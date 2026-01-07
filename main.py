import asyncio
import datetime
import logging
import numpy as np

from pocketoptionapi_async import AsyncPocketOptionClient, OrderDirection
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
)

# ========================= CONFIG =========================
SSID = "n5jlYj9XTDjpMpoH/A2DysApxaK_JCsxpm"
TELEGRAM_TOKEN = "8276370676:AAGh5VqkG7b4cvpfRIVwY_rtaBlIiNwCTDM"
TELEGRAM_CHAT_ID = "7634818949"

ASSETS = [
    "EURUSD_otc", "GBPUSD_otc", "AUDUSD_otc", "USDJPY_otc",
    "EURGBP_otc", "EURJPY_otc", "GBPJPY_otc", "AUDJPY_otc",
]

DEFAULT_TRADE_AMOUNT = 1.0
MIN_EXPIRY = 180  # 3 minutes

MAX_TRADES_PER_DAY = 10
MAX_CONSECUTIVE_LOSSES = 3
# =========================================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


# ================= INDICATORS =================
def ema_last(values, period):
    if values is None or len(values) < period:
        return None
    k = 2 / (period + 1)
    ema_val = float(values[0])
    for price in values[1:]:
        ema_val = float(price) * k + ema_val * (1 - k)
    return float(ema_val)


def rsi_last(values, period=14):
    if values is None or len(values) < period + 1:
        return None
    v = values[-(period + 1):]
    deltas = np.diff(v)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = float(np.mean(gains))
    avg_loss = float(np.mean(losses))
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _fmt_time(dt_obj):
    return dt_obj.strftime("%H:%M:%S UTC") if dt_obj else "None"


# ================= CORE BOT CLASS =================
class TradingBot:
    def __init__(self):  # Fixed: added double underscores
        self.app = None
        self.client = None

        self.is_demo = True
        self.trade_amount = float(DEFAULT_TRADE_AMOUNT)

        self.daily_trades = 0
        self.consec_losses = 0
        self.daily_pnl = 0.0
        self.current_date = datetime.date.today()

        self.running = False
        self.paused = False

        self.last_scan_time = None
        self.last_signal_time = None
        self.last_trade_info = None

        self.active_trade = False
        self.active_trade_asset = None

        self.trade_lock = asyncio.Lock()
        self.asset_locks = {a: asyncio.Lock() for a in ASSETS}

        self._runner_task = None
        self._handlers_added = False

    async def send(self, msg: str):
        try:
            if self.app:
                await self.app.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg)
        except Exception as e:
            logging.error(f"Telegram error: {e}")

    def reset_daily(self):
        today = datetime.date.today()
        if today != self.current_date:
            self.daily_trades = 0
            self.consec_losses = 0
            self.daily_pnl = 0.0
            self.current_date = today

    def status_line(self):
        if not self.running:
            return "🛑 Stopped"
        if self.paused:
            return "⏸ Paused"
        if self.active_trade:
            return f"🎯 In trade ({self.active_trade_asset})"
        return "🔎 Searching for signal…"

    async def connect(self) -> bool:
        try:
            self.client = AsyncPocketOptionClient(ssid=SSID, is_demo=self.is_demo)
            await self.client.connect()

            if hasattr(self.client, "connected") and not getattr(self.client, "connected"):
                await self.send("❌ Connection failed (client not connected).")
                return False

            balance = await self.client.get_balance()
            mode = "DEMO" if self.is_demo else "LIVE"
            await self.send(f"✅ Connected to {mode}\nBalance: ${balance}")

            for asset in ASSETS:
                await self.client.subscribe_candles(asset, 300)

            if not self._handlers_added:
                for asset in ASSETS:
                    self.client.add_candle_handler(
                        asset,
                        lambda candle, a=asset: asyncio.create_task(self.on_candle(candle, a))
                    )
                self._handlers_added = True

            return True

        except Exception as e:
            logging.error(f"Connection failed: {e}")
            await self.send(f"⚠️ Connection failed: {e}")
            return False

    async def get_candles(self, asset, tf, count=220):
        try:
            candles = await self.client.get_candles(asset, tf, count)
            if not candles or len(candles) < 210:
                return None
            return {
                "c": np.array([c["close"] for c in candles], dtype=float)
            }
        except Exception:
            return None

    async def check_entry(self, asset):
        self.last_scan_time = datetime.datetime.utcnow()

        h1_data = await self.get_candles(asset, 3600, count=220)
        if not h1_data:
            return
        ema200_h1 = ema_last(h1_data["c"], 200)
        if ema200_h1 is None:
            return

        bias = "buy" if h1_data["c"][-1] > ema200_h1 else "sell"

        m5_data = await self.get_candles(asset, 300, count=220)
        if not m5_data:
            return

        c = m5_data["c"]
        rsi_val = rsi_last(c, 14)
        ema200_m5 = ema_last(c, 200)

        if rsi_val is None or ema200_m5 is None:
            return

        signal = None
        if bias == "buy" and c[-1] > ema200_m5 and 45 <= rsi_val <= 65:
            signal = OrderDirection.CALL
        elif bias == "sell" and c[-1] < ema200_m5 and 35 <= rsi_val <= 55:
            signal = OrderDirection.PUT

        if signal:
            self.last_signal_time = datetime.datetime.utcnow()
            await self.execute_trade(asset, signal)

    async def execute_trade(self, asset, direction):
        async with self.trade_lock:
            self.reset_daily()

            if self.paused or (not self.running) or self.active_trade:
                return

            if self.daily_trades >= MAX_TRADES_PER_DAY or self.consec_losses >= MAX_CONSECUTIVE_LOSSES:
                self.paused = True
                await self.send("🛑 Limits reached. Bot paused.")
                return

            balance_before = await self.client.get_balance()

            order = await self.client.place_order(
                asset=asset,
                amount=float(self.trade_amount),
                direction=direction,
                duration=MIN_EXPIRY,
            )

            if not order:
                logging.warning(f"Order failed for {asset}")
                return

            order_id = order.get("id") if isinstance(order, dict) else None

            self.active_trade = True
            self.active_trade_asset = asset
            self.daily_trades += 1
            self.last_trade_info = f"{direction.name} {asset} @ ${self.trade_amount:.2f}"

            await self.send(
                f"🚀 Trade Placed\n"
                f"{'🟢 CALL' if direction == OrderDirection.CALL else '🔴 PUT'} {asset}\n"
                f"Stake: ${self.trade_amount:.2f}\n"
                f"Expiry: {MIN_EXPIRY//60}m\n"
                f"Trades today: {self.daily_trades}/{MAX_TRADES_PER_DAY}"
            )

            asyncio.create_task(self.check_result(asset, order_id, balance_before))

    async def check_result(self, asset, order_id, balance_before):
        try:
            await asyncio.sleep(MIN_EXPIRY + 8)
            profit = None
            try:
                history = await self.client.get_order_history()
                if history and order_id is not None:
                    my_order = next((o for o in history if str(o.get("id")) == str(order_id)), None)
                    if my_order is not None:
                        profit = float(my_order.get("profit", 0.0))
            except Exception:
                profit = None

            if profit is None:
                balance_after = await self.client.get_balance()
                diff = float(balance_after) - float(balance_before)
                profit = diff

            if profit > 0:
                self.consec_losses = 0
                self.daily_pnl += float(profit)
                await self.send(f"✅ WIN {asset}\n+${profit:.2f}\nDaily PnL: ${self.daily_pnl:.2f}")
            else:
                self.consec_losses += 1
                self.daily_pnl += float(profit)
                await self.send(
                    f"❌ LOSS {asset}\n${profit:.2f}\n"
                    f"Consec losses: {self.consec_losses}/{MAX_CONSECUTIVE_LOSSES}\n"
                    f"Daily PnL: ${self.daily_pnl:.2f}"
                )
        finally:
            self.active_trade = False
            self.active_trade_asset = None

    async def on_candle(self, candle, asset):
        if not self.running or self.paused or self.active_trade:
            return
        async with self.asset_locks[asset]:
            await self.check_entry(asset)

    async def runner(self):
        ok = await self.connect()
        if not ok:
            self.running = False
            return

        while True:
            if self.running and self.client:
                if hasattr(self.client, "connected") and not getattr(self.client, "connected"):
                    await self.send("⚠️ Disconnected. Reconnecting...")
                    self._handlers_added = False
                    await self.connect()
            await asyncio.sleep(15)

    def start(self):
        self.running = True
        self.paused = False
        if self._runner_task is None or self._runner_task.done():
            self._runner_task = asyncio.create_task(self.runner())

    def stop(self):
        self.running = False
        self.paused = True


# ================= TELEGRAM HANDLERS =================
bot_instance = TradingBot()


def keyboard():
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("▶️ Run", callback_data="RUN"),
            InlineKeyboardButton("⏸ Stop", callback_data="STOP")
        ],
        [
            InlineKeyboardButton("📊 Status", callback_data="STATUS")
        ],
    ])


async def start_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    await u.message.reply_text("🤖 PocketOption Bot Ready", reply_markup=keyboard())


async def btn_handler(u: Update, c: ContextTypes.DEFAULT_TYPE):
    q = u.callback_query
    await q.answer()

    if q.data == "RUN":
        bot_instance.start()
        await q.edit_message_text("▶️ Bot Started", reply_markup=keyboard())

    elif q.data == "STOP":
        bot_instance.stop()
        await q.edit_message_text("⏸ Bot Stopped", reply_markup=keyboard())

    elif q.data == "STATUS":
        txt = (
            f"📊 Status\n{bot_instance.status_line()}\n\n"
            f"Trades today: {bot_instance.daily_trades}/{MAX_TRADES_PER_DAY}\n"
            f"Consecutive losses: {bot_instance.consec_losses}/{MAX_CONSECUTIVE_LOSSES}\n"
            f"Daily PnL: ${bot_instance.daily_pnl:.2f}\n"
            f"Last scan: {_fmt_time(bot_instance.last_scan_time)}\n"
            f"Last signal: {_fmt_time(bot_instance.last_signal_time)}\n"
            f"Last trade: {bot_instance.last_trade_info or 'None'}"
        )
        await q.edit_message_text(txt, reply_markup=keyboard())


def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    bot_instance.app = app

    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CallbackQueryHandler(btn_handler))

    print("Bot is live...")
    app.run_polling()


if __name__ == "__main__":
    main()
