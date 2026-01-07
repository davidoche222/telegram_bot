import asyncio
import datetime
import logging
import numpy as np
from scipy.signal import find_peaks

from pocketoptionapi_async import AsyncPocketOptionClient, OrderDirection
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# ========================= CONFIG =========================
SSID = "n5jlYj9XTDjpMpoH/A2DysApxaK_JCsxpm"
TELEGRAM_TOKEN = "8276370676:AAGh5VqkG7b4cvpfRIVwY_rtaBlIiNwCTDM"
TELEGRAM_CHAT_ID = "7634818949"

ASSETS = [
    "EURUSD_otc",
    "GBPUSD_otc",
    "AUDUSD_otc",
    "USDJPY_otc",
    "EURGBP_otc",
    "EURJPY_otc",
    "GBPJPY_otc",
    "AUDJPY_otc",
]

DEFAULT_TRADE_AMOUNT = 1.0
MIN_EXPIRY = 180
MAX_EXPIRY = 300

MAX_TRADES_PER_DAY = 5
MAX_CONSECUTIVE_LOSSES = 2
# =========================================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# ================= INDICATORS =================
def ema_last(values, period):
    if len(values) < period:
        return None
    k = 2 / (period + 1)
    ema_val = values[0]
    for price in values[1:]:
        ema_val = price * k + ema_val * (1 - k)
    return ema_val


def rsi_last(values, period=14):
    if len(values) < period + 1:
        return None
    deltas = np.diff(values[-(period + 1):])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def atr_last(h, l, c, period=14):
    if len(c) < period + 1:
        return None
    trs = []
    for i in range(-period, 0):
        tr = max(
            h[i] - l[i],
            abs(h[i] - c[i - 1]),
            abs(l[i] - c[i - 1]),
        )
        trs.append(tr)
    return np.mean(trs)
# =========================================================


class TradingBot:
    def init(self):
        self.app = None
        self.client = None

        self.is_demo = True
        self.trade_amount = DEFAULT_TRADE_AMOUNT

        self.daily_trades = 0
        self.consec_losses = 0
        self.daily_pnl = 0.0
        self.current_date = datetime.date.today()

        self.running = False
        self.paused = False

        self.last_trade_info = None
        self.last_scan_time = None
        self.last_signal_time = None
        self.active_trade = False
        self.active_trade_asset = None

        self.trade_lock = asyncio.Lock()
        self.asset_locks = {a: asyncio.Lock() for a in ASSETS}

    async def send(self, msg):
        await self.app.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg)

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
  async def connect(self):
        self.client = AsyncPocketOptionClient(ssid=SSID, is_demo=self.is_demo)
        await self.client.connect()

        balance = await self.client.get_balance()
        mode = "DEMO" if self.is_demo else "LIVE"
        await self.send(f"🚀 Connected\nMode: {mode}\nBalance: ${balance}")

        for asset in ASSETS:
            await self.client.subscribe_candles(asset, 300)

    async def get_candles(self, asset, tf, count=220):
        candles = await self.client.get_candles(asset, tf, count)
if not candles or len(candles) < 60:
            return None
        return {
            "o": np.array([c["open"] for c in candles]),
            "h": np.array([c["high"] for c in candles]),
            "l": np.array([c["low"] for c in candles]),
            "c": np.array([c["close"] for c in candles]),
        }
 async def get_bias(self, asset):
        data = await self.get_candles(asset, 3600)
        if not data:
            return None
  ema200 = ema_last(data["c"], 200)
        if ema200 is None:
            return None

        return "buy" if data["c"][-1] > ema200 else "sell"

    async def check_entry(self, asset):
        self.last_scan_time = datetime.datetime.utcnow()

        bias = await self.get_bias(asset)
        if not bias:
            return

        data = await self.get_candles(asset, 300)
        if not data:
            return

        o, h, l, c = data["o"], data["h"], data["l"], data["c"]

        rsi = rsi_last(c)
        atr = atr_last(h, l, c)
        ema200 = ema_last(c, 200)

        if rsi is None or atr is None or ema200 is None:
            return

        if bias == "buy" and (c[-1] <= ema200 or not 45 <= rsi <= 65):
            return
        if bias == "sell" and (c[-1] >= ema200 or not 35 <= rsi <= 55):
            return

        self.last_signal_time = datetime.datetime.utcnow()

        async with self.trade_lock:
            self.reset_daily()
            if self.daily_trades >= MAX_TRADES_PER_DAY:
                return
            if self.consec_losses >= MAX_CONSECUTIVE_LOSSES:
                return

            direction = OrderDirection.CALL if bias == "buy" else OrderDirection.PUT
            expiry = MIN_EXPIRY

            balance_before = await self.client.get_balance()

            order = await self.client.place_order(
                asset=asset,
                amount=self.trade_amount,
                direction=direction,
                duration=expiry,
            )

            if order and order.get("status") == "success":
                self.daily_trades += 1
                self.active_trade = True
                self.active_trade_asset = asset

                await self.send(
                    f"{'🟢 CALL' if bias=='buy' else '🔴 PUT'} {asset}\n"
                    f"Stake: ${self.trade_amount}\nExpiry: {expiry//60}m"
                )

                asyncio.create_task(self.check_result(asset, expiry, balance_before))

    async def check_result(self, asset, expiry, balance_before):
        await asyncio.sleep(expiry + 5)
        balance_after = await self.client.get_balance()
        diff = balance_after - balance_before

        self.daily_pnl += diff

        if diff > 0:
            self.consec_losses = 0
            await self.send(f"✅ WIN {asset}\n+${diff:.2f}")
        else:
            self.consec_losses += 1
            await self.send(f"❌ LOSS {asset}\n{diff:.2f}")

        self.active_trade = False
        self.active_trade_asset = None

    async def on_candle(self, candle, asset):
        if self.running and not self.paused:
            async with self.asset_locks[asset]:
                await self.check_entry(asset)

    async def runner(self):
        await self.connect()
        self.running = True

        for asset in ASSETS:
            self.client.add_candle_handler(
                asset,
                lambda c, a=asset: asyncio.create_task(self.on_candle(c, a))
            )

        while self.running:
            await asyncio.sleep(30)

    def start(self):
        if not self.running:
            asyncio.create_task(self.runner())

    def stop(self):
        self.running = False
        self.paused = True


# ================= TELEGRAM =================
bot_instance = TradingBot()

def keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("▶️ Run", callback_data="RUN"),
         InlineKeyboardButton("⏸ Stop", callback_data="STOP")],
        [InlineKeyboardButton("📊 Status", callback_data="STATUS")],
        [InlineKeyboardButton("💰 Set Stake", callback_data="STAKE")],
    ])
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🤖 Bot Ready", reply_markup=keyboard())

async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(bot_instance.status_line(), reply_markup=keyboard())

async def stake_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    amt = float(context.args[0])
    bot_instance.trade_amount = amt
    await update.message.reply_text(f"Stake set to ${amt}", reply_markup=keyboard())

async def buttons(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()

    if q.data == "RUN":
        bot_instance.start()
        await q.edit_message_text("▶️ Running", reply_markup=keyboard())
    elif q.data == "STOP":
        bot_instance.stop()
        await q.edit_message_text("⏸ Stopped", reply_markup=keyboard())
    elif q.data == "STATUS":
        await q.edit_message_text(bot_instance.status_line(), reply_markup=keyboard())
    elif q.data == "STAKE":
        await q.edit_message_text("Use /stake 2", reply_markup=keyboard())


def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    bot_instance.app = app

    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(CommandHandler("stake", stake_cmd))
    app.add_handler(CallbackQueryHandler(buttons))

    app.run_polling()


if __name__ == "__main__":
    main()
