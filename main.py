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
# Copy the FULL string: 42["auth",{"session":"...","isDemo":1,...}]
DEMO_SSID = "AUzrqQYcmseidQS_7"
LIVE_SSID = "AE7fchX1LKy3h7hUd"

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
    if avg_loss == 0: return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))

def _fmt_time(dt_obj):
    return dt_obj.strftime("%H:%M:%S UTC") if dt_obj else "None"

# ================= CORE BOT CLASS =================
class TradingBot:
    def __init__(self):
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

    async def send(self, msg: str):
        try:
            if self.app:
                await self.app.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg)
        except Exception as e:
            logging.error(f"Telegram error: {e}")

    def status_line(self):
        if not self.running: return "🛑 Stopped"
        if self.paused: return "⏸ Paused"
        if self.active_trade: return f"🎯 In trade ({self.active_trade_asset})"
        return "🔎 Searching..."

    async def connect(self) -> bool:
        """Enhanced Connection with Dual SSID Support"""
        try:
            ssid = (DEMO_SSID if self.is_demo else LIVE_SSID).strip()
            if not ssid or "PASTE_" in ssid:
                await self.send("❌ SSID is not set in the CONFIG section.")
                return False

            if self.client:
                try: await self.client.disconnect()
                except: pass

            self.client = AsyncPocketOptionClient(ssid=ssid, is_demo=self.is_demo)
            await self.client.connect()
            await asyncio.sleep(5) # Handshake buffer

            if hasattr(self.client, "connected") and not getattr(self.client, "connected"):
                await self.send("❌ Connection failed (Check SSID or IP block).")
                return False

            balance = await self.client.get_balance()
            mode = "DEMO" if self.is_demo else "LIVE"
            await self.send(f"✅ Connected to {mode}\nBalance: ${balance}")

            for asset in ASSETS:
                await self.client.subscribe_candles(asset, 300)
                # Re-add handlers on every reconnection
                self.client.add_candle_handler(
                    asset, lambda candle, a=asset: asyncio.create_task(self.on_candle(candle, a))
                )
            return True
        except Exception as e:
            await self.send(f"❌ Connection error: {repr(e)}")
            return False

    async def get_candles(self, asset, tf, count=220):
        try:
            candles = await self.client.get_candles(asset, tf, count)
            if not candles or len(candles) < 210: return None
            return {"c": np.array([c["close"] for c in candles], dtype=float)}
        except: return None

    async def check_entry(self, asset):
        self.last_scan_time = datetime.datetime.utcnow()
        h1_data = await self.get_candles(asset, 3600)
        if not h1_data: return
        
        ema200_h1 = ema_last(h1_data["c"], 200)
        if ema200_h1 is None: return
        bias = "buy" if h1_data["c"][-1] > ema200_h1 else "sell"

        m5_data = await self.get_candles(asset, 300)
        if not m5_data: return
        c = m5_data["c"]
        rsi_val = rsi_last(c, 14)
        ema200_m5 = ema_last(c, 200)

        if rsi_val is None or ema200_m5 is None: return

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
            if self.paused or not self.running or self.active_trade: return
            
            order = await self.client.place_order(asset, float(self.trade_amount), direction, MIN_EXPIRY)
            if order:
                self.active_trade = True
                self.active_trade_asset = asset
                self.daily_trades += 1
                await self.send(f"🚀 Trade Placed ({'DEMO' if self.is_demo else 'LIVE'})\n{direction.name} {asset}")
                asyncio.create_task(self.finish_trade())

    async def finish_trade(self):
        await asyncio.sleep(MIN_EXPIRY + 5)
        self.active_trade = False
        self.active_trade_asset = None

    async def on_candle(self, candle, asset):
        if not self.running or self.paused or self.active_trade: return
        async with self.asset_locks[asset]:
            await self.check_entry(asset)

    async def runner(self):
        ok = await self.connect()
        if not ok: 
            self.running = False
            return
        while True:
            await asyncio.sleep(30)

    def start(self):
        self.running = True
        self.paused = False
        if self._runner_task is None or self._runner_task.done():
            self._runner_task = asyncio.create_task(self.runner())

# ================= TELEGRAM HANDLERS =================
bot_instance = TradingBot()

def main_menu():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("▶️ Run Bot", callback_data="PROMPT_MODE"),
         InlineKeyboardButton("🛑 Stop", callback_data="STOP")],
        [InlineKeyboardButton("📊 Status", callback_data="STATUS")]
    ])

def mode_menu():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🧪 Start on DEMO", callback_data="START_DEMO")],
        [InlineKeyboardButton("💎 Start on LIVE", callback_data="START_LIVE")],
        [InlineKeyboardButton("⬅️ Back", callback_data="BACK")]
    ])

async def start_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    await u.message.reply_text("🤖 PocketOption Bot Ready", reply_markup=main_menu())

async def btn_handler(u: Update, c: ContextTypes.DEFAULT_TYPE):
    q = u.callback_query
    await q.answer()

    if q.data == "PROMPT_MODE":
        await q.edit_message_text("Select Account Type:", reply_markup=mode_menu())
    
    elif q.data == "START_DEMO":
        bot_instance.is_demo = True
        await q.edit_message_text("⏳ Initializing Demo...")
        bot_instance.start()

    elif q.data == "START_LIVE":
        bot_instance.is_demo = False
        await q.edit_message_text("⏳ Initializing LIVE...")
        bot_instance.start()

    elif q.data == "STOP":
        bot_instance.running = False
        await q.edit_message_text("⏸ Bot Stopped", reply_markup=main_menu())

    elif q.data == "BACK":
        await q.edit_message_text("Bot Menu:", reply_markup=main_menu())

    elif q.data == "STATUS":
        txt = f"📊 {bot_instance.status_line()}\nMode: {'DEMO' if bot_instance.is_demo else 'LIVE'}\nTrades: {bot_instance.daily_trades}"
        await q.edit_message_text(txt, reply_markup=main_menu())

def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    bot_instance.app = app
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CallbackQueryHandler(btn_handler))
    print("Bot is live...")
    app.run_polling()

if __name__ == "__main__":
    main()
