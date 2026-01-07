import asyncio
import logging
import datetime
import numpy as np
from scipy.signal import find_peaks

from pocketoptionapi_async import AsyncPocketOptionClient, OrderDirection
from telegram.ext import Application, CommandHandler, ContextTypes

# ========================= HARDCODED CONFIG =========================
SSID = "n5jlYj9XTDjpMpoH/A2DysApxaK_JCsxpm"

TELEGRAM_TOKEN = "8365422417:AAEqVxqOuEBMm2PBQUACa5Una5K6O7qNlQQ"
TELEGRAM_CHAT_ID = "7634818949"

ASSETS = ["EURUSD_otc", "GBPUSD_otc", "AUDUSD_otc", "USDJPY_otc"]

TRADE_AMOUNT = 1.0
MIN_EXPIRY = 180
MAX_EXPIRY = 300

MAX_TRADES_PER_DAY = 5
MAX_CONSECUTIVE_LOSSES = 2
COOLDOWN_SECONDS = 60
# ===================================================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


# ================= PURE PYTHON INDICATORS =================
def ema(values, period):
    if len(values) < period:
        return None
    k = 2 / (period + 1)
    ema_val = values[0]
    for price in values[1:]:
        ema_val = price * k + ema_val * (1 - k)
    return ema_val


def rsi(values, period=14):
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


def atr(highs, lows, closes, period=14):
    if len(closes) < period + 1:
        return None
    trs = []
    for i in range(-period, 0):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1])
        )
        trs.append(tr)
    return np.mean(trs)
# ==========================================================


class TradingBot:
    def init(self, app: Application):
        self.app = app
        self.client = None
        self.is_demo = None
        self.daily_trades = 0
        self.consec_losses = 0
        self.current_date = datetime.date.today()
        self.last_trade_time = {}

    async def send_telegram(self, text):
        await self.app.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=text)

    async def connect(self):
        try:
            self.client = AsyncPocketOptionClient(
                ssid=SSID,
                is_demo=self.is_demo
            )
            await self.client.connect()

            if not getattr(self.client, "connected", False):
                await self.send_telegram("❌ Connection failed")
                return False

            balance = await self.client.get_balance()
            mode = "DEMO" if self.is_demo else "LIVE"

            await self.send_telegram(
                f"🚀 Connected\nMode: {mode}\nBalance: ${balance}"
            )

            for asset in ASSETS:
                await self.client.subscribe_candles(asset, 300)

            return True

        except Exception as e:
            await self.send_telegram(f"⚠️ Error: {e}")
            return False

    async def get_candles(self, asset, timeframe, count=200):
        candles = await self.client.get_candles(asset, timeframe, count)
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

        c = data["c"]
        ema200 = ema(c, 200)
        if ema200 is None:
            return None

        return "buy" if c[-1] > ema200 else "sell"

    async def check_trade(self, asset):
        bias = await self.get_bias(asset)
        if not bias:
            return

        data = await self.get_candles(asset, 300)
        if not data:
            return
            o, h, l, c = data["o"], data["h"], data["l"], data["c"]

        rsi_val = rsi(c)
        atr_val = atr(h, l, c)

        if rsi_val is None or atr_val is None:
            return

        direction = OrderDirection.CALL if bias == "buy" else OrderDirection.PUT

        await self.client.place_order(
            asset=asset,
            amount=TRADE_AMOUNT,
            direction=direction,
            duration=MIN_EXPIRY
        )

        await self.send_telegram(
            f"{'🟢 CALL' if bias=='buy' else '🔴 PUT'} {asset}"
        )

    async def on_candle(self, candle, asset):
        await self.check_trade(asset)

    async def run(self):
        if not await self.connect():
            return

        for asset in ASSETS:
            self.client.add_candle_handler(
                asset,
                lambda c, a=asset: asyncio.create_task(self.on_candle(c, a))
            )

        while True:
            await asyncio.sleep(30)


bot = None


async def start(update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🤖 Bot ready\n/demo or /live")


async def demo(update, context: ContextTypes.DEFAULT_TYPE):
    global bot
    bot.is_demo = True
    asyncio.create_task(bot.run())
    await update.message.reply_text("✅ DEMO started")


async def live(update, context: ContextTypes.DEFAULT_TYPE):
    global bot
    bot.is_demo = False
    asyncio.create_task(bot.run())
    await update.message.reply_text("⚠️ LIVE started")


def main():
    global bot
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    bot = TradingBot(app)

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("demo", demo))
    app.add_handler(CommandHandler("live", live))

    app.run_polling()


if __name__ == "__main__":
    main()
