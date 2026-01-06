import os
import asyncio
import logging
import datetime
import numpy as np
import talib
from scipy.signal import find_peaks

from pocketoptionapi_async import AsyncPocketOptionClient, OrderDirection
from telegram.ext import Application, CommandHandler, ContextTypes

# ========================= ENV CONFIG =========================
SSID = os.getenv("SSID", "n5jlYj9XTDjpMpoH/A2DysApxaK_JCsxpm").strip()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN","8365422417:AAEqVxqOuEBMm2PBQUACa5Una5K6O7qNlQQ").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "7634818949").strip()

ASSETS = ["EURUSD_otc", "GBPUSD_otc", "AUDUSD_otc", "USDJPY_otc"]

TRADE_AMOUNT = float(os.getenv("TRADE_AMOUNT", "1.0"))
MIN_EXPIRY = int(os.getenv("MIN_EXPIRY", "180"))   # 3 minutes
MAX_EXPIRY = int(os.getenv("MAX_EXPIRY", "300"))   # 5 minutes

MAX_TRADES_PER_DAY = int(os.getenv("MAX_TRADES_PER_DAY", "5"))
MAX_CONSECUTIVE_LOSSES = int(os.getenv("MAX_CONSECUTIVE_LOSSES", "2"))

COOLDOWN_SECONDS = int(os.getenv("COOLDOWN_SECONDS", "60"))
# =============================================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


class UpgradedTradingBot:
    def init(self, app: Application):
        self.app = app
        self.client = None
        self.is_demo = None

        self.daily_trades = 0
        self.consec_losses = 0
        self.current_date = datetime.date.today()

        self.asset_locks = {a: asyncio.Lock() for a in ASSETS}
        self.last_trade_time = {a: None for a in ASSETS}

        self.running_task: asyncio.Task | None = None

    async def send_telegram(self, message: str):
        try:
            await self.app.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
        except Exception as e:
            logging.error(f"Telegram failed: {e}")

    def reset_daily_limits(self):
        today = datetime.date.today()
        if today != self.current_date:
            self.daily_trades = 0
            self.consec_losses = 0
            self.current_date = today

    async def can_place_trade(self) -> bool:
        self.reset_daily_limits()
        if self.daily_trades >= MAX_TRADES_PER_DAY:
            await self.send_telegram("🛑 Max trades per day reached. Bot paused until tomorrow.")
            return False
        if self.consec_losses >= MAX_CONSECUTIVE_LOSSES:
            await self.send_telegram("🛑 Max consecutive losses reached. Bot paused until next session.")
            return False
        return True

    async def connect(self) -> bool:
        if not SSID:
            await self.send_telegram("❌ SSID is empty. Set SSID in Railway Variables before running.")
            return False

        try:
            self.client = AsyncPocketOptionClient(ssid=SSID, is_demo=self.is_demo)
            await self.client.connect()

            if not getattr(self.client, "connected", False):
                await self.send_telegram("❌ Connection failed. Check your SSID and try again.")
                return False

            balance = await self.client.get_balance()
            mode = "DEMO" if self.is_demo else "REAL (LIVE)"
            await self.send_telegram(
                f"🚀 Connected!\nMode: {mode}\nBalance: ${balance}\nMonitoring {len(ASSETS)} assets..."
            )

            for asset in ASSETS:
                await self.client.subscribe_candles(asset, 300)  # 5M candles

            return True

        except Exception as e:
            s = str(e).lower()
            if "auth" in s or "unauthorized" in s:
                await self.send_telegram("⚠️ SESSION EXPIRED. Get a new SSID and update Railway Variable SSID.")
            else:
                await self.send_telegram(f"⚠️ Connection error: {e}")
            return False

    async def get_candles(self, asset: str, timeframe: int, count: int = 200):
        try:
            candles = await self.client.get_candles(asset, timeframe, count)
            if not candles or len(candles) < 60:
                return None
o = np.array([c["open"] for c in candles], dtype=float)
            h = np.array([c["high"] for c in candles], dtype=float)
            l = np.array([c["low"] for c in candles], dtype=float)
            c_ = np.array([c["close"] for c in candles], dtype=float)
            return {"o": o, "h": h, "l": l, "c": c_}
        except Exception:
            return None

    async def get_1h_bias(self, asset: str):
        data = await self.get_candles(asset, 3600, 220)
        if not data:
            return None

        o, h, l, c = data["o"], data["h"], data["l"], data["c"]

        ema200 = talib.EMA(c, 200)
        if ema200 is None or len(ema200) < 2 or np.isnan(ema200[-1]) or np.isnan(ema200[-2]):
            return None

        slope = ema200[-1] - ema200[-2]
        is_up_trend = slope > 0.0001
        is_down_trend = slope < -0.0001

        peaks, _ = find_peaks(h, distance=4)
        valleys, _ = find_peaks(-l, distance=4)
        if len(peaks) < 2 or len(valleys) < 2:
            return None

        bull_structure = h[peaks[-1]] > h[peaks[-2]] and l[valleys[-1]] > l[valleys[-2]]
        bear_structure = h[peaks[-1]] < h[peaks[-2]] and l[valleys[-1]] < l[valleys[-2]]

        last_range = (h[-1] - l[-1])
        if last_range <= 0:
            return None

        last_bullish = c[-1] > o[-1] and abs(c[-1] - o[-1]) > 0.5 * last_range
        last_bearish = c[-1] < o[-1] and abs(o[-1] - c[-1]) > 0.5 * last_range

        if bull_structure and c[-1] > ema200[-1] and is_up_trend and last_bullish:
            return "buy"
        if bear_structure and c[-1] < ema200[-1] and is_down_trend and last_bearish:
            return "sell"
        return None

    async def check_entry(self, asset: str):
        async with self.asset_locks[asset]:
            now = datetime.datetime.utcnow()
            lt = self.last_trade_time[asset]
            if lt and (now - lt).total_seconds() < COOLDOWN_SECONDS:
                return

            bias = await self.get_1h_bias(asset)
            if not bias:
                return

            data_5m = await self.get_candles(asset, 300, 220)
            if not data_5m:
                return

            o, h, l, c = data_5m["o"], data_5m["h"], data_5m["l"], data_5m["c"]

            valleys, _ = find_peaks(-l[-30:], distance=3)
            peaks, _ = find_peaks(h[-30:], distance=3)

            zone = None
            if bias == "buy" and len(valleys) > 0:
                zone = l[-30:][valleys[-1]]
            elif bias == "sell" and len(peaks) > 0:
                zone = h[-30:][peaks[-1]]

            if zone is None:
                return

            atr14 = talib.ATR(h, l, c, 14)
            if atr14 is None or len(atr14) == 0 or np.isnan(atr14[-1]):
                return

            tolerance = atr14[-1] * 0.5
            near_zone = (min(l[-3:]) <= zone + tolerance) and (max(h[-3:]) >= zone - tolerance)
            if not near_zone:
                return

            # Triggers
            is_engulfing = is_pinbar = is_micro_bos = False
            strength = "moderate"

            body = abs(c[-1] - o[-1])
            candle_range = (h[-1] - l[-1])
            if candle_range <= 0:
                return

            if bias == "buy":
                if c[-1] > o[-1] and c[-1] > h[-2] and o[-1] < c[-2]:
                    is_engulfing = True
                    strength = "strong"
                lower_wick = min(o[-1], c[-1]) - l[-1]
                if body > 0 and lower_wick >= 2 * body and c[-1] > (h[-1] + l[-1]) / 2:
                    is_pinbar = True
                if len(c) >= 3 and c[-2] < o[-2] and c[-1] > h[-2]:
                    is_micro_bos = True
                    strength = "strong"
            else:
                if c[-1] < o[-1] and c[-1] < l[-2] and o[-1] > c[-2]:
is_engulfing = True
                    strength = "strong"
                upper_wick = h[-1] - max(o[-1], c[-1])
                if body > 0 and upper_wick >= 2 * body and c[-1] < (h[-1] + l[-1]) / 2:
                    is_pinbar = True
                if len(c) >= 3 and c[-2] > o[-2] and c[-1] < l[-2]:
                    is_micro_bos = True
                    strength = "strong"

            if not (is_engulfing or is_pinbar or is_micro_bos):
                return

            # Filters
            ema200_5m = talib.EMA(c, 200)
            rsi14 = talib.RSI(c, 14)

            if (
                ema200_5m is None
                or rsi14 is None
                or np.isnan(ema200_5m[-1])
                or np.isnan(rsi14[-1])
            ):
                return

            ema200_5m_last = ema200_5m[-1]
            rsi = rsi14[-1]
            atr = atr14[-1]
            prev_body = abs(c[-2] - o[-2])

            if bias == "buy":
                if c[-1] <= ema200_5m_last or not (45 <= rsi <= 65):
                    return
            else:
                if c[-1] >= ema200_5m_last or not (35 <= rsi <= 55):
                    return

            if body < 0.7 * atr or body <= prev_body:
                return

            if not await self.can_place_trade():
                return

            direction = OrderDirection.CALL if bias == "buy" else OrderDirection.PUT
            expiry = MIN_EXPIRY if strength == "strong" else MAX_EXPIRY

            order = await self.client.place_order(
                asset=asset,
                amount=TRADE_AMOUNT,
                direction=direction,
                duration=expiry,
            )

            if order and order.get("status") == "success":
                self.daily_trades += 1
                self.last_trade_time[asset] = now

                await self.send_telegram(
                    f"{'🟢 CALL' if bias=='buy' else '🔴 PUT'} on {asset}\n"
                    f"${TRADE_AMOUNT} | {expiry//60} min expiry\nStrength: {strength.upper()}"
                )
            else:
                await self.send_telegram(f"❌ Trade failed on {asset}")

    async def on_candle_update(self, candle, asset: str):
        if self.client and getattr(self.client, "connected", False):
            await self.check_entry(asset)

    async def run_trading(self):
        if not await self.connect():
            return

        for asset in ASSETS:
            self.client.add_candle_handler(
                asset,
                lambda c, a=asset: asyncio.create_task(self.on_candle_update(c, a)),
            )

        await self.send_telegram("🤖 Bot is now monitoring the market 24/7...")
        while True:
            await asyncio.sleep(30)


bot_instance: UpgradedTradingBot | None = None


async def start(update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 Trading Bot Online!\n\nChoose mode:\n"
        "🔹 /demo - Practice account\n"
        "🔸 /live - Real money account\n"
    )


async def demo(update, context: ContextTypes.DEFAULT_TYPE):
    global bot_instance
    bot_instance.is_demo = True
    await update.message.reply_text("✅ Mode set to DEMO. Starting bot...")
    if bot_instance.running_task is None or bot_instance.running_task.done():
        bot_instance.running_task = asyncio.create_task(bot_instance.run_trading())


async def live(update, context: ContextTypes.DEFAULT_TYPE):
    global bot_instance
    bot_instance.is_demo = False
    await update.message.reply_text("⚠️ Mode set to LIVE. Starting bot...")
    if bot_instance.running_task is None or bot_instance.running_task.done():
        bot_instance.running_task = asyncio.create_task(bot_instance.run_trading())


async def post_init(app: Application):
    if TELEGRAM_CHAT_ID:
        await app.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text="🤖 Bot deployed and online!\nSend /start")


def main():
    global bot_instance
pycache/
*.pyc
.env
.venv/
venv/
.DS_Store
.idea/
.vscode/
if c[-1] > ema200[-1]:
            return "buy"
        if c[-1] < ema200[-1]:
            return "sell"

        return None

    async def check_entry(self, asset):
        async with self.asset_locks[asset]:
            now = datetime.datetime.utcnow()
            last = self.last_trade_time[asset]
            if last and (now - last).total_seconds() < COOLDOWN_SECONDS:
                return

            bias = await self.get_1h_bias(asset)
            if not bias:
                return

            if not await self.can_place_trade():
                return

            direction = OrderDirection.CALL if bias == "buy" else OrderDirection.PUT
            expiry = MIN_EXPIRY

            order = await self.client.place_order(
                asset=asset,
                amount=TRADE_AMOUNT,
                direction=direction,
                duration=expiry
            )

            if order and order.get("status") == "success":
                self.daily_trades += 1
                self.last_trade_time[asset] = now

                await self.send_telegram(
                    f"{'🟢 CALL' if bias=='buy' else '🔴 PUT'} {asset}\n"
                    f"${TRADE_AMOUNT} | {expiry//60} min"
                )

    async def on_candle_update(self, candle, asset):
        if self.client and self.client.connected:
            await self.check_entry(asset)

    async def run_trading(self):
        if not await self.connect():
            return

        for asset in ASSETS:
            self.client.add_candle_handler(
                asset,
                lambda c, a=asset: asyncio.create_task(self.on_candle_update(c, a))
            )

        await self.send_telegram("🤖 Bot running 24/7")

        while True:
            await asyncio.sleep(30)


bot_instance = None


async def start(update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 Bot Ready\n/demo - Demo\n/live - Live"
    )


async def demo(update, context: ContextTypes.DEFAULT_TYPE):
    bot_instance.is_demo = True
    await update.message.reply_text("Demo mode selected")
    asyncio.create_task(bot_instance.run_trading())


async def live(update, context: ContextTypes.DEFAULT_TYPE):
    bot_instance.is_demo = False
    await update.message.reply_text("⚠️ Live mode selected")
    asyncio.create_task(bot_instance.run_trading())


def main():
    global bot_instance

    application = Application.builder().token(TELEGRAM_TOKEN).build()
    bot_instance = UpgradedTradingBot(application)

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("demo", demo))
    application.add_handler(CommandHandler("live", live))

    application.run_polling()


if name == "main":
    main()
