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
MIN_EXPIRY = 180   # 3 minutes
MAX_EXPIRY = 300   # 5 minutes

MAX_TRADES_PER_DAY = 5
MAX_CONSECUTIVE_LOSSES = 2

COOLDOWN_SECONDS = 60
# ===================================================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


# ================= PURE PYTHON INDICATORS (NO TA-LIB) =================
def ema_last(values: np.ndarray, period: int):
    """Return last EMA value for array; None if not enough data."""
    if values is None or len(values) < period:
        return None
    k = 2 / (period + 1)
    ema_val = float(values[0])
    for price in values[1:]:
        ema_val = float(price) * k + ema_val * (1 - k)
    return ema_val


def rsi_last(values: np.ndarray, period: int = 14):
    """Return last RSI value; None if not enough data."""
    if values is None or len(values) < period + 1:
        return None

    # Use last (period+1) points
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


def atr_last(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14):
    """Return last ATR value; None if not enough data."""
    if closes is None or len(closes) < period + 1:
        return None

    h = highs[-(period + 1):]
    l = lows[-(period + 1):]
    c = closes[-(period + 1):]

    trs = []
    for i in range(1, len(c)):
        tr = max(
            float(h[i] - l[i]),
            abs(float(h[i] - c[i - 1])),
            abs(float(l[i] - c[i - 1])),
        )
        trs.append(tr)

    return float(np.mean(trs)) if trs else None
# =====================================================================


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
            logging.error(f"Telegram error: {e}")

    def reset_daily_limits(self):
        today = datetime.date.today()
        if today != self.current_date:
            self.daily_trades = 0
            self.consec_losses = 0
            self.current_date = today

    async def can_place_trade(self):
        self.reset_daily_limits()
        if self.daily_trades >= MAX_TRADES_PER_DAY:
            await self.send_telegram("🛑 Max 5 trades per day reached. Bot paused until tomorrow.")
            return False
        if self.consec_losses >= MAX_CONSECUTIVE_LOSSES:
            await self.send_telegram("🛑 Max consecutive losses reached. Bot paused until next session.")
            return False
        return True

    async def connect(self):
        if not SSID or "PASTE_" in SSID:
            await self.send_telegram("❌ SSID missing. Paste your PocketOption SSID in main.py and redeploy.")
            return False
try:
            self.client = AsyncPocketOptionClient(ssid=SSID, is_demo=self.is_demo)
            await self.client.connect()

            if not getattr(self.client, "connected", False):
                await self.send_telegram("❌ Connection failed. Check SSID and try again.")
                return False

            balance = await self.client.get_balance()
            mode = "DEMO" if self.is_demo else "REAL (LIVE)"
            await self.send_telegram(
                f"🚀 Connected!\nMode: {mode}\nBalance: ${balance}\nMonitoring {len(ASSETS)} assets..."
            )

            for asset in ASSETS:
                await self.client.subscribe_candles(asset, 300)  # 5m candles stream
            return True

        except Exception as e:
            s = str(e).lower()
            if "auth" in s or "unauthorized" in s:
                await self.send_telegram("⚠️ SESSION EXPIRED. Get a new SSID and update main.py.")
            else:
                await self.send_telegram(f"⚠️ Connection error: {e}")
            return False

    async def get_candles(self, asset, timeframe, count=220):
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

    async def get_1h_bias(self, asset):
        data = await self.get_candles(asset, 3600, 260)
        if not data:
            return None

        o, h, l, c = data["o"], data["h"], data["l"], data["c"]

        ema200_now = ema_last(c, 200)
        ema200_prev = ema_last(c[:-1], 200) if len(c) > 201 else None
        if ema200_now is None or ema200_prev is None:
            return None

        slope = ema200_now - ema200_prev
        is_up_trend = slope > 0.0001
        is_down_trend = slope < -0.0001

        peaks, _ = find_peaks(h, distance=4)
        valleys, _ = find_peaks(-l, distance=4)
        if len(peaks) < 2 or len(valleys) < 2:
            return None

        bull_structure = h[peaks[-1]] > h[peaks[-2]] and l[valleys[-1]] > l[valleys[-2]]
        bear_structure = h[peaks[-1]] < h[peaks[-2]] and l[valleys[-1]] < l[valleys[-2]]

        last_range = h[-1] - l[-1]
        if last_range <= 0:
            return None

        last_bullish = c[-1] > o[-1] and abs(c[-1] - o[-1]) > 0.5 * last_range
        last_bearish = c[-1] < o[-1] and abs(o[-1] - c[-1]) > 0.5 * last_range

        if bull_structure and c[-1] > ema200_now and is_up_trend and last_bullish:
            return "buy"
        if bear_structure and c[-1] < ema200_now and is_down_trend and last_bearish:
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

            data_5m = await self.get_candles(asset, 300, 220)
            if not data_5m:
                return

            o, h, l, c = data_5m["o"], data_5m["h"], data_5m["l"], data_5m["c"]

            # Find zone from last 30 candles
            valleys, _ = find_peaks(-l[-30:], distance=3)
            peaks, _ = find_peaks(h[-30:], distance=3)

            zone = None
            if bias == "buy" and len(valleys) > 0:
                zone = l[-30:][valleys[-1]]
            elif bias == "sell" and len(peaks) > 0:
                zone = h[-30:][peaks[-1]]
            if zone is None:
                return
atr14 = atr_last(h, l, c, 14)
            if atr14 is None or atr14 <= 0:
                return

            tolerance = atr14 * 0.5
            near_zone = (min(l[-3:]) <= zone + tolerance) and (max(h[-3:]) >= zone - tolerance)
            if not near_zone:
                return

            # Triggers
            is_engulfing = is_pinbar = is_micro_bos = False
            strength = "moderate"

            body = abs(c[-1] - o[-1])
            candle_range = h[-1] - l[-1]
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
            ema200_5m = ema_last(c, 200)
            rsi14 = rsi_last(c, 14)

            if ema200_5m is None or rsi14 is None:
                return

            prev_body = abs(c[-2] - o[-2])

            if bias == "buy":
                if c[-1] <= ema200_5m or not (45 <= rsi14 <= 65):
                    return
            else:
                if c[-1] >= ema200_5m or not (35 <= rsi14 <= 55):
                    return

            if body < 0.7 * atr14 or body <= prev_body:
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

    async def on_candle_update(self, candle, asset):
        if self.client and getattr(self.client, "connected", False):
            await self.check_entry(asset)

    async def run_trading(self):
        if not await self.connect():
            return

        for asset in ASSETS:
            self.client.add_candle_handler(
                asset, lambda c, a=asset: asyncio.create_task(self.on_candle_update(c, a))
            )

        await self.send_telegram("🤖 Bot is now monitoring the market 24/7...")
        while True:
            await asyncio.sleep(30)


bot_instance: UpgradedTradingBot | None = None


async def start(update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 Welcome!\n\nChoose mode:\n"
        "🔹 /demo - Practice\n"
        "🔸 /live - Real money\n"
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


def main():
    global bot_instance

    if "PASTE_" in TELEGRAM_TOKEN or not TELEGRAM_TOKEN:
        raise RuntimeError("Paste your TELEGRAM_TOKEN in main.py before deploying.")
    if "PASTE_" in TELEGRAM_CHAT_ID or not TELEGRAM_CHAT_ID:
        raise RuntimeError("Paste your TELEGRAM_CHAT_ID in main.py before deploying.")

    application = Application.builder().token(TELEGRAM_TOKEN).build()
    bot_instance = UpgradedTradingBot(application)

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("demo", demo))
    application.add_handler(CommandHandler("live", live))

    application.run_polling()


if name == "main":
    main()
