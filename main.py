# ⚠️ SECURITY NOTE:
# Do NOT hardcode tokens in public code.
# Paste tokens only on your local machine.
# If you ever leaked tokens, ROTATE them immediately.

import asyncio
import logging
import random
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
from deriv_api import DerivAPI
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

# ========================= CONFIG =========================
DEMO_TOKEN = "tIrfitLjqeBxCOM"
REAL_TOKEN = "ZkOFWOlPtwnjqTS"
APP_ID = 1089

# ✅ markets
MARKETS = ["R_10", "R_25"]

TELEGRAM_TOKEN = "8253450930:AAHUhPk9TML-8kZlA9UaHZZvTUGdurN9MVY"
TELEGRAM_CHAT_ID = "7634818949"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ========================= TWO MODES (MANUAL SWITCHING) =========================
# M5 = 5-minute candles, 20-minute expiry (your original)
# M1 = 1-minute candles, 5-minute expiry (your request)
MODE_CONFIG = {
    "M5": {"TF_SEC": 300, "DURATION_MIN": 20},
    "M1": {"TF_SEC": 60,  "DURATION_MIN": 5},
}
DEFAULT_MODE = "M5"

CANDLES_COUNT = 220

# ========================= TRADE & RISK SETTINGS =========================
COOLDOWN_SEC = 180
MAX_TRADES_PER_DAY = 20
MAX_CONSEC_LOSSES = 6
STOP_AFTER_LOSSES = 3
DAILY_LOSS_LIMIT = -2.0
DAILY_PROFIT_TARGET = 2.0

# ========================= PAYOUT MODE =========================
USE_PAYOUT_MODE = True
PAYOUT_TARGET = 1.0
MIN_PAYOUT = 0.35
MAX_STAKE_ALLOWED = 10.00

# ========================= MARTINGALE (✅ 3 steps, 1.8x) =========================
MARTINGALE_MULT = 1.80
MARTINGALE_MAX_STEPS = 3          # steps after a loss: 1..3
MARTINGALE_HALT_ON_MAX = True     # stop bot when step hits max and loses again

# ========================= INDICATORS =========================
RSI_PERIOD = 14
RSI_CALL_MIN, RSI_CALL_MAX = 50.0, 70.0
RSI_PUT_MIN, RSI_PUT_MAX = 30.0, 50.0

ADX_PERIOD = 14
ATR_PERIOD = 14
ADX_MIN = 25.0  # (kept, but for breakout we use ADX rising instead of >= ADX_MIN)

# ========================= BREAKOUT (Donchian) =========================
DONCHIAN_LEN = 20
ATR_BREAKOUT_K = 0.15  # 0.05–0.30

# ========================= ATR-NORMALIZED FILTERS =========================
EMA_DIFF_ATR_MIN = 0.25
EMA_SLOPE_LOOKBACK = 10
EMA_SLOPE_ATR_MIN = 0.10

# Candle / spike
MIN_BODY_RATIO = 0.55
MIN_CANDLE_RANGE = 1e-6
SPIKE_RANGE_ATR = 2.5
SPIKE_BODY_ATR = 1.8

# ========================= TRADE QUALITY UPGRADES =========================
ATR_COMPRESSION_LOOKBACK = 30       # candles
ATR_COMPRESSION_MAX = 0.90          # ATR_now must be < ATR_mean * this

DONCHIAN_WIDTH_ATR_MAX = 8.0        # (Donchian width / ATR) must be <= this

FOLLOW_THROUGH_BODY_ATR_MIN = 0.60  # body/ATR in breakout direction must be >= this
# ================================================================================

# ========================= ANTI RATE-LIMIT =========================
TICKS_GLOBAL_MIN_INTERVAL = 0.45
RATE_LIMIT_BACKOFF_BASE = 20

# ========================= UI =========================
STATUS_REFRESH_COOLDOWN_SEC = 8

# ========================= INDICATOR MATH =========================
def calculate_ema(values, period: int):
    values = np.array(values, dtype=float)
    if len(values) < period:
        return np.array([])
    k = 2.0 / (period + 1.0)
    ema = np.zeros_like(values)
    ema[0] = values[0]
    for i in range(1, len(values)):
        ema[i] = values[i] * k + ema[i - 1] * (1 - k)
    return ema


def calculate_rsi(values, period=14):
    values = np.array(values, dtype=float)
    n = len(values)
    if n < period + 2:
        return np.array([])

    deltas = np.diff(values)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    rsi = np.full(n, np.nan, dtype=float)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    rs = avg_gain / (avg_loss + 1e-12)
    rsi[period] = 100.0 - (100.0 / (1.0 + rs))

    for i in range(period + 1, n):
        gain = gains[i - 1]
        loss = losses[i - 1]
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        rs = avg_gain / (avg_loss + 1e-12)
        rsi[i] = 100.0 - (100.0 / (1.0 + rs))

    return rsi


def calculate_atr(highs, lows, closes, period=14):
    highs = np.array(highs, dtype=float)
    lows = np.array(lows, dtype=float)
    closes = np.array(closes, dtype=float)

    n = len(closes)
    if n < period + 2:
        return np.array([])

    prev_close = np.roll(closes, 1)
    prev_close[0] = closes[0]

    tr = np.maximum(highs - lows, np.maximum(np.abs(highs - prev_close), np.abs(lows - prev_close)))

    atr = np.full(n, np.nan, dtype=float)
    atr[period] = np.mean(tr[1:period + 1])
    for i in range(period + 1, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    return atr


def calculate_adx(highs, lows, closes, period=14):
    highs = np.array(highs, dtype=float)
    lows = np.array(lows, dtype=float)
    closes = np.array(closes, dtype=float)

    n = len(closes)
    if n < period * 2 + 2:
        return np.array([])

    up_move = highs[1:] - highs[:-1]
    down_move = lows[:-1] - lows[1:]

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    prev_close = closes[:-1]
    tr = np.maximum(highs[1:] - lows[1:], np.maximum(np.abs(highs[1:] - prev_close), np.abs(lows[1:] - prev_close)))

    tr_s = np.zeros_like(tr)
    plus_s = np.zeros_like(plus_dm)
    minus_s = np.zeros_like(minus_dm)

    tr_s[period - 1] = np.sum(tr[:period])
    plus_s[period - 1] = np.sum(plus_dm[:period])
    minus_s[period - 1] = np.sum(minus_dm[:period])

    for i in range(period, len(tr)):
        tr_s[i] = tr_s[i - 1] - (tr_s[i - 1] / period) + tr[i]
        plus_s[i] = plus_s[i - 1] - (plus_s[i - 1] / period) + plus_dm[i]
        minus_s[i] = minus_s[i - 1] - (minus_s[i - 1] / period) + minus_dm[i]

    plus_di = 100.0 * (plus_s / (tr_s + 1e-12))
    minus_di = 100.0 * (minus_s / (tr_s + 1e-12))
    dx = 100.0 * (np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-12))

    adx = np.full(n, np.nan, dtype=float)
    dx_full = np.full(n, np.nan, dtype=float)
    dx_full[1:] = dx

    start = period * 2
    adx[start] = np.nanmean(dx_full[period:start + 1])
    for i in range(start + 1, n):
        adx[i] = ((adx[i - 1] * (period - 1)) + dx_full[i]) / period

    return adx


def build_candles_from_deriv(candles_raw):
    out = []
    for x in candles_raw:
        out.append(
            {
                "t0": int(x.get("epoch", 0)),
                "o": float(x.get("open", 0)),
                "h": float(x.get("high", 0)),
                "l": float(x.get("low", 0)),
                "c": float(x.get("close", 0)),
            }
        )
    return out


def fmt_time_hhmmss(epoch):
    try:
        return datetime.fromtimestamp(epoch, ZoneInfo("Africa/Lagos")).strftime("%H:%M:%S")
    except Exception:
        return "—"


def fmt_dt(epoch):
    try:
        return datetime.fromtimestamp(epoch, ZoneInfo("Africa/Lagos")).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "—"


def money2(x: float) -> float:
    import math
    return math.ceil(float(x) * 100.0) / 100.0


def is_finite(x) -> bool:
    return isinstance(x, (int, float)) and np.isfinite(x)


# ========================= BOT CORE =========================
class DerivBreakoutBot:
    def __init__(self):
        self.api = None
        self.app = None
        self.active_token = None
        self.account_type = "None"

        # ✅ manual switching mode
        self.mode = DEFAULT_MODE  # "M5" or "M1"

        self.is_scanning = False
        self.scanner_task = None
        self.market_tasks = {}

        # Active trade stored as dict so mode switch won't break remaining calc
        # {"cid": int, "symbol": str, "start_ts": float, "duration_min": int, "tf_sec": int, "mode": str}
        self.active_trade = None

        self.cooldown_until = 0.0
        self.trades_today = 0
        self.total_losses_today = 0
        self.consecutive_losses = 0
        self.total_profit_today = 0.0
        self.balance = "0.00"

        self.session_losses = 0

        # ✅ martingale state
        self.martingale_step = 0  # 0 = base, 1..MARTINGALE_MAX_STEPS = recovery levels
        self.martingale_halt = False

        self.trade_lock = asyncio.Lock()

        # market debug & processing must be per (symbol, mode)
        self.market_debug = {(m, md): {} for m in MARKETS for md in MODE_CONFIG.keys()}
        self.last_processed_closed_t0 = {(m, md): 0 for m in MARKETS for md in MODE_CONFIG.keys()}

        self.tz = ZoneInfo("Africa/Lagos")
        self.current_day = datetime.now(self.tz).date()
        self.pause_until = 0.0

        self._ticks_lock = asyncio.Lock()
        self._last_ticks_ts = 0.0
        self._next_poll_epoch = {(m, md): 0.0 for m in MARKETS for md in MODE_CONFIG.keys()}
        self._rate_limit_strikes = {(m, md): 0 for m in MARKETS for md in MODE_CONFIG.keys()}

        self.status_cooldown_until = 0.0

    # ---------- mode helpers ----------
    def get_mode_cfg(self) -> dict:
        return MODE_CONFIG.get(self.mode, MODE_CONFIG[DEFAULT_MODE])

    def tf_sec(self) -> int:
        return int(self.get_mode_cfg()["TF_SEC"])

    def duration_min(self) -> int:
        return int(self.get_mode_cfg()["DURATION_MIN"])

    def toggle_mode(self):
        self.mode = "M1" if self.mode == "M5" else "M5"

    @staticmethod
    def _is_gatewayish_error(msg: str) -> bool:
        m = (msg or "").lower()
        return any(
            k in m
            for k in [
                "gateway", "bad gateway", "502", "503", "504",
                "timeout", "timed out",
                "temporarily unavailable",
                "connection", "websocket", "not connected", "disconnect",
                "internal server error", "service unavailable",
            ]
        )

    @staticmethod
    def _is_rate_limit_error(msg: str) -> bool:
        m = (msg or "").lower()
        return ("rate limit" in m) or ("reached the rate limit" in m) or ("too many requests" in m) or ("429" in m)

    async def safe_send_tg(self, text: str, retries: int = 5):
        if not self.app:
            return
        last_err = None
        for i in range(1, retries + 1):
            try:
                await self.app.bot.send_message(TELEGRAM_CHAT_ID, text)
                return
            except Exception as e:
                last_err = e
                msg = str(e)
                if self._is_gatewayish_error(msg):
                    await asyncio.sleep(0.8 * i + random.random() * 0.4)
                else:
                    await asyncio.sleep(0.4 * i)
        logger.warning(f"Telegram send failed after retries: {last_err}")

    # ---------- resets ----------
    def _next_midnight_epoch(self) -> float:
        now = datetime.now(self.tz)
        next_midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        return next_midnight.timestamp()

    def _daily_reset_if_needed(self):
        today = datetime.now(self.tz).date()
        if today != self.current_day:
            self.current_day = today
            self.trades_today = 0
            self.total_losses_today = 0
            self.consecutive_losses = 0
            self.total_profit_today = 0.0
            self.cooldown_until = 0.0
            self.pause_until = 0.0
            self.session_losses = 0
            self.martingale_step = 0
            self.martingale_halt = False

    def can_auto_trade(self) -> tuple[bool, str]:
        self._daily_reset_if_needed()

        if self.martingale_halt:
            return False, f"Stopped: Martingale max step ({MARTINGALE_MAX_STEPS}) reached"

        if time.time() < self.pause_until:
            left = int(self.pause_until - time.time())
            return False, f"Paused until 12:00am WAT ({left}s)"

        if self.total_profit_today >= DAILY_PROFIT_TARGET:
            self.pause_until = self._next_midnight_epoch()
            return False, f"Daily target reached (+${self.total_profit_today:.2f})"

        if self.total_profit_today <= float(DAILY_LOSS_LIMIT):
            self.pause_until = self._next_midnight_epoch()
            return False, f"Stopped: Daily loss limit ({DAILY_LOSS_LIMIT:+.2f}) reached"

        if self.consecutive_losses >= MAX_CONSEC_LOSSES:
            return False, "Stopped: max loss streak reached"

        if self.session_losses >= STOP_AFTER_LOSSES:
            return False, f"Soft stop: {STOP_AFTER_LOSSES} losses in session. Reset by STOP/START."

        if self.trades_today >= MAX_TRADES_PER_DAY:
            return False, "Stopped: daily trade limit reached"

        if time.time() < self.cooldown_until:
            return False, f"Cooldown {int(self.cooldown_until - time.time())}s"

        if self.active_trade:
            return False, "Trade in progress"

        if not self.api:
            return False, "Not connected"

        return True, "OK"

    # ---------- Deriv connection ----------
    async def connect(self) -> bool:
        try:
            if not self.active_token:
                return False
            self.api = DerivAPI(app_id=APP_ID)
            await self.api.authorize(self.active_token)
            await self.fetch_balance()
            return True
        except Exception as e:
            logger.error(f"Connect error: {e}")
            return False

    async def safe_reconnect(self) -> bool:
        try:
            if self.api:
                try:
                    await self.api.disconnect()
                except Exception:
                    pass
        except Exception:
            pass
        self.api = None
        return await self.connect()

    async def safe_deriv_call(self, fn_name: str, payload: dict, retries: int = 6):
        last_err = None
        for attempt in range(1, retries + 1):
            try:
                if not self.api:
                    ok = await self.safe_reconnect()
                    if not ok:
                        raise RuntimeError("Reconnect failed")
                fn = getattr(self.api, fn_name)
                return await fn(payload)
            except Exception as e:
                last_err = e
                msg = str(e)
                if self._is_gatewayish_error(msg):
                    await self.safe_reconnect()
                if self._is_rate_limit_error(msg):
                    await asyncio.sleep(min(20.0, 2.5 * attempt + random.random()))
                else:
                    await asyncio.sleep(min(8.0, 0.6 * attempt + random.random() * 0.5))
        raise last_err

    async def safe_ticks_history(self, payload: dict, retries: int = 4):
        async with self._ticks_lock:
            now = time.time()
            gap = (self._last_ticks_ts + TICKS_GLOBAL_MIN_INTERVAL) - now
            if gap > 0:
                await asyncio.sleep(gap)
            self._last_ticks_ts = time.time()
        return await self.safe_deriv_call("ticks_history", payload, retries=retries)

    async def fetch_balance(self):
        if not self.api:
            return
        try:
            bal = await self.safe_deriv_call("balance", {"balance": 1}, retries=4)
            self.balance = f"{float(bal['balance']['balance']):.2f} {bal['balance']['currency']}"
        except Exception:
            pass

    # ---------- market data ----------
    async def fetch_candles(self, symbol: str, tf_sec: int):
        payload = {
            "ticks_history": symbol,
            "end": "latest",
            "count": CANDLES_COUNT,
            "style": "candles",
            "granularity": int(tf_sec),
        }
        data = await self.safe_ticks_history(payload, retries=4)
        return build_candles_from_deriv(data.get("candles", []))

    async def fetch_last_price(self, symbol: str) -> float:
        payload = {"ticks_history": symbol, "end": "latest", "count": 1, "style": "ticks"}
        data = await self.safe_ticks_history(payload, retries=3)
        try:
            ticks = data.get("history", {}).get("prices", [])
            return float(ticks[-1]) if ticks else float("nan")
        except Exception:
            return float("nan")

    # ---------- scanner ----------
    async def background_scanner(self):
        if not self.api:
            return
        self.market_tasks = {sym: asyncio.create_task(self.scan_market(sym)) for sym in MARKETS}
        try:
            while self.is_scanning:
                # safety release if Deriv doesn't return result
                if self.active_trade:
                    dur = int(self.active_trade.get("duration_min", 0))
                    if time.time() - float(self.active_trade.get("start_ts", 0.0)) > (dur * 60 + 180):
                        self.active_trade = None
                await asyncio.sleep(1)
        finally:
            for t in self.market_tasks.values():
                t.cancel()
            self.market_tasks.clear()

    def _missing_for_side(self, side: str, checks: dict) -> list[str]:
        if side == "CALL":
            needed = [
                "GATE_OK", "ATR_OK",
                "ATR_COMPRESS_OK", "DONCHIAN_SQUEEZE_OK",
                "ADX_OK", "BIAS_UP",
                "EMA50_RISING", "EMA_DIFF_OK",
                "STRONG_CANDLE", "SPIKE_OK",
                "RSI_OK_CALL",
                "BREAKOUT_CALL", "FOLLOW_THROUGH_CALL",
            ]
        else:
            needed = [
                "GATE_OK", "ATR_OK",
                "ATR_COMPRESS_OK", "DONCHIAN_SQUEEZE_OK",
                "ADX_OK", "BIAS_DOWN",
                "EMA50_FALLING", "EMA_DIFF_OK",
                "STRONG_CANDLE", "SPIKE_OK",
                "RSI_OK_PUT",
                "BREAKOUT_PUT", "FOLLOW_THROUGH_PUT",
            ]
        return [k for k in needed if not checks.get(k, False)]

    async def scan_market(self, symbol: str):
        # scan BOTH modes in the same bot, but trades only trigger for CURRENT selected mode
        for md in MODE_CONFIG.keys():
            self._next_poll_epoch[(symbol, md)] = time.time() + random.random() * 0.6

        while self.is_scanning:
            try:
                # run both modes each loop (lightweight); rate limiting already handled globally
                for md, cfg in MODE_CONFIG.items():
                    tf_sec = int(cfg["TF_SEC"])
                    now = time.time()
                    nxt = float(self._next_poll_epoch.get((symbol, md), 0.0))
                    if now < nxt:
                        continue

                    ok_gate, gate = self.can_auto_trade()

                    candles = await self.fetch_candles(symbol, tf_sec=tf_sec)
                    if len(candles) < (DONCHIAN_LEN + 80):
                        self.market_debug[(symbol, md)] = {
                            "time": time.time(),
                            "mode": md,
                            "tf_sec": tf_sec,
                            "gate": "Waiting",
                            "why": [f"Need more candles (have {len(candles)})."],
                            "checks": {},
                            "missing_call": [],
                            "missing_put": [],
                            "signal": None,
                            "last_price": float("nan"),
                            "next_close": 0,
                        }
                        self._next_poll_epoch[(symbol, md)] = time.time() + 10
                        continue

                    confirm = candles[-2]
                    confirm_t0 = int(confirm["t0"])
                    next_closed_epoch = confirm_t0 + tf_sec
                    self._next_poll_epoch[(symbol, md)] = float(next_closed_epoch + 0.30)

                    if self.last_processed_closed_t0[(symbol, md)] == confirm_t0:
                        continue

                    closes = np.array([x["c"] for x in candles], dtype=float)
                    highs = np.array([x["h"] for x in candles], dtype=float)
                    lows = np.array([x["l"] for x in candles], dtype=float)

                    ema20 = calculate_ema(closes, 20)
                    ema50 = calculate_ema(closes, 50)
                    if len(ema20) < 70 or len(ema50) < 70:
                        self.market_debug[(symbol, md)] = {
                            "time": time.time(),
                            "mode": md,
                            "tf_sec": tf_sec,
                            "gate": "Indicators",
                            "why": ["EMA not ready."],
                            "checks": {},
                            "missing_call": [],
                            "missing_put": [],
                            "signal": None,
                            "last_price": float("nan"),
                            "next_close": next_closed_epoch,
                        }
                        self.last_processed_closed_t0[(symbol, md)] = confirm_t0
                        continue

                    ema20_now = float(ema20[-2])
                    ema50_now = float(ema50[-2])
                    bias_up = ema20_now > ema50_now
                    bias_down = ema20_now < ema50_now

                    atr = calculate_atr(highs, lows, closes, ATR_PERIOD)
                    adx = calculate_adx(highs, lows, closes, ADX_PERIOD)
                    atr_now = float(atr[-2]) if len(atr) and not np.isnan(atr[-2]) else float("nan")
                    adx_now = float(adx[-2]) if len(adx) and not np.isnan(adx[-2]) else float("nan")
                    atr_ok = is_finite(atr_now) and atr_now > 1e-12

                    # ===================== ADX (BREAKOUT) =====================
                    # breakout: require ADX to be rising (momentum expansion), not necessarily already high
                    adx_prev = float(adx[-3]) if len(adx) >= 4 and not np.isnan(adx[-3]) else float("nan")
                    adx_ok = is_finite(adx_now) and is_finite(adx_prev) and (adx_now > adx_prev)
                    # ==========================================================

                    # ATR compression
                    atr_mean = float("nan")
                    atr_compress_ok = False
                    if atr_ok and len(atr) >= (ATR_COMPRESSION_LOOKBACK + 5):
                        recent = atr[-(ATR_COMPRESSION_LOOKBACK + 2):-2]
                        try:
                            atr_mean = float(np.nanmean(recent))
                        except Exception:
                            atr_mean = float("nan")
                        if is_finite(atr_mean) and atr_mean > 0:
                            atr_compress_ok = atr_now < (atr_mean * float(ATR_COMPRESSION_MAX))

                    rsi = calculate_rsi(closes, RSI_PERIOD)
                    rsi_now = float(rsi[-2]) if len(rsi) and not np.isnan(rsi[-2]) else float("nan")
                    rsi_ok_call = is_finite(rsi_now) and (RSI_CALL_MIN <= rsi_now <= RSI_CALL_MAX)
                    rsi_ok_put = is_finite(rsi_now) and (RSI_PUT_MIN <= rsi_now <= RSI_PUT_MAX)

                    ema_diff = abs(ema20_now - ema50_now)
                    ema_diff_atr = (ema_diff / atr_now) if (atr_ok and atr_now > 0) else float("nan")
                    ema_diff_ok = is_finite(ema_diff_atr) and (ema_diff_atr >= float(EMA_DIFF_ATR_MIN))

                    ema50_slope = float("nan")
                    ema50_slope_atr = float("nan")
                    slope_ok = False
                    ema50_rising = False
                    ema50_falling = False
                    if len(ema50) >= (EMA_SLOPE_LOOKBACK + 3):
                        ema50_prev2 = float(ema50[-(EMA_SLOPE_LOOKBACK + 2)])
                        ema50_slope = float(ema50_now - ema50_prev2)
                        if atr_ok and atr_now > 0:
                            ema50_slope_atr = float(ema50_slope / atr_now)
                        slope_ok = is_finite(ema50_slope_atr) and (abs(ema50_slope_atr) >= float(EMA_SLOPE_ATR_MIN))
                        ema50_rising = slope_ok and (ema50_slope_atr >= float(EMA_SLOPE_ATR_MIN))
                        ema50_falling = slope_ok and (ema50_slope_atr <= -float(EMA_SLOPE_ATR_MIN))

                    c_open = float(confirm["o"])
                    c_close = float(confirm["c"])
                    c_high = float(confirm["h"])
                    c_low = float(confirm["l"])
                    c_range = max(MIN_CANDLE_RANGE, c_high - c_low)
                    body = abs(c_close - c_open)
                    body_ratio = body / c_range
                    strong_candle = body_ratio >= float(MIN_BODY_RATIO)

                    range_atr = (c_range / atr_now) if (atr_ok and atr_now > 0) else float("nan")
                    body_atr = (body / atr_now) if (atr_ok and atr_now > 0) else float("nan")
                    spike_range_block = is_finite(range_atr) and (range_atr > float(SPIKE_RANGE_ATR))
                    spike_body_block = is_finite(body_atr) and (body_atr > float(SPIKE_BODY_ATR))
                    spike_block = spike_range_block or spike_body_block

                    window_high = float(np.max(highs[-(DONCHIAN_LEN + 2):-2]))
                    window_low = float(np.min(lows[-(DONCHIAN_LEN + 2):-2]))

                    donchian_width = float(window_high - window_low)
                    donchian_width_atr = (donchian_width / atr_now) if (atr_ok and atr_now > 0) else float("nan")
                    donchian_squeeze_ok = is_finite(donchian_width_atr) and (donchian_width_atr <= float(DONCHIAN_WIDTH_ATR_MAX))

                    buf = (float(ATR_BREAKOUT_K) * atr_now) if atr_ok else 0.0
                    call_break_level = window_high + buf
                    put_break_level = window_low - buf

                    breakout_call = c_close > call_break_level
                    breakout_put = c_close < put_break_level

                    body_dir_call_atr = ((c_close - c_open) / atr_now) if (atr_ok and atr_now > 0) else float("nan")
                    body_dir_put_atr = ((c_open - c_close) / atr_now) if (atr_ok and atr_now > 0) else float("nan")
                    follow_through_call = is_finite(body_dir_call_atr) and (body_dir_call_atr >= float(FOLLOW_THROUGH_BODY_ATR_MIN))
                    follow_through_put = is_finite(body_dir_put_atr) and (body_dir_put_atr >= float(FOLLOW_THROUGH_BODY_ATR_MIN))

                    last_price = await self.fetch_last_price(symbol)

                    checks = {
                        "GATE_OK": ok_gate,
                        "ATR_OK": atr_ok,
                        "ATR_COMPRESS_OK": atr_compress_ok,
                        "DONCHIAN_SQUEEZE_OK": donchian_squeeze_ok,
                        "ADX_OK": adx_ok,
                        "BIAS_UP": bias_up,
                        "BIAS_DOWN": bias_down,
                        "EMA_DIFF_OK": ema_diff_ok,
                        "SLOPE_OK": slope_ok,
                        "EMA50_RISING": ema50_rising,
                        "EMA50_FALLING": ema50_falling,
                        "STRONG_CANDLE": strong_candle,
                        "SPIKE_OK": (not spike_block),
                        "RSI_OK_CALL": rsi_ok_call,
                        "RSI_OK_PUT": rsi_ok_put,
                        "BREAKOUT_CALL": breakout_call,
                        "BREAKOUT_PUT": breakout_put,
                        "FOLLOW_THROUGH_CALL": follow_through_call,
                        "FOLLOW_THROUGH_PUT": follow_through_put,
                    }

                    call_ready = (
                        checks["GATE_OK"] and checks["ATR_OK"]
                        and checks["ATR_COMPRESS_OK"] and checks["DONCHIAN_SQUEEZE_OK"]
                        and checks["ADX_OK"] and checks["BIAS_UP"]
                        and checks["EMA50_RISING"] and checks["EMA_DIFF_OK"]
                        and checks["STRONG_CANDLE"] and checks["SPIKE_OK"]
                        and checks["RSI_OK_CALL"] and checks["BREAKOUT_CALL"]
                        and checks["FOLLOW_THROUGH_CALL"]
                    )
                    put_ready = (
                        checks["GATE_OK"] and checks["ATR_OK"]
                        and checks["ATR_COMPRESS_OK"] and checks["DONCHIAN_SQUEEZE_OK"]
                        and checks["ADX_OK"] and checks["BIAS_DOWN"]
                        and checks["EMA50_FALLING"] and checks["EMA_DIFF_OK"]
                        and checks["STRONG_CANDLE"] and checks["SPIKE_OK"]
                        and checks["RSI_OK_PUT"] and checks["BREAKOUT_PUT"]
                        and checks["FOLLOW_THROUGH_PUT"]
                    )

                    signal = "CALL" if call_ready else "PUT" if put_ready else None

                    missing_call = self._missing_for_side("CALL", checks)
                    missing_put = self._missing_for_side("PUT", checks)

                    why = []
                    why.append(f"Mode: {md} | Gate: {gate}")
                    why.append(f"Confirm close: {c_close:.5f} | Live price: {last_price:.5f}" if is_finite(last_price) else f"Confirm close: {c_close:.5f} | Live price: —")
                    why.append(f"Next candle closes: {fmt_time_hhmmss(next_closed_epoch)} (WAT)")
                    why.append(f"ADX {adx_now:.2f} (prev {adx_prev:.2f}) | ATR {atr_now:.5f}")
                    if atr_ok:
                        why.append(f"EMA diff/ATR {ema_diff_atr:.3f} (min {EMA_DIFF_ATR_MIN})")
                        why.append(f"EMA50 slope/ATR {ema50_slope_atr:.3f} (min {EMA_SLOPE_ATR_MIN})")
                        why.append(f"ATR compression: ATR_now {atr_now:.5f} | ATR_mean({ATR_COMPRESSION_LOOKBACK}) {atr_mean:.5f} | need < {ATR_COMPRESSION_MAX:.2f}x")
                        why.append(f"Donchian width/ATR {donchian_width_atr:.2f} (max {DONCHIAN_WIDTH_ATR_MAX:.2f})")
                        why.append(f"Follow-through: body_dir_call/ATR {body_dir_call_atr:.2f} | body_dir_put/ATR {body_dir_put_atr:.2f} (min {FOLLOW_THROUGH_BODY_ATR_MIN:.2f})")
                    why.append(f"Donchian({DONCHIAN_LEN}) H/L {window_high:.5f}/{window_low:.5f} | buffer {buf:.5f}")
                    why.append(f"CALL needs close > {call_break_level:.5f} | PUT needs close < {put_break_level:.5f}")
                    why.append(f"Body ratio {body_ratio:.2f} (min {MIN_BODY_RATIO}) | Spike? range/ATR {range_atr:.2f} body/ATR {body_atr:.2f}")
                    why.append(f"Martingale step: {self.martingale_step}/{MARTINGALE_MAX_STEPS} (x{MARTINGALE_MULT:.2f})")
                    if signal:
                        why.append(f"READY: {signal}")

                    self.market_debug[(symbol, md)] = {
                        "time": time.time(),
                        "mode": md,
                        "tf_sec": tf_sec,
                        "gate": gate,
                        "last_closed": confirm_t0,
                        "next_close": next_closed_epoch,
                        "signal": signal,
                        "checks": checks,
                        "missing_call": missing_call,
                        "missing_put": missing_put,
                        "why": why[:20],
                        "confirm_close": c_close,
                        "last_price": last_price,
                        "call_break_level": call_break_level,
                        "put_break_level": put_break_level,
                        "next_poll_epoch": self._next_poll_epoch.get((symbol, md), 0.0),
                    }

                    self.last_processed_closed_t0[(symbol, md)] = confirm_t0

                    # ✅ Only place trades for the CURRENT selected mode (manual switching)
                    if md == self.mode:
                        if call_ready:
                            await self.execute_trade("CALL", symbol, source="AUTO")
                        elif put_ready:
                            await self.execute_trade("PUT", symbol, source="AUTO")

                await asyncio.sleep(0.05)

            except asyncio.CancelledError:
                break
            except Exception as e:
                msg = str(e)
                logger.error(f"Scanner Error ({symbol}): {msg}")
                # apply backoff to BOTH modes for this symbol
                for md in MODE_CONFIG.keys():
                    if self._is_rate_limit_error(msg):
                        self._rate_limit_strikes[(symbol, md)] = int(self._rate_limit_strikes.get((symbol, md), 0)) + 1
                        backoff = RATE_LIMIT_BACKOFF_BASE * self._rate_limit_strikes[(symbol, md)]
                        backoff = min(240, backoff)
                        self._next_poll_epoch[(symbol, md)] = time.time() + backoff
                    else:
                        self._next_poll_epoch[(symbol, md)] = time.time() + (6 if self._is_gatewayish_error(msg) else 2)

    # ---------- trading (✅ MARTINGALE payout scaling) ----------
    def calc_payout_for_step(self) -> float:
        base = max(float(MIN_PAYOUT), float(PAYOUT_TARGET))
        payout = float(base) * (float(MARTINGALE_MULT) ** int(self.martingale_step))
        return money2(payout)

    async def execute_trade(self, side: str, symbol: str, source="MANUAL"):
        if not self.api or self.active_trade:
            return

        async with self.trade_lock:
            ok, gate = self.can_auto_trade()
            if not ok:
                return

            try:
                import math

                payout = self.calc_payout_for_step()
                tf_sec = self.tf_sec()
                dur_min = self.duration_min()
                mode = self.mode

                proposal_req = {
                    "proposal": 1,
                    "amount": float(payout),
                    "basis": "payout",
                    "contract_type": side,
                    "currency": "USD",
                    "duration": int(dur_min),
                    "duration_unit": "m",
                    "symbol": symbol,
                }

                prop = await self.safe_deriv_call("proposal", proposal_req, retries=6)
                if "error" in prop:
                    err = prop["error"].get("message", "Proposal error")
                    await self.safe_send_tg(f"❌ Proposal Error:\n{err}")
                    return

                p = prop["proposal"]
                proposal_id = p["id"]
                ask_price = float(p.get("ask_price", 0.0))

                if not math.isfinite(ask_price) or ask_price <= 0:
                    await self.safe_send_tg("❌ Proposal returned invalid ask_price.")
                    return

                if ask_price > float(MAX_STAKE_ALLOWED):
                    await self.safe_send_tg(
                        f"⛔️ Skipped trade: payout=${payout:.2f} needs stake=${ask_price:.2f} > max ${MAX_STAKE_ALLOWED:.2f}\n"
                        f"Martingale step: {self.martingale_step}/{MARTINGALE_MAX_STEPS}"
                    )
                    self.cooldown_until = time.time() + COOLDOWN_SEC
                    return

                buy = await self.safe_deriv_call(
                    "buy",
                    {"buy": proposal_id, "price": float(MAX_STAKE_ALLOWED)},
                    retries=6,
                )
                if "error" in buy:
                    err_msg = str(buy["error"].get("message", "Buy error"))
                    await self.safe_send_tg(f"❌ Trade Refused:\n{err_msg}")
                    return

                cid = int(buy["buy"]["contract_id"])
                self.active_trade = {
                    "cid": cid,
                    "symbol": symbol,
                    "start_ts": time.time(),
                    "duration_min": int(dur_min),
                    "tf_sec": int(tf_sec),
                    "mode": str(mode),
                }

                if source == "AUTO":
                    self.trades_today += 1

                safe_symbol = str(symbol).replace("_", " ")
                await self.safe_send_tg(
                    f"🚀 {side} TRADE OPENED\n"
                    f"🧭 Mode: {mode}\n"
                    f"🛒 Market: {safe_symbol}\n"
                    f"🕯 {int(tf_sec)//60}m candles | ⏱ Expiry: {int(dur_min)}m\n"
                    f"🎲 Martingale: step {self.martingale_step}/{MARTINGALE_MAX_STEPS} (x{MARTINGALE_MULT:.2f})\n"
                    f"🎁 Payout: ${payout:.2f}\n"
                    f"💵 Stake (Deriv): ${ask_price:.2f}\n"
                    f"🤖 Source: {source}\n"
                    f"📊 Today PnL: {self.total_profit_today:+.2f}\n"
                    f"🚦 Gate: {gate}"
                )

                asyncio.create_task(self.check_result(cid, source, dur_min))

            except Exception as e:
                logger.error(f"Trade error: {e}")
                await self.safe_send_tg(f"⚠️ Trade error:\n{e}")

    async def check_result(self, cid: int, source: str, dur_min: int):
        await asyncio.sleep(int(dur_min) * 60 + 8)
        try:
            res = await self.safe_deriv_call(
                "proposal_open_contract",
                {"proposal_open_contract": 1, "contract_id": cid},
                retries=6,
            )
            profit = float(res["proposal_open_contract"].get("profit", 0))

            if source == "AUTO":
                self.total_profit_today += profit

                if profit <= 0:
                    self.consecutive_losses += 1
                    self.total_losses_today += 1
                    self.session_losses += 1

                    if self.martingale_step < MARTINGALE_MAX_STEPS:
                        self.martingale_step += 1
                    else:
                        if MARTINGALE_HALT_ON_MAX:
                            self.martingale_halt = True
                            self.is_scanning = False

                else:
                    self.consecutive_losses = 0
                    self.session_losses = 0
                    self.martingale_step = 0
                    self.martingale_halt = False

                if self.total_profit_today >= DAILY_PROFIT_TARGET:
                    self.pause_until = self._next_midnight_epoch()

                if self.total_profit_today <= float(DAILY_LOSS_LIMIT):
                    self.pause_until = self._next_midnight_epoch()

            await self.fetch_balance()

            next_payout = self.calc_payout_for_step()

            await self.safe_send_tg(
                f"🏁 FINISH: {'WIN' if profit > 0 else 'LOSS'} ({profit:+.2f})\n"
                f"🎲 Martingale: step {self.martingale_step}/{MARTINGALE_MAX_STEPS} | Next payout: ${next_payout:.2f}\n"
                f"📊 Trades: {self.trades_today}/{MAX_TRADES_PER_DAY} | ❌ Losses: {self.total_losses_today}\n"
                f"📉 Streak: {self.consecutive_losses}/{MAX_CONSEC_LOSSES} | Session losses: {self.session_losses}/{STOP_AFTER_LOSSES}\n"
                f"💵 Today PnL: {self.total_profit_today:+.2f} (target +{DAILY_PROFIT_TARGET:.2f}, limit {DAILY_LOSS_LIMIT:+.2f})\n"
                f"💰 Balance: {self.balance}"
            )
        finally:
            self.active_trade = None
            self.cooldown_until = time.time() + COOLDOWN_SEC


# ========================= UI =========================
bot_logic = DerivBreakoutBot()

def main_keyboard():
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("▶️ START", callback_data="START_SCAN"),
                InlineKeyboardButton("⏹️ STOP", callback_data="STOP_SCAN"),
            ],
            [
                InlineKeyboardButton(f"🧭 MODE: {bot_logic.mode} (SWITCH)", callback_data="SWITCH_MODE"),
            ],
            [
                InlineKeyboardButton("📊 STATUS (DETAILED)", callback_data="STATUS"),
                InlineKeyboardButton("🔄 REFRESH", callback_data="STATUS"),
            ],
            [InlineKeyboardButton("🧪 TEST BUY", callback_data="TEST_BUY")],
            [
                InlineKeyboardButton("🧪 DEMO", callback_data="SET_DEMO"),
                InlineKeyboardButton("💰 LIVE", callback_data="SET_REAL"),
            ],
        ]
    )

# ========================= STATUS DISPLAY =========================
def _yn(v: bool) -> str:
    return "✅" if v else "❌"


def _humanize_key(k: str) -> str:
    mapping = {
        "GATE_OK": "Gate (risk limits/cooldown)",
        "ATR_OK": "ATR ready",
        "ATR_COMPRESS_OK": "ATR compression (calm before breakout)",
        "DONCHIAN_SQUEEZE_OK": "Donchian squeeze (tight range)",
        "ADX_OK": "ADX momentum expanding (rising)",
        "BIAS_UP": "Trend bias UP (EMA20 > EMA50)",
        "BIAS_DOWN": "Trend bias DOWN (EMA20 < EMA50)",
        "EMA50_RISING": "EMA50 rising strongly (slope filter)",
        "EMA50_FALLING": "EMA50 falling strongly (slope filter)",
        "EMA_DIFF_OK": "EMA separation OK (not flat)",
        "STRONG_CANDLE": "Strong confirm candle (body ratio)",
        "SPIKE_OK": "No spike candle",
        "RSI_OK_CALL": "RSI ok for CALL",
        "RSI_OK_PUT": "RSI ok for PUT",
        "BREAKOUT_CALL": "Breakout above CALL level",
        "BREAKOUT_PUT": "Breakout below PUT level",
        "FOLLOW_THROUGH_CALL": "Follow-through CALL (body/ATR)",
        "FOLLOW_THROUGH_PUT": "Follow-through PUT (body/ATR)",
        "SLOPE_OK": "EMA50 slope strong enough",
    }
    return mapping.get(k, k)


def _regime_from_checks(checks: dict) -> str:
    adx_ok = bool(checks.get("ADX_OK", False))
    sep_ok = bool(checks.get("EMA_DIFF_OK", False))
    rising = bool(checks.get("EMA50_RISING", False))
    falling = bool(checks.get("EMA50_FALLING", False))
    if adx_ok and sep_ok and (rising or falling):
        return "TREND"
    return "RANGE"


def _breakout_state(last_price: float, call_lvl: float, put_lvl: float, checks: dict) -> tuple[str, str]:
    b_call = bool(checks.get("BREAKOUT_CALL", False))
    b_put = bool(checks.get("BREAKOUT_PUT", False))

    if b_call:
        return ("BROKE OUT ✅ (CALL breakout confirmed)", "")
    if b_put:
        return ("BROKE OUT ✅ (PUT breakout confirmed)", "")

    if not is_finite(last_price) or not is_finite(call_lvl) or not is_finite(put_lvl):
        return ("NO BREAKOUT (no live price yet)", "")

    near = abs(last_price) * 0.0015
    dist_to_call = call_lvl - last_price
    dist_to_put = last_price - put_lvl

    approaching = []
    if dist_to_call >= 0 and dist_to_call <= near:
        approaching.append(f"Approaching CALL breakout (≈ {dist_to_call:.5f} away)")
    if dist_to_put >= 0 and dist_to_put <= near:
        approaching.append(f"Approaching PUT breakout (≈ {dist_to_put:.5f} away)")

    if approaching:
        return ("APPROACHING ⚠️ (near breakout level)", " | ".join(approaching))

    return ("NO BREAKOUT (still inside range)", "")


def _focus_side_from_checks(checks: dict) -> str:
    if bool(checks.get("BIAS_UP", False)) and not bool(checks.get("BIAS_DOWN", False)):
        return "CALL"
    if bool(checks.get("BIAS_DOWN", False)) and not bool(checks.get("BIAS_UP", False)):
        return "PUT"
    return "BOTH"


def _waiting_list_for_side(side: str, checks: dict) -> list[str]:
    if side == "CALL":
        needed = [
            "GATE_OK", "ATR_OK",
            "ATR_COMPRESS_OK", "DONCHIAN_SQUEEZE_OK",
            "ADX_OK", "BIAS_UP",
            "EMA50_RISING", "EMA_DIFF_OK",
            "STRONG_CANDLE", "SPIKE_OK",
            "RSI_OK_CALL", "BREAKOUT_CALL",
            "FOLLOW_THROUGH_CALL",
        ]
    else:
        needed = [
            "GATE_OK", "ATR_OK",
            "ATR_COMPRESS_OK", "DONCHIAN_SQUEEZE_OK",
            "ADX_OK", "BIAS_DOWN",
            "EMA50_FALLING", "EMA_DIFF_OK",
            "STRONG_CANDLE", "SPIKE_OK",
            "RSI_OK_PUT", "BREAKOUT_PUT",
            "FOLLOW_THROUGH_PUT",
        ]
    return [k for k in needed if not bool(checks.get(k, False))]


def format_market_detail(sym: str, d: dict) -> str:
    if not d:
        return f"📍 {sym.replace('_',' ')}\n⏳ No scan data yet\n"

    age = int(time.time() - d.get("time", time.time()))
    gate = d.get("gate", "—")
    last_closed = int(d.get("last_closed", 0))
    next_close = int(d.get("next_close", 0))
    signal = d.get("signal") or "—"
    mode = d.get("mode", "—")
    tf_sec = int(d.get("tf_sec", 0) or 0)

    checks = d.get("checks", {}) or {}

    def f(x, fmt=".5f"):
        return "—" if (x is None or (isinstance(x, float) and not np.isfinite(x))) else format(float(x), fmt)

    next_poll_epoch = float(d.get("next_poll_epoch", 0.0))
    next_poll_in = max(0, int(next_poll_epoch - time.time())) if next_poll_epoch else 0

    confirm_close = d.get("confirm_close", float("nan"))
    last_price = d.get("last_price", float("nan"))
    call_lvl = d.get("call_break_level", float("nan"))
    put_lvl = d.get("put_break_level", float("nan"))

    regime = _regime_from_checks(checks)
    bo_state, bo_approach = _breakout_state(last_price, call_lvl, put_lvl, checks)
    focus = _focus_side_from_checks(checks)

    missing_call = _waiting_list_for_side("CALL", checks)
    missing_put = _waiting_list_for_side("PUT", checks)

    if focus == "CALL":
        waiting_keys = missing_call
        focus_line = "Focus: CALL (bias UP)"
    elif focus == "PUT":
        waiting_keys = missing_put
        focus_line = "Focus: PUT (bias DOWN)"
    else:
        focus_side = "CALL" if len(missing_call) <= len(missing_put) else "PUT"
        waiting_keys = missing_call if focus_side == "CALL" else missing_put
        focus_line = f"Focus: BOTH (mixed bias) | Nearest: {focus_side}"

    waiting_pretty = ", ".join([_humanize_key(k) for k in waiting_keys]) if waiting_keys else "NONE ✅ (trade should trigger)"

    summary = [f"Regime: {regime}", f"Breakout: {bo_state}"]
    if bo_approach:
        summary.append(bo_approach)

    key_checks = [
        ("Gate", "GATE_OK"),
        ("ADX", "ADX_OK"),
        ("Compress", "ATR_COMPRESS_OK"),
        ("Squeeze", "DONCHIAN_SQUEEZE_OK"),
        ("EMA sep", "EMA_DIFF_OK"),
        ("Slope up", "EMA50_RISING"),
        ("Slope dn", "EMA50_FALLING"),
        ("Candle", "STRONG_CANDLE"),
        ("FollowC", "FOLLOW_THROUGH_CALL"),
        ("FollowP", "FOLLOW_THROUGH_PUT"),
        ("Spike", "SPIKE_OK"),
        ("RSI call", "RSI_OK_CALL"),
        ("RSI put", "RSI_OK_PUT"),
        ("BO call", "BREAKOUT_CALL"),
        ("BO put", "BREAKOUT_PUT"),
    ]
    checks_line = " | ".join([f"{name}:{_yn(bool(checks.get(k, False)))}" for name, k in key_checks])

    tf_txt = f"{tf_sec//60}m" if tf_sec else "—"

    return (
        f"📍 {sym.replace('_',' ')} | 🧭 {mode} | 🕯 {tf_txt} ({age}s)\n"
        f"Gate: {gate} | Next scan in: {next_poll_in}s\n"
        f"Last closed: {fmt_time_hhmmss(last_closed)} | Next close: {fmt_time_hhmmss(next_close)}\n"
        f"Live: {f(last_price)} | Confirm close: {f(confirm_close)}\n"
        f"Break levels → CALL: {f(call_lvl)} | PUT: {f(put_lvl)}\n"
        f"Signal: {signal}\n"
        f"━━━━━━━━━━━━━━\n"
        f"{' | '.join(summary)}\n"
        f"{focus_line}\n"
        f"WAITING FOR: {waiting_pretty}\n"
        f"Checks: {checks_line}\n"
    )
# ========================= END STATUS DISPLAY =========================

async def _safe_answer(q, text: str | None = None, show_alert: bool = False):
    try:
        await q.answer(text=text, show_alert=show_alert)
    except Exception as e:
        logger.warning(f"Callback answer ignored: {e}")

async def _safe_edit(q, text: str, reply_markup=None):
    try:
        await q.edit_message_text(text, reply_markup=reply_markup)
    except Exception as e:
        logger.warning(f"Edit failed: {e}")

async def btn_handler(u: Update, c: ContextTypes.DEFAULT_TYPE):
    q = u.callback_query
    await _safe_answer(q)
    await _safe_edit(q, "⏳ Working...", reply_markup=main_keyboard())

    if q.data == "SET_DEMO":
        bot_logic.active_token, bot_logic.account_type = DEMO_TOKEN, "DEMO"
        ok = await bot_logic.connect()
        await _safe_edit(q, "✅ Connected to DEMO" if ok else "❌ DEMO Failed", reply_markup=main_keyboard())

    elif q.data == "SET_REAL":
        bot_logic.active_token, bot_logic.account_type = REAL_TOKEN, "LIVE"
        ok = await bot_logic.connect()
        await _safe_edit(q, "✅ LIVE CONNECTED" if ok else "❌ LIVE Failed", reply_markup=main_keyboard())

    elif q.data == "SWITCH_MODE":
        bot_logic.toggle_mode()
        cfg = bot_logic.get_mode_cfg()
        await _safe_edit(
            q,
            f"✅ Mode switched to {bot_logic.mode}\n"
            f"🕯 Candles: {int(cfg['TF_SEC'])//60}m | ⏱ Expiry: {int(cfg['DURATION_MIN'])}m\n"
            f"Trades will trigger ONLY for the selected mode.",
            reply_markup=main_keyboard(),
        )

    elif q.data == "START_SCAN":
        if not bot_logic.api:
            await _safe_edit(q, "❌ Connect first.", reply_markup=main_keyboard())
            return
        bot_logic.session_losses = 0
        bot_logic.martingale_halt = False
        bot_logic.is_scanning = True
        bot_logic.scanner_task = asyncio.create_task(bot_logic.background_scanner())

        cfg = bot_logic.get_mode_cfg()
        await _safe_edit(
            q,
            f"🔍 SCANNER ACTIVE\n"
            f"🧭 Selected mode: {bot_logic.mode} (manual switch)\n"
            f"✅ Donchian({DONCHIAN_LEN}) breakout + ATR buffer(k={ATR_BREAKOUT_K})\n"
            f"🕯 {int(cfg['TF_SEC'])//60}m candles | ⏱ {int(cfg['DURATION_MIN'])}m expiry\n"
            f"🎲 Martingale: {MARTINGALE_MAX_STEPS} steps | x{MARTINGALE_MULT:.2f}\n"
            f"📌 Note: Bot scans BOTH M1 & M5, but only trades the selected mode.",
            reply_markup=main_keyboard(),
        )

    elif q.data == "STOP_SCAN":
        bot_logic.is_scanning = False
        if bot_logic.scanner_task and not bot_logic.scanner_task.done():
            bot_logic.scanner_task.cancel()
        await _safe_edit(q, "⏹️ Scanner stopped.", reply_markup=main_keyboard())

    elif q.data == "TEST_BUY":
        asyncio.create_task(bot_logic.execute_trade("CALL", MARKETS[0], source="MANUAL"))
        await _safe_edit(
            q,
            f"🧪 Test trade triggered (CALL {MARKETS[0].replace('_',' ')} | Mode {bot_logic.mode}).",
            reply_markup=main_keyboard(),
        )

    elif q.data == "STATUS":
        now = time.time()
        if now < bot_logic.status_cooldown_until:
            left = int(bot_logic.status_cooldown_until - now)
            await _safe_edit(q, f"⏳ Refresh cooldown: {left}s\n\nPress again after cooldown.", reply_markup=main_keyboard())
            return
        bot_logic.status_cooldown_until = now + STATUS_REFRESH_COOLDOWN_SEC

        bot_logic._daily_reset_if_needed()
        await bot_logic.fetch_balance()

        now_time = datetime.now(ZoneInfo("Africa/Lagos")).strftime("%Y-%m-%d %H:%M:%S")
        _ok, gate = bot_logic.can_auto_trade()

        cfg = bot_logic.get_mode_cfg()

        trade_status = "No Active Trade"
        if bot_logic.active_trade and bot_logic.api:
            try:
                cid = int(bot_logic.active_trade["cid"])
                res = await bot_logic.safe_deriv_call(
                    "proposal_open_contract",
                    {"proposal_open_contract": 1, "contract_id": cid},
                    retries=4,
                )
                pnl = float(res["proposal_open_contract"].get("profit", 0))
                dur = int(bot_logic.active_trade.get("duration_min", 0))
                rem = max(0, int(dur * 60) - int(time.time() - float(bot_logic.active_trade.get("start_ts", 0.0))))
                icon = "✅ PROFIT" if pnl > 0 else "❌ LOSS" if pnl < 0 else "➖ FLAT"
                mkt_clean = str(bot_logic.active_trade.get("symbol", "—")).replace("_", " ")
                trade_status = (
                    f"🚀 Active Trade ({mkt_clean})\n"
                    f"Mode: {bot_logic.active_trade.get('mode','—')} | Candles: {int(bot_logic.active_trade.get('tf_sec',0))//60}m | Expiry: {dur}m\n"
                    f"Opened: {fmt_dt(int(bot_logic.active_trade.get('start_ts', time.time())))}\n"
                    f"📈 PnL: {icon} ({pnl:+.2f})\n"
                    f"⏳ Remaining: {rem}s"
                )
            except Exception:
                trade_status = "🚀 Active Trade: Syncing..."

        cooldown_left = max(0, int(bot_logic.cooldown_until - time.time()))
        pause_left = max(0, int(bot_logic.pause_until - time.time()))
        next_payout = bot_logic.calc_payout_for_step()

        header = (
            f"🕒 Time (WAT): {now_time}\n"
            f"🤖 Bot: {'ACTIVE' if bot_logic.is_scanning else 'OFFLINE'} ({bot_logic.account_type})\n"
            f"🧭 Mode: {bot_logic.mode}\n"
            f"🕯 Candles: {int(cfg['TF_SEC'])//60}m | ⏱ Expiry: {int(cfg['DURATION_MIN'])}m\n"
            f"🎲 Martingale: step {bot_logic.martingale_step}/{MARTINGALE_MAX_STEPS} (x{MARTINGALE_MULT:.2f}) | Next payout: ${next_payout:.2f}\n"
            f"🧊 Cooldown: {cooldown_left}s left (base {COOLDOWN_SEC}s)\n"
            f"⏸ Pause: {pause_left}s left\n"
            f"🧯 Risk: max/day {MAX_TRADES_PER_DAY} | max streak {MAX_CONSEC_LOSSES} | soft stop {STOP_AFTER_LOSSES}\n"
            f"🎯 Daily: target +{DAILY_PROFIT_TARGET:.2f} | limit {DAILY_LOSS_LIMIT:+.2f}\n"
            f"━━━━━━━━━━━━━━━\n{trade_status}\n━━━━━━━━━━━━━━━\n"
            f"💵 Today PnL: {bot_logic.total_profit_today:+.2f}\n"
            f"🎯 Trades: {bot_logic.trades_today}/{MAX_TRADES_PER_DAY} | ❌ Losses: {bot_logic.total_losses_today}\n"
            f"📉 Streak: {bot_logic.consecutive_losses}/{MAX_CONSEC_LOSSES} | Session losses: {bot_logic.session_losses}/{STOP_AFTER_LOSSES}\n"
            f"🚦 Gate: {gate}\n"
            f"💰 Balance: {bot_logic.balance}\n"
            f"📡 Markets: {', '.join(MARKETS).replace('_',' ')}\n"
        )

        # show BOTH mode scans so you can compare quickly
        details = "\n\n📌 LIVE SCAN (M1 + M5 • BREAKOUT • WAITING-FOR)\n\n"
        blocks = []
        for md in ("M1", "M5"):
            blocks.append(f"===== {md} =====")
            for sym in MARKETS:
                blocks.append(format_market_detail(sym, bot_logic.market_debug.get((sym, md), {})))
        details += "\n\n".join(blocks)

        await _safe_edit(q, header + details, reply_markup=main_keyboard())

async def start_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    cfg = bot_logic.get_mode_cfg()
    await u.message.reply_text(
        "💎 Deriv Breakout Bot (M1 + M5)\n"
        f"🧭 Mode: {bot_logic.mode} (use MODE button to switch)\n"
        f"🕯 Timeframe: {int(cfg['TF_SEC'])//60}m | ⏱ Expiry: {int(cfg['DURATION_MIN'])}m\n"
        f"📌 Donchian({DONCHIAN_LEN}) breakout + ATR buffer + RSI + ADX(rising)\n"
        f"🎲 Martingale: {MARTINGALE_MAX_STEPS} steps | x{MARTINGALE_MULT:.2f}\n",
        reply_markup=main_keyboard(),
    )

if __name__ == "__main__":
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    bot_logic.app = app
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CallbackQueryHandler(btn_handler))
    app.run_polling(drop_pending_updates=True)
