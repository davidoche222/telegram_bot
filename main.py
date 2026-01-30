# ⚠️ SECURITY NOTE:
# DO NOT hardcode Deriv / Telegram tokens in public code.
# Put them only on your local machine (env vars / config file).
#
# This version keeps your UI (STATUS button etc.) and uses a
# "powerful but not too restrictive" strategy + a safer martingale buy flow.

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

MARKETS = ["R_10", "R_25"]

# Trade pacing
COOLDOWN_SEC = 120                 # faster trading than 120s
MAX_TRADES_PER_DAY = 60
MAX_CONSEC_LOSSES = 10

# Telegram
TELEGRAM_TOKEN = "8253450930:AAHUhPk9TML-8kZlA9UaHZZvTUGdurN9MVY"
TELEGRAM_CHAT_ID = "7634818949"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ========================= STRATEGY SETTINGS =========================
TF_SEC = 60            # M1 candles
CANDLES_COUNT = 150    # enough for EMA/ADX/ATR

RSI_PERIOD = 14
DURATION_MIN = 2       # 2-minute expiry (more stable than 1m)

# EMA slope
EMA_SLOPE_LOOKBACK = 10
EMA_SLOPE_MIN = 0.20   # avoid flat conditions

# ========================= TARGETS / SAFETY =========================
DAILY_PROFIT_TARGET = 5
DAILY_LOSS_LIMIT = -2.0            # optional stop for the day

# Instead of pausing whole section, do a short pause after a loss streak
LOSS_STREAK_PAUSE_SEC = 10 * 60    # 10 minutes
LOSS_STREAK_PAUSE_AT = 3

# ========================= PAYOUT / MARTINGALE =========================
USE_PAYOUT_MODE = True
PAYOUT_TARGET = 1.00
MIN_PAYOUT = 0.35

# If your goal is ONLY to recover losses, 1.65 is a safer recovery-style multiplier
MARTINGALE_MULT = 1.8
MARTINGALE_MAX_STEPS = 4

# Buy protections
MAX_STAKE_ALLOWED = 10.00
BUY_PRICE_BUFFER = 0.05            # buffer above ask_price to reduce "price moved" refusals

# ========================= CANDLE QUALITY (NOT TOO STRICT) =========================
MIN_BODY_RATIO = 0.45              # not too strict
MIN_CANDLE_RANGE = 1e-6

# ========================= FILTERS =========================
# Spike block prevents entering right after unusually large candle body
SPIKE_MULT = 1.6

# Flat trend block (EMA20/EMA50 too close)
EMA_DIFF_MIN = 0.25

# RSI windows (balanced: not too tight)
RSI_CALL_MIN, RSI_CALL_MAX = 42.0, 68.0
RSI_PUT_MIN,  RSI_PUT_MAX  = 32.0, 58.0

# Close position filter (acts like "breakout strength" but more frequent)
# Call wants close in top 25% of candle range; Put wants bottom 25%
CLOSE_POS_CALL_MIN = 0.72
CLOSE_POS_PUT_MAX  = 0.28

# ========================= ADX + ATR FILTERS =========================
ADX_PERIOD = 14
ATR_PERIOD = 14

ADX_MIN = 22.0         # stronger than 20 but still allows trades
ATR_MIN = 0.0          # show ATR but don't block by ATR unless you raise it

# "EITHER" => passes if ADX ok OR ATR ok (ATR ok is almost always true if ATR_MIN=0.0)
# To make ADX matter, keep ATR_MIN=0.0 but TREND_FILTER_MODE="BOTH" OR raise ATR_MIN slightly
TREND_FILTER_MODE = "EITHER"

# ========================= ANTI RATE-LIMIT =========================
TICKS_GLOBAL_MIN_INTERVAL = 0.35
RATE_LIMIT_BACKOFF_BASE = 20

# ========================= UI: REFRESH COOLDOWN =========================
STATUS_REFRESH_COOLDOWN_SEC = 10

# ========================= UTIL =========================
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


def fmt_hhmm(epoch):
    try:
        return datetime.fromtimestamp(epoch, ZoneInfo("Africa/Lagos")).strftime("%H:%M")
    except Exception:
        return "—"


def money2(x: float) -> float:
    import math
    return math.ceil(float(x) * 100.0) / 100.0


# ========================= BOT CORE =========================
class DerivSniperBot:
    def __init__(self):
        self.api = None
        self.app = None
        self.active_token = None
        self.account_type = "None"

        self.is_scanning = False
        self.scanner_task = None
        self.market_tasks = {}

        self.active_trade_info = None
        self.active_market = "None"
        self.trade_start_time = 0.0

        self.cooldown_until = 0.0
        self.trades_today = 0
        self.total_losses_today = 0
        self.consecutive_losses = 0
        self.total_profit_today = 0.0
        self.balance = "0.00"

        self.current_stake = 0.0
        self.martingale_step = 0
        self.martingale_halt = False

        # Soft pause until timestamp (used for loss-streak pause)
        self.soft_pause_until = 0.0

        self.trade_lock = asyncio.Lock()

        self.market_debug = {m: {} for m in MARKETS}
        self.last_processed_closed_t0 = {m: 0 for m in MARKETS}

        self.tz = ZoneInfo("Africa/Lagos")
        self.current_day = datetime.now(self.tz).date()
        self.pause_until = 0.0

        # anti rate-limit state
        self._ticks_lock = asyncio.Lock()
        self._last_ticks_ts = 0.0
        self._next_poll_epoch = {m: 0.0 for m in MARKETS}
        self._rate_limit_strikes = {m: 0 for m in MARKETS}

        # refresh cooldown
        self.status_cooldown_until = 0.0

    # ---------- helpers ----------
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

    # ---------- Daily reset ----------
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
            self.soft_pause_until = 0.0

            self.martingale_step = 0
            self.current_stake = 0.0
            self.martingale_halt = False

    def can_auto_trade(self) -> tuple[bool, str]:
        self._daily_reset_if_needed()

        if self.martingale_halt:
            return False, f"Stopped: Martingale {MARTINGALE_MAX_STEPS} steps completed"

        now = time.time()

        if now < self.soft_pause_until:
            left = int(self.soft_pause_until - now)
            return False, f"Soft pause after losses ({left}s)"

        if now < self.pause_until:
            left = int(self.pause_until - now)
            return False, f"Paused until 12:00am WAT ({left}s)"

        if self.total_profit_today >= DAILY_PROFIT_TARGET:
            self.pause_until = self._next_midnight_epoch()
            return False, f"Daily target reached (+${self.total_profit_today:.2f})"

        if self.total_profit_today <= float(DAILY_LOSS_LIMIT):
            self.pause_until = self._next_midnight_epoch()
            return False, f"Stopped: Daily loss limit ({DAILY_LOSS_LIMIT:.2f}) reached"

        if self.consecutive_losses >= MAX_CONSEC_LOSSES:
            return False, "Stopped: max loss streak reached"

        if self.trades_today >= MAX_TRADES_PER_DAY:
            return False, "Stopped: daily trade limit reached"

        if now < self.cooldown_until:
            return False, f"Cooldown {int(self.cooldown_until - now)}s"

        if self.active_trade_info:
            return False, "Trade in progress"

        if not self.api:
            return False, "Not connected"

        return True, "OK"

    # ---------- Scanner loop ----------
    async def background_scanner(self):
        if not self.api:
            return
        self.market_tasks = {sym: asyncio.create_task(self.scan_market(sym)) for sym in MARKETS}
        try:
            while self.is_scanning:
                # safety: clear stuck trade after expiry+grace
                if self.active_trade_info and (time.time() - self.trade_start_time > (DURATION_MIN * 60 + 120)):
                    self.active_trade_info = None
                await asyncio.sleep(1)
        finally:
            for t in self.market_tasks.values():
                t.cancel()
            self.market_tasks.clear()

    async def fetch_real_m1_candles(self, symbol: str):
        payload = {
            "ticks_history": symbol,
            "end": "latest",
            "count": CANDLES_COUNT,
            "style": "candles",
            "granularity": TF_SEC,
        }
        data = await self.safe_ticks_history(payload, retries=4)
        return build_candles_from_deriv(data.get("candles", []))

    async def scan_market(self, symbol: str):
        self._next_poll_epoch[symbol] = time.time() + random.random() * 0.5

        while self.is_scanning:
            try:
                now = time.time()
                nxt = float(self._next_poll_epoch.get(symbol, 0.0))
                if now < nxt:
                    await asyncio.sleep(min(1.0, nxt - now))
                    continue

                # hard stop if risk limits hit
                if self.consecutive_losses >= MAX_CONSEC_LOSSES or self.trades_today >= MAX_TRADES_PER_DAY:
                    self.is_scanning = False
                    break

                ok_gate, gate = self.can_auto_trade()

                candles = await self.fetch_real_m1_candles(symbol)
                if len(candles) < 80:
                    self.market_debug[symbol] = {
                        "time": time.time(),
                        "gate": "Waiting for more candles",
                        "why": [f"Need more candle history (have {len(candles)})."],
                    }
                    self._next_poll_epoch[symbol] = time.time() + 10
                    continue

                pullback = candles[-3]
                confirm = candles[-2]
                confirm_t0 = int(confirm["t0"])

                # poll shortly after candle close
                next_closed_epoch = confirm_t0 + TF_SEC
                self._next_poll_epoch[symbol] = float(next_closed_epoch + 0.20)

                if self.last_processed_closed_t0[symbol] == confirm_t0:
                    continue

                closes = [x["c"] for x in candles]
                highs = [x["h"] for x in candles]
                lows = [x["l"] for x in candles]

                ema20_arr = calculate_ema(closes, 20)
                ema50_arr = calculate_ema(closes, 50)
                if len(ema20_arr) < 70 or len(ema50_arr) < 70:
                    self.market_debug[symbol] = {
                        "time": time.time(),
                        "gate": "Indicators",
                        "why": ["EMA20/EMA50 not ready yet."],
                    }
                    self.last_processed_closed_t0[symbol] = confirm_t0
                    continue

                ema20_pullback = float(ema20_arr[-3])
                ema20_confirm = float(ema20_arr[-2])
                ema50_confirm = float(ema50_arr[-2])

                # slope
                slope_ok = False
                ema50_slope = 0.0
                ema50_rising = False
                ema50_falling = False
                if len(ema50_arr) >= (EMA_SLOPE_LOOKBACK + 3):
                    ema50_prev = float(ema50_arr[-(EMA_SLOPE_LOOKBACK + 2)])
                    ema50_slope = ema50_confirm - ema50_prev
                    ema50_rising = ema50_slope > EMA_SLOPE_MIN
                    ema50_falling = ema50_slope < -EMA_SLOPE_MIN
                    slope_ok = True

                rsi_arr = calculate_rsi(closes, RSI_PERIOD)
                if len(rsi_arr) < 70 or np.isnan(rsi_arr[-2]):
                    self.market_debug[symbol] = {
                        "time": time.time(),
                        "gate": "Indicators",
                        "why": ["RSI not ready yet."],
                    }
                    self.last_processed_closed_t0[symbol] = confirm_t0
                    continue
                rsi_now = float(rsi_arr[-2])

                # ADX/ATR
                atr_arr = calculate_atr(highs, lows, closes, ATR_PERIOD)
                adx_arr = calculate_adx(highs, lows, closes, ADX_PERIOD)

                atr_now = float(atr_arr[-2]) if len(atr_arr) and not np.isnan(atr_arr[-2]) else float("nan")
                adx_now = float(adx_arr[-2]) if len(adx_arr) and not np.isnan(adx_arr[-2]) else float("nan")

                atr_ok = (not np.isnan(atr_now)) and (atr_now >= float(ATR_MIN))
                adx_ok = (not np.isnan(adx_now)) and (adx_now >= float(ADX_MIN))

                if TREND_FILTER_MODE.upper() == "EITHER":
                    trend_filter_ok = adx_ok or atr_ok
                else:
                    trend_filter_ok = adx_ok and atr_ok

                # pullback touch
                pb_high = float(pullback["h"])
                pb_low = float(pullback["l"])
                touched_ema20 = (pb_low <= ema20_pullback <= pb_high)

                # confirm candle
                c_open = float(confirm["o"])
                c_close = float(confirm["c"])
                c_high = float(confirm["h"])
                c_low = float(confirm["l"])

                bull_confirm = c_close > c_open
                bear_confirm = c_close < c_open

                close_above_ema20 = c_close > ema20_confirm
                close_below_ema20 = c_close < ema20_confirm

                # candle strength
                c_range = max(MIN_CANDLE_RANGE, c_high - c_low)
                last_body = abs(c_close - c_open)
                body_ratio = last_body / c_range
                strong_candle = body_ratio >= float(MIN_BODY_RATIO)

                # spike block (compare to recent bodies)
                bodies = [abs(float(candles[i]["c"]) - float(candles[i]["o"])) for i in range(-22, -2)]
                avg_body = float(np.mean(bodies)) if len(bodies) >= 10 else float(np.mean([abs(float(c["c"]) - float(c["o"])) for c in candles[-60:-2]]))
                spike_block = (avg_body > 0 and last_body > SPIKE_MULT * avg_body)

                # flat block
                ema_diff = abs(ema20_confirm - ema50_confirm)
                flat_block = ema_diff < EMA_DIFF_MIN

                uptrend = ema20_confirm > ema50_confirm
                downtrend = ema20_confirm < ema50_confirm

                call_rsi_ok = (RSI_CALL_MIN <= rsi_now <= RSI_CALL_MAX)
                put_rsi_ok = (RSI_PUT_MIN <= rsi_now <= RSI_PUT_MAX)

                # close position filter (more entries than strict breakout)
                close_pos = (c_close - c_low) / (c_range + 1e-12)
                strength_call = close_pos >= CLOSE_POS_CALL_MIN
                strength_put = close_pos <= CLOSE_POS_PUT_MAX

                call_ready = (
                    uptrend and slope_ok and ema50_rising and touched_ema20 and bull_confirm
                    and strong_candle and close_above_ema20 and call_rsi_ok
                    and strength_call
                    and (not spike_block) and (not flat_block)
                    and trend_filter_ok
                )
                put_ready = (
                    downtrend and slope_ok and ema50_falling and touched_ema20 and bear_confirm
                    and strong_candle and close_below_ema20 and put_rsi_ok
                    and strength_put
                    and (not spike_block) and (not flat_block)
                    and trend_filter_ok
                )

                signal = "CALL" if call_ready else "PUT" if put_ready else None

                block_parts = []
                if not trend_filter_ok:
                    block_parts.append("TREND FILTER")
                if spike_block:
                    block_parts.append("SPIKE")
                if flat_block:
                    block_parts.append("FLAT")
                if not strong_candle:
                    block_parts.append("WEAK CANDLE")
                if not slope_ok:
                    block_parts.append("SLOPE N/A")
                block_label = " | ".join(block_parts) if block_parts else "OK"

                why = []
                if not ok_gate:
                    why.append(f"Gate blocked: {gate}")
                if signal:
                    why.append(f"READY: {signal}")
                else:
                    why.append("No entry yet.")

                self.market_debug[symbol] = {
                    "time": time.time(),
                    "gate": gate,
                    "last_closed": confirm_t0,
                    "signal": signal,
                    "why": why[:8],
                    "rsi_now": rsi_now,
                    "ema50_slope": ema50_slope,
                    "body_ratio": body_ratio,
                    "adx_now": adx_now,
                    "atr_now": atr_now,
                    "block_label": block_label,
                }

                self.last_processed_closed_t0[symbol] = confirm_t0

                if not ok_gate:
                    continue

                if call_ready:
                    await self.execute_trade("CALL", symbol, source="AUTO")
                elif put_ready:
                    await self.execute_trade("PUT", symbol, source="AUTO")

            except asyncio.CancelledError:
                break
            except Exception as e:
                msg = str(e)
                logger.error(f"Scanner Error ({symbol}): {msg}")

                if self._is_rate_limit_error(msg):
                    self._rate_limit_strikes[symbol] = int(self._rate_limit_strikes.get(symbol, 0)) + 1
                    backoff = RATE_LIMIT_BACKOFF_BASE * self._rate_limit_strikes[symbol]
                    backoff = min(180, backoff)
                    self._next_poll_epoch[symbol] = time.time() + backoff
                else:
                    await asyncio.sleep(2 if not self._is_gatewayish_error(msg) else 5)

            await asyncio.sleep(0.05)

    # ========================= PAYOUT MODE + MARTINGALE =========================
    async def execute_trade(self, side: str, symbol: str, reason="AUTO", source="AUTO"):
        if not self.api or self.active_trade_info:
            return

        async with self.trade_lock:
            ok, gate = self.can_auto_trade()
            if not ok:
                return

            try:
                import math

                # recovery-style martingale payout
                step = int(self.martingale_step)
                payout = float(PAYOUT_TARGET) * (float(MARTINGALE_MULT) ** step)
                payout = money2(payout)

                payout = max(0.01, float(payout))
                if not math.isfinite(payout):
                    payout = 0.01

                payout = max(float(MIN_PAYOUT), float(payout))
                payout = money2(payout)

                proposal_req = {
                    "proposal": 1,
                    "amount": payout,
                    "basis": "payout",
                    "contract_type": side,
                    "currency": "USD",
                    "duration": int(DURATION_MIN),
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
                if ask_price <= 0:
                    await self.safe_send_tg("❌ Proposal returned invalid ask_price.")
                    return

                if ask_price > float(MAX_STAKE_ALLOWED):
                    await self.safe_send_tg(
                        f"⛔️ Skipped trade: payout=${payout:.2f} needs stake=${ask_price:.2f} > max ${MAX_STAKE_ALLOWED:.2f}"
                    )
                    self.cooldown_until = time.time() + COOLDOWN_SEC
                    return

                # ✅ IMPORTANT FIX (reduces trading errors):
                # Buy cap tied to the current ask_price + buffer (but never above MAX_STAKE_ALLOWED).
                buy_price_cap = min(float(MAX_STAKE_ALLOWED), money2(float(ask_price) + float(BUY_PRICE_BUFFER)))

                buy = await self.safe_deriv_call("buy", {"buy": proposal_id, "price": buy_price_cap}, retries=6)
                if "error" in buy:
                    err_msg = str(buy["error"].get("message", "Buy error")).strip()
                    low = err_msg.lower()

                    # Retry once if price changed
                    if ("moved too much" in low) or ("price has changed" in low):
                        await asyncio.sleep(0.25)

                        prop2 = await self.safe_deriv_call("proposal", proposal_req, retries=6)
                        if "error" in prop2:
                            err2 = prop2["error"].get("message", "Proposal error")
                            await self.safe_send_tg(f"❌ Proposal Error (retry):\n{err2}")
                            return

                        p2 = prop2["proposal"]
                        proposal_id2 = p2["id"]
                        ask_price2 = float(p2.get("ask_price", 0.0))
                        if ask_price2 <= 0:
                            await self.safe_send_tg("❌ Proposal retry returned invalid ask_price.")
                            return

                        if ask_price2 > float(MAX_STAKE_ALLOWED):
                            await self.safe_send_tg(
                                f"⛔️ Skipped (retry): payout=${payout:.2f} needs stake=${ask_price2:.2f} > max ${MAX_STAKE_ALLOWED:.2f}"
                            )
                            self.cooldown_until = time.time() + COOLDOWN_SEC
                            return

                        buy_price_cap2 = min(float(MAX_STAKE_ALLOWED), money2(float(ask_price2) + float(BUY_PRICE_BUFFER)))
                        buy2 = await self.safe_deriv_call("buy", {"buy": proposal_id2, "price": buy_price_cap2}, retries=6)

                        if "error" in buy2:
                            err3 = str(buy2["error"].get("message", "Buy error")).strip()
                            await self.safe_send_tg(f"❌ Trade Refused (retry):\n{err3}")
                            return

                        buy = buy2
                        ask_price = ask_price2
                        buy_price_cap = buy_price_cap2
                    else:
                        await self.safe_send_tg(f"❌ Trade Refused:\n{err_msg}")
                        return

                self.active_trade_info = int(buy["buy"]["contract_id"])
                self.active_market = symbol
                self.trade_start_time = time.time()
                self.current_stake = ask_price

                if source == "AUTO":
                    self.trades_today += 1

                safe_symbol = str(symbol).replace("_", " ")
                msg = (
                    f"🚀 {side} TRADE OPENED\n"
                    f"🛒 Market: {safe_symbol}\n"
                    f"⏱ Expiry: {DURATION_MIN}m\n"
                    f"🎁 Payout: ${payout:.2f}\n"
                    f"🎲 Martingale step: {self.martingale_step}/{MARTINGALE_MAX_STEPS} (x{MARTINGALE_MULT})\n"
                    f"💵 Stake (ask_price): ${ask_price:.2f}\n"
                    f"🧾 Buy cap: ${buy_price_cap:.2f}\n"
                    f"🚦 Gate: {gate}\n"
                    f"🎯 Today PnL: {self.total_profit_today:+.2f} / +{DAILY_PROFIT_TARGET:.2f}\n"
                )
                await self.safe_send_tg(msg)

                asyncio.create_task(self.check_result(self.active_trade_info, source))

            except Exception as e:
                logger.error(f"Trade error: {e}")
                await self.safe_send_tg(f"⚠️ Trade error:\n{e}")

    async def check_result(self, cid: int, source: str):
        await asyncio.sleep(int(DURATION_MIN) * 60 + 5)
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

                    if self.martingale_step < MARTINGALE_MAX_STEPS:
                        self.martingale_step += 1
                    else:
                        self.martingale_halt = True
                        self.is_scanning = False

                    # Soft pause after a streak (doesn't kill the whole day)
                    if self.consecutive_losses >= LOSS_STREAK_PAUSE_AT:
                        self.soft_pause_until = time.time() + float(LOSS_STREAK_PAUSE_SEC)

                else:
                    self.consecutive_losses = 0
                    self.martingale_step = 0
                    self.martingale_halt = False

                if self.total_profit_today >= DAILY_PROFIT_TARGET:
                    self.pause_until = self._next_midnight_epoch()

                if self.total_profit_today <= float(DAILY_LOSS_LIMIT):
                    self.pause_until = self._next_midnight_epoch()

            await self.fetch_balance()

            pause_note = "\n⏸ Paused until 12:00am WAT" if time.time() < self.pause_until else ""
            soft_note = f"\n🟡 Soft pause until {fmt_hhmm(self.soft_pause_until)}" if time.time() < self.soft_pause_until else ""
            halt_note = f"\n🛑 Martingale stopped after {MARTINGALE_MAX_STEPS} steps" if self.martingale_halt else ""

            next_payout = money2(float(PAYOUT_TARGET) * (float(MARTINGALE_MULT) ** int(self.martingale_step)))

            await self.safe_send_tg(
                (
                    f"🏁 FINISH: {'WIN' if profit > 0 else 'LOSS'} ({profit:+.2f})\n"
                    f"📊 Today: {self.trades_today}/{MAX_TRADES_PER_DAY} | ❌ Losses: {self.total_losses_today} | Streak: {self.consecutive_losses}/{MAX_CONSEC_LOSSES}\n"
                    f"💵 Today PnL: {self.total_profit_today:+.2f} / +{DAILY_PROFIT_TARGET:.2f}\n"
                    f"🎁 Next payout: ${next_payout:.2f} (step {self.martingale_step}/{MARTINGALE_MAX_STEPS})\n"
                    f"💰 Balance: {self.balance}"
                    f"{soft_note}{pause_note}{halt_note}"
                )
            )
        finally:
            self.active_trade_info = None
            self.cooldown_until = time.time() + COOLDOWN_SEC


# ========================= UI =========================
bot_logic = DerivSniperBot()


def main_keyboard():
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("▶️ START", callback_data="START_SCAN"),
                InlineKeyboardButton("⏹️ STOP", callback_data="STOP_SCAN"),
            ],
            [
                InlineKeyboardButton("📊 STATUS", callback_data="STATUS"),
                InlineKeyboardButton("🔄 REFRESH", callback_data="STATUS"),
            ],
            [InlineKeyboardButton("🧪 TEST BUY", callback_data="TEST_BUY")],
            [
                InlineKeyboardButton("🧪 DEMO", callback_data="SET_DEMO"),
                InlineKeyboardButton("💰 LIVE", callback_data="SET_REAL"),
            ],
        ]
    )


def format_market_detail(sym: str, d: dict) -> str:
    if not d:
        return f"📍 {sym.replace('_',' ')}\n⏳ No scan data yet"

    age = int(time.time() - d.get("time", time.time()))
    gate = d.get("gate", "—")
    last_closed = d.get("last_closed", 0)
    signal = d.get("signal") or "—"

    block_label = d.get("block_label", "—")
    rsi_now = d.get("rsi_now", None)
    ema50_slope = d.get("ema50_slope", None)
    body_ratio = d.get("body_ratio", None)
    adx_now = d.get("adx_now", None)
    atr_now = d.get("atr_now", None)

    extra = []
    if isinstance(rsi_now, (int, float)) and not np.isnan(rsi_now):
        extra.append(f"RSI: {rsi_now:.2f}")
    if isinstance(ema50_slope, (int, float)) and not np.isnan(ema50_slope):
        extra.append(f"EMA50 slope: {ema50_slope:.3f}")
    if isinstance(body_ratio, (int, float)) and not np.isnan(body_ratio):
        extra.append(f"Body ratio: {body_ratio:.2f}")
    if isinstance(adx_now, (int, float)) and not np.isnan(adx_now):
        extra.append(f"ADX: {adx_now:.2f}")
    if isinstance(atr_now, (int, float)) and not np.isnan(atr_now):
        extra.append(f"ATR: {atr_now:.5f}")

    extra_line = " | ".join(extra) if extra else "—"

    why = d.get("why", [])
    why_line = "Why: " + (str(why[0]) if why else "—")

    return (
        f"📍 {sym.replace('_',' ')} ({age}s)\n"
        f"Gate: {gate}\n"
        f"Last closed: {fmt_time_hhmmss(last_closed)}\n"
        f"────────────────\n"
        f"Signal: {signal}\n"
        f"Filters: {block_label}\n"
        f"Stats: {extra_line}\n"
        f"{why_line}\n"
    )


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

    elif q.data == "START_SCAN":
        if not bot_logic.api:
            await _safe_edit(q, "❌ Connect first.", reply_markup=main_keyboard())
            return
        bot_logic.is_scanning = True
        bot_logic.scanner_task = asyncio.create_task(bot_logic.background_scanner())
        await _safe_edit(
            q,
            (
                "🔍 SCANNER ACTIVE\n"
                "Strategy: EMA trend + EMA50 slope + pullback touch + strong confirm + RSI + ADX filter\n"
                f"Expiry: {DURATION_MIN}m | Cooldown: {COOLDOWN_SEC}s\n"
                f"Martingale: x{MARTINGALE_MULT} up to {MARTINGALE_MAX_STEPS} steps (recovery-style)\n"
                "✅ STATUS to monitor.\n"
            ),
            reply_markup=main_keyboard(),
        )

    elif q.data == "STOP_SCAN":
        bot_logic.is_scanning = False
        if bot_logic.scanner_task and not bot_logic.scanner_task.done():
            bot_logic.scanner_task.cancel()
        await _safe_edit(q, "⏹️ Scanner stopped.", reply_markup=main_keyboard())

    elif q.data == "TEST_BUY":
        asyncio.create_task(bot_logic.execute_trade("CALL", "R_10", "Manual Test", source="MANUAL"))
        await _safe_edit(q, "🧪 Test trade triggered (CALL R 10).", reply_markup=main_keyboard())

    elif q.data == "STATUS":
        now = time.time()
        if now < bot_logic.status_cooldown_until:
            left = int(bot_logic.status_cooldown_until - now)
            await _safe_edit(q, f"⏳ Refresh cooldown: {left}s\n\nPress again after cooldown.", reply_markup=main_keyboard())
            return
        bot_logic.status_cooldown_until = now + STATUS_REFRESH_COOLDOWN_SEC

        await bot_logic.fetch_balance()
        now_time = datetime.now(ZoneInfo("Africa/Lagos")).strftime("%Y-%m-%d %H:%M:%S")
        _ok, gate = bot_logic.can_auto_trade()

        trade_status = "No Active Trade"
        if bot_logic.active_trade_info and bot_logic.api:
            try:
                res = await bot_logic.safe_deriv_call(
                    "proposal_open_contract",
                    {"proposal_open_contract": 1, "contract_id": bot_logic.active_trade_info},
                    retries=4,
                )
                pnl = float(res["proposal_open_contract"].get("profit", 0))
                rem = max(0, int(DURATION_MIN * 60) - int(time.time() - bot_logic.trade_start_time))
                icon = "✅ PROFIT" if pnl > 0 else "❌ LOSS" if pnl < 0 else "➖ FLAT"
                mkt_clean = str(bot_logic.active_market).replace("_", " ")
                trade_status = f"🚀 Active Trade ({mkt_clean})\n📈 PnL: {icon} ({pnl:+.2f})\n⏳ Left: {rem}s"
            except Exception:
                trade_status = "🚀 Active Trade: Syncing..."

        pause_line = "⏸ Paused until 12:00am WAT\n" if time.time() < bot_logic.pause_until else ""
        soft_line = f"🟡 Soft pause until {fmt_hhmm(bot_logic.soft_pause_until)}\n" if time.time() < bot_logic.soft_pause_until else ""

        next_payout = money2(float(PAYOUT_TARGET) * (float(MARTINGALE_MULT) ** int(bot_logic.martingale_step)))

        header = (
            f"🕒 Time (WAT): {now_time}\n"
            f"🤖 Bot: {'ACTIVE' if bot_logic.is_scanning else 'OFFLINE'} ({bot_logic.account_type})\n"
            f"{pause_line}{soft_line}"
            f"🎁 Next payout: ${next_payout:.2f} | Step: {bot_logic.martingale_step}/{MARTINGALE_MAX_STEPS} (x{MARTINGALE_MULT})\n"
            f"🧯 Max stake allowed: ${MAX_STAKE_ALLOWED:.2f}\n"
            f"⏱ Expiry: {DURATION_MIN}m | Cooldown: {COOLDOWN_SEC}s\n"
            f"🎯 Daily Target: +${DAILY_PROFIT_TARGET:.2f} | Daily Loss Limit: {DAILY_LOSS_LIMIT:.2f}\n"
            f"📡 Markets: {', '.join(MARKETS).replace('_',' ')}\n"
            f"📌 Filters: ADX(min {ADX_MIN}) | RSI({RSI_CALL_MIN}-{RSI_CALL_MAX}/{RSI_PUT_MIN}-{RSI_PUT_MAX})\n"
            f"━━━━━━━━━━━━━━━\n{trade_status}\n━━━━━━━━━━━━━━━\n"
            f"💵 Total Profit Today: {bot_logic.total_profit_today:+.2f}\n"
            f"🎯 Trades: {bot_logic.trades_today}/{MAX_TRADES_PER_DAY} | ❌ Losses: {bot_logic.total_losses_today}\n"
            f"📉 Loss Streak: {bot_logic.consecutive_losses}/{MAX_CONSEC_LOSSES}\n"
            f"🚦 Gate: {gate}\n"
            f"💰 Balance: {bot_logic.balance}\n"
        )

        details = "\n\n📌 LIVE SCAN\n\n" + "\n\n".join(
            [format_market_detail(sym, bot_logic.market_debug.get(sym, {})) for sym in MARKETS]
        )

        await _safe_edit(q, header + details, reply_markup=main_keyboard())


async def start_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    await u.message.reply_text(
        "💎 Deriv Bot\n"
        f"🕯 Timeframe: M1 | ⏱ Expiry: {DURATION_MIN}m\n"
        "✅ Anti-rate-limit enabled\n",
        reply_markup=main_keyboard(),
    )


if __name__ == "__main__":
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    bot_logic.app = app
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CallbackQueryHandler(btn_handler))
    app.run_polling(drop_pending_updates=True)
