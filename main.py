# ✅ DONCHIAN BREAKOUT VERSION (2-minute expiry untouched)
# Strategy:
# Breakout:
#   - CALL when confirm candle CLOSES above Donchian Upper (period N)
#   - PUT when confirm candle CLOSES below Donchian Lower (period N)
# Filters:
#   - EMA50 slope must be strong in trade direction (avoids chop)
#   - Donchian width must be "expanded" vs recent average (avoids tight ranges)
# Execution:
#   - 1M chart, 2-minute expiry (UNCHANGED)
# Risk:
#   - Martingale lock (same direction only during martingale steps)
#   - Max 1 trade per breakout structure (prevents overtrading same breakout)

# ⚠️ SECURITY NOTE:
# DO NOT paste tokens publicly. Rotate any tokens you already shared.

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

COOLDOWN_SEC = 120
MAX_TRADES_PER_DAY = 60
MAX_CONSEC_LOSSES = 10

TELEGRAM_TOKEN = "8253450930:AAHUhPk9TML-8kZlA9UaHZZvTUGdurN9MVY"
TELEGRAM_CHAT_ID = "7634818949"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ========================= STRATEGY SETTINGS =========================
TF_SEC = 60  # M1 candles
CANDLES_COUNT = 200  # a bit more history helps Donchian + filters

DURATION_MIN = 4 # ✅ keep 2-minute expiry (DO NOT TOUCH)

# ---- Donchian breakout ----
DONCHIAN_PERIOD = 14         # classic
BREAKOUT_BUFFER = 0.0         # set to small value if you want: e.g. 0.05 (points)
USE_HIGH_LOW_BANDS = True     # True = highest high/lowest low; False = uses closes

# ---- Trend / chop filters ----
EMA_SLOPE_LOOKBACK = 10
EMA_SLOPE_MIN = 0.20          # "strong slope" threshold (tune per index)

# Donchian width filter (avoid tight ranges / fakeouts)
WIDTH_AVG_LOOKBACK = 30       # compare current width to avg width of last N
WIDTH_MIN_RATIO = 1.10        # width_now must be >= avg_width * 1.10 (10% wider)
MIN_WIDTH_ABS = 0.0           # optional absolute minimum width; keep 0.0 if unsure

# Rejection / candle strength (optional but useful on M1)
MIN_BODY_RATIO = 0.45
MIN_CANDLE_RANGE = 1e-6

# ========================= SECTIONS =========================
SECTIONS_PER_DAY = 4
SECTION_PROFIT_TARGET = 0.15
SECTION_LENGTH_SEC = int(24 * 60 * 60 / SECTIONS_PER_DAY)

# ========================= PAYOUT MODE =========================
USE_PAYOUT_MODE = True
PAYOUT_TARGET = 1
MIN_PAYOUT = 0.35
MAX_STAKE_ALLOWED = 10.00

# ========================= MARTINGALE SETTINGS =========================
MARTINGALE_MULT = 1.8
MARTINGALE_MAX_STEPS = 4
MARTINGALE_MAX_STAKE = 16.0

# ========================= ANTI RATE-LIMIT =========================
TICKS_GLOBAL_MIN_INTERVAL = 0.35
RATE_LIMIT_BACKOFF_BASE = 20

# ========================= UI: REFRESH COOLDOWN =========================
STATUS_REFRESH_COOLDOWN_SEC = 10


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


def calculate_donchian(highs, lows, closes, period=20, use_high_low=True):
    """
    Returns arrays: upper, lower, mid (same length), NaNs until ready.
    If use_high_low True: upper=max(high), lower=min(low).
    Else: upper=max(close), lower=min(close).
    """
    highs = np.array(highs, dtype=float)
    lows = np.array(lows, dtype=float)
    closes = np.array(closes, dtype=float)

    n = len(closes)
    if n < period:
        return np.array([]), np.array([]), np.array([])

    upper = np.full(n, np.nan, dtype=float)
    lower = np.full(n, np.nan, dtype=float)
    mid = np.full(n, np.nan, dtype=float)

    for i in range(period - 1, n):
        if use_high_low:
            hh = float(np.max(highs[i - period + 1 : i + 1]))
            ll = float(np.min(lows[i - period + 1 : i + 1]))
        else:
            hh = float(np.max(closes[i - period + 1 : i + 1]))
            ll = float(np.min(closes[i - period + 1 : i + 1]))
        upper[i] = hh
        lower[i] = ll
        mid[i] = (hh + ll) / 2.0

    return upper, lower, mid


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

        # Martingale lock direction (only lock when step > 0)
        self.last_signal = None  # "CALL" or "PUT"

        # sections
        self.section_profit = 0.0
        self.sections_won_today = 0
        self.section_index = 1
        self.section_pause_until = 0.0

        self.trade_lock = asyncio.Lock()

        self.market_debug = {m: {} for m in MARKETS}
        self.last_processed_closed_t0 = {m: 0 for m in MARKETS}

        # ✅ Max 1 trade per breakout structure:
        # store last breakout key per symbol: (confirm_t0, direction, band_value_rounded)
        self.last_structure_key = {m: None for m in MARKETS}

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

    # ---------- Sections ----------
    def _today_midnight_epoch(self) -> float:
        now = datetime.now(self.tz)
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return midnight.timestamp()

    def _get_section_index_for_epoch(self, epoch_ts: float) -> int:
        midnight = self._today_midnight_epoch()
        sec_into_day = max(0, int(epoch_ts - midnight))
        idx0 = min(SECTIONS_PER_DAY - 1, sec_into_day // SECTION_LENGTH_SEC)
        return int(idx0 + 1)

    def _next_section_start_epoch(self, epoch_ts: float) -> float:
        midnight = self._today_midnight_epoch()
        sec_into_day = max(0, int(epoch_ts - midnight))
        idx0 = min(SECTIONS_PER_DAY - 1, sec_into_day // SECTION_LENGTH_SEC)
        next_start = midnight + (idx0 + 1) * SECTION_LENGTH_SEC
        if idx0 + 1 >= SECTIONS_PER_DAY:
            next_midnight = (datetime.fromtimestamp(midnight, self.tz) + timedelta(days=1)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            return next_midnight.timestamp()
        return float(next_start)

    def _sync_section_if_needed(self):
        now = time.time()
        today = datetime.now(self.tz).date()
        if today != self.current_day:
            return
        new_idx = self._get_section_index_for_epoch(now)
        if new_idx != self.section_index:
            self.section_index = new_idx
            self.section_profit = 0.0
            self.section_pause_until = 0.0

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
            self.martingale_step = 0
            self.current_stake = 0.0
            self.martingale_halt = False
            self.last_signal = None

            self.section_profit = 0.0
            self.sections_won_today = 0
            self.section_index = self._get_section_index_for_epoch(time.time())
            self.section_pause_until = 0.0

            self.last_structure_key = {m: None for m in MARKETS}

        self._sync_section_if_needed()

    def can_auto_trade(self) -> tuple[bool, str]:
        self._daily_reset_if_needed()

        if self.martingale_halt:
            return False, f"Stopped: Martingale {MARTINGALE_MAX_STEPS} steps completed"

        if time.time() < self.section_pause_until:
            left = int(self.section_pause_until - time.time())
            return False, f"Section paused. Resumes {fmt_hhmm(self.section_pause_until)} ({left}s)"

        if time.time() < self.pause_until:
            left = int(self.pause_until - time.time())
            return False, f"Paused until 12:00am WAT ({left}s)"

        if self.total_profit_today >= 2.0:
            self.pause_until = self._next_midnight_epoch()
            return False, f"Daily target reached (+${self.total_profit_today:.2f})"

        if self.total_profit_today <= -2.0:
            self.pause_until = self._next_midnight_epoch()
            return False, "Stopped: Daily loss limit (-$2.00) reached"

        if self.consecutive_losses >= MAX_CONSEC_LOSSES:
            return False, "Stopped: max loss streak reached"
        if self.trades_today >= MAX_TRADES_PER_DAY:
            return False, "Stopped: daily trade limit reached"
        if time.time() < self.cooldown_until:
            return False, f"Cooldown {int(self.cooldown_until - time.time())}s"
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
                if self.active_trade_info and (time.time() - self.trade_start_time > (DURATION_MIN * 60 + 90)):
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

                if self.consecutive_losses >= MAX_CONSEC_LOSSES or self.trades_today >= MAX_TRADES_PER_DAY:
                    self.is_scanning = False
                    break

                ok_gate, gate = self.can_auto_trade()

                candles = await self.fetch_real_m1_candles(symbol)
                if len(candles) < 120:
                    self.market_debug[symbol] = {
                        "time": time.time(),
                        "gate": "Waiting for more candles",
                        "why": [f"Need more history (have {len(candles)}, need ~120)."],
                    }
                    self._next_poll_epoch[symbol] = time.time() + 10
                    continue

                # confirm candle = candles[-2] (closed)
                confirm = candles[-2]
                confirm_t0 = int(confirm["t0"])

                next_closed_epoch = confirm_t0 + TF_SEC
                self._next_poll_epoch[symbol] = float(next_closed_epoch + 0.35)

                if self.last_processed_closed_t0[symbol] == confirm_t0:
                    continue

                closes = [x["c"] for x in candles]
                highs = [x["h"] for x in candles]
                lows = [x["l"] for x in candles]

                # EMA50 slope filter
                ema50_arr = calculate_ema(closes, 50)
                if len(ema50_arr) < 80:
                    self.market_debug[symbol] = {
                        "time": time.time(),
                        "gate": "Indicators",
                        "why": ["EMA50 not ready yet."],
                    }
                    self.last_processed_closed_t0[symbol] = confirm_t0
                    continue

                ema50_now = float(ema50_arr[-2])
                slope_ok = False
                ema50_slope = 0.0
                ema50_rising = False
                ema50_falling = False
                if len(ema50_arr) >= (EMA_SLOPE_LOOKBACK + 3):
                    ema50_prev = float(ema50_arr[-(EMA_SLOPE_LOOKBACK + 2)])
                    ema50_slope = ema50_now - ema50_prev
                    ema50_rising = ema50_slope > EMA_SLOPE_MIN
                    ema50_falling = ema50_slope < -EMA_SLOPE_MIN
                    slope_ok = True

                # Donchian bands
                dc_upper, dc_lower, dc_mid = calculate_donchian(
                    highs, lows, closes,
                    period=DONCHIAN_PERIOD,
                    use_high_low=USE_HIGH_LOW_BANDS
                )
                if len(dc_upper) < 80 or np.isnan(dc_upper[-2]) or np.isnan(dc_lower[-2]):
                    self.market_debug[symbol] = {
                        "time": time.time(),
                        "gate": "Indicators",
                        "why": ["Donchian not ready yet."],
                    }
                    self.last_processed_closed_t0[symbol] = confirm_t0
                    continue

                # ✅ DONCHIAN "PREVIOUS BAND" FIX:
                # confirm candle must be compared to the band computed BEFORE it closed (index -3)
                if np.isnan(dc_upper[-3]) or np.isnan(dc_lower[-3]):
                    self.market_debug[symbol] = {
                        "time": time.time(),
                        "gate": "Indicators",
                        "why": ["Donchian prev band not ready yet."],
                    }
                    self.last_processed_closed_t0[symbol] = confirm_t0
                    continue

                upper_prev = float(dc_upper[-3])
                lower_prev = float(dc_lower[-3])
                width_now = max(0.0, upper_prev - lower_prev)

                # width avg filter (exclude the confirm candle)
                start_idx = max(0, len(dc_upper) - 3 - WIDTH_AVG_LOOKBACK)
                widths = []
                for i in range(start_idx, len(dc_upper) - 2):
                    if not np.isnan(dc_upper[i]) and not np.isnan(dc_lower[i]):
                        widths.append(float(dc_upper[i] - dc_lower[i]))
                width_avg = float(np.mean(widths)) if widths else width_now
                width_ok = (width_now >= width_avg * WIDTH_MIN_RATIO) and (width_now >= MIN_WIDTH_ABS)

                # confirm candle stats (optional strength)
                c_open = float(confirm["o"])
                c_close = float(confirm["c"])
                c_high = float(confirm["h"])
                c_low = float(confirm["l"])
                c_range = max(MIN_CANDLE_RANGE, c_high - c_low)
                body_ratio = abs(c_close - c_open) / c_range
                strong_candle = body_ratio >= float(MIN_BODY_RATIO)

                # breakout conditions (compare to PREVIOUS band + optional buffer)
                breakout_call = c_close > (upper_prev + float(BREAKOUT_BUFFER))
                breakout_put = c_close < (lower_prev - float(BREAKOUT_BUFFER))

                # structure key so we don't trade same breakout repeatedly
                call_key = (confirm_t0, "CALL", round(upper_prev, 5))
                put_key = (confirm_t0, "PUT", round(lower_prev, 5))

                call_ready = (
                    slope_ok and ema50_rising
                    and width_ok
                    and strong_candle
                    and breakout_call
                    and (self.last_structure_key[symbol] != call_key)
                )

                put_ready = (
                    slope_ok and ema50_falling
                    and width_ok
                    and strong_candle
                    and breakout_put
                    and (self.last_structure_key[symbol] != put_key)
                )

                # ✅ Martingale direction lock
                if self.martingale_step > 0 and self.last_signal:
                    if (call_ready or put_ready) and (("CALL" if call_ready else "PUT") != self.last_signal):
                        call_ready = False
                        put_ready = False
                    elif (not call_ready and not put_ready):
                        # no ready signal anyway; leave as is
                        pass

                # recompute signal for status display (does not change trade logic)
                signal = "CALL" if call_ready else "PUT" if put_ready else None

                # Debug labels
                slope_label = "EMA50 SLOPE ↑" if ema50_rising else "EMA50 SLOPE ↓" if ema50_falling else "EMA50 SLOPE FLAT"
                dc_label = f"Donchian({DONCHIAN_PERIOD}) U:{upper_prev:.5f} L:{lower_prev:.5f} W:{width_now:.5f}"
                width_label = f"Width OK: {'✅' if width_ok else '❌'} (now {width_now:.5f} vs avg {width_avg:.5f})"

                block_parts = []
                if not slope_ok:
                    block_parts.append("SLOPE N/A")
                if not ema50_rising and not ema50_falling:
                    block_parts.append("EMA50 FLAT")
                if not width_ok:
                    block_parts.append("RANGE TOO TIGHT")
                if not strong_candle:
                    block_parts.append("WEAK CANDLE")
                if signal is None:
                    block_parts.append("NO SIGNAL")
                if self.martingale_step > 0 and self.last_signal:
                    block_parts.append(f"MARTI LOCK: {self.last_signal}")

                block_label = " | ".join(block_parts) if block_parts else "OK"

                why = []
                if not ok_gate:
                    why.append(f"Gate blocked: {gate}")
                if signal and (call_ready or put_ready):
                    why.append(f"READY: {signal} (enter next candle)")
                else:
                    why.append("No entry yet (conditions not aligned).")

                # ---- Extra STATUS info (approaching breakout + waiting-for checklist) ----
                live = candles[-1]
                live_close = float(live["c"])

                dist_to_upper = float((upper_prev + float(BREAKOUT_BUFFER)) - live_close)
                dist_to_lower = float(live_close - (lower_prev - float(BREAKOUT_BUFFER)))

                blocked_by_marti_lock = False
                if self.martingale_step > 0 and self.last_signal:
                    # if there is a breakout but wrong direction, show block
                    if breakout_call and self.last_signal != "CALL":
                        blocked_by_marti_lock = True
                    if breakout_put and self.last_signal != "PUT":
                        blocked_by_marti_lock = True

                structure_seen = (self.last_structure_key[symbol] == call_key) or (self.last_structure_key[symbol] == put_key)

                self.market_debug[symbol] = {
                    "time": time.time(),
                    "gate": gate,
                    "last_closed": confirm_t0,

                    # display signal (ready or not)
                    "signal": signal or "—",

                    # labels
                    "slope_label": slope_label,
                    "dc_label": dc_label,
                    "width_label": width_label,
                    "block_label": block_label,

                    # raw values for richer status
                    "upper_now": upper_prev,
                    "lower_now": lower_prev,
                    "width_now": width_now,
                    "width_avg": width_avg,
                    "body_ratio": body_ratio,
                    "ema50_slope": ema50_slope,

                    # filter booleans
                    "slope_ok": slope_ok,
                    "ema50_rising": ema50_rising,
                    "ema50_falling": ema50_falling,
                    "width_ok": width_ok,
                    "strong_candle": strong_candle,
                    "breakout_call": breakout_call,
                    "breakout_put": breakout_put,
                    "blocked_by_marti_lock": blocked_by_marti_lock,
                    "structure_seen": structure_seen,

                    # live candle / approaching breakout info
                    "last_live_close": live_close,
                    "dist_to_upper": dist_to_upper,
                    "dist_to_lower": dist_to_lower,

                    "why": why[:10],
                }

                self.last_processed_closed_t0[symbol] = confirm_t0

                if not ok_gate:
                    continue

                if call_ready:
                    self.last_structure_key[symbol] = call_key
                    await self.execute_trade("CALL", symbol, source="AUTO")
                elif put_ready:
                    self.last_structure_key[symbol] = put_key
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
    async def execute_trade(self, side: str, symbol: str, reason="MANUAL", source="MANUAL"):
        if not self.api or self.active_trade_info:
            return

        async with self.trade_lock:
            ok, _gate = self.can_auto_trade()
            if not ok:
                return

            try:
                import math

                # ✅ Martingale by PAYOUT (Deriv will quote the stake that matches this payout)
                payout = float(PAYOUT_TARGET) * (float(MARTINGALE_MULT) ** int(self.martingale_step))
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
                    "duration": int(DURATION_MIN),     # ✅ UNCHANGED
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
                ask_price = float(p.get("ask_price", 0.0))  # ✅ stake required for requested payout
                if ask_price <= 0:
                    await self.safe_send_tg("❌ Proposal returned invalid ask_price.")
                    return

                # ✅ Cap martingale using MARTINGALE_MAX_STAKE (not MAX_STAKE_ALLOWED)
                if ask_price > float(MARTINGALE_MAX_STAKE):
                    await self.safe_send_tg(
                        f"⛔️ Skipped trade: payout=${payout:.2f} requires stake=${ask_price:.2f} "
                        f"> martingale cap ${MARTINGALE_MAX_STAKE:.2f}"
                    )
                    self.cooldown_until = time.time() + COOLDOWN_SEC
                    return

                # ✅ Buy using SAFE cap (prevents rejection if ask_price shifts slightly)
                buy = await self.safe_deriv_call("buy", {"buy": proposal_id, "price": float(MARTINGALE_MAX_STAKE)}, retries=6)
                if "error" in buy:
                    err_msg = str(buy["error"].get("message", "Buy error"))
                    await self.safe_send_tg(f"❌ Trade Refused:\n{err_msg}")
                    return

                self.active_trade_info = int(buy["buy"]["contract_id"])
                self.active_market = symbol
                self.trade_start_time = time.time()
                self.current_stake = ask_price

                if source == "AUTO":
                    self.trades_today += 1
                    self.last_signal = side  # lock direction during martingale sequence

                safe_symbol = str(symbol).replace("_", " ")
                msg = (
                    f"🚀 {side} TRADE OPENED (Donchian Breakout)\n"
                    f"🛒 Market: {safe_symbol}\n"
                    f"⏱ Expiry: {DURATION_MIN}m\n"
                    f"🎁 Payout: ${payout:.2f}\n"
                    f"🎲 Martingale step: {self.martingale_step}/{MARTINGALE_MAX_STEPS}\n"
                    f"💵 Stake (Deriv): ${ask_price:.2f}\n"
                    f"🤖 Source: {source}\n"
                    f"🛡️ Martingale Lock: {self.last_signal if self.martingale_step > 0 else 'OFF'}\n"
                    f"🎯 Today PnL: {self.total_profit_today:+.2f}"
                )
                await self.safe_send_tg(msg)

                asyncio.create_task(self.check_result(self.active_trade_info, source))

            except Exception as e:
                logger.error(f"Trade error: {e}")
                await self.safe_send_tg(f"⚠️ Trade error:\n{e}")

    async def check_result(self, cid: int, source: str):
        await asyncio.sleep(int(DURATION_MIN) * 60 + 5)  # ✅ UNCHANGED
        try:
            res = await self.safe_deriv_call(
                "proposal_open_contract",
                {"proposal_open_contract": 1, "contract_id": cid},
                retries=6,
            )
            profit = float(res["proposal_open_contract"].get("profit", 0))

            if source == "AUTO":
                self.total_profit_today += profit
                self.section_profit += profit

                if self.section_profit >= float(SECTION_PROFIT_TARGET):
                    self.sections_won_today += 1
                    self.section_pause_until = self._next_section_start_epoch(time.time())

                if profit <= 0:
                    self.consecutive_losses += 1
                    self.total_losses_today += 1
                    if self.martingale_step < MARTINGALE_MAX_STEPS:
                        self.martingale_step += 1
                    else:
                        self.martingale_halt = True
                        self.is_scanning = False
                    if self.consecutive_losses >= 3:
                        self.section_pause_until = self._next_section_start_epoch(time.time())
                else:
                    self.consecutive_losses = 0
                    self.martingale_step = 0
                    self.martingale_halt = False
                    self.last_signal = None  # clear after win

                if self.total_profit_today >= 2.0:
                    self.pause_until = self._next_midnight_epoch()

            await self.fetch_balance()

            pause_note = "\n⏸ Paused until 12:00am WAT" if time.time() < self.pause_until else ""
            halt_note = f"\n🛑 Martingale stopped after {MARTINGALE_MAX_STEPS} steps" if self.martingale_halt else ""
            section_note = (
                f"\n🧩 Section paused until {fmt_hhmm(self.section_pause_until)}"
                if time.time() < self.section_pause_until
                else ""
            )

            next_payout = money2(float(PAYOUT_TARGET) * (float(MARTINGALE_MULT) ** int(self.martingale_step)))

            await self.safe_send_tg(
                (
                    f"🏁 FINISH: {'WIN' if profit > 0 else 'LOSS'} ({profit:+.2f})\n"
                    f"🧩 Section: {self.section_index}/{SECTIONS_PER_DAY} | Section PnL: {self.section_profit:+.2f} | Sections won: {self.sections_won_today}\n"
                    f"📊 Today: {self.trades_today}/{MAX_TRADES_PER_DAY} | ❌ Losses: {self.total_losses_today} | Streak: {self.consecutive_losses}/{MAX_CONSEC_LOSSES}\n"
                    f"💵 Today PnL: {self.total_profit_today:+.2f}\n"
                    f"🎁 Next payout: ${next_payout:.2f} (step {self.martingale_step}/{MARTINGALE_MAX_STEPS})\n"
                    f"🛡️ Martingale Lock: {self.last_signal if self.martingale_step > 0 else 'OFF'}\n"
                    f"💰 Balance: {self.balance}"
                    f"{pause_note}{section_note}{halt_note}"
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
            [InlineKeyboardButton("🧩 SECTION", callback_data="NEXT_SECTION")],
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

    upper_now = d.get("upper_now", None)
    lower_now = d.get("lower_now", None)
    width_now = d.get("width_now", None)
    width_avg = d.get("width_avg", None)

    last_live_close = d.get("last_live_close", None)
    dist_to_upper = d.get("dist_to_upper", None)
    dist_to_lower = d.get("dist_to_lower", None)

    signal = d.get("signal") or "—"
    body_ratio = d.get("body_ratio", None)
    ema50_slope = d.get("ema50_slope", None)

    slope_ok = d.get("slope_ok", False)
    ema50_rising = d.get("ema50_rising", False)
    ema50_falling = d.get("ema50_falling", False)
    width_ok = d.get("width_ok", False)
    strong_candle = d.get("strong_candle", False)
    breakout_call = d.get("breakout_call", False)
    breakout_put = d.get("breakout_put", False)
    blocked_by_marti_lock = d.get("blocked_by_marti_lock", False)
    structure_seen = d.get("structure_seen", False)

    slope_label = d.get("slope_label", "—")
    dc_label = d.get("dc_label", "—")
    width_label = d.get("width_label", "—")
    block_label = d.get("block_label", "—")

    approach_lines = []
    if isinstance(last_live_close, (int, float)) and not np.isnan(last_live_close) and \
       isinstance(upper_now, (int, float)) and isinstance(lower_now, (int, float)):

        if isinstance(dist_to_upper, (int, float)) and isinstance(dist_to_lower, (int, float)):
            approach_lines.append(f"Live close: {last_live_close:.5f}")
            approach_lines.append(f"Δ to Upper: {dist_to_upper:+.5f} | Δ to Lower: {dist_to_lower:+.5f}")

            thr = 0.00001
            if isinstance(width_now, (int, float)) and width_now and width_now > 0:
                thr = max(0.15 * width_now, 0.00001)

            if 0 < dist_to_upper <= thr:
                approach_lines.append("🟡 Approaching CALL breakout (near Upper)")
            if 0 < dist_to_lower <= thr:
                approach_lines.append("🟡 Approaching PUT breakout (near Lower)")
            if dist_to_upper <= 0:
                approach_lines.append("🟣 Above Upper NOW (breakout zone)")
            if dist_to_lower <= 0:
                approach_lines.append("🟣 Below Lower NOW (breakout zone)")

    approach_block = "\n".join(approach_lines) if approach_lines else "—"

    wait_lines = []
    wait_lines.append("✅ Conditions checklist (what bot is waiting for):")
    wait_lines.append(f"• Gate OK: {'✅' if gate == 'OK' else '❌'} ({gate})")
    wait_lines.append(f"• EMA slope ready: {'✅' if slope_ok else '❌'}" + (f" (slope {ema50_slope:+.3f})" if isinstance(ema50_slope, (int, float)) else ""))
    if isinstance(width_now, (int, float)) and isinstance(width_avg, (int, float)):
        wait_lines.append(f"• Width OK: {'✅' if width_ok else '❌'} (now {width_now:.5f} vs avg {width_avg:.5f})")
    else:
        wait_lines.append(f"• Width OK: {'✅' if width_ok else '❌'}")

    wait_lines.append(f"• Strong candle body: {'✅' if strong_candle else '❌'}" + (f" (body {body_ratio:.2f})" if isinstance(body_ratio, (int, float)) else ""))
    wait_lines.append(f"• CALL breakout: {'✅' if breakout_call else '❌'} | Trend rising: {'✅' if ema50_rising else '❌'}")
    wait_lines.append(f"• PUT breakout: {'✅' if breakout_put else '❌'} | Trend falling: {'✅' if ema50_falling else '❌'}")
    wait_lines.append(f"• Martingale Lock: {'❌ BLOCKED' if blocked_by_marti_lock else '✅ OK'}")
    wait_lines.append(f"• Structure already traded: {'❌ YES (blocked)' if structure_seen else '✅ NO'}")

    wait_block = "\n".join(wait_lines)

    extra = []
    if isinstance(body_ratio, (int, float)) and not np.isnan(body_ratio):
        extra.append(f"Body ratio: {body_ratio:.2f}")
    extra_line = " | ".join(extra) if extra else "—"

    why = d.get("why", [])
    why_line = "Why: " + (str(why[0]) if why else "—")

    return (
        f"📍 {sym.replace('_',' ')} ({age}s)\n"
        f"Gate: {gate}\n"
        f"Last closed: {fmt_time_hhmmss(last_closed)}\n"
        f"────────────────\n"
        f"{slope_label}\n"
        f"{dc_label}\n"
        f"{width_label}\n"
        f"Stats: {extra_line}\n"
        f"Filters: {block_label}\n"
        f"Signal: {signal}\n"
        f"────────────────\n"
        f"📌 Approaching breakout:\n{approach_block}\n"
        f"────────────────\n"
        f"{wait_block}\n"
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
        await _safe_edit(q, "🔍 SCANNER ACTIVE (Donchian Breakout)\n✅ Press STATUS to monitor.", reply_markup=main_keyboard())

    elif q.data == "STOP_SCAN":
        bot_logic.is_scanning = False
        if bot_logic.scanner_task and not bot_logic.scanner_task.done():
            bot_logic.scanner_task.cancel()
        await _safe_edit(q, "⏹️ Scanner stopped.", reply_markup=main_keyboard())

    elif q.data == "NEXT_SECTION":
        bot_logic._daily_reset_if_needed()
        now = time.time()
        nxt = bot_logic._next_section_start_epoch(now)
        if nxt <= now + 1:
            nxt = now + 1

        forced_idx = bot_logic._get_section_index_for_epoch(nxt + 1)
        bot_logic.section_index = forced_idx
        bot_logic.section_profit = 0.0
        bot_logic.section_pause_until = 0.0

        await _safe_edit(
            q,
            f"🧩 Moved to Section {bot_logic.section_index}/{SECTIONS_PER_DAY}. Reset section PnL to 0.00.",
            reply_markup=main_keyboard(),
        )

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

        section_pause_line = ""
        if time.time() < bot_logic.section_pause_until:
            section_pause_line = f"🧩 Section paused until {fmt_hhmm(bot_logic.section_pause_until)}\n"

        header = (
            f"🕒 Time (WAT): {now_time}\n"
            f"🤖 Bot: {'ACTIVE' if bot_logic.is_scanning else 'OFFLINE'} ({bot_logic.account_type})\n"
            f"🧩 Section: {bot_logic.section_index}/{SECTIONS_PER_DAY} | Section PnL: {bot_logic.section_profit:+.2f} / +{SECTION_PROFIT_TARGET:.2f} | Sections won: {bot_logic.sections_won_today}\n"
            f"{section_pause_line}"
            f"🎁 Step: {bot_logic.martingale_step}/{MARTINGALE_MAX_STEPS} | Lock: {bot_logic.last_signal if bot_logic.martingale_step > 0 else 'OFF'}\n"
            f"⏱ Expiry: {DURATION_MIN}m | Cooldown: {COOLDOWN_SEC}s\n"
            f"📡 Markets: {', '.join(MARKETS).replace('_',' ')}\n"
            f"━━━━━━━━━━━━━━━\n"
            f"💵 Total Profit Today: {bot_logic.total_profit_today:+.2f}\n"
            f"🎯 Trades: {bot_logic.trades_today}/{MAX_TRADES_PER_DAY} | ❌ Losses: {bot_logic.total_losses_today}\n"
            f"📉 Loss Streak: {bot_logic.consecutive_losses}/{MAX_CONSEC_LOSSES}\n"
            f"🚦 Gate: {gate}\n"
            f"💰 Balance: {bot_logic.balance}\n"
        )

        details = "\n\n📌 LIVE SCAN (FULL)\n\n" + "\n\n".join(
            [format_market_detail(sym, bot_logic.market_debug.get(sym, {})) for sym in MARKETS]
        )

        await _safe_edit(q, header + details, reply_markup=main_keyboard())


async def start_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    await u.message.reply_text(
        "💎 Deriv Bot (Donchian Breakout)\n"
        f"🕯 Timeframe: M1 | ⏱ Expiry: {DURATION_MIN}m\n",
        reply_markup=main_keyboard(),
    )


if __name__ == "__main__":
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    bot_logic.app = app
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CallbackQueryHandler(btn_handler))
    app.run_polling(drop_pending_updates=True)
