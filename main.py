# ⚠️ SECURITY NOTE:
# You posted your Deriv + Telegram tokens publicly.
# Regenerate/revoke them in Deriv and BotFather, then replace below.

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
REAL_TOKEN = "ZkOFWOlPtwnjqTS" # replace with your full real token
APP_ID = 1089

MARKETS = ["R_10", "R_25"]

COOLDOWN_SEC = 120
MAX_TRADES_PER_DAY = 60
MAX_CONSEC_LOSSES = 10  # ✅ unchanged (NOT 2)

TELEGRAM_TOKEN = "8253450930:AAHUhPk9TML-8kZlA9UaHZZvTUGdurN9MVY"
TELEGRAM_CHAT_ID = "7634818949"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ========================= STRATEGY SETTINGS =========================
TF_SEC = 60  # M1 candles
CANDLES_COUNT = 150
SCAN_SLEEP_SEC = 2

RSI_PERIOD = 14
DURATION_MIN = 1  # 1-minute expiry

# ========================= EMA50 SLOPE + DAILY TARGET =========================
EMA_SLOPE_LOOKBACK = 10
EMA_SLOPE_MIN = 0.0
DAILY_PROFIT_TARGET = 5.0  # keep as-is unless you change it

# ========================= SECTIONS (✅ NEW) =========================
SECTIONS_PER_DAY = 4
SECTION_PROFIT_TARGET = 0.48  # stop this section after +$0.48, then auto-resume next section
SECTION_LENGTH_SEC = int(24 * 60 * 60 / SECTIONS_PER_DAY)  # 6 hours when 4 sections

# ========================= PAYOUT MODE =========================
USE_PAYOUT_MODE = True
PAYOUT_TARGET = 1.00          # base payout (martingale will multiply this)
MAX_STAKE_ALLOWED = 5.00      # safety cap: if Deriv needs > this stake, skip trade
BUY_PRICE_BUFFER = 0.02       # kept (not used in fixed-cap buy)

# ========================= MARTINGALE SETTINGS =========================
MARTINGALE_MULT = 2.0         # 2x martingale
MARTINGALE_MAX_STEPS = 5      # stop AFTER completing 5 martingales
MARTINGALE_MAX_STAKE = 16.0   # kept for display only

# ========================= ✅ HIGHER TIMEFRAME BIAS (ADDED) =========================
# This is ONLY a filter (does not change your UI/flow):
# - For CALL on M1: HTF must be BULL (HTF EMA20 > EMA50)
# - For PUT on M1:  HTF must be BEAR (HTF EMA20 < EMA50)
USE_HTF_BIAS = True
HTF_SEC = 300                 # M5 candles
HTF_CANDLES_COUNT = 140        # enough for EMA50 stability
HTF_EMA_FAST = 20
HTF_EMA_SLOW = 50
HTF_REFRESH_SEC = 20           # cache refresh to reduce API calls

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


def build_candles_from_deriv(candles_raw):
    out = []
    for x in candles_raw:
        out.append({
            "t0": int(x.get("epoch", 0)),
            "o": float(x.get("open", 0)),
            "h": float(x.get("high", 0)),
            "l": float(x.get("low", 0)),
            "c": float(x.get("close", 0)),
        })
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


# ✅ round UP to 2dp
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

        # for UI display
        self.current_stake = 0.0
        self.martingale_step = 0
        self.martingale_halt = False  # stop-after-5-martingales flag

        # ✅ SECTIONS state
        self.section_profit = 0.0
        self.sections_won_today = 0
        self.section_index = 1  # 1..SECTIONS_PER_DAY
        self.section_pause_until = 0.0  # when section target hit, wait to next section boundary

        self.trade_lock = asyncio.Lock()

        self.market_debug = {m: {} for m in MARKETS}
        self.last_processed_closed_t0 = {m: 0 for m in MARKETS}

        self.tz = ZoneInfo("Africa/Lagos")
        self.current_day = datetime.now(self.tz).date()
        self.pause_until = 0.0

        # ✅ HTF cache (ADDED)
        self.htf_cache = {m: {"ts": 0.0, "bias": "N/A"} for m in MARKETS}

    # ---------- Gateway helpers ----------
    @staticmethod
    def _is_gatewayish_error(msg: str) -> bool:
        m = (msg or "").lower()
        return any(k in m for k in [
            "gateway", "bad gateway", "502", "503", "504",
            "timeout", "timed out", "temporarily unavailable",
            "connection", "websocket", "not connected", "disconnect",
            "internal server error", "service unavailable"
        ])

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

    # ---------- Section helpers ----------
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
        # if already past last section start, next is next midnight
        if idx0 + 1 >= SECTIONS_PER_DAY:
            next_midnight = (datetime.fromtimestamp(midnight, self.tz) + timedelta(days=1)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            return next_midnight.timestamp()
        return float(next_start)

    def _section_boundaries(self):
        midnight = self._today_midnight_epoch()
        starts = [midnight + i * SECTION_LENGTH_SEC for i in range(SECTIONS_PER_DAY)]
        ends = [s + SECTION_LENGTH_SEC for s in starts]
        return starts, ends

    def _sync_section_if_needed(self):
        """Auto-advance section by time (and reset section_profit)"""
        now = time.time()
        today = datetime.now(self.tz).date()
        if today != self.current_day:
            return  # daily reset handles everything

        new_idx = self._get_section_index_for_epoch(now)
        if new_idx != self.section_index:
            # new time window -> reset section state
            self.section_index = new_idx
            self.section_profit = 0.0
            self.section_pause_until = 0.0

    # ---------- Deriv connection helpers ----------
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
                low = msg.lower()
                if self._is_gatewayish_error(low):
                    await self.safe_reconnect()
                await asyncio.sleep(min(8.0, 0.6 * attempt + random.random() * 0.5))
        raise last_err

    async def safe_ticks_history(self, payload: dict, retries: int = 6):
        return await self.safe_deriv_call("ticks_history", payload, retries=retries)

    async def fetch_balance(self):
        if not self.api:
            return
        try:
            bal = await self.safe_deriv_call("balance", {"balance": 1}, retries=4)
            self.balance = f"{float(bal['balance']['balance']):.2f} {bal['balance']['currency']}"
        except Exception:
            pass

    # ========================= ✅ HTF BIAS (ADDED) =========================
    async def _get_htf_bias(self, symbol: str) -> str:
        """
        Returns: "BULL", "BEAR", or "N/A"
        Cache is used to avoid hammering the API.
        """
        if not USE_HTF_BIAS:
            return "N/A"

        now = time.time()
        cached = self.htf_cache.get(symbol, {"ts": 0.0, "bias": "N/A"})
        if (now - float(cached.get("ts", 0.0))) < float(HTF_REFRESH_SEC) and cached.get("bias") != "N/A":
            return str(cached.get("bias", "N/A"))

        payload = {
            "ticks_history": symbol,
            "end": "latest",
            "count": int(HTF_CANDLES_COUNT),
            "style": "candles",
            "granularity": int(HTF_SEC),
        }
        data = await self.safe_ticks_history(payload, retries=6)
        candles = build_candles_from_deriv(data.get("candles", []))
        if len(candles) < (HTF_EMA_SLOW + 5):
            self.htf_cache[symbol] = {"ts": now, "bias": "N/A"}
            return "N/A"

        closes = [x["c"] for x in candles]
        ema_fast_arr = calculate_ema(closes, int(HTF_EMA_FAST))
        ema_slow_arr = calculate_ema(closes, int(HTF_EMA_SLOW))
        if len(ema_fast_arr) == 0 or len(ema_slow_arr) == 0:
            self.htf_cache[symbol] = {"ts": now, "bias": "N/A"}
            return "N/A"

        ema_fast = float(ema_fast_arr[-2])  # last closed HTF candle
        ema_slow = float(ema_slow_arr[-2])

        bias = "BULL" if ema_fast > ema_slow else "BEAR" if ema_fast < ema_slow else "N/A"
        self.htf_cache[symbol] = {"ts": now, "bias": bias}
        return bias

    # ========================= DAILY RESET / PAUSE =========================
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

            # ✅ reset sections daily
            self.section_profit = 0.0
            self.sections_won_today = 0
            self.section_index = self._get_section_index_for_epoch(time.time())
            self.section_pause_until = 0.0

            # ✅ reset HTF cache daily (ADDED)
            self.htf_cache = {m: {"ts": 0.0, "bias": "N/A"} for m in MARKETS}

        # Also auto-advance section by time within the same day
        self._sync_section_if_needed()

    def can_auto_trade(self) -> tuple[bool, str]:
        self._daily_reset_if_needed()

        # ✅ stop only when 5 martingales have been completed and it still lost
        if self.martingale_halt:
            return False, f"Stopped: Martingale {MARTINGALE_MAX_STEPS} steps completed"

        # ✅ section pause (auto resumes when time passes)
        if time.time() < self.section_pause_until:
            left = int(self.section_pause_until - time.time())
            return False, f"Section target hit (+${SECTION_PROFIT_TARGET:.2f}). Resumes {fmt_hhmm(self.section_pause_until)} ({left}s)"

        if time.time() < self.pause_until:
            left = int(self.pause_until - time.time())
            return False, f"Paused until 12:00am WAT ({left}s)"

        if self.total_profit_today >= DAILY_PROFIT_TARGET:
            self.pause_until = self._next_midnight_epoch()
            return False, f"Daily target reached (+${self.total_profit_today:.2f})"

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
            "granularity": TF_SEC
        }
        data = await self.safe_ticks_history(payload, retries=6)
        return build_candles_from_deriv(data.get("candles", []))

    async def _sleep_until(self, epoch_ts: float, buffer_s: float = 0.15):
        wait = (epoch_ts - time.time()) + buffer_s
        if wait > 0:
            await asyncio.sleep(wait)
        else:
            await asyncio.sleep(0.10)

    async def _sync_to_next_candle_open(self, last_closed_t0: int):
        next_open = int(last_closed_t0) + int(TF_SEC)
        await self._sleep_until(next_open, buffer_s=0.20)

    async def scan_market(self, symbol: str):
        while self.is_scanning:
            try:
                if self.consecutive_losses >= MAX_CONSEC_LOSSES or self.trades_today >= MAX_TRADES_PER_DAY:
                    self.is_scanning = False
                    break

                ok_gate, gate = self.can_auto_trade()

                candles = await self.fetch_real_m1_candles(symbol)
                if len(candles) < 70:
                    self.market_debug[symbol] = {
                        "time": time.time(),
                        "gate": "Waiting for more candles",
                        "why": [f"Not enough candle history yet (need ~70, have {len(candles)})."],
                    }
                    await asyncio.sleep(SCAN_SLEEP_SEC)
                    continue

                pullback = candles[-3]
                confirm = candles[-2]
                confirm_t0 = int(confirm["t0"])

                if self.last_processed_closed_t0[symbol] == confirm_t0:
                    await self._sync_to_next_candle_open(confirm_t0)
                    continue

                closes = [x["c"] for x in candles]

                ema20_arr = calculate_ema(closes, 20)
                ema50_arr = calculate_ema(closes, 50)
                if len(ema20_arr) < 60 or len(ema50_arr) < 60:
                    self.market_debug[symbol] = {"time": time.time(), "gate": "Indicators", "why": ["EMA20/EMA50 not ready yet."]}
                    await asyncio.sleep(SCAN_SLEEP_SEC)
                    continue

                ema20_pullback = float(ema20_arr[-3])
                ema20_confirm = float(ema20_arr[-2])
                ema50_confirm = float(ema50_arr[-2])

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
                if len(rsi_arr) < 60 or np.isnan(rsi_arr[-2]):
                    self.market_debug[symbol] = {"time": time.time(), "gate": "Indicators", "why": ["RSI not ready yet."]}
                    await asyncio.sleep(SCAN_SLEEP_SEC)
                    continue
                rsi_now = float(rsi_arr[-2])

                pb_high = float(pullback["h"])
                pb_low = float(pullback["l"])
                touched_ema20 = (pb_low <= ema20_pullback <= pb_high)

                c_open = float(confirm["o"])
                c_close = float(confirm["c"])
                bull_confirm = c_close > c_open
                bear_confirm = c_close < c_open

                close_above_ema20 = c_close > ema20_confirm
                close_below_ema20 = c_close < ema20_confirm

                bodies = [abs(float(candles[i]["c"]) - float(candles[i]["o"])) for i in range(-22, -2)]
                avg_body = float(np.mean(bodies)) if len(bodies) >= 10 else float(
                    np.mean([abs(float(c["c"]) - float(c["o"])) for c in candles[-60:-2]])
                )
                last_body = abs(c_close - c_open)

                if symbol == "R_50":
                    spike_mult = 1.2
                    rsi_call_min, rsi_call_max = 53.0, 62.0
                    rsi_put_min, rsi_put_max = 38.0, 47.0
                    ema_diff_min = 0.30
                else:
                    spike_mult = 1.5
                    rsi_call_min, rsi_call_max = 52.0, 60.0
                    rsi_put_min, rsi_put_max = 40.0, 48.0
                    ema_diff_min = 0.20

                spike_block = (avg_body > 0 and last_body > spike_mult * avg_body)

                ema_diff = abs(ema20_confirm - ema50_confirm)
                flat_block = ema_diff < ema_diff_min

                uptrend = ema20_confirm > ema50_confirm
                downtrend = ema20_confirm < ema50_confirm

                call_rsi_ok = (rsi_call_min <= rsi_now <= rsi_call_max)
                put_rsi_ok = (rsi_put_min <= rsi_now <= rsi_put_max)

                call_ready = (
                    uptrend and slope_ok and ema50_rising and touched_ema20 and bull_confirm
                    and close_above_ema20 and call_rsi_ok and not spike_block and not flat_block
                )

                put_ready = (
                    downtrend and slope_ok and ema50_falling and touched_ema20 and bear_confirm
                    and close_below_ema20 and put_rsi_ok and not spike_block and not flat_block
                )

                # ✅ APPLY HTF BIAS FILTER (ADDED) — no UI changes
                if USE_HTF_BIAS:
                    htf_bias = await self._get_htf_bias(symbol)
                    if call_ready and htf_bias != "BULL":
                        call_ready = False
                    if put_ready and htf_bias != "BEAR":
                        put_ready = False

                signal = "CALL" if call_ready else "PUT" if put_ready else None

                trend_label = "UPTREND" if uptrend else "DOWNTREND" if downtrend else "SIDEWAYS"
                ema_label = "EMA20 ABOVE EMA50" if uptrend else "EMA20 BELOW EMA50" if downtrend else "EMA20 = EMA50"
                trend_strength = "STRONG" if not flat_block else "WEAK"
                pullback_label = "PULLBACK TOUCHED ✅" if touched_ema20 else "WAITING PULLBACK…"
                confirm_close_label = (
                    "CONFIRM CLOSE > EMA20 ✅" if close_above_ema20 else
                    "CONFIRM CLOSE < EMA20 ✅" if close_below_ema20 else
                    "CONFIRM CLOSE ON EMA20"
                )
                slope_label = "EMA50 SLOPE ↑" if ema50_rising else "EMA50 SLOPE ↓" if ema50_falling else "EMA50 SLOPE FLAT"

                block_label = []
                if spike_block:
                    block_label.append("SPIKE BLOCK")
                if flat_block:
                    block_label.append("WEAK/FLAT TREND")
                if not slope_ok:
                    block_label.append("SLOPE N/A")
                block_label = " | ".join(block_label) if block_label else "OK"

                why = []
                if not ok_gate:
                    why.append(f"Gate blocked: {gate}")
                if signal:
                    why.append(f"READY: {signal} (enter next candle)")

                self.market_debug[symbol] = {
                    "time": time.time(),
                    "gate": gate,
                    "last_closed": confirm_t0,
                    "signal": signal,
                    "trend_label": trend_label,
                    "ema_label": ema_label,
                    "trend_strength": trend_strength,
                    "pullback_label": pullback_label,
                    "block_label": block_label,
                    "confirm_close_label": confirm_close_label,
                    "slope_label": slope_label,
                    "ema50_slope": ema50_slope,
                    "rsi_now": rsi_now,
                    "avg_body": avg_body,
                    "last_body": last_body,
                    "spike_block": spike_block,
                    "flat_block": flat_block,
                    "why": why[:10],
                }

                self.last_processed_closed_t0[symbol] = confirm_t0

                if not ok_gate:
                    await self._sync_to_next_candle_open(confirm_t0)
                    continue

                if call_ready or put_ready:
                    await self._sync_to_next_candle_open(confirm_t0)

                if call_ready:
                    await self.execute_trade(
                        "CALL", symbol,
                        reason="Trend + EMA50SlopeUp + PullbackTouch + ConfirmGreen + Close>EMA20 + RSI + Filters",
                        source="AUTO",
                    )
                elif put_ready:
                    await self.execute_trade(
                        "PUT", symbol,
                        reason="Trend + EMA50SlopeDown + PullbackTouch + ConfirmRed + Close<EMA20 + RSI + Filters",
                        source="AUTO",
                    )

                await asyncio.sleep(0.25)

            except asyncio.CancelledError:
                break
            except Exception as e:
                msg = str(e)
                logger.error(f"Scanner Error ({symbol}): {msg}")
                self.market_debug[symbol] = {"time": time.time(), "gate": "Error", "why": [msg[:160]]}

                if self._is_gatewayish_error(msg):
                    await asyncio.sleep(4 + random.random() * 2)
                else:
                    await asyncio.sleep(2)

            await asyncio.sleep(SCAN_SLEEP_SEC)

    # ========================= PAYOUT MODE + 2x MARTINGALE (STOP AFTER 5) =========================
    async def execute_trade(self, side: str, symbol: str, reason="MANUAL", source="MANUAL"):
        if not self.api or self.active_trade_info:
            return

        async with self.trade_lock:
            ok, _gate = self.can_auto_trade()
            if not ok:
                return

            try:
                payout = float(PAYOUT_TARGET) * (float(MARTINGALE_MULT) ** int(self.martingale_step))
                payout = money2(payout)

                proposal_req = {
                    "proposal": 1,
                    "amount": payout,
                    "basis": "payout",          # ✅ payout mode (NOT stake)
                    "contract_type": side,
                    "currency": "USD",
                    "duration": int(DURATION_MIN),
                    "duration_unit": "m",
                    "symbol": symbol
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

                buy_price_cap = float(MAX_STAKE_ALLOWED)

                buy = await self.safe_deriv_call("buy", {"buy": proposal_id, "price": buy_price_cap}, retries=6)
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

                safe_symbol = str(symbol).replace("_", " ")
                msg = (
                    f"🚀 {side} TRADE OPENED\n"
                    f"🛒 Market: {safe_symbol}\n"
                    f"⏱ Expiry: {DURATION_MIN}m\n"
                    f"🎁 Payout: ${payout:.2f}\n"
                    f"🎲 Martingale step: {self.martingale_step}/{MARTINGALE_MAX_STEPS}\n"
                    f"🧩 Section: {self.section_index}/{SECTIONS_PER_DAY} | Section PnL: {self.section_profit:+.2f} / +{SECTION_PROFIT_TARGET:.2f}\n"
                    f"💵 Stake (Deriv): ${ask_price:.2f}\n"
                    f"🧾 Buy cap used: ${buy_price_cap:.2f}\n"
                    f"🧠 Reason: {reason}\n"
                    f"🤖 Source: {source}\n"
                    f"🎯 Today PnL: {self.total_profit_today:+.2f} / +{DAILY_PROFIT_TARGET:.2f}"
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
                retries=6
            )
            profit = float(res["proposal_open_contract"].get("profit", 0))

            if source == "AUTO":
                self.total_profit_today += profit
                self.section_profit += profit

                # ✅ Section target logic
                if self.section_profit >= float(SECTION_PROFIT_TARGET):
                    self.sections_won_today += 1
                    # pause until next section boundary (auto resume)
                    self.section_pause_until = self._next_section_start_epoch(time.time())

                if profit <= 0:
                    self.consecutive_losses += 1
                    self.total_losses_today += 1

                    if self.martingale_step < MARTINGALE_MAX_STEPS:
                        self.martingale_step += 1
                    else:
                        self.martingale_halt = True
                        self.is_scanning = False
                else:
                    self.consecutive_losses = 0
                    self.martingale_step = 0
                    self.martingale_halt = False

                if self.total_profit_today >= DAILY_PROFIT_TARGET:
                    self.pause_until = self._next_midnight_epoch()

            await self.fetch_balance()

            pause_note = ""
            if time.time() < self.pause_until:
                pause_note = f"\n⏸ Paused until 12:00am WAT"

            halt_note = ""
            if self.martingale_halt:
                halt_note = f"\n🛑 Martingale stopped after {MARTINGALE_MAX_STEPS} steps"

            section_note = ""
            if time.time() < self.section_pause_until:
                section_note = f"\n🧩 Section hit (+${SECTION_PROFIT_TARGET:.2f}). Auto-resume at {fmt_hhmm(self.section_pause_until)}"

            next_payout = money2(float(PAYOUT_TARGET) * (float(MARTINGALE_MULT) ** int(self.martingale_step)))

            await self.safe_send_tg(
                (
                    f"🏁 FINISH: {'WIN' if profit > 0 else 'LOSS'} ({profit:+.2f})\n"
                    f"🧩 Section: {self.section_index}/{SECTIONS_PER_DAY} | Section PnL: {self.section_profit:+.2f} | Sections won: {self.sections_won_today}\n"
                    f"📊 Today: {self.trades_today}/{MAX_TRADES_PER_DAY} | ❌ Losses: {self.total_losses_today} | Streak: {self.consecutive_losses}/{MAX_CONSEC_LOSSES}\n"
                    f"💵 Today PnL: {self.total_profit_today:+.2f} / +{DAILY_PROFIT_TARGET:.2f}\n"
                    f"🎁 Next payout: ${next_payout:.2f} (step {self.martingale_step}/{MARTINGALE_MAX_STEPS})\n"
                    f"💰 Balance: {self.balance}"
                    f"{pause_note}"
                    f"{section_note}"
                    f"{halt_note}"
                )
            )
        finally:
            self.active_trade_info = None
            self.cooldown_until = time.time() + COOLDOWN_SEC


# ========================= UI =========================
bot_logic = DerivSniperBot()


def main_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("▶️ START", callback_data="START_SCAN"),
         InlineKeyboardButton("⏹️ STOP", callback_data="STOP_SCAN")],
        [InlineKeyboardButton("📊 STATUS", callback_data="STATUS"),
         InlineKeyboardButton("🔄 REFRESH", callback_data="STATUS")],
        [InlineKeyboardButton("🧩 SECTION", callback_data="NEXT_SECTION")],  # ✅ NEW
        [InlineKeyboardButton("🧪 TEST BUY", callback_data="TEST_BUY")],
        [InlineKeyboardButton("🧪 DEMO", callback_data="SET_DEMO"),
         InlineKeyboardButton("💰 LIVE", callback_data="SET_REAL")]
    ])


def format_market_detail(sym: str, d: dict) -> str:
    if not d:
        return f"📍 {sym.replace('_',' ')}\n⏳ No scan data yet"

    age = int(time.time() - d.get("time", time.time()))
    gate = d.get("gate", "—")
    last_closed = d.get("last_closed", 0)
    signal = d.get("signal") or "—"

    trend_label = d.get("trend_label", "—")
    ema_label = d.get("ema_label", "—")
    trend_strength = d.get("trend_strength", "—")
    pullback_label = d.get("pullback_label", "—")
    block_label = d.get("block_label", "—")
    confirm_close_label = d.get("confirm_close_label", "—")
    slope_label = d.get("slope_label", "—")

    return (
        f"📍 {sym.replace('_',' ')} ({age}s)\n"
        f"Gate: {gate}\n"
        f"Last closed: {fmt_time_hhmmss(last_closed)}\n"
        f"────────────────\n"
        f"Trend: {trend_label} ({trend_strength})\n"
        f"{ema_label}\n"
        f"{slope_label}\n"
        f"{pullback_label}\n"
        f"{confirm_close_label}\n"
        f"Filters: {block_label}\n"
        f"Signal: {signal}\n"
    )


# ========================= FIX: CALLBACK "query too old" =========================
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
        await _safe_edit(q, "⚠️ LIVE CONNECTED" if ok else "❌ LIVE Failed", reply_markup=main_keyboard())

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
                "📌 Strategy: Trend(EMA20/EMA50) + EMA50 Slope + Pullback touch EMA20 + Confirm color + Close vs EMA20 + RSI (M1)\n"
                "🕯 Timeframe: M1\n"
                f"⏱ Expiry: {DURATION_MIN}m\n"
                f"🎁 PAYOUT MODE: ${PAYOUT_TARGET:.2f} base payout (Martingale {MARTINGALE_MULT:.0f}x up to {MARTINGALE_MAX_STEPS})\n"
                f"🧩 Sections: {SECTIONS_PER_DAY}/day | Target per section: +${SECTION_PROFIT_TARGET:.2f} (auto resume next section)\n"
                f"🧯 Max stake allowed: ${MAX_STAKE_ALLOWED:.2f} (skips if higher)\n"
                f"🎯 Daily Target: +${DAILY_PROFIT_TARGET:.2f} (pause till 12am WAT)"
            ),
            reply_markup=main_keyboard()
        )

    elif q.data == "STOP_SCAN":
        bot_logic.is_scanning = False
        await _safe_edit(q, "⏹️ Scanner stopped.", reply_markup=main_keyboard())

    elif q.data == "NEXT_SECTION":
        # ✅ NEW: manually advance to next section immediately, and auto resume (no other changes)
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
            f"🧩 Moved to Section {bot_logic.section_index}/{SECTIONS_PER_DAY}. Reset section PnL to 0.00.\n"
            f"✅ Auto trading continues normally (if START is on).",
            reply_markup=main_keyboard()
        )

    elif q.data == "TEST_BUY":
        asyncio.create_task(bot_logic.execute_trade("CALL", "R_10", "Manual Test", source="MANUAL"))
        await _safe_edit(q, "🧪 Test trade triggered (CALL R 10).", reply_markup=main_keyboard())

    elif q.data == "STATUS":
        await bot_logic.fetch_balance()
        now_time = datetime.now(ZoneInfo("Africa/Lagos")).strftime("%Y-%m-%d %H:%M:%S")
        _ok, gate = bot_logic.can_auto_trade()

        trade_status = "No Active Trade"
        if bot_logic.active_trade_info and bot_logic.api:
            try:
                res = await bot_logic.safe_deriv_call(
                    "proposal_open_contract",
                    {"proposal_open_contract": 1, "contract_id": bot_logic.active_trade_info},
                    retries=4
                )
                pnl = float(res["proposal_open_contract"].get("profit", 0))
                rem = max(0, int(DURATION_MIN * 60) - int(time.time() - bot_logic.trade_start_time))
                icon = "✅ PROFIT" if pnl > 0 else "❌ LOSS" if pnl < 0 else "➖ FLAT"
                mkt_clean = str(bot_logic.active_market).replace("_", " ")
                trade_status = f"🚀 Active Trade ({mkt_clean})\n📈 PnL: {icon} ({pnl:+.2f})\n⏳ Left: {rem}s"
            except Exception:
                trade_status = "🚀 Active Trade: Syncing..."

        pause_line = ""
        if time.time() < bot_logic.pause_until:
            pause_line = "⏸ Paused until 12:00am WAT\n"

        section_line = ""
        if time.time() < bot_logic.section_pause_until:
            section_line = f"🧩 Section paused until {fmt_hhmm(bot_logic.section_pause_until)}\n"

        next_payout = money2(float(PAYOUT_TARGET) * (float(MARTINGALE_MULT) ** int(bot_logic.martingale_step)))

        header = (
            f"🕒 Time (WAT): {now_time}\n"
            f"🤖 Bot: {'ACTIVE' if bot_logic.is_scanning else 'OFFLINE'} ({bot_logic.account_type})\n"
            f"{pause_line}"
            f"{section_line}"
            f"🧩 Section: {bot_logic.section_index}/{SECTIONS_PER_DAY} | Section PnL: {bot_logic.section_profit:+.2f} / +{SECTION_PROFIT_TARGET:.2f} | Sections won: {bot_logic.sections_won_today}\n"
            f"🎁 Next payout: ${next_payout:.2f} | Step: {bot_logic.martingale_step}/{MARTINGALE_MAX_STEPS}\n"
            f"🧯 Max stake allowed: ${MAX_STAKE_ALLOWED:.2f}\n"
            f"⏱ Expiry: {DURATION_MIN}m | Cooldown: {COOLDOWN_SEC}s\n"
            f"🎯 Daily Target: +${DAILY_PROFIT_TARGET:.2f}\n"
            f"📡 Markets: {', '.join(MARKETS).replace('_',' ')}\n"
            f"━━━━━━━━━━━━━━━\n{trade_status}\n━━━━━━━━━━━━━━━\n"
            f"💵 Total Profit Today: {bot_logic.total_profit_today:+.2f}\n"
            f"🎯 Trades: {bot_logic.trades_today}/{MAX_TRADES_PER_DAY} | ❌ Losses: {bot_logic.total_losses_today}\n"
            f"📉 Loss Streak: {bot_logic.consecutive_losses}/{MAX_CONSEC_LOSSES}\n"
            f"🚦 Gate: {gate}\n"
            f"💰 Balance: {bot_logic.balance}\n"
        )

        details = "\n\n📌 LIVE SCAN (Simple)\n\n" + "\n\n".join(
            [format_market_detail(sym, bot_logic.market_debug.get(sym, {})) for sym in MARKETS]
        )

        await _safe_edit(q, header + details, reply_markup=main_keyboard())


async def start_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    await u.message.reply_text(
        "💎 Deriv Bot\n"
        "📌 Strategy: Trend + EMA50Slope + Pullback + ConfirmColor + CloseVsEMA20 + RSI (M1)\n"
        "🕯 Timeframe: M1\n"
        f"⏱ Expiry: {DURATION_MIN}m\n"
        f"🎁 PAYOUT MODE: ${PAYOUT_TARGET:.2f} base payout (Martingale {MARTINGALE_MULT:.0f}x up to {MARTINGALE_MAX_STEPS})\n"
        f"🧩 Sections: {SECTIONS_PER_DAY}/day | Target per section: +${SECTION_PROFIT_TARGET:.2f}\n"
        f"🧯 Max stake allowed: ${MAX_STAKE_ALLOWED:.2f}\n"
        f"🎯 Daily Target: +${DAILY_PROFIT_TARGET:.2f} (pause till 12am WAT)\n",
        reply_markup=main_keyboard()
    )


if __name__ == "__main__":
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    bot_logic.app = app
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CallbackQueryHandler(btn_handler))
    app.run_polling(drop_pending_updates=True)
