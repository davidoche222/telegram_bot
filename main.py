# ⚠️ SECURITY NOTE:
# Do NOT post your Deriv / Telegram tokens publicly.
# Paste them only on your local machine.

import asyncio
import logging
import random
import time
import json
import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
from deriv_api import DerivAPI
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

# ========================= CONFIG =========================
# ✅ KEEP YOUR REAL TOKENS ON YOUR PC ONLY
DEMO_TOKEN = "tIrfitLjqeBxCOM"
REAL_TOKEN = "ZkOFWOlPtwnjqTS"
APP_ID = 1089

# ✅ R_75 and R_100 only — stable, less API pressure
MARKETS = ["R_75", "R_100"]

# ✅ cooldown
COOLDOWN_SEC = 3
MAX_TRADES_PER_DAY = 80
MAX_CONSEC_LOSSES = 10

# ✅ KEEP YOUR TELEGRAM TOKEN ON YOUR PC ONLY
TELEGRAM_TOKEN = "8589420556:AAHmB6YE9KIEu0tBIgWdd9baBDt0eDh5FY8"
TELEGRAM_CHAT_ID = "7634818949"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ========================= STRATEGY SETTINGS =========================
TF_SEC = 60  # M1 candles
CANDLES_COUNT = 210  # EMA200 needs 200+, no need for 300
RSI_PERIOD = 14
DURATION_MIN = 3  # 3-minute expiry

# ========================= SESSION FILTER =========================
ALLOWED_SESSIONS_UTC = None  # None = trade any time

# ========================= TREND/PULLBACK INDICATORS =========================
EMA_TREND_FAST = 50
EMA_TREND_SLOW = 200
EMA_PULLBACK = 20

RSI_BUY_MIN = 45.0
RSI_SELL_MAX = 55.0

ATR_PERIOD = 14
PULLBACK_ATR_MULT = 0.60

MIN_BODY_RATIO = 0.32
MIN_CANDLE_RANGE = 1e-6

# ========================= DAILY TARGETS / LIMITS =========================
DAILY_PROFIT_TARGET = 10.0

# ========================= SECTIONS =========================
SECTIONS_PER_DAY = 1
SECTION_PROFIT_TARGET = 3.0
SECTION_LENGTH_SEC = int(24 * 60 * 60 / SECTIONS_PER_DAY)

# ========================= PAYOUT MODE =========================
USE_PAYOUT_MODE = True
PAYOUT_TARGET = 1
MIN_PAYOUT = 0.35
MAX_STAKE_ALLOWED = 10.00

# ========================= MARTINGALE SETTINGS =========================
# ✅ UNTOUCHED — DO NOT MODIFY
MARTINGALE_MULT = 2
MARTINGALE_MAX_STEPS = 4
MARTINGALE_MAX_STAKE = 16.0

# ========================= ENTRY QUALITY TOGGLES =========================
USE_SPIKE_BLOCK = True
USE_STRONG_CANDLE_FILTER = True

# ========================= ANTI RATE-LIMIT =========================
TICKS_GLOBAL_MIN_INTERVAL = 0.35
RATE_LIMIT_BACKOFF_BASE = 20

# ========================= UI: REFRESH COOLDOWN =========================
STATUS_REFRESH_COOLDOWN_SEC = 10

# ========================= STATS (30 DAYS) =========================
STATS_DAYS = 30
TRADE_LOG_FILE = "trade_log.jsonl"

# ========================= SESSION BUCKETS (UTC) =========================
SESSION_BUCKETS = [
    ("ASIA", 0, 6),
    ("LONDON", 7, 11),
    ("OVERLAP", 12, 15),
    ("NEWYORK", 16, 20),
    ("LATE_NY", 21, 23),
]

# ========================= NEW: PROTECTIVE FILTER SETTINGS =========================
# Filter 1: Market Quality Score — only trade if score >= this
QUALITY_SCORE_MIN = 70

# Filter 2: EMA Slope — EMA50 must be moving, not flat
EMA_SLOPE_MIN = 0.00003  # minimum slope over last 5 candles (adjust per market)

# Filter 3: 2-loss pause
CONSEC_LOSS_PAUSE_COUNT = 2        # pause after this many consecutive losses
CONSEC_LOSS_PAUSE_MINUTES = 15     # pause for this many minutes

# Filter 4: Multi-timeframe agreement — 5M trend must match 15M trend
USE_MTF_CONFIRMATION = True

# Filter 5: Hourly block — block hour only if it has enough data AND win rate is below 30%
HOURLY_BLOCK_MIN_TRADES = 4       # need at least this many trades before blocking an hour
HOURLY_BLOCK_MAX_WINRATE = 0.30   # block if win rate drops below 30%


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


def money2(x: float) -> float:
    import math
    return math.ceil(float(x) * 100.0) / 100.0


def session_bucket(epoch_ts: float) -> str:
    dt = datetime.fromtimestamp(epoch_ts, ZoneInfo("UTC"))
    h = dt.hour
    for name, start_h, end_h in SESSION_BUCKETS:
        if start_h <= h <= end_h:
            return name
    return "UNKNOWN"


def is_strong_candle(candle: dict) -> tuple[bool, float]:
    c_open = float(candle["o"])
    c_close = float(candle["c"])
    c_high = float(candle["h"])
    c_low = float(candle["l"])
    rng = max(MIN_CANDLE_RANGE, c_high - c_low)
    body = abs(c_close - c_open)
    ratio = body / rng
    return ratio >= float(MIN_BODY_RATIO), ratio


def is_engulfing(prev: dict, cur: dict, direction: str) -> bool:
    po, pc = float(prev["o"]), float(prev["c"])
    co, cc = float(cur["o"]), float(cur["c"])
    prev_bear = pc < po
    prev_bull = pc > po
    cur_bull = cc > co
    cur_bear = cc < co

    if direction == "BUY":
        return prev_bear and cur_bull and (cc >= po) and (co <= pc)
    else:
        return prev_bull and cur_bear and (cc <= po) and (co >= pc)


def is_rejection(candle: dict, direction: str) -> bool:
    o = float(candle["o"])
    c = float(candle["c"])
    h = float(candle["h"])
    l = float(candle["l"])
    rng = max(MIN_CANDLE_RANGE, h - l)
    body = abs(c - o)
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l

    if direction == "BUY":
        return (lower_wick / rng) >= 0.45 and (body / rng) <= 0.55 and c >= o
    else:
        return (upper_wick / rng) >= 0.45 and (body / rng) <= 0.55 and c <= o


# ========================= NEW: MARKET QUALITY SCORE =========================
def calculate_quality_score(
    trend_up: bool,
    trend_down: bool,
    ema_slope: float,
    rsi_now: float,
    near_ema20: bool,
    atr_now: float,
    body_ratio: float,
    spike_block: bool,
    direction: str,
) -> tuple[int, list]:
    """
    Scores the trade setup from 0-100.
    Only trade if score >= QUALITY_SCORE_MIN (70).
    Returns (score, reasons_list).
    """
    score = 0
    reasons = []

    # 1. Trend clarity (30 points)
    if trend_up or trend_down:
        score += 30
        reasons.append("✅ Clear trend (+30)")
    else:
        reasons.append("❌ No trend (0)")

    # 2. EMA slope — trend must be actively moving (20 points)
    abs_slope = abs(ema_slope)
    if abs_slope >= EMA_SLOPE_MIN * 2:
        score += 20
        reasons.append(f"✅ Strong slope (+20)")
    elif abs_slope >= EMA_SLOPE_MIN:
        score += 10
        reasons.append(f"⚠️ Weak slope (+10)")
    else:
        reasons.append(f"❌ Flat EMA slope (0)")

    # 3. RSI quality (20 points)
    if direction == "BUY" and 42 <= rsi_now <= 58:
        score += 20
        reasons.append(f"✅ RSI in buy zone {rsi_now:.1f} (+20)")
    elif direction == "SELL" and 42 <= rsi_now <= 58:
        score += 20
        reasons.append(f"✅ RSI in sell zone {rsi_now:.1f} (+20)")
    elif 35 <= rsi_now <= 65:
        score += 10
        reasons.append(f"⚠️ RSI acceptable {rsi_now:.1f} (+10)")
    else:
        reasons.append(f"❌ RSI out of range {rsi_now:.1f} (0)")

    # 4. Pullback quality — price near EMA20 (15 points)
    if near_ema20:
        score += 15
        reasons.append("✅ Clean pullback to EMA20 (+15)")
    else:
        reasons.append("❌ Not near EMA20 (0)")

    # 5. Candle body quality (10 points)
    if body_ratio >= 0.45:
        score += 10
        reasons.append(f"✅ Strong candle body {body_ratio:.2f} (+10)")
    elif body_ratio >= 0.32:
        score += 5
        reasons.append(f"⚠️ Acceptable candle {body_ratio:.2f} (+5)")
    else:
        reasons.append(f"❌ Weak candle {body_ratio:.2f} (0)")

    # 6. No spike (5 points)
    if not spike_block:
        score += 5
        reasons.append("✅ No spike (+5)")
    else:
        reasons.append("❌ Spike detected (0)")

    return score, reasons


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
        self.active_trade_meta = None

        self.cooldown_until = 0.0
        self.trades_today = 0
        self.total_losses_today = 0
        self.consecutive_losses = 0
        self.total_profit_today = 0.0
        self.balance = "0.00"

        self.max_loss_streak_today = 0
        self.hit_5_losses_today = False

        self.current_stake = 0.0
        self.martingale_step = 0
        self.martingale_halt = False

        # sections
        self.section_profit = 0.0
        self.sections_won_today = 0
        self.section_index = 1
        self.section_pause_until = 0.0

        self.trade_lock = asyncio.Lock()
        self._pending_buy = False

        self.market_debug = {m: {} for m in MARKETS}
        self.last_processed_closed_t0 = {m: 0 for m in MARKETS}

        self.tz = ZoneInfo("Africa/Lagos")
        self.current_day = datetime.now(self.tz).date()
        self.pause_until = 0.0

        self._ticks_lock = asyncio.Lock()
        self._last_ticks_ts = 0.0
        self._next_poll_epoch = {m: 0.0 for m in MARKETS}
        self._rate_limit_strikes = {m: 0 for m in MARKETS}

        self.status_cooldown_until = 0.0

        self.trade_log_path = os.path.abspath(TRADE_LOG_FILE)
        self.trade_records = []
        self._load_trade_log()

        # ========================= NEW: PROTECTIVE STATE =========================
        # Filter 3: 2-loss pause
        self.consec_loss_pause_until = 0.0

        # Filter 5: Hourly performance tracker (resets daily)
        # Structure: { hour_int: {"wins": 0, "losses": 0} }
        self.hourly_stats_today = {}

        # Last quality score per market (for STATUS display)
        self.last_quality_score = {m: 0 for m in MARKETS}
        self.last_quality_reasons = {m: [] for m in MARKETS}

    # ---------- 30-day stats ----------
    def _load_trade_log(self):
        self.trade_records = []
        if not os.path.exists(self.trade_log_path):
            return
        try:
            with open(self.trade_log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        if isinstance(rec, dict) and "t" in rec:
                            self.trade_records.append(rec)
                    except Exception:
                        continue
            self._prune_trade_records()
        except Exception as e:
            logger.warning(f"Failed to load trade log: {e}")

    def _append_trade_log(self, rec: dict):
        try:
            with open(self.trade_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write trade log: {e}")

    def _prune_trade_records(self):
        cutoff = time.time() - (STATS_DAYS * 24 * 3600)
        self.trade_records = [r for r in self.trade_records if float(r.get("t", 0)) >= cutoff]

    def record_trade_result(self, symbol: str, open_epoch: float, profit: float):
        sess = session_bucket(open_epoch)
        win = 1 if profit > 0 else 0
        rec = {"t": float(open_epoch), "symbol": str(symbol), "session": str(sess), "win": int(win), "profit": float(profit)}
        self.trade_records.append(rec)
        self._append_trade_log(rec)
        self._prune_trade_records()

        # ========================= NEW: Update hourly stats =========================
        dt = datetime.fromtimestamp(open_epoch, self.tz)
        hour = dt.hour
        if hour not in self.hourly_stats_today:
            self.hourly_stats_today[hour] = {"wins": 0, "losses": 0}
        if win:
            self.hourly_stats_today[hour]["wins"] += 1
        else:
            self.hourly_stats_today[hour]["losses"] += 1

    def stats_30d(self):
        self._prune_trade_records()
        by_market = {}
        by_session = {}

        for r in self.trade_records:
            sym = r.get("symbol", "—")
            sess = r.get("session", "—")
            win = int(r.get("win", 0))

            by_market.setdefault(sym, {"wins": 0, "losses": 0, "trades": 0})
            by_session.setdefault(sess, {"wins": 0, "losses": 0, "trades": 0})

            by_market[sym]["trades"] += 1
            by_session[sess]["trades"] += 1
            if win == 1:
                by_market[sym]["wins"] += 1
                by_session[sess]["wins"] += 1
            else:
                by_market[sym]["losses"] += 1
                by_session[sess]["losses"] += 1

        def wr(d):
            t = d["trades"]
            return (100.0 * d["wins"] / t) if t > 0 else 0.0

        return by_market, by_session, wr

    # ========================= NEW: FILTER CHECKS =========================

    def _is_hour_blocked(self) -> tuple[bool, str]:
        """Filter 5: Block hour only if at least 4 trades have happened AND win rate < 30%.
        Needs real evidence before blocking — not just 2 bad trades."""
        now_hour = datetime.now(self.tz).hour
        stats = self.hourly_stats_today.get(now_hour, {"wins": 0, "losses": 0})
        wins = stats["wins"]
        losses = stats["losses"]
        total = wins + losses
        if total < HOURLY_BLOCK_MIN_TRADES:
            return False, "OK"  # not enough data yet — keep trading
        win_rate = wins / total
        if win_rate < HOURLY_BLOCK_MAX_WINRATE:
            return True, f"Hour {now_hour}:00 blocked ({wins}W/{losses}L = {win_rate*100:.0f}% WR)"
        return False, "OK"

    def _is_2loss_paused(self) -> tuple[bool, str]:
        """Filter 3: Pause 15 min after 2 consecutive losses."""
        if time.time() < self.consec_loss_pause_until:
            left = int(self.consec_loss_pause_until - time.time())
            return True, f"2-loss pause: {left}s remaining"
        return False, "OK"

    # ---------- helpers ----------
    @staticmethod
    def _is_gatewayish_error(msg: str) -> bool:
        m = (msg or "").lower()
        return any(k in m for k in [
            "gateway", "bad gateway", "502", "503", "504",
            "timeout", "timed out", "temporarily unavailable",
            "connection", "websocket", "not connected",
            "disconnect", "internal server error", "service unavailable",
        ])

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

    async def fetch_candles_with_timeout(self, symbol: str, granularity_sec: int, count: int, timeout_sec: float = 12.0):
        """Wraps fetch_candles with a timeout so one stuck market can't freeze others."""
        try:
            return await asyncio.wait_for(
                self.fetch_candles(symbol, granularity_sec, count),
                timeout=timeout_sec
            )
        except asyncio.TimeoutError:
            logger.warning(f"fetch_candles TIMEOUT ({symbol} gran={granularity_sec}s) — forcing reconnect")
            asyncio.create_task(self.safe_reconnect())
            return []
        except Exception as e:
            logger.warning(f"fetch_candles ERROR ({symbol} gran={granularity_sec}s): {e}")
            if self._is_gatewayish_error(str(e)):
                asyncio.create_task(self.safe_reconnect())
            return []

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

            self.section_profit = 0.0
            self.sections_won_today = 0
            self.section_index = self._get_section_index_for_epoch(time.time())
            self.section_pause_until = 0.0

            self.max_loss_streak_today = 0
            self.hit_5_losses_today = False

            # ========================= NEW: Reset daily protective state =========================
            self.consec_loss_pause_until = 0.0
            self.hourly_stats_today = {}

        self._sync_section_if_needed()

    def _session_gate_ok(self) -> tuple[bool, str]:
        if not ALLOWED_SESSIONS_UTC:
            return True, "OK"
        sess = session_bucket(time.time())
        if sess in ALLOWED_SESSIONS_UTC:
            return True, "OK"
        return False, f"Session blocked: {sess} (allowed: {', '.join(sorted(ALLOWED_SESSIONS_UTC))})"

    def can_auto_trade(self) -> tuple[bool, str]:
        self._daily_reset_if_needed()

        ok_sess, msg_sess = self._session_gate_ok()
        if not ok_sess:
            return False, msg_sess

        if self.martingale_halt:
            return False, f"Stopped: Martingale {MARTINGALE_MAX_STEPS} steps completed"

        if time.time() < self.section_pause_until:
            left = int(self.section_pause_until - time.time())
            return False, f"Section paused. Resumes {fmt_hhmm(self.section_pause_until)} ({left}s)"

        if time.time() < self.pause_until:
            left = int(self.pause_until - time.time())
            return False, f"Paused until 12:00am WAT ({left}s)"

        if self.total_profit_today >= DAILY_PROFIT_TARGET:
            self.pause_until = self._next_midnight_epoch()
            return False, f"Daily target reached (+${self.total_profit_today:.2f})"

        if self.total_profit_today <= -8.0:
            self.pause_until = self._next_midnight_epoch()
            return False, "Stopped: Daily loss limit (-$8.00) reached"

        if self.consecutive_losses >= MAX_CONSEC_LOSSES:
            return False, "Stopped: max loss streak reached"
        if self.trades_today >= MAX_TRADES_PER_DAY:
            return False, "Stopped: daily trade limit reached"
        if time.time() < self.cooldown_until:
            return False, f"Cooldown {int(self.cooldown_until - time.time())}s"
        if self.active_trade_info:
            return False, "Trade in progress"
        if self._pending_buy:
            return False, "Trade in progress (pending buy)"
        if not self.api:
            return False, "Not connected"

        # ========================= NEW: Protective filter gates =========================
        paused_2loss, msg_2loss = self._is_2loss_paused()
        if paused_2loss:
            return False, msg_2loss

        hour_blocked, msg_hour = self._is_hour_blocked()
        if hour_blocked:
            return False, msg_hour

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
                    self.active_trade_meta = None
                await asyncio.sleep(1)
        finally:
            for t in self.market_tasks.values():
                t.cancel()
            self.market_tasks.clear()

    async def fetch_candles(self, symbol: str, granularity_sec: int, count: int):
        payload = {
            "ticks_history": symbol,
            "end": "latest",
            "count": int(count),
            "style": "candles",
            "granularity": int(granularity_sec),
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

                candles_1m = await self.fetch_candles_with_timeout(symbol, 60, CANDLES_COUNT)
                candles_5m = await self.fetch_candles_with_timeout(symbol, 300, CANDLES_COUNT)
                candles_15m = await self.fetch_candles_with_timeout(symbol, 900, CANDLES_COUNT)

                # If any timeframe returned empty, skip and retry
                if not candles_1m or not candles_5m or not candles_15m:
                    self.market_debug[symbol] = {
                        "time": time.time(),
                        "gate": "Candle fetch failed",
                        "why": ["One or more timeframes returned empty — retrying in 8s"],
                    }
                    self._next_poll_epoch[symbol] = time.time() + 8
                    continue

                # Reset rate limit strikes after successful fetch
                self._rate_limit_strikes[symbol] = 0

                if len(candles_1m) < 30 or len(candles_5m) < 60 or len(candles_15m) < 220:
                    self.market_debug[symbol] = {
                        "time": time.time(),
                        "gate": "Waiting for more candles",
                        "why": [f"Need 15M>=220, 5M>=60, 1M>=30 | got 15M={len(candles_15m)} 5M={len(candles_5m)} 1M={len(candles_1m)}"],
                    }
                    self._next_poll_epoch[symbol] = time.time() + 12
                    continue

                confirm = candles_1m[-2]
                confirm_t0 = int(confirm["t0"])

                next_closed_epoch = confirm_t0 + TF_SEC
                self._next_poll_epoch[symbol] = float(next_closed_epoch + 0.35)

                if self.last_processed_closed_t0[symbol] == confirm_t0:
                    continue

                # ========================= 15M TREND =========================
                closes_15 = [x["c"] for x in candles_15m]
                ema_fast_15 = calculate_ema(closes_15, EMA_TREND_FAST)
                ema_slow_15 = calculate_ema(closes_15, EMA_TREND_SLOW)

                if len(ema_fast_15) < 10 or len(ema_slow_15) < 10:
                    self.market_debug[symbol] = {"time": time.time(), "gate": "Indicators", "why": ["15M EMA not ready."]}
                    self.last_processed_closed_t0[symbol] = confirm_t0
                    continue

                ema_fast_now = float(ema_fast_15[-2])
                ema_slow_now = float(ema_slow_15[-2])
                price_15 = float(closes_15[-2])

                trend_up_15 = (ema_fast_now > ema_slow_now) and (price_15 > ema_fast_now) and (price_15 > ema_slow_now)
                trend_down_15 = (ema_fast_now < ema_slow_now) and (price_15 < ema_fast_now) and (price_15 < ema_slow_now)

                # ========================= NEW FILTER 2: EMA SLOPE =========================
                # EMA50 slope over last 5 candles — must be actively moving
                ema_slope = (float(ema_fast_15[-2]) - float(ema_fast_15[-7])) / 5.0
                slope_ok_up = trend_up_15 and (ema_slope >= EMA_SLOPE_MIN)
                slope_ok_down = trend_down_15 and (ema_slope <= -EMA_SLOPE_MIN)
                slope_ok = slope_ok_up or slope_ok_down
                slope_label = f"EMA50 slope: {ema_slope:.6f} ({'✅ Active' if slope_ok else '❌ Flat'})"

                # ========================= 5M PULLBACK =========================
                closes_5 = [x["c"] for x in candles_5m]
                highs_5 = [x["h"] for x in candles_5m]
                lows_5 = [x["l"] for x in candles_5m]

                ema20_5 = calculate_ema(closes_5, EMA_PULLBACK)
                rsi_5 = calculate_rsi(closes_5, RSI_PERIOD)
                atr_5 = calculate_atr(highs_5, lows_5, closes_5, ATR_PERIOD)

                if len(ema20_5) < 10 or len(rsi_5) < 10 or len(atr_5) < 10 or np.isnan(rsi_5[-2]) or np.isnan(atr_5[-2]):
                    self.market_debug[symbol] = {"time": time.time(), "gate": "Indicators", "why": ["5M EMA/RSI/ATR not ready."]}
                    self.last_processed_closed_t0[symbol] = confirm_t0
                    continue

                ema20_now = float(ema20_5[-2])
                rsi5_now = float(rsi_5[-2])
                atr5_now = float(atr_5[-2])
                price_5 = float(closes_5[-2])

                near_ema20 = abs(price_5 - ema20_now) <= (atr5_now * float(PULLBACK_ATR_MULT))
                pullback_buy_ok = near_ema20 and (rsi5_now >= float(RSI_BUY_MIN))
                pullback_sell_ok = near_ema20 and (rsi5_now <= float(RSI_SELL_MAX))

                # ========================= NEW FILTER 4: MTF CONFIRMATION =========================
                # 5M trend must agree with 15M trend
                ema_fast_5 = calculate_ema(closes_5, EMA_TREND_FAST)
                trend_up_5m = len(ema_fast_5) >= 5 and (price_5 > float(ema_fast_5[-2]))
                trend_down_5m = len(ema_fast_5) >= 5 and (price_5 < float(ema_fast_5[-2]))

                mtf_buy_ok = trend_up_15 and trend_up_5m if USE_MTF_CONFIRMATION else trend_up_15
                mtf_sell_ok = trend_down_15 and trend_down_5m if USE_MTF_CONFIRMATION else trend_down_15
                mtf_label = f"MTF: 15M={'↑' if trend_up_15 else '↓' if trend_down_15 else '—'} 5M={'↑' if trend_up_5m else '↓' if trend_down_5m else '—'} ({'✅' if (mtf_buy_ok or mtf_sell_ok) else '❌'})"

                # ========================= 1M ENTRY =========================
                prev_1m = candles_1m[-3]
                cur_1m = candles_1m[-2]

                strong_ok, body_ratio = is_strong_candle(cur_1m)
                spike_block = False
                if USE_SPIKE_BLOCK:
                    bodies = [abs(float(candles_1m[i]["c"]) - float(candles_1m[i]["o"])) for i in range(-22, -2)]
                    avg_body = float(np.mean(bodies)) if bodies else 0.0
                    last_body = abs(float(cur_1m["c"]) - float(cur_1m["o"]))
                    spike_block = (avg_body > 0 and last_body > 1.5 * avg_body)

                entry_buy = (is_engulfing(prev_1m, cur_1m, "BUY") or is_rejection(cur_1m, "BUY"))
                entry_sell = (is_engulfing(prev_1m, cur_1m, "SELL") or is_rejection(cur_1m, "SELL"))

                strong_filter_ok = (strong_ok if USE_STRONG_CANDLE_FILTER else True)
                spike_ok = ((not spike_block) if USE_SPIKE_BLOCK else True)

                # ========================= NEW FILTER 1: QUALITY SCORE =========================
                direction = "BUY" if (mtf_buy_ok and pullback_buy_ok and entry_buy) else "SELL" if (mtf_sell_ok and pullback_sell_ok and entry_sell) else "NONE"
                quality_score, quality_reasons = calculate_quality_score(
                    trend_up=mtf_buy_ok,
                    trend_down=mtf_sell_ok,
                    ema_slope=ema_slope,
                    rsi_now=rsi5_now,
                    near_ema20=near_ema20,
                    atr_now=atr5_now,
                    body_ratio=body_ratio,
                    spike_block=spike_block,
                    direction=direction,
                )
                self.last_quality_score[symbol] = quality_score
                self.last_quality_reasons[symbol] = quality_reasons
                quality_ok = quality_score >= QUALITY_SCORE_MIN

                call_ready = mtf_buy_ok and pullback_buy_ok and entry_buy and strong_filter_ok and spike_ok and slope_ok and quality_ok
                put_ready = mtf_sell_ok and pullback_sell_ok and entry_sell and strong_filter_ok and spike_ok and slope_ok and quality_ok

                signal = "CALL" if call_ready else "PUT" if put_ready else None

                trend_label = "UPTREND" if trend_up_15 else "DOWNTREND" if trend_down_15 else "SIDEWAYS"
                ema_label = f"15M EMA{EMA_TREND_FAST}={'↑' if ema_fast_now > ema_slow_now else '↓'} EMA{EMA_TREND_SLOW}"
                pullback_label = f"5M Pullback: {'✅' if near_ema20 else '❌'} | dist={abs(price_5-ema20_now):.3f} <= ATR*{PULLBACK_ATR_MULT} ({atr5_now*PULLBACK_ATR_MULT:.3f})"
                confirm_close_label = f"1M Entry: {'✅' if (entry_buy or entry_sell) else '❌'} (engulf/reject)"

                block_parts = []
                if not (trend_up_15 or trend_down_15):
                    block_parts.append("NO 15M TREND")
                if not slope_ok:
                    block_parts.append("FLAT EMA SLOPE")
                if USE_MTF_CONFIRMATION and not (mtf_buy_ok or mtf_sell_ok):
                    block_parts.append("MTF MISMATCH")
                if trend_up_15 and not pullback_buy_ok:
                    block_parts.append("5M BUY FILTER FAIL")
                if trend_down_15 and not pullback_sell_ok:
                    block_parts.append("5M SELL FILTER FAIL")
                if USE_SPIKE_BLOCK and spike_block:
                    block_parts.append("SPIKE BLOCK")
                if USE_STRONG_CANDLE_FILTER and not strong_ok:
                    block_parts.append("WEAK CANDLE")
                if not quality_ok:
                    block_parts.append(f"LOW QUALITY SCORE ({quality_score}/100)")

                block_label = " | ".join(block_parts) if block_parts else "OK"

                why = []
                if not ok_gate:
                    why.append(f"Gate blocked: {gate}")
                if signal:
                    why.append(f"READY: {signal} | Score: {quality_score}/100")
                else:
                    why.append(f"No entry. Score: {quality_score}/100")

                self.market_debug[symbol] = {
                    "time": time.time(),
                    "gate": gate,
                    "last_closed": confirm_t0,
                    "signal": signal,
                    "trend_label": trend_label,
                    "ema_label": ema_label,
                    "trend_strength": "STRONG" if (trend_up_15 or trend_down_15) else "WEAK",
                    "pullback_label": pullback_label,
                    "confirm_close_label": confirm_close_label,
                    "slope_label": slope_label,
                    "mtf_label": mtf_label,
                    "block_label": block_label,
                    "quality_score": quality_score,
                    "rsi_now": rsi5_now,
                    "body_ratio": body_ratio,
                    "atr_now": atr5_now,
                    "why": why[:10],
                }

                self.last_processed_closed_t0[symbol] = confirm_t0

                if not ok_gate:
                    continue

                if call_ready:
                    await self.execute_trade("CALL", symbol, source="AUTO", rsi_now=rsi5_now, ema50_slope=ema_slope)
                elif put_ready:
                    await self.execute_trade("PUT", symbol, source="AUTO", rsi_now=rsi5_now, ema50_slope=ema_slope)

            except asyncio.CancelledError:
                break
            except Exception as e:
                msg = str(e)
                logger.error(f"Scanner Error ({symbol}): {msg}")

                if self._is_rate_limit_error(msg):
                    self._rate_limit_strikes[symbol] = int(self._rate_limit_strikes.get(symbol, 0)) + 1
                    backoff = RATE_LIMIT_BACKOFF_BASE * self._rate_limit_strikes[symbol]
                    backoff = min(60, backoff)
                    logger.warning(f"Rate limit on {symbol} — backoff {backoff}s")
                    self._next_poll_epoch[symbol] = time.time() + backoff
                elif self._is_gatewayish_error(msg):
                    logger.warning(f"Connection error on {symbol} — reconnecting...")
                    await self.safe_reconnect()
                    self._next_poll_epoch[symbol] = time.time() + 5
                else:
                    self._next_poll_epoch[symbol] = time.time() + 3

            await asyncio.sleep(0.05)

    # ========================= PAYOUT MODE + MARTINGALE =========================
    async def execute_trade(self, side: str, symbol: str, reason="MANUAL", source="MANUAL", rsi_now: float = 0.0, ema50_slope: float = 0.0):
        if not self.api or self.active_trade_info:
            return

        async with self.trade_lock:
            ok, _gate = self.can_auto_trade()
            if not ok:
                return
            if self._pending_buy:
                return

            self._pending_buy = True
            try:
                import math

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
                    await self.safe_send_tg(f"⛔️ Skipped trade: payout=${payout:.2f} needs stake=${ask_price:.2f} > max ${MAX_STAKE_ALLOWED:.2f}")
                    self.cooldown_until = time.time() + COOLDOWN_SEC
                    return

                buy_price_cap = float(MAX_STAKE_ALLOWED)

                # Don't retry BUY — network hiccup can cause DOUBLE PURCHASE
                buy = await self.safe_deriv_call("buy", {"buy": proposal_id, "price": buy_price_cap}, retries=1)

                if "error" in buy:
                    err_msg = str(buy["error"].get("message", "Buy error"))
                    await self.safe_send_tg(f"❌ Trade Refused:\n{err_msg}")
                    return

                self.active_trade_info = int(buy["buy"]["contract_id"])
                self.active_market = symbol
                self.trade_start_time = time.time()
                self.current_stake = ask_price

                self.active_trade_meta = {
                    "symbol": symbol,
                    "side": side,
                    "open_epoch": float(self.trade_start_time),
                    "source": source,
                }

                if source == "AUTO":
                    self.trades_today += 1

                quality_score = self.last_quality_score.get(symbol, 0)
                safe_symbol = str(symbol).replace("_", " ")
                msg = (
                    f"🚀 {side} TRADE OPENED\n"
                    f"🛒 Market: {safe_symbol}\n"
                    f"⏱ Expiry: {DURATION_MIN}m\n"
                    f"🎁 Payout: ${payout:.2f}\n"
                    f"🎲 Martingale step: {self.martingale_step}/{MARTINGALE_MAX_STEPS}\n"
                    f"💵 Stake (Deriv): ${ask_price:.2f}\n"
                    f"🏆 Quality Score: {quality_score}/100\n"
                    f"🕓 Session (UTC): {session_bucket(self.trade_start_time)}\n"
                    f"🤖 Source: {source}\n"
                    f"🎯 Today PnL: {self.total_profit_today:+.2f} / +{DAILY_PROFIT_TARGET:.2f}"
                )
                await self.safe_send_tg(msg)

                asyncio.create_task(self.check_result(self.active_trade_info, source, side, rsi_now, ema50_slope))

            except Exception as e:
                logger.error(f"Trade error: {e}")
                await self.safe_send_tg(f"⚠️ Trade error:\n{e}")
            finally:
                self._pending_buy = False

    async def check_result(self, cid: int, source: str, side: str, rsi_now: float, ema50_slope: float):
        await asyncio.sleep(int(DURATION_MIN) * 60 + 5)
        try:
            res = await self.safe_deriv_call("proposal_open_contract", {"proposal_open_contract": 1, "contract_id": cid}, retries=6)
            profit = float(res["proposal_open_contract"].get("profit", 0))

            if source == "AUTO" and self.active_trade_meta:
                sym = self.active_trade_meta.get("symbol", "—")
                open_epoch = float(self.active_trade_meta.get("open_epoch", time.time()))
                self.record_trade_result(sym, open_epoch, profit)

            if source == "AUTO":
                self.total_profit_today += profit
                self.section_profit += profit

                if self.section_profit >= float(SECTION_PROFIT_TARGET):
                    self.sections_won_today += 1
                    self.section_pause_until = self._next_section_start_epoch(time.time())

                if profit <= 0:
                    self.consecutive_losses += 1
                    self.total_losses_today += 1

                    if self.consecutive_losses > self.max_loss_streak_today:
                        self.max_loss_streak_today = self.consecutive_losses
                    if self.consecutive_losses >= 5 and not self.hit_5_losses_today:
                        self.hit_5_losses_today = True
                        await self.safe_send_tg("⚠️ ALERT: You have hit 5 losses in a row today (at least once).")

                    if self.martingale_step < MARTINGALE_MAX_STEPS:
                        self.martingale_step += 1
                    else:
                        self.martingale_halt = True
                        self.is_scanning = False

                    # ========================= NEW FILTER 3: 2-LOSS PAUSE =========================
                    if self.consecutive_losses >= CONSEC_LOSS_PAUSE_COUNT:
                        pause_secs = CONSEC_LOSS_PAUSE_MINUTES * 60
                        self.consec_loss_pause_until = time.time() + pause_secs
                        resume_time = fmt_hhmm(self.consec_loss_pause_until)
                        await self.safe_send_tg(
                            f"⏸ 2-LOSS PAUSE TRIGGERED\n"
                            f"📉 {self.consecutive_losses} consecutive losses detected\n"
                            f"⏳ Pausing for {CONSEC_LOSS_PAUSE_MINUTES} minutes\n"
                            f"▶️ Resumes at: {resume_time} WAT\n"
                            f"💡 Market may be choppy — waiting for it to stabilise"
                        )
                else:
                    self.consecutive_losses = 0
                    self.martingale_step = 0
                    self.martingale_halt = False

                if self.total_profit_today >= DAILY_PROFIT_TARGET:
                    self.pause_until = self._next_midnight_epoch()

            await self.fetch_balance()

            pause_note = "\n⏸ Paused until 12:00am WAT" if time.time() < self.pause_until else ""
            halt_note = f"\n🛑 Martingale stopped after {MARTINGALE_MAX_STEPS} steps" if self.martingale_halt else ""
            section_note = f"\n🧩 Section paused until {fmt_hhmm(self.section_pause_until)}" if time.time() < self.section_pause_until else ""
            pause_2loss_note = f"\n⏸ 2-Loss pause until {fmt_hhmm(self.consec_loss_pause_until)}" if time.time() < self.consec_loss_pause_until else ""

            next_payout = money2(float(PAYOUT_TARGET) * (float(MARTINGALE_MULT) ** int(self.martingale_step)))

            await self.safe_send_tg(
                f"🏁 FINISH: {'WIN ✅' if profit > 0 else 'LOSS ❌'} ({profit:+.2f})\n"
                f"🧩 Section: {self.section_index}/{SECTIONS_PER_DAY} | Section PnL: {self.section_profit:+.2f} / +{SECTION_PROFIT_TARGET:.2f}\n"
                f"📊 Today: {self.trades_today}/{MAX_TRADES_PER_DAY} | ❌ Losses: {self.total_losses_today} | Streak: {self.consecutive_losses}/{MAX_CONSEC_LOSSES}\n"
                f"📌 Max streak today: {self.max_loss_streak_today} | Hit 5-loss today: {'YES' if self.hit_5_losses_today else 'NO'}\n"
                f"💵 Today PnL: {self.total_profit_today:+.2f} / +{DAILY_PROFIT_TARGET:.2f}\n"
                f"🎁 Next payout: ${next_payout:.2f} (step {self.martingale_step}/{MARTINGALE_MAX_STEPS})\n"
                f"💰 Balance: {self.balance}"
                f"{pause_note}{section_note}{pause_2loss_note}{halt_note}"
            )
        finally:
            self.active_trade_info = None
            self.active_trade_meta = None
            self.cooldown_until = time.time() + COOLDOWN_SEC


# ========================= UI =========================
bot_logic = DerivSniperBot()


def main_keyboard():
    return InlineKeyboardMarkup([
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
    confirm_close_label = d.get("confirm_close_label", "—")
    slope_label = d.get("slope_label", "—")
    mtf_label = d.get("mtf_label", "—")
    block_label = d.get("block_label", "—")
    quality_score = d.get("quality_score", 0)

    rsi_now = d.get("rsi_now", None)
    body_ratio = d.get("body_ratio", None)
    atr_now = d.get("atr_now", None)

    extra = []
    if isinstance(rsi_now, (int, float)) and not np.isnan(rsi_now):
        extra.append(f"RSI(5M): {rsi_now:.2f}")
    if isinstance(atr_now, (int, float)) and not np.isnan(atr_now):
        extra.append(f"ATR(5M): {atr_now:.5f}")
    if isinstance(body_ratio, (int, float)) and not np.isnan(body_ratio):
        extra.append(f"Body: {body_ratio:.2f}")

    extra_line = " | ".join(extra) if extra else "—"
    why = d.get("why", [])
    why_line = "Why: " + (str(why[0]) if why else "—")

    score_bar = "🟢" if quality_score >= 70 else "🟡" if quality_score >= 50 else "🔴"

    return (
        f"📍 {sym.replace('_',' ')} ({age}s)\n"
        f"Gate: {gate}\n"
        f"Last closed: {fmt_time_hhmmss(last_closed)}\n"
        f"────────────────\n"
        f"Trend: {trend_label} ({trend_strength})\n"
        f"{ema_label}\n"
        f"{slope_label}\n"
        f"{mtf_label}\n"
        f"{pullback_label}\n"
        f"{confirm_close_label}\n"
        f"Stats: {extra_line}\n"
        f"Filters: {block_label}\n"
        f"{score_bar} Quality Score: {quality_score}/100\n"
        f"Signal: {signal}\n"
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
        await _safe_edit(q, "🔍 SCANNER ACTIVE\n✅ Press STATUS to monitor.", reply_markup=main_keyboard())

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

        await _safe_edit(q, f"🧩 Moved to Section {bot_logic.section_index}/{SECTIONS_PER_DAY}. Reset section PnL to 0.00.", reply_markup=main_keyboard())

    elif q.data == "TEST_BUY":
        test_symbol = MARKETS[0] if MARKETS else "R_75"
        asyncio.create_task(bot_logic.execute_trade("CALL", test_symbol, "Manual Test", source="MANUAL"))
        await _safe_edit(q, f"🧪 Test trade triggered (CALL {test_symbol.replace('_',' ')}).", reply_markup=main_keyboard())

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
                sess = session_bucket(bot_logic.trade_start_time)
                trade_status = f"🚀 Active Trade ({mkt_clean})\n🕓 Session(UTC): {sess}\n📈 PnL: {icon} ({pnl:+.2f})\n⏳ Left: {rem}s"
            except Exception:
                trade_status = "🚀 Active Trade: Syncing..."

        pause_line = "⏸ Paused until 12:00am WAT\n" if time.time() < bot_logic.pause_until else ""
        section_line = f"🧩 Section paused until {fmt_hhmm(bot_logic.section_pause_until)}\n" if time.time() < bot_logic.section_pause_until else ""
        pause_2loss_line = f"⏸ 2-Loss pause until {fmt_hhmm(bot_logic.consec_loss_pause_until)}\n" if time.time() < bot_logic.consec_loss_pause_until else ""

        next_payout = money2(float(PAYOUT_TARGET) * (float(MARTINGALE_MULT) ** int(bot_logic.martingale_step)))

        by_mkt, by_sess, wr = bot_logic.stats_30d()

        def fmt_stats_block(title: str, items: dict):
            rows = []
            for k, v in items.items():
                rows.append((k, wr(v), v["trades"], v["wins"], v["losses"]))
            rows.sort(key=lambda x: (x[1], x[2]), reverse=True)
            lines = [f"{title} (last {STATS_DAYS}d):"]
            if not rows:
                lines.append("— No trades recorded yet")
                return "\n".join(lines)
            for k, wrr, t, w, l in rows:
                lines.append(f"- {k.replace('_',' ')}: {wrr:.1f}% ({w}/{t})")
            return "\n".join(lines)

        # ========================= NEW: Hourly tracker display =========================
        hourly_lines = ["⏰ Hourly stats today (WAT):"]
        if bot_logic.hourly_stats_today:
            for h in sorted(bot_logic.hourly_stats_today.keys()):
                hs = bot_logic.hourly_stats_today[h]
                w = hs["wins"]
                l = hs["losses"]
                total = w + l
                wrate = (100 * w / total) if total > 0 else 0
                blocked = total >= HOURLY_BLOCK_MIN_TRADES and (w / total if total > 0 else 0) < HOURLY_BLOCK_MAX_WINRATE
                flag = "🚫" if blocked else ("⚠️" if total < HOURLY_BLOCK_MIN_TRADES else "✅")
                note = " (need more data)" if total < HOURLY_BLOCK_MIN_TRADES else (" BLOCKED" if blocked else "")
                hourly_lines.append(f"{flag} {h:02d}:00 — {wrate:.0f}% WR ({w}W/{l}L){note}")
        else:
            hourly_lines.append("— No trades yet today")
        hourly_block = "\n".join(hourly_lines)

        stats_block = (
            "📈 PERFORMANCE TRACKER\n"
            + fmt_stats_block("Markets", by_mkt)
            + "\n"
            + fmt_stats_block("Sessions(UTC)", by_sess)
            + "\n"
        )

        allowed_sess_line = (
            f"🕓 Allowed sessions(UTC): {', '.join(sorted(ALLOWED_SESSIONS_UTC))}\n" if ALLOWED_SESSIONS_UTC else "🕓 Allowed sessions(UTC): ALL\n"
        )

        header = (
            f"🕒 Time (WAT): {now_time}\n"
            f"🤖 Bot: {'ACTIVE' if bot_logic.is_scanning else 'OFFLINE'} ({bot_logic.account_type})\n"
            f"{allowed_sess_line}"
            f"{pause_line}{section_line}{pause_2loss_line}"
            f"🧩 Section: {bot_logic.section_index}/{SECTIONS_PER_DAY} | Section PnL: {bot_logic.section_profit:+.2f} / +{SECTION_PROFIT_TARGET:.2f}\n"
            f"🎁 Next payout: ${next_payout:.2f} | Step: {bot_logic.martingale_step}/{MARTINGALE_MAX_STEPS}\n"
            f"🧯 Max stake allowed: ${MAX_STAKE_ALLOWED:.2f}\n"
            f"⏱ Expiry: {DURATION_MIN}m | Cooldown: {COOLDOWN_SEC}s\n"
            f"🎯 Daily Target: +${DAILY_PROFIT_TARGET:.2f}\n"
            f"📡 Markets: {', '.join(MARKETS).replace('_',' ')}\n"
            f"🧭 Strategy: 15M trend + 5M pullback + 1M entry | MTF+Slope+Score\n"
            f"🏆 Min Quality Score: {QUALITY_SCORE_MIN}/100\n"
            f"━━━━━━━━━━━━━━━\n{trade_status}\n━━━━━━━━━━━━━━━\n"
            f"{stats_block}"
            f"{hourly_block}\n"
            f"━━━━━━━━━━━━━━━\n"
            f"💵 Total Profit Today: {bot_logic.total_profit_today:+.2f}\n"
            f"🎯 Trades: {bot_logic.trades_today}/{MAX_TRADES_PER_DAY} | ❌ Losses: {bot_logic.total_losses_today}\n"
            f"📉 Loss Streak: {bot_logic.consecutive_losses}/{MAX_CONSEC_LOSSES} | Max streak today: {bot_logic.max_loss_streak_today}\n"
            f"⚠️ Hit 5-loss streak today: {'YES' if bot_logic.hit_5_losses_today else 'NO'}\n"
            f"🚦 Gate: {gate}\n"
            f"💰 Balance: {bot_logic.balance}\n"
        )

        details = "\n\n📌 LIVE SCAN (FULL)\n\n" + "\n\n".join([format_market_detail(sym, bot_logic.market_debug.get(sym, {})) for sym in MARKETS])

        await _safe_edit(q, header + details, reply_markup=main_keyboard())


async def start_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    await u.message.reply_text(
        "💎 Deriv Sniper Bot v2\n"
        f"🕯 Entry: M1 | ⏱ Expiry: {DURATION_MIN}m\n"
        f"✅ 5 Protective Filters Active\n"
        f"✅ Quality Score Gate: {QUALITY_SCORE_MIN}/100\n"
        f"✅ MTF Confirmation: ON\n"
        f"✅ EMA Slope Filter: ON\n"
        f"✅ 2-Loss Pause: {CONSEC_LOSS_PAUSE_MINUTES}min\n"
        f"✅ Hourly Block: ON\n"
        f"✅ 30-day tracker: {TRADE_LOG_FILE}\n",
        reply_markup=main_keyboard(),
    )


if __name__ == "__main__":
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    bot_logic.app = app
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CallbackQueryHandler(btn_handler))
    app.run_polling(drop_pending_updates=True)
