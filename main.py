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
MARKETS = ["R_10", "R_25"]  # you can add "R_50" later if you want

TELEGRAM_TOKEN = "8253450930:AAHUhPk9TML-8kZlA9UaHZZvTUGdurN9MVY"
TELEGRAM_CHAT_ID = "7634818949"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ========================= TWO MODES (MANUAL SWITCHING) =========================
# NOTE: Strategy logic uses:
# - 5M for STRUCTURE + breakout confirmation
# - 1M for ENTRY (retest/pullback)
# Mode ONLY controls expiry duration (M1=5m, M5=20m).
MODE_CONFIG = {
    "M5": {"TF_SEC": 300, "DURATION_MIN": 20},
    "M1": {"TF_SEC": 60,  "DURATION_MIN": 5},
}
DEFAULT_MODE = "M1"  # recommended with this approach
CANDLES_COUNT = 220

# ========================= TRADE & RISK SETTINGS =========================
COOLDOWN_SEC = 180
MAX_TRADES_PER_DAY = 20
MAX_CONSEC_LOSSES = 6
STOP_AFTER_LOSSES = 3
DAILY_LOSS_LIMIT = -8000
DAILY_PROFIT_TARGET = 2.0

# ========================= PAYOUT MODE =========================
USE_PAYOUT_MODE = True
PAYOUT_TARGET = 1
MIN_PAYOUT = 0.35
MAX_STAKE_ALLOWED = 10.00

# ========================= MARTINGALE (3 steps, 1.8x) =========================
MARTINGALE_MULT = 1.80
MARTINGALE_MAX_STEPS = 3
MARTINGALE_HALT_ON_MAX = True

# ========================= CLEAN BREAKOUT (5M structure + 1M entry) =========================
# Indicators used:
# - EMA20 / EMA50 on 5M (trend filter)
# - Donchian range on 5M (support/resistance box)
EMA_FAST = 20
EMA_SLOW = 50
DONCHIAN_LEN = 20

USE_EMA_BIAS = True  # recommended: BUY only if EMA20>EMA50, SELL only if EMA20<EMA50

# Breakout confirmation on 5M (clean candle body vs recent)
BODY_LOOKBACK = 10
BODY_STRONG_MULT = 1.10  # breakout candle body must be >= median_body * this

# 1M retest entry tolerance
RETEST_TOL_PCT = 0.0003  # 0.03% of price (e.g., 5000 -> 1.5 points)
SETUP_TTL_MIN = 25       # expire setup if no retest within this time window

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


def _yn(v: bool) -> str:
    return "YES ✅" if v else "NO ❌"


# ========================= BOT CORE =========================
class DerivBreakoutBot:
    def __init__(self):
        self.api = None
        self.app = None
        self.active_token = None
        self.account_type = "None"

        # manual switching: affects expiry only
        self.mode = DEFAULT_MODE  # "M5" or "M1"

        self.is_scanning = False
        self.scanner_task = None
        self.market_tasks = {}

        # {"cid": int, "symbol": str, "start_ts": float, "duration_min": int, "tf_sec": int, "mode": str}
        self.active_trade = None

        self.cooldown_until = 0.0
        self.trades_today = 0
        self.total_losses_today = 0
        self.consecutive_losses = 0
        self.total_profit_today = 0.0
        self.balance = "0.00"
        self.session_losses = 0

        # martingale
        self.martingale_step = 0
        self.martingale_halt = False

        self.trade_lock = asyncio.Lock()

        # debug per (symbol, tf_name)
        # tf_name in {"ENTRY_1M", "STRUCTURE_5M"}
        self.market_debug = {(m, tf): {} for m in MARKETS for tf in ("ENTRY_1M", "STRUCTURE_5M")}
        self.last_processed_closed_t0 = {(m, tf): 0 for m in MARKETS for tf in ("ENTRY_1M", "STRUCTURE_5M")}

        # breakout setups created from 5M, entered from 1M
        # self.setups[symbol] = {
        #   "dir": "CALL"/"PUT",
        #   "level": float,          # broken resistance/support
        #   "created_t0": int,       # breakout confirm candle epoch
        #   "expires_at": float,     # epoch
        #   "ema_fast": float, "ema_slow": float,
        #   "waiting": str,          # text
        # }
        self.setups = {}

        self.tz = ZoneInfo("Africa/Lagos")
        self.current_day = datetime.now(self.tz).date()
        self.pause_until = 0.0

        self._ticks_lock = asyncio.Lock()
        self._last_ticks_ts = 0.0
        self._next_poll_epoch = {(m, tf): 0.0 for m in MARKETS for tf in ("ENTRY_1M", "STRUCTURE_5M")}
        self._rate_limit_strikes = {(m, tf): 0 for m in MARKETS for tf in ("ENTRY_1M", "STRUCTURE_5M")}

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
            self.setups = {}

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

                # expire old setups
                now = time.time()
                for sym in list(self.setups.keys()):
                    if now >= float(self.setups[sym].get("expires_at", 0)):
                        self.setups[sym]["waiting"] = "Setup expired (no retest)"
                        # keep it a little while for status visibility, then delete next loop
                        if now - float(self.setups[sym].get("expires_at", 0)) > 60:
                            del self.setups[sym]

                await asyncio.sleep(1)
        finally:
            for t in self.market_tasks.values():
                t.cancel()
            self.market_tasks.clear()

    def _bias_ok(self, direction: str, ema_fast_now: float, ema_slow_now: float) -> bool:
        if not USE_EMA_BIAS:
            return True
        if not (is_finite(ema_fast_now) and is_finite(ema_slow_now)):
            return False
        if direction == "CALL":
            return ema_fast_now > ema_slow_now
        return ema_fast_now < ema_slow_now

    @staticmethod
    def _near_threshold(price: float) -> float:
        # "approaching" threshold: 0.15% of price (same idea you saw before)
        return abs(price) * 0.0015

    async def scan_market(self, symbol: str):
        # schedule both loops (5M + 1M) without extra websockets
        self._next_poll_epoch[(symbol, "STRUCTURE_5M")] = time.time() + random.random() * 0.6
        self._next_poll_epoch[(symbol, "ENTRY_1M")] = time.time() + random.random() * 0.6

        while self.is_scanning:
            try:
                # -------------------- 5M STRUCTURE --------------------
                now = time.time()
                nxt5 = float(self._next_poll_epoch.get((symbol, "STRUCTURE_5M"), 0.0))
                if now >= nxt5:
                    ok_gate, gate = self.can_auto_trade()

                    candles_5m = await self.fetch_candles(symbol, tf_sec=300)
                    if len(candles_5m) >= (DONCHIAN_LEN + 60):
                        confirm = candles_5m[-2]
                        confirm_t0 = int(confirm["t0"])
                        next_close = confirm_t0 + 300
                        self._next_poll_epoch[(symbol, "STRUCTURE_5M")] = float(next_close + 0.30)

                        if self.last_processed_closed_t0[(symbol, "STRUCTURE_5M")] != confirm_t0:
                            closes = np.array([x["c"] for x in candles_5m], dtype=float)
                            highs = np.array([x["h"] for x in candles_5m], dtype=float)
                            lows = np.array([x["l"] for x in candles_5m], dtype=float)

                            ema_fast = calculate_ema(closes, EMA_FAST)
                            ema_slow = calculate_ema(closes, EMA_SLOW)
                            ema_fast_now = float(ema_fast[-2]) if len(ema_fast) else float("nan")
                            ema_slow_now = float(ema_slow[-2]) if len(ema_slow) else float("nan")

                            bias_up = is_finite(ema_fast_now) and is_finite(ema_slow_now) and (ema_fast_now > ema_slow_now)
                            bias_down = is_finite(ema_fast_now) and is_finite(ema_slow_now) and (ema_fast_now < ema_slow_now)

                            # Donchian window excludes confirm candle
                            window_high = float(np.max(highs[-(DONCHIAN_LEN + 2):-2]))
                            window_low = float(np.min(lows[-(DONCHIAN_LEN + 2):-2]))

                            c_open = float(confirm["o"])
                            c_close = float(confirm["c"])
                            c_high = float(confirm["h"])
                            c_low = float(confirm["l"])
                            body = abs(c_close - c_open)

                            # "strong body" vs recent median body
                            bodies = np.array([abs(float(x["c"]) - float(x["o"])) for x in candles_5m[-(BODY_LOOKBACK + 2):-2]], dtype=float)
                            med_body = float(np.median(bodies)) if len(bodies) else float("nan")
                            strong_body = is_finite(med_body) and med_body > 0 and (body >= med_body * float(BODY_STRONG_MULT))

                            breakout_call = (c_close > window_high)
                            breakout_put = (c_close < window_low)

                            # apply EMA bias filter for confirmation
                            bo_confirm_call = breakout_call and strong_body and self._bias_ok("CALL", ema_fast_now, ema_slow_now)
                            bo_confirm_put = breakout_put and strong_body and self._bias_ok("PUT", ema_fast_now, ema_slow_now)

                            last_price = await self.fetch_last_price(symbol)

                            # approaching?
                            near = self._near_threshold(last_price) if is_finite(last_price) else float("nan")
                            approaching_call = is_finite(last_price) and is_finite(window_high) and (0 <= (window_high - last_price) <= near)
                            approaching_put = is_finite(last_price) and is_finite(window_low) and (0 <= (last_price - window_low) <= near)

                            confirmed = bo_confirm_call or bo_confirm_put
                            confirmed_dir = "CALL" if bo_confirm_call else "PUT" if bo_confirm_put else None
                            broken_level = window_high if bo_confirm_call else window_low if bo_confirm_put else float("nan")

                            waiting = "—"
                            if confirmed:
                                waiting = "Waiting for 1M retest/pullback to broken level"
                                # create/replace setup
                                self.setups[symbol] = {
                                    "dir": confirmed_dir,
                                    "level": float(broken_level),
                                    "created_t0": int(confirm_t0),
                                    "expires_at": float(time.time() + SETUP_TTL_MIN * 60),
                                    "ema_fast": float(ema_fast_now),
                                    "ema_slow": float(ema_slow_now),
                                    "waiting": waiting,
                                    "gate": gate,
                                }
                            else:
                                # if no breakout confirmed, keep setup only if still valid; otherwise remove
                                if symbol in self.setups:
                                    # keep it until TTL, but update waiting text
                                    self.setups[symbol]["waiting"] = "Waiting for 5M breakout confirmation"

                            self.market_debug[(symbol, "STRUCTURE_5M")] = {
                                "time": time.time(),
                                "gate": gate,
                                "ok_gate": ok_gate,
                                "last_closed": confirm_t0,
                                "next_close": next_close,
                                "window_high": window_high,
                                "window_low": window_low,
                                "confirm_close": c_close,
                                "last_price": last_price,
                                "ema_fast": ema_fast_now,
                                "ema_slow": ema_slow_now,
                                "bias_up": bias_up,
                                "bias_down": bias_down,
                                "body": body,
                                "med_body": med_body,
                                "strong_body": strong_body,
                                "breakout_confirmed": bool(confirmed),
                                "breakout_dir": confirmed_dir,
                                "broken_level": broken_level if confirmed else float("nan"),
                                "approaching_call": bool(approaching_call),
                                "approaching_put": bool(approaching_put),
                            }
                            self.last_processed_closed_t0[(symbol, "STRUCTURE_5M")] = confirm_t0
                    else:
                        self._next_poll_epoch[(symbol, "STRUCTURE_5M")] = time.time() + 10

                # -------------------- 1M ENTRY (RETEST) --------------------
                now = time.time()
                nxt1 = float(self._next_poll_epoch.get((symbol, "ENTRY_1M"), 0.0))
                if now >= nxt1:
                    ok_gate, gate = self.can_auto_trade()

                    candles_1m = await self.fetch_candles(symbol, tf_sec=60)
                    if len(candles_1m) >= 40:
                        confirm = candles_1m[-2]
                        confirm_t0 = int(confirm["t0"])
                        next_close = confirm_t0 + 60
                        self._next_poll_epoch[(symbol, "ENTRY_1M")] = float(next_close + 0.25)

                        if self.last_processed_closed_t0[(symbol, "ENTRY_1M")] != confirm_t0:
                            last_price = await self.fetch_last_price(symbol)
                            c_open = float(confirm["o"])
                            c_close = float(confirm["c"])
                            c_high = float(confirm["h"])
                            c_low = float(confirm["l"])

                            setup = self.setups.get(symbol)
                            retest_ok = False
                            retest_dir = None
                            level = float("nan")
                            tol = float("nan")
                            waiting = "No 5M setup yet"

                            if setup and time.time() < float(setup.get("expires_at", 0)):
                                retest_dir = str(setup.get("dir"))
                                level = float(setup.get("level", float("nan")))
                                tol = max(0.0, abs(level) * float(RETEST_TOL_PCT)) if is_finite(level) else float("nan")

                                # retest rules:
                                # CALL: 1M candle touches level (low <= level+tol) and closes back ABOVE level
                                # PUT:  1M candle touches level (high >= level-tol) and closes back BELOW level
                                if retest_dir == "CALL":
                                    retest_ok = is_finite(level) and is_finite(tol) and (c_low <= (level + tol)) and (c_close > level)
                                elif retest_dir == "PUT":
                                    retest_ok = is_finite(level) and is_finite(tol) and (c_high >= (level - tol)) and (c_close < level)

                                waiting = setup.get("waiting", "Waiting for 1M retest")
                                if retest_ok:
                                    waiting = "Retest confirmed ✅ (entry allowed)"

                            # record debug
                            self.market_debug[(symbol, "ENTRY_1M")] = {
                                "time": time.time(),
                                "gate": gate,
                                "ok_gate": ok_gate,
                                "last_closed": confirm_t0,
                                "next_close": next_close,
                                "last_price": last_price,
                                "confirm_close": c_close,
                                "setup_active": bool(setup and time.time() < float(setup.get("expires_at", 0))),
                                "setup_dir": retest_dir,
                                "setup_level": level,
                                "tol": tol,
                                "retest_ok": bool(retest_ok),
                                "waiting": waiting,
                            }
                            self.last_processed_closed_t0[(symbol, "ENTRY_1M")] = confirm_t0

                            # if retest ok, place trade (expiry depends on MODE)
                            if retest_ok and ok_gate:
                                await self.execute_trade(retest_dir, symbol, source="AUTO")
                                # clear setup after entry to avoid duplicate entries
                                if symbol in self.setups:
                                    self.setups[symbol]["waiting"] = "Trade opened (setup consumed)"
                                    del self.setups[symbol]
                    else:
                        self._next_poll_epoch[(symbol, "ENTRY_1M")] = time.time() + 6

                await asyncio.sleep(0.05)

            except asyncio.CancelledError:
                break
            except Exception as e:
                msg = str(e)
                logger.error(f"Scanner Error ({symbol}): {msg}")
                for tf in ("STRUCTURE_5M", "ENTRY_1M"):
                    if self._is_rate_limit_error(msg):
                        self._rate_limit_strikes[(symbol, tf)] = int(self._rate_limit_strikes.get((symbol, tf), 0)) + 1
                        backoff = RATE_LIMIT_BACKOFF_BASE * self._rate_limit_strikes[(symbol, tf)]
                        backoff = min(240, backoff)
                        self._next_poll_epoch[(symbol, tf)] = time.time() + backoff
                    else:
                        self._next_poll_epoch[(symbol, tf)] = time.time() + (6 if self._is_gatewayish_error(msg) else 2)

    # ---------- trading ----------
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

                # expiry controlled by MODE
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
                    f"🧭 Mode (expiry profile): {mode}\n"
                    f"🛒 Market: {safe_symbol}\n"
                    f"⏱ Expiry: {int(dur_min)}m\n"
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


def _fnum(x, fmt=".5f"):
    try:
        if x is None:
            return "—"
        if isinstance(x, float) and not np.isfinite(x):
            return "—"
        return format(float(x), fmt)
    except Exception:
        return "—"


def format_market_detail(sym: str, d5: dict, d1: dict, setup: dict | None) -> str:
    # Very informative but still structured + readable
    sym_clean = sym.replace("_", " ")

    # 5M structure
    if not d5:
        s5 = f"STRUCTURE(5M): no data yet"
    else:
        bo_conf = bool(d5.get("breakout_confirmed", False))
        bo_dir = d5.get("breakout_dir", None)
        app_call = bool(d5.get("approaching_call", False))
        app_put = bool(d5.get("approaching_put", False))
        approaching = "CALL" if app_call else "PUT" if app_put else "NO"
        bias = "UP" if bool(d5.get("bias_up", False)) else "DOWN" if bool(d5.get("bias_down", False)) else "MIXED/NA"
        s5 = (
            f"STRUCTURE(5M)\n"
            f"- Breakout confirmed? {_yn(bo_conf)}"
            + (f" ({bo_dir})" if bo_conf and bo_dir else "") + "\n"
            f"- Approaching breakout? {_yn(approaching != 'NO')} ({approaching})\n"
            f"- Range High/Low: {_fnum(d5.get('window_high'))} / {_fnum(d5.get('window_low'))}\n"
            f"- EMA{EMA_FAST}/{EMA_SLOW} bias: {bias} | EMA20={_fnum(d5.get('ema_fast'))} EMA50={_fnum(d5.get('ema_slow'))}\n"
            f"- Breakout candle: close={_fnum(d5.get('confirm_close'))} body={_fnum(d5.get('body'), '.2f')} (median={_fnum(d5.get('med_body'), '.2f')}) strong={_yn(bool(d5.get('strong_body', False)))}\n"
            f"- Last closed: {fmt_time_hhmmss(int(d5.get('last_closed', 0)))} | Next close: {fmt_time_hhmmss(int(d5.get('next_close', 0)))}\n"
        )

    # 1M entry
    if not d1:
        s1 = f"ENTRY(1M): no data yet"
    else:
        active = bool(d1.get("setup_active", False))
        retest_ok = bool(d1.get("retest_ok", False))
        s1 = (
            f"ENTRY(1M)\n"
            f"- Setup active? {_yn(active)}"
            + (f" ({d1.get('setup_dir')})" if active and d1.get("setup_dir") else "") + "\n"
            f"- Retest confirmed? {_yn(retest_ok)}\n"
            f"- Level / tol: {_fnum(d1.get('setup_level'))} ± {_fnum(d1.get('tol'), '.5f')}\n"
            f"- Confirm close: {_fnum(d1.get('confirm_close'))} | Live: {_fnum(d1.get('last_price'))}\n"
            f"- Waiting: {d1.get('waiting', '—')}\n"
            f"- Last closed: {fmt_time_hhmmss(int(d1.get('last_closed', 0)))} | Next close: {fmt_time_hhmmss(int(d1.get('next_close', 0)))}\n"
        )

    # setup summary
    if setup:
        ttl = max(0, int(float(setup.get("expires_at", 0)) - time.time()))
        ssetup = (
            f"SETUP TRACKER\n"
            f"- Dir: {setup.get('dir','—')} | Broken level: {_fnum(setup.get('level'))}\n"
            f"- Created: {fmt_time_hhmmss(int(setup.get('created_t0', 0)))} | Expires in: {ttl}s\n"
            f"- Waiting: {setup.get('waiting','—')}\n"
        )
    else:
        ssetup = "SETUP TRACKER\n- No active setup\n"

    return (
        f"📍 {sym_clean}\n"
        f"{s5}"
        f"{ssetup}"
        f"{s1}"
        f"━━━━━━━━━━━━━━━━━━━━\n"
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

    elif q.data == "SWITCH_MODE":
        bot_logic.toggle_mode()
        cfg = bot_logic.get_mode_cfg()
        await _safe_edit(
            q,
            f"✅ Mode switched to {bot_logic.mode}\n"
            f"⏱ Expiry profile: {int(cfg['DURATION_MIN'])} minutes\n"
            f"📌 Strategy stays the same: 5M structure + 1M retest entry.",
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
            "🔍 SCANNER ACTIVE\n"
            f"🧭 Mode (expiry profile): {bot_logic.mode} | Expiry: {int(cfg['DURATION_MIN'])}m\n"
            "📌 Strategy:\n"
            f"- STRUCTURE: 5M Donchian({DONCHIAN_LEN}) + EMA{EMA_FAST}/{EMA_SLOW} trend filter\n"
            "- CONFIRM: 5M candle CLOSES outside range + strong body\n"
            "- ENTRY: 1M retest/pullback to broken level, then enter\n"
            f"- Retest tolerance: {RETEST_TOL_PCT*100:.3f}% | Setup TTL: {SETUP_TTL_MIN}m\n"
            f"🎲 Martingale: {MARTINGALE_MAX_STEPS} steps | x{MARTINGALE_MULT:.2f}\n",
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
            f"🧪 Test trade triggered (CALL {MARKETS[0].replace('_',' ')} | Expiry mode {bot_logic.mode}).",
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
                    f"Expiry mode: {bot_logic.active_trade.get('mode','—')} | Expiry: {dur}m\n"
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
            f"🧭 Expiry mode: {bot_logic.mode} | Expiry: {int(cfg['DURATION_MIN'])}m\n"
            f"🎲 Martingale: step {bot_logic.martingale_step}/{MARTINGALE_MAX_STEPS} (x{MARTINGALE_MULT:.2f}) | Next payout: ${next_payout:.2f}\n"
            f"🧊 Cooldown: {cooldown_left}s | ⏸ Pause: {pause_left}s\n"
            f"🧯 Risk: max/day {MAX_TRADES_PER_DAY} | max streak {MAX_CONSEC_LOSSES} | soft stop {STOP_AFTER_LOSSES}\n"
            f"🎯 Daily: target +{DAILY_PROFIT_TARGET:.2f} | limit {DAILY_LOSS_LIMIT:+.2f}\n"
            f"━━━━━━━━━━━━━━━\n{trade_status}\n━━━━━━━━━━━━━━━\n"
            f"💵 Today PnL: {bot_logic.total_profit_today:+.2f}\n"
            f"🎯 Trades: {bot_logic.trades_today}/{MAX_TRADES_PER_DAY} | ❌ Losses: {bot_logic.total_losses_today}\n"
            f"📉 Streak: {bot_logic.consecutive_losses}/{MAX_CONSEC_LOSSES} | Session losses: {bot_logic.session_losses}/{STOP_AFTER_LOSSES}\n"
            f"🚦 Gate: {gate}\n"
            f"💰 Balance: {bot_logic.balance}\n"
            f"📡 Markets: {', '.join(MARKETS).replace('_',' ')}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
        )

        body = ""
        for sym in MARKETS:
            d5 = bot_logic.market_debug.get((sym, "STRUCTURE_5M"), {})
            d1 = bot_logic.market_debug.get((sym, "ENTRY_1M"), {})
            setup = bot_logic.setups.get(sym)
            body += format_market_detail(sym, d5, d1, setup)

        await _safe_edit(q, header + body, reply_markup=main_keyboard())


async def start_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    cfg = bot_logic.get_mode_cfg()
    await u.message.reply_text(
        "💎 Deriv Breakout Bot (5M structure + 1M retest entry)\n"
        f"🧭 Expiry mode: {bot_logic.mode} | Expiry: {int(cfg['DURATION_MIN'])}m\n"
        f"📌 Indicators: EMA{EMA_FAST}/EMA{EMA_SLOW} + Donchian({DONCHIAN_LEN})\n"
        "📌 Logic:\n"
        "- Confirm breakout on 5M close outside the box (strong body)\n"
        "- Enter on 1M retest/pullback to the broken level\n",
        reply_markup=main_keyboard(),
    )


if __name__ == "__main__":
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    bot_logic.app = app
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CallbackQueryHandler(btn_handler))
    app.run_polling(drop_pending_updates=True)
