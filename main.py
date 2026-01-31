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
# M5 = 5-minute candles, 20-minute expiry
# M1 = 1-minute candles, 5-minute expiry
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
DAILY_LOSS_LIMIT = -8000
DAILY_PROFIT_TARGET = 2.0

# ========================= PAYOUT MODE =========================
USE_PAYOUT_MODE = True
PAYOUT_TARGET = 500
MIN_PAYOUT = 0.35
MAX_STAKE_ALLOWED = 10.00

# ========================= MARTINGALE (3 steps, 1.8x) =========================
MARTINGALE_MULT = 1.80
MARTINGALE_MAX_STEPS = 3
MARTINGALE_HALT_ON_MAX = True

# ========================= CLEAN BREAKOUT SETTINGS =========================
# Keep it simple:
# - Donchian breakout using ATR buffer
# - Spike filter (avoid huge candles)
# - Optional EMA bias filter (kept ON by default)
ATR_PERIOD = 14
DONCHIAN_LEN = 20
ATR_BREAKOUT_K = 0.15  # buffer = k * ATR

USE_EMA_BIAS = True    # If you want more trades, set False
EMA_FAST = 20
EMA_SLOW = 50

MIN_CANDLE_RANGE = 1e-6
SPIKE_RANGE_ATR = 2.5
SPIKE_BODY_ATR = 1.8

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

        # martingale
        self.martingale_step = 0
        self.martingale_halt = False

        self.trade_lock = asyncio.Lock()

        # debug per (symbol, mode)
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
            needed = ["GATE_OK", "ATR_OK", "SPIKE_OK", "BREAKOUT_CALL"]
            if USE_EMA_BIAS:
                needed.append("BIAS_UP")
        else:
            needed = ["GATE_OK", "ATR_OK", "SPIKE_OK", "BREAKOUT_PUT"]
            if USE_EMA_BIAS:
                needed.append("BIAS_DOWN")
        return [k for k in needed if not checks.get(k, False)]

    async def scan_market(self, symbol: str):
        for md in MODE_CONFIG.keys():
            self._next_poll_epoch[(symbol, md)] = time.time() + random.random() * 0.6

        while self.is_scanning:
            try:
                for md, cfg in MODE_CONFIG.items():
                    tf_sec = int(cfg["TF_SEC"])
                    now = time.time()
                    nxt = float(self._next_poll_epoch.get((symbol, md), 0.0))
                    if now < nxt:
                        continue

                    ok_gate, gate = self.can_auto_trade()

                    candles = await self.fetch_candles(symbol, tf_sec=tf_sec)
                    if len(candles) < (DONCHIAN_LEN + 10):
                        self.market_debug[(symbol, md)] = {
                            "time": time.time(),
                            "mode": md,
                            "tf_sec": tf_sec,
                            "gate": "Waiting",
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

                    atr = calculate_atr(highs, lows, closes, ATR_PERIOD)
                    atr_now = float(atr[-2]) if len(atr) and not np.isnan(atr[-2]) else float("nan")
                    atr_ok = is_finite(atr_now) and atr_now > 1e-12

                    # optional EMA bias (trend direction)
                    bias_up = bias_down = False
                    ema_fast_now = ema_slow_now = float("nan")
                    if USE_EMA_BIAS:
                        ema_fast = calculate_ema(closes, EMA_FAST)
                        ema_slow = calculate_ema(closes, EMA_SLOW)
                        if len(ema_fast) and len(ema_slow):
                            ema_fast_now = float(ema_fast[-2])
                            ema_slow_now = float(ema_slow[-2])
                            bias_up = ema_fast_now > ema_slow_now
                            bias_down = ema_fast_now < ema_slow_now

                    # spike filter on confirm candle
                    c_open = float(confirm["o"])
                    c_close = float(confirm["c"])
                    c_high = float(confirm["h"])
                    c_low = float(confirm["l"])
                    c_range = max(MIN_CANDLE_RANGE, c_high - c_low)
                    body = abs(c_close - c_open)

                    range_atr = (c_range / atr_now) if (atr_ok and atr_now > 0) else float("nan")
                    body_atr = (body / atr_now) if (atr_ok and atr_now > 0) else float("nan")
                    spike_block = (is_finite(range_atr) and range_atr > float(SPIKE_RANGE_ATR)) or (
                        is_finite(body_atr) and body_atr > float(SPIKE_BODY_ATR)
                    )

                    # Donchian window excludes confirm candle
                    window_high = float(np.max(highs[-(DONCHIAN_LEN + 2):-2]))
                    window_low = float(np.min(lows[-(DONCHIAN_LEN + 2):-2]))

                    buf = (float(ATR_BREAKOUT_K) * atr_now) if atr_ok else 0.0
                    call_break_level = window_high + buf
                    put_break_level = window_low - buf

                    breakout_call = c_close > call_break_level
                    breakout_put = c_close < put_break_level

                    last_price = await self.fetch_last_price(symbol)

                    checks = {
                        "GATE_OK": ok_gate,
                        "ATR_OK": atr_ok,
                        "SPIKE_OK": (not spike_block),
                        "BREAKOUT_CALL": breakout_call,
                        "BREAKOUT_PUT": breakout_put,
                        "BIAS_UP": bias_up,
                        "BIAS_DOWN": bias_down,
                    }

                    call_ready = checks["GATE_OK"] and checks["ATR_OK"] and checks["SPIKE_OK"] and checks["BREAKOUT_CALL"]
                    put_ready = checks["GATE_OK"] and checks["ATR_OK"] and checks["SPIKE_OK"] and checks["BREAKOUT_PUT"]
                    if USE_EMA_BIAS:
                        call_ready = call_ready and checks["BIAS_UP"]
                        put_ready = put_ready and checks["BIAS_DOWN"]

                    signal = "CALL" if call_ready else "PUT" if put_ready else None

                    self.market_debug[(symbol, md)] = {
                        "time": time.time(),
                        "mode": md,
                        "tf_sec": tf_sec,
                        "gate": gate,
                        "last_closed": confirm_t0,
                        "next_close": next_closed_epoch,
                        "signal": signal,
                        "checks": checks,
                        "missing_call": self._missing_for_side("CALL", checks),
                        "missing_put": self._missing_for_side("PUT", checks),
                        "confirm_close": c_close,
                        "last_price": last_price,
                        "call_break_level": call_break_level,
                        "put_break_level": put_break_level,
                        "next_poll_epoch": self._next_poll_epoch.get((symbol, md), 0.0),
                        "atr_now": atr_now,
                        "ema_fast": ema_fast_now,
                        "ema_slow": ema_slow_now,
                        "range_atr": range_atr,
                        "body_atr": body_atr,
                    }

                    self.last_processed_closed_t0[(symbol, md)] = confirm_t0

                    # trade ONLY selected mode
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
                for md in MODE_CONFIG.keys():
                    if self._is_rate_limit_error(msg):
                        self._rate_limit_strikes[(symbol, md)] = int(self._rate_limit_strikes.get((symbol, md), 0)) + 1
                        backoff = RATE_LIMIT_BACKOFF_BASE * self._rate_limit_strikes[(symbol, md)]
                        backoff = min(240, backoff)
                        self._next_poll_epoch[(symbol, md)] = time.time() + backoff
                    else:
                        self._next_poll_epoch[(symbol, md)] = time.time() + (6 if self._is_gatewayish_error(msg) else 2)

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


def _yn(v: bool) -> str:
    return "✅" if v else "❌"


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

    confirm_close = d.get("confirm_close", float("nan"))
    last_price = d.get("last_price", float("nan"))
    call_lvl = d.get("call_break_level", float("nan"))
    put_lvl = d.get("put_break_level", float("nan"))

    tf_txt = f"{tf_sec//60}m" if tf_sec else "—"

    # Keep status clean: show only the checks that matter
    keys = ["GATE_OK", "ATR_OK", "SPIKE_OK", "BREAKOUT_CALL", "BREAKOUT_PUT"]
    if USE_EMA_BIAS:
        keys += ["BIAS_UP", "BIAS_DOWN"]

    pretty = {
        "GATE_OK": "Gate",
        "ATR_OK": "ATR",
        "SPIKE_OK": "SpikeOK",
        "BREAKOUT_CALL": "BO Call",
        "BREAKOUT_PUT": "BO Put",
        "BIAS_UP": "BiasUp",
        "BIAS_DOWN": "BiasDn",
    }

    checks_line = " | ".join([f"{pretty[k]}:{_yn(bool(checks.get(k, False)))}" for k in keys])

    return (
        f"📍 {sym.replace('_',' ')} | 🧭 {mode} | 🕯 {tf_txt} ({age}s)\n"
        f"Gate: {gate}\n"
        f"Last closed: {fmt_time_hhmmss(last_closed)} | Next close: {fmt_time_hhmmss(next_close)}\n"
        f"Live: {f(last_price)} | Confirm close: {f(confirm_close)}\n"
        f"Break levels → CALL: {f(call_lvl)} | PUT: {f(put_lvl)}\n"
        f"Signal: {signal}\n"
        f"Checks: {checks_line}\n"
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
            f"🕯 Candles: {int(cfg['TF_SEC'])//60}m | ⏱ Expiry: {int(cfg['DURATION_MIN'])}m\n"
            f"Trades trigger ONLY for selected mode.",
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
            f"🧭 Selected mode: {bot_logic.mode}\n"
            f"📌 Strategy: Donchian breakout + ATR buffer + spike filter"
            + (f" + EMA({EMA_FAST}/{EMA_SLOW}) bias" if USE_EMA_BIAS else "") + "\n"
            f"🕯 {int(cfg['TF_SEC'])//60}m candles | ⏱ {int(cfg['DURATION_MIN'])}m expiry\n"
            f"🎲 Martingale: {MARTINGALE_MAX_STEPS} steps | x{MARTINGALE_MULT:.2f}\n"
            f"Note: Bot scans BOTH M1 & M5, but trades selected mode only.",
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

        details = "\n\n📌 LIVE SCAN (M1 + M5)\n\n"
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
        "💎 Deriv Breakout Bot (CLEAN)\n"
        f"🧭 Mode: {bot_logic.mode} (use MODE button to switch)\n"
        f"🕯 Timeframe: {int(cfg['TF_SEC'])//60}m | ⏱ Expiry: {int(cfg['DURATION_MIN'])}m\n"
        f"📌 Strategy: Donchian breakout + ATR buffer + spike filter"
        + (f" + EMA({EMA_FAST}/{EMA_SLOW}) bias" if USE_EMA_BIAS else "") + "\n"
        f"🎲 Martingale: {MARTINGALE_MAX_STEPS} steps | x{MARTINGALE_MULT:.2f}\n",
        reply_markup=main_keyboard(),
    )


if __name__ == "__main__":
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    bot_logic.app = app
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CallbackQueryHandler(btn_handler))
    app.run_polling(drop_pending_updates=True)
