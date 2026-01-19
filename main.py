import asyncio
import logging
import time
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
from deriv_api import DerivAPI
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

# ========================= CONFIG =========================
DEMO_TOKEN = "tIrfitLjqeBxCOM"
REAL_TOKEN = "2hsJzopRHG5w"
APP_ID = 1089

MARKETS = ["R_10", "R_25", "R_50", "R_75", "R_100"]

TELEGRAM_TOKEN = "8253450930:AAHUhPk9TML-8kZlA9UaHZZvTUGdurN9MVY"
TELEGRAM_CHAT_ID = "7634818949"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ========================= RISK / LIMITS =========================
MAX_TRADES_PER_DAY_TOTAL = 40                 # ✅ changed from 10 -> 40
MAX_TRADES_PER_MARKET_PER_DAY = 3
STOP_DAY_AFTER_TOTAL_LOSSES = 3
STOP_SYMBOL_AFTER_LOSSES = 2
MAX_CONSECUTIVE_LOSSES = 7                    # ✅ stop after 7 loss streak

COOLDOWN_AFTER_TRADE_SEC = 120
COOLDOWN_PER_SYMBOL_SEC = 120

BASE_STAKE = 1.00

# ========================= STRATEGY SETTINGS =========================
ENTRY_TF_SEC = 60  # M1 entries

STRUCT_TF_SEC = {
    "R_10": 300,   # M5
    "R_25": 300,   # M5
    "R_50": 900,   # M15
    "R_75": 900,   # M15
    "R_100": 900,  # M15
}

# ✅ Expiry set to 3 minutes for ALL markets
EXPIRY_MIN = {
    "R_10": 3,
    "R_25": 3,
    "R_50": 3,
    "R_75": 3,
    "R_100": 3,
}

SWING_N = 2
SWEEP_BUFFER_PCT = 0.0005

RSI_PERIOD = 14
RSI_LO = 30
RSI_HI = 70
USE_RSI_FILTER = True

DOJI_BODY_PCT = 0.25
CHOP_DOJI_COUNT = 6
CHOP_LOOKBACK = 10
CHOP_PAUSE_SEC = 600

SPIKE_MULTIPLIER = 2.0
AVG_RANGE_LOOKBACK = 20

# ========================= EMA STRATEGY SETTINGS (NEW) =========================
EMA_FAST = 20
EMA_SLOW = 50
EMA_PULLBACK_TOL_PCT = 0.0006   # how close price must come to EMA20 (0.06%)
EMA_MIN_SEPARATION_PCT = 0.0004 # avoid flat markets; require EMA20-EMA50 separation (0.04%)

# ========================= FETCH THROTTLES (RATE LIMIT FIX) =========================
M1_FETCH_MIN_INTERVAL_SEC = 15  # per symbol

# ========================= HELPERS =========================
def build_candles_from_deriv(candles_raw):
    out = []
    for x in candles_raw:
        out.append({
            "t": int(x.get("epoch", x.get("t", 0)) or 0),
            "o": float(x.get("open", x.get("o", 0))),
            "h": float(x.get("high", x.get("h", 0))),
            "l": float(x.get("low",  x.get("l", 0))),
            "c": float(x.get("close", x.get("c", 0))),
        })
    return out

def calculate_rsi(closes, period=14):
    closes = np.array(closes, dtype=float)
    if len(closes) < period + 1:
        return np.array([])
    delta = np.diff(closes)
    gain = np.clip(delta, 0, None)
    loss = np.clip(-delta, 0, None)
    avg_gain = np.convolve(gain, np.ones(period), "valid") / period
    avg_loss = np.convolve(loss, np.ones(period), "valid") / period
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_ema(values, period):
    values = np.array(values, dtype=float)
    if len(values) < period:
        return np.array([])
    k = 2.0 / (period + 1.0)
    ema = np.zeros_like(values)
    ema[0] = values[0]
    for i in range(1, len(values)):
        ema[i] = values[i] * k + ema[i-1] * (1.0 - k)
    return ema

def candle_range(c):
    return max(1e-9, c["h"] - c["l"])

def candle_body(c):
    return abs(c["c"] - c["o"])

def is_doji(c):
    return candle_body(c) <= DOJI_BODY_PCT * candle_range(c)

def avg_range(candles, lookback=20):
    if len(candles) < lookback + 1:
        lookback = max(2, len(candles) - 1)
    rngs = [candle_range(x) for x in candles[-lookback:]]
    return float(sum(rngs) / max(1, len(rngs)))

def strong_bull_close(c):
    rng = candle_range(c)
    return c["c"] >= (c["h"] - 0.30 * rng) and (c["c"] > c["o"])

def strong_bear_close(c):
    rng = candle_range(c)
    return c["c"] <= (c["l"] + 0.30 * rng) and (c["c"] < c["o"])

def bull_engulf(prev, cur):
    return (prev["c"] < prev["o"]) and (cur["c"] > cur["o"]) and (cur["c"] >= prev["o"]) and (cur["o"] <= prev["c"])

def bear_engulf(prev, cur):
    return (prev["c"] > prev["o"]) and (cur["c"] < cur["o"]) and (cur["c"] <= prev["o"]) and (cur["o"] >= prev["c"])

def find_swings(candles, n=2):
    highs = []
    lows = []
    if len(candles) < (2 * n + 1):
        return highs, lows

    for i in range(n, len(candles) - n):
        hi = candles[i]["h"]
        lo = candles[i]["l"]
        if all(hi > candles[j]["h"] for j in range(i - n, i)) and all(hi > candles[j]["h"] for j in range(i + 1, i + n + 1)):
            highs.append(i)
        if all(lo < candles[j]["l"] for j in range(i - n, i)) and all(lo < candles[j]["l"] for j in range(i + 1, i + n + 1)):
            lows.append(i)
    return highs, lows

def determine_bias(struct_candles):
    highs_idx, lows_idx = find_swings(struct_candles, SWING_N)

    if len(highs_idx) < 2 or len(lows_idx) < 2:
        return "RANGE", None, None

    h1, h2 = highs_idx[-2], highs_idx[-1]
    l1, l2 = lows_idx[-2], lows_idx[-1]

    last_high_1 = struct_candles[h1]["h"]
    last_high_2 = struct_candles[h2]["h"]
    last_low_1 = struct_candles[l1]["l"]
    last_low_2 = struct_candles[l2]["l"]

    up = (last_high_2 > last_high_1) and (last_low_2 > last_low_1)
    down = (last_high_2 < last_high_1) and (last_low_2 < last_low_1)

    liquidity_high = struct_candles[highs_idx[-1]]["h"]
    liquidity_low = struct_candles[lows_idx[-1]]["l"]

    if up:
        return "UP", liquidity_high, liquidity_low
    if down:
        return "DOWN", liquidity_high, liquidity_low
    return "RANGE", liquidity_high, liquidity_low

# ========================= LIVE SCAN (CLEAN FORMAT) =========================
def fmt_scan(sym, d):
    age = int(time.time() - d.get("time", time.time()))
    bias = d.get("bias", "?")
    setup = d.get("setup", "-")
    signal = d.get("signal", "-")
    waiting = (d.get("waiting", "") or "").strip()

    if "🚦" in waiting:
        status = "🚦 COOLDOWN"
        reason = waiting.replace("🚦", "").strip()
    elif "Chop detected" in waiting or "Chop filter" in waiting:
        status = "🟨 CHOP"
        reason = waiting
    elif "Spike candle" in waiting:
        status = "🟧 SPIKE"
        reason = "Spike candle (skipping this signal)"
    elif "rate limit" in waiting.lower():
        status = "🛑 RATE LIMIT"
        reason = waiting
    elif bias == "RANGE":
        status = "🚫 NO TRADE"
        reason = "Market is RANGE (no clear swing trend)"
    elif signal != "-" and ("BUY" in signal or "SELL" in signal):
        status = "✅ SIGNAL"
        reason = f"{setup} → {signal}"
    else:
        status = "⏳ WAIT"
        if waiting:
            reason = waiting
        else:
            reason = "Waiting for setup"

    levels = d.get("levels", "")
    ind = d.get("ind", "")

    short_ind = ""
    if ind:
        parts = [p.strip() for p in ind.split("|")]
        keep = []
        for p in parts:
            if p.lower().startswith("rsi"):
                keep.append(p.replace("RSI(14):", "RSI:").strip())
            if p.lower().startswith("doji"):
                keep.append(p.strip())
        short_ind = " | ".join(keep)

    out = []
    out.append(f"{sym.replace('_',' ')} ({age}s) → {status}  |  Trend: {bias}")
    out.append(f"Reason: {reason}")

    if status in ("✅ SIGNAL", "⏳ WAIT") and levels:
        out.append(f"Levels: {levels}")

    if short_ind:
        out.append(f"{short_ind}")

    return "\n".join(out)

# ========================= BOT CORE =========================
class DerivBot:
    def __init__(self):
        self.api = None
        self.app = None
        self.active_token = None
        self.account_type = "None"

        self.is_scanning = False
        self.scanner_task = None
        self.market_tasks = {}

        self.active_trade_info = None
        self.active_market = None
        self.trade_start_time = 0.0
        self.global_cooldown_until = 0.0

        self.day_key = None
        self.trades_today_total = 0
        self.losses_today_total = 0
        self.wins_today_total = 0
        self.trades_today_by_symbol = {m: 0 for m in MARKETS}
        self.losses_today_by_symbol = {m: 0 for m in MARKETS}
        self.disabled_symbol_today = {m: False for m in MARKETS}

        self.symbol_cooldown_until = {m: 0.0 for m in MARKETS}
        self.symbol_chop_until = {m: 0.0 for m in MARKETS}

        self.balance = "0.00"
        self.start_balance_value = None
        self.start_balance_text = None
        self.total_profit_today = 0.0
        self.current_stake = BASE_STAKE
        self.consecutive_losses = 0
        self.trade_lock = asyncio.Lock()

        self.market_debug = {}

        self.cache_m1 = {m: [] for m in MARKETS}
        self.cache_struct = {m: [] for m in MARKETS}
        self.last_fetch_m1 = {m: 0.0 for m in MARKETS}
        self.last_fetch_struct = {m: 0.0 for m in MARKETS}
        self.rate_backoff_until = {m: 0.0 for m in MARKETS}

        # ✅ anti-overtrade: trade only once per CLOSED M1 candle
        self.last_signal_candle_t = {m: 0 for m in MARKETS}

    def reset_day_if_needed(self):
        now = datetime.now(ZoneInfo("Africa/Lagos"))
        key = now.strftime("%Y-%m-%d")
        if self.day_key != key:
            self.day_key = key
            self.trades_today_total = 0
            self.losses_today_total = 0
            self.wins_today_total = 0
            self.trades_today_by_symbol = {m: 0 for m in MARKETS}
            self.losses_today_by_symbol = {m: 0 for m in MARKETS}
            self.disabled_symbol_today = {m: False for m in MARKETS}
            self.symbol_cooldown_until = {m: 0.0 for m in MARKETS}
            self.symbol_chop_until = {m: 0.0 for m in MARKETS}
            self.total_profit_today = 0.0
            self.current_stake = BASE_STAKE
            self.consecutive_losses = 0
            self.start_balance_value = None
            self.start_balance_text = None
            self.last_signal_candle_t = {m: 0 for m in MARKETS}

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

    async def fetch_balance(self):
        if not self.api:
            return
        try:
            bal = await self.api.balance({"balance": 1})
            bal_value = float(bal["balance"]["balance"])
            bal_ccy = bal["balance"]["currency"]
            self.balance = f"{bal_value:.2f} {bal_ccy}"

            self.reset_day_if_needed()
            if self.start_balance_value is None:
                self.start_balance_value = bal_value
                self.start_balance_text = self.balance
        except:
            pass

    def gate(self, symbol: str):
        self.reset_day_if_needed()

        if self.consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
            return False, f"Stopped: loss streak {self.consecutive_losses}/{MAX_CONSECUTIVE_LOSSES}"

        if self.disabled_symbol_today.get(symbol, False):
            return False, "Symbol disabled (loss limit)"
        if self.losses_today_total >= STOP_DAY_AFTER_TOTAL_LOSSES:
            return False, "Stopped: daily loss limit"
        if self.trades_today_total >= MAX_TRADES_PER_DAY_TOTAL:
            return False, "Daily trade limit reached"
        if self.trades_today_by_symbol.get(symbol, 0) >= MAX_TRADES_PER_MARKET_PER_DAY:
            return False, "Symbol trade cap reached"

        now = time.time()
        if now < self.global_cooldown_until:
            return False, f"Global cooldown {int(self.global_cooldown_until - now)}s"
        if now < self.symbol_cooldown_until.get(symbol, 0.0):
            return False, f"Symbol cooldown {int(self.symbol_cooldown_until[symbol] - now)}s"
        if now < self.symbol_chop_until.get(symbol, 0.0):
            return False, f"Chop filter {int(self.symbol_chop_until[symbol] - now)}s"

        if self.active_trade_info:
            return False, "Trade in progress"
        if not self.api:
            return False, "Not connected"
        return True, "OK"

    async def background_scanner(self):
        if not self.api:
            return
        self.market_tasks = {sym: asyncio.create_task(self.scan_symbol(sym)) for sym in MARKETS}
        try:
            while self.is_scanning:
                if self.active_trade_info and (time.time() - self.trade_start_time > (EXPIRY_MIN.get(self.active_market, 3) * 60 + 90)):
                    self.active_trade_info = None
                await asyncio.sleep(1)
        finally:
            for t in self.market_tasks.values():
                t.cancel()
            self.market_tasks.clear()

    async def fetch_candles_raw(self, symbol: str, granularity: int, count: int):
        res = await self.api.ticks_history({
            "ticks_history": symbol,
            "end": "latest",
            "count": count,
            "style": "candles",
            "granularity": granularity
        })
        raw = res.get("candles", [])
        return build_candles_from_deriv(raw)

    async def get_m1(self, symbol: str):
        now = time.time()
        if now < self.rate_backoff_until[symbol]:
            return self.cache_m1[symbol]

        if (now - self.last_fetch_m1[symbol]) < M1_FETCH_MIN_INTERVAL_SEC and len(self.cache_m1[symbol]) > 0:
            return self.cache_m1[symbol]

        try:
            candles = await self.fetch_candles_raw(symbol, ENTRY_TF_SEC, 220)
            self.cache_m1[symbol] = candles
            self.last_fetch_m1[symbol] = now
            return candles
        except Exception as e:
            msg = str(e)
            if "rate limit" in msg.lower() and "ticks_history" in msg.lower():
                self.rate_backoff_until[symbol] = now + 20
            raise

    async def get_struct(self, symbol: str):
        now = time.time()
        if now < self.rate_backoff_until[symbol]:
            return self.cache_struct[symbol]

        struct_tf = STRUCT_TF_SEC[symbol]
        if (now - self.last_fetch_struct[symbol]) < struct_tf and len(self.cache_struct[symbol]) > 0:
            return self.cache_struct[symbol]

        try:
            candles = await self.fetch_candles_raw(symbol, struct_tf, 220)
            self.cache_struct[symbol] = candles
            self.last_fetch_struct[symbol] = now
            return candles
        except Exception as e:
            msg = str(e)
            if "rate limit" in msg.lower() and "ticks_history" in msg.lower():
                self.rate_backoff_until[symbol] = now + 20
            raise

    async def scan_symbol(self, symbol: str):
        while self.is_scanning:
            try:
                ok, g = self.gate(symbol)
                if not ok:
                    self.market_debug[symbol] = {"time": time.time(), "bias": "-", "setup": "-", "signal": "-", "levels": "", "ind": "", "waiting": f"🚦 {g}"}
                    await asyncio.sleep(2)
                    continue

                struct = await self.get_struct(symbol)
                m1 = await self.get_m1(symbol)

                if len(struct) < 50 or len(m1) < 60:
                    self.market_debug[symbol] = {"time": time.time(), "bias": "…", "setup": "-", "signal": "-", "levels": "", "ind": "", "waiting": "Syncing candles..."}
                    await asyncio.sleep(3)
                    continue

                # ✅ use only CLOSED candle for signals
                c_prev = m1[-3]     # candle before confirmation
                c_confirm = m1[-2]  # last closed candle (confirmation)
                confirm_t = int(c_confirm.get("t", 0) or 0)

                # ✅ anti-overtrade: only one trade decision per closed candle
                if confirm_t != 0 and self.last_signal_candle_t[symbol] == confirm_t:
                    self.market_debug[symbol] = {
                        "time": time.time(),
                        "bias": "-",
                        "setup": "-",
                        "signal": "-",
                        "levels": "",
                        "ind": "",
                        "waiting": "Waiting: next M1 candle close"
                    }
                    await asyncio.sleep(2)
                    continue

                # Chop filter (unchanged)
                last10 = m1[-(CHOP_LOOKBACK+1):-1]
                dojis = sum(1 for x in last10 if is_doji(x))
                if dojis >= CHOP_DOJI_COUNT:
                    self.symbol_chop_until[symbol] = time.time() + CHOP_PAUSE_SEC
                    self.market_debug[symbol] = {
                        "time": time.time(),
                        "bias": "-",
                        "setup": "-",
                        "signal": "-",
                        "levels": "",
                        "ind": f"RSI(14): - | Doji:{dojis}/{CHOP_LOOKBACK}",
                        "waiting": "Chop detected → pausing 10m"
                    }
                    await asyncio.sleep(2)
                    continue

                # ✅ EMA strategy: trend from STRUCT timeframe (M5/M15) via EMA20/EMA50
                struct_closes = [x["c"] for x in struct]
                ema20_s = calculate_ema(struct_closes, EMA_FAST)
                ema50_s = calculate_ema(struct_closes, EMA_SLOW)
                if len(ema20_s) == 0 or len(ema50_s) == 0:
                    await asyncio.sleep(2)
                    continue

                ema20_struct = float(ema20_s[-1])
                ema50_struct = float(ema50_s[-1])

                if ema20_struct > ema50_struct:
                    bias = "UP"
                elif ema20_struct < ema50_struct:
                    bias = "DOWN"
                else:
                    bias = "RANGE"

                # ✅ avoid flat EMA (no trend strength)
                sep = abs(ema20_struct - ema50_struct) / max(1e-9, ema50_struct)
                if sep < EMA_MIN_SEPARATION_PCT:
                    self.market_debug[symbol] = {
                        "time": time.time(),
                        "bias": bias,
                        "setup": "-",
                        "signal": "-",
                        "levels": "",
                        "ind": f"RSI(14): - | Doji:{dojis}/10",
                        "waiting": "EMAs too close (choppy trend) → wait"
                    }
                    self.last_signal_candle_t[symbol] = confirm_t
                    await asyncio.sleep(2)
                    continue

                # ✅ M1 EMA for pullback entry
                closes_m1 = [x["c"] for x in m1]
                ema20_m1_arr = calculate_ema(closes_m1, EMA_FAST)
                ema50_m1_arr = calculate_ema(closes_m1, EMA_SLOW)
                if len(ema20_m1_arr) == 0 or len(ema50_m1_arr) == 0:
                    await asyncio.sleep(2)
                    continue

                ema20_m1 = float(ema20_m1_arr[-2])  # align with last CLOSED candle
                ema50_m1 = float(ema50_m1_arr[-2])

                # Optional RSI (kept, but now used as confirmation)
                rsi_arr = calculate_rsi(closes_m1, RSI_PERIOD)
                rsi_now = float(rsi_arr[-1]) if len(rsi_arr) > 0 else 50.0

                # Spike filter (unchanged)
                avg_rng = avg_range(m1[-(AVG_RANGE_LOOKBACK+1):-1], AVG_RANGE_LOOKBACK)
                sig_rng = candle_range(c_confirm)
                if sig_rng > SPIKE_MULTIPLIER * avg_rng:
                    self.market_debug[symbol] = {
                        "time": time.time(),
                        "bias": bias,
                        "setup": "-",
                        "signal": "-",
                        "levels": "",
                        "ind": f"RSI(14): {rsi_now:.0f} | Doji:{dojis}/10 | avgR:{avg_rng:.2f}",
                        "waiting": "Spike candle → skip"
                    }
                    self.last_signal_candle_t[symbol] = confirm_t
                    await asyncio.sleep(2)
                    continue

                # ✅ Pullback test: candle touches/near EMA20
                price = float(c_confirm["c"])
                tol = price * EMA_PULLBACK_TOL_PCT

                touched_ema20 = (c_confirm["l"] <= (ema20_m1 + tol)) and (c_confirm["h"] >= (ema20_m1 - tol))

                # ✅ Entry confirmation: rejection wick OR engulfing
                bullish_confirm = strong_bull_close(c_confirm) or bull_engulf(c_prev, c_confirm)
                bearish_confirm = strong_bear_close(c_confirm) or bear_engulf(c_prev, c_confirm)

                # ✅ RSI confirmation (single-indicator confirm). If USE_RSI_FILTER=True:
                # BUY: RSI > 50 ; SELL: RSI < 50
                rsi_buy_ok = (not USE_RSI_FILTER) or (rsi_now > 50)
                rsi_sell_ok = (not USE_RSI_FILTER) or (rsi_now < 50)

                buy_ready = (bias == "UP") and touched_ema20 and bullish_confirm and rsi_buy_ok and (ema20_m1 > ema50_m1)
                sell_ready = (bias == "DOWN") and touched_ema20 and bearish_confirm and rsi_sell_ok and (ema20_m1 < ema50_m1)

                levels_txt = f"EMA20:{ema20_m1:.2f} EMA50:{ema50_m1:.2f} | Struct EMA20:{ema20_struct:.2f} EMA50:{ema50_struct:.2f}"
                ind_txt = f"RSI(14): {rsi_now:.0f} | Doji:{dojis}/10 | avgR:{avg_rng:.2f}"

                if buy_ready:
                    self.market_debug[symbol] = {
                        "time": time.time(),
                        "bias": bias,
                        "setup": "EMA Pullback",
                        "signal": "BUY (RISE)",
                        "levels": levels_txt,
                        "ind": ind_txt,
                        "waiting": "✅ Trend UP + pullback to EMA20 + bullish confirm"
                    }
                    self.last_signal_candle_t[symbol] = confirm_t
                    await self.execute_trade("CALL", symbol, f"EMA20/50 BUY | {levels_txt} | RSI {rsi_now:.0f}")

                elif sell_ready:
                    self.market_debug[symbol] = {
                        "time": time.time(),
                        "bias": bias,
                        "setup": "EMA Pullback",
                        "signal": "SELL (FALL)",
                        "levels": levels_txt,
                        "ind": ind_txt,
                        "waiting": "✅ Trend DOWN + pullback to EMA20 + bearish confirm"
                    }
                    self.last_signal_candle_t[symbol] = confirm_t
                    await self.execute_trade("PUT", symbol, f"EMA20/50 SELL | {levels_txt} | RSI {rsi_now:.0f}")

                else:
                    why = []
                    if bias == "UP":
                        why.append("Need UP pullback to EMA20 + bullish confirm")
                    elif bias == "DOWN":
                        why.append("Need DOWN pullback to EMA20 + bearish confirm")
                    else:
                        why.append("No trend bias")

                    if not touched_ema20:
                        why.append("No EMA20 touch")
                    if USE_RSI_FILTER:
                        why.append("RSI confirm ON (>50 buy / <50 sell)")

                    self.market_debug[symbol] = {
                        "time": time.time(),
                        "bias": bias,
                        "setup": "-",
                        "signal": "-",
                        "levels": levels_txt,
                        "ind": ind_txt,
                        "waiting": " | ".join(why)
                    }
                    self.last_signal_candle_t[symbol] = confirm_t

            except asyncio.CancelledError:
                break
            except Exception as e:
                msg = str(e)
                if "rate limit" in msg.lower() and "ticks_history" in msg.lower():
                    self.market_debug[symbol] = {"time": time.time(), "bias": "-", "setup": "-", "signal": "-", "levels": "", "ind": "", "waiting": "⚠️ Rate limit hit for ticks_history → backing off 20s"}
                else:
                    logger.error(f"Scan error {symbol}: {e}")
                    self.market_debug[symbol] = {"time": time.time(), "bias": "-", "setup": "-", "signal": "-", "levels": "", "ind": "", "waiting": f"⚠️ Error: {msg[:120]}"}

            await asyncio.sleep(2)

    async def execute_trade(self, side: str, symbol: str, reason: str):
        if not self.api or self.active_trade_info:
            return

        async with self.trade_lock:
            ok, _g = self.gate(symbol)
            if not ok:
                return

            try:
                duration = int(EXPIRY_MIN.get(symbol, 3))
                stake = self.current_stake

                prop = await self.api.proposal({
                    "proposal": 1,
                    "amount": stake,
                    "basis": "stake",
                    "contract_type": side,
                    "currency": "USD",
                    "duration": duration,
                    "duration_unit": "m",
                    "symbol": symbol
                })
                buy = await self.api.buy({"buy": prop["proposal"]["id"], "price": float(prop["proposal"]["ask_price"])})

                self.active_trade_info = int(buy["buy"]["contract_id"])
                self.active_market = symbol
                self.trade_start_time = time.time()

                self.trades_today_total += 1
                self.trades_today_by_symbol[symbol] += 1

                self.global_cooldown_until = time.time() + COOLDOWN_AFTER_TRADE_SEC
                self.symbol_cooldown_until[symbol] = time.time() + COOLDOWN_PER_SYMBOL_SEC

                msg = (
                    f"🚀 {side} OPENED (${stake:.2f})\n"
                    f"🛒 Market: {symbol.replace('_',' ')}\n"
                    f"⏱ Expiry: {duration}m\n"
                    f"📌 Strategy: EMA20/EMA50 Pullback + RSI Confirm\n"
                    f"🧠 {reason}"
                )
                await self.app.bot.send_message(TELEGRAM_CHAT_ID, msg)

                asyncio.create_task(self.check_result(self.active_trade_info, symbol))

            except Exception as e:
                logger.error(f"Trade error: {e}")

    async def check_result(self, cid: int, symbol: str):
        await asyncio.sleep(EXPIRY_MIN.get(symbol, 3) * 60 + 5)

        try:
            res = await self.api.proposal_open_contract({"proposal_open_contract": 1, "contract_id": cid})
            profit = float(res["proposal_open_contract"].get("profit", 0.0))

            self.total_profit_today += profit

            if profit > 0:
                self.wins_today_total += 1
                self.consecutive_losses = 0
                self.current_stake = BASE_STAKE
            else:
                self.losses_today_total += 1
                self.losses_today_by_symbol[symbol] += 1
                self.consecutive_losses += 1

                if self.losses_today_by_symbol[symbol] >= STOP_SYMBOL_AFTER_LOSSES:
                    self.disabled_symbol_today[symbol] = True

                self.current_stake *= 2

            await self.fetch_balance()
            await self.app.bot.send_message(
                TELEGRAM_CHAT_ID,
                f"🏁 FINISH: {'WIN' if profit > 0 else 'LOSS'} ({profit:+.2f})\n"
                f"📊 Today: ✅{self.wins_today_total} / ❌{self.losses_today_total} | Streak ❌{self.consecutive_losses}/{MAX_CONSECUTIVE_LOSSES}\n"
                f"💰 Balance: {self.balance}"
            )

        finally:
            self.active_trade_info = None
            self.active_market = None


# ========================= UI =========================
bot_logic = DerivBot()

def main_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("▶️ START", callback_data="START_SCAN"),
         InlineKeyboardButton("⏹️ STOP", callback_data="STOP_SCAN")],
        [InlineKeyboardButton("📊 STATUS", callback_data="STATUS"),
         InlineKeyboardButton("🔄 REFRESH", callback_data="STATUS")],
        [InlineKeyboardButton("🧪 TEST BUY", callback_data="TEST_BUY")],
        [InlineKeyboardButton("🧪 DEMO", callback_data="SET_DEMO"),
         InlineKeyboardButton("💰 LIVE", callback_data="SET_REAL")]
    ])

async def btn_handler(u: Update, c: ContextTypes.DEFAULT_TYPE):
    q = u.callback_query
    await q.answer()

    if q.data == "SET_DEMO":
        bot_logic.active_token, bot_logic.account_type = DEMO_TOKEN, "DEMO"
        ok = await bot_logic.connect()
        await q.edit_message_text("✅ Connected to DEMO" if ok else "❌ DEMO Failed", reply_markup=main_keyboard())

    elif q.data == "SET_REAL":
        bot_logic.active_token, bot_logic.account_type = REAL_TOKEN, "LIVE"
        ok = await bot_logic.connect()
        await q.edit_message_text("⚠️ LIVE CONNECTED" if ok else "❌ LIVE Failed", reply_markup=main_keyboard())

    elif q.data == "START_SCAN":
        if not bot_logic.api:
            await q.edit_message_text("❌ Connect first.", reply_markup=main_keyboard())
            return
        bot_logic.is_scanning = True
        bot_logic.scanner_task = asyncio.create_task(bot_logic.background_scanner())
        await q.edit_message_text("🔍 SCANNER ACTIVE\n📌 Strategy: EMA20/EMA50 Pullback + RSI Confirm", reply_markup=main_keyboard())

    elif q.data == "STOP_SCAN":
        bot_logic.is_scanning = False
        await q.edit_message_text("⏹️ Scanner stopped.", reply_markup=main_keyboard())

    elif q.data == "TEST_BUY":
        await bot_logic.execute_trade("CALL", "R_10", "Manual test trade")

    elif q.data == "STATUS":
        await bot_logic.fetch_balance()
        now_time = datetime.now(ZoneInfo("Africa/Lagos")).strftime("%Y-%m-%d %H:%M:%S")
        _ok, gate = bot_logic.gate("R_10")

        pl_line = "P/L Today: (start balance not set yet)"
        if bot_logic.start_balance_value is not None:
            try:
                cur_val = float(bot_logic.balance.split()[0])
                pl = cur_val - float(bot_logic.start_balance_value)
                pl_line = f"P/L Today: {pl:+.2f} (Start: {bot_logic.start_balance_text} → Now: {bot_logic.balance})"
            except:
                pl_line = f"P/L Today: (calc error) | Start: {bot_logic.start_balance_text} | Now: {bot_logic.balance}"

        trade_status = "No Active Trade"
        if bot_logic.active_trade_info and bot_logic.api:
            try:
                res = await bot_logic.api.proposal_open_contract({"proposal_open_contract": 1, "contract_id": bot_logic.active_trade_info})
                pnl = float(res["proposal_open_contract"].get("profit", 0.0))
                rem = int(max(0, (EXPIRY_MIN.get(bot_logic.active_market, 3) * 60) - (time.time() - bot_logic.trade_start_time)))
                mkt = (bot_logic.active_market or "—").replace("_", " ")
                trade_status = f"🚀 Active Trade ({mkt})\n📈 Live PnL: {pnl:+.2f}\n⏳ Left: {rem}s"
            except:
                trade_status = "🚀 Active Trade: Syncing..."

        status_msg = (
            f"🕒 Time (WAT): {now_time}\n"
            f"🤖 Bot: {'ACTIVE' if bot_logic.is_scanning else 'OFFLINE'} ({bot_logic.account_type})\n"
            f"📌 Strategy: EMA20/EMA50 Pullback + RSI Confirm\n"
            f"🚦 Gate: {gate}\n"
            f"📡 Markets: {', '.join(MARKETS).replace('_',' ')}\n"
            f"━━━━━━━━━━━━━━━\n{trade_status}\n━━━━━━━━━━━━━━━\n"
            f"{pl_line}\n"
            f"📊 Today: ✅ Wins {bot_logic.wins_today_total} | ❌ Losses {bot_logic.losses_today_total} | Total {bot_logic.trades_today_total}/{MAX_TRADES_PER_DAY_TOTAL}\n"
            f"📉 Loss Streak: {bot_logic.consecutive_losses}/{MAX_CONSECUTIVE_LOSSES}\n"
            f"🧪 Next Stake: ${bot_logic.current_stake:.2f}\n"
            f"💰 Balance: {bot_logic.balance}"
        )

        debug_lines = []
        for sym in MARKETS:
            d = bot_logic.market_debug.get(sym)
            if not d:
                debug_lines.append(f"{sym.replace('_',' ')} → ⏳ No scan data yet")
            else:
                debug_lines.append(fmt_scan(sym, d))

        status_msg += "\n\n📌 LIVE SCAN (Clean)\n\n" + "\n\n".join(debug_lines)
        await q.edit_message_text(status_msg, reply_markup=main_keyboard())

async def start_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    await u.message.reply_text(
        "💎 Deriv Bot\n"
        "📌 Strategy: EMA20/EMA50 Pullback + RSI Confirm\n"
        "🕯 Entry: M1 (confirm candle) | Trend: M5/M15 (EMA bias) | Expiry: 3 minutes\n",
        reply_markup=main_keyboard()
    )

if __name__ == "__main__":
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    bot_logic.app = app
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CallbackQueryHandler(btn_handler))
    app.run_polling(drop_pending_updates=True)
