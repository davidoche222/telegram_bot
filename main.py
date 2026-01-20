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
MAX_TRADES_PER_DAY_TOTAL = 40
MAX_TRADES_PER_MARKET_PER_DAY = 3
STOP_DAY_AFTER_TOTAL_LOSSES = 10**9
STOP_SYMBOL_AFTER_LOSSES = 10**9
MAX_CONSECUTIVE_LOSSES = 7

COOLDOWN_AFTER_TRADE_SEC = 120
COOLDOWN_PER_SYMBOL_SEC = 120
BASE_STAKE = 1.00

# ========================= STRATEGY SETTINGS =========================
ENTRY_TF_SEC = 60
STRUCT_TF_SEC = {"R_10": 300, "R_25": 300, "R_50": 900, "R_75": 900, "R_100": 900}
EXPIRY_MIN = {"R_10": 3, "R_25": 3, "R_50": 3, "R_75": 3, "R_100": 3}

RSI_PERIOD = 14

DOJI_BODY_PCT = 0.25
CHOP_DOJI_COUNT = 6
CHOP_LOOKBACK = 10
CHOP_PAUSE_SEC = 600

SPIKE_MULTIPLIER = 2.0
AVG_RANGE_LOOKBACK = 20

# ========================= EMA & ATR SETTINGS =========================
EMA_FAST = 20
EMA_SLOW = 50

ATR_PERIOD = 14
ATR_TOUCH_MULT = 0.25  # EMA20 touch tolerance in ATR units
EMA_MIN_SEPARATION_PCT = 0.001
RSI_BUY_MIN = 52
RSI_SELL_MAX = 48

# ✅ NEW: ATR volatility filter (makes signals stronger)
# Use ATR / AvgRange ratio so it adapts per market.
ATR_RATIO_MIN = 0.60   # below this = too quiet / weak movement
ATR_RATIO_MAX = 1.80   # above this = too wild / messy movement (spike filter still catches extremes)

M1_FETCH_MIN_INTERVAL_SEC = 15

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

def calculate_atr(candles, period=14):
    if len(candles) < period + 1:
        return 0.0
    tr_list = []
    for i in range(1, len(candles)):
        h, l, pc = candles[i]["h"], candles[i]["l"], candles[i-1]["c"]
        tr = max(h - l, abs(h - pc), abs(l - pc))
        tr_list.append(tr)
    return float(np.mean(tr_list[-period:]))

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

def fmt_scan(sym, d):
    age = int(time.time() - d.get("time", time.time()))
    status = d.get("status", "⏳ WAIT")
    bias = d.get("bias", "?")
    reason = d.get("reason", d.get("waiting", "Waiting for setup"))
    levels = d.get("levels", "")
    ind = d.get("ind", "")

    lines = []
    lines.append(f"• {sym.replace('_',' ')} ({age}s)  |  {status}  |  Trend: {bias}")
    lines.append(f"  - {reason}")
    if levels:
        lines.append(f"  - Levels: {levels}")
    if ind:
        lines.append(f"  - {ind}")
    return "\n".join(lines)

def fmt_secs_left(ts):
    if not ts:
        return 0
    return max(0, int(ts - time.time()))

# ========================= BOT CORE =========================
class DerivBot:
    def __init__(self):
        self.api, self.app = None, None
        self.active_token, self.account_type = None, "None"

        self.is_scanning, self.scanner_task = False, None
        self.market_tasks = {}

        self.active_trade_info, self.active_market = None, None
        self.trade_start_time = 0.0

        self.global_cooldown_until = 0.0
        self.symbol_cooldown_until = {m: 0.0 for m in MARKETS}
        self.symbol_chop_until = {m: 0.0 for m in MARKETS}

        self.day_key = None
        self.trades_today_total = 0
        self.losses_today_total, self.wins_today_total = 0, 0
        self.trades_today_by_symbol = {m: 0 for m in MARKETS}

        self.balance, self.start_balance_value, self.start_balance_text = "0.00", None, None
        self.current_stake, self.consecutive_losses = BASE_STAKE, 0
        self.trade_lock = asyncio.Lock()
        self.market_debug = {}

        self.cache_m1 = {m: [] for m in MARKETS}
        self.cache_struct = {m: [] for m in MARKETS}
        self.last_fetch_m1 = {m: 0.0 for m in MARKETS}
        self.last_fetch_struct = {m: 0.0 for m in MARKETS}
        self.rate_backoff_until = {m: 0.0 for m in MARKETS}

        # ✅ anti-overtrade: once per CLOSED candle
        self.last_signal_candle_t = {m: 0 for m in MARKETS}

    def reset_day_if_needed(self):
        now = datetime.now(ZoneInfo("Africa/Lagos"))
        key = now.strftime("%Y-%m-%d")
        if self.day_key != key:
            self.day_key = key
            self.trades_today_total = 0
            self.losses_today_total, self.wins_today_total = 0, 0
            self.trades_today_by_symbol = {m: 0 for m in MARKETS}
            self.current_stake = BASE_STAKE
            self.consecutive_losses = 0
            self.start_balance_value, self.start_balance_text = None, None

            self.global_cooldown_until = 0.0
            self.symbol_cooldown_until = {m: 0.0 for m in MARKETS}
            self.symbol_chop_until = {m: 0.0 for m in MARKETS}
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

        if self.trades_today_total >= MAX_TRADES_PER_DAY_TOTAL:
            return False, "Daily limit reached"

        if self.trades_today_by_symbol.get(symbol, 0) >= MAX_TRADES_PER_MARKET_PER_DAY:
            return False, "Symbol cap reached"

        now = time.time()
        if now < self.global_cooldown_until:
            return False, f"Global cooldown {int(self.global_cooldown_until - now)}s"
        if now < self.symbol_cooldown_until.get(symbol, 0.0):
            return False, f"Symbol cooldown {int(self.symbol_cooldown_until[symbol] - now)}s"
        if now < self.symbol_chop_until.get(symbol, 0.0):
            return False, f"Chop filter {int(self.symbol_chop_until[symbol] - now)}s"

        if self.active_trade_info:
            return False, "Active trade in progress"
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
                    self.active_market = None
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
        return build_candles_from_deriv(res.get("candles", []))

    async def get_m1(self, symbol: str):
        now = time.time()
        if now < self.rate_backoff_until[symbol]:
            return self.cache_m1[symbol]
        if (now - self.last_fetch_m1[symbol]) < M1_FETCH_MIN_INTERVAL_SEC and self.cache_m1[symbol]:
            return self.cache_m1[symbol]
        try:
            candles = await self.fetch_candles_raw(symbol, ENTRY_TF_SEC, 220)
            self.cache_m1[symbol], self.last_fetch_m1[symbol] = candles, now
            return candles
        except Exception as e:
            if "rate limit" in str(e).lower():
                self.rate_backoff_until[symbol] = now + 20
            raise

    async def get_struct(self, symbol: str):
        now = time.time()
        struct_tf = STRUCT_TF_SEC[symbol]
        if (now - self.last_fetch_struct[symbol]) < struct_tf and self.cache_struct[symbol]:
            return self.cache_struct[symbol]
        candles = await self.fetch_candles_raw(symbol, struct_tf, 220)
        self.cache_struct[symbol], self.last_fetch_struct[symbol] = candles, now
        return candles

    async def scan_symbol(self, symbol: str):
        while self.is_scanning:
            try:
                ok, gate_reason = self.gate(symbol)
                if not ok:
                    self.market_debug[symbol] = {
                        "time": time.time(),
                        "status": "🚦 GATED",
                        "bias": "-",
                        "reason": gate_reason,
                        "levels": "",
                        "ind": ""
                    }
                    await asyncio.sleep(2)
                    continue

                struct, m1 = await self.get_struct(symbol), await self.get_m1(symbol)
                if len(struct) < 50 or len(m1) < 60:
                    self.market_debug[symbol] = {
                        "time": time.time(),
                        "status": "⏳ SYNC",
                        "bias": "…",
                        "reason": "Syncing candles...",
                        "levels": "",
                        "ind": ""
                    }
                    await asyncio.sleep(3)
                    continue

                c_prev, c_confirm = m1[-3], m1[-2]
                confirm_t = int(c_confirm.get("t", 0) or 0)

                # anti-overtrade: once per closed candle
                if confirm_t and self.last_signal_candle_t[symbol] == confirm_t:
                    self.market_debug[symbol] = {
                        "time": time.time(),
                        "status": "⏳ WAIT",
                        "bias": self.market_debug.get(symbol, {}).get("bias", "-"),
                        "reason": "Waiting: next M1 candle close",
                        "levels": "",
                        "ind": ""
                    }
                    await asyncio.sleep(2)
                    continue

                # Chop filter
                last10 = m1[-(CHOP_LOOKBACK + 1):-1]
                dojis = sum(1 for x in last10 if is_doji(x))
                if dojis >= CHOP_DOJI_COUNT:
                    self.symbol_chop_until[symbol] = time.time() + CHOP_PAUSE_SEC
                    self.market_debug[symbol] = {
                        "time": time.time(),
                        "status": "🟨 CHOP",
                        "bias": "-",
                        "reason": f"Chop detected → pausing {CHOP_PAUSE_SEC//60}m",
                        "levels": "",
                        "ind": f"Doji:{dojis}/{CHOP_LOOKBACK}"
                    }
                    self.last_signal_candle_t[symbol] = confirm_t
                    await asyncio.sleep(2)
                    continue

                # Indicators
                struct_closes = [x["c"] for x in struct]
                ema20_s, ema50_s = calculate_ema(struct_closes, EMA_FAST), calculate_ema(struct_closes, EMA_SLOW)
                if len(ema20_s) == 0 or len(ema50_s) == 0:
                    self.last_signal_candle_t[symbol] = confirm_t
                    await asyncio.sleep(2)
                    continue

                ema20_struct, ema50_struct = float(ema20_s[-1]), float(ema50_s[-1])
                bias = "UP" if ema20_struct > ema50_struct else "DOWN" if ema20_struct < ema50_struct else "RANGE"

                closes_m1 = [x["c"] for x in m1]
                ema20_m1_arr, ema50_m1_arr = calculate_ema(closes_m1, EMA_FAST), calculate_ema(closes_m1, EMA_SLOW)
                if len(ema20_m1_arr) == 0 or len(ema50_m1_arr) == 0:
                    self.last_signal_candle_t[symbol] = confirm_t
                    await asyncio.sleep(2)
                    continue

                ema20_m1, ema50_m1 = float(ema20_m1_arr[-2]), float(ema50_m1_arr[-2])

                rsi_arr = calculate_rsi(closes_m1, RSI_PERIOD)
                rsi_now = float(rsi_arr[-1]) if len(rsi_arr) > 0 else 50.0
                atr_now = calculate_atr(m1, ATR_PERIOD)

                avg_rng = avg_range(m1[-(AVG_RANGE_LOOKBACK + 1):-1], AVG_RANGE_LOOKBACK)
                sig_rng = candle_range(c_confirm)

                # Spike filter (hard block)
                if sig_rng > SPIKE_MULTIPLIER * avg_rng:
                    self.market_debug[symbol] = {
                        "time": time.time(),
                        "status": "🟧 SPIKE",
                        "bias": bias,
                        "reason": "Spike candle detected → skip",
                        "levels": f"avgR:{avg_rng:.4f} sigR:{sig_rng:.4f}",
                        "ind": f"RSI:{rsi_now:.0f} | ATR:{atr_now:.4f} | Doji:{dojis}/{CHOP_LOOKBACK}"
                    }
                    self.last_signal_candle_t[symbol] = confirm_t
                    await asyncio.sleep(2)
                    continue

                # ✅ NEW: ATR volatility filter (stronger entries)
                atr_ratio = (atr_now / (avg_rng + 1e-9)) if avg_rng > 0 else 0.0
                if atr_ratio < ATR_RATIO_MIN:
                    self.market_debug[symbol] = {
                        "time": time.time(),
                        "status": "🚫 FILTER",
                        "bias": bias,
                        "reason": f"ATR too low (quiet market) → skip | ratio={atr_ratio:.2f}",
                        "levels": f"avgR:{avg_rng:.4f}",
                        "ind": f"RSI:{rsi_now:.0f} | ATR:{atr_now:.4f} | Doji:{dojis}/{CHOP_LOOKBACK}"
                    }
                    self.last_signal_candle_t[symbol] = confirm_t
                    await asyncio.sleep(2)
                    continue

                if atr_ratio > ATR_RATIO_MAX:
                    self.market_debug[symbol] = {
                        "time": time.time(),
                        "status": "🚫 FILTER",
                        "bias": bias,
                        "reason": f"ATR too high (too volatile) → skip | ratio={atr_ratio:.2f}",
                        "levels": f"avgR:{avg_rng:.4f}",
                        "ind": f"RSI:{rsi_now:.0f} | ATR:{atr_now:.4f} | Doji:{dojis}/{CHOP_LOOKBACK}"
                    }
                    self.last_signal_candle_t[symbol] = confirm_t
                    await asyncio.sleep(2)
                    continue

                # Core logic (unchanged)
                tol = atr_now * ATR_TOUCH_MULT
                touched_ema20 = (c_confirm["l"] <= (ema20_m1 + tol)) and (c_confirm["h"] >= (ema20_m1 - tol))
                bullish_confirm = strong_bull_close(c_confirm) or bull_engulf(c_prev, c_confirm)
                bearish_confirm = strong_bear_close(c_confirm) or bear_engulf(c_prev, c_confirm)

                rsi_buy_ok = (rsi_now > RSI_BUY_MIN) and (rsi_now < 70)
                rsi_sell_ok = (rsi_now < RSI_SELL_MAX) and (rsi_now > 30)
                trend_sep = abs(ema20_m1 - ema50_m1) / max(1e-9, abs(ema50_m1))
                trend_sep_ok = trend_sep > EMA_MIN_SEPARATION_PCT

                levels_txt = f"EMA20:{ema20_m1:.2f} EMA50:{ema50_m1:.2f} | tol(ATR):{tol:.4f} | sep:{trend_sep:.4f}"
                ind_txt = f"RSI:{rsi_now:.0f} | ATR:{atr_now:.4f} (ratio:{atr_ratio:.2f}) | Doji:{dojis}/{CHOP_LOOKBACK}"

                wait_reason = "Waiting for setup"
                if bias == "RANGE":
                    wait_reason = "Structural trend is neutral"
                elif not trend_sep_ok:
                    wait_reason = "EMA20/EMA50 too close (flat)"
                elif not touched_ema20:
                    wait_reason = "Price has not touched EMA20 zone yet"
                elif bias == "UP" and not rsi_buy_ok:
                    wait_reason = f"RSI {rsi_now:.0f} too weak for BUY"
                elif bias == "DOWN" and not rsi_sell_ok:
                    wait_reason = f"RSI {rsi_now:.0f} too weak for SELL"
                elif bias == "UP" and not bullish_confirm:
                    wait_reason = "Need bullish confirm (strong close / engulf)"
                elif bias == "DOWN" and not bearish_confirm:
                    wait_reason = "Need bearish confirm (strong close / engulf)"

                buy_ready = (bias == "UP") and touched_ema20 and bullish_confirm and rsi_buy_ok and trend_sep_ok
                sell_ready = (bias == "DOWN") and touched_ema20 and bearish_confirm and rsi_sell_ok and trend_sep_ok

                if buy_ready:
                    self.market_debug[symbol] = {
                        "time": time.time(),
                        "status": "✅ SIGNAL",
                        "bias": bias,
                        "reason": "Trend UP + EMA20 pullback + bullish confirm + RSI OK",
                        "levels": levels_txt,
                        "ind": ind_txt
                    }
                    self.last_signal_candle_t[symbol] = confirm_t
                    await self.execute_trade("CALL", symbol, f"ATR BUY | RSI {rsi_now:.0f}")

                elif sell_ready:
                    self.market_debug[symbol] = {
                        "time": time.time(),
                        "status": "✅ SIGNAL",
                        "bias": bias,
                        "reason": "Trend DOWN + EMA20 pullback + bearish confirm + RSI OK",
                        "levels": levels_txt,
                        "ind": ind_txt
                    }
                    self.last_signal_candle_t[symbol] = confirm_t
                    await self.execute_trade("PUT", symbol, f"ATR SELL | RSI {rsi_now:.0f}")

                else:
                    self.market_debug[symbol] = {
                        "time": time.time(),
                        "status": "⏳ WAIT",
                        "bias": bias,
                        "reason": wait_reason,
                        "levels": levels_txt,
                        "ind": ind_txt
                    }
                    self.last_signal_candle_t[symbol] = confirm_t

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scan error {symbol}: {e}")
                self.market_debug[symbol] = {
                    "time": time.time(),
                    "status": "⚠️ ERROR",
                    "bias": "-",
                    "reason": str(e)[:160],
                    "levels": "",
                    "ind": ""
                }
            await asyncio.sleep(2)

    async def execute_trade(self, side: str, symbol: str, reason: str):
        if not self.api or self.active_trade_info:
            return

        async with self.trade_lock:
            ok, _ = self.gate(symbol)
            if not ok:
                return
            try:
                duration = int(EXPIRY_MIN.get(symbol, 3))
                stake = float(self.current_stake)

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

                # cooldowns
                self.global_cooldown_until = time.time() + COOLDOWN_AFTER_TRADE_SEC
                self.symbol_cooldown_until[symbol] = time.time() + COOLDOWN_PER_SYMBOL_SEC

                await self.app.bot.send_message(
                    TELEGRAM_CHAT_ID,
                    f"🚀 {side} OPENED (${stake:.2f})\n"
                    f"🛒 {symbol.replace('_',' ')}\n"
                    f"⏱ Expiry: {duration}m\n"
                    f"🧠 {reason}"
                )
                asyncio.create_task(self.check_result(self.active_trade_info, symbol))

            except Exception as e:
                logger.error(f"Trade error: {e}")

    async def check_result(self, cid: int, symbol: str):
        await asyncio.sleep(EXPIRY_MIN.get(symbol, 3) * 60 + 5)
        try:
            res = await self.api.proposal_open_contract({"proposal_open_contract": 1, "contract_id": cid})
            profit = float(res["proposal_open_contract"].get("profit", 0.0))

            if profit > 0:
                self.wins_today_total += 1
                self.consecutive_losses = 0
                self.current_stake = BASE_STAKE
            else:
                self.losses_today_total += 1
                self.consecutive_losses += 1
                self.current_stake *= 2

            await self.fetch_balance()
            await self.app.bot.send_message(
                TELEGRAM_CHAT_ID,
                f"🏁 {'WIN' if profit > 0 else 'LOSS'} ({profit:+.2f})\n"
                f"📊 Today: ✅{self.wins_today_total} / ❌{self.losses_today_total} | Streak ❌{self.consecutive_losses}/{MAX_CONSECUTIVE_LOSSES}\n"
                f"🧪 Next Stake: ${self.current_stake:.2f}\n"
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
        await q.edit_message_text("🔍 SCANNING ACTIVE\n📌 Strategy: EMA20/50 Pullback + RSI + ATR Filters", reply_markup=main_keyboard())

    elif q.data == "STOP_SCAN":
        bot_logic.is_scanning = False
        await q.edit_message_text("⏹️ STOPPED", reply_markup=main_keyboard())

    elif q.data == "TEST_BUY":
        await bot_logic.execute_trade("CALL", "R_10", "Manual test trade")

    elif q.data == "STATUS":
        await bot_logic.fetch_balance()

        now_time = datetime.now(ZoneInfo("Africa/Lagos")).strftime("%Y-%m-%d %H:%M:%S")

        # P/L today
        pl_line = "P/L Today: (start balance not set yet)"
        if bot_logic.start_balance_value is not None:
            try:
                cur_val = float(bot_logic.balance.split()[0])
                pl = cur_val - float(bot_logic.start_balance_value)
                pl_line = f"P/L Today: {pl:+.2f} (Start: {bot_logic.start_balance_text} → Now: {bot_logic.balance})"
            except:
                pl_line = f"P/L Today: (calc error) | Start: {bot_logic.start_balance_text} | Now: {bot_logic.balance}"

        # cooldown info
        g_cd = fmt_secs_left(bot_logic.global_cooldown_until)
        cd_line = f"Cooldowns: Global {g_cd}s"
        # show per-symbol cooldown short summary
        sym_cd_bits = []
        for s in MARKETS:
            scd = fmt_secs_left(bot_logic.symbol_cooldown_until.get(s, 0.0))
            if scd > 0:
                sym_cd_bits.append(f"{s.replace('_',' ')} {scd}s")
        if sym_cd_bits:
            cd_line += " | " + ", ".join(sym_cd_bits)

        # active trade info
        trade_status = "No Active Trade"
        if bot_logic.active_trade_info and bot_logic.api:
            try:
                res = await bot_logic.api.proposal_open_contract({
                    "proposal_open_contract": 1,
                    "contract_id": bot_logic.active_trade_info
                })
                pnl = float(res["proposal_open_contract"].get("profit", 0.0))
                rem = int(max(0, (EXPIRY_MIN.get(bot_logic.active_market, 3) * 60) - (time.time() - bot_logic.trade_start_time)))
                mkt = (bot_logic.active_market or "—").replace("_", " ")
                trade_status = f"🚀 Active Trade: {mkt} | PnL: {pnl:+.2f} | Left: {rem}s"
            except:
                trade_status = "🚀 Active Trade: Syncing..."

        header = (
            f"🕒 Time (WAT): {now_time}\n"
            f"🤖 Bot: {'ON ✅' if bot_logic.is_scanning else 'OFF ⛔'} ({bot_logic.account_type})\n"
            f"📌 Strategy: EMA20/50 Pullback + RSI + ATR Touch + ATR Volatility Filter\n"
            f"{trade_status}\n"
            f"━━━━━━━━━━━━━━━\n"
            f"{pl_line}\n"
            f"📊 Today: ✅{bot_logic.wins_today_total} ❌{bot_logic.losses_today_total} | Trades {bot_logic.trades_today_total}/{MAX_TRADES_PER_DAY_TOTAL}\n"
            f"📉 Loss Streak: {bot_logic.consecutive_losses}/{MAX_CONSECUTIVE_LOSSES}\n"
            f"🧪 Next Stake: ${bot_logic.current_stake:.2f}\n"
            f"💰 Balance: {bot_logic.balance}\n"
            f"{cd_line}\n"
        )

        debug_lines = []
        for sym in MARKETS:
            d = bot_logic.market_debug.get(sym)
            if not d:
                debug_lines.append(f"• {sym.replace('_',' ')} → ⏳ No scan data yet (press START)")
            else:
                debug_lines.append(fmt_scan(sym, d))

        status_msg = header + "\n━━━━━━━━━━━━━━━\n📌 LIVE SCAN\n\n" + "\n\n".join(debug_lines)
        await q.edit_message_text(status_msg, reply_markup=main_keyboard())

async def start_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    await u.message.reply_text(
        "💎 Deriv Bot\n"
        "📌 Strategy: EMA20/50 Pullback + RSI + ATR Filters\n"
        "🕯 Entry: M1 | Trend: M5/M15 | Expiry: 3 minutes\n",
        reply_markup=main_keyboard()
    )

if __name__ == "__main__":
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    bot_logic.app = app
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CallbackQueryHandler(btn_handler))
    app.run_polling(drop_pending_updates=True)
