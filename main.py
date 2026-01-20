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

MARKETS = ["R_10", "R_25", "R_50"]

COOLDOWN_SEC = 120
MAX_TRADES_PER_DAY = 40
MAX_CONSEC_LOSSES = 10

BASE_STAKE = 1.00  # Initial trade amount

TELEGRAM_TOKEN = "8253450930:AAHUhPk9TML-8kZlA9UaHZZvTUGdurN9MVY"
TELEGRAM_CHAT_ID = "7634818949"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ========================= STRATEGY SETTINGS =========================
# Strategy 1: EMA 9/21 + RSI14 (your current)
S1_EMA_FAST = 9
S1_EMA_SLOW = 21
S1_RSI_PERIOD = 14
S1_EXPIRY_MIN = 5  # keep as your current code

# Strategy 2: EMA-RSI Bounce (EMA10/EMA20 + RSI7) from your spec
S2_EMA_FAST = 10
S2_EMA_SLOW = 20
S2_RSI_PERIOD = 7
S2_RSI_BUY_MAX = 35
S2_RSI_SELL_MIN = 65
S2_RSI_RESET_LOW = 45   # RSI reset zone (developer note)
S2_RSI_RESET_HIGH = 55
S2_EXPIRY_MIN = 1       # per your spec (60 seconds)

# Flat filter for Strategy 2 (prevents sideways): EMA gap must exceed this percent of price
S2_FLAT_MIN_PCT = 0.00035  # tweak if too strict/loose

# ========================= STRATEGY MATH =========================
def calculate_ema(data, period):
    if len(data) < period:
        return np.array([])
    values = np.array(data, dtype=float)
    ema = np.zeros_like(values)
    k = 2 / (period + 1)
    ema[0] = values[0]
    for i in range(1, len(values)):
        ema[i] = values[i] * k + ema[i - 1] * (1 - k)
    return ema

def calculate_rsi(data, period=14):
    data = np.array(data, dtype=float)
    if len(data) < period + 2:
        return np.array([])
    delta = np.diff(data)
    gain = delta.clip(min=0)
    loss = (-delta).clip(min=0)
    avg_gain = np.convolve(gain, np.ones(period), "valid") / period
    avg_loss = np.convolve(loss, np.ones(period), "valid") / period
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))

def build_m1_candles_from_ticks(times, prices):
    if not times or not prices:
        return []
    candles = []
    curr_t0 = times[0] - (times[0] % 60)
    o = h = l = c = float(prices[0])
    for t, p in zip(times, prices):
        t0 = t - (t % 60)
        p = float(p)
        if t0 != curr_t0:
            candles.append({"o": o, "h": h, "l": l, "c": c, "t0": curr_t0})
            curr_t0, o, h, l, c = t0, p, p, p, p
        else:
            h, l, c = max(h, p), min(l, p), p
    candles.append({"o": o, "h": h, "l": l, "c": c, "t0": curr_t0})
    return candles

def fmt_time_hhmmss(epoch):
    try:
        return datetime.fromtimestamp(epoch, ZoneInfo("Africa/Lagos")).strftime("%H:%M:%S")
    except:
        return "—"

def rsi_dir_arrow(rsi, rsi_prev):
    if rsi > rsi_prev:
        return "↑"
    if rsi < rsi_prev:
        return "↓"
    return "→"

# ========================= STRATEGY 1: EMA9/21 + RSI14 =========================
def s1_calculate_indicators(candles):
    c = np.array([x["c"] for x in candles], dtype=float)
    o = np.array([x["o"] for x in candles], dtype=float)

    ema_fast = calculate_ema(c, S1_EMA_FAST)
    ema_slow = calculate_ema(c, S1_EMA_SLOW)
    if len(ema_slow) < 2 or len(ema_fast) < 1:
        return None

    rsi_vals = calculate_rsi(c, S1_RSI_PERIOD)
    if len(rsi_vals) < 2:
        return None

    slope = float(ema_slow[-1] - ema_slow[-2])

    return {
        "ema_fast": float(ema_fast[-1]),
        "ema_slow": float(ema_slow[-1]),
        "ema_slope": slope,
        "rsi": float(rsi_vals[-1]),
        "rsi_prev": float(rsi_vals[-2]),
        "o": float(o[-1]),
        "c": float(c[-1]),
    }

def s1_analyze_block_reason(ind):
    ema9 = ind["ema_fast"]
    ema21 = ind["ema_slow"]
    slope = ind["ema_slope"]
    rsi = ind["rsi"]
    rsi_prev = ind["rsi_prev"]
    price = ind["c"]

    trend_up = (ema9 > ema21 and slope > 0)
    trend_down = (ema9 < ema21 and slope < 0)

    rsi_rising = rsi > rsi_prev
    rsi_falling = rsi < rsi_prev

    above_ema21 = price > ema21
    below_ema21 = price < ema21
    above_ema9 = price > ema9
    below_ema9 = price < ema9

    if trend_up:
        trend = "UP"
    elif trend_down:
        trend = "DOWN"
    else:
        trend = "RANGE"

    reasons = []
    if trend == "RANGE":
        reasons.append("Trend not clear (EMA9/EMA21 + slope not aligned)")

    if trend == "UP":
        if not above_ema21:
            reasons.append("Price not above EMA21")
        if not (45 <= rsi <= 60):
            reasons.append(f"RSI not in BUY window 45–60 (now {rsi:.0f})")
        if not rsi_rising:
            reasons.append("RSI not rising (needs RSI > previous)")
        if not above_ema9:
            reasons.append("Price not above EMA9")
        if not (35 <= rsi <= 65):
            reasons.append("RSI safety filter failed (needs 35–65)")

    if trend == "DOWN":
        if not below_ema21:
            reasons.append("Price not below EMA21")
        if not (40 <= rsi <= 55):
            reasons.append(f"RSI not in SELL window 40–55 (now {rsi:.0f})")
        if not rsi_falling:
            reasons.append("RSI not falling (needs RSI < previous)")
        if not below_ema9:
            reasons.append("Price not below EMA9")
        if not (35 <= rsi <= 65):
            reasons.append("RSI safety filter failed (needs 35–65)")

    if not reasons:
        reasons.append("All conditions met ✅ (signal ready)")
    return trend, reasons

# ========================= STRATEGY 2: EMA10/20 + RSI7 BOUNCE =========================
def s2_calculate_indicators(candles):
    c = np.array([x["c"] for x in candles], dtype=float)
    ema10 = calculate_ema(c, S2_EMA_FAST)
    ema20 = calculate_ema(c, S2_EMA_SLOW)
    if len(ema10) < 1 or len(ema20) < 1:
        return None
    rsi_vals = calculate_rsi(c, S2_RSI_PERIOD)
    if len(rsi_vals) < 2:
        return None
    return {
        "ema_fast": float(ema10[-1]),
        "ema_slow": float(ema20[-1]),
        "rsi": float(rsi_vals[-1]),
        "rsi_prev": float(rsi_vals[-2]),
    }

def s2_analyze_block_reason(ema10, ema20, rsi, rsi_prev, sig_candle):
    """
    sig_candle is the last CLOSED candle (we decide on it, then enter next candle).
    """
    trend_up = ema10 > ema20
    trend_down = ema10 < ema20
    price = float(sig_candle["c"])
    low = float(sig_candle["l"])
    high = float(sig_candle["h"])

    gap = abs(ema10 - ema20)
    flat_ok = gap >= (price * S2_FLAT_MIN_PCT)

    reasons = []
    trend = "RANGE"
    if trend_up:
        trend = "UP"
    elif trend_down:
        trend = "DOWN"

    if trend == "RANGE":
        reasons.append("Trend not clear (EMA10 equals EMA20)")

    if not flat_ok:
        reasons.append(f"Flat filter: EMA gap too small (gap {gap:.4f})")

    if trend == "UP":
        touch = low <= ema20
        close_confirm = price > ema20
        if not touch:
            reasons.append("Pullback missing: candle did not touch EMA20 (LOW must hit EMA20)")
        if not (rsi <= S2_RSI_BUY_MAX):
            reasons.append(f"RSI not oversold enough (needs ≤ {S2_RSI_BUY_MAX}, now {rsi:.0f})")
        if not close_confirm:
            reasons.append("Confirmation missing: candle did NOT close above EMA20")

    if trend == "DOWN":
        touch = high >= ema20
        close_confirm = price < ema20
        if not touch:
            reasons.append("Pullback missing: candle did not touch EMA20 (HIGH must hit EMA20)")
        if not (rsi >= S2_RSI_SELL_MIN):
            reasons.append(f"RSI not overbought enough (needs ≥ {S2_RSI_SELL_MIN}, now {rsi:.0f})")
        if not close_confirm:
            reasons.append("Confirmation missing: candle did NOT close below EMA20")

    if not reasons:
        reasons.append("All conditions met ✅ (bounce signal ready)")
    return trend, reasons

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
        self.current_stake = BASE_STAKE

        self.trade_lock = asyncio.Lock()

        # Strategy switch
        self.strategy = "S1"  # "S1" = EMA9/21 RSI14 | "S2" = EMA10/20 RSI7 Bounce

        # Per-market live scan details
        self.market_debug = {m: {} for m in MARKETS}
        self.last_closed_m1_time = {m: 0 for m in MARKETS}

        # Prevent repeat decisions on same closed candle (per market)
        self.last_decision_candle_t0 = {m: 0 for m in MARKETS}

        # Strategy 2: after taking a trade, wait until RSI resets near 50 before trading again (per market)
        self.s2_rsi_reset_lock = {m: False for m in MARKETS}

    def strategy_name(self):
        if self.strategy == "S2":
            return "EMA-RSI Bounce (EMA10/20 + RSI7)"
        return "EMA Cross (EMA9/21 + RSI14)"

    def current_expiry_min(self):
        return S2_EXPIRY_MIN if self.strategy == "S2" else S1_EXPIRY_MIN

    def toggle_strategy(self):
        self.strategy = "S2" if self.strategy == "S1" else "S1"

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
            self.balance = f"{float(bal['balance']['balance']):.2f} {bal['balance']['currency']}"
        except:
            pass

    def can_auto_trade(self) -> tuple[bool, str]:
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
                if self.active_trade_info and (time.time() - self.trade_start_time > (self.current_expiry_min() * 60 + 90)):
                    self.active_trade_info = None
                await asyncio.sleep(1)
        finally:
            for t in self.market_tasks.values():
                t.cancel()
            self.market_tasks.clear()

    async def scan_market(self, symbol: str):
        while self.is_scanning:
            try:
                if self.consecutive_losses >= MAX_CONSEC_LOSSES or self.trades_today >= MAX_TRADES_PER_DAY:
                    self.is_scanning = False
                    break

                data = await self.api.ticks_history({
                    "ticks_history": symbol,
                    "end": "latest",
                    "count": 1000,
                    "style": "ticks"
                })
                candles = build_m1_candles_from_ticks(data["history"]["times"], data["history"]["prices"])
                if len(candles) < 35:
                    await asyncio.sleep(3)
                    continue

                # last CLOSED candle (signal candle) = candles[-2]
                if len(candles) < 2:
                    await asyncio.sleep(3)
                    continue
                sig = candles[-2]
                sig_t0 = int(sig.get("t0", 0) or 0)
                self.last_closed_m1_time[symbol] = sig_t0

                ok, gate = self.can_auto_trade()

                # Update RSI reset lock for Strategy 2 (it unlocks when RSI returns near 50)
                if self.strategy == "S2":
                    ind2_now = s2_calculate_indicators(candles)
                    if ind2_now and self.s2_rsi_reset_lock[symbol]:
                        rsi_now = ind2_now["rsi"]
                        if S2_RSI_RESET_LOW <= rsi_now <= S2_RSI_RESET_HIGH:
                            self.s2_rsi_reset_lock[symbol] = False

                # Only evaluate once per new closed candle
                if sig_t0 and self.last_decision_candle_t0[symbol] == sig_t0:
                    # Still refresh debug so STATUS stays alive
                    self.market_debug[symbol] = self.market_debug.get(symbol, {}) or {
                        "time": time.time(),
                        "strategy": self.strategy_name(),
                        "trend": "—",
                        "gate": gate,
                        "reasons": ["Waiting for next M1 candle to close"],
                        "last_closed": sig_t0
                    }
                    await asyncio.sleep(2)
                    continue

                # Mark this candle as evaluated (prevents repeated “same thing” spam)
                self.last_decision_candle_t0[symbol] = sig_t0

                # ===================== STRATEGY 1 =====================
                if self.strategy == "S1":
                    ind = s1_calculate_indicators(candles)
                    if not ind:
                        await asyncio.sleep(2)
                        continue

                    trend, reasons = s1_analyze_block_reason(ind)

                    self.market_debug[symbol] = {
                        "time": time.time(),
                        "strategy": self.strategy_name(),
                        "trend": trend,
                        "gate": gate,
                        "price": ind["c"],
                        "ema_fast": ind["ema_fast"],
                        "ema_slow": ind["ema_slow"],
                        "slope": ind["ema_slope"],
                        "rsi": ind["rsi"],
                        "rsi_prev": ind["rsi_prev"],
                        "reasons": reasons,
                        "last_closed": sig_t0,
                    }

                    if not ok:
                        await asyncio.sleep(2)
                        continue

                    # --- Your original trading logic (same rules, but using generic names) ---
                    if ind["ema_fast"] > ind["ema_slow"] and ind["ema_slope"] > 0:
                        if ind["c"] > ind["ema_slow"]:
                            if 45 <= ind["rsi"] <= 60 and ind["rsi"] > ind["rsi_prev"]:
                                if ind["c"] > ind["ema_fast"]:
                                    if 35 <= ind["rsi"] <= 65:
                                        await self.execute_trade("CALL", symbol, "S1: EMA Cross + RSI Rising", source="AUTO")

                    elif ind["ema_fast"] < ind["ema_slow"] and ind["ema_slope"] < 0:
                        if ind["c"] < ind["ema_slow"]:
                            if 40 <= ind["rsi"] <= 55 and ind["rsi"] < ind["rsi_prev"]:
                                if ind["c"] < ind["ema_fast"]:
                                    if 35 <= ind["rsi"] <= 65:
                                        await self.execute_trade("PUT", symbol, "S1: EMA Cross + RSI Falling", source="AUTO")

                # ===================== STRATEGY 2 =====================
                else:
                    ind = s2_calculate_indicators(candles)
                    if not ind:
                        await asyncio.sleep(2)
                        continue

                    ema10 = ind["ema_fast"]
                    ema20 = ind["ema_slow"]
                    rsi = ind["rsi"]
                    rsi_prev = ind["rsi_prev"]

                    trend, reasons = s2_analyze_block_reason(ema10, ema20, rsi, rsi_prev, sig)

                    lock_txt = "LOCKED (wait RSI reset to ~50)" if self.s2_rsi_reset_lock[symbol] else "OK"
                    # add lock to reasons (readable)
                    reasons2 = reasons[:]
                    if self.s2_rsi_reset_lock[symbol]:
                        reasons2.insert(0, f"Consecutive filter: {lock_txt}")

                    self.market_debug[symbol] = {
                        "time": time.time(),
                        "strategy": self.strategy_name(),
                        "trend": trend,
                        "gate": gate,
                        "price": float(sig["c"]),
                        "ema_fast": ema10,
                        "ema_slow": ema20,
                        "slope": float(ema10 - ema20),  # just show gap for S2
                        "rsi": rsi,
                        "rsi_prev": rsi_prev,
                        "touch": None,
                        "confirm": None,
                        "reasons": reasons2,
                        "last_closed": sig_t0,
                        "lock": lock_txt,
                    }

                    if not ok:
                        await asyncio.sleep(2)
                        continue

                    # Block if RSI reset lock is active (per your spec)
                    if self.s2_rsi_reset_lock[symbol]:
                        await asyncio.sleep(2)
                        continue

                    # Flat filter
                    gap = abs(ema10 - ema20)
                    flat_ok = gap >= (float(sig["c"]) * S2_FLAT_MIN_PCT)
                    if not flat_ok:
                        await asyncio.sleep(2)
                        continue

                    # Decide trade based on last CLOSED candle, then enter immediately (we are now in next candle)
                    # CALL
                    if ema10 > ema20:
                        touched = float(sig["l"]) <= ema20
                        confirmed = float(sig["c"]) > ema20
                        if touched and confirmed and (rsi <= S2_RSI_BUY_MAX):
                            await self.execute_trade("CALL", symbol, "S2: Bounce BUY (touch EMA20 + RSI oversold + close above)", source="AUTO")
                            self.s2_rsi_reset_lock[symbol] = True

                    # PUT
                    elif ema10 < ema20:
                        touched = float(sig["h"]) >= ema20
                        confirmed = float(sig["c"]) < ema20
                        if touched and confirmed and (rsi >= S2_RSI_SELL_MIN):
                            await self.execute_trade("PUT", symbol, "S2: Bounce SELL (touch EMA20 + RSI overbought + close below)", source="AUTO")
                            self.s2_rsi_reset_lock[symbol] = True

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scanner Error ({symbol}): {e}")
                self.market_debug[symbol] = {
                    "time": time.time(),
                    "strategy": self.strategy_name(),
                    "trend": "—",
                    "gate": f"Error: {str(e)[:80]}",
                    "reasons": [f"Error: {str(e)[:120]}"],
                    "last_closed": self.last_closed_m1_time.get(symbol, 0),
                }
            await asyncio.sleep(2)

    async def execute_trade(self, side: str, symbol: str, reason="MANUAL", source="MANUAL"):
        if not self.api or self.active_trade_info:
            return

        async with self.trade_lock:
            try:
                stake = self.current_stake if source == "AUTO" else BASE_STAKE
                duration_min = self.current_expiry_min()

                prop = await self.api.proposal({
                    "proposal": 1,
                    "amount": stake,
                    "basis": "stake",
                    "contract_type": side,
                    "currency": "USD",
                    "duration": int(duration_min),
                    "duration_unit": "m",
                    "symbol": symbol
                })
                buy = await self.api.buy({
                    "buy": prop["proposal"]["id"],
                    "price": float(prop["proposal"]["ask_price"])
                })

                self.active_trade_info = int(buy["buy"]["contract_id"])
                self.active_market = symbol
                self.trade_start_time = time.time()

                if source == "AUTO":
                    self.trades_today += 1

                safe_symbol = str(symbol).replace("_", " ")
                msg = (
                    f"🚀 {side} TRADE OPENED (${stake})\n"
                    f"🛒 Market: {safe_symbol}\n"
                    f"⏱ Expiry: {duration_min}m\n"
                    f"🧠 Reason: {reason}\n"
                    f"📌 Strategy: {self.strategy_name()}\n"
                    f"🤖 Source: {source}"
                )
                await self.app.bot.send_message(TELEGRAM_CHAT_ID, msg)

                asyncio.create_task(self.check_result(self.active_trade_info, source, duration_min))
            except Exception as e:
                logger.error(f"Trade error: {e}")

    async def check_result(self, cid: int, source: str, duration_min: int):
        await asyncio.sleep(int(duration_min) * 60 + 5)
        try:
            res = await self.api.proposal_open_contract({"proposal_open_contract": 1, "contract_id": cid})
            profit = float(res["proposal_open_contract"].get("profit", 0))

            if source == "AUTO":
                self.total_profit_today += profit
                if profit <= 0:
                    self.consecutive_losses += 1
                    self.total_losses_today += 1
                    self.current_stake *= 2
                else:
                    self.consecutive_losses = 0
                    self.current_stake = BASE_STAKE

            await self.fetch_balance()
            await self.app.bot.send_message(
                TELEGRAM_CHAT_ID,
                f"🏁 FINISH: {'WIN' if profit > 0 else 'LOSS'} (${profit:.2f})\n💰 Balance: {self.balance}"
            )
        finally:
            self.active_trade_info = None
            self.cooldown_until = time.time() + COOLDOWN_SEC

# ========================= UI =========================
bot_logic = DerivSniperBot()

def main_keyboard():
    # Button label reflects current strategy
    strat_label = "🧠 STRATEGY: S1 (EMA9/21)" if bot_logic.strategy == "S1" else "🧠 STRATEGY: S2 (EMA10/20 Bounce)"
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("▶️ START", callback_data="START_SCAN"),
         InlineKeyboardButton("⏹️ STOP", callback_data="STOP_SCAN")],
        [InlineKeyboardButton("📊 STATUS", callback_data="STATUS"),
         InlineKeyboardButton("🔄 REFRESH", callback_data="STATUS")],
        [InlineKeyboardButton(strat_label, callback_data="TOGGLE_STRATEGY")],
        [InlineKeyboardButton("🧪 TEST BUY", callback_data="TEST_BUY")],
        [InlineKeyboardButton("🧪 DEMO", callback_data="SET_DEMO"),
         InlineKeyboardButton("💰 LIVE", callback_data="SET_REAL")]
    ])

def format_market_detail(sym: str, d: dict) -> str:
    if not d:
        return f"📍 {sym.replace('_',' ')}\nStatus: ⏳ No scan data yet"

    age = int(time.time() - d.get("time", time.time()))
    trend = d.get("trend", "—")
    gate = d.get("gate", "—")
    strat = d.get("strategy", "—")

    ema_fast = d.get("ema_fast", None)
    ema_slow = d.get("ema_slow", None)
    slope = d.get("slope", None)
    rsi = d.get("rsi", None)
    rsi_prev = d.get("rsi_prev", None)
    price = d.get("price", None)
    last_closed = d.get("last_closed", 0)

    reasons = d.get("reasons", [])
    why = "\n".join([f"• {x}" for x in reasons[:5]])

    # Status label
    status = "WAIT"
    if str(gate) == "OK":
        status = "SCANNING"
    if "Stopped:" in str(gate):
        status = "STOPPED"
    if "Cooldown" in str(gate):
        status = "COOLDOWN"
    if "Trade in progress" in str(gate):
        status = "IN TRADE"

    block = (
        f"📍 {sym.replace('_',' ')} ({age}s ago)\n"
        f"Strategy: {strat}\n"
        f"Status: {status} | Trend: {trend}\n"
        f"Gate: {gate}\n"
        f"Why:\n{why}\n"
        f"Last closed M1 candle: {fmt_time_hhmmss(last_closed)}\n"
    )

    if all(v is not None for v in [ema_fast, ema_slow, slope, rsi, rsi_prev, price]):
        dir_arrow = rsi_dir_arrow(rsi, rsi_prev)
        block += (
            f"Data: Price {price:.2f}\n"
            f"EMA fast {ema_fast:.2f} | EMA slow {ema_slow:.2f} | Gap/Slope {slope:+.4f}\n"
            f"RSI {rsi:.0f} {dir_arrow} (prev {rsi_prev:.0f})\n"
        )

    # Strategy 2 lock display
    if "lock" in d and d["lock"] is not None:
        block += f"Reset lock: {d['lock']}\n"

    return block

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
        await q.edit_message_text(f"🔍 SCANNER ACTIVE\n📌 Using: {bot_logic.strategy_name()}", reply_markup=main_keyboard())

    elif q.data == "STOP_SCAN":
        bot_logic.is_scanning = False
        await q.edit_message_text("⏹️ Scanner stopped.", reply_markup=main_keyboard())

    elif q.data == "TOGGLE_STRATEGY":
        bot_logic.toggle_strategy()
        # Reset per-market candle decision so it updates cleanly on next status
        bot_logic.last_decision_candle_t0 = {m: 0 for m in MARKETS}
        await q.edit_message_text(f"✅ Switched to: {bot_logic.strategy_name()}\n⏱ Expiry: {bot_logic.current_expiry_min()}m", reply_markup=main_keyboard())

    elif q.data == "TEST_BUY":
        await bot_logic.execute_trade("CALL", "R_10", "Manual Test", source="MANUAL")

    elif q.data == "STATUS":
        await bot_logic.fetch_balance()

        now_time = datetime.now(ZoneInfo("Africa/Lagos")).strftime("%Y-%m-%d %H:%M:%S")
        ok, gate = bot_logic.can_auto_trade()

        trade_status = "No Active Trade"
        if bot_logic.active_trade_info and bot_logic.api:
            try:
                res = await bot_logic.api.proposal_open_contract({"proposal_open_contract": 1, "contract_id": bot_logic.active_trade_info})
                pnl = float(res["proposal_open_contract"].get("profit", 0))
                rem = max(0, int(bot_logic.current_expiry_min() * 60) - int(time.time() - bot_logic.trade_start_time))
                icon = "✅ PROFIT" if pnl > 0 else "❌ LOSS" if pnl < 0 else "➖ FLAT"
                mkt_clean = str(bot_logic.active_market).replace("_", " ")
                trade_status = f"🚀 Active Trade ({mkt_clean})\n📈 PnL: {icon} (${pnl:.2f})\n⏳ Left: {rem}s"
            except:
                trade_status = "🚀 Active Trade: Syncing..."

        header = (
            f"🕒 Time (WAT): {now_time}\n"
            f"🤖 Bot: {'ACTIVE' if bot_logic.is_scanning else 'OFFLINE'} ({bot_logic.account_type})\n"
            f"📌 Strategy: {bot_logic.strategy_name()}\n"
            f"⏱ Expiry: {bot_logic.current_expiry_min()}m\n"
            f"📡 Markets: {', '.join(MARKETS).replace('_', ' ')}\n"
            f"━━━━━━━━━━━━━━━\n{trade_status}\n━━━━━━━━━━━━━━━\n"
            f"💵 Total Profit: ${bot_logic.total_profit_today:.2f}\n"
            f"🎯 Today: {bot_logic.trades_today}/{MAX_TRADES_PER_DAY} | ❌ Losses: {bot_logic.total_losses_today}\n"
            f"📉 Streak: {bot_logic.consecutive_losses}/{MAX_CONSEC_LOSSES} | 🧪 Next Stake: ${bot_logic.current_stake:.2f}\n"
            f"🚦 Gate: {gate}\n"
            f"💰 Balance: {bot_logic.balance}\n"
        )

        details = "\n\n📌 LIVE SCAN (Detailed)\n\n" + "\n\n".join(
            [format_market_detail(sym, bot_logic.market_debug.get(sym, {})) for sym in MARKETS]
        )

        await q.edit_message_text(header + details, reply_markup=main_keyboard())

async def start_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    await u.message.reply_text(
        "💎 Deriv Bot\n"
        "✅ Two strategies included:\n"
        "S1: EMA9/21 + RSI14 (5m expiry)\n"
        "S2: EMA10/20 Bounce + RSI7 (1m expiry)\n\n"
        "Use the 🧠 STRATEGY button to switch.",
        reply_markup=main_keyboard()
    )

if __name__ == "__main__":
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    bot_logic.app = app
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CallbackQueryHandler(btn_handler))
    app.run_polling(drop_pending_updates=True)
