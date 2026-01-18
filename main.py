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
# Your strategy target: ~10 trades/day (cap)
MAX_TRADES_PER_DAY_TOTAL = 10
MAX_TRADES_PER_MARKET_PER_DAY = 3
STOP_DAY_AFTER_TOTAL_LOSSES = 3
STOP_SYMBOL_AFTER_LOSSES = 2

COOLDOWN_AFTER_TRADE_SEC = 120          # extra cooldown after a trade (global)
COOLDOWN_PER_SYMBOL_SEC = 120           # 2 full M1 candles cooldown on same symbol

BASE_STAKE = 1.00

# ========================= STRATEGY SETTINGS =========================
ENTRY_TF_SEC = 60  # M1 entries

# Structure TF by market (as per your rules)
STRUCT_TF_SEC = {
    "R_10": 300,   # M5
    "R_25": 300,   # M5
    "R_50": 900,   # M15
    "R_75": 900,   # M15
    "R_100": 900,  # M15
}

# Expiry per index (minutes)
EXPIRY_MIN = {
    "R_10": 3,   # 2-3 ok; we use 3
    "R_25": 2,
    "R_50": 2,   # 1-2 ok; we use 2
    "R_75": 1,
    "R_100": 1,
}

# Swing fractal N (5-candle fractal)
SWING_N = 2

# Sweep buffer (0.05% of price)
SWEEP_BUFFER_PCT = 0.0005

# Optional RSI filter on M1
RSI_PERIOD = 14
RSI_LO = 30
RSI_HI = 70
USE_RSI_FILTER = True

# Filters
DOJI_BODY_PCT = 0.25        # doji if body <= 25% of range
CHOP_DOJI_COUNT = 6         # if >=6 dojis in last 10 -> chop
CHOP_LOOKBACK = 10
CHOP_PAUSE_SEC = 600        # pause 10 minutes on that symbol

SPIKE_MULTIPLIER = 2.0
AVG_RANGE_LOOKBACK = 20

# ========================= HELPERS =========================
def build_candles_from_deriv(candles_raw):
    out = []
    for x in candles_raw:
        out.append({
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
    # returns indices of swing highs and swing lows
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

    # liquidity levels = most recent confirmed swing high/low
    liquidity_high = struct_candles[highs_idx[-1]]["h"]
    liquidity_low = struct_candles[lows_idx[-1]]["l"]

    if up:
        return "UP", liquidity_high, liquidity_low
    if down:
        return "DOWN", liquidity_high, liquidity_low
    return "RANGE", liquidity_high, liquidity_low

def fmt_scan(sym, d):
    age = int(time.time() - d.get("time", time.time()))
    lines = [
        f"• {sym.replace('_',' ')} ({age}s)",
        f"  Bias: {d.get('bias','?')} | Setup: {d.get('setup','-')} | Signal: {d.get('signal','-')}",
    ]
    if d.get("levels"):
        lines.append(f"  Levels: {d['levels']}")
    if d.get("ind"):
        lines.append(f"  {d['ind']}")
    if d.get("waiting"):
        lines.append(f"  ⏳ {d['waiting']}")
    return "\n".join(lines)

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

        # Daily tracking
        self.day_key = None
        self.trades_today_total = 0
        self.losses_today_total = 0
        self.trades_today_by_symbol = {m: 0 for m in MARKETS}
        self.losses_today_by_symbol = {m: 0 for m in MARKETS}
        self.disabled_symbol_today = {m: False for m in MARKETS}

        # Symbol cooldown + chop lock
        self.symbol_cooldown_until = {m: 0.0 for m in MARKETS}
        self.symbol_chop_until = {m: 0.0 for m in MARKETS}

        self.balance = "0.00"
        self.total_profit_today = 0.0
        self.current_stake = BASE_STAKE
        self.consecutive_losses = 0
        self.trade_lock = asyncio.Lock()

        self.market_debug = {}

    def reset_day_if_needed(self):
        # Reset at local day change (Africa/Lagos)
        now = datetime.now(ZoneInfo("Africa/Lagos"))
        key = now.strftime("%Y-%m-%d")
        if self.day_key != key:
            self.day_key = key
            self.trades_today_total = 0
            self.losses_today_total = 0
            self.trades_today_by_symbol = {m: 0 for m in MARKETS}
            self.losses_today_by_symbol = {m: 0 for m in MARKETS}
            self.disabled_symbol_today = {m: False for m in MARKETS}
            self.symbol_cooldown_until = {m: 0.0 for m in MARKETS}
            self.symbol_chop_until = {m: 0.0 for m in MARKETS}
            self.total_profit_today = 0.0
            self.current_stake = BASE_STAKE
            self.consecutive_losses = 0

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

    def gate(self, symbol: str):
        self.reset_day_if_needed()

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
                # safety timeout
                if self.active_trade_info and (time.time() - self.trade_start_time > (EXPIRY_MIN.get(self.active_market, 2) * 60 + 90)):
                    self.active_trade_info = None
                await asyncio.sleep(1)
        finally:
            for t in self.market_tasks.values():
                t.cancel()
            self.market_tasks.clear()

    async def fetch_candles(self, symbol: str, granularity: int, count: int):
        res = await self.api.ticks_history({
            "ticks_history": symbol,
            "end": "latest",
            "count": count,
            "style": "candles",
            "granularity": granularity
        })
        raw = res.get("candles", [])
        return build_candles_from_deriv(raw)

    async def scan_symbol(self, symbol: str):
        while self.is_scanning:
            try:
                ok, g = self.gate(symbol)
                if not ok:
                    self.market_debug[symbol] = {
                        "time": time.time(),
                        "bias": "-",
                        "setup": "-",
                        "signal": "-",
                        "levels": "",
                        "ind": "",
                        "waiting": f"🚦 {g}"
                    }
                    await asyncio.sleep(2)
                    continue

                # Fetch structure candles (bias + liquidity levels)
                struct_tf = STRUCT_TF_SEC[symbol]
                struct = await self.fetch_candles(symbol, struct_tf, 220)

                # Fetch entry candles (M1)
                m1 = await self.fetch_candles(symbol, ENTRY_TF_SEC, 220)

                if len(struct) < 50 or len(m1) < 30:
                    self.market_debug[symbol] = {
                        "time": time.time(),
                        "bias": "…",
                        "setup": "-",
                        "signal": "-",
                        "levels": "",
                        "ind": "",
                        "waiting": "Syncing candles..."
                    }
                    await asyncio.sleep(3)
                    continue

                # Use only CLOSED candles: last closed = -2
                if len(m1) < 5:
                    await asyncio.sleep(2)
                    continue
                c_confirm = m1[-2]  # confirmation / continuation candle (closed)
                c_sweep = m1[-3]    # sweep candle (closed)
                c_prev = m1[-4]     # candle before sweep

                # Chop filter (doji count last 10 closed candles)
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
                        "ind": f"Dojis: {dojis}/{CHOP_LOOKBACK}",
                        "waiting": "Chop detected → pausing 10m"
                    }
                    await asyncio.sleep(2)
                    continue

                # Bias + liquidity levels
                bias, liq_high, liq_low = determine_bias(struct)

                # ✅ OPTION 1 CHANGE:
                # If we have NO levels at all, we must skip (can't do Setup A or B).
                if liq_high is None or liq_low is None:
                    self.market_debug[symbol] = {
                        "time": time.time(),
                        "bias": bias,
                        "setup": "-",
                        "signal": "-",
                        "levels": "Need clearer swings",
                        "ind": "",
                        "waiting": "No levels yet (waiting swings)"
                    }
                    await asyncio.sleep(2)
                    continue
                # (We DO NOT block RANGE here anymore — because Setup A can trade in RANGE)

                # RSI optional filter
                closes_m1 = [x["c"] for x in m1]
                rsi_arr = calculate_rsi(closes_m1, RSI_PERIOD)
                rsi_now = float(rsi_arr[-1]) if len(rsi_arr) > 0 else 50.0
                rsi_prev = float(rsi_arr[-2]) if len(rsi_arr) > 1 else rsi_now

                # Sweep buffer (percent of price)
                price = float(c_sweep["c"])
                buf = price * SWEEP_BUFFER_PCT

                # Spike filter on the SIGNAL candle (confirmation candle)
                avg_rng = avg_range(m1[-(AVG_RANGE_LOOKBACK+1):-1], AVG_RANGE_LOOKBACK)
                sig_rng = candle_range(c_confirm)
                if sig_rng > SPIKE_MULTIPLIER * avg_rng:
                    self.market_debug[symbol] = {
                        "time": time.time(),
                        "bias": bias,
                        "setup": "-",
                        "signal": "-",
                        "levels": f"H:{liq_high:.2f} L:{liq_low:.2f} | buf:{buf:.2f}",
                        "ind": f"RSI: {rsi_now:.0f} | Spike: {sig_rng:.2f}>{SPIKE_MULTIPLIER}x{avg_rng:.2f}",
                        "waiting": "Spike candle → skip"
                    }
                    await asyncio.sleep(2)
                    continue

                # ========================= SETUP A: LIQUIDITY REVERSAL =========================
                # SELL (Fall): sweep above swing high, close back below, then confirmation bearish
                sweep_sell = (c_sweep["h"] >= (liq_high + buf)) and (c_sweep["c"] < liq_high)
                confirm_sell = (bear_engulf(c_sweep, c_confirm) or strong_bear_close(c_confirm))
                rsi_sell_ok = (rsi_now < RSI_HI and rsi_prev >= RSI_HI) or (rsi_prev > RSI_HI) or (rsi_now > RSI_HI)

                sell_A_ready = sweep_sell and confirm_sell and ((not USE_RSI_FILTER) or rsi_sell_ok)

                # BUY (Rise): sweep below swing low, close back above, then confirmation bullish
                sweep_buy = (c_sweep["l"] <= (liq_low - buf)) and (c_sweep["c"] > liq_low)
                confirm_buy = (bull_engulf(c_sweep, c_confirm) or strong_bull_close(c_confirm))
                rsi_buy_ok = (rsi_now > RSI_LO and rsi_prev <= RSI_LO) or (rsi_prev < RSI_LO) or (rsi_now < RSI_LO)

                buy_A_ready = sweep_buy and confirm_buy and ((not USE_RSI_FILTER) or rsi_buy_ok)

                # ========================= SETUP B: STRUCTURE CONTINUATION =========================
                # ✅ Still requires non-RANGE bias (UP/DOWN only)
                pb1 = m1[-4]
                pb2 = m1[-3]
                cont = m1[-2]
                prev_for_break = m1[-3]

                if bias == "UP":
                    pullback_ok = (pb1["c"] < pb1["o"]) and (pb2["c"] < pb2["o"])
                    continuation_ok = (cont["c"] > cont["o"]) and (cont["c"] > prev_for_break["h"]) and strong_bull_close(cont)
                    rsi_ok = (rsi_now > 50)
                    buy_B_ready = pullback_ok and continuation_ok and ((not USE_RSI_FILTER) or rsi_ok)
                    sell_B_ready = False
                elif bias == "DOWN":
                    pullback_ok = (pb1["c"] > pb1["o"]) and (pb2["c"] > pb2["o"])
                    continuation_ok = (cont["c"] < cont["o"]) and (cont["c"] < prev_for_break["l"]) and strong_bear_close(cont)
                    rsi_ok = (rsi_now < 50)
                    sell_B_ready = pullback_ok and continuation_ok and ((not USE_RSI_FILTER) or rsi_ok)
                    buy_B_ready = False
                else:
                    buy_B_ready = False
                    sell_B_ready = False

                # Priority: Setup A > Setup B (as per your rule)
                setup = "-"
                signal = "-"
                waiting = ""
                levels_txt = f"H:{liq_high:.2f}  L:{liq_low:.2f}  buf:{buf:.2f}"
                ind_txt = f"RSI(14): {rsi_now:.0f} | Doji:{dojis}/10 | avgR:{avg_rng:.2f}"

                if buy_A_ready:
                    setup = "A-Reversal"
                    signal = "BUY (RISE)"
                    waiting = "✅ Liquidity sweep low + bullish confirm"
                    self.market_debug[symbol] = {"time": time.time(), "bias": bias, "setup": setup, "signal": signal, "levels": levels_txt, "ind": ind_txt, "waiting": waiting}
                    await self.execute_trade("CALL", symbol, f"Setup A BUY | {levels_txt} | RSI {rsi_now:.0f}")
                elif sell_A_ready:
                    setup = "A-Reversal"
                    signal = "SELL (FALL)"
                    waiting = "✅ Liquidity sweep high + bearish confirm"
                    self.market_debug[symbol] = {"time": time.time(), "bias": bias, "setup": setup, "signal": signal, "levels": levels_txt, "ind": ind_txt, "waiting": waiting}
                    await self.execute_trade("PUT", symbol, f"Setup A SELL | {levels_txt} | RSI {rsi_now:.0f}")
                elif buy_B_ready:
                    setup = "B-Continuation"
                    signal = "BUY (RISE)"
                    waiting = "✅ Pullback then continuation break"
                    self.market_debug[symbol] = {"time": time.time(), "bias": bias, "setup": setup, "signal": signal, "levels": levels_txt, "ind": ind_txt, "waiting": waiting}
                    await self.execute_trade("CALL", symbol, f"Setup B BUY | Bias {bias} | RSI {rsi_now:.0f}")
                elif sell_B_ready:
                    setup = "B-Continuation"
                    signal = "SELL (FALL)"
                    waiting = "✅ Pullback then continuation break"
                    self.market_debug[symbol] = {"time": time.time(), "bias": bias, "setup": setup, "signal": signal, "levels": levels_txt, "ind": ind_txt, "waiting": waiting}
                    await self.execute_trade("PUT", symbol, f"Setup B SELL | Bias {bias} | RSI {rsi_now:.0f}")
                else:
                    # Helpful waiting text
                    notes = []
                    if USE_RSI_FILTER:
                        notes.append("RSI filter ON")
                    notes.append("Need sweep+reject+confirm (A) OR pullback+break (B)")
                    self.market_debug[symbol] = {
                        "time": time.time(),
                        "bias": bias,
                        "setup": "-",
                        "signal": "-",
                        "levels": levels_txt,
                        "ind": ind_txt,
                        "waiting": " | ".join(notes)
                    }

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scan error {symbol}: {e}")
                self.market_debug[symbol] = {
                    "time": time.time(),
                    "bias": "-",
                    "setup": "-",
                    "signal": "-",
                    "levels": "",
                    "ind": "",
                    "waiting": f"⚠️ Error: {str(e)[:120]}"
                }

            await asyncio.sleep(2)

    async def execute_trade(self, side: str, symbol: str, reason: str):
        if not self.api or self.active_trade_info:
            return

        async with self.trade_lock:
            ok, g = self.gate(symbol)
            if not ok:
                return

            try:
                duration = int(EXPIRY_MIN.get(symbol, 2))
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

                # update caps
                self.trades_today_total += 1
                self.trades_today_by_symbol[symbol] += 1

                # cooldowns
                self.global_cooldown_until = time.time() + COOLDOWN_AFTER_TRADE_SEC
                self.symbol_cooldown_until[symbol] = time.time() + COOLDOWN_PER_SYMBOL_SEC

                msg = (
                    f"🚀 {side} OPENED (${stake:.2f})\n"
                    f"🛒 Market: {symbol.replace('_',' ')}\n"
                    f"⏱ Expiry: {duration}m\n"
                    f"📌 Strategy: Liquidity + Structure (A/B internal)\n"
                    f"🧠 {reason}"
                )
                await self.app.bot.send_message(TELEGRAM_CHAT_ID, msg)

                asyncio.create_task(self.check_result(self.active_trade_info, symbol))

            except Exception as e:
                logger.error(f"Trade error: {e}")

    async def check_result(self, cid: int, symbol: str):
        # wait for expiry + small buffer
        await asyncio.sleep(EXPIRY_MIN.get(symbol, 2) * 60 + 5)

        try:
            res = await self.api.proposal_open_contract({"proposal_open_contract": 1, "contract_id": cid})
            profit = float(res["proposal_open_contract"].get("profit", 0.0))

            self.total_profit_today += profit

            if profit <= 0:
                self.losses_today_total += 1
                self.losses_today_by_symbol[symbol] += 1
                self.consecutive_losses += 1

                # disable symbol if loss cap reached
                if self.losses_today_by_symbol[symbol] >= STOP_SYMBOL_AFTER_LOSSES:
                    self.disabled_symbol_today[symbol] = True

                # optional martingale kept as your earlier bot
                self.current_stake *= 2
            else:
                self.consecutive_losses = 0
                self.current_stake = BASE_STAKE

            await self.fetch_balance()
            await self.app.bot.send_message(
                TELEGRAM_CHAT_ID,
                f"🏁 FINISH: {'WIN' if profit > 0 else 'LOSS'} ({profit:+.2f})\n💰 Balance: {self.balance}"
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
        await q.edit_message_text("🔍 SCANNER ACTIVE\n📌 Strategy: Liquidity + Structure (A/B internal)", reply_markup=main_keyboard())

    elif q.data == "STOP_SCAN":
        bot_logic.is_scanning = False
        await q.edit_message_text("⏹️ Scanner stopped.", reply_markup=main_keyboard())

    elif q.data == "TEST_BUY":
        await bot_logic.execute_trade("CALL", "R_10", "Manual test trade")

    elif q.data == "STATUS":
        await bot_logic.fetch_balance()
        now_time = datetime.now(ZoneInfo("Africa/Lagos")).strftime("%Y-%m-%d %H:%M:%S")
        ok, gate = bot_logic.gate("R_10")  # just to show global readiness

        trade_status = "No Active Trade"
        if bot_logic.active_trade_info and bot_logic.api:
            try:
                res = await bot_logic.api.proposal_open_contract({"proposal_open_contract": 1, "contract_id": bot_logic.active_trade_info})
                pnl = float(res["proposal_open_contract"].get("profit", 0.0))
                rem = int(max(0, (EXPIRY_MIN.get(bot_logic.active_market, 2) * 60) - (time.time() - bot_logic.trade_start_time)))
                mkt = (bot_logic.active_market or "—").replace("_", " ")
                trade_status = f"🚀 Active Trade ({mkt})\n📈 PnL: {pnl:+.2f}\n⏳ Left: {rem}s"
            except:
                trade_status = "🚀 Active Trade: Syncing..."

        status_msg = (
            f"🕒 Time (WAT): {now_time}\n"
            f"🤖 Bot: {'ACTIVE' if bot_logic.is_scanning else 'OFFLINE'} ({bot_logic.account_type})\n"
            f"📌 Strategy: Price Action Liquidity + Structure (Setup A + Setup B)\n"
            f"📡 Markets: {', '.join(MARKETS).replace('_',' ')}\n"
            f"━━━━━━━━━━━━━━━\n{trade_status}\n━━━━━━━━━━━━━━━\n"
            f"🎯 Trades Today: {bot_logic.trades_today_total}/{MAX_TRADES_PER_DAY_TOTAL}\n"
            f"❌ Losses Today: {bot_logic.losses_today_total}/{STOP_DAY_AFTER_TOTAL_LOSSES}\n"
            f"🧪 Next Stake: ${bot_logic.current_stake:.2f}\n"
            f"💰 Balance: {bot_logic.balance}"
        )

        debug_lines = []
        for sym in MARKETS:
            d = bot_logic.market_debug.get(sym)
            if not d:
                debug_lines.append(f"• {sym.replace('_',' ')}\n  ⏳ No scan data yet")
            else:
                debug_lines.append(fmt_scan(sym, d))

        status_msg += "\n\n📌 LIVE SCAN\n" + "\n\n".join(debug_lines)
        await q.edit_message_text(status_msg, reply_markup=main_keyboard())

async def start_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    await u.message.reply_text(
        "💎 Deriv Bot\n"
        "📌 Strategy: Price Action Liquidity + Structure\n"
        "✅ Setup A: Liquidity sweep reversal\n"
        "✅ Setup B: Trend continuation\n"
        "🕯 Entry: M1 | Bias: M5/M15 | Expiry: per index\n",
        reply_markup=main_keyboard()
    )

if __name__ == "__main__":
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    bot_logic.app = app
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CallbackQueryHandler(btn_handler))
    app.run_polling(drop_pending_updates=True)
