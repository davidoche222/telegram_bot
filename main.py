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

# ADDED R_75 and R_100 (only change here)
MARKETS = ["R_10", "R_25", "R_50", "R_75", "R_100"]

COOLDOWN_SEC = 120
MAX_TRADES_PER_DAY = 40
MAX_CONSEC_LOSSES = 10

BASE_STAKE = 1.00  # <--- Initial trade amount

TELEGRAM_TOKEN = "8253450930:AAHUhPk9TML-8kZlA9UaHZZvTUGdurN9MVY"
TELEGRAM_CHAT_ID = "7634818949"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ========================= STRATEGY CONFIG (ONLY STRATEGY SETTINGS) =========================
TIMEFRAME_SEC = 300          # 5 minutes candles
EXPIRY_CANDLES = 2           # 2 candles expiry
TRADE_DURATION_MIN = (TIMEFRAME_SEC * EXPIRY_CANDLES) // 60  # 10 minutes
TRADE_DURATION_SEC = TIMEFRAME_SEC * EXPIRY_CANDLES          # 600 seconds

RSI_FAST_PERIOD = 1
RSI_LEVEL_BUY = 20
RSI_LEVEL_SELL = 80

MFI_PERIOD = 50
MFI_LEVEL_BUY = 20
MFI_LEVEL_SELL = 80

# Alligator: using periods=2, shifts=10 (as per your settings)
ALLIGATOR_JAWS_PERIOD = 2
ALLIGATOR_TEETH_PERIOD = 2
ALLIGATOR_LIPS_PERIOD = 2
ALLIGATOR_JAWS_SHIFT = 10
ALLIGATOR_TEETH_SHIFT = 10
ALLIGATOR_LIPS_SHIFT = 10

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
    # Works for period=1 too.
    if len(data) < (period + 1):
        return np.array([])
    values = np.array(data, dtype=float)
    delta = np.diff(values)
    gain = delta.clip(min=0)
    loss = (-delta).clip(min=0)

    # Wilder-style smoothing approximation using rolling mean for simplicity
    avg_gain = np.convolve(gain, np.ones(period), "valid") / period
    avg_loss = np.convolve(loss, np.ones(period), "valid") / period
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def _smma(values, period):
    # Smoothed moving average (Wilder / SMMA)
    values = np.array(values, dtype=float)
    if len(values) < period:
        return np.array([])
    out = np.zeros_like(values)
    out[:period - 1] = np.nan
    out[period - 1] = np.mean(values[:period])
    for i in range(period, len(values)):
        out[i] = (out[i - 1] * (period - 1) + values[i]) / period
    return out

def calculate_mfi(high, low, close, volume, period=14):
    # Money Flow Index using tick-count volume per candle (since synth indices don't provide real volume).
    high = np.array(high, dtype=float)
    low = np.array(low, dtype=float)
    close = np.array(close, dtype=float)
    volume = np.array(volume, dtype=float)

    if len(close) < (period + 1):
        return np.array([])

    typical = (high + low + close) / 3.0
    raw_mf = typical * volume

    tp_diff = np.diff(typical)
    pos_mf = np.where(tp_diff > 0, raw_mf[1:], 0.0)
    neg_mf = np.where(tp_diff < 0, raw_mf[1:], 0.0)

    # rolling sums
    pos_sum = np.convolve(pos_mf, np.ones(period), "valid")
    neg_sum = np.convolve(neg_mf, np.ones(period), "valid")

    mfr = pos_sum / (neg_sum + 1e-9)
    mfi = 100 - (100 / (1 + mfr))
    return mfi

def calculate_indicators(candles):
    # candles have: o,h,l,c,v
    c = np.array([x["c"] for x in candles], dtype=float)
    o = np.array([x["o"] for x in candles], dtype=float)
    h = np.array([x["h"] for x in candles], dtype=float)
    l = np.array([x["l"] for x in candles], dtype=float)
    v = np.array([x.get("v", 1) for x in candles], dtype=float)

    # RSI Fast (period=1)
    rsi_fast = calculate_rsi(c, RSI_FAST_PERIOD)
    if len(rsi_fast) < 2:
        return None

    # MFI (period=50)
    mfi_vals = calculate_mfi(h, l, c, v, MFI_PERIOD)
    if len(mfi_vals) < 2:
        return None

    # Alligator lines: SMMA of median price
    median = (h + l) / 2.0
    jaws = _smma(median, ALLIGATOR_JAWS_PERIOD)
    teeth = _smma(median, ALLIGATOR_TEETH_PERIOD)
    lips = _smma(median, ALLIGATOR_LIPS_PERIOD)
    if len(jaws) == 0 or len(teeth) == 0 or len(lips) == 0:
        return None

    # Apply shift by referencing earlier values (shift forward visually means using older value).
    def shifted(arr, shift):
        if len(arr) <= shift:
            return np.array([])
        return arr[:-shift]

    jaws_s = shifted(jaws, ALLIGATOR_JAWS_SHIFT)
    teeth_s = shifted(teeth, ALLIGATOR_TEETH_SHIFT)
    lips_s = shifted(lips, ALLIGATOR_LIPS_SHIFT)

    # align lengths by trimming to shortest
    min_len = min(len(jaws_s), len(teeth_s), len(lips_s), len(c))
    if min_len < 5:
        return None

    # take last aligned
    jaws_v = jaws_s[-1]
    teeth_v = teeth_s[-1]
    lips_v = lips_s[-1]

    # basic "awake" check: lines not too tangled
    spread = abs(lips_v - teeth_v) + abs(teeth_v - jaws_v)
    price_scale = max(1e-9, abs(c[-1]))
    is_awake = spread > (price_scale * 1e-5)  # tiny but avoids totally flat

    # Trend direction by alligator ordering
    alligator_up = (lips_v > teeth_v > jaws_v)
    alligator_down = (lips_v < teeth_v < jaws_v)

    return {
        "o": o[-1],
        "c": c[-1],

        "rsi_fast": rsi_fast[-1],
        "rsi_fast_prev": rsi_fast[-2],

        "mfi": mfi_vals[-1],
        "mfi_prev": mfi_vals[-2],

        "jaws": jaws_v,
        "teeth": teeth_v,
        "lips": lips_v,
        "is_awake": is_awake,
        "allig_up": alligator_up,
        "allig_down": alligator_down,
    }

def build_m1_candles_from_ticks(times, prices):
    # NOTE: kept the function name the same, but it now builds 5-minute candles (M5) for the strategy.
    if not times or not prices:
        return []
    candles = []
    # bucket by 5 minutes
    bucket = TIMEFRAME_SEC
    curr_t0 = times[0] - (times[0] % bucket)

    o = h = l = c = float(prices[0])
    v = 0  # tick count per candle

    for t, p in zip(times, prices):
        t0 = t - (t % bucket)
        p = float(p)
        if t0 != curr_t0:
            candles.append({"o": o, "h": h, "l": l, "c": c, "v": v})
            curr_t0, o, h, l, c, v = t0, p, p, p, p, 0
        else:
            h, l, c = max(h, p), min(l, p), p
        v += 1

    candles.append({"o": o, "h": h, "l": l, "c": c, "v": v})
    return candles

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
        self.total_profit_today = 0.0  # <--- ADDED PROFIT TRACKER
        self.balance = "0.00"
        self.current_stake = BASE_STAKE  # <--- ADDED MARTINGALE TRACKER
        self.trade_lock = asyncio.Lock()
        self.last_scan_symbol = "None"
        self.last_signal_reason = "None"
        self.last_block_reason = "None"
        self.last_trade_side = "None"
        self.last_trade_source = "None"

    async def connect(self) -> bool:
        try:
            if not self.active_token:
                return False
            self.api = DerivAPI(app_id=APP_ID)
            await self.api.authorize(self.active_token)
            await self.fetch_balance()
            return True
        except:
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
            return False, "Max Streak Loss"
        if self.trades_today >= MAX_TRADES_PER_DAY:
            return False, "Daily Limit Met"
        if time.time() < self.cooldown_until:
            return False, f"Cooldown {int(self.cooldown_until - time.time())}s"
        if self.active_trade_info:
            return False, "Trade in Progress"
        if not self.api:
            return False, "Not Connected"
        return True, "OK"

    async def background_scanner(self):
        if not self.api:
            return
        self.market_tasks = {sym: asyncio.create_task(self.scan_market(sym)) for sym in MARKETS}
        try:
            while self.is_scanning:
                # safety timeout for a trade
                if self.active_trade_info and (time.time() - self.trade_start_time > (TRADE_DURATION_SEC + 30)):
                    self.active_trade_info = None
                await asyncio.sleep(1)
        finally:
            for t in self.market_tasks.values():
                t.cancel()
            self.market_tasks.clear()

    async def scan_market(self, symbol: str):
        while self.is_scanning:
            self.last_scan_symbol = symbol
            try:
                if self.consecutive_losses >= MAX_CONSEC_LOSSES or self.trades_today >= MAX_TRADES_PER_DAY:
                    self.is_scanning = False
                    break

                data = await self.api.ticks_history(
                    {"ticks_history": symbol, "end": "latest", "count": 1000, "style": "ticks"}
                )
                candles = build_m1_candles_from_ticks(data["history"]["times"], data["history"]["prices"])

                # Need enough 5-min candles for MFI(50) + buffer
                if len(candles) < (MFI_PERIOD + 5):
                    await asyncio.sleep(10)
                    continue

                ind = calculate_indicators(candles)
                if not ind:
                    await asyncio.sleep(10)
                    continue

                ok, gate = self.can_auto_trade()
                self.last_block_reason = gate
                if not ok:
                    await asyncio.sleep(10)
                    continue

                # ================= NEW STRATEGY LOGIC: RSI(1) + ALLIGATOR + MFI(50) =================
                # Main confirmation = MFI, and Alligator gives direction (must not be sleeping),
                # RSI(1) gives quick trigger.

                is_green = ind["c"] > ind["o"]
                is_red = ind["c"] < ind["o"]

                # BUY (CALL)
                if ind["is_awake"] and ind["allig_up"]:
                    if (ind["mfi"] <= MFI_LEVEL_BUY) and (ind["rsi_fast"] <= RSI_LEVEL_BUY):
                        # extra safety: both are turning up (momentum shift)
                        if (ind["mfi"] >= ind["mfi_prev"]) and (ind["rsi_fast"] >= ind["rsi_fast_prev"]):
                            if is_green:
                                await self.execute_trade(
                                    "CALL",
                                    symbol,
                                    "M5 RSI(1)<=20 + MFI(50)<=20 + Alligator UP (Awake)",
                                    source="AUTO",
                                )

                # SELL (PUT)
                elif ind["is_awake"] and ind["allig_down"]:
                    if (ind["mfi"] >= MFI_LEVEL_SELL) and (ind["rsi_fast"] >= RSI_LEVEL_SELL):
                        # extra safety: both are turning down from the top (momentum shift)
                        if (ind["mfi"] <= ind["mfi_prev"]) and (ind["rsi_fast"] <= ind["rsi_fast_prev"]):
                            if is_red:
                                await self.execute_trade(
                                    "PUT",
                                    symbol,
                                    "M5 RSI(1)>=80 + MFI(50)>=80 + Alligator DOWN (Awake)",
                                    source="AUTO",
                                )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scanner Error: {e}")
            await asyncio.sleep(5)

    async def execute_trade(self, side: str, symbol: str, reason="MANUAL", source="MANUAL"):
        if not self.api or self.active_trade_info:
            return
        async with self.trade_lock:
            try:
                stake = self.current_stake if source == "AUTO" else BASE_STAKE
                prop = await self.api.proposal(
                    {
                        "proposal": 1,
                        "amount": stake,
                        "basis": "stake",
                        "contract_type": side,
                        "currency": "USD",
                        "duration": int(TRADE_DURATION_MIN),
                        "duration_unit": "m",
                        "symbol": symbol,
                    }
                )
                buy = await self.api.buy(
                    {"buy": prop["proposal"]["id"], "price": float(prop["proposal"]["ask_price"])}
                )
                self.active_trade_info, self.active_market, self.trade_start_time = (
                    int(buy["buy"]["contract_id"]),
                    symbol,
                    time.time(),
                )
                self.last_signal_reason, self.last_trade_side, self.last_trade_source = reason, side, source
                if source == "AUTO":
                    self.trades_today += 1

                safe_symbol = str(symbol).replace("_", " ")
                msg = f"🚀 {side} TRADE OPENED (${stake})\n🛒 Market: {safe_symbol}\n🧠 Source: {source}"
                await self.app.bot.send_message(TELEGRAM_CHAT_ID, msg)
                asyncio.create_task(self.check_result(self.active_trade_info, source))
            except:
                pass

    async def check_result(self, cid: int, source: str):
        await asyncio.sleep(TRADE_DURATION_SEC + 5)
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
                f"🏁 FINISH: {'WIN' if profit > 0 else 'LOSS'} (${profit:.2f})\n💰 Balance: {self.balance}",
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
            [InlineKeyboardButton("📊 STATUS", callback_data="STATUS"), InlineKeyboardButton("🔄 REFRESH", callback_data="STATUS")],
            [InlineKeyboardButton("🧪 TEST BUY", callback_data="TEST_BUY")],
            [InlineKeyboardButton("🧪 DEMO", callback_data="SET_DEMO"), InlineKeyboardButton("💰 LIVE", callback_data="SET_REAL")],
        ]
    )

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
        await q.edit_message_text("🔍 SCANNER ACTIVE", reply_markup=main_keyboard())
    elif q.data == "STOP_SCAN":
        bot_logic.is_scanning = False
        await q.edit_message_text("⏹️ Scanner stopped.", reply_markup=main_keyboard())
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
                rem = max(0, TRADE_DURATION_SEC - int(time.time() - bot_logic.trade_start_time))
                icon = "PROFIT" if pnl > 0 else "LOSS"
                mkt_clean = str(bot_logic.active_market).replace("_", " ")
                trade_status = f"🚀 Active Trade ({mkt_clean})\n📈 PnL: {icon} (${pnl:.2f})\n⏳ Left: {rem}s"
            except:
                trade_status = "🚀 Active Trade: Syncing..."

        status_msg = (
            f"🕒 Time (WAT): {now_time}\n"
            f"🤖 Bot: {'ACTIVE' if bot_logic.is_scanning else 'OFFLINE'} ({bot_logic.account_type})\n"
            f"📡 Markets: {', '.join(MARKETS).replace('_', ' ')}\n"
            f"━━━━━━━━━━━━━━━\n{trade_status}\n━━━━━━━━━━━━━━━\n"
            f"💵 Total Profit: ${bot_logic.total_profit_today:.2f}\n"
            f"🎯 Today: {bot_logic.trades_today}/{MAX_TRADES_PER_DAY} | ❌ Losses: {bot_logic.total_losses_today}\n"
            f"📉 Streak: {bot_logic.consecutive_losses}/{MAX_CONSEC_LOSSES} | 🧪 Next Stake: ${bot_logic.current_stake:.2f}\n"
            f"🚦 Gate: {gate}\n💰 Balance: {bot_logic.balance}"
        )
        await q.edit_message_text(status_msg, reply_markup=main_keyboard())

async def start_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    await u.message.reply_text("💎 Sniper Survival M5 (RSI(1) + Alligator + MFI(50) Edition)", reply_markup=main_keyboard())

if __name__ == "__main__":
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    bot_logic.app = app
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CallbackQueryHandler(btn_handler))
    app.run_polling(drop_pending_updates=True)
