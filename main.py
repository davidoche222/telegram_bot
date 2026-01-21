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
BASE_STAKE = 1.00

TELEGRAM_TOKEN = "8253450930:AAHUhPk9TML-8kZlA9UaHZZvTUGdurN9MVY"
TELEGRAM_CHAT_ID = "7634818949"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ========================= STRATEGY SETTINGS =========================
TF_SEC = 60                # M1 candles (REAL candles from Deriv)
CANDLES_COUNT = 300        # enough for EMA50 + RSI + PSAR
SCAN_SLEEP_SEC = 2

EMA_PERIOD = 50

# Parabolic SAR defaults
PSAR_AF_STEP = 0.02
PSAR_AF_MAX = 0.2

# RSI
RSI_PERIOD = 14

# Contract expiry for binary
DURATION_MIN = 5  # ✅ requested: 5 minute expiry

# ===== CROSS STRATEGY SETTINGS =====
PSAR_MIN_DIST_MULT = 0.50   # PSAR must be at least 0.5 * candle range away from close (tune 0.3–1.0)

# ========================= INDICATOR MATH =========================
def calculate_ema(values, period):
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
    """
    Wilder's RSI. Returns array same length as values.
    First (period) values will be np.nan.
    """
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


def calculate_psar(candles, af_step=0.02, af_max=0.2):
    """
    Standard Parabolic SAR implementation.
    Returns PSAR list (len == len(candles)).
    """
    n = len(candles)
    if n < 3:
        return []

    highs = [c["h"] for c in candles]
    lows = [c["l"] for c in candles]

    psar = [0.0] * n

    uptrend = highs[1] > highs[0]
    ep = highs[1] if uptrend else lows[1]
    af = af_step
    psar[1] = lows[0] if uptrend else highs[0]

    for i in range(2, n):
        prev_psar = psar[i - 1]
        psar_i = prev_psar + af * (ep - prev_psar)

        if uptrend:
            psar_i = min(psar_i, lows[i - 1], lows[i - 2])
        else:
            psar_i = max(psar_i, highs[i - 1], highs[i - 2])

        if uptrend:
            if lows[i] < psar_i:
                uptrend = False
                psar_i = ep
                ep = lows[i]
                af = af_step
            else:
                if highs[i] > ep:
                    ep = highs[i]
                    af = min(af + af_step, af_max)
        else:
            if highs[i] > psar_i:
                uptrend = True
                psar_i = ep
                ep = highs[i]
                af = af_step
            else:
                if lows[i] < ep:
                    ep = lows[i]
                    af = min(af + af_step, af_max)

        psar[i] = psar_i

    psar[0] = psar[1]
    return psar


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
    except:
        return "—"


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

        self.market_debug = {m: {} for m in MARKETS}
        self.last_processed_closed_t0 = {m: 0 for m in MARKETS}

        # Cross-state per symbol (ONE trade per cross)
        self.last_cross_dir = {m: None for m in MARKETS}   # "BULL" / "BEAR" / None
        self.last_cross_t0 = {m: 0 for m in MARKETS}       # time of cross candle
        self.cross_traded = {m: False for m in MARKETS}    # prevents multiple trades per cross

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
                if self.active_trade_info and (time.time() - self.trade_start_time > (DURATION_MIN * 60 + 90)):
                    self.active_trade_info = None
                await asyncio.sleep(1)
        finally:
            for t in self.market_tasks.values():
                t.cancel()
            self.market_tasks.clear()

    async def fetch_real_m1_candles(self, symbol: str):
        data = await self.api.ticks_history({
            "ticks_history": symbol,
            "end": "latest",
            "count": CANDLES_COUNT,
            "style": "candles",
            "granularity": TF_SEC
        })
        return build_candles_from_deriv(data.get("candles", []))

    async def scan_market(self, symbol: str):
        while self.is_scanning:
            try:
                if self.consecutive_losses >= MAX_CONSEC_LOSSES or self.trades_today >= MAX_TRADES_PER_DAY:
                    self.is_scanning = False
                    break

                candles = await self.fetch_real_m1_candles(symbol)
                need = max(EMA_PERIOD + 10, RSI_PERIOD + 10, 60)
                if len(candles) < need:
                    self.market_debug[symbol] = {
                        "time": time.time(),
                        "gate": "Waiting for more candles",
                        "why": [f"Not enough candle history yet (need {need}, have {len(candles)})."],
                    }
                    await asyncio.sleep(SCAN_SLEEP_SEC)
                    continue

                ok_gate, gate = self.can_auto_trade()

                # candles[-1] forming; candles[-2] last closed
                prev2 = candles[-4]          # candle BEFORE cross candle
                cross_candle = candles[-3]   # Candle A (cross candle)
                confirm = candles[-2]        # Candle B (2nd candle after cross)

                cross_t0 = int(cross_candle["t0"])
                confirm_t0 = int(confirm["t0"])

                prev2_close = float(prev2["c"])
                cross_close = float(cross_candle["c"])
                confirm_close = float(confirm["c"])

                confirm_h = float(confirm["h"])
                confirm_l = float(confirm["l"])
                confirm_range = max(1e-9, confirm_h - confirm_l)

                closes = [x["c"] for x in candles]

                ema50_arr = calculate_ema(closes, EMA_PERIOD)
                if len(ema50_arr) < 10:
                    self.market_debug[symbol] = {"time": time.time(), "gate": "Indicators", "why": ["EMA50 not ready yet."]}
                    await asyncio.sleep(SCAN_SLEEP_SEC)
                    continue

                ema_prev2 = float(ema50_arr[-4])
                ema_cross = float(ema50_arr[-3])
                ema_confirm = float(ema50_arr[-2])

                psar_list = calculate_psar(candles, PSAR_AF_STEP, PSAR_AF_MAX)
                if not psar_list or len(psar_list) < 10:
                    self.market_debug[symbol] = {"time": time.time(), "gate": "Indicators", "why": ["PSAR not ready yet."]}
                    await asyncio.sleep(SCAN_SLEEP_SEC)
                    continue
                psar_confirm = float(psar_list[-2])

                rsi_arr = calculate_rsi(closes, RSI_PERIOD)
                if len(rsi_arr) < 10 or np.isnan(rsi_arr[-2]) or np.isnan(rsi_arr[-3]) or np.isnan(rsi_arr[-4]):
                    self.market_debug[symbol] = {"time": time.time(), "gate": "Indicators", "why": ["RSI not ready yet."]}
                    await asyncio.sleep(SCAN_SLEEP_SEC)
                    continue

                rsi_now = float(rsi_arr[-2])    # confirm candle RSI
                rsi_prev = float(rsi_arr[-3])
                rsi_prev2 = float(rsi_arr[-4])

                # ---------------- CROSS EVENT DETECTION ----------------
                bull_cross = (cross_close > ema_cross) and (prev2_close <= ema_prev2)
                bear_cross = (cross_close < ema_cross) and (prev2_close >= ema_prev2)

                if bull_cross and self.last_cross_t0[symbol] != cross_t0:
                    self.last_cross_dir[symbol] = "BULL"
                    self.last_cross_t0[symbol] = cross_t0
                    self.cross_traded[symbol] = False
                elif bear_cross and self.last_cross_t0[symbol] != cross_t0:
                    self.last_cross_dir[symbol] = "BEAR"
                    self.last_cross_t0[symbol] = cross_t0
                    self.cross_traded[symbol] = False

                # If price invalidates the cross on confirm candle, block this cycle
                if self.last_cross_dir[symbol] == "BULL" and confirm_close < ema_confirm:
                    self.cross_traded[symbol] = True
                if self.last_cross_dir[symbol] == "BEAR" and confirm_close > ema_confirm:
                    self.cross_traded[symbol] = True

                # ---------------- CONFIRMATIONS ON 2ND CANDLE ----------------
                psar_below_confirm = psar_confirm < confirm_l
                psar_above_confirm = psar_confirm > confirm_h
                psar_dist = abs(confirm_close - psar_confirm)
                psar_far = psar_dist >= (PSAR_MIN_DIST_MULT * confirm_range)

                # RSI "turn" logic (like your oscillator turn idea)
                rsi_turn_up = (rsi_prev2 > rsi_prev) and (rsi_prev < rsi_now)
                rsi_turn_down = (rsi_prev2 < rsi_prev) and (rsi_prev > rsi_now)

                # Bias filter (keeps it cleaner)
                rsi_call_bias = rsi_now >= 50.0
                rsi_put_bias = rsi_now <= 50.0

                call_ready = (
                    self.last_cross_dir[symbol] == "BULL"
                    and self.last_cross_t0[symbol] == cross_t0
                    and not self.cross_traded[symbol]
                    and (confirm_close > ema_confirm)
                    and psar_below_confirm
                    and psar_far
                    and rsi_call_bias
                    and rsi_turn_up
                )

                put_ready = (
                    self.last_cross_dir[symbol] == "BEAR"
                    and self.last_cross_t0[symbol] == cross_t0
                    and not self.cross_traded[symbol]
                    and (confirm_close < ema_confirm)
                    and psar_above_confirm
                    and psar_far
                    and rsi_put_bias
                    and rsi_turn_down
                )

                # ---------------- STATUS / WHY ----------------
                why = []

                if not ok_gate:
                    why.append(f"Gate blocked: {gate}")

                if self.last_cross_dir[symbol] is None:
                    why.append("Waiting: no EMA50 cross detected yet.")
                else:
                    cross_time = fmt_time_hhmmss(self.last_cross_t0[symbol])
                    why.append(f"Last EMA50 cross: {self.last_cross_dir[symbol]} at {cross_time}.")
                    if self.cross_traded[symbol]:
                        why.append("Already traded this cross (waiting for a NEW cross).")
                    else:
                        why.append("Waiting for 2nd candle confirmation after the cross.")

                # Checklist style (more readable)
                if self.last_cross_dir[symbol] == "BULL" and not self.cross_traded[symbol]:
                    why += [
                        f"{'✅' if confirm_close > ema_confirm else '❌'} Confirm candle HOLDS above EMA50",
                        f"{'✅' if psar_below_confirm else '❌'} PSAR below confirm candle",
                        f"{'✅' if psar_far else '❌'} PSAR far enough (filter)",
                        f"{'✅' if rsi_call_bias else '❌'} RSI >= 50 (bias)",
                        f"{'✅' if rsi_turn_up else '❌'} RSI just turned UP",
                    ]

                if self.last_cross_dir[symbol] == "BEAR" and not self.cross_traded[symbol]:
                    why += [
                        f"{'✅' if confirm_close < ema_confirm else '❌'} Confirm candle HOLDS below EMA50",
                        f"{'✅' if psar_above_confirm else '❌'} PSAR above confirm candle",
                        f"{'✅' if psar_far else '❌'} PSAR far enough (filter)",
                        f"{'✅' if rsi_put_bias else '❌'} RSI <= 50 (bias)",
                        f"{'✅' if rsi_turn_down else '❌'} RSI just turned DOWN",
                    ]

                if call_ready:
                    why = ["✅ CALL READY: EMA50 cross + 2nd candle hold + PSAR far + RSI turned UP (>=50)."]
                elif put_ready:
                    why = ["✅ PUT READY: EMA50 cross + 2nd candle hold + PSAR far + RSI turned DOWN (<=50)."]
                elif not why:
                    why = ["Waiting for conditions."]

                signal = "CALL" if call_ready else "PUT" if put_ready else None

                self.market_debug[symbol] = {
                    "time": time.time(),
                    "gate": gate,
                    "last_closed": confirm_t0,
                    "signal": signal,
                    "why": why[:10],

                    "cross_dir": self.last_cross_dir[symbol],
                    "cross_t0": self.last_cross_t0[symbol],
                    "cross_traded": self.cross_traded[symbol],

                    "c_cross": cross_close,
                    "c_confirm": confirm_close,
                    "ema_confirm": ema_confirm,

                    "psar_confirm": psar_confirm,
                    "psar_far": psar_far,
                    "psar_dist": psar_dist,

                    "rsi_prev2": rsi_prev2,
                    "rsi_prev": rsi_prev,
                    "rsi_now": rsi_now,
                    "rsi_turn": "UP" if rsi_turn_up else "DOWN" if rsi_turn_down else "NO_TURN",
                }

                if not ok_gate:
                    await asyncio.sleep(SCAN_SLEEP_SEC)
                    continue

                if self.last_processed_closed_t0[symbol] == confirm_t0:
                    await asyncio.sleep(SCAN_SLEEP_SEC)
                    continue
                self.last_processed_closed_t0[symbol] = confirm_t0

                if call_ready:
                    self.cross_traded[symbol] = True
                    await self.execute_trade("CALL", symbol, reason="EMA50 CROSS → 2nd candle + PSAR far + RSI turn UP", source="AUTO")
                elif put_ready:
                    self.cross_traded[symbol] = True
                    await self.execute_trade("PUT", symbol, reason="EMA50 CROSS → 2nd candle + PSAR far + RSI turn DOWN", source="AUTO")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scanner Error ({symbol}): {e}")
                self.market_debug[symbol] = {"time": time.time(), "gate": "Error", "why": [str(e)[:140]]}

            await asyncio.sleep(SCAN_SLEEP_SEC)

    async def execute_trade(self, side: str, symbol: str, reason="MANUAL", source="MANUAL"):
        if not self.api or self.active_trade_info:
            return

        async with self.trade_lock:
            ok, _gate = self.can_auto_trade()
            if not ok:
                return

            try:
                stake = float(self.current_stake if source == "AUTO" else BASE_STAKE)

                prop = await self.api.proposal({
                    "proposal": 1,
                    "amount": stake,
                    "basis": "stake",
                    "contract_type": side,
                    "currency": "USD",
                    "duration": int(DURATION_MIN),
                    "duration_unit": "m",
                    "symbol": symbol
                })
                buy = await self.api.buy({"buy": prop["proposal"]["id"], "price": float(prop["proposal"]["ask_price"])})

                self.active_trade_info = int(buy["buy"]["contract_id"])
                self.active_market = symbol
                self.trade_start_time = time.time()

                if source == "AUTO":
                    self.trades_today += 1

                safe_symbol = str(symbol).replace("_", " ")
                msg = (
                    f"🚀 {side} TRADE OPENED (${stake:.2f})\n"
                    f"🛒 Market: {safe_symbol}\n"
                    f"⏱ Expiry: {DURATION_MIN}m\n"
                    f"🧠 Reason: {reason}\n"
                    f"🤖 Source: {source}"
                )
                await self.app.bot.send_message(TELEGRAM_CHAT_ID, msg)
                asyncio.create_task(self.check_result(self.active_trade_info, source))

            except Exception as e:
                logger.error(f"Trade error: {e}")

    async def check_result(self, cid: int, source: str):
        await asyncio.sleep(int(DURATION_MIN) * 60 + 5)
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
                (
                    f"🏁 FINISH: {'WIN' if profit > 0 else 'LOSS'} ({profit:+.2f})\n"
                    f"📊 Today: {self.trades_today}/{MAX_TRADES_PER_DAY} | ❌ Losses: {self.total_losses_today} | Streak: {self.consecutive_losses}/{MAX_CONSEC_LOSSES}\n"
                    f"🧪 Next Stake: ${self.current_stake:.2f}\n"
                    f"💰 Balance: {self.balance}"
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
    cross_dir = d.get("cross_dir") or "—"
    cross_t0 = d.get("cross_t0", 0)
    traded = "YES" if d.get("cross_traded") else "NO"

    why_lines = d.get("why", ["Waiting..."])[:10]
    why = "\n".join([f"• {x}" for x in why_lines])

    if "ema_confirm" not in d:
        return (
            f"📍 {sym.replace('_',' ')} ({age}s)\n"
            f"Gate: {gate}\n"
            f"Last closed: {fmt_time_hhmmss(last_closed)}\n"
            f"Signal: {signal}\n"
            f"Why:\n{why}"
        )

    return (
        f"📍 {sym.replace('_',' ')} ({age}s)\n"
        f"Gate: {gate}\n"
        f"Last closed: {fmt_time_hhmmss(last_closed)}\n"
        f"Signal: {signal}\n"
        f"──────────────\n"
        f"Cross: {cross_dir} @ {fmt_time_hhmmss(cross_t0)} | Traded: {traded}\n"
        f"Price: CrossClose={d.get('c_cross', 0):.2f} | ConfirmClose={d.get('c_confirm', 0):.2f}\n"
        f"EMA50(confirm): {d.get('ema_confirm', 0):.2f}\n"
        f"PSAR(confirm): {d.get('psar_confirm', 0):.2f} | Far: {'YES' if d.get('psar_far') else 'NO'}\n"
        f"RSI: prev2={d.get('rsi_prev2', 0):.2f} → prev1={d.get('rsi_prev', 0):.2f} → now={d.get('rsi_now', 0):.2f}\n"
        f"RSI turn: {d.get('rsi_turn', '—')}\n"
        f"──────────────\n"
        f"Checklist / Notes:\n{why}"
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
        await q.edit_message_text(
            f"🔍 SCANNER ACTIVE\n"
            f"📌 Strategy: EMA50 CROSS (enter on 2nd candle) + PSAR far + RSI turn\n"
            f"🕯 Timeframe: M1\n"
            f"⏱ Expiry: {DURATION_MIN}m",
            reply_markup=main_keyboard()
        )

    elif q.data == "STOP_SCAN":
        bot_logic.is_scanning = False
        await q.edit_message_text("⏹️ Scanner stopped.", reply_markup=main_keyboard())

    elif q.data == "TEST_BUY":
        await bot_logic.execute_trade("CALL", "R_10", "Manual Test", source="MANUAL")

    elif q.data == "STATUS":
        await bot_logic.fetch_balance()
        now_time = datetime.now(ZoneInfo("Africa/Lagos")).strftime("%Y-%m-%d %H:%M:%S")
        _ok, gate = bot_logic.can_auto_trade()

        trade_status = "No Active Trade"
        if bot_logic.active_trade_info and bot_logic.api:
            try:
                res = await bot_logic.api.proposal_open_contract({"proposal_open_contract": 1, "contract_id": bot_logic.active_trade_info})
                pnl = float(res["proposal_open_contract"].get("profit", 0))
                rem = max(0, int(DURATION_MIN * 60) - int(time.time() - bot_logic.trade_start_time))
                icon = "✅ PROFIT" if pnl > 0 else "❌ LOSS" if pnl < 0 else "➖ FLAT"
                mkt_clean = str(bot_logic.active_market).replace("_", " ")
                trade_status = f"🚀 Active Trade ({mkt_clean})\n📈 PnL: {icon} ({pnl:+.2f})\n⏳ Left: {rem}s"
            except:
                trade_status = "🚀 Active Trade: Syncing..."

        header = (
            f"🕒 Time (WAT): {now_time}\n"
            f"🤖 Bot: {'ACTIVE' if bot_logic.is_scanning else 'OFFLINE'} ({bot_logic.account_type})\n"
            f"📌 Strategy: EMA50 CROSS (2nd candle) + PSAR far + RSI turn (M1)\n"
            f"⏱ Expiry: {DURATION_MIN}m | Cooldown: {COOLDOWN_SEC}s\n"
            f"📡 Markets: {', '.join(MARKETS).replace('_',' ')}\n"
            f"━━━━━━━━━━━━━━━\n{trade_status}\n━━━━━━━━━━━━━━━\n"
            f"💵 Total Profit Today: {bot_logic.total_profit_today:+.2f}\n"
            f"🎯 Trades: {bot_logic.trades_today}/{MAX_TRADES_PER_DAY} | ❌ Losses: {bot_logic.total_losses_today}\n"
            f"📉 Loss Streak: {bot_logic.consecutive_losses}/{MAX_CONSEC_LOSSES} | 🧪 Next Stake: ${bot_logic.current_stake:.2f}\n"
            f"🚦 Gate: {gate}\n"
            f"💰 Balance: {bot_logic.balance}\n"
        )

        details = "\n\n📌 LIVE SCAN (Readable)\n\n" + "\n\n".join(
            [format_market_detail(sym, bot_logic.market_debug.get(sym, {})) for sym in MARKETS]
        )

        await q.edit_message_text(header + details, reply_markup=main_keyboard())

async def start_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    await u.message.reply_text(
        "💎 Deriv Bot\n"
        "📌 Strategy: EMA50 CROSS (enter 2nd candle) + PSAR far + RSI turn\n"
        "🕯 Timeframe: M1\n"
        f"⏱ Expiry: {DURATION_MIN}m\n",
        reply_markup=main_keyboard()
    )

if __name__ == "__main__":
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    bot_logic.app = app
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CallbackQueryHandler(btn_handler))
    app.run_polling(drop_pending_updates=True)
