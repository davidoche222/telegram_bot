import asyncio
import datetime
import logging
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from deriv_api import DerivAPI
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

# ========================= CONFIG =========================
DEMO_TOKEN = "YYSlMBIcTXqOozU"
REAL_TOKEN = "2NFJTH3JgXWFCcv"
APP_ID = 1089

SYMBOL = "R_10"
CANDLE_SECONDS = 30
EXPIRY_CANDLES = 5  
DURATION_MINUTES = (CANDLE_SECONDS * EXPIRY_CANDLES) / 60.0

STAKE = 0.35
CURRENCY = "USD"

EMA_PERIOD = 100
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

PSAR_STEP = 0.02
PSAR_MAX = 0.2

COOLDOWN_SECONDS = 10 * 60
MAX_TRADES_PER_DAY = 20  # <--- Updated as requested
STOP_AFTER_LOSSES = 5    # <--- Stop after 5 consecutive losses

MIN_CANDLES_REQUIRED = 220

TELEGRAM_TOKEN = "8276370676:AAGh5VqkG7b4cvpfRIVwY_rtaBlIiNwCTDM"
TELEGRAM_CHAT_ID = "7634818949"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ========================= INDICATORS =========================
def ema(values: np.ndarray, period: int) -> np.ndarray:
    if len(values) < period: return np.array([])
    k = 2.0 / (period + 1.0)
    out = np.zeros_like(values, dtype=float)
    out[0] = float(values[0])
    for i in range(1, len(values)):
        out[i] = values[i] * k + out[i - 1] * (1 - k)
    return out

def macd_hist(values: np.ndarray, fast=12, slow=26, signal=9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(values) < slow + signal + 5: return np.array([]), np.array([]), np.array([])
    e_fast = ema(values, fast)
    e_slow = ema(values, slow)
    m = e_fast - e_slow
    sig = ema(m, signal)
    hist = m - sig
    return m, sig, hist

def psar(high: np.ndarray, low: np.ndarray, step=0.02, max_step=0.2) -> np.ndarray:
    n = len(high)
    if n < 5: return np.array([])
    ps = np.zeros(n, dtype=float)
    up = True if high[1] + low[1] > high[0] + low[0] else False
    af, ep = step, (high[0] if up else low[0])
    ps[0] = low[0] if up else high[0]
    for i in range(1, n):
        ps_i = ps[i - 1] + af * (ep - ps[i - 1])
        if up:
            ps_i = min(ps_i, low[i - 1], low[i - 2] if i >= 2 else low[i - 1])
            if low[i] < ps_i: up, ps_i, af, ep = False, ep, step, low[i]
            else:
                if high[i] > ep: ep, af = high[i], min(max_step, af + step)
        else:
            ps_i = max(ps_i, high[i - 1], high[i - 2] if i >= 2 else high[i - 1])
            if high[i] > ps_i: up, ps_i, af, ep = True, ep, step, high[i]
            else:
                if low[i] < ep: ep, af = low[i], min(max_step, af + step)
        ps[i] = ps_i
    return ps

def candle_body_strength(open_: float, close: float, bodies: np.ndarray) -> bool:
    body = abs(close - open_)
    if len(bodies) < 6: return False
    avg5 = float(np.mean(bodies[-6:-1]))
    return body >= avg5 and body > 0.0

@dataclass
class Candle:
    t0: int
    open: float
    high: float
    low: float
    close: float

def align_time(ts: int, step_sec: int) -> int:
    return ts - (ts % step_sec)

def ticks_to_candles_30s(ticks: List[Tuple[int, float]], step_sec: int = 30) -> List[Candle]:
    if not ticks: return []
    candles = []
    cur_t0 = align_time(int(ticks[0][0]), step_sec)
    o = h = l = c = float(ticks[0][1])
    for ts, price in ticks[1:]:
        t0 = align_time(int(ts), step_sec)
        if t0 != cur_t0:
            candles.append(Candle(cur_t0, o, h, l, c))
            cur_t0, o, h, l, c = t0, price, price, price, price
        else:
            h, l, c = max(h, price), min(l, price), price
    candles.append(Candle(cur_t0, o, h, l, c))
    return candles

# ========================= BOT CORE =========================
class DerivStrictSecondSARBot:
    def __init__(self):
        self.api = None
        self.app = None
        self.active_token = None
        self.account_type = "None"
        self.running = False
        self.status = "💤 Offline"
        self.active_contract_id = None
        self.trade_open_time = 0.0
        self.trades_today = 0
        self.losses_today = 0
        self.pnl_today = 0.0
        self.balance = "0.00"
        self.cooldown_until = 0.0
        self.trade_lock = asyncio.Lock()
        self.buy_stage = 0
        self.sell_stage = 0
        self.last_reason = "Waiting…"
        self.last_scan_time = None
        self._scanner_task = None

    async def connect(self) -> bool:
        try:
            self.api = DerivAPI(app_id=APP_ID)
            await self.api.authorize(self.active_token)
            await self.fetch_balance()
            return True
        except Exception: return False

    async def fetch_balance(self):
        if not self.api: return
        try:
            bal = await self.api.balance({"balance": 1})
            b = bal.get("balance", {})
            self.balance = f"{float(b.get('balance', 0.0)):.2f} {b.get('currency', CURRENCY)}"
        except Exception: pass

    def _signal_logic(self, candles: List[Candle]) -> Optional[str]:
        if len(candles) < MIN_CANDLES_REQUIRED:
            self.last_reason = f"Need candles ({len(candles)}/{MIN_CANDLES_REQUIRED})"
            return None

        o = np.array([x.open for x in candles], dtype=float)
        h = np.array([x.high for x in candles], dtype=float)
        l = np.array([x.low for x in candles], dtype=float)
        c = np.array([x.close for x in candles], dtype=float)

        ema100 = ema(c, EMA_PERIOD)
        slope = ema100[-1] - ema100[-6]
        ps = psar(h, l, step=PSAR_STEP, max_step=PSAR_MAX)
        _, _, hist = macd_hist(c)
        bodies = np.abs(c - o)

        ps_above, ps_below = ps[-1] > h[-1], ps[-1] < l[-1]
        f_above, f_below = (not (ps[-2] > h[-2])) and ps_above, (not (ps[-2] < l[-2])) and ps_below
        
        avg_range = max(float(np.mean(h[-20:] - l[-20:])), 1e-6)
        sar_far = abs(ps[-1] - c[-1]) >= 0.6 * avg_range
        macd_bear, macd_bull = (hist[-1] < 0 and hist[-1] < hist[-2]), (hist[-1] > 0 and hist[-1] > hist[-2])

        # SELL logic
        if slope < 0 and c[-1] < ema100[-1]:
            if f_above: self.sell_stage = 1; self.buy_stage = 0; self.last_reason = "SELL: Flip 1 seen"
            elif self.sell_stage == 1 and ps_above: self.sell_stage = 2; self.last_reason = "SELL: Dot 2 confirmed"
            if self.sell_stage == 2 and sar_far and macd_bear and c[-1] < o[-1] and candle_body_strength(o[-1], c[-1], bodies):
                self.last_reason = "SELL: Entry met ✅"; return "PUT"
        else: self.sell_stage = 0

        # BUY logic
        if slope > 0 and c[-1] > ema100[-1]:
            if f_below: self.buy_stage = 1; self.sell_stage = 0; self.last_reason = "BUY: Flip 1 seen"
            elif self.buy_stage == 1 and ps_below: self.buy_stage = 2; self.last_reason = "BUY: Dot 2 confirmed"
            if self.buy_stage == 2 and sar_far and macd_bull and c[-1] > o[-1] and candle_body_strength(o[-1], c[-1], bodies):
                self.last_reason = "BUY: Entry met ✅"; return "CALL"
        else: self.buy_stage = 0
        
        return None

    async def _place_trade(self, side: str, source="AUTO"):
        if not self.api: return
        async with self.trade_lock:
            if self.active_contract_id: return
            if self.losses_today >= STOP_AFTER_LOSSES:
                self.running = False; await self.app.bot.send_message(TELEGRAM_CHAT_ID, "🛑 Stopped: 5 consecutive losses.")
                return
            if self.trades_today >= MAX_TRADES_PER_DAY:
                self.running = False; await self.app.bot.send_message(TELEGRAM_CHAT_ID, "🛑 Daily Limit reached.")
                return
            if source == "AUTO" and time.time() < self.cooldown_until: return

            try:
                prop = await self.api.proposal({"proposal": 1, "amount": float(STAKE), "basis": "stake", "contract_type": side, "currency": CURRENCY, "duration": int(round(DURATION_MINUTES)), "duration_unit": "m", "symbol": SYMBOL})
                buy = await self.api.buy({"buy": prop["proposal"]["id"], "price": float(prop["proposal"]["ask_price"]) + 0.05})
                self.active_contract_id = int(buy["buy"]["contract_id"])
                self.trade_open_time = time.time()
                self.trades_today += 1
                self.buy_stage = 0; self.sell_stage = 0
                await self.app.bot.send_message(TELEGRAM_CHAT_ID, f"🚀 **{side} OPENED ({source})**\nDaily: {self.trades_today}/{MAX_TRADES_PER_DAY}")
                asyncio.create_task(self._check_result(self.active_contract_id))
            except Exception as e: logger.error(f"Trade Error: {e}")

    async def _check_result(self, cid):
        await asyncio.sleep((CANDLE_SECONDS * EXPIRY_CANDLES) + 8)
        try:
            res = await self.api.proposal_open_contract({"proposal_open_contract": 1, "contract_id": cid})
            profit = float(res["proposal_open_contract"].get("profit", 0.0))
            self.pnl_today += profit
            self.losses_today = (self.losses_today + 1) if profit <= 0 else 0
            await self.fetch_balance()
            await self.app.bot.send_message(TELEGRAM_CHAT_ID, f"🏁 **CLOSED**\nResult: {'✅ WIN' if profit > 0 else '❌ LOSS'} (${profit:.2f})\nStreak: {self.losses_today}/5")
        finally:
            self.active_contract_id = None
            self.cooldown_until = time.time() + COOLDOWN_SECONDS

    async def scanner_loop(self):
        self.running = True
        while self.running:
            try:
                if not self.active_contract_id and time.time() >= self.cooldown_until:
                    resp = await self.api.ticks_history({"ticks_history": SYMBOL, "end": "latest", "count": 1800, "style": "ticks"})
                    ticks = sorted(list(zip(resp["history"]["times"], resp["history"]["prices"])), key=lambda x: x[0])
                    sig = self._signal_logic(ticks_to_candles_30s(ticks))
                    if sig: await self._place_trade(sig, "AUTO")
            except Exception as e: logger.error(f"Scanner error: {e}")
            await asyncio.sleep(5)

# ========================= TELEGRAM UI =========================
bot_logic = DerivStrictSecondSARBot()

def keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("▶️ START", callback_data="START"), InlineKeyboardButton("⏹️ STOP", callback_data="STOP")],
        [InlineKeyboardButton("🧪 TEST BUY", callback_data="TEST_BUY"), InlineKeyboardButton("📊 STATUS", callback_data="STATUS")],
        [InlineKeyboardButton("🧪 DEMO", callback_data="SET_DEMO"), InlineKeyboardButton("💰 LIVE", callback_data="SET_REAL")],
    ])

async def btn_handler(u: Update, c: ContextTypes.DEFAULT_TYPE):
    q = u.callback_query
    await q.answer()
    if q.data == "SET_DEMO":
        bot_logic.active_token = DEMO_TOKEN; await bot_logic.connect()
        await q.edit_message_text(f"✅ DEMO Connected\nBalance: {bot_logic.balance}", reply_markup=keyboard())
    elif q.data == "SET_REAL":
        bot_logic.active_token = REAL_TOKEN; await bot_logic.connect()
        await q.edit_message_text(f"⚠️ LIVE Connected\nBalance: {bot_logic.balance}", reply_markup=keyboard())
    elif q.data == "START":
        if not bot_logic.api: await q.message.reply_text("Connect first!"); return
        bot_logic._scanner_task = asyncio.create_task(bot_logic.scanner_loop())
        await q.edit_message_text("✅ Scanner Running", reply_markup=keyboard())
    elif q.data == "STOP":
        bot_logic.running = False; await q.edit_message_text("🛑 Stopped", reply_markup=keyboard())
    elif q.data == "TEST_BUY":
        if not bot_logic.api: await q.message.reply_text("Connect first!")
        else: await bot_logic._place_trade("CALL", "TEST")
    elif q.data == "STATUS":
        await bot_logic.fetch_balance()
        cd = max(0, int(bot_logic.cooldown_until - time.time()))
        msg = f"📊 **STATUS**\nTrades: {bot_logic.trades_today}/20\nLoss Streak: {bot_logic.losses_today}/5\nBalance: {bot_logic.balance}\nCooldown: {cd}s\nStage: B:{bot_logic.buy_stage} S:{bot_logic.sell_stage}\nReason: {bot_logic.last_reason}"
        await q.edit_message_text(msg, reply_markup=keyboard(), parse_mode="Markdown")

if __name__ == "__main__":
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    bot_logic.app = app
    app.add_handler(CommandHandler("start", lambda u, c: u.message.reply_text("🤖 Sniper Bot", reply_markup=keyboard())))
    app.add_handler(CallbackQueryHandler(btn_handler))
    app.run_polling(drop_pending_updates=True)
