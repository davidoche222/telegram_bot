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
EXPIRY_CANDLES = 5  # 2.5 minutes
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
MAX_TRADES_PER_DAY = 20
STOP_AFTER_LOSSES = 5  # UPDATED: Stop only after 5 consecutive losses

MIN_CANDLES_REQUIRED = 220

TELEGRAM_TOKEN = "PASTE_TELEGRAM_BOT_TOKEN"
TELEGRAM_CHAT_ID = "PASTE_TELEGRAM_CHAT_ID"
# =========================================================

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
            if low[i] < ps_i:
                up, ps_i, af, ep = False, ep, step, low[i]
            else:
                if high[i] > ep: ep, af = high[i], min(max_step, af + step)
        else:
            ps_i = max(ps_i, high[i - 1], high[i - 2] if i >= 2 else high[i - 1])
            if high[i] > ps_i:
                up, ps_i, af, ep = True, ep, step, high[i]
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
            cur_t0, o, h = t0, price, price
            l = c = price
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
        self.losses_today = 0 # This now tracks CONSECUTIVE losses
        self.pnl_today = 0.0
        self.balance = "0.00"
        self.cooldown_until = 0.0
        self.trade_lock = asyncio.Lock()
        self.buy_stage = 0
        self.sell_stage = 0
        self.last_reason = "Waiting..."
        self.last_scan_time = None
        self.day = datetime.date.today()
        self._scanner_task = None

    def _reset_daily(self):
        today = datetime.date.today()
        if today != self.day:
            self.day, self.trades_today, self.losses_today = today, 0, 0
            self.pnl_today, self.buy_stage, self.sell_stage = 0.0, 0, 0
            self.cooldown_until = 0.0

    async def send(self, text: str):
        if not self.app: return
        try: await self.app.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=text, parse_mode="Markdown")
        except: pass

    async def connect(self) -> bool:
        try:
            self.api = DerivAPI(app_id=APP_ID)
            await self.api.authorize(self.active_token)
            await self.fetch_balance()
            return True
        except: return False

    async def fetch_balance(self):
        if not self.api: return
        try:
            bal = await self.api.balance({"balance": 1})
            b = bal.get("balance", {})
            self.balance = f"{float(b.get('balance', 0.0)):.2f} {b.get('currency', CURRENCY)}"
        except: pass

    async def _get_recent_ticks(self, count: int = 1500) -> List[Tuple[int, float]]:
        resp = await self.api.ticks_history({"ticks_history": SYMBOL, "end": "latest", "count": count, "style": "ticks"})
        h = resp.get("history", {})
        ticks = list(zip(h.get("times", []), h.get("prices", [])))
        ticks.sort(key=lambda x: x[0])
        return [(int(t), float(p)) for t, p in ticks]

    def _signal_logic(self, candles: List[Candle]) -> Optional[str]:
        if len(candles) < MIN_CANDLES_REQUIRED:
            self.last_reason = f"Need more candles"
            return None
        c = np.array([x.close for x in candles], dtype=float)
        o = np.array([x.open for x in candles], dtype=float)
        h = np.array([x.high for x in candles], dtype=float)
        l = np.array([x.low for x in candles], dtype=float)

        ema100 = ema(c, EMA_PERIOD)
        slope = ema100[-1] - ema100[-6]
        slope_thr = max(1e-6, float(np.std(c[-60:])) * 0.001)

        if abs(slope) < slope_thr:
            self.buy_stage = self.sell_stage = 0
            self.last_reason = "Flat EMA (Chop)"
            return None

        ps = psar(h, l, PSAR_STEP, PSAR_MAX)
        _, _, hist = macd_hist(c)
        bodies = np.abs(c - o)
        ps_above, ps_below = ps[-1] > h[-1], ps[-1] < l[-1]
        flipped_to_above = (not (ps[-2] > h[-2])) and ps_above
        flipped_to_below = (not (ps[-2] < l[-2])) and ps_below

        avg_range = max(float(np.mean(h[-20:] - l[-20:])), 1e-6)
        sar_far = abs(ps[-1] - c[-1]) >= 0.6 * avg_range

        # SELL Logic
        if slope < 0 and c[-1] < ema100[-1]:
            if flipped_to_above: self.sell_stage, self.buy_stage = 1, 0
            elif self.sell_stage == 1 and ps_above: self.sell_stage = 2
            elif self.sell_stage == 2:
                if sar_far and hist[-1] < hist[-2] and c[-1] < o[-1] and candle_body_strength(o[-1], c[-1], bodies):
                    return "PUT"
        # BUY Logic
        if slope > 0 and c[-1] > ema100[-1]:
            if flipped_to_below: self.buy_stage, self.sell_stage = 1, 0
            elif self.buy_stage == 1 and ps_below: self.buy_stage = 2
            elif self.buy_stage == 2:
                if sar_far and hist[-1] > hist[-2] and c[-1] > o[-1] and candle_body_strength(o[-1], c[-1], bodies):
                    return "CALL"
        return None

    async def _place_trade(self, side: str):
        async with self.trade_lock:
            if self.active_contract_id: return
            self._reset_daily()
            if self.losses_today >= STOP_AFTER_LOSSES: return
            
            try:
                prop = await self.api.proposal({"proposal": 1, "amount": float(STAKE), "basis": "stake", "contract_type": side, "currency": CURRENCY, "duration": int(round(DURATION_MINUTES)), "duration_unit": "m", "symbol": SYMBOL})
                buy = await self.api.buy({"buy": prop["proposal"]["id"], "price": round(float(prop["proposal"]["ask_price"]) + 0.01, 2)})
                self.active_contract_id = int(buy["buy"]["contract_id"])
                self.trade_open_time = time.time()
                self.trades_today += 1
                self.buy_stage = self.sell_stage = 0
                asyncio.create_task(self._check_result(self.active_contract_id))
            except Exception as e: logger.error(f"Trade Error: {e}")

    async def _check_result(self, cid: int):
        await asyncio.sleep((CANDLE_SECONDS * EXPIRY_CANDLES) + 8)
        try:
            res = await self.api.proposal_open_contract({"proposal_open_contract": 1, "contract_id": cid})
            profit = float(res["proposal_open_contract"].get("profit", 0.0))
            self.pnl_today += profit
            
            # CONSECUTIVE LOSS LOGIC
            if profit <= 0:
                self.losses_today += 1
            else:
                self.losses_today = 0 # Reset on win!
                
            await self.fetch_balance()
            status_msg = "✅ WIN" if profit > 0 else "❌ LOSS"
            await self.send(f"🏁 TRADE CLOSED\nResult: {status_msg}\nConsecutive Losses: {self.losses_today}/{STOP_AFTER_LOSSES}\nBalance: {self.balance}")
            
            if self.losses_today >= STOP_AFTER_LOSSES:
                self.running = False
                await self.send(f"🛑 STOPPED: {STOP_AFTER_LOSSES} Consecutive losses hit.")
        finally:
            self.active_contract_id = None
            self.cooldown_until = time.time() + COOLDOWN_SECONDS

    async def scanner_loop(self):
        if not self.api and not await self.connect(): return
        self.running = True
        while self.running:
            try:
                self._reset_daily()
                if self.active_contract_id or time.time() < self.cooldown_until:
                    await asyncio.sleep(2)
                    continue
                ticks = await self._get_recent_ticks()
                sig = self._signal_logic(ticks_to_candles_30s(ticks))
                if sig: await self._place_trade(sig)
            except: pass
            await asyncio.sleep(3)

    def start(self):
        if not self._scanner_task or self._scanner_task.done():
            self._scanner_task = asyncio.create_task(self.scanner_loop())

    def stop(self): self.running = False

# ========================= TELEGRAM UI =========================
bot_logic = DerivStrictSecondSARBot()

def keyboard():
    return InlineKeyboardMarkup([[InlineKeyboardButton("▶️ START", callback_data="START"), InlineKeyboardButton("⏹️ STOP", callback_data="STOP")], [InlineKeyboardButton("📊 STATUS", callback_data="STATUS")], [InlineKeyboardButton("🧪 DEMO", callback_data="SET_DEMO"), InlineKeyboardButton("💰 LIVE", callback_data="SET_REAL")]])

async def btn_handler(u: Update, c: ContextTypes.DEFAULT_TYPE):
    q = u.callback_query
    await q.answer()
    if q.data == "STATUS":
        live_trade_info = ""
        if bot_logic.active_contract_id:
            try:
                res = await bot_logic.api.proposal_open_contract({"proposal_open_contract": 1, "contract_id": bot_logic.active_contract_id})
                pnl = float(res['proposal_open_contract'].get('profit', 0))
                rem = max(0, int((DURATION_MINUTES * 60) - (time.time() - bot_logic.trade_open_time)))
                live_trade_info = f"\n⏳ **Active Trade**: `${pnl:.2f}`\n⏱️ **Time Left**: `{rem//60:02d}:{rem%60:02d}`"
            except: pass
        
        await bot_logic.fetch_balance()
        msg = f"📊 STATUS\nAccount: {bot_logic.account_type}\nConsecutive Losses: {bot_logic.losses_today}/{STOP_AFTER_LOSSES}\nBalance: {bot_logic.balance}{live_trade_info}"
        await q.edit_message_text(msg, reply_markup=keyboard(), parse_mode="Markdown")
    elif q.data == "START": bot_logic.start()
    elif q.data == "STOP": bot_logic.stop()
    elif "SET_" in q.data:
        bot_logic.active_token = DEMO_TOKEN if "DEMO" in q.data else REAL_TOKEN
        bot_logic.account_type = "DEMO" if "DEMO" in q.data else "LIVE"
        await bot_logic.connect()
        await q.edit_message_text(f"Connected to {bot_logic.account_type}", reply_markup=keyboard())

if __name__ == "__main__":
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    bot_logic.app = app
    app.add_handler(CommandHandler("start", lambda u, c: u.message.reply_text("🤖 Bot Ready", reply_markup=keyboard())))
    app.add_handler(CallbackQueryHandler(btn_handler))
    app.run_polling()
