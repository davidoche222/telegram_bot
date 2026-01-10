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
DEMO_TOKEN = "tIrfitLjqeBxCOM"
REAL_TOKEN = "2hsJzopRHG5wUEb"
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
STOP_AFTER_LOSSES = 5  # <--- Still stops ONLY after 5 consecutive losses

MIN_CANDLES_REQUIRED = 220

TELEGRAM_TOKEN = "8276370676:AAGh5VqkG7b4cvpfRIVwY_rtaBlIiNwCTDM"
TELEGRAM_CHAT_ID = "7634818949"
# =========================================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# --- INDICATORS (From First Code) ---
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
            cur_t0, o, h = t0, price, price
            l = c = price
        else:
            h, l, c = max(h, price), min(l, price), price
    candles.append(Candle(cur_t0, o, h, l, c))
    return candles

# ========================= BOT CORE =========================
class DerivSniperBotMerged:
    def __init__(self):
        self.api = None
        self.app = None
        self.active_token = None
        self.account_type = "None"
        self.is_scanning = False
        self.scanner_status = "💤 Offline"
        self.active_contract_id = None
        self.trade_start_time = 0
        self.trades_today = 0
        self.losses_today = 0  # CONSECUTIVE LOSSES
        self.balance = "0.00"
        self.cooldown_until = 0.0
        self.trade_lock = asyncio.Lock()
        self.buy_stage = self.sell_stage = 0
        self._scanner_task = None

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
            self.balance = f"{float(bal['balance']['balance']):.2f} {bal['balance']['currency']}"
        except: pass

    def _signal_logic(self, candles: List[Candle]) -> Optional[str]:
        if len(candles) < MIN_CANDLES_REQUIRED: return None
        c = np.array([x.close for x in candles], dtype=float)
        o = np.array([x.open for x in candles], dtype=float)
        h = np.array([x.high for x in candles], dtype=float)
        l = np.array([x.low for x in candles], dtype=float)

        ema100 = ema(c, EMA_PERIOD)
        slope = ema100[-1] - ema100[-6]
        ps = psar(h, l, PSAR_STEP, PSAR_MAX)
        _, _, hist = macd_hist(c)
        bodies = np.abs(c - o)
        
        # Original Buy/Sell Logic
        if slope > 0 and c[-1] > ema100[-1]:
             if ps[-1] < l[-1] and hist[-1] > hist[-2] and c[-1] > o[-1] and candle_body_strength(o[-1], c[-1], bodies):
                 return "CALL"
        if slope < 0 and c[-1] < ema100[-1]:
             if ps[-1] > h[-1] and hist[-1] < hist[-2] and c[-1] < o[-1] and candle_body_strength(o[-1], c[-1], bodies):
                 return "PUT"
        return None

    async def execute_trade(self, side: str, source="AUTO"):
        async with self.trade_lock:
            if self.active_contract_id or self.losses_today >= STOP_AFTER_LOSSES: return
            try:
                prop = await self.api.proposal({"proposal": 1, "amount": float(STAKE), "basis": "stake", "contract_type": side, "currency": CURRENCY, "duration": int(round(DURATION_MINUTES)), "duration_unit": "m", "symbol": SYMBOL})
                buy = await self.api.buy({"buy": prop["proposal"]["id"], "price": 10.0})
                self.active_contract_id = int(buy["buy"]["contract_id"])
                self.trade_start_time = time.time()
                self.trades_today += 1
                await self.app.bot.send_message(TELEGRAM_CHAT_ID, f"🚀 **{side} TRADE EXECUTED ({source})**\nMarket: {SYMBOL}")
                asyncio.create_task(self.check_result(self.active_contract_id))
            except: pass

    async def check_result(self, cid):
        await asyncio.sleep((CANDLE_SECONDS * EXPIRY_CANDLES) + 8)
        try:
            res = await self.api.proposal_open_contract({"proposal_open_contract": 1, "contract_id": cid})
            profit = float(res["proposal_open_contract"].get("profit", 0.0))
            if profit <= 0: self.losses_today += 1
            else: self.losses_today = 0 # Reset on Win
            
            await self.fetch_balance()
            status_msg = "✅ WIN" if profit > 0 else "❌ LOSS"
            await self.app.bot.send_message(TELEGRAM_CHAT_ID, f"🏁 **TRADE FINISHED**\nResult: {status_msg} (${profit:.2f})\nConsecutive Losses: {self.losses_today}/{STOP_AFTER_LOSSES}")
            
            if self.losses_today >= STOP_AFTER_LOSSES:
                self.is_scanning = False
                self.scanner_status = "🛑 STOPPED: STREAK HIT"
                await self.app.bot.send_message(TELEGRAM_CHAT_ID, f"🛑 **CRITICAL STOP**: {STOP_AFTER_LOSSES} consecutive losses hit.")
        finally:
            self.active_contract_id = None
            self.cooldown_until = time.time() + COOLDOWN_SECONDS

    async def scanner_loop(self):
        if not self.api and not await self.connect(): return
        self.is_scanning = True
        self.scanner_status = "📡 Scanning"
        while self.is_scanning:
            try:
                if not self.active_contract_id and time.time() >= self.cooldown_until:
                    resp = await self.api.ticks_history({"ticks_history": SYMBOL, "end": "latest", "count": 1000, "style": "ticks"})
                    h = resp.get("history", {})
                    ticks = sorted(list(zip(h.get("times", []), h.get("prices", []))), key=lambda x: x[0])
                    sig = self._signal_logic(ticks_to_candles_30s([(int(t), float(p)) for t, p in ticks]))
                    if sig: await self.execute_trade(sig, "AUTO")
            except: pass
            await asyncio.sleep(10)

# ========================= UI (New Style) =========================
bot_logic = DerivSniperBotMerged()

def main_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("▶️ START SCANNER", callback_data="START_SCAN"), InlineKeyboardButton("⏹️ STOP", callback_data="STOP_SCAN")],
        [InlineKeyboardButton("🧪 TEST BUY", callback_data="TEST_BUY"), InlineKeyboardButton("📊 STATUS", callback_data="STATUS")],
        [InlineKeyboardButton("🧪 DEMO", callback_data="SET_DEMO"), InlineKeyboardButton("💰 LIVE", callback_data="SET_REAL")]
    ])

async def btn_handler(u: Update, c: ContextTypes.DEFAULT_TYPE):
    q = u.callback_query
    await q.answer()
    
    if q.data == "STATUS":
        await bot_logic.fetch_balance()
        status_header = f"🤖 **Bot State**: `{bot_logic.scanner_status}`\n🔑 **Account**: `{bot_logic.account_type}`\n"
        
        trade_text = ""
        if bot_logic.active_contract_id:
            try:
                res = await bot_logic.api.proposal_open_contract({"proposal_open_contract": 1, "contract_id": bot_logic.active_contract_id})
                current_pnl = float(res['proposal_open_contract'].get('profit', 0))
                elapsed = int(time.time() - bot_logic.trade_start_time)
                remaining = max(0, int(DURATION_MINUTES * 60) - elapsed)
                trade_text = f"\n⏳ **Live Trade**: `${current_pnl:.2f}`\n⏱️ **Remaining**: `{remaining//60:02d}:{remaining%60:02d}`\n"
            except: trade_text = "\n⏳ **Trade Status**: `Updating...`\n"

        summary = f"\n💰 **Balance**: `{bot_logic.balance}`\n🎯 **Streak**: `{bot_logic.losses_today}/{STOP_AFTER_LOSSES}`"
        await q.edit_message_text(f"📊 **DETAILED STATUS**\n{status_header}{trade_text}{summary}", reply_markup=main_keyboard(), parse_mode="Markdown")

    elif q.data == "START_SCAN":
        if not bot_logic.api:
            await q.edit_message_text("❌ Connect Account First!", reply_markup=main_keyboard())
            return
        bot_logic.is_scanning = True
        asyncio.create_task(bot_logic.scanner_loop())
        await q.edit_message_text("🔍 **SCANNER ACTIVE**\nStrategy: Parabolic SAR + MACD", reply_markup=main_keyboard(), parse_mode="Markdown")

    elif q.data == "SET_DEMO":
        bot_logic.active_token = DEMO_TOKEN
        if await bot_logic.connect():
            bot_logic.account_type = "DEMO"
            await q.edit_message_text(f"✅ Connected to DEMO\nBal: {bot_logic.balance}", reply_markup=main_keyboard())
            
    elif q.data == "SET_REAL":
        bot_logic.active_token = REAL_TOKEN
        if await bot_logic.connect():
            bot_logic.account_type = "LIVE 💰"
            await q.edit_message_text(f"⚠️ **CONNECTED TO LIVE**\nBal: {bot_logic.balance}", reply_markup=main_keyboard(), parse_mode="Markdown")
            
    elif q.data == "TEST_BUY":
        await bot_logic.execute_trade("CALL", "MANUAL-TEST")

    elif q.data == "STOP_SCAN":
        bot_logic.is_scanning = False
        bot_logic.scanner_status = "💤 Offline"
        await q.edit_message_text("🛑 Scanner stopped.", reply_markup=main_keyboard())

if __name__ == "__main__":
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    bot_logic.app = app
    app.add_handler(CommandHandler("start", lambda u, c: u.message.reply_text("💎 **Sniper v5.7 (Live Status Edition)**", reply_markup=main_keyboard())))
    app.add_handler(CallbackQueryHandler(btn_handler))
    app.run_polling(drop_pending_updates=True)
