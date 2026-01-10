import asyncio
import logging
import time
import numpy as np
from datetime import datetime
from deriv_api import DerivAPI
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

# ========================= CONFIG =========================
DEMO_TOKEN = "tIrfitLjqeBxCOM"
REAL_TOKEN = "2hsJzopRHG5wUEb"
APP_ID = 1089

MARKET = "R_10"
CANDLE_SEC = 30
EXPIRY_CANDLES = 5
DURATION_MINUTES = (CANDLE_SEC * EXPIRY_CANDLES) / 60.0 # 2.5m

STAKE = 0.35
EMA_PERIOD = 100
COOLDOWN_SEC = 600 # 10 Minutes

TELEGRAM_TOKEN = "8276370676:AAGh5VqkG7b4cvpfRIVwY_rtaBlIiNwCTDM"
TELEGRAM_CHAT_ID = "7634818949"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ========================= MATH & INDICATORS =========================
def get_ema(values, period):
    if len(values) < period: return np.array([])
    return np.array([sum(values[i-period:i])/period if i==period else 0 for i in range(len(values))]) # Simple init then EMA logic below

def calculate_indicators(candles):
    c = np.array([x['c'] for x in candles])
    h = np.array([x['h'] for x in candles])
    l = np.array([x['l'] for x in candles])
    o = np.array([x['o'] for x in candles])
    
    # EMA 100
    ema_vals = [c[0]]
    k = 2 / (EMA_PERIOD + 1)
    for i in range(1, len(c)):
        ema_vals.append(c[i] * k + ema_vals[-1] * (1 - k))
    ema100 = np.array(ema_vals)

    # MACD (12, 26, 9)
    def ema_func(data, p):
        res = [data[0]]
        alpha = 2/(p+1)
        for val in data[1:]: res.append(val*alpha + res[-1]*(1-alpha))
        return np.array(res)
    
    macd_line = ema_func(c, 12) - ema_func(c, 26)
    signal_line = ema_func(macd_line, 9)
    hist = macd_line - signal_line

    # PSAR (0.02, 0.2)
    psar = np.zeros(len(c))
    up = True; af = 0.02; ep = h[0]; psar[0] = l[0]
    for i in range(1, len(c)):
        psar[i] = psar[i-1] + af * (ep - psar[i-1])
        if up:
            psar[i] = min(psar[i], l[i-1], l[max(0, i-2)])
            if l[i] < psar[i]: up = False; psar[i] = ep; af = 0.02; ep = l[i]
            elif h[i] > ep: ep = h[i]; af = min(0.2, af + 0.02)
        else:
            psar[i] = max(psar[i], h[i-1], h[max(0, i-2)])
            if h[i] > psar[i]: up = True; psar[i] = ep; af = 0.02; ep = h[i]
            elif l[i] < ep: ep = l[i]; af = min(0.2, af + 0.02)
            
    return ema100, psar, hist, o, h, l, c

# ========================= BOT CORE =========================
class DerivSniperBot:
    def __init__(self):
        self.api = None
        self.app = None
        self.active_token = None
        self.account_type = "None"
        self.is_scanning = False
        self.scanner_status = "💤 Offline"
        
        self.active_trade_id = None
        self.cooldown_until = 0
        self.last_reason = "Waiting for data..."
        
        # Stages
        self.buy_stage = 0
        self.sell_stage = 0
        
        # Stats
        self.trades_today = 0
        self.consecutive_losses = 0
        self.pnl_today = 0.0
        self.balance = "0.00"
        self.trade_lock = asyncio.Lock()

    async def connect(self):
        try:
            self.api = DerivAPI(app_id=APP_ID)
            await self.api.authorize(self.active_token)
            await self.fetch_balance()
            return True
        except: return False

    async def fetch_balance(self):
        try:
            bal = await self.api.balance({"balance": 1})
            self.balance = f"{float(bal['balance']['balance']):.2f} {bal['balance']['currency']}"
        except: pass

    async def background_scanner(self):
        while self.is_scanning:
            if self.consecutive_losses >= 5:
                self.is_scanning = False; self.scanner_status = "🛑 STOPPED (5 LOSSES)"
                await self.app.bot.send_message(TELEGRAM_CHAT_ID, "🛑 **Bot Stopped**: 5 consecutive losses reached.")
                break
            
            if self.trades_today >= 20:
                self.is_scanning = False; self.scanner_status = "🛑 DAILY LIMIT (20)"
                break

            if self.active_trade_id or time.time() < self.cooldown_until:
                await asyncio.sleep(5); continue

            try:
                # 1. Get Ticks and build 30s Candles
                resp = await self.api.ticks_history({"ticks_history": MARKET, "end": "latest", "count": 2000, "style": "ticks"})
                ticks = list(zip(resp['history']['times'], resp['history']['prices']))
                
                candles = []
                curr_t0 = ticks[0][0] - (ticks[0][0] % 30)
                o = h = l = c = ticks[0][1]
                for t, p in ticks:
                    t0 = t - (t % 30)
                    if t0 != curr_t0:
                        candles.append({'o':o, 'h':h, 'l':l, 'c':c})
                        curr_t0, o, h, l, c = t0, p, p, p, p
                    else:
                        h, l, c = max(h, p), min(l, p), p
                
                if len(candles) < 110: continue

                # 2. Indicators
                ema100, psar, hist, op, hi, lo, cl = calculate_indicators(candles)
                
                # 3. Trend & Slope
                slope = ema100[-1] - ema100[-6]
                slope_threshold = max(1e-5, np.std(cl[-20:]) * 0.01)
                
                # 4. Strength Filters
                avg_range = np.mean(hi[-20:] - lo[-20:])
                sar_dist = abs(psar[-1] - cl[-1])
                body = abs(cl[-1] - op[-1])
                avg_body = np.mean(np.abs(cl[-6:-1] - op[-6:-1]))

                # 5. SAR Flip Logic
                ps_above = psar[-1] > hi[-1]
                ps_below = psar[-1] < lo[-1]
                prev_ps_above = psar[-2] > hi[-2]
                prev_ps_below = psar[-2] < lo[-2]

                # --- SELL (PUT) LOGIC ---
                if slope < -slope_threshold and cl[-1] < ema100[-1]:
                    if (not prev_ps_above) and ps_above:
                        self.sell_stage = 1; self.buy_stage = 0; self.last_reason = "SELL: Stage 1 (1st Dot)"
                    elif self.sell_stage == 1 and ps_above:
                        self.sell_stage = 2; self.last_reason = "SELL: Stage 2 (Confirmed)"
                    
                    if self.sell_stage == 2:
                        if sar_dist >= 0.6 * avg_range and hist[-1] < 0 and hist[-1] < hist[-2] and cl[-1] < op[-1] and body >= avg_body:
                            await self.execute_trade("PUT"); self.sell_stage = 0
                        else: self.last_reason = "SELL: Waiting for MACD/Strong Candle"
                else: self.sell_stage = 0

                # --- BUY (CALL) LOGIC ---
                if slope > slope_threshold and cl[-1] > ema100[-1]:
                    if (not prev_ps_below) and ps_below:
                        self.buy_stage = 1; self.sell_stage = 0; self.last_reason = "BUY: Stage 1 (1st Dot)"
                    elif self.buy_stage == 1 and ps_below:
                        self.buy_stage = 2; self.last_reason = "BUY: Stage 2 (Confirmed)"
                    
                    if self.buy_stage == 2:
                        if sar_dist >= 0.6 * avg_range and hist[-1] > 0 and hist[-1] > hist[-2] and cl[-1] > op[-1] and body >= avg_body:
                            await self.execute_trade("CALL"); self.buy_stage = 0
                        else: self.last_reason = "BUY: Waiting for MACD/Strong Candle"
                else: self.buy_stage = 0

                self.scanner_status = "📡 Scanning..."
            except Exception as e: logger.error(f"Scanner Error: {e}")
            await asyncio.sleep(10)

    async def execute_trade(self, side):
        async with self.trade_lock:
            try:
                prop = await self.api.proposal({"proposal": 1, "amount": STAKE, "basis": "stake", "contract_type": side, "currency": "USD", "duration": int(EXPIRY_CANDLES * 30), "duration_unit": "s", "symbol": MARKET})
                buy = await self.api.buy({"buy": prop["proposal"]["id"], "price": float(prop["proposal"]["ask_price"]) + 0.01})
                self.active_trade_id = buy["buy"]["contract_id"]
                self.trades_today += 1
                await self.app.bot.send_message(TELEGRAM_CHAT_ID, f"🚀 **{side} Entered** (Second SAR Confirmed)\nDaily Trade: {self.trades_today}/20")
                asyncio.create_task(self.check_result(self.active_trade_id))
            except Exception as e: logger.error(f"Trade Error: {e}")

    async def check_result(self, cid):
        await asyncio.sleep(160)
        try:
            res = await self.api.proposal_open_contract({"proposal_open_contract": 1, "contract_id": cid})
            profit = float(res['proposal_open_contract'].get('profit', 0))
            self.pnl_today += profit
            if profit <= 0: self.consecutive_losses += 1
            else: self.consecutive_losses = 0
            
            await self.fetch_balance()
            await self.app.bot.send_message(TELEGRAM_CHAT_ID, f"🏁 **Finish**: {'✅ WIN' if profit > 0 else '❌ LOSS'} (${profit:.2f})\nStreak: {self.consecutive_losses}/5")
        finally:
            self.active_trade_id = None
            self.cooldown_until = time.time() + COOLDOWN_SEC

# ========================= UI =========================
bot_logic = DerivSniperBot()

def main_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("▶️ START", callback_data="START"), InlineKeyboardButton("⏹️ STOP", callback_data="STOP")],
        [InlineKeyboardButton("📊 STATUS", callback_data="STATUS")],
        [InlineKeyboardButton("🧪 DEMO", callback_data="DEMO"), InlineKeyboardButton("💰 LIVE", callback_data="REAL")]
    ])

async def btn_handler(u: Update, c: ContextTypes.DEFAULT_TYPE):
    q = u.callback_query; await q.answer()
    if q.data == "STATUS":
        cd = max(0, int(bot_logic.cooldown_until - time.time()))
        msg = (f"📊 **STATUS**\nState: `{bot_logic.scanner_status}`\n"
               f"Account: `{bot_logic.account_type}`\n"
               f"Stages: B:{bot_logic.buy_stage} S:{bot_logic.sell_stage}\n"
               f"Cooldown: `{cd}s`\n"
               f"Last Reason: `{bot_logic.last_reason}`\n"
               f"Today: {bot_logic.trades_today}/20 | Streak: {bot_logic.consecutive_losses}/5\n"
               f"PnL: ${bot_logic.pnl_today:.2f} | Bal: {bot_logic.balance}")
        await q.edit_message_text(msg, reply_markup=main_keyboard(), parse_mode="Markdown")
    elif q.data == "START":
        if not bot_logic.api: await q.message.reply_text("Connect first!"); return
        bot_logic.is_scanning = True; asyncio.create_task(bot_logic.background_scanner())
        await q.edit_message_text("🔍 **Scanner Started**", reply_markup=main_keyboard())
    elif q.data == "STOP":
        bot_logic.is_scanning = False; await q.edit_message_text("🛑 **Stopped**", reply_markup=main_keyboard())
    elif q.data == "DEMO":
        bot_logic.active_token = DEMO_TOKEN; await bot_logic.connect(); bot_logic.account_type = "DEMO"
        await q.edit_message_text(f"Connected DEMO. Bal: {bot_logic.balance}", reply_markup=main_keyboard())
    elif q.data == "REAL":
        bot_logic.active_token = REAL_TOKEN; await bot_logic.connect(); bot_logic.account_type = "REAL"
        await q.edit_message_text(f"Connected REAL. Bal: {bot_logic.balance}", reply_markup=main_keyboard())

if __name__ == "__main__":
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    bot_logic.app = app
    app.add_handler(CommandHandler("start", lambda u,c: u.message.reply_text("🤖 Second SAR Bot", reply_markup=main_keyboard())))
    app.add_handler(CallbackQueryHandler(btn_handler))
    app.run_polling()
