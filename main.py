import asyncio
import logging
import time
import numpy as np
from datetime import datetime
from decimal import Decimal
from deriv_api import DerivAPI
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

# ========================= CONFIG =========================
DEMO_TOKEN = "tIrfitLjqeBxCOM"
REAL_TOKEN = "2hsJzopRHG5wUEb"
APP_ID = 1089

MARKETS = ["R_10", "R_25", "R_50", "R_75", "R_100"] 
TARGET_PAYOUT = 1.00  
EMA_PERIOD = 100
COOLDOWN_SEC = 300  
MAX_TRADES_PER_DAY = 30
MAX_CONSECUTIVE_LOSSES = 5

TELEGRAM_TOKEN = "8276370676:AAGh5VqkG7b4cvpfRIVwY_rtaBlIiNwCTDM"
TELEGRAM_CHAT_ID = "7634818949"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ========================= CORE BOT CLASS =========================
class DerivSniperBot:
    def __init__(self):
        self.api = None
        self.app = None
        self.active_token = None
        self.is_scanning = False
        self.scanner_status = "💤 Offline"
        self.active_trade_info = None 
        self.cooldown_until = 0
        self.balance = "0.00"
        self.trade_lock = asyncio.Lock()
        
        self.trades_today = 0
        self.consecutive_losses = 0
        self.weekly_wins = 0
        self.weekly_losses = 0
        self.weekly_profit = Decimal('0.00')
        self.current_market = "None"

    async def connect(self):
        try:
            self.api = DerivAPI(app_id=APP_ID)
            await self.api.authorize(self.active_token)
            await self.fetch_balance()
            return True
        except Exception as e:
            logger.error(f"Connect error: {e}")
            return False

    async def fetch_balance(self):
        if not self.api: return
        try:
            bal = await self.api.balance({"balance": 1})
            self.balance = f"{float(bal['balance']['balance']):.2f} {bal['balance']['currency']}"
        except: pass

    async def execute_trade(self, symbol, side, retry=False):
        if not self.api or self.active_trade_info: return
        async with self.trade_lock:
            try:
                # FIX: Market Moved protection
                prop = await self.api.proposal({
                    "proposal": 1, "amount": float(TARGET_PAYOUT), "basis": "payout",
                    "contract_type": side, "currency": "USD", "duration": 150,
                    "duration_unit": "s", "symbol": symbol, "subscribe": 1
                })
                ask_price = float(prop['proposal']['ask_price'])
                # 1.5% Slippage buffer
                max_buy_price = round(ask_price * 1.015, 2)

                buy = await self.api.buy({"buy": prop["proposal"]["id"], "price": max_buy_price})
                self.active_trade_info = buy["buy"]["contract_id"]
                self.trades_today += 1
                
                msg = f"🚀 **{side} TRADE**\nMarket: `{symbol}`\nStake: `${ask_price:.2f}`"
                await self.app.bot.send_message(TELEGRAM_CHAT_ID, msg, parse_mode="Markdown")
                asyncio.create_task(self.check_result(self.active_trade_info))
            except Exception as e:
                if "moved too much" in str(e) and not retry:
                    await asyncio.sleep(0.3)
                    asyncio.create_task(self.execute_trade(symbol, side, retry=True))
                else:
                    logger.error(f"Trade Error: {e}")

    async def check_result(self, cid):
        await asyncio.sleep(160)
        try:
            res = await self.api.proposal_open_contract({"proposal_open_contract": 1, "contract_id": cid})
            p = Decimal(str(res['proposal_open_contract'].get('profit', '0')))
            if p <= 0:
                self.weekly_losses += 1
                self.consecutive_losses += 1
            else:
                self.weekly_wins += 1
                self.consecutive_losses = 0
            await self.fetch_balance()
            status = '✅ WIN' if p > 0 else '❌ LOSS'
            await self.app.bot.send_message(TELEGRAM_CHAT_ID, f"{status} (${float(p):.2f})")
        finally:
            self.active_trade_info = None
            self.cooldown_until = time.time() + COOLDOWN_SEC

    async def background_scanner(self):
        while self.is_scanning:
            if not self.api: await asyncio.sleep(2); continue
            for symbol in MARKETS:
                if not self.is_scanning: break
                self.current_market = symbol
                if self.active_trade_info or time.time() < self.cooldown_until:
                    await asyncio.sleep(1); break
                try:
                    data = await self.api.ticks_history({"ticks_history": symbol, "end": "latest", "count": 600, "style": "ticks"})
                    ticks = list(zip(data['history']['times'], data['history']['prices']))
                    candles = []; curr_t0 = ticks[0][0] - (ticks[0][0] % 30); o = h = l = c = ticks[0][1]
                    for t, p in ticks:
                        t0 = t - (t % 30)
                        if t0 != curr_t0:
                            candles.append({'o':o, 'h':h, 'l':l, 'c':c})
                            curr_t0, o, h, l, c = t0, p, p, p, p
                        else: h, l, c = max(h, p), min(l, p), p
                    if len(candles) < 110: continue
                    ema100, psar, hist, op, hi, lo, cl = calculate_indicators(candles)
                    if cl[-1] > ema100[-1] and hist[-1] > 0 and psar[-1] < lo[-1] and psar[-2] > hi[-2]:
                        await self.execute_trade(symbol, "CALL")
                    elif cl[-1] < ema100[-1] and hist[-1] < 0 and psar[-1] > hi[-1] and psar[-2] < lo[-2]:
                        await self.execute_trade(symbol, "PUT")
                except: pass
                await asyncio.sleep(1)

# ========================= INDICATORS & SETUP =========================
def calculate_indicators(candles):
    c = np.array([x['c'] for x in candles]); h = np.array([x['h'] for x in candles])
    l = np.array([x['l'] for x in candles]); o = np.array([x['o'] for x in candles])
    ema = [c[0]]; k = 2 / (101)
    for price in c[1:]: ema.append(price * k + ema[-1] * (1 - k))
    ema100 = np.array(ema)
    def get_ema(data, p):
        res = [data[0]]; alpha = 2/(p+1)
        for v in data[1:]: res.append(v*alpha + res[-1]*(1-alpha))
        return np.array(res)
    macd_line = get_ema(c, 12) - get_ema(c, 26)
    sig_line = get_ema(macd_line, 9); hist = macd_line - sig_line
    psar = np.zeros(len(c)); up = True; af = 0.02; ep = h[0]; psar[0] = l[0]
    for i in range(1, len(c)):
        psar[i] = psar[i-1] + af * (ep - psar[i-1])
        if up:
            psar[i] = min(psar[i], l[max(0, i-1)], l[max(0, i-2)])
            if l[i] < psar[i]: up = False; psar[i] = ep; af = 0.02; ep = l[i]
            elif h[i] > ep: ep = h[i]; af = min(0.2, af + 0.02)
        else:
            psar[i] = max(psar[i], h[max(0, i-1)], h[max(0, i-2)])
            if h[i] > psar[i]: up = True; psar[i] = ep; af = 0.02; ep = h[i]
            elif l[i] < ep: ep = l[i]; af = min(0.2, af + 0.02)
    return ema100, psar, hist, o, h, l, c

# --- IMPORTANT: DEFINING THE INSTANCE BEFORE THE HANDLERS ---
bot_logic = DerivSniperBot()

def main_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("▶️ START SCANNER", callback_data="START_SCAN"), InlineKeyboardButton("⏹️ STOP", callback_data="STOP_SCAN")],
        [InlineKeyboardButton("🧪 TEST BUY", callback_data="TEST_BUY"), InlineKeyboardButton("📊 STATUS", callback_data="STATUS")],
        [InlineKeyboardButton("🧪 DEMO", callback_data="SET_DEMO"), InlineKeyboardButton("💰 LIVE", callback_data="SET_REAL")]
    ])

async def btn_handler(u: Update, c: ContextTypes.DEFAULT_TYPE):
    q = u.callback_query; await q.answer()
    if q.data == "STATUS":
        await bot_logic.fetch_balance()
        await q.edit_message_text(f"💰 Balance: {bot_logic.balance}\nStat: {bot_logic.scanner_status}", reply_markup=main_keyboard())
    elif q.data == "START_SCAN":
        bot_logic.is_scanning = True
        bot_logic.scanner_status = "📡 Active"
        asyncio.create_task(bot_logic.background_scanner())
    elif q.data == "TEST_BUY":
        await bot_logic.execute_trade("R_10", "CALL")
    elif q.data == "SET_DEMO":
        bot_logic.active_token = DEMO_TOKEN; await bot_logic.connect()
    elif q.data == "SET_REAL":
        bot_logic.active_token = REAL_TOKEN; await bot_logic.connect()
    elif q.data == "STOP_SCAN":
        bot_logic.is_scanning = False

async def start_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    await u.message.reply_text("💎 Sniper Bot v8.8", reply_markup=main_keyboard())

if __name__ == "__main__":
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    bot_logic.app = app
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CallbackQueryHandler(btn_handler))
    # Conflict Fix
    app.run_polling(drop_pending_updates=True)
