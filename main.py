import asyncio
import datetime
import logging
import numpy as np
from deriv_api import DerivAPI
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.error import BadRequest
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

# ========================= 1. CONFIG =========================
DEMO_TOKEN = "YYSlMBIcTXqOozU"
REAL_TOKEN = "2NFJTH3JgXWFCcv"
APP_ID = "1089" 

SYMBOL = "R_100"      # Changed to R_100 for better limits
STAKE = 0.35          # Minimum stake
DURATION = 5          # Increased to 5m to avoid "Max Purchase Price" errors
TELEGRAM_TOKEN = "8276370676:AAGh5VqkG7b4cvpfRIVwY_rtaBlIiNwCTDM"
TELEGRAM_CHAT_ID = "7634818949"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# ========================= 2. STRATEGY ENGINE =========================
def get_ema(values, period=50):
    if len(values) < period: return None
    values = np.array(values, dtype=float)
    k = 2 / (period + 1)
    ema = values[0]
    for p in values[1:]:
        ema = p * k + ema * (1 - k)
    return ema

def get_rsi(prices, period=14):
    if len(prices) < period + 1: return 50
    deltas = np.diff(prices)
    up = deltas[deltas >= 0].sum() / period
    down = -deltas[deltas < 0].sum() / period
    if down == 0: return 100
    rs = up / down
    return 100.0 - (100.0 / (1.0 + rs))

def check_v3_momentum(candle, prev_3, direction, ema):
    try:
        o, h, l, c = float(candle['open']), float(candle['high']), float(candle['low']), float(candle['close'])
        rng = h - l
        if rng <= 0: return False
        
        # Rule: Candle must be touching or crossing the EMA
        if not (l <= ema <= h): return False
        
        # Rule: Candle body must be significant
        avg_body = sum([abs(float(x['close']) - float(x['open'])) for x in prev_3]) / 3
        if abs(c - o) <= avg_body: return False
        
        if direction == "CALL":
            return (h - c) / rng <= 0.30 and c > float(prev_3[-1]['high']) and c > o
        else: 
            return (c - l) / rng <= 0.30 and c < float(prev_3[-1]['low']) and c < o
    except: return False

# ========================= 3. BOT CORE =========================
class DerivSniperBot:
    def __init__(self):
        self.api = None
        self.running = False
        self.account_mode = "Disconnected"
        self.active_token = None
        self.current_status = "🛑 Stopped"
        self.active_trade_info = None  
        self.balance = "0.00"
        self.wins_today = 0
        self.losses_today = 0
        self.pnl_today = 0.0
        self.app = None

    async def connect(self):
        try:
            self.api = DerivAPI(app_id=APP_ID)
            await self.api.authorize(self.active_token)
            return True
        except: return False

    async def update_balance(self):
        if self.api:
            try:
                info = await self.api.authorize(self.active_token)
                self.balance = info['authorize']['balance']
            except: pass

    async def run_scanner(self):
        self.running = True
        while self.running:
            try:
                if not self.api: await self.connect()
                if self.active_trade_info:
                    self.current_status = f"🚀 ACTIVE: {self.active_trade_info['side']}"
                    await asyncio.sleep(10)
                    continue

                # Data Fetching using proper ticks_history
                m5_resp = await self.api.ticks_history({"ticks_history": SYMBOL, "count": 60, "granularity": 300, "style": "candles"})
                m1_resp = await self.api.ticks_history({"ticks_history": SYMBOL, "count": 60, "granularity": 60, "style": "candles"})
                
                m5_c = [float(x['close']) for x in m5_resp['candles']]
                m1_c = [float(x['close']) for x in m1_resp['candles']]
                m1_ema = get_ema(m1_c, 50)
                m5_ema = get_ema(m5_c, 50)
                rsi = get_rsi(m1_c, 14)

                self.current_status = f"🔎 Scanning (RSI: {round(rsi)})"

                m5_bias = "CALL" if m5_c[-1] > m5_ema else "PUT"
                if 30 <= rsi <= 70:
                    last_c, prev_3 = m1_resp['candles'][-1], m1_resp['candles'][-4:-1]
                    signal = None
                    if m5_bias == "CALL" and m1_c[-1] > m1_ema:
                        if check_v3_momentum(last_c, prev_3, "CALL", m1_ema): signal = "CALL"
                    elif m5_bias == "PUT" and m1_c[-1] < m1_ema:
                        if check_v3_momentum(last_c, prev_3, "PUT", m1_ema): signal = "PUT"
                    
                    if signal: await self.execute_trade(signal)
                await asyncio.sleep(10)
            except Exception as e:
                logging.error(f"Loop Error: {e}")
                await asyncio.sleep(10)

    async def execute_trade(self, side):
        try:
            req = {"buy": 1, "price": STAKE, "parameters": {"amount": STAKE, "basis": "stake", 
                   "contract_type": side, "currency": "USD", "duration": DURATION, "duration_unit": "m", "symbol": SYMBOL}}
            resp = await self.api.buy(req)
            
            if 'error' in resp:
                err = resp['error']['message']
                await self.app.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=f"❌ **Trade Refused:**\n{err}")
                return

            self.active_trade_info = {"side": side, "id": resp['buy']['contract_id'], "start": datetime.datetime.now().strftime("%H:%M:%S")}
            await self.app.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=f"🚀 **TRADE PLACED**\nSide: `{side}`\nStake: `${STAKE}`", parse_mode="Markdown")
            asyncio.create_task(self.check_result(self.active_trade_info['id']))
        except Exception as e:
            await self.app.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=f"⚠️ **Trade Error:** {e}")

    async def check_result(self, cid):
        await asyncio.sleep(DURATION * 60 + 5)
        try:
            table = await self.api.profit_table({"limit": 5})
            for trans in table['profit_table']['transactions']:
                if str(trans['contract_id']) == str(cid):
                    profit = float(trans['sell_price']) - float(trans['buy_price'])
                    self.pnl_today += profit
                    res = "✅ WIN" if profit > 0 else "❌ LOSS"
                    if profit > 0: self.wins_today += 1
                    else: self.losses_today += 1
                    await self.update_balance()
                    await self.app.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=f"🏁 **RESULT**: {res}\nProfit: `${round(profit, 2)}`", parse_mode="Markdown")
            self.active_trade_info = None 
        except: self.active_trade_info = None

# ========================= 4. TELEGRAM UI =========================
bot = DerivSniperBot()

async def start_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    kb = [
        [InlineKeyboardButton("🚀 START SCANNER", callback_data="PROMPT_MODE")],
        [InlineKeyboardButton("📊 STATUS", callback_data="STATUS"), InlineKeyboardButton("💰 BALANCE", callback_data="CHECK_BAL")],
        [InlineKeyboardButton("🧪 TEST BUY (CALL)", callback_data="TEST_BUY")],
        [InlineKeyboardButton("🛑 STOP", callback_data="STOP")]
    ]
    text = "💎 **Deriv Sniper v3**\nLogic: M5 Trend + M1 EMA Bounce."
    if u.message: await u.message.reply_text(text, reply_markup=InlineKeyboardMarkup(kb), parse_mode="Markdown")
    else: await u.callback_query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(kb), parse_mode="Markdown")

async def btn_handler(u: Update, c: ContextTypes.DEFAULT_TYPE):
    q = u.callback_query
    await q.answer()
    
    if q.data == "PROMPT_MODE":
        kb = [[InlineKeyboardButton("🧪 DEMO", callback_data="SET_DEMO"), InlineKeyboardButton("💰 LIVE", callback_data="SET_REAL")], [InlineKeyboardButton("⬅️ BACK", callback_data="BACK")]]
        await q.edit_message_text("💳 **Select Account:**", reply_markup=InlineKeyboardMarkup(kb))

    elif q.data in ["SET_DEMO", "SET_REAL"]:
        bot.active_token = DEMO_TOKEN if q.data == "SET_DEMO" else REAL_TOKEN
        bot.account_mode = "🧪 DEMO" if q.data == "SET_DEMO" else "💰 REAL"
        bot.running = False
        await asyncio.sleep(1) 
        if await bot.connect():
            await bot.update_balance()
            asyncio.create_task(bot.run_scanner())
            await q.edit_message_text(f"✅ **Scanner Online**\nAccount: {bot.account_mode}\nBalance: ${bot.balance}", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("📊 STATUS", callback_data="STATUS")]]))
        else: await q.edit_message_text("❌ Connection Error.")

    elif q.data == "TEST_BUY":
        if not bot.api:
            await q.edit_message_text("❌ Start scanner first!", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ BACK", callback_data="BACK")]]))
        else:
            await bot.execute_trade("CALL")

    elif q.data == "STATUS":
        pnl_str = f"+${round(bot.pnl_today, 2)}" if bot.pnl_today >= 0 else f"-${round(abs(bot.pnl_today), 2)}"
        msg = f"📊 **DASHBOARD**\nAccount: `{bot.account_mode}`\nState: `{bot.current_status}`\nPnL: `{pnl_str}`\nWins: {bot.wins_today} | Loss: {bot.losses_today}"
        kb = [[InlineKeyboardButton("🔄 Refresh", callback_data="STATUS")], [InlineKeyboardButton("⬅️ BACK", callback_data="BACK")]]
        try: await q.edit_message_text(msg, reply_markup=InlineKeyboardMarkup(kb), parse_mode="Markdown")
        except BadRequest: pass

    elif q.data == "CHECK_BAL":
        await bot.update_balance()
        await q.edit_message_text(f"💰 **Balance**: `${bot.balance}`", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ BACK", callback_data="BACK")]]))

    elif q.data == "STOP":
        bot.running = False
        bot.current_status = "🛑 Stopped"
        await q.edit_message_text("🛑 Bot Paused.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🚀 RESTART", callback_data="PROMPT_MODE")]]))

    elif q.data == "BACK":
        await start_cmd(u, c)

def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    bot.app = app
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CallbackQueryHandler(btn_handler))
    app.run_polling()

if __name__ == "__main__": main()
