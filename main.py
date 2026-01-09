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

SYMBOL = "R_10"
STAKE = 0.35
DURATION = 3
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
    o, h, l, c = candle['open'], candle['high'], candle['low'], candle['close']
    rng = h - l
    if rng <= 0: return False
    if not (l <= ema <= h): return False
    avg_body = sum([abs(x['close'] - x['open']) for x in prev_3]) / 3
    if abs(c - o) <= avg_body: return False
    if direction == "CALL":
        return (h - c) / rng <= 0.30 and c > prev_3[-1]['high'] and c > o
    else: 
        return (c - l) / rng <= 0.30 and c < prev_3[-1]['low'] and c < o

# ========================= 3. BOT CORE =========================
class DerivSniperBot:
    def __init__(self):
        self.api = None
        self.running = False
        self.account_mode = "Disconnected"
        self.active_token = None
        self.current_status = "🛑 Stopped" # Default state
        self.active_trade_info = None  
        self.balance = "0.00"
        self.trades_today = 0
        self.wins_today = 0
        self.losses_today = 0
        self.pnl_today = 0.0
        self.app = None

    async def notify(self, text):
        if self.app:
            try: await self.app.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=text, parse_mode="Markdown")
            except: pass

    async def connect(self, token):
        try:
            self.api = DerivAPI(app_id=APP_ID)
            await self.api.authorize(token)
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
        # IMMEDIATE STATE UPDATE
        self.current_status = "🔎 Starting Scanner..." 
        
        while self.running:
            try:
                # If a trade is active, status shows the trade
                if self.active_trade_info:
                    self.current_status = f"🚀 ACTIVE: {self.active_trade_info['side']}"
                    await asyncio.sleep(5)
                    continue

                # Analysis logic
                m5_data = await self.api.get_candles({"ticks_history": SYMBOL, "granularity": 300, "count": 60})
                m5_c = [x['close'] for x in m5_data['candles']]
                m5_ema = get_ema(m5_c, 50)
                m5_bias = "CALL" if m5_c[-1] > m5_ema else "PUT"

                m1_data = await self.api.get_candles({"ticks_history": SYMBOL, "granularity": 60, "count": 60})
                m1_c = [x['close'] for x in m1_data['candles']]
                m1_ema = get_ema(m1_c, 50)
                rsi = get_rsi(m1_c, 14)

                # UPDATE STATUS EVERY LOOP
                self.current_status = f"🔎 Searching (RSI: {round(rsi)})"

                if 40 <= rsi <= 60:
                    last_c = m1_data['candles'][-1]
                    prev_3 = m1_data['candles'][-4:-1]
                    
                    signal = None
                    if m5_bias == "CALL" and m1_c[-1] > m1_ema:
                        if check_v3_momentum(last_c, prev_3, "CALL", m1_ema): signal = "CALL"
                    elif m5_bias == "PUT" and m1_c[-1] < m1_ema:
                        if check_v3_momentum(last_c, prev_3, "PUT", m1_ema): signal = "PUT"

                    if signal:
                        await self.execute_trade(signal)

                await asyncio.sleep(5) # Faster scanning
            except Exception as e:
                self.current_status = f"⚠️ Error: {str(e)[:20]}"
                await asyncio.sleep(5)

    async def execute_trade(self, side):
        try:
            req = {"buy": 1, "price": STAKE, "parameters": {"amount": STAKE, "basis": "stake", 
                   "contract_type": side, "currency": "USD", "duration": DURATION, "duration_unit": "m", "symbol": SYMBOL}}
            resp = await self.api.buy(req)
            
            self.trades_today += 1
            self.active_trade_info = {
                "side": side,
                "id": resp['buy']['contract_id'],
                "start": datetime.datetime.now().strftime("%H:%M:%S")
            }
            
            await self.notify(f"🚀 **AUTO-TRADE OPENED**\nSide: `{side}`\nID: `{self.active_trade_info['id']}`")
            asyncio.create_task(self.check_result(self.active_trade_info['id']))
        except Exception as e:
            logging.error(f"Trade Execution Failed: {e}")

    async def check_result(self, cid):
        await asyncio.sleep(DURATION * 60 + 5)
        try:
            table = await self.api.get_profit_table({"limit": 5})
            for trans in table['profit_table']['transactions']:
                if str(trans['contract_id']) == str(cid):
                    profit = float(trans['sell_price']) - float(trans['buy_price'])
                    self.pnl_today += profit
                    res = "✅ WIN" if profit > 0 else "❌ LOSS"
                    if profit > 0: self.wins_today += 1
                    else: self.losses_today += 1
                    
                    await self.update_balance()
                    await self.notify(f"🏁 **RESULT**\nOutcome: {res}\nProfit: `${round(profit, 2)}`")
                    break
            self.active_trade_info = None 
        except:
            self.active_trade_info = None

# ========================= 4. TELEGRAM UI =========================
bot = DerivSniperBot()

async def start_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    kb = [[InlineKeyboardButton("🚀 START SCANNER", callback_data="PROMPT_MODE")],
          [InlineKeyboardButton("📊 STATUS", callback_data="STATUS"), InlineKeyboardButton("💰 BALANCE", callback_data="CHECK_BAL")],
          [InlineKeyboardButton("🛑 STOP", callback_data="STOP")]]
    text = "💎 **Deriv Sniper v3**\nBot initialized and ready."
    if u.message: await u.message.reply_text(text, reply_markup=InlineKeyboardMarkup(kb), parse_mode="Markdown")
    else: await u.callback_query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(kb), parse_mode="Markdown")

async def btn_handler(u: Update, c: ContextTypes.DEFAULT_TYPE):
    q = u.callback_query
    await q.answer()
    
    if q.data == "PROMPT_MODE":
        kb = [[InlineKeyboardButton("🧪 DEMO", callback_data="SET_DEMO"), InlineKeyboardButton("💰 LIVE", callback_data="SET_REAL")],
              [InlineKeyboardButton("⬅️ BACK", callback_data="BACK")]]
        await q.edit_message_text("💳 **Mode Selection:**", reply_markup=InlineKeyboardMarkup(kb), parse_mode="Markdown")

    elif q.data in ["SET_DEMO", "SET_REAL"]:
        is_demo = q.data == "SET_DEMO"
        bot.active_token = DEMO_TOKEN if is_demo else REAL_TOKEN
        bot.account_mode = "🧪 DEMO" if is_demo else "💰 REAL"
        
        # Ensure the previous loop is killed before starting a new one
        bot.running = False
        await asyncio.sleep(1) 
        
        if await bot.connect(bot.active_token):
            await bot.update_balance()
            # This triggers the background loop
            asyncio.create_task(bot.run_scanner())
            await q.edit_message_text(f"✅ **Scanner Online**\nAccount: {bot.account_mode}\n\n*Watching for signals...*", 
                                      reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("📊 STATUS", callback_data="STATUS")]]), parse_mode="Markdown")
        else:
            await q.edit_message_text("❌ Connection Error.")

    elif q.data == "STATUS":
        pnl_str = f"+${round(bot.pnl_today, 2)}" if bot.pnl_today >= 0 else f"-${round(abs(bot.pnl_today), 2)}"
        
        msg = (f"📊 **BOT DASHBOARD**\n"
               f"------------------\n"
               f"Account: `{bot.account_mode}`\n"
               f"State: `{bot.current_status}`\n"
               f"Wins/Loss: {bot.wins_today}W - {bot.losses_today}L\n"
               f"Total PnL: `{pnl_str}`")
        
        kb = [[InlineKeyboardButton("🔄 Refresh", callback_data="STATUS")], [InlineKeyboardButton("⬅️ BACK", callback_data="BACK")]]
        try: await q.edit_message_text(msg, reply_markup=InlineKeyboardMarkup(kb), parse_mode="Markdown")
        except BadRequest: pass

    elif q.data == "CHECK_BAL":
        await bot.update_balance()
        await q.edit_message_text(f"💰 **Balance**: `${bot.balance}`", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ BACK", callback_data="BACK")]]))

    elif q.data == "STOP":
        bot.running = False
        bot.current_status = "🛑 Stopped"
        await q.edit_message_text("🛑 Bot Stopped.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🚀 RESTART", callback_data="PROMPT_MODE")]]))

    elif q.data == "BACK":
        await start_cmd(u, c)

def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    bot.app = app
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CallbackQueryHandler(btn_handler))
    app.run_polling()

if __name__ == "__main__": main()
