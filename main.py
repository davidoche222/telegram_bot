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
        self.account_mode = "None"
        self.active_token = None
        self.current_status = "🛑 Stopped"
        self.active_trade_msg = "None"
        self.balance = "0.00"
        self.trades_today = 0
        self.wins_today = 0
        self.losses_today = 0
        self.pnl_today = 0.0
        self.last_trade_time = None
        self.last_ema_level = 0
        self.app = None

    async def notify(self, text):
        if self.app:
            await self.app.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=text, parse_mode="Markdown")

    async def connect(self, token):
        try:
            self.api = DerivAPI(app_id=APP_ID)
            account_info = await self.api.authorize(token)
            self.balance = account_info['authorize']['balance']
            return True
        except: return False

    async def update_balance(self):
        if self.api and self.active_token:
            try:
                account_info = await self.api.authorize(self.active_token)
                self.balance = account_info['authorize']['balance']
            except: pass

    async def run_scanner(self):
        self.running = True
        while self.running:
            try:
                if self.trades_today >= 2:
                    self.current_status = "🛑 Limit Reached"
                    await self.notify("⚠️ **Daily Limit Reached.** Bot stopped.")
                    self.running = False
                    break

                # Analysis logic...
                m5 = await self.api.get_candles({"ticks_history": SYMBOL, "granularity": 300, "count": 60})
                m5_c = [x['close'] for x in m5['candles']]
                m5_ema = get_ema(m5_c, 50)
                m5_bias = "CALL" if m5_c[-1] > m5_ema else "PUT"

                m1 = await self.api.get_candles({"ticks_history": SYMBOL, "granularity": 60, "count": 60})
                m1_c = [x['close'] for x in m1['candles']]
                m1_ema = get_ema(m1_c, 50)
                rsi = get_rsi(m1_c, 14)

                if not (40 <= rsi <= 60):
                    self.current_status = "🔎 RSI Filter Active"
                    await asyncio.sleep(10)
                    continue

                self.current_status = "🔎 Scanning..."
                was_away = abs(m1_c[-6] - m1_ema) > (m1_ema * 0.0004)
                
                last_c, prev_3 = m1['candles'][-1], m1['candles'][-4:-1]
                signal = None
                if m5_bias == "CALL" and m1_c[-1] > m1_ema and was_away:
                    if check_v3_momentum(last_c, prev_3, "CALL", m1_ema): signal = "CALL"
                elif m5_bias == "PUT" and m1_c[-1] < m1_ema and was_away:
                    if check_v3_momentum(last_c, prev_3, "PUT", m1_ema): signal = "PUT"

                if signal and abs(m1_ema - self.last_ema_level) > (m1_ema * 0.0002):
                    await self.execute_trade(signal)
                    self.last_ema_level = m1_ema
                    await asyncio.sleep(DURATION * 60 + 10) 

                await asyncio.sleep(5)
            except: await asyncio.sleep(10)

    async def execute_trade(self, side):
        try:
            req = {"buy": 1, "price": STAKE, "parameters": {"amount": STAKE, "basis": "stake", 
                   "contract_type": side, "currency": "USD", "duration": DURATION, "duration_unit": "m", "symbol": SYMBOL}}
            resp = await self.api.buy(req)
            self.trades_today += 1
            self.active_trade_msg = f"📡 {side} Active"
            await self.notify(f"🚀 **AUTOMATIC TRADE OPENED!**\nType: `{side}`\nStake: `${STAKE}`\nAccount: `{self.account_mode}`")
            asyncio.create_task(self.check_result(resp['buy']['contract_id']))
        except: pass

    async def check_result(self, cid):
        await asyncio.sleep(DURATION * 60 + 5)
        try:
            table = await self.api.get_profit_table({"limit": 1})
            last = table['profit_table']['transactions'][0]
            profit = float(last['sell_price']) - float(last['buy_price'])
            self.pnl_today += profit
            res = "✅ WIN" if profit > 0 else "❌ LOSS"
            if profit > 0: self.wins_today += 1
            else: self.losses_today += 1
            await self.update_balance()
            await self.notify(f"🏁 **TRADE FINISHED**\nResult: {res}\nProfit: `${round(profit, 2)}`")
            self.active_trade_msg = "None"
        except: pass

# ========================= 4. TELEGRAM UI =========================
bot = DerivSniperBot()

async def start_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    kb = [[InlineKeyboardButton("🚀 START SCANNER", callback_data="PROMPT_MODE")],
          [InlineKeyboardButton("📊 STATUS", callback_data="STATUS"), InlineKeyboardButton("💰 BALANCE", callback_data="CHECK_BAL")],
          [InlineKeyboardButton("🛑 STOP", callback_data="STOP")]]
    text = "💎 **Deriv Sniper v3**\n$5 Account Protection ON."
    if u.message: await u.message.reply_text(text, reply_markup=InlineKeyboardMarkup(kb), parse_mode="Markdown")
    else: await u.callback_query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(kb), parse_mode="Markdown")

async def btn_handler(u: Update, c: ContextTypes.DEFAULT_TYPE):
    q = u.callback_query
    await q.answer()
    
    if q.data == "PROMPT_MODE":
        kb = [[InlineKeyboardButton("🧪 DEMO", callback_data="SET_DEMO"), InlineKeyboardButton("💰 LIVE", callback_data="SET_REAL")],
              [InlineKeyboardButton("⬅️ BACK", callback_data="BACK")]]
        await q.edit_message_text("💳 **Select Mode:**", reply_markup=InlineKeyboardMarkup(kb), parse_mode="Markdown")

    elif q.data in ["SET_DEMO", "SET_REAL"]:
        is_demo = q.data == "SET_DEMO"
        bot.active_token = DEMO_TOKEN if is_demo else REAL_TOKEN
        bot.account_mode = "🧪 DEMO" if is_demo else "💰 REAL"
        if await bot.connect(bot.active_token):
            asyncio.create_task(bot.run_scanner())
            await q.edit_message_text(f"✅ **Scanner Started!**\nMode: {bot.account_mode}", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("📊 STATUS", callback_data="STATUS")]]))
        else: await q.edit_message_text("❌ Connection Error.")

    elif q.data == "STATUS":
        pnl = f"+${round(bot.pnl_today, 2)}" if bot.pnl_today >= 0 else f"-${round(abs(bot.pnl_today), 2)}"
        msg = (f"📊 **STATUS**\nMode: `{bot.account_mode}`\nState: `{bot.current_status}`\n"
               f"Bullets: {bot.trades_today}/2\nWins: {bot.wins_today} | Loss: {bot.losses_today}\nPnL: `{pnl}`")
        kb = [[InlineKeyboardButton("🔄 Refresh", callback_data="STATUS")], [InlineKeyboardButton("⬅️ BACK", callback_data="BACK")]]
        try: await q.edit_message_text(msg, reply_markup=InlineKeyboardMarkup(kb), parse_mode="Markdown")
        except BadRequest: pass # SILENCE THE ERROR

    elif q.data == "CHECK_BAL":
        await bot.update_balance()
        await q.edit_message_text(f"💰 **Balance**\nAccount: {bot.account_mode}\nBalance: `${bot.balance}`", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ BACK", callback_data="BACK")]]))

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
