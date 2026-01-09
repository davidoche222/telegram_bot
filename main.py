import asyncio
import datetime
import logging
import numpy as np
import os
from deriv_api import DerivAPI
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.error import BadRequest
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

# ========================= 1. CONFIG =========================
DEMO_TOKEN = "YYSlMBIcTXqOozU"
REAL_TOKEN = "2NFJTH3JgXWFCcv"
APP_ID = "1089"

SYMBOL = "R_50" 
STAKE = 0.35
DURATION = 15 

TELEGRAM_TOKEN = "8276370676:AAGh5VqkG7b4cvpfRIVwY_rtaBlIiNwCTDM"
TELEGRAM_CHAT_ID = "7634818949"

# ========================= 2. LOGGING SETUP =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("bot_log.txt"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ========================= 3. STRATEGY ENGINE =========================
def get_ema(values, period=50):
    if len(values) < period: return None
    values = np.array(values, dtype=float)
    k = 2 / (period + 1)
    ema = values[0]
    for p in values[1:]:
        ema = p * k + ema * (1 - k)
    return float(ema)

def get_rsi(prices, period=14):
    if len(prices) < period + 1: return 50.0
    deltas = np.diff(prices)
    up = deltas[deltas >= 0].sum() / period
    down = -deltas[deltas < 0].sum() / period
    if down == 0: return 100.0
    rs = up / down
    return 100.0 - (100.0 / (1.0 + rs))

# ========================= 4. BOT CORE =========================
class DerivSniperBot:
    def __init__(self):
        self.api = None
        self.app = None
        self.running = False
        self.account_mode = "Disconnected"
        self.active_token = None
        self.current_status = "🛑 Stopped"
        self.active_trade_info = None
        self.balance = "0.00"
        self.wins_today, self.losses_today, self.pnl_today = 0, 0, 0.0
        self.trade_lock = asyncio.Lock()

    async def send(self, text):
        try:
            await self.app.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=text, parse_mode="Markdown")
        except: pass

    async def fetch_balance(self):
        try:
            bal_data = await self.api.balance()
            self.balance = f"{bal_data['balance']['balance']:.2f} {bal_data['balance']['currency']}"
            return self.balance
        except: return "Error"

    async def connect(self):
        try:
            self.api = DerivAPI(app_id=APP_ID)
            await self.api.authorize(self.active_token)
            await self.fetch_balance()
            logger.info(f"Connected. Balance: {self.balance}")
            return True
        except Exception as e:
            logger.error(f"Connect error: {e}")
            return False

    async def get_candles(self, gran):
        resp = await self.api.ticks_history({"ticks_history": SYMBOL, "count": 60, "end": "latest", "granularity": gran, "style": "candles"})
        return resp["candles"]

    async def run_scanner(self):
        self.running = True
        while self.running:
            try:
                if not self.api: await self.connect()
                if self.active_trade_info:
                    await asyncio.sleep(15)
                    continue

                m5 = await self.get_candles(300)
                m1 = await self.get_candles(60)
                m1_c = [float(x["close"]) for x in m1]
                rsi = get_rsi(m1_c, 14)

                self.current_status = f"🔎 Scanning (RSI: {round(rsi)})"
                await asyncio.sleep(10)
            except Exception as e:
                logger.error(f"Scanner Loop Error: {e}")
                await asyncio.sleep(10)

    async def execute_trade(self, side):
        if not self.api:
            await self.send("❌ API not connected.")
            return

        async with self.trade_lock:
            if self.active_trade_info: return
            
            try:
                req = {"buy": 1, "price": STAKE, "parameters": {"amount": STAKE, "basis": "stake", "contract_type": side, "currency": "USD", "duration": DURATION, "duration_unit": "m", "symbol": SYMBOL}}
                resp = await self.api.buy(req)
                
                if "error" in resp:
                    await self.send(f"❌ **Trade Refused:**\n`{resp['error'].get('message')}`")
                    return

                cid = resp["buy"]["contract_id"]
                self.active_trade_info = {"side": side, "id": cid}
                await self.send(f"🚀 **TRADE PLACED**\nSide: `{side}`")
                asyncio.create_task(self.check_result(cid))
            except Exception as e:
                logger.error(f"Trade Error: {e}")

    async def check_result(self, cid):
        await asyncio.sleep(DURATION * 60 + 5)
        try:
            table = await self.api.profit_table({"limit": 5})
            for trans in table["profit_table"]["transactions"]:
                if str(trans.get("contract_id")) == str(cid):
                    profit = float(trans["sell_price"]) - float(trans["buy_price"])
                    self.pnl_today += profit
                    res = "✅ WIN" if profit > 0 else "❌ LOSS"
                    if profit > 0: self.wins_today += 1
                    else: self.losses_today += 1
                    
                    await self.fetch_balance() # Update balance after trade
                    await self.send(f"🏁 **RESULT**: {res}\nProfit: `${round(profit, 2)}`")
                    break
        finally: self.active_trade_info = None

# ========================= 5. TELEGRAM UI =========================
bot = DerivSniperBot()

def main_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🚀 START SCANNER", callback_data="PROMPT_MODE")],
        [InlineKeyboardButton("📊 STATUS/BALANCE", callback_data="STATUS"), InlineKeyboardButton("🧪 TEST BUY", callback_data="TEST_BUY")],
        [InlineKeyboardButton("🛑 STOP", callback_data="STOP")]
    ])

async def start_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    await u.message.reply_text("💎 **Deriv Sniper v3.2**", reply_markup=main_keyboard())

async def btn_handler(u: Update, c: ContextTypes.DEFAULT_TYPE):
    q = u.callback_query
    await q.answer()
    
    if q.data == "PROMPT_MODE":
        await q.edit_message_text("💳 **Select Account:**", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🧪 DEMO", callback_data="SET_DEMO"), InlineKeyboardButton("💰 LIVE", callback_data="SET_REAL")]]))
    
    elif q.data in ("SET_DEMO", "SET_REAL"):
        bot.active_token = DEMO_TOKEN if q.data == "SET_DEMO" else REAL_TOKEN
        bot.account_mode = "🧪 DEMO" if q.data == "SET_DEMO" else "💰 REAL"
        if await bot.connect():
            asyncio.create_task(bot.run_scanner())
            await q.edit_message_text(f"✅ **Scanner Online**\nMode: {bot.account_mode}", reply_markup=main_keyboard())

    elif q.data == "TEST_BUY":
        if bot.api: await bot.execute_trade("CALL")
        else: await q.answer("Start scanner first!", show_alert=True)

    elif q.data == "STATUS":
        await bot.fetch_balance()
        pnl_str = f"+${round(bot.pnl_today, 2)}" if bot.pnl_today >= 0 else f"-${round(abs(bot.pnl_today), 2)}"
        msg = (f"📊 **DASHBOARD**\n"
               f"💰 Balance: `{bot.balance}`\n"
               f"📉 PnL Today: `{pnl_str}`\n"
               f"🏆 Wins: {bot.wins_today} | ❌ Loss: {bot.losses_today}\n"
               f"📡 Status: {bot.current_status}")
        try: await q.edit_message_text(msg, reply_markup=main_keyboard(), parse_mode="Markdown")
        except BadRequest: pass

    elif q.data == "STOP":
        bot.running = False
        await q.edit_message_text("🛑 Bot Stopped.")

if __name__ == "__main__":
    app = Application.builder().token(TELEGRAM_TOKEN).post_init(lambda a: a.bot.delete_webhook(drop_pending_updates=True)).build()
    bot.app = app
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CallbackQueryHandler(btn_handler))
    app.run_polling()
