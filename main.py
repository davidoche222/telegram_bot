import asyncio
import datetime
import logging
import numpy as np
from deriv_api import DerivAPI
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.error import BadRequest, Conflict
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

# ========================= 1. CONFIG =========================
DEMO_TOKEN = "YYSlMBIcTXqOozU"
REAL_TOKEN = "2NFJTH3JgXWFCcv"
APP_ID = "1089"

# We now use a list of symbols to rotate through if one is blocked
MARKETS = ["R_10", "R_25", "R_50", "R_100"] 
current_market_index = 0

SYMBOL = MARKETS[current_market_index]
STAKE = 0.35
DURATION = 5 # 5 minutes is a good middle ground

TELEGRAM_TOKEN = "8276370676:AAGh5VqkG7b4cvpfRIVwY_rtaBlIiNwCTDM"
TELEGRAM_CHAT_ID = "7634818949"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ========================= 2. BOT CORE =========================
class DerivSniperBot:
    def __init__(self):
        self.api = None
        self.app = None
        self.running = False
        self.account_mode = "Disconnected"
        self.active_token = None
        self.current_symbol = SYMBOL
        self.current_status = "🛑 Stopped"
        self.active_trade_info = None
        self.balance = "0.00"
        self.wins_today, self.losses_today, self.pnl_today = 0, 0, 0.0
        self.trade_lock = asyncio.Lock()

    async def send(self, text):
        try: await self.app.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=text, parse_mode="Markdown")
        except: pass

    async def connect(self):
        try:
            self.api = DerivAPI(app_id=APP_ID)
            await self.api.authorize(self.active_token)
            bal = await self.api.balance()
            self.balance = f"{bal['balance']['balance']:.2f}"
            return True
        except Exception as e:
            logger.error(f"Connect error: {e}")
            return False

    async def switch_market(self):
        """Switches to the next symbol in the MARKETS list"""
        global current_market_index
        current_market_index = (current_market_index + 1) % len(MARKETS)
        self.current_symbol = MARKETS[current_market_index]
        logger.info(f"🔄 Switching to Market: {self.current_symbol}")
        await self.send(f"🔄 **Market Switch:** Trying `{self.current_symbol}` due to limits.")

    async def execute_trade(self, side):
        if not self.api: return
        
        async with self.trade_lock:
            if self.active_trade_info: return
            
            try:
                req = {
                    "buy": 1, 
                    "price": STAKE, 
                    "parameters": {
                        "amount": STAKE, "basis": "stake", "contract_type": side,
                        "currency": "USD", "duration": DURATION, "duration_unit": "m",
                        "symbol": self.current_symbol
                    }
                }
                resp = await self.api.buy(req)
                
                if "error" in resp:
                    err_msg = resp['error'].get('message', "")
                    # Check if error is related to purchase limits
                    if "maximum purchase price" in err_msg.lower():
                        logger.warning(f"Limit hit on {self.current_symbol}. Switching...")
                        await self.switch_market()
                    else:
                        await self.send(f"❌ **Trade Error:** {err_msg}")
                    return

                cid = resp["buy"]["contract_id"]
                self.active_trade_info = {"side": side, "id": cid}
                await self.send(f"🚀 **TRADE PLACED**\nMarket: `{self.current_symbol}`\nSide: `{side}`")
                asyncio.create_task(self.check_result(cid))

            except Exception as e:
                logger.error(f"Trade Error: {e}")

    async def run_scanner(self):
        self.running = True
        while self.running:
            try:
                # Basic scan just to keep connection alive and show RSI
                if not self.api: await self.connect()
                # (Scanner logic for RSI omitted for brevity, but same as before)
                self.current_status = f"🔎 Scanning {self.current_symbol}"
                await asyncio.sleep(10)
            except Exception as e:
                logger.error(f"Scanner Error: {e}")
                await asyncio.sleep(10)

    async def check_result(self, cid):
        await asyncio.sleep(DURATION * 60 + 5)
        self.active_trade_info = None # Reset for next trade
        await self.send(f"🏁 Trade `{cid}` finished. Check balance!")

# ========================= 3. TELEGRAM UI =========================
bot = DerivSniperBot()

async def btn_handler(u: Update, c: ContextTypes.DEFAULT_TYPE):
    q = u.callback_query
    await q.answer()
    
    if q.data == "PROMPT_MODE":
        kb = [[InlineKeyboardButton("🧪 DEMO", callback_data="SET_DEMO")]]
        await q.edit_message_text("Select Account:", reply_markup=InlineKeyboardMarkup(kb))
    
    elif q.data == "SET_DEMO":
        bot.active_token = DEMO_TOKEN
        if await bot.connect():
            asyncio.create_task(bot.run_scanner())
            await q.edit_message_text(f"✅ Online: {bot.current_symbol}", reply_markup=main_keyboard())

    elif q.data == "TEST_BUY":
        await bot.execute_trade("CALL")

def main_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🧪 TEST BUY", callback_data="TEST_BUY")],
        [InlineKeyboardButton("🛑 STOP", callback_data="STOP")]
    ])

async def start_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    await u.message.reply_text("💎 **Sniper v3.3**", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🚀 START", callback_data="PROMPT_MODE")]]))

if __name__ == "__main__":
    # The 'drop_pending_updates' fixes the Conflict error you saw!
    app = Application.builder().token(TELEGRAM_TOKEN).post_init(lambda a: a.bot.delete_webhook(drop_pending_updates=True)).build()
    bot.app = app
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CallbackQueryHandler(btn_handler))
    app.run_polling()
