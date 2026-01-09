import asyncio
import datetime
import logging
import numpy as np
from deriv_api import DerivAPI
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.error import BadRequest
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

# ========================= CONFIG =========================
DEMO_TOKEN = "YYSlMBIcTXqOozU"
REAL_TOKEN = "2NFJTH3JgXWFCcv"
APP_ID = 1089

SYMBOL = "R_10"   # Volatility 10 is more stable for $0.35 stakes
STAKE = 0.35
DURATION = 5      # 5 Minutes

TELEGRAM_TOKEN = "8276370676:AAGh5VqkG7b4cvpfRIVwY_rtaBlIiNwCTDM"
TELEGRAM_CHAT_ID = "7634818949"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ========================= BOT CORE =========================
class DerivSniperBot:
    def __init__(self):
        self.api = None
        self.app = None
        self.running = False
        self.active_token = None
        self.current_symbol = SYMBOL
        self.current_status = "🛑 Stopped"
        self.active_trade_info = None
        self.balance = "0.00"
        self.wins_today, self.losses_today, self.pnl_today = 0, 0, 0.0
        self.trade_lock = asyncio.Lock()
        self._scanner_task = None

    async def send(self, text: str):
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

    async def execute_trade(self, side: str):
        """Your new Improved Logic: Proposal -> Extract Price -> Buy"""
        if not self.api:
            await self.send("❌ API not connected.")
            return

        async with self.trade_lock:
            if self.active_trade_info: return
            
            try:
                # 1) Get Proposal
                proposal_req = {
                    "proposal": 1, "amount": STAKE, "basis": "stake",
                    "contract_type": side, "duration": DURATION,
                    "duration_unit": "m", "symbol": self.current_symbol
                }
                prop = await self.api.proposal(proposal_req)
                
                if "error" in prop:
                    await self.send(f"❌ Proposal Error:\n`{prop['error'].get('message')}`")
                    return

                proposal = prop["proposal"]
                ask_price = float(proposal.get("ask_price", 0.0))
                
                # 2) Buy using the exact price Deriv just gave us
                max_price = round(ask_price + 0.01, 2)
                buy_req = {"buy": proposal["id"], "price": max_price}
                resp = await self.api.buy(buy_req)

                if "error" in resp:
                    await self.send(f"❌ Trade Refused:\n`{resp['error'].get('message')}`")
                    return

                cid = resp["buy"]["contract_id"]
                self.active_trade_info = {"side": side, "id": cid}
                await self.send(f"🚀 **TRADE PLACED**\nSide: `{side}`\nPrice: `${ask_price}`")
                asyncio.create_task(self.check_result(cid))

            except Exception as e:
                await self.send(f"⚠️ Trade Error: {e}")

    async def run_scanner(self):
        self.running = True
        while self.running:
            try:
                if not self.api: await self.connect()
                self.current_status = f"🔎 Scanning {self.current_symbol}"
                await asyncio.sleep(10)
            except: await asyncio.sleep(10)

    async def check_result(self, cid):
        await asyncio.sleep(DURATION * 60 + 5)
        try:
            poc = await self.api.proposal_open_contract({"proposal_open_contract": 1, "contract_id": cid})
            c = poc.get("proposal_open_contract", {})
            if c.get("is_sold"):
                profit = float(c.get("profit", 0.0))
                self.pnl_today += profit
                res = "✅ WIN" if profit > 0 else "❌ LOSS"
                if profit > 0: self.wins_today += 1
                else: self.losses_today += 1
                await self.send(f"🏁 **{res}**\nProfit: `${round(profit, 2)}`")
        finally: self.active_trade_info = None

# ========================= TELEGRAM UI =========================
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
    elif q.data == "STOP":
        bot.running = False
        await q.edit_message_text("🛑 Stopped.")

def main_keyboard():
    return InlineKeyboardMarkup([[InlineKeyboardButton("🧪 TEST BUY", callback_data="TEST_BUY")], [InlineKeyboardButton("🛑 STOP", callback_data="STOP")]])

async def start_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    await u.message.reply_text("💎 **Sniper v3.5**", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🚀 START", callback_data="PROMPT_MODE")]]))

if __name__ == "__main__":
    app = Application.builder().token(TELEGRAM_TOKEN).post_init(lambda a: a.bot.delete_webhook(drop_pending_updates=True)).build()
    bot.app = app
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CallbackQueryHandler(btn_handler))
    app.run_polling(drop_pending_updates=True)
