import asyncio
import logging
import re
from decimal import Decimal, ROUND_HALF_UP
from deriv_api import DerivAPI
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

# ========================= CONFIG =========================
DEMO_TOKEN = "YYSlMBIcTXqOozU"
REAL_TOKEN = "2NFJTH3JgXWFCcv"
APP_ID = 1089

MARKETS = ["R_10", "R_25", "R_50", "R_100"]
STAKE = 0.50 
DURATION = 5

TELEGRAM_TOKEN = "8276370676:AAGh5VqkG7b4cvpfRIVwY_rtaBlIiNwCTDM"
TELEGRAM_CHAT_ID = "7634818949"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class DerivSniperBot:
    def __init__(self):
        self.api = None
        self.app = None
        self.running = False
        self.active_token = None
        self.current_symbol = MARKETS[0]
        self.balance = "0.00"
        self.pnl_today = 0.0
        self.wins_today = 0
        self.losses_today = 0
        self.active_trade_info = None
        self.trade_lock = asyncio.Lock()

    async def connect(self) -> bool:
        try:
            self.api = DerivAPI(app_id=APP_ID)
            await self.api.authorize(self.active_token)
            await self.fetch_balance()
            return True
        except: return False

    async def fetch_balance(self):
        try:
            bal = await self.api.balance({"balance": 1})
            b = bal.get("balance", {})
            self.balance = f"{float(b.get('balance', 0.0)):.2f} {b.get('currency', 'USD')}"
        except: pass

    async def execute_trade(self, side: str):
        if not self.api: return
        async with self.trade_lock:
            if self.active_trade_info: return
            
            try:
                # 🛑 NUCLEAR FIX: Using hard-coded values and explicit types
                # Stripping any hidden spaces from symbol and side
                proposal_data = {
                    "proposal": 1,
                    "amount": float(0.50),            # Explicit 0.50
                    "basis": "stake",
                    "contract_type": str(side).strip(), 
                    "currency": "USD",
                    "duration": int(5),               # Explicit 5
                    "duration_unit": "m",
                    "symbol": str(self.current_symbol).strip()
                }

                logger.info(f"🚀 SENDING PROPOSAL: {proposal_data}")
                prop = await self.api.proposal(proposal_data)

                if "error" in prop:
                    msg = prop["error"].get("message", "Unknown Error")
                    logger.error(f"❌ PROPOSAL REJECTED: {msg}")
                    await self.app.bot.send_message(TELEGRAM_CHAT_ID, f"❌ Broker Error: {msg}")
                    return

                # If successful, buy with a very high price limit to bypass 'Price Changed'
                pid = prop["proposal"]["id"]
                buy_res = await self.api.buy({"buy": pid, "price": 10.0})
                
                if "error" in buy_res:
                    logger.error(f"❌ BUY ERROR: {buy_res['error'].get('message')}")
                    await self.app.bot.send_message(TELEGRAM_CHAT_ID, f"❌ Buy Error: {buy_res['error'].get('message')}")
                    return

                self.active_trade_info = buy_res["buy"]["contract_id"]
                await self.app.bot.send_message(TELEGRAM_CHAT_ID, "✅ SUCCESS! Trade placed at $0.50")
                asyncio.create_task(self.check_result(self.active_trade_info))

            except Exception as e:
                logger.error(f"⚠️ CRITICAL SYSTEM ERROR: {e}")

    async def check_result(self, cid):
        await asyncio.sleep((DURATION * 60) + 10)
        try:
            poc = await self.api.proposal_open_contract({"proposal_open_contract": 1, "contract_id": cid})
            c = poc.get("proposal_open_contract", {})
            if c.get("is_sold"):
                profit = float(c.get("profit", 0.0))
                self.pnl_today += profit
                if profit > 0: self.wins_today += 1
                else: self.losses_today += 1
                await self.fetch_balance()
                await self.app.bot.send_message(TELEGRAM_CHAT_ID, f"🏁 Result: {'✅ WIN' if profit > 0 else '❌ LOSS'} (${profit:.2f})")
        finally:
            self.active_trade_info = None

# ========================= TELEGRAM UI =========================
bot_logic = DerivSniperBot()

async def btn_handler(u: Update, c: ContextTypes.DEFAULT_TYPE):
    q = u.callback_query
    await q.answer()
    if q.data == "PROMPT_MODE":
        kb = InlineKeyboardMarkup([[InlineKeyboardButton("🧪 DEMO", callback_data="SET_DEMO"), InlineKeyboardButton("💰 LIVE", callback_data="SET_REAL")]])
        await q.edit_message_text("Choose Account:", reply_markup=kb)
    elif q.data in ("SET_DEMO", "SET_REAL"):
        bot_logic.active_token = DEMO_TOKEN if q.data == "SET_DEMO" else REAL_TOKEN
        if await bot_logic.connect():
            await q.edit_message_text(f"✅ Connected!\nBal: {bot_logic.balance}", reply_markup=main_keyboard())
    elif q.data == "TEST_BUY":
        await bot_logic.execute_trade("CALL")
    elif q.data == "STATUS":
        await bot_logic.fetch_balance()
        status = f"📊 Bal: {bot_logic.balance}\n🏆 W/L: {bot_logic.wins_today}/{bot_logic.losses_today}\n💵 PnL: ${bot_logic.pnl_today:.2f}"
        await q.edit_message_text(status, reply_markup=main_keyboard())

def main_keyboard():
    return InlineKeyboardMarkup([[InlineKeyboardButton("🚀 START", callback_data="PROMPT_MODE")], 
                                 [InlineKeyboardButton("📊 STATUS", callback_data="STATUS"), InlineKeyboardButton("🧪 TEST BUY", callback_data="TEST_BUY")]])

async def start_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    await u.message.reply_text("💎 Sniper v4.5", reply_markup=main_keyboard())

if __name__ == "__main__":
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    bot_logic.app = app
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CallbackQueryHandler(btn_handler))
    
    # drop_pending_updates=True clears old conflict errors immediately
    print("Bot is starting...")
    app.run_polling(drop_pending_updates=True, allowed_updates=Update.ALL_TYPES)
