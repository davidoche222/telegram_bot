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

# ========================= HELPERS =========================
def clean_money(amount):
    """Forces number to exactly 2 decimal places to stop the 0.35 error"""
    return float(Decimal(str(amount)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

def find_min_stake(error_msg):
    """Regex to find the number Deriv is asking for in the error message"""
    match = re.search(r"at least\s*(\d+\.?\d*)", error_msg.lower())
    return float(match.group(1)) if match else None

# ========================= BOT CORE =========================
class DerivSniperBot:
    def __init__(self):
        self.api = None
        self.app = None
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
        except Exception as e:
            logger.error(f"Connect failed: {e}")
            return False

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
            
            # Use clean_money to ensure we send exactly 0.50
            current_stake = clean_money(STAKE)

            for attempt in range(2):
                try:
                    # 1. Proposal
                    prop_req = {
                        "proposal": 1,
                        "amount": current_stake,
                        "basis": "stake",
                        "contract_type": side,
                        "currency": "USD",
                        "duration": int(DURATION),
                        "duration_unit": "m",
                        "symbol": self.current_symbol
                    }
                    
                    prop = await self.api.proposal(prop_req)

                    if "error" in prop:
                        msg = prop["error"].get("message", "")
                        # If broker says 'at least X', grab X and retry once
                        required_min = find_min_stake(msg)
                        if required_min and attempt == 0:
                            current_stake = clean_money(required_min + 0.01)
                            logger.info(f"Retrying with higher stake: {current_stake}")
                            continue
                        
                        await self.app.bot.send_message(TELEGRAM_CHAT_ID, f"❌ Stake Error: {msg}")
                        return

                    # 2. Buy
                    proposal_id = prop["proposal"]["id"]
                    ask_price = float(prop["proposal"]["ask_price"])
                    
                    # Buffer to prevent 'Price has changed' error
                    buy_price = clean_money(ask_price + 0.10)
                    
                    buy_res = await self.api.buy({"buy": proposal_id, "price": buy_price})
                    
                    if "error" in buy_res:
                        await self.app.bot.send_message(TELEGRAM_CHAT_ID, f"❌ Buy Error: {buy_res['error'].get('message')}")
                        return

                    self.active_trade_info = buy_res["buy"]["contract_id"]
                    await self.app.bot.send_message(TELEGRAM_CHAT_ID, f"🚀 **TRADE PLACED**\nStake: `${current_stake}`\nMarket: `{self.current_symbol}`")
                    asyncio.create_task(self.check_result(self.active_trade_info))
                    return

                except Exception as e:
                    logger.error(f"Trade process error: {e}")
                    return

    async def check_result(self, cid):
        await asyncio.sleep((DURATION * 60) + 10)
        try:
            poc = await self.api.proposal_open_contract({"proposal_open_contract": 1, "contract_id": cid})
            c = poc.get("proposal_open_contract", {})
            if c.get("is_sold"):
                profit = float(c.get("profit", 0.0))
                self.pnl_today += profit
                res = "✅ WIN" if profit > 0 else "❌ LOSS"
                if profit > 0: self.wins_today += 1
                else: self.losses_today += 1
                await self.fetch_balance()
                await self.app.bot.send_message(TELEGRAM_CHAT_ID, f"🏁 Result: {res} (${profit:.2f})")
        finally:
            self.active_trade_info = None

# ========================= TELEGRAM UI =========================
bot_logic = DerivSniperBot()

async def btn_handler(u: Update, c: ContextTypes.DEFAULT_TYPE):
    q = u.callback_query
    await q.answer()
    if q.data == "PROMPT_MODE":
        kb = InlineKeyboardMarkup([[InlineKeyboardButton("🧪 DEMO", callback_data="SET_DEMO"), InlineKeyboardButton("💰 LIVE", callback_data="SET_REAL")]])
        await q.edit_message_text("Select Account:", reply_markup=kb)
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
    await u.message.reply_text("💎 Sniper v4.4", reply_markup=main_keyboard())

if __name__ == "__main__":
    # drop_pending_updates=True stops the "Conflict" error
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    bot_logic.app = app
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CallbackQueryHandler(btn_handler))
    app.run_polling(drop_pending_updates=True)
