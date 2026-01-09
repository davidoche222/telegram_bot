import asyncio
import logging
import re
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
from deriv_api import DerivAPI
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.error import BadRequest
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

# ========================= CONFIG =========================
DEMO_TOKEN = "YYSlMBIcTXqOozU"
REAL_TOKEN = "2NFJTH3JgXWFCcv"
APP_ID = 1089

MARKETS = ["R_10", "R_25", "R_50", "R_100"]
STAKE = 0.50 
MIN_STAKE = 0.35
DURATION = 5

TELEGRAM_TOKEN = "8276370676:AAGh5VqkG7b4cvpfRIVwY_rtaBlIiNwCTDM"
TELEGRAM_CHAT_ID = "7634818949"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ========================= HELPERS =========================
def money(x: float) -> float:
    return float(Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

def extract_min_stake(msg: str) -> float | None:
    # This searches for numbers in the error "stake must be at least X.XX"
    m = re.search(r"at least\s*([0-9]+(?:\.[0-9]+)?)", (msg or "").lower())
    if not m: return None
    try: return money(float(m.group(1)))
    except: return None

# ========================= BOT CORE =========================
class DerivSniperBot:
    def __init__(self):
        self.api = None
        self.app = None
        self.running = False
        self.active_token = None
        self.current_symbol = MARKETS[0]
        self.current_status = "🛑 Stopped"
        self.balance = "0.00"
        self.wins_today = 0
        self.losses_today = 0
        self.pnl_today = 0.0
        self.active_trade_info = None
        self.trade_lock = asyncio.Lock()

    async def connect(self) -> bool:
        try:
            self.api = DerivAPI(app_id=APP_ID)
            await self.api.authorize(self.active_token)
            await self.fetch_balance()
            return True
        except Exception as e:
            return False

    async def fetch_balance(self):
        if not self.api: return
        try:
            bal = await self.api.balance({"balance": 1})
            b = bal.get("balance", {})
            self.balance = f"{float(b.get('balance', 0.0)):.2f} {b.get('currency', 'USD')}"
        except: pass

    async def execute_trade(self, side: str):
        if not self.api: return
        async with self.trade_lock:
            if self.active_trade_info: return
            
            # Start with your chosen stake
            current_stake = money(max(float(STAKE), float(MIN_STAKE)))

            for attempt in range(3): # Increased to 3 attempts
                try:
                    # 1. Get the Proposal
                    prop = await self.api.proposal({
                        "proposal": 1, 
                        "amount": current_stake, 
                        "basis": "stake",
                        "contract_type": side, 
                        "duration": DURATION,
                        "duration_unit": "m", 
                        "symbol": self.current_symbol, 
                        "currency": "USD"
                    })

                    if "error" in prop:
                        msg = prop["error"].get("message", "")
                        new_min = extract_min_stake(msg)
                        if new_min and attempt < 2:
                            current_stake = new_min + 0.01 # Adjust stake to broker requirement
                            continue
                        await self.app.bot.send_message(TELEGRAM_CHAT_ID, f"❌ Proposal Error: {msg}")
                        return
                    
                    # 2. Extract price details
                    pid = prop["proposal"]["id"]
                    ask = float(prop["proposal"].get("ask_price", current_stake))
                    
                    # 3. Buy with a generous price buffer (slippage protection)
                    # We set the price to ask + 5% to ensure it hits
                    buy_limit = money(ask * 1.05)
                    buy_res = await self.api.buy({"buy": pid, "price": buy_limit})
                    
                    if "error" in buy_res:
                        msg = buy_res['error'].get('message', "")
                        new_min = extract_min_stake(msg)
                        if new_min and attempt < 2:
                            current_stake = new_min + 0.01
                            continue
                        await self.app.bot.send_message(TELEGRAM_CHAT_ID, f"❌ Buy Failed: {msg}")
                        return

                    # Success!
                    cid = buy_res["buy"]["contract_id"]
                    self.active_trade_info = cid
                    await self.app.bot.send_message(TELEGRAM_CHAT_ID, f"🚀 **TRADE PLACED**\nMarket: `{self.current_symbol}`\nStake: `${current_stake}`")
                    asyncio.create_task(self.check_result(cid))
                    return

                except Exception as e:
                    logger.error(f"Trade Error: {e}")
                    return

    async def check_result(self, cid):
        # Wait for duration + safety buffer
        await asyncio.sleep(DURATION * 60 + 10)
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
                await self.app.bot.send_message(TELEGRAM_CHAT_ID, f"🏁 {res} | Profit: `${profit:.2f}`\nBalance: {self.balance}")
        finally:
            self.active_trade_info = None

# ========================= TELEGRAM UI =========================
bot_logic = DerivSniperBot()

def main_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🚀 START SCANNER", callback_data="PROMPT_MODE")],
        [InlineKeyboardButton("📊 STATUS", callback_data="STATUS"), InlineKeyboardButton("🧪 TEST BUY", callback_data="TEST_BUY")],
        [InlineKeyboardButton("🛑 STOP", callback_data="STOP")]
    ])

async def btn_handler(u: Update, c: ContextTypes.DEFAULT_TYPE):
    q = u.callback_query
    if q.data == "PROMPT_MODE":
        kb = InlineKeyboardMarkup([[InlineKeyboardButton("🧪 DEMO", callback_data="SET_DEMO"), InlineKeyboardButton("💰 LIVE", callback_data="SET_REAL")]])
        await q.edit_message_text("Choose Account:", reply_markup=kb)
    elif q.data in ("SET_DEMO", "SET_REAL"):
        bot_logic.active_token = DEMO_TOKEN if q.data == "SET_DEMO" else REAL_TOKEN
        if await bot_logic.connect():
            bot_logic.running = True
            await q.edit_message_text(f"✅ Connected to {q.data.split('_')[1]}!\nBalance: {bot_logic.balance}", reply_markup=main_keyboard())
    elif q.data == "STATUS":
        await bot_logic.fetch_balance()
        msg = f"📊 **STATUS**\n💰 Bal: {bot_logic.balance}\n🏆 W/L: {bot_logic.wins_today}/{bot_logic.losses_today}\n💵 PnL: ${bot_logic.pnl_today:.2f}"
        await q.edit_message_text(msg, reply_markup=main_keyboard())
    elif q.data == "TEST_BUY":
        await q.answer("Placing trade...")
        await bot_logic.execute_trade("CALL")
    elif q.data == "STOP":
        bot_logic.running = False
        await q.edit_message_text("🛑 Bot Stopped.", reply_markup=main_keyboard())

async def start_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    await u.message.reply_text("💎 **Deriv Sniper v4.2**", reply_markup=main_keyboard())

if __name__ == "__main__":
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    bot_logic.app = app
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CallbackQueryHandler(btn_handler))
    app.run_polling()
