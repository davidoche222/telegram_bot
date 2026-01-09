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
current_market_idx = 0

STAKE = 0.40
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
    m = re.search(r"at least\s*([0-9]+(?:\.[0-9]+)?)", (msg or "").lower())
    if not m: return None
    try: return money(float(m.group(1)))
    except: return None

def get_rsi(prices, period=14):
    if len(prices) < period + 1: return 50.0
    deltas = np.diff(prices)
    up = deltas[deltas >= 0].sum() / period
    down = -deltas[deltas < 0].sum() / period
    return 100.0 - (100.0 / (1.0 + (up / down))) if down != 0 else 100.0

# ========================= BOT CORE =========================
class DerivSniperBot:
    def __init__(self): # Fixed typo
        self.api = None
        self.app = None
        self.running = False
        self.account_mode = "Disconnected"
        self.active_token = None
        self.current_symbol = MARKETS[0]
        self.current_status = "🛑 Stopped"
        self.active_trade_info = None
        self.balance = "0.00"
        self.wins_today = 0
        self.losses_today = 0
        self.pnl_today = 0.0
        self.trade_lock = asyncio.Lock()
        self._scanner_task = None

    async def send(self, text: str):
        if not self.app: return
        try: await self.app.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=text, parse_mode="Markdown")
        except: pass

    async def connect(self) -> bool:
        try:
            self.api = DerivAPI(app_id=APP_ID)
            await self.api.authorize(self.active_token)
            await self.fetch_balance()
            return True
        except Exception as e:
            await self.send(f"❌ Connection failed: {e}")
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
            
            stake = money(max(float(STAKE), float(MIN_STAKE)))

            for attempt in range(2):
                try:
                    prop = await self.api.proposal({
                        "proposal": 1, "amount": stake, "basis": "stake",
                        "contract_type": side, "duration": DURATION,
                        "duration_unit": "m", "symbol": self.current_symbol,
                        "currency": "USD"
                    })

                    if "error" in prop:
                        msg = prop["error"].get("message", "")
                        min_req = extract_min_stake(msg)
                        if min_req and attempt == 0:
                            stake = min_req
                            continue
                        await self.send(f"❌ Proposal Error: {msg}")
                        return

                    proposal_id = prop["proposal"]["id"]
                    ask_price = float(prop["proposal"].get("ask_price", stake))
                    max_price = money(ask_price + 0.01)

                    resp = await self.api.buy({"buy": proposal_id, "price": max_price})
                    
                    if "error" in resp:
                        msg = resp["error"].get("message", "")
                        min_req = extract_min_stake(msg)
                        if min_req and attempt == 0:
                            stake = min_req
                            continue
                        await self.send(f"❌ Buy Error: {msg}")
                        return

                    cid = resp["buy"]["contract_id"]
                    self.active_trade_info = {"side": side, "id": cid}
                    await self.send(f"🚀 **TRADE PLACED**\nMarket: `{self.current_symbol}`\nStake: `${stake}`")
                    asyncio.create_task(self.check_result(cid))
                    return
                except Exception as e:
                    await self.send(f"⚠️ Error: {e}")
                    return

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
                await self.fetch_balance()
                await self.send(f"🏁 {res} | Profit: ${profit:.2f}\nBalance: {self.balance}")
        finally: self.active_trade_info = None

    def start_scanner(self):
        if not self._scanner_task or self._scanner_task.done():
            self._scanner_task = asyncio.create_task(self.run_scanner())

    async def run_scanner(self):
        self.running = True
        while self.running:
            await asyncio.sleep(10) # Scanner logic here

# ========================= TELEGRAM UI =========================
bot = DerivSniperBot()

async def btn_handler(u: Update, c: ContextTypes.DEFAULT_TYPE):
    q = u.callback_query
    await q.answer()
    if q.data == "PROMPT_MODE":
        kb = InlineKeyboardMarkup([[InlineKeyboardButton("🧪 DEMO", callback_data="SET_DEMO"), InlineKeyboardButton("💰 LIVE", callback_data="SET_REAL")]])
        await q.edit_message_text("Select Account Type:", reply_markup=kb)
    elif q.data in ("SET_DEMO", "SET_REAL"):
        bot.active_token = DEMO_TOKEN if q.data == "SET_DEMO" else REAL_TOKEN
        if await bot.connect():
            bot.start_scanner()
            await q.edit_message_text(f"✅ Online: {bot.current_symbol}", reply_markup=main_keyboard())
    elif q.data == "TEST_BUY":
        await bot.execute_trade("CALL")
    elif q.data == "STOP":
        bot.running = False
        await q.edit_message_text("🛑 Bot Stopped.")

def main_keyboard():
    return InlineKeyboardMarkup([[InlineKeyboardButton("🚀 START SCANNER", callback_data="PROMPT_MODE")], 
                                 [InlineKeyboardButton("📊 STATUS", callback_data="STATUS"), InlineKeyboardButton("🧪 TEST BUY", callback_data="TEST_BUY")],
                                 [InlineKeyboardButton("🛑 STOP", callback_data="STOP")]])

async def start_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    await u.message.reply_text("💎 **Deriv Sniper v4.0**", reply_markup=main_keyboard())

if __name__ == "__main__": # Fixed typo
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    bot.app = app
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CallbackQueryHandler(btn_handler))
    app.run_polling()
