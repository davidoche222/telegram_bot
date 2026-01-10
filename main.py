import asyncio
import logging
import pandas as pd
import numpy as np
from deriv_api import DerivAPI
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

# ========================= CONFIG =========================
DEMO_TOKEN = "YYSlMBIcTXqOozU"
REAL_TOKEN = "2NFJTH3JgXWFCcv"
APP_ID = 1089

MARKETS = ["R_10", "R_25", "R_50", "R_100"]
DURATION = 5 
EMA_PERIOD = 20
RSI_PERIOD = 14

TELEGRAM_TOKEN = "8276370676:AAGh5VqkG7b4cvpfRIVwY_rtaBlIiNwCTDM"
TELEGRAM_CHAT_ID = "7634818949"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ========================= STRATEGY =========================
def calculate_indicators(ticks):
    df = pd.DataFrame(ticks)
    # EMA
    df['ema'] = df['quote'].ewm(span=EMA_PERIOD, adjust=False).mean()
    # RSI
    delta = df['quote'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=RSI_PERIOD).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=RSI_PERIOD).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    return df.iloc[-1]

# ========================= BOT CORE =========================
class DerivSniperBot:
    def __init__(self):
        self.api = None
        self.app = None
        self.active_token = None
        self.current_symbol = MARKETS[0]
        self.is_scanning = False
        self.active_trade_info = None
        self.trade_lock = asyncio.Lock()
        
        self.balance = "0.00"
        self.pnl_today = 0.0
        self.wins_today = 0
        self.losses_today = 0

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
            self.balance = f"{float(bal['balance']['balance']):.2f} USD"
        except: pass

    async def background_scanner(self):
        logger.info("🔍 Scanner starting...")
        while self.is_scanning:
            try:
                ticks_data = await self.api.ticks_history({
                    "ticks_history": self.current_symbol,
                    "end": "latest", "count": 50, "style": "ticks"
                })
                prices = [{"quote": float(t['quote'])} for t in ticks_data['history']['prices']]
                latest = calculate_indicators(prices)
                
                # Signal Logic
                if latest['quote'] > latest['ema'] and latest['rsi'] < 40:
                    await self.execute_trade("CALL", "AUTO-STRATEGY")
                    await asyncio.sleep(305) # Prevent double trades
                elif latest['quote'] < latest['ema'] and latest['rsi'] > 60:
                    await self.execute_trade("PUT", "AUTO-STRATEGY")
                    await asyncio.sleep(305)
            except Exception as e:
                logger.error(f"Scanner Loop Error: {e}")
            await asyncio.sleep(15)

    async def execute_trade(self, side: str, source="MANUAL"):
        if not self.api or self.active_trade_info: return
        async with self.trade_lock:
            try:
                # Use Payout basis so Deriv decides the stake
                proposal = await self.api.proposal({
                    "proposal": 1, "amount": 1.00, "basis": "payout",
                    "contract_type": side, "currency": "USD",
                    "duration": DURATION, "duration_unit": "m", "symbol": self.current_symbol
                })
                
                if "error" in proposal:
                    await self.app.bot.send_message(TELEGRAM_CHAT_ID, f"❌ Error: {proposal['error']['message']}")
                    return

                buy = await self.api.buy({"buy": proposal["proposal"]["id"], "price": 10.0})
                self.active_trade_info = buy["buy"]["contract_id"]
                
                await self.app.bot.send_message(
                    TELEGRAM_CHAT_ID, 
                    f"🚀 **{source} TRADE**\nSide: {side}\nStake: ${buy['buy']['buy_price']}\nMarket: {self.current_symbol}"
                )
                asyncio.create_task(self.check_result(self.active_trade_info))
            except Exception as e:
                logger.error(f"Trade Execution Error: {e}")

    async def check_result(self, cid):
        await asyncio.sleep((DURATION * 60) + 10)
        try:
            res = await self.api.proposal_open_contract({"proposal_open_contract": 1, "contract_id": cid})
            c = res['proposal_open_contract']
            profit = float(c.get('profit', 0))
            self.pnl_today += profit
            if profit > 0: self.wins_today += 1 
            else: self.losses_today += 1
            await self.fetch_balance()
            await self.app.bot.send_message(TELEGRAM_CHAT_ID, f"🏁 **FINISH**\nResult: {'✅ WIN' if profit > 0 else '❌ LOSS'} (${profit:.2f})")
        finally:
            self.active_trade_info = None

# ========================= UI =========================
bot_logic = DerivSniperBot()

def main_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🧪 DEMO", callback_data="SET_DEMO"), InlineKeyboardButton("💰 LIVE", callback_data="SET_REAL")],
        [InlineKeyboardButton("▶️ START SCANNER", callback_data="START_SCAN"), InlineKeyboardButton("⏹️ STOP", callback_data="STOP_SCAN")],
        [InlineKeyboardButton("🧪 TEST BUY (CALL)", callback_data="TEST_BUY"), InlineKeyboardButton("📊 STATUS", callback_data="STATUS")]
    ])

async def btn_handler(u: Update, c: ContextTypes.DEFAULT_TYPE):
    q = u.callback_query
    await q.answer()
    
    if q.data in ("SET_DEMO", "SET_REAL"):
        bot_logic.active_token = DEMO_TOKEN if q.data == "SET_DEMO" else REAL_TOKEN
        if await bot_logic.connect():
            await q.edit_message_text(f"✅ Connected!\nBal: {bot_logic.balance}", reply_markup=main_keyboard())
    
    elif q.data == "START_SCAN":
        bot_logic.is_scanning = True
        asyncio.create_task(bot_logic.background_scanner())
        await q.edit_message_text("🔍 **SCANNER ON** (RSI/EMA)", reply_markup=main_keyboard())

    elif q.data == "STOP_SCAN":
        bot_logic.is_scanning = False
        await q.edit_message_text("🛑 **SCANNER OFF**", reply_markup=main_keyboard())

    elif q.data == "TEST_BUY":
        await bot_logic.execute_trade("CALL", "MANUAL-TEST")

    elif q.data == "STATUS":
        await bot_logic.fetch_balance()
        txt = f"📊 **STATUS**\n💰 Bal: {bot_logic.balance}\n🏆 W/L: {bot_logic.wins_today}/{bot_logic.losses_today}\n💵 PnL: ${bot_logic.pnl_today:.2f}"
        await q.edit_message_text(txt, reply_markup=main_keyboard())

async def start_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    await u.message.reply_text("💎 **Sniper v4.8 Strategy + Manual**", reply_markup=main_keyboard())

if __name__ == "__main__":
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    bot_logic.app = app
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CallbackQueryHandler(btn_handler))
    app.run_polling(drop_pending_updates=True)
