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

# List for the Auto-Switcher
MARKETS = ["R_10", "R_25", "R_50", "R_100"]
current_market_idx = 0

STAKE = 0.35
DURATION = 5  # minutes

TELEGRAM_TOKEN = "8276370676:AAGh5VqkG7b4cvpfRIVwY_rtaBlIiNwCTDM"
TELEGRAM_CHAT_ID = "7634818949"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ========================= STRATEGY HELPERS =========================
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

# ========================= BOT CORE =========================
class DerivSniperBot:
    def __init__(self):
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
            logger.error(f"Connect error: {e}")
            return False

    async def fetch_balance(self):
        if not self.api: return
        try:
            bal = await self.api.balance()
            b = bal.get("balance", {})
            self.balance = f"{float(b.get('balance', 0.0)):.2f} USD"
        except: pass

    async def get_candles(self, sym, gran):
        try:
            resp = await self.api.ticks_history({"ticks_history": sym, "count": 60, "end": "latest", "granularity": gran, "style": "candles"})
            return resp.get("candles")
        except: return None

    async def switch_market(self):
        global current_market_idx
        current_market_idx = (current_market_idx + 1) % len(MARKETS)
        self.current_symbol = MARKETS[current_market_idx]
        await self.send(f"🔄 **Switching to {self.current_symbol}** (Limit bypass)")

    async def run_scanner(self):
        self.running = True
        while self.running:
            try:
                if not self.api: await self.connect()
                if self.active_trade_info:
                    await asyncio.sleep(15)
                    continue

                m5 = await self.get_candles(self.current_symbol, 300)
                m1 = await self.get_candles(self.current_symbol, 60)
                
                if m1 and m5:
                    m1_c = [float(x["close"]) for x in m1]
                    rsi = get_rsi(m1_c, 14)
                    self.current_status = f"🔎 {self.current_symbol} | RSI: {round(rsi)}"
                
                await asyncio.sleep(10)
            except Exception as e:
                logger.error(f"Scanner Loop Error: {e}")
                await asyncio.sleep(10)

    async def execute_trade(self, side: str):
        async with self.trade_lock:
            if self.active_trade_info or not self.api: return
            try:
                # Direct Buy attempt with limit check
                req = {"buy": 1, "price": STAKE, "parameters": {"amount": STAKE, "basis": "stake", "contract_type": side, "currency": "USD", "duration": DURATION, "duration_unit": "m", "symbol": self.current_symbol}}
                resp = await self.api.buy(req)

                if "error" in resp:
                    err = resp['error'].get('message', "").lower()
                    if "maximum purchase price" in err:
                        await self.switch_market()
                    else:
                        await self.send(f"❌ **Error:** {err}")
                    return

                cid = resp["buy"]["contract_id"]
                self.active_trade_info = {"side": side, "id": cid}
                await self.send(f"🚀 **TRADE PLACED**\nMarket: {self.current_symbol}\nSide: {side}")
                asyncio.create_task(self.check_result(cid))
            except Exception as e: logger.error(f"Trade Error: {e}")

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
                await self.send(f"🏁 **{res}** | Profit: ${profit:.2f}\nNew Balance: {self.balance}")
        finally: self.active_trade_info = None

    def start_scanner(self):
        if self._scanner_task and not self._scanner_task.done(): return
        self._scanner_task = asyncio.create_task(self.run_scanner())

# ========================= TELEGRAM UI =========================
bot = DerivSniperBot()

def main_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🚀 START SCANNER", callback_data="PROMPT_MODE")],
        [InlineKeyboardButton("📊 STATUS", callback_data="STATUS"), InlineKeyboardButton("🧪 TEST BUY", callback_data="TEST_BUY")],
        [InlineKeyboardButton("🛑 STOP", callback_data="STOP")]
    ])

async def start_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    await u.message.reply_text("💎 **Deriv Sniper v3.4**", reply_markup=main_keyboard())

async def btn_handler(u: Update, c: ContextTypes.DEFAULT_TYPE):
    q = u.callback_query
    await q.answer()
    if q.data == "PROMPT_MODE":
        kb = InlineKeyboardMarkup([[InlineKeyboardButton("🧪 DEMO", callback_data="SET_DEMO"), InlineKeyboardButton("💰 LIVE", callback_data="SET_REAL")]])
        await q.edit_message_text("Select Account:", reply_markup=kb)
    elif q.data in ("SET_DEMO", "SET_REAL"):
        bot.active_token = DEMO_TOKEN if q.data == "SET_DEMO" else REAL_TOKEN
        if await bot.connect():
            bot.start_scanner()
            await q.edit_message_text(f"✅ Scanner Online: {bot.current_symbol}", reply_markup=main_keyboard())
    elif q.data == "TEST_BUY":
        await bot.execute_trade("CALL")
    elif q.data == "STATUS":
        await bot.fetch_balance()
        pnl_str = f"+${bot.pnl_today:.2f}" if bot.pnl_today >= 0 else f"-${abs(bot.pnl_today):.2f}"
        msg = f"📊 DASHBOARD\n💰 Balance: {bot.balance}\n📉 PnL Today: {pnl_str}\n🏆 W/L: {bot.wins_today}/{bot.losses_today}\n📡 Status: {bot.current_status}"
        try: await q.edit_message_text(msg, reply_markup=main_keyboard())
        except BadRequest: pass
    elif q.data == "STOP":
        bot.running = False
        await q.edit_message_text("🛑 Bot Stopped.")

if __name__ == "__main__":
    app = Application.builder().token(TELEGRAM_TOKEN).post_init(lambda a: a.bot.delete_webhook(drop_pending_updates=True)).build()
    bot.app = app
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CallbackQueryHandler(btn_handler))
    app.run_polling(drop_pending_updates=True)
