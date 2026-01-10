import asyncio
import logging
from decimal import Decimal, ROUND_HALF_UP
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

# ========================= MANUAL MATH BRAIN =========================
def calculate_ema_manual(prices, period):
    if len(prices) < period: return prices[-1]
    multiplier = 2 / (period + 1)
    ema = sum(prices[:period]) / period
    for price in prices[period:]:
        ema = (price - ema) * multiplier + ema
    return ema

def calculate_rsi_manual(prices, period):
    if len(prices) < period + 1: return 50
    gains, losses = [], []
    for i in range(1, len(prices)):
        change = prices[i] - prices[i-1]
        gains.append(max(0, change))
        losses.append(max(0, -change))
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    if avg_loss == 0: return 100
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    return 100 - (100 / (1 + rs))

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
        self.pnl_today, self.wins_today, self.losses_today = 0.0, 0, 0

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
        logger.info("🔍 Scanner active...")
        while self.is_scanning:
            if not self.api:
                await asyncio.sleep(5)
                continue

            try:
                # 1. Fetch History
                ticks_data = await self.api.ticks_history({
                    "ticks_history": self.current_symbol,
                    "end": "latest", "count": 60, "style": "ticks"
                })
                
                # 🔥 THE FIX: Flexible Data Extraction
                raw_prices = ticks_data.get('history', {}).get('prices', [])
                
                clean_prices = []
                for p in raw_prices:
                    if isinstance(p, dict):
                        clean_prices.append(float(p.get('quote', 0)))
                    else:
                        clean_prices.append(float(p))

                if not clean_prices:
                    logger.warning("No price data received.")
                    await asyncio.sleep(5)
                    continue

                cur_price = clean_prices[-1]
                ema_val = calculate_ema_manual(clean_prices, EMA_PERIOD)
                rsi_val = calculate_rsi_manual(clean_prices, RSI_PERIOD)

                logger.info(f"📊 {self.current_symbol} | Price: {cur_price:.2f} | RSI: {rsi_val:.1f} | EMA: {ema_val:.2f}")

                # 2. STRATEGY TRIGGER
                if cur_price > ema_val and rsi_val < 40:
                    await self.execute_trade("CALL", "AUTO-SIGNAL")
                    await asyncio.sleep(305) 
                elif cur_price < ema_val and rsi_val > 60:
                    await self.execute_trade("PUT", "AUTO-SIGNAL")
                    await asyncio.sleep(305)
                    
            except Exception as e:
                logger.error(f"Scanner Loop Error: {e}")
            
            await asyncio.sleep(10)

    async def execute_trade(self, side: str, source="MANUAL"):
        if not self.api or self.active_trade_info: return
        async with self.trade_lock:
            try:
                proposal = await self.api.proposal({
                    "proposal": 1, "amount": 1.00, "basis": "payout",
                    "contract_type": side, "currency": "USD",
                    "duration": DURATION, "duration_unit": "m", "symbol": self.current_symbol
                })
                if "error" in proposal: return

                buy = await self.api.buy({"buy": proposal["proposal"]["id"], "price": 10.0})
                self.active_trade_info = buy["buy"]["contract_id"]
                
                await self.app.bot.send_message(
                    TELEGRAM_CHAT_ID, 
                    f"🚀 **{source}**\n{side} on {self.current_symbol}\nStake: ${buy['buy']['buy_price']}"
                )
                asyncio.create_task(self.check_result(self.active_trade_info))
            except: pass

    async def check_result(self, cid):
        await asyncio.sleep((DURATION * 60) + 10)
        try:
            res = await self.api.proposal_open_contract({"proposal_open_contract": 1, "contract_id": cid})
            profit = float(res['proposal_open_contract'].get('profit', 0))
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
        [InlineKeyboardButton("🧪 CONNECT DEMO", callback_data="SET_DEMO"), InlineKeyboardButton("💰 CONNECT LIVE", callback_data="SET_REAL")],
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
        await q.edit_message_text("🔍 **SCANNER ACTIVE**\nWatching RSI & EMA...", reply_markup=main_keyboard())
    elif q.data == "STOP_SCAN":
        bot_logic.is_scanning = False
        await q.edit_message_text("🛑 **SCANNER STOPPED**", reply_markup=main_keyboard())
    elif q.data == "TEST_BUY":
        await bot_logic.execute_trade("CALL", "MANUAL-TEST")
    elif q.data == "STATUS":
        await bot_logic.fetch_balance()
        txt = f"📊 **STATS**\n💰 Bal: {bot_logic.balance}\n🏆 W/L: {bot_logic.wins_today}/{bot_logic.losses_today}\n💵 PnL: ${bot_logic.pnl_today:.2f}"
        await q.edit_message_text(txt, reply_markup=main_keyboard(), parse_mode="Markdown")

async def start_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    await u.message.reply_text("💎 **Sniper v5.1 (Data Fix)**", reply_markup=main_keyboard())

if __name__ == "__main__":
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    bot_logic.app = app
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CallbackQueryHandler(btn_handler))
    app.run_polling(drop_pending_updates=True)
