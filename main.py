import asyncio
import logging
import time
from decimal import Decimal, ROUND_HALF_UP
from deriv_api import DerivAPI
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

# ========================= CONFIG =========================
DEMO_TOKEN = "YYSlMBIcTXqOozU"
REAL_TOKEN = "2NFJTH3JgXWFCcv"
APP_ID = 1089

MARKET = "R_10"  # Locking to R_10 for best trend stability
DURATION = 5 
EMA_PERIOD = 50  # 50 EMA for strong trend filtering
RSI_PERIOD = 14

TELEGRAM_TOKEN = "8276370676:AAGh5VqkG7b4cvpfRIVwY_rtaBlIiNwCTDM"
TELEGRAM_CHAT_ID = "7634818949"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ========================= MATH BRAIN =========================
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
        self.account_type = "None"
        self.current_symbol = MARKET
        self.is_scanning = False
        self.scanner_status = "💤 Offline"
        
        # Live Tracking
        self.active_trade_info = None 
        self.trade_start_time = 0
        self.last_rsi = 50
        self.m5_trend = "Neutral"
        
        # Survival Limits
        self.trades_today = 0
        self.losses_today = 0
        self.pnl_today = 0.0
        self.balance = "0.00"
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
            self.balance = f"{float(bal['balance']['balance']):.2f} {bal['balance']['currency']}"
        except: pass

    async def background_scanner(self):
        while self.is_scanning:
            # 🛑 CHECK SURVIVAL LIMITS (3 Trades or 3 Losses)
            if self.trades_today >= 3 or self.losses_today >= 3:
                self.is_scanning = False
                self.scanner_status = "🛑 DAILY LIMIT REACHED"
                await self.app.bot.send_message(TELEGRAM_CHAT_ID, "🏁 **Daily Session Complete.** (3 Trades/Losses Hit). Stopping bot for safety.")
                break

            try:
                # 1. GET M5 TREND (Filter)
                m5_data = await self.api.ticks_history({"ticks_history": MARKET, "end": "latest", "count": 100, "style": "ticks"})
                m5_prices = [float(p) for p in m5_data['history']['prices']]
                m5_ema = calculate_ema_manual(m5_prices, EMA_PERIOD)
                self.m5_trend = "CALL" if m5_prices[-1] > m5_ema else "PUT"

                # 2. GET M1 DATA (Entry Timing)
                m1_data = await self.api.ticks_history({"ticks_history": MARKET, "end": "latest", "count": 60, "style": "ticks"})
                m1_prices = [float(p) for p in m1_data['history']['prices']]
                self.last_rsi = calculate_rsi_manual(m1_prices, RSI_PERIOD)
                curr_p = m1_prices[-1]
                m1_ema = calculate_ema_manual(m1_prices, EMA_PERIOD)

                # 3. SIGNAL LOGIC (80% Price Action Pullback / 20% RSI)
                if self.m5_trend == "CALL":
                    # Pullback to EMA + RSI in warm range
                    if curr_p <= (m1_ema * 1.0005) and 45 <= self.last_rsi <= 60:
                        await self.execute_trade("CALL", "AUTO")
                
                elif self.m5_trend == "PUT":
                    # Bounce to EMA + RSI in cool range
                    if curr_p >= (m1_ema * 0.9995) and 40 <= self.last_rsi <= 55:
                        await self.execute_trade("PUT", "AUTO")

                self.scanner_status = f"📡 Scanning ({self.m5_trend})"
            except Exception as e:
                logger.error(f"Scanner Error: {e}")
            
            await asyncio.sleep(15)

    async def execute_trade(self, side: str, source="MANUAL"):
        if not self.api or self.active_trade_info: return
        async with self.trade_lock:
            try:
                proposal = await self.api.proposal({
                    "proposal": 1, "amount": 1.00, "basis": "payout",
                    "contract_type": side, "currency": "USD",
                    "duration": DURATION, "duration_unit": "m", "symbol": self.current_symbol
                })
                buy = await self.api.buy({"buy": proposal["proposal"]["id"], "price": 10.0})
                
                self.active_trade_info = buy["buy"]["contract_id"]
                self.trade_start_time = time.time()
                
                # Only increment daily trade count for AUTO/REAL trades, not manual tests
                if source == "AUTO":
                    self.trades_today += 1
                
                await self.app.bot.send_message(TELEGRAM_CHAT_ID, f"🚀 **{side} TRADE EXECUTED ({source})**\nTrend: {self.m5_trend}\nMarket: {MARKET}")
                asyncio.create_task(self.check_result(self.active_trade_info, source))
                await asyncio.sleep(305) # Prevent overlapping trades
            except Exception as e:
                logger.error(f"Execution Error: {e}")

    async def check_result(self, cid, source):
        await asyncio.sleep((DURATION * 60) + 10)
        try:
            res = await self.api.proposal_open_contract({"proposal_open_contract": 1, "contract_id": cid})
            profit = float(res['proposal_open_contract'].get('profit', 0))
            
            if source == "AUTO":
                self.pnl_today += profit
                if profit <= 0: self.losses_today += 1
            
            await self.fetch_balance()
            await self.app.bot.send_message(TELEGRAM_CHAT_ID, f"🏁 **TRADE FINISHED**\nResult: {'✅ WIN' if profit > 0 else '❌ LOSS'} (${profit:.2f})\nSession: {self.trades_today}/3 trades.")
        finally:
            self.active_trade_info = None

# ========================= UI =========================
bot_logic = DerivSniperBot()

def main_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("▶️ START SCANNER", callback_data="START_SCAN"), InlineKeyboardButton("⏹️ STOP", callback_data="STOP_SCAN")],
        [InlineKeyboardButton("🧪 TEST BUY", callback_data="TEST_BUY"), InlineKeyboardButton("📊 STATUS", callback_data="STATUS")],
        [InlineKeyboardButton("🧪 DEMO", callback_data="SET_DEMO"), InlineKeyboardButton("💰 LIVE", callback_data="SET_REAL")]
    ])

async def btn_handler(u: Update, c: ContextTypes.DEFAULT_TYPE):
    q = u.callback_query
    await q.answer()
    
    if q.data == "STATUS":
        await bot_logic.fetch_balance()
        status_header = f"🤖 **Bot State**: `{bot_logic.scanner_status}`\n🔑 **Account**: `{bot_logic.account_type}`\n"
        ind_text = f"📈 **M5 Trend**: `{bot_logic.m5_trend}`\n⚡ **RSI**: `{bot_logic.last_rsi:.1f}`\n"
        summary = f"\n💰 **Balance**: `{bot_logic.balance}`\n🎯 **Trades**: `{bot_logic.trades_today}/3` | **Losses**: `{bot_logic.losses_today}/3`"
        await q.edit_message_text(f"📊 **DETAILED STATUS**\n{status_header}{ind_text}{summary}", reply_markup=main_keyboard(), parse_mode="Markdown")

    elif q.data == "START_SCAN":
        if not bot_logic.api:
            await q.edit_message_text("❌ Connect Account First!", reply_markup=main_keyboard())
            return
        bot_logic.is_scanning = True
        asyncio.create_task(bot_logic.background_scanner())
        await q.edit_message_text("🔍 **SCANNER ACTIVE**\nStrategy: M5 Trend + M1 EMA Pullback", reply_markup=main_keyboard(), parse_mode="Markdown")

    elif q.data == "SET_DEMO":
        bot_logic.active_token = DEMO_TOKEN
        if await bot_logic.connect():
            bot_logic.account_type = "DEMO"
            await q.edit_message_text(f"✅ Connected to DEMO\nBal: {bot_logic.balance}", reply_markup=main_keyboard())
            
    elif q.data == "SET_REAL":
        bot_logic.active_token = REAL_TOKEN
        if await bot_logic.connect():
            bot_logic.account_type = "LIVE 💰"
            await q.edit_message_text(f"⚠️ **CONNECTED TO LIVE**\nBal: {bot_logic.balance}", reply_markup=main_keyboard(), parse_mode="Markdown")
            
    elif q.data == "TEST_BUY":
        await bot_logic.execute_trade("CALL", "MANUAL-TEST")

    elif q.data == "STOP_SCAN":
        bot_logic.is_scanning = False
        bot_logic.scanner_status = "💤 Offline"
        await q.edit_message_text("🛑 Scanner stopped.", reply_markup=main_keyboard())

async def start_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    await u.message.reply_text("💎 **Sniper v5.6 (Survival Mode)**", reply_markup=main_keyboard())

if __name__ == "__main__":
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    bot_logic.app = app
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CallbackQueryHandler(btn_handler))
    app.run_polling(drop_pending_updates=True)
