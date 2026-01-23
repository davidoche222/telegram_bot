import asyncio
import logging
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
from deriv_api import DerivAPI
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

# ========================= CONFIG =========================
DEMO_TOKEN = "tIrfitLjqeBxCOM"
REAL_TOKEN = "2hsJzopRHG5w"
APP_ID = 1089

MARKETS = ["R_10", "R_25", "R_50"]

COOLDOWN_SEC = 120
MAX_TRADES_PER_DAY = 60
MAX_CONSEC_LOSSES = 10

TELEGRAM_TOKEN = "8253450930:AAHUhPk9TML-8kZlA9UaHZZvTUGdurN9MVY"
TELEGRAM_CHAT_ID = "7634818949"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ========================= STRATEGY SETTINGS =========================
TF_SEC = 60
CANDLES_COUNT = 150
SCAN_SLEEP_SEC = 2
RSI_PERIOD = 14
DURATION_MIN = 1

# ========================= EMA50 SLOPE + MARTINGALE =========================
EMA_SLOPE_LOOKBACK = 10
EMA_SLOPE_MIN = 0.0

MARTINGALE_MULT = 2.0
MARTINGALE_MAX_STEPS = 4
MARTINGALE_MAX_STAKE = 8.00  # Max stake allowed during martingale

DAILY_PROFIT_TARGET = 5.0

# ========================= STAKE SETTINGS (FIXED) =========================
BASE_STAKE = 0.40  # <--- YOUR REQUESTED RISK
MIN_STAKE = 0.35   # Deriv API Minimum floor

# ========================= INDICATOR MATH =========================
def calculate_ema(values, period: int):
    values = np.array(values, dtype=float)
    if len(values) < period:
        return np.array([])
    k = 2.0 / (period + 1.0)
    ema = np.zeros_like(values)
    ema[0] = values[0]
    for i in range(1, len(values)):
        ema[i] = values[i] * k + ema[i - 1] * (1 - k)
    return ema

def calculate_rsi(values, period=14):
    values = np.array(values, dtype=float)
    n = len(values)
    if n < period + 2:
        return np.array([])
    deltas = np.diff(values)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    rsi = np.full(n, np.nan, dtype=float)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    rs = avg_gain / (avg_loss + 1e-12)
    rsi[period] = 100.0 - (100.0 / (1.0 + rs))
    for i in range(period + 1, n):
        gain = gains[i - 1]
        loss = losses[i - 1]
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        rs = avg_gain / (avg_loss + 1e-12)
        rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def build_candles_from_deriv(candles_raw):
    return [{"t0": int(x.get("epoch", 0)), "o": float(x.get("open", 0)), 
             "h": float(x.get("high", 0)), "l": float(x.get("low", 0)), 
             "c": float(x.get("close", 0))} for x in candles_raw]

# ========================= BOT CORE =========================
class DerivSniperBot:
    def __init__(self):
        self.api = None
        self.app = None
        self.active_token = None
        self.account_type = "None"
        self.is_scanning = False
        self.active_trade_info = None
        self.active_market = "None"
        self.trade_start_time = 0.0
        self.cooldown_until = 0.0
        self.trades_today = 0
        self.total_losses_today = 0
        self.consecutive_losses = 0
        self.total_profit_today = 0.0
        self.balance = "0.00"
        self.trade_lock = asyncio.Lock()
        self.market_debug = {m: {} for m in MARKETS}
        self.last_processed_closed_t0 = {m: 0 for m in MARKETS}
        self.tz = ZoneInfo("Africa/Lagos")
        self.current_day = datetime.now(self.tz).date()
        self.pause_until = 0.0

        # ✅ FIXED STAKE LOGIC
        self.base_stake = float(BASE_STAKE)
        self.current_stake = self.base_stake
        self.martingale_step = 0

    async def connect(self) -> bool:
        try:
            if not self.active_token: return False
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
            bal = await self.api.balance({"balance": 1})
            self.balance = f"{float(bal['balance']['balance']):.2f} {bal['balance']['currency']}"
        except: pass

    def _next_midnight_epoch(self) -> float:
        now = datetime.now(self.tz)
        next_midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        return next_midnight.timestamp()

    def _daily_reset_if_needed(self):
        today = datetime.now(self.tz).date()
        if today != self.current_day:
            self.current_day, self.trades_today, self.total_losses_today = today, 0, 0
            self.consecutive_losses, self.total_profit_today, self.cooldown_until = 0, 0.0, 0.0
            self.pause_until, self.martingale_step = 0.0, 0
            self.current_stake = self.base_stake

    def can_auto_trade(self) -> tuple[bool, str]:
        self._daily_reset_if_needed()
        if time.time() < self.pause_until: return False, "Paused until midnight"
        if self.total_profit_today >= DAILY_PROFIT_TARGET: return False, "Target reached"
        if self.consecutive_losses >= MAX_CONSEC_LOSSES: return False, "Max losses"
        if self.trades_today >= MAX_TRADES_PER_DAY: return False, "Limit reached"
        if time.time() < self.cooldown_until: return False, "Cooldown"
        if self.active_trade_info: return False, "Trade active"
        if not self.api: return False, "No API"
        return True, "OK"

    async def fetch_real_m1_candles(self, symbol: str):
        payload = {"ticks_history": symbol, "end": "latest", "count": CANDLES_COUNT, "style": "candles", "granularity": TF_SEC}
        data = await self.api.ticks_history(payload)
        return build_candles_from_deriv(data.get("candles", []))

    async def execute_trade(self, side: str, symbol: str, reason="MANUAL", source="MANUAL"):
        if not self.api or self.active_trade_info: return
        async with self.trade_lock:
            ok, _ = self.can_auto_trade()
            if not ok: return
            try:
                # ✅ Set exact stake
                stake = float(self.current_stake if source == "AUTO" else self.base_stake)
                stake = round(max(stake, MIN_STAKE), 2)

                prop = await self.api.proposal({
                    "proposal": 1,
                    "amount": stake,
                    "basis": "stake", # <--- NOW USING STAKE BASIS
                    "contract_type": side,
                    "currency": "USD",
                    "duration": int(DURATION_MIN),
                    "duration_unit": "m",
                    "symbol": symbol
                })

                if "error" in prop:
                    logger.error(f"Prop Error: {prop['error']['message']}")
                    return

                buy = await self.api.buy({"buy": prop["proposal"]["id"], "price": stake})
                self.active_trade_info = int(buy["buy"]["contract_id"])
                self.active_market, self.trade_start_time = symbol, time.time()
                if source == "AUTO": self.trades_today += 1

                msg = (f"🚀 {side} OPENED\n🛒 Market: {symbol}\n💵 Stake: ${stake:.2f}\n🧠 {reason}")
                await self.app.bot.send_message(TELEGRAM_CHAT_ID, msg)
                asyncio.create_task(self.check_result(self.active_trade_info, source, stake_used=stake))
            except Exception as e:
                logger.error(f"Trade error: {e}")

    async def check_result(self, cid: int, source: str, stake_used: float):
        await asyncio.sleep(int(DURATION_MIN) * 60 + 5)
        try:
            res = await self.api.proposal_open_contract({"proposal_open_contract": 1, "contract_id": cid})
            profit = float(res["proposal_open_contract"].get("profit", 0))
            if source == "AUTO":
                self.total_profit_today += profit
                if profit <= 0:
                    self.consecutive_losses += 1
                    self.martingale_step = min(self.martingale_step + 1, MARTINGALE_MAX_STEPS)
                    self.current_stake = min(stake_used * MARTINGALE_MULT, MARTINGALE_MAX_STAKE)
                else:
                    self.consecutive_losses, self.martingale_step = 0, 0
                    self.current_stake = self.base_stake

            await self.fetch_balance()
            res_msg = "WIN ✅" if profit > 0 else "LOSS ❌"
            await self.app.bot.send_message(TELEGRAM_CHAT_ID, f"🏁 {res_msg} ({profit:+.2f})\n💰 PnL: {self.total_profit_today:.2f}")
        finally:
            self.active_trade_info = None
            self.cooldown_until = time.time() + COOLDOWN_SEC

    # ... (Rest of scanner logic remains the same, just calling execute_trade)
    async def background_scanner(self):
        self.market_tasks = {sym: asyncio.create_task(self.scan_market(sym)) for sym in MARKETS}
        while self.is_scanning: await asyncio.sleep(1)

    async def scan_market(self, symbol: str):
        while self.is_scanning:
            try:
                ok, gate = self.can_auto_trade()
                candles = await self.fetch_real_m1_candles(symbol)
                if len(candles) < 70: 
                    await asyncio.sleep(5)
                    continue
                
                # ... (Insert your RSI/EMA strategy logic here from your original script)
                # For brevity, I'm focusing on the trade execution fix.
                # When a signal is found:
                # await self.execute_trade("CALL", symbol, "Strategy", "AUTO")
                
            except Exception as e: logger.error(e)
            await asyncio.sleep(SCAN_SLEEP_SEC)

# ========================= UI =========================
bot_logic = DerivSniperBot()

def main_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("▶️ START", callback_data="START_SCAN"), InlineKeyboardButton("⏹️ STOP", callback_data="STOP_SCAN")],
        [InlineKeyboardButton("📊 STATUS", callback_data="STATUS")],
        [InlineKeyboardButton("💰 LIVE", callback_data="SET_REAL"), InlineKeyboardButton("🧪 DEMO", callback_data="SET_DEMO")],
        [InlineKeyboardButton("➖ Stake -0.05", callback_data="STAKE_DOWN"), InlineKeyboardButton("➕ Stake +0.05", callback_data="STAKE_UP")]
    ])

async def btn_handler(u: Update, c: ContextTypes.DEFAULT_TYPE):
    q = u.callback_query
    await q.answer()
    if q.data == "SET_REAL":
        bot_logic.active_token, bot_logic.account_type = REAL_TOKEN, "LIVE"
        await bot_logic.connect()
    elif q.data == "STAKE_UP":
        bot_logic.base_stake = round(bot_logic.base_stake + 0.05, 2)
        bot_logic.current_stake = bot_logic.base_stake
    elif q.data == "STAKE_DOWN":
        bot_logic.base_stake = round(max(bot_logic.base_stake - 0.05, MIN_STAKE), 2)
        bot_logic.current_stake = bot_logic.base_stake
    elif q.data == "START_SCAN":
        bot_logic.is_scanning = True
        asyncio.create_task(bot_logic.background_scanner())
    
    await q.edit_message_text(f"Bot Status: {bot_logic.account_type}\nStake: ${bot_logic.base_stake:.2f}\nBalance: {bot_logic.balance}", reply_markup=main_keyboard())

async def start_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    await u.message.reply_text(f"💎 Deriv Stake Bot\nRisking: ${bot_logic.base_stake:.2f}", reply_markup=main_keyboard())

if __name__ == "__main__":
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    bot_logic.app = app
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CallbackQueryHandler(btn_handler))
    app.run_polling()
