import asyncio
import datetime
import logging
import numpy as np
import os
from deriv_api import DerivAPI
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.error import BadRequest
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

# ========================= 1. CONFIG =========================
# REPLACE THESE WITH YOUR ACTUAL TOKENS
DEMO_TOKEN = "YYSlMBIcTXqOozU"
REAL_TOKEN = "2NFJTH3JgXWFCcv"
APP_ID = "1089"

SYMBOL = "R_50"       # R_50 and R_100 are best for this strategy
STAKE = 0.35
DURATION = 15         # 15 minutes ensures Deriv accepts the $0.35 stake

TELEGRAM_TOKEN = "8276370676:AAGh5VqkG7b4cvpfRIVwY_rtaBlIiNwCTDM"
TELEGRAM_CHAT_ID = "7634818949"

# ========================= 2. LOGGING SETUP =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("bot_log.txt"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ========================= 3. STRATEGY ENGINE =========================
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

def check_v3_momentum(candle, prev_3, direction, ema):
    try:
        o, h, l, c = float(candle["open"]), float(candle["high"]), float(candle["low"]), float(candle["close"])
        rng = h - l
        if rng <= 0 or ema is None: return False

        # Rule: Candle must touch the EMA
        if not (l <= ema <= h): return False

        avg_body = sum(abs(float(x["close"]) - float(x["open"])) for x in prev_3) / 3.0
        if abs(c - o) <= avg_body: return False

        if direction == "CALL":
            return (h - c) / rng <= 0.30 and c > float(prev_3[-1]["high"]) and c > o
        else:
            return (c - l) / rng <= 0.30 and c < float(prev_3[-1]["low"]) and c < o
    except: return False

# ========================= 4. BOT CORE =========================
class DerivSniperBot:
    def __init__(self):
        self.api = None
        self.app = None
        self.running = False
        self.account_mode = "Disconnected"
        self.active_token = None
        self.current_status = "🛑 Stopped"
        self.active_trade_info = None
        self.wins_today, self.losses_today, self.pnl_today = 0, 0, 0.0
        self.trade_lock = asyncio.Lock()

    async def send(self, text):
        try: await self.app.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=text, parse_mode="Markdown")
        except: pass

    async def connect(self):
        try:
            self.api = DerivAPI(app_id=APP_ID)
            await self.api.authorize(self.active_token)
            logger.info(f"Connected to {self.account_mode}")
            return True
        except Exception as e:
            logger.error(f"Connect error: {e}")
            return False

    async def get_candles(self, gran):
        resp = await self.api.ticks_history({"ticks_history": SYMBOL, "count": 60, "end": "latest", "granularity": gran, "style": "candles"})
        return resp["candles"]

    async def run_scanner(self):
        self.running = True
        logger.info("Scanner started.")
        while self.running:
            try:
                if not self.api: await self.connect()
                if self.active_trade_info:
                    await asyncio.sleep(15)
                    continue

                m5 = await self.get_candles(300)
                m1 = await self.get_candles(60)
                m5_c, m1_c = [float(x["close"]) for x in m5], [float(x["close"]) for x in m1]
                m5_ema, m1_ema = get_ema(m5_c, 50), get_ema(m1_c, 50)
                rsi = get_rsi(m1_c, 14)

                self.current_status = f"🔎 Scanning (RSI: {round(rsi)})"
                
                if m5_ema and m1_ema and 30 <= rsi <= 70:
                    m5_bias = "CALL" if m5_c[-1] > m5_ema else "PUT"
                    last_c, prev_3 = m1[-1], m1[-4:-1]
                    
                    signal = None
                    if m5_bias == "CALL" and m1_c[-1] > m1_ema and check_v3_momentum(last_c, prev_3, "CALL", m1_ema):
                        signal = "CALL"
                    elif m5_bias == "PUT" and m1_c[-1] < m1_ema and check_v3_momentum(last_c, prev_3, "PUT", m1_ema):
                        signal = "PUT"

                    if signal:
                        logger.info(f"STRATEGY SIGNAL: {signal}")
                        await self.execute_trade(signal)

                await asyncio.sleep(10)
            except Exception as e:
                logger.error(f"Scanner Loop Error: {e}")
                await asyncio.sleep(10)

    async def execute_trade(self, side):
        async with self.trade_lock:
            if self.active_trade_info: return
            try:
                req = {"buy": 1, "price": STAKE, "parameters": {"amount": STAKE, "basis": "stake", "contract_type": side, "currency": "USD", "duration": DURATION, "duration_unit": "m", "symbol": SYMBOL}}
                resp = await self.api.buy(req)
                
                if "error" in resp:
                    err_msg = resp['error'].get('message')
                    logger.warning(f"Trade Refused by Deriv: {err_msg}")
                    await self.send(f"❌ **Trade Refused:**\n`{err_msg}`")
                    return

                cid = resp["buy"]["contract_id"]
                self.active_trade_info = {"side": side, "id": cid}
                logger.info(f"BUY SUCCESS: {side} ID {cid}")
                await self.send(f"🚀 **TRADE PLACED**\nSide: `{side}`\nStake: `${STAKE}`")
                asyncio.create_task(self.check_result(cid))
            except Exception as e:
                logger.error(f"Execution Error: {e}")

    async def check_result(self, cid):
        await asyncio.sleep(DURATION * 60 + 5)
        try:
            table = await self.api.profit_table({"limit": 10})
            for trans in table["profit_table"]["transactions"]:
                if str(trans.get("contract_id")) == str(cid):
                    profit = float(trans["sell_price"]) - float(trans["buy_price"])
                    self.pnl_today += profit
                    res = "✅ WIN" if profit > 0 else "❌ LOSS"
                    if profit > 0: self.wins_today += 1
                    else: self.losses_today += 1
                    logger.info(f"TRADE CLOSED: {res} (${profit})")
                    await self.send(f"🏁 **RESULT**: {res}\nProfit: `${round(profit, 2)}`")
                    break
        except Exception as e: logger.error(f"Result Check Error: {e}")
        finally: self.active_trade_info = None

# ========================= 5. TELEGRAM UI =========================
bot = DerivSniperBot()

def main_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🚀 START SCANNER", callback_data="PROMPT_MODE")],
        [InlineKeyboardButton("📊 STATUS", callback_data="STATUS"), InlineKeyboardButton("🧪 TEST BUY", callback_data="TEST_BUY")],
        [InlineKeyboardButton("🛑 STOP", callback_data="STOP")]
    ])

async def start_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    await u.message.reply_text("💎 **Deriv Sniper v3.1**\nLogic: EMA Touch + RSI Filter", reply_markup=main_keyboard())

async def logs_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    """Sends the last 10 lines of the log file to Telegram"""
    try:
        with open("bot_log.txt", "r") as f:
            lines = f.readlines()
            last_lines = "".join(lines[-10:])
            await u.message.reply_text(f"📜 **Latest Logs:**\n```\n{last_lines}\n```", parse_mode="Markdown")
    except: await u.message.reply_text("❌ No log file found.")

async def btn_handler(u: Update, c: ContextTypes.DEFAULT_TYPE):
    q = u.callback_query
    await q.answer()
    if q.data == "PROMPT_MODE":
        await q.edit_message_text("💳 **Select Account:**", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🧪 DEMO", callback_data="SET_DEMO"), InlineKeyboardButton("💰 LIVE", callback_data="SET_REAL")]]))
    elif q.data in ("SET_DEMO", "SET_REAL"):
        bot.active_token = DEMO_TOKEN if q.data == "SET_DEMO" else REAL_TOKEN
        bot.account_mode = "🧪 DEMO" if q.data == "SET_DEMO" else "💰 REAL"
        bot.running = False
        await asyncio.sleep(1)
        if await bot.connect():
            asyncio.create_task(bot.run_scanner())
            await q.edit_message_text(f"✅ **Scanner Online** ({bot.account_mode})", reply_markup=main_keyboard())
    elif q.data == "TEST_BUY":
        if bot.api: await bot.execute_trade("CALL")
        else: await q.edit_message_text("❌ Connect account first!")
    elif q.data == "STATUS":
        pnl_str = f"+${round(bot.pnl_today, 2)}" if bot.pnl_today >= 0 else f"-${round(abs(bot.pnl_today), 2)}"
        msg = f"📊 **DASHBOARD**\nStatus: `{bot.current_status}`\nPnL: `{pnl_str}`\nWins: {bot.wins_today} | Loss: {bot.losses_today}"
        try: await q.edit_message_text(msg, reply_markup=main_keyboard(), parse_mode="Markdown")
        except: pass
    elif q.data == "STOP":
        bot.running = False
        await q.edit_message_text("🛑 Bot Stopped.")

async def post_init(application: Application):
    await application.bot.delete_webhook(drop_pending_updates=True)

if __name__ == "__main__":
    app = Application.builder().token(TELEGRAM_TOKEN).post_init(post_init).build()
    bot.app = app
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("logs", logs_cmd))
    app.add_handler(CallbackQueryHandler(btn_handler))
    app.run_polling()
