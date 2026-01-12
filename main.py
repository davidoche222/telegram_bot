import asyncio
import logging
import time
import numpy as np
from deriv_api import DerivAPI
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

# ========================= CONFIG =========================
DEMO_TOKEN = "tIrfitLjqeBxCOM"
REAL_TOKEN = "2hsJzopRHG5wUEb"
APP_ID = 1089

MARKETS = ["R_10", "R_25", "R_50", "R_75", "R_100"] 
EMA_PERIOD = 100
COOLDOWN_SEC = 300 # Set to 5 minutes to catch the next range cycle

TELEGRAM_TOKEN = "8276370676:AAGh5VqkG7b4cvpfRIVwY_rtaBlIiNwCTDM"
TELEGRAM_CHAT_ID = "7634818949"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ========================= STRATEGY MATH (TRIPLE-LOCK) =========================
def calculate_indicators(candles):
    c = np.array([x['c'] for x in candles])
    h = np.array([x['h'] for x in candles])
    l = np.array([x['l'] for x in candles])
    o = np.array([x['o'] for x in candles])

    # 1. Bollinger Bands (20, 2)
    period = 20
    sma = np.convolve(c, np.ones(period), 'valid') / period
    std_dev = np.array([np.std(c[i : i + period]) for i in range(len(sma))])
    upper_band = sma + (std_dev * 2)
    lower_band = sma - (std_dev * 2)

    # 2. ADX (14) - Trend Filter
    adx_p = 14
    tr = np.maximum(h[1:]-l[1:], np.maximum(abs(h[1:]-c[:-1]), abs(l[1:]-c[:-1])))
    plus_dm = np.where((h[1:]-h[:-1]) > (l[:-1]-l[1:]), np.maximum(h[1:]-h[:-1], 0), 0)
    minus_dm = np.where((l[:-1]-l[1:]) > (h[1:]-h[:-1]), np.maximum(l[:-1]-l[1:], 0), 0)
    
    def smooth(data, p):
        res = [np.mean(data[:p])]
        for x in data[p:]: res.append((res[-1]*(p-1)+x)/p)
        return np.array(res)

    trs = smooth(tr, adx_p)
    pdms, mdms = smooth(plus_dm, adx_p), smooth(minus_dm, adx_p)
    pdi, mdi = 100*pdms/trs, 100*mdms/trs
    dx = 100*abs(pdi-mdi)/(pdi+mdi)
    adx = smooth(dx, adx_p)

    # 3. Stochastic (14, 3)
    stoch_p = 14
    l_min = np.array([np.min(l[i:i+stoch_p]) for i in range(len(l)-stoch_p+1)])
    h_max = np.array([np.max(h[i:i+stoch_p]) for i in range(len(h)-stoch_p+1)])
    pk = 100*(c[stoch_p-1:]-l_min)/(h_max-l_min+0.000001)
    pd = np.convolve(pk, np.ones(3), 'valid') / 3

    return upper_band[-1], lower_band[-1], adx[-1], pk[-1], pd[-1], o[-1], c[-1], h[-1], l[-1]

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
        self.trade_start_time = 0
        self.cooldown_until = 0
        
        self.trades_today = 0
        self.total_losses_today = 0
        self.consecutive_losses = 0
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

    async def scan_market(self, symbol):
        while self.is_scanning:
            if self.consecutive_losses >= 5 or self.trades_today >= 20:
                self.is_scanning = False
                break
            try:
                # Building 1-Minute Candles from Ticks
                data = await self.api.ticks_history({"ticks_history": symbol, "end": "latest", "count": 1500, "style": "ticks"})
                ticks = list(zip(data['history']['times'], data['history']['prices']))
                candles = []
                curr_t0 = ticks[0][0] - (ticks[0][0] % 60)
                o = h = l = c = ticks[0][1]
                for t, p in ticks:
                    t0 = t - (t % 60)
                    if t0 != curr_t0:
                        candles.append({'o':o, 'h':h, 'l':l, 'c':c})
                        curr_t0, o, h, l, c = t0, p, p, p, p
                    else: h, l, c = max(h, p), min(l, p), p
                
                if len(candles) < 40: continue

                up, lw, adx_val, pk, pd, op, cl, hi, lo = calculate_indicators(candles)

                # TRIGGER LOGIC: Ranging market (ADX < 25) + Bollinger Touch + Stoch Cross
                if adx_val < 25 and time.time() >= self.cooldown_until:
                    # CALL Signal
                    if lo <= lw and pk < 20 and pk > pd and cl > op:
                        await self.execute_trade("CALL", symbol, "AUTO")
                    
                    # PUT Signal
                    elif hi >= up and pk > 80 and pk < pd and cl < op:
                        await self.execute_trade("PUT", symbol, "AUTO")

            except Exception as e: logger.error(f"Error {symbol}: {e}")
            await asyncio.sleep(10)

    async def background_scanner(self):
        tasks = [asyncio.create_task(self.scan_market(m)) for m in MARKETS]
        await asyncio.gather(*tasks)

    async def execute_trade(self, side: str, symbol: str, source="MANUAL"):
        if not self.api or self.active_trade_info: return
        async with self.trade_lock:
            try:
                # 3-Minute Expiration (180s)
                proposal = await self.api.proposal({"proposal": 1, "amount": 1.00, "basis": "stake", "contract_type": side, "currency": "USD", "duration": 180, "duration_unit": "s", "symbol": symbol})
                buy = await self.api.buy({"buy": proposal["proposal"]["id"], "price": float(proposal["proposal"]["ask_price"]) + 0.02})
                self.active_trade_info = buy["buy"]["contract_id"]
                self.active_market = symbol
                self.trade_start_time = time.time()
                if source == "AUTO": self.trades_today += 1
                await self.app.bot.send_message(TELEGRAM_CHAT_ID, f"🚀 **{side} TRADE OPENED**\nMarket: `{symbol}`")
                asyncio.create_task(self.check_result(self.active_trade_info, source))
            except Exception as e: logger.error(f"Trade Error: {e}")

    async def check_result(self, cid, source):
        # Wait for 3-minute expiration + buffer
        await asyncio.sleep(185)
        try:
            res = await self.api.proposal_open_contract({"proposal_open_contract": 1, "contract_id": cid})
            profit = float(res['proposal_open_contract'].get('profit', 0))
            if source == "AUTO":
                if profit <= 0: 
                    self.consecutive_losses += 1
                    self.total_losses_today += 1
                else: self.consecutive_losses = 0
            await self.fetch_balance()
            await self.app.bot.send_message(TELEGRAM_CHAT_ID, f"🏁 **FINISH**: {'✅ WIN' if profit > 0 else '❌ LOSS'} (${profit:.2f})")
        finally:
            self.active_trade_info = None
            self.active_market = "None"
            self.cooldown_until = time.time() + COOLDOWN_SEC

# ========================= UI =========================
bot_logic = DerivSniperBot()

def main_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("▶️ START", callback_data="START_SCAN"), InlineKeyboardButton("⏹️ STOP", callback_data="STOP_SCAN")],
        [InlineKeyboardButton("🧪 TEST BUY", callback_data="TEST_BUY"), InlineKeyboardButton("📊 STATUS", callback_data="STATUS")],
        [InlineKeyboardButton("🧪 DEMO", callback_data="SET_DEMO"), InlineKeyboardButton("💰 LIVE", callback_data="SET_REAL")]
    ])

async def btn_handler(u: Update, c: ContextTypes.DEFAULT_TYPE):
    q = u.callback_query; await q.answer()
    if q.data == "STATUS":
        await bot_logic.fetch_balance()
        status_msg = (
            f"🤖 **Bot**: `{'ACTIVE' if bot_logic.is_scanning else 'OFFLINE'}`\n"
            f"📡 **Scanning**: `R10, R25, R50, R75, R100`\n"
            f"━━━━━━━━━━━━━━━\n"
            f"🎯 **Trades Today**: `{bot_logic.trades_today}/20`\n"
            f"❌ **Total Lost**: `{bot_logic.total_losses_today}`\n"
            f"📉 **Streak**: `{bot_logic.consecutive_losses}/5` losses\n"
            f"💰 **Balance**: `{bot_logic.balance}`"
        )
        await q.edit_message_text(status_msg, reply_markup=main_keyboard(), parse_mode="Markdown")
    elif q.data == "START_SCAN":
        bot_logic.is_scanning = True; asyncio.create_task(bot_logic.background_scanner())
        await q.edit_message_text("🔍 **RANGE SCANNER ACTIVE**", reply_markup=main_keyboard())
    elif q.data == "TEST_BUY":
        await bot_logic.execute_trade("CALL", "R_10", "MANUAL")
    elif q.data == "SET_DEMO":
        bot_logic.active_token = DEMO_TOKEN; await bot_logic.connect()
        await q.edit_message_text(f"✅ Connected to DEMO", reply_markup=main_keyboard())
    elif q.data == "SET_REAL":
        bot_logic.active_token = REAL_TOKEN; await bot_logic.connect()
        await q.edit_message_text(f"⚠️ **LIVE CONNECTED**", reply_markup=main_keyboard())
    elif q.data == "STOP_SCAN": bot_logic.is_scanning = False

async def start_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    await u.message.reply_text("💎 **Sniper Range v4.0**", reply_markup=main_keyboard())

if __name__ == "__main__":
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    bot_logic.app = app
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CallbackQueryHandler(btn_handler))
    app.run_polling(drop_pending_updates=True)
