import asyncio
import logging
import time
import numpy as np
from datetime import datetime
from deriv_api import DerivAPI
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

# ========================= CONFIG =========================
DEMO_TOKEN = "tIrfitLjqeBxCOM"
REAL_TOKEN = "2hsJzopRHG5wUEb"
APP_ID = 1089

MARKETS = ["R_10", "R_25", "R_50", "R_75", "R_100"] 
BASE_STAKE = 0.52
MARTINGALE_MULTIPLIER = 2.1
EMA_PERIOD = 100
COOLDOWN_SEC = 600  

TELEGRAM_TOKEN = "8276370676:AAGh5VqkG7b4cvpfRIVwY_rtaBlIiNwCTDM"
TELEGRAM_CHAT_ID = "7634818949"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ========================= STRATEGY MATH =========================
def calculate_indicators(candles):
    c = np.array([x['c'] for x in candles])
    h = np.array([x['h'] for x in candles])
    l = np.array([x['l'] for x in candles])
    o = np.array([x['o'] for x in candles])

    ema = [c[0]]
    k = 2 / (EMA_PERIOD + 1)
    for price in c[1:]: ema.append(price * k + ema[-1] * (1 - k))
    ema100 = np.array(ema)

    def get_ema(data, p):
        res = [data[0]]; alpha = 2/(p+1)
        for v in data[1:]: res.append(v*alpha + res[-1]*(1-alpha))
        return np.array(res)
    
    macd_line = get_ema(c, 12) - get_ema(c, 26)
    sig_line = get_ema(macd_line, 9)
    hist = macd_line - sig_line

    psar = np.zeros(len(c))
    up = True; af = 0.02; ep = h[0]; psar[0] = l[0]
    for i in range(1, len(c)):
        psar[i] = psar[i-1] + af * (ep - psar[i-1])
        if up:
            psar[i] = min(psar[i], l[max(0, i-1)], l[max(0, i-2)])
            if l[i] < psar[i]: up = False; psar[i] = ep; af = 0.02; ep = l[i]
            elif h[i] > ep: ep = h[i]; af = min(0.2, af + 0.02)
        else:
            psar[i] = max(psar[i], h[max(0, i-1)], h[max(0, i-2)])
            if h[i] > psar[i]: up = True; psar[i] = ep; af = 0.02; ep = h[i]
            elif l[i] < ep: ep = l[i]; af = min(0.2, af + 0.02)

    return ema100, psar, hist, o, h, l, c

# ========================= BOT CORE =========================
class DerivSniperBot:
    def __init__(self):
        self.api = None
        self.app = None
        self.active_token = None
        self.is_scanning = False
        self.scanner_status = "💤 Offline"
        self.active_trade_info = None 
        self.cooldown_until = 0
        self.last_reason = "System Initialized"
        self.current_stake = BASE_STAKE
        self.stages = {m: {"buy": 0, "sell": 0} for m in MARKETS}
        self.trades_today = 0
        self.consecutive_losses = 0
        self.balance = "0.00"
        self.trade_lock = asyncio.Lock()
        
        # Weekly Reporting Stats
        self.weekly_wins = 0
        self.weekly_losses = 0
        self.weekly_profit = 0.0

    async def connect(self):
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

    async def report_scheduler(self):
        """Checks every hour if it's Sunday night to send the report."""
        while True:
            now = datetime.now()
            # If Sunday and 23:00 - 23:59
            if now.weekday() == 6 and now.hour == 23:
                report = (
                    f"📅 **WEEKLY PROFIT REPORT**\n"
                    f"━━━━━━━━━━━━━━━\n"
                    f"✅ Wins: `{self.weekly_wins}`\n"
                    f"❌ Losses: `{self.weekly_losses}`\n"
                    f"💰 Net Profit: `${self.weekly_profit:.2f}`\n"
                    f"━━━━━━━━━━━━━━━\n"
                    f"Resetting stats for the new week..."
                )
                await self.app.bot.send_message(TELEGRAM_CHAT_ID, report, parse_mode="Markdown")
                # Reset weekly counters
                self.weekly_wins = 0
                self.weekly_losses = 0
                self.weekly_profit = 0.0
                await asyncio.sleep(3600) # Sleep 1 hour so it doesn't double-send
            await asyncio.sleep(1800) # Check every 30 mins

    async def background_scanner(self):
        while self.is_scanning:
            for symbol in MARKETS:
                if not self.is_scanning: break
                if self.active_trade_info: await asyncio.sleep(5); continue
                
                if time.time() < self.cooldown_until:
                    self.scanner_status = "⏱️ Cooldown"; await asyncio.sleep(2); break 

                self.scanner_status = f"📡 Scanning {symbol}..."
                try:
                    data = await self.api.ticks_history({"ticks_history": symbol, "end": "latest", "count": 1000, "style": "ticks"})
                    ticks = list(zip(data['history']['times'], data['history']['prices']))
                    
                    candles = []
                    curr_t0 = ticks[0][0] - (ticks[0][0] % 30)
                    o = h = l = c = ticks[0][1]
                    for t, p in ticks:
                        t0 = t - (t % 30)
                        if t0 != curr_t0:
                            candles.append({'o':o, 'h':h, 'l':l, 'c':c})
                            curr_t0, o, h, l, c = t0, p, p, p, p
                        else: h, l, c = max(h, p), min(l, p), p
                    
                    if len(candles) < 110: continue
                    ema100, psar, hist, op, hi, lo, cl = calculate_indicators(candles)
                    
                    vol_factor = 10.0 if "100" in symbol else (5.0 if "75" in symbol else 1.0)
                    slope = ema100[-1] - ema100[-5]
                    slope_threshold = 0.005 * vol_factor 
                    avg_range = np.mean(hi[-10:] - lo[-10:])

                    ps_above = psar[-1] > hi[-1]
                    ps_below = psar[-1] < lo[-1]
                    prev_ps_above = psar[-2] > hi[-2]
                    prev_ps_below = psar[-2] < lo[-2]

                    if cl[-1] < ema100[-1] and slope < -slope_threshold:
                        if (not prev_ps_above) and ps_above: self.stages[symbol]["sell"] = 1
                        elif self.stages[symbol]["sell"] == 1 and ps_above:
                            if hist[-1] < 0 and abs(cl[-1]-op[-1]) > (avg_range * 0.1):
                                await self.execute_trade(symbol, "PUT")
                                self.stages[symbol]["sell"] = 0
                    else: self.stages[symbol]["sell"] = 0

                    if cl[-1] > ema100[-1] and slope > slope_threshold:
                        if (not prev_ps_below) and ps_below: self.stages[symbol]["buy"] = 1
                        elif self.stages[symbol]["buy"] == 1 and ps_below:
                            if hist[-1] > 0 and abs(cl[-1]-op[-1]) > (avg_range * 0.1):
                                await self.execute_trade(symbol, "CALL")
                                self.stages[symbol]["buy"] = 0
                    else: self.stages[symbol]["buy"] = 0

                except Exception as e: logger.error(f"Error on {symbol}: {e}")
                await asyncio.sleep(1.5)

    async def execute_trade(self, symbol, side):
        if self.active_trade_info: return
        async with self.trade_lock:
            try:
                prop = await self.api.proposal({"proposal": 1, "amount": self.current_stake, "basis": "stake", "contract_type": side, "currency": "USD", "duration": 150, "duration_unit": "s", "symbol": symbol})
                buy = await self.api.buy({"buy": prop["proposal"]["id"], "price": float(prop["proposal"]["ask_price"])})
                self.active_trade_info = buy["buy"]["contract_id"]
                self.trades_today += 1
                await self.app.bot.send_message(TELEGRAM_CHAT_ID, f"🚀 **{side} TRADE: {symbol}**\nStake: ${self.current_stake:.2f}")
                asyncio.create_task(self.check_result(self.active_trade_info))
            except Exception as e: logger.error(f"Trade Error: {e}")

    async def check_result(self, cid):
        await asyncio.sleep(160)
        try:
            res = await self.api.proposal_open_contract({"proposal_open_contract": 1, "contract_id": cid})
            profit = float(res['proposal_open_contract'].get('profit', 0))
            
            # Update Weekly stats
            self.weekly_profit += profit
            if profit <= 0:
                self.weekly_losses += 1
                self.consecutive_losses += 1
                self.current_stake = round(self.current_stake * MARTINGALE_MULTIPLIER, 2)
            else:
                self.weekly_wins += 1
                self.consecutive_losses = 0
                self.current_stake = BASE_STAKE
            
            await self.fetch_balance()
            status = '✅ WIN' if profit > 0 else '❌ LOSS'
            await self.app.bot.send_message(TELEGRAM_CHAT_ID, f"🏁 **{status}** (${profit:.2f})\nNext Stake: ${self.current_stake:.2f}")
        finally:
            self.active_trade_info = None
            self.cooldown_until = time.time() + COOLDOWN_SEC

# ========================= UI =========================
bot_logic = DerivSniperBot()

def main_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("▶️ START SCANNER", callback_data="START_SCAN"), InlineKeyboardButton("⏹️ STOP", callback_data="STOP_SCAN")],
        [InlineKeyboardButton("🧪 TEST BUY", callback_data="TEST_BUY"), InlineKeyboardButton("📊 STATUS", callback_data="STATUS")],
        [InlineKeyboardButton("🧪 DEMO", callback_data="SET_DEMO"), InlineKeyboardButton("💰 LIVE", callback_data="SET_REAL")]
    ])

async def btn_handler(u: Update, c: ContextTypes.DEFAULT_TYPE):
    q = u.callback_query; await q.answer()
    if q.data == "STATUS":
        await bot_logic.fetch_balance()
        cd = max(0, int(bot_logic.cooldown_until - time.time()))
        
        market_ov = ""
        for m in MARKETS:
            s_icon = "📍" if (bot_logic.stages[m]["buy"] == 1 or bot_logic.stages[m]["sell"] == 1) else "⚪"
            market_ov += f"`{m}:{s_icon}` "

        txt = (
            f"📊 **DETAILED BOT STATUS**\n"
            f"━━━━━━━━━━━━━━━\n"
            f"🤖 **Scan Status**: `{bot_logic.scanner_status}`\n"
            f"💰 **Balance**: `{bot_logic.balance}`\n"
            f"🎲 **Next Stake**: `${bot_logic.current_stake}`\n"
            f"🔥 **Loss Streak**: `{bot_logic.consecutive_losses}`\n"
            f"━━━━━━━━━━━━━━━\n"
            f"📈 **Market Stages (1=Ready)**:\n{market_ov}\n"
            f"⏱️ **Cooldown**: `{cd}s` remaining\n"
            f"📅 **Week P/L**: `${bot_logic.weekly_profit:.2f}`\n"
            f"🎯 **Total Trades**: `{bot_logic.trades_today}`"
        )
        await q.edit_message_text(txt, reply_markup=main_keyboard(), parse_mode="Markdown")
    elif q.data == "START_SCAN":
        bot_logic.is_scanning = True; asyncio.create_task(bot_logic.background_scanner())
        await q.edit_message_text("🔍 **5-MARKET SCANNER ACTIVE**\nStake: $0.52 + Martingale", reply_markup=main_keyboard())
    elif q.data == "SET_DEMO":
        bot_logic.active_token = DEMO_TOKEN; await bot_logic.connect()
        await q.edit_message_text(f"✅ Connected DEMO", reply_markup=main_keyboard())
    elif q.data == "SET_REAL":
        bot_logic.active_token = REAL_TOKEN; await bot_logic.connect()
        await q.edit_message_text(f"⚠️ **LIVE CONNECTED**", reply_markup=main_keyboard())
    elif q.data == "TEST_BUY":
        await bot_logic.execute_trade("R_10", "CALL")
    elif q.data == "STOP_SCAN": bot_logic.is_scanning = False

async def start_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    # Start the weekly report timer when bot starts
    asyncio.create_task(bot_logic.report_scheduler())
    await u.message.reply_text("💎 **Sniper v7.2 (Penta-Scanner + Weekly Reports)**", reply_markup=main_keyboard())

if __name__ == "__main__":
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    bot_logic.app = app
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CallbackQueryHandler(btn_handler))
    app.run_polling()
