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

MARKET = "R_10"
EMA_PERIOD = 100
COOLDOWN_SEC = 600  # 10 Minutes mandatory cooldown

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

    # EMA 100
    ema = [c[0]]
    k = 2 / (EMA_PERIOD + 1)
    for price in c[1:]:
        ema.append(price * k + ema[-1] * (1 - k))
    ema100 = np.array(ema)

    # MACD (12, 26, 9)
    def get_ema(data, p):
        res = [data[0]]; alpha = 2/(p+1)
        for v in data[1:]: res.append(v*alpha + res[-1]*(1-alpha))
        return np.array(res)
    macd_line = get_ema(c, 12) - get_ema(c, 26)
    sig_line = get_ema(macd_line, 9)
    hist = macd_line - sig_line

    # PSAR (0.02, 0.2)
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
        self.account_type = "None"
        self.is_scanning = False
        self.scanner_status = "💤 Offline"
        
        # Live Tracking
        self.active_trade_info = None 
        self.trade_start_time = 0
        self.cooldown_until = 0
        self.last_reason = "Waiting for data..."
        
        # Strategy Stages
        self.buy_stage = 0
        self.sell_stage = 0
        
        # Risk Management
        self.trades_today = 0
        self.consecutive_losses = 0
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
            # --- SMART STATUS UPDATE ---
            if self.active_trade_info:
                self.scanner_status = "🚀 In Trade"
            elif time.time() < self.cooldown_until:
                self.scanner_status = "⏱️ In Cooldown"
            else:
                self.scanner_status = "📡 Searching for Signal..."

            if self.consecutive_losses >= 5:
                self.is_scanning = False
                self.scanner_status = "🛑 STOPPED (5 LOSSES)"
                await self.app.bot.send_message(TELEGRAM_CHAT_ID, "🛑 **Safety Stop**: 5 consecutive losses reached. Bot deactivated.")
                break
            
            if self.trades_today >= 20:
                self.is_scanning = False
                self.scanner_status = "🛑 DAILY LIMIT (20) REACHED"
                break

            try:
                data = await self.api.ticks_history({"ticks_history": MARKET, "end": "latest", "count": 1500, "style": "ticks"})
                ticks = list(zip(data['history']['times'], data['history']['prices']))
                
                candles = []
                curr_t0 = ticks[0][0] - (ticks[0][0] % 30)
                o = h = l = c = ticks[0][1]
                for t, p in ticks:
                    t0 = t - (t % 30)
                    if t0 != curr_t0:
                        candles.append({'o':o, 'h':h, 'l':l, 'c':c})
                        curr_t0, o, h, l, c = t0, p, p, p, p
                    else:
                        h, l, c = max(h, p), min(l, p), p
                
                if len(candles) < 110: continue

                ema100, psar, hist, op, hi, lo, cl = calculate_indicators(candles)
                
                slope = ema100[-1] - ema100[-6]
                slope_threshold = 0.00005 
                avg_range = np.mean(hi[-20:] - lo[-20:])
                body = abs(cl[-1] - op[-1])
                avg_body = np.mean(np.abs(cl[-6:-1] - op[-6:-1]))

                ps_above = psar[-1] > hi[-1]
                ps_below = psar[-1] < lo[-1]
                prev_ps_above = psar[-2] > hi[-2]
                prev_ps_below = psar[-2] < lo[-2]

                # --- SELL Logic ---
                if slope < -slope_threshold and cl[-1] < ema100[-1]:
                    if (not prev_ps_above) and ps_above:
                        self.sell_stage = 1; self.buy_stage = 0; self.last_reason = "SELL: Flip 1 detected"
                    elif self.sell_stage == 1 and ps_above:
                        self.sell_stage = 2; self.last_reason = "SELL: Stage 2 confirmed"
                    
                    if self.sell_stage == 2 and time.time() >= self.cooldown_until:
                        if abs(psar[-1] - cl[-1]) >= 0.6 * avg_range and hist[-1] < hist[-2] and cl[-1] < op[-1] and body >= avg_body:
                            await self.execute_trade("PUT", "AUTO"); self.sell_stage = 0
                        else: self.last_reason = "SELL: Waiting for Momentum/Body"
                else: 
                    if cl[-1] > ema100[-1] or slope > 0: self.sell_stage = 0

                # --- BUY Logic ---
                if slope > slope_threshold and cl[-1] > ema100[-1]:
                    if (not prev_ps_below) and ps_below:
                        self.buy_stage = 1; self.sell_stage = 0; self.last_reason = "BUY: Flip 1 detected"
                    elif self.buy_stage == 1 and ps_below:
                        self.buy_stage = 2; self.last_reason = "BUY: Stage 2 confirmed"
                    
                    if self.buy_stage == 2 and time.time() >= self.cooldown_until:
                        if abs(psar[-1] - cl[-1]) >= 0.6 * avg_range and hist[-1] > hist[-2] and cl[-1] > op[-1] and body >= avg_body:
                            await self.execute_trade("CALL", "AUTO"); self.buy_stage = 0
                        else: self.last_reason = "BUY: Waiting for Momentum/Body"
                else:
                    if cl[-1] < ema100[-1] or slope < 0: self.buy_stage = 0

            except Exception as e:
                logger.error(f"Scanner Error: {e}")
            
            await asyncio.sleep(15)

    async def execute_trade(self, side: str, source="MANUAL"):
        if not self.api or self.active_trade_info: return
        async with self.trade_lock:
            try:
                proposal = await self.api.proposal({
                    "proposal": 1, "amount": 1.00, "basis": "stake",
                    "contract_type": side, "currency": "USD",
                    "duration": 150, "duration_unit": "s", "symbol": MARKET
                })
                buy = await self.api.buy({"buy": proposal["proposal"]["id"], "price": float(proposal["proposal"]["ask_price"]) + 0.02})
                
                self.active_trade_info = buy["buy"]["contract_id"]
                self.trade_start_time = time.time()
                if source == "AUTO": self.trades_today += 1
                
                await self.app.bot.send_message(TELEGRAM_CHAT_ID, f"🚀 **{side} TRADE EXECUTED ({source})**\nStage 2 PSAR Confirmation\nMarket: {MARKET}")
                asyncio.create_task(self.check_result(self.active_trade_info, source))
            except Exception as e:
                logger.error(f"Execution Error: {e}")

    async def check_result(self, cid, source):
        await asyncio.sleep(160)
        try:
            res = await self.api.proposal_open_contract({"proposal_open_contract": 1, "contract_id": cid})
            profit = float(res['proposal_open_contract'].get('profit', 0))
            if source == "AUTO":
                self.pnl_today += profit
                if profit <= 0: self.consecutive_losses += 1
                else: self.consecutive_losses = 0
            
            await self.fetch_balance()
            await self.app.bot.send_message(TELEGRAM_CHAT_ID, f"🏁 **TRADE FINISHED**\nResult: {'✅ WIN' if profit > 0 else '❌ LOSS'} (${profit:.2f})\nStreak: {self.consecutive_losses}/5 losses.")
        finally:
            self.active_trade_info = None
            self.cooldown_until = time.time() + COOLDOWN_SEC
            # Notification for cooldown ending
            asyncio.create_task(self.notify_cooldown_end())

    async def notify_cooldown_end(self):
        await asyncio.sleep(COOLDOWN_SEC)
        if self.is_scanning:
            await self.app.bot.send_message(TELEGRAM_CHAT_ID, "📡 **Cooldown Ended**: Bot is now searching for the next signal.")

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
        
        # Priority-based status display
        if not bot_logic.is_scanning:
            current_state = "💤 Offline"
        elif bot_logic.active_trade_info:
            current_state = "🚀 In Trade"
        elif time.time() < bot_logic.cooldown_until:
            current_state = "⏱️ In Cooldown"
        else:
            current_state = "📡 Searching for Signal"

        status_header = f"🤖 **Bot State**: `{current_state}`\n🔑 **Account**: `{bot_logic.account_type}`\n"
        
        trade_text = ""
        if bot_logic.active_trade_info:
            try:
                res = await bot_logic.api.proposal_open_contract({"proposal_open_contract": 1, "contract_id": bot_logic.active_trade_info})
                current_pnl = float(res['proposal_open_contract'].get('profit', 0))
                pnl_icon = "🟢" if current_pnl >= 0 else "🔴"
                elapsed = int(time.time() - bot_logic.trade_start_time)
                remaining = max(0, 150 - elapsed)
                trade_text = f"\n⏳ **Live Trade**: {pnl_icon} `${current_pnl:.2f}`\n⏱️ **Remaining**: `{remaining}s`\n"
            except: trade_text = "\n⏳ **Trade Status**: `Updating...`\n"
        
        cd = max(0, int(bot_logic.cooldown_until - time.time()))
        summary = f"\n💰 **Balance**: `{bot_logic.balance}`\n🎯 **Today**: `{bot_logic.trades_today}/20` | **Streak**: `{bot_logic.consecutive_losses}/5`"
        details = f"\n📈 **Stages**: `B:{bot_logic.buy_stage} S:{bot_logic.sell_stage}`\n⏱️ **Cooldown**: `{cd}s`\n💬 **Last**: `{bot_logic.last_reason}`"
        
        await q.edit_message_text(f"📊 **DETAILED STATUS**\n{status_header}{trade_text}{details}{summary}", reply_markup=main_keyboard(), parse_mode="Markdown")

    elif q.data == "START_SCAN":
        if not bot_logic.api: await q.edit_message_text("❌ Connect Account First!", reply_markup=main_keyboard()); return
        bot_logic.is_scanning = True; asyncio.create_task(bot_logic.background_scanner())
        await q.edit_message_text("🔍 **SCANNER ACTIVE**\nStrategy: Strict Second SAR Confirmation", reply_markup=main_keyboard(), parse_mode="Markdown")

    elif q.data == "SET_DEMO":
        bot_logic.active_token = DEMO_TOKEN; await bot_logic.connect(); bot_logic.account_type = "DEMO"
        await q.edit_message_text(f"✅ Connected to DEMO\nBal: {bot_logic.balance}", reply_markup=main_keyboard())
            
    elif q.data == "SET_REAL":
        bot_logic.active_token = REAL_TOKEN; await bot_logic.connect(); bot_logic.account_type = "LIVE 💰"
        await q.edit_message_text(f"⚠️ **CONNECTED TO LIVE**\nBal: {bot_logic.balance}", reply_markup=main_keyboard(), parse_mode="Markdown")
            
    elif q.data == "TEST_BUY": await bot_logic.execute_trade("CALL", "MANUAL-TEST")
    elif q.data == "STOP_SCAN": bot_logic.is_scanning = False; bot_logic.scanner_status = "💤 Offline"

async def start_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    await u.message.reply_text("💎 **Sniper v6.3 (Smart Status Edition)**", reply_markup=main_keyboard())

if __name__ == "__main__":
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    bot_logic.app = app
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CallbackQueryHandler(btn_handler))
    app.run_polling(drop_pending_updates=True)
