import asyncio
import logging
import time
from datetime import datetime
import numpy as np
from deriv_api import DerivAPI
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

# ========================= CONFIG =========================
DEMO_TOKEN = "tIrfitLjqeBxCOM"
REAL_TOKEN = "2hsJzopRHG5wUEb"
APP_ID = 1089

MARKETS = ["R_10", "R_25", "R_50", "R_75", "R_100"]
COOLDOWN_SEC = 300 
MAX_TRADES_PER_DAY = 20
MAX_CONSEC_LOSSES = 5

TELEGRAM_TOKEN = "8276370676:AAGh5VqkG7b4cvpfRIVwY_rtaBlIiNwCTDM"
TELEGRAM_CHAT_ID = "7634818949"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ========================= STRATEGY MATH =========================
def calculate_indicators(candles):
    c = np.array([x["c"] for x in candles], dtype=float)
    h = np.array([x["h"] for x in candles], dtype=float)
    l = np.array([x["l"] for x in candles], dtype=float)
    o = np.array([x["o"] for x in candles], dtype=float)

    period = 20
    sma = np.convolve(c, np.ones(period), "valid") / period
    std_dev = np.array([np.std(c[i : i + period]) for i in range(len(sma))], dtype=float)
    upper_band = sma + (std_dev * 2)
    lower_band = sma - (std_dev * 2)

    adx_p = 14
    tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    plus_dm = np.where((h[1:] - h[:-1]) > (l[:-1] - l[1:]), np.maximum(h[1:] - h[:-1], 0), 0)
    minus_dm = np.where((l[:-1] - l[1:]) > (h[1:] - h[:-1]), np.maximum(l[:-1] - l[1:], 0), 0)

    def smooth(data, p):
        if len(data) < p: return np.array([], dtype=float)
        res = [np.mean(data[:p])]
        for x in data[p:]: res.append((res[-1] * (p - 1) + x) / p)
        return np.array(res, dtype=float)

    trs = smooth(tr, adx_p)
    pdms = smooth(plus_dm, adx_p)
    mdms = smooth(minus_dm, adx_p)
    eps = 1e-9
    pdi = 100 * (pdms / (trs + eps))
    mdi = 100 * (mdms / (trs + eps))
    dx = 100 * (np.abs(pdi - mdi) / (pdi + mdi + eps))
    adx = smooth(dx, adx_p)

    stoch_p = 14
    if len(l) < stoch_p: return None
    l_min = np.array([np.min(l[i : i + stoch_p]) for i in range(len(l) - stoch_p + 1)], dtype=float)
    h_max = np.array([np.max(h[i : i + stoch_p]) for i in range(len(h) - stoch_p + 1)], dtype=float)
    pk = 100 * ((c[stoch_p - 1 :] - l_min) / (h_max - l_min + 1e-6))
    pd = np.convolve(pk, np.ones(3), "valid") / 3

    return upper_band[-1], lower_band[-1], adx[-1], pk[-1], pd[-1], o[-1], c[-1], h[-1], l[-1]

def build_candles(times, prices):
    candles = []
    curr_t0 = times[0] - (times[0] % 60)
    o = h = l = c = float(prices[0])
    for t, p in zip(times, prices):
        t0 = t - (t % 60)
        p = float(p)
        if t0 != curr_t0:
            candles.append({"o": o, "h": h, "l": l, "c": c})
            curr_t0, o, h, l, c = t0, p, p, p, p
        else: h, l, c = max(h, p), min(l, p), p
    candles.append({"o": o, "h": h, "l": l, "c": c})
    return candles

# ========================= BOT CORE =========================
class DerivSniperBot:
    def __init__(self):
        self.api = None
        self.app = None
        self.active_token = None
        self.account_type = "None"
        self.is_scanning = False
        self.market_tasks = {}
        self.active_trade_info = None
        self.active_market = "None"
        self.trade_start_time = 0.0
        self.cooldown_until = 0.0
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
        if not self.api: return
        try:
            bal = await self.api.balance({"balance": 1})
            self.balance = f"{float(bal['balance']['balance']):.2f} {bal['balance']['currency']}"
        except: pass

    def can_auto_trade(self) -> tuple[bool, str]:
        if self.consecutive_losses >= MAX_CONSEC_LOSSES: return False, "Max Loss Reached"
        if self.trades_today >= MAX_TRADES_PER_DAY: return False, "Daily Limit Met"
        if time.time() < self.cooldown_until: return False, f"Cooldown {int(self.cooldown_until - time.time())}s"
        if self.active_trade_info: return False, "Trade in Progress"
        if not self.api: return False, "No Connection"
        return True, "OK"

    async def background_scanner(self):
        self.market_tasks = {sym: asyncio.create_task(self.scan_market(sym)) for sym in MARKETS}
        try:
            while self.is_scanning:
                if self.active_trade_info and (time.time() - self.trade_start_time > 240):
                    self.active_trade_info = None
                await asyncio.sleep(1)
        finally:
            for t in self.market_tasks.values(): t.cancel()
            self.market_tasks.clear()

    async def scan_market(self, symbol: str):
        while self.is_scanning:
            try:
                data = await self.api.ticks_history({"ticks_history": symbol, "end": "latest", "count": 1000, "style": "ticks"})
                candles = build_candles(data["history"]["times"], data["history"]["prices"])
                if len(candles) < 40: 
                    await asyncio.sleep(10); continue
                ind = calculate_indicators(candles)
                if not ind: continue
                up, lw, adx_v, pk, pd, op, cl, hi, lo = ind
                ok, gate = self.can_auto_trade()
                if not ok: 
                    await asyncio.sleep(5); continue
                if adx_v < 30:
                    if (lo <= lw) and (pk < 25) and (pk > pd):
                        await self.execute_trade("CALL", symbol, "Oversold BB", source="AUTO")
                    elif (hi >= up) and (pk > 75) and (pk < pd):
                        await self.execute_trade("PUT", symbol, "Overbought BB", source="AUTO")
            except asyncio.CancelledError: break
            except: pass
            await asyncio.sleep(5)

    async def execute_trade(self, side: str, symbol: str, reason="MANUAL", source="MANUAL"):
        if not self.api or self.active_trade_info: return
        async with self.trade_lock:
            try:
                prop = await self.api.proposal({"proposal": 1, "amount": 1.00, "basis": "stake", "contract_type": side, "currency": "USD", "duration": 180, "duration_unit": "s", "symbol": symbol})
                buy = await self.api.buy({"buy": prop["proposal"]["id"], "price": float(prop["proposal"]["ask_price"])})
                self.active_trade_info = int(buy["buy"]["contract_id"])
                self.active_market, self.trade_start_time = symbol, time.time()
                if source == "AUTO": self.trades_today += 1
                await self.app.bot.send_message(TELEGRAM_CHAT_ID, f"🚀 *{side} OPENED*\n🛒 `{symbol}`\n📝 {reason}", parse_mode="Markdown")
                asyncio.create_task(self.check_result(self.active_trade_info, source))
            except: pass

    async def check_result(self, cid: int, source: str):
        await asyncio.sleep(185)
        try:
            res = await self.api.proposal_open_contract({"proposal_open_contract": 1, "contract_id": cid})
            profit = float(res["proposal_open_contract"].get("profit", 0))
            if source == "AUTO":
                if profit <= 0: self.consecutive_losses += 1; self.total_losses_today += 1
                else: self.consecutive_losses = 0
            await self.fetch_balance()
            await self.app.bot.send_message(TELEGRAM_CHAT_ID, f"🏁 *FINISH*: {'✅ WIN' if profit > 0 else '❌ LOSS'} (${profit:.2f})", parse_mode="Markdown")
        finally:
            self.active_trade_info = None
            self.cooldown_until = time.time() + COOLDOWN_SEC

# ========================= UI =========================
bot_logic = DerivSniperBot()

def main_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("▶️ START", callback_data="START_SCAN"), InlineKeyboardButton("⏹️ STOP", callback_data="STOP_SCAN")],
        [InlineKeyboardButton("📊 STATUS", callback_data="STATUS"), InlineKeyboardButton("🔄 REFRESH", callback_data="STATUS")],
        [InlineKeyboardButton("🧪 TEST BUY", callback_data="TEST_BUY")],
        [InlineKeyboardButton("🧪 DEMO", callback_data="SET_DEMO"), InlineKeyboardButton("💰 LIVE", callback_data="SET_REAL")]
    ])

async def btn_handler(u: Update, c: ContextTypes.DEFAULT_TYPE):
    q = u.callback_query
    await q.answer()
    
    if q.data == "SET_DEMO":
        bot_logic.active_token = DEMO_TOKEN; bot_logic.account_type = "DEMO"
        ok = await bot_logic.connect()
        await q.edit_message_text("✅ DEMO CONNECTED" if ok else "❌ FAILED", reply_markup=main_keyboard())
    
    elif q.data == "SET_REAL":
        bot_logic.active_token = REAL_TOKEN; bot_logic.account_type = "LIVE"
        ok = await bot_logic.connect()
        await q.edit_message_text("⚠️ LIVE CONNECTED" if ok else "❌ FAILED", reply_markup=main_keyboard())
    
    elif q.data == "START_SCAN":
        if not bot_logic.api: 
            await q.edit_message_text("❌ Connect First!", reply_markup=main_keyboard())
            return
        bot_logic.is_scanning = True
        asyncio.create_task(bot_logic.background_scanner())
        await q.edit_message_text("🔍 *SCANNER ACTIVE*", reply_markup=main_keyboard(), parse_mode="Markdown")
    
    elif q.data == "STOP_SCAN":
        bot_logic.is_scanning = False
        await q.edit_message_text("⏹️ STOPPED", reply_markup=main_keyboard())
    
    elif q.data == "TEST_BUY":
        await bot_logic.execute_trade("CALL", "R_10", "Manual Test")
    
    elif q.data == "STATUS":
        await bot_logic.fetch_balance()
        now_time = datetime.now().strftime("%H:%M:%S")
        ok, gate = bot_logic.can_auto_trade()
        
        trade_status = "No active trade"
        if bot_logic.active_trade_info and bot_logic.api:
            try:
                res = await bot_logic.api.proposal_open_contract({"proposal_open_contract": 1, "contract_id": bot_logic.active_trade_info})
                pnl = float(res['proposal_open_contract'].get('profit', 0))
                elapsed = int(time.time() - bot_logic.trade_start_time)
                rem = max(0, 180 - elapsed)
                icon = "🟢" if pnl >= 0 else "🔴"
                trade_status = f"Active: {bot_logic.active_market} | {icon} PnL: ${pnl:.2f} | {rem}s"
            except:
                trade_status = "Syncing trade..."

        status_text = (
            f"🕒 **Time:** `{now_time}`\n"
            f"🤖 **Bot:** `{'ACTIVE' if bot_logic.is_scanning else 'OFFLINE'}` (`{bot_logic.account_type}`)\n"
            f"📡 **Scanning:** `{', '.join(MARKETS)}`\n"
            f"━━━━━━━━━━━━━━━\n"
            f"📌 **Trade:** {trade_status}\n"
            f"💰 **Balance:** `{bot_logic.balance}`\n"
            f"━━━━━━━━━━━━━━━\n"
            f"🎯 **Today:** `{bot_logic.trades_today}/20` | 📉 **Loss Streak:** `{bot_logic.consecutive_losses}/5` \n"
            f"🚦 **Gate:** `{gate}`"
        )
        await q.edit_message_text(status_text, reply_markup=main_keyboard(), parse_mode="Markdown")

async def start_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    await u.message.reply_text("💎 **Sniper M1 v4.9 Stable**", reply_markup=main_keyboard())

if __name__ == "__main__":
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    bot_logic.app = app
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CallbackQueryHandler(btn_handler))
    app.run_polling(drop_pending_updates=True)
