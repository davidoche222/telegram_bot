import asyncio
import logging
import time
import numpy as np
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from deriv_api import DerivAPI
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

# ========================= CONFIG =========================
DEMO_TOKEN = "tIrfitLjqeBxCOM"
REAL_TOKEN = "2hsJzopRHG5wUEb"
APP_ID = 1089

MARKETS = ["R_10", "R_25", "R_50", "R_75", "R_100"] 
BASE_PAYOUT = 1.00  
MARTINGALE_MULTIPLIER = 2.1
EMA_PERIOD = 100
COOLDOWN_SEC = 300  

TELEGRAM_TOKEN = "8276370676:AAGh5VqkG7b4cvpfRIVwY_rtaBlIiNwCTDM"
TELEGRAM_CHAT_ID = "7634818949"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ========================= CORE BOT CLASS =========================
class DerivSniperBot:
    def __init__(self):
        self.api = None
        self.app = None
        self.active_token = None
        self.is_scanning = False
        self.scanner_status = "💤 Offline"
        self.active_trade_info = None 
        self.cooldown_until = 0
        self.current_payout = Decimal(str(BASE_PAYOUT))
        self.stages = {m: {"buy": 0, "sell": 0} for m in MARKETS}
        self.balance = "0.00"
        self.trade_lock = asyncio.Lock()
        
        self.weekly_wins = 0
        self.weekly_losses = 0
        self.weekly_profit = Decimal('0.00')

    async def connect(self):
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

    async def execute_trade(self, symbol, side):
        """Martingale on Payout to ensure full recovery + profit."""
        if not self.api or self.active_trade_info: return

        async with self.trade_lock:
            try:
                # 1. Precise Payout calculation
                target_payout = float(self.current_payout.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))
                
                # 2. Proposal Request
                prop = await self.api.proposal({
                    "proposal": 1,
                    "amount": target_payout,
                    "basis": "payout",
                    "contract_type": side,
                    "currency": "USD",
                    "duration": 150,
                    "duration_unit": "s",
                    "symbol": symbol
                })
                
                # 3. Stake Floor Check (0.35 USD)
                required_stake = float(prop['proposal']['ask_price'])
                if required_stake < 0.35:
                    # Force payout higher to meet minimum stake requirement
                    prop = await self.api.proposal({
                        "proposal": 1,
                        "amount": 0.35,
                        "basis": "stake",
                        "contract_type": side,
                        "currency": "USD",
                        "duration": 150,
                        "duration_unit": "s",
                        "symbol": symbol
                    })

                # 4. Final Purchase
                buy = await self.api.buy({"buy": prop["proposal"]["id"], "price": float(prop["proposal"]["ask_price"])})
                self.active_trade_info = buy["buy"]["contract_id"]
                
                msg = f"🚀 **{side} TRADE**\nMarket: `{symbol}`\nTarget Payout: `${target_payout:.2f}`\nStake: `${float(prop['proposal']['ask_price']):.2f}`"
                await self.app.bot.send_message(TELEGRAM_CHAT_ID, msg, parse_mode="Markdown")
                asyncio.create_task(self.check_result(self.active_trade_info))
                
            except Exception as e:
                logger.error(f"Trade Execution Failed: {e}")

    async def check_result(self, cid):
        await asyncio.sleep(160)
        try:
            res = await self.api.proposal_open_contract({"proposal_open_contract": 1, "contract_id": cid})
            profit = Decimal(str(res['proposal_open_contract'].get('profit', '0')))
            self.weekly_profit += profit
            
            if profit <= 0:
                self.weekly_losses += 1
                # Multiply payout for recovery + profit on next win
                self.current_payout = (self.current_payout * Decimal(str(MARTINGALE_MULTIPLIER))).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
            else:
                self.weekly_wins += 1
                # Reset to base payout
                self.current_payout = Decimal(str(BASE_PAYOUT))
            
            await self.fetch_balance()
            status = '✅ WIN' if profit > 0 else '❌ LOSS'
            msg = f"🏁 **{status}** (${float(profit):.2f})\nNext Payout Goal: `${float(self.current_payout):.2f}`"
            await self.app.bot.send_message(TELEGRAM_CHAT_ID, msg, parse_mode="Markdown")
        finally:
            self.active_trade_info = None
            self.cooldown_until = time.time() + COOLDOWN_SEC

# ========================= INDICATORS =========================
def calculate_indicators(candles):
    c = np.array([x['c'] for x in candles]); h = np.array([x['h'] for x in candles])
    l = np.array([x['l'] for x in candles]); o = np.array([x['o'] for x in candles])
    ema = [c[0]]; k = 2 / (EMA_PERIOD + 1)
    for price in c[1:]: ema.append(price * k + ema[-1] * (1 - k))
    ema100 = np.array(ema)
    def get_ema(data, p):
        res = [data[0]]; alpha = 2/(p+1)
        for v in data[1:]: res.append(v*alpha + res[-1]*(1-alpha))
        return np.array(res)
    macd_line = get_ema(c, 12) - get_ema(c, 26)
    sig_line = get_ema(macd_line, 9); hist = macd_line - sig_line
    psar = np.zeros(len(c)); up = True; af = 0.02; ep = h[0]; psar[0] = l[0]
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

# ========================= SCANNER & UI =========================
    async def background_scanner(self):
        while self.is_scanning:
            if not self.api: await asyncio.sleep(2); continue
            for symbol in MARKETS:
                if not self.is_scanning: break
                if self.active_trade_info or time.time() < self.cooldown_until: await asyncio.sleep(2); break 
                try:
                    self.scanner_status = f"📡 Scanning {symbol}..."
                    data = await self.api.ticks_history({"ticks_history": symbol, "end": "latest", "count": 1000, "style": "ticks"})
                    ticks = list(zip(data['history']['times'], data['history']['prices']))
                    candles = []; curr_t0 = ticks[0][0] - (ticks[0][0] % 30); o = h = l = c = ticks[0][1]
                    for t, p in ticks:
                        t0 = t - (t % 30)
                        if t0 != curr_t0: candles.append({'o':o, 'h':h, 'l':l, 'c':c}); curr_t0, o, h, l, c = t0, p, p, p, p
                        else: h, l, c = max(h, p), min(l, p), p
                    if len(candles) < 110: continue
                    ema100, psar, hist, op, hi, lo, cl = calculate_indicators(candles)
                    vol_factor = 10.0 if "100" in symbol else (5.0 if "75" in symbol else 1.0)
                    slope = ema100[-1] - ema100[-5]; slope_threshold = 0.005 * vol_factor; avg_range = np.mean(hi[-10:] - lo[-10:])
                    
                    if cl[-1] < ema100[-1] and slope < -slope_threshold:
                        if psar[-1] > hi[-1] and not (psar[-2] > hi[-2]): self.stages[symbol]["sell"] = 1
                        elif self.stages[symbol]["sell"] == 1 and psar[-1] > hi[-1]:
                            if hist[-1] < 0 and abs(cl[-1]-op[-1]) > (avg_range * 0.1): await self.execute_trade(symbol, "PUT"); self.stages[symbol]["sell"] = 0
                    else: self.stages[symbol]["sell"] = 0
                    if cl[-1] > ema100[-1] and slope > slope_threshold:
                        if psar[-1] < lo[-1] and not (psar[-2] < lo[-2]): self.stages[symbol]["buy"] = 1
                        elif self.stages[symbol]["buy"] == 1 and psar[-1] < lo[-1]:
                            if hist[-1] > 0 and abs(cl[-1]-op[-1]) > (avg_range * 0.1): await self.execute_trade(symbol, "CALL"); self.stages[symbol]["buy"] = 0
                    else: self.stages[symbol]["buy"] = 0
                except: pass
                await asyncio.sleep(1.2)

# ... [Standard Boilerplate for report_scheduler, btn_handler, main_keyboard, and app.run_polling() as in v7.5/7.6] ...
