import asyncio
import datetime
import logging
import numpy as np
from deriv_api import DerivAPI
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

# ========================= 1. CONFIG (STRICT v3) =========================
DERIV_TOKEN = "2NFJTH3JgXWFCcv" 
APP_ID = "1089"        # Default testing ID
SYMBOL = "R_10"        # Volatility 10 (1s)
STAKE = 0.35           # Min Stake
DURATION = 3           # 3 Minutes
TELEGRAM_TOKEN = "8276370676:AAGh5VqkG7b4cvpfRIVwY_rtaBlIiNwCTDM"
TELEGRAM_CHAT_ID = "7634818949"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# ========================= 2. STRATEGY ENGINE =========================
def get_ema(values, period=50):
    if len(values) < period: return None
    values = np.array(values, dtype=float)
    k = 2 / (period + 1)
    ema = values[0]
    for p in values[1:]:
        ema = p * k + ema * (1 - k)
    return ema

def get_rsi(prices, period=14):
    if len(prices) < period + 1: return 50
    deltas = np.diff(prices)
    up = deltas[deltas >= 0].sum() / period
    down = -deltas[deltas < 0].sum() / period
    if down == 0: return 100
    rs = up / down
    return 100.0 - (100.0 / (1.0 + rs))

def check_v3_momentum(candle, prev_3, direction, ema):
    """Rule 7: Momentum Candle Gate"""
    o, h, l, c = candle['open'], candle['high'], candle['low'], candle['close']
    rng = h - l
    if rng <= 0: return False
    
    # 7.1: Touch EMA
    if not (l <= ema <= h): return False
    
    # 7.3: Body vs Avg Body of last 3
    avg_body = sum([abs(x['close'] - x['open']) for x in prev_3]) / 3
    curr_body = abs(c - o)
    if curr_body <= avg_body: return False
    
    # 7.4/7.6: Close Position & Trend Confirmation
    if direction == "CALL":
        is_top_30 = (h - c) / rng <= 0.30  # Closes in top 30%
        breaks_prior = c > prev_3[-1]['high']
        return is_top_30 and breaks_prior and c > o
    else: 
        is_bottom_30 = (c - l) / rng <= 0.30 # Closes in bottom 30%
        breaks_prior = c < prev_3[-1]['low']
        return is_bottom_30 and breaks_prior and c < o

# ========================= 3. BOT CORE =========================
class DerivSniperBot:
    def __init__(self):
        self.api = None
        self.running = False
        self.current_status = "🛑 Stopped"
        self.active_trade_msg = "None"
        
        # Stats Tracking
        self.trades_today = 0
        self.wins_today = 0
        self.losses_today = 0
        self.pnl_today = 0.0
        
        self.app = None
        self.last_trade_time = None
        self.last_ema_level = 0

    async def send(self, msg):
        if self.app:
            try: await self.app.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg)
            except: pass

    async def connect(self):
        try:
            self.api = DerivAPI(app_id=APP_ID)
            await self.api.authorize(DERIV_TOKEN)
            return True
        except: return False

    async def get_candles(self, gran, count):
        data = await self.api.get_candles({"ticks_history": SYMBOL, "granularity": gran, "count": count})
        return data['candles']

    async def run_scanner(self):
        self.current_status = "🔎 Searching for Signal..."
        while self.running:
            try:
                # RULE 1: Daily Limits (2 trades max)
                if self.trades_today >= 2:
                    self.current_status = "🛑 Daily Limit Reached (2/2)"
                    self.running = False
                    break

                # RULE 9: Cooldown (45 mins after a loss)
                if self.last_trade_time and self.losses_today > 0:
                    elapsed = (datetime.datetime.now() - self.last_trade_time).seconds
                    if elapsed < 2700:
                        self.current_status = f"⏳ Cooling down ({45 - (elapsed//60)}m left)"
                        await asyncio.sleep(30)
                        continue

                # RULE 3: M5 Bias Check
                m5 = await self.get_candles(300, 60)
                m5_closes = [x['close'] for x in m5]
                m5_ema = get_ema(m5_closes, 50)
                if not m5_ema: continue
                m5_bias = "CALL" if m5_closes[-1] > m5_ema else "PUT"

                # RULE 4 & 6: M1 Alignment & RSI
                m1 = await self.get_candles(60, 60)
                m1_closes = [x['close'] for x in m1]
                m1_ema = get_ema(m1_closes, 50)
                rsi = get_rsi(m1_closes, 14)

                # RSI Exhaustion Filter (Rule 6: 40-60 zone only)
                if not (40 <= rsi <= 60):
                    self.current_status = f"🔎 RSI Outside Zone ({round(rsi)})"
                    await asyncio.sleep(10)
                    continue

                self.current_status = "🔎 Perfect Conditions... Waiting for Pullback"

                # RULE 5: First Pullback detection
                # Check if price was significantly away from EMA recently
                was_away = abs(m1_closes[-6] - m1_ema) > (m1_ema * 0.0004)
                
                last_c = m1[-1]
                prev_3 = m1[-4:-1]

                signal = None
                if m5_bias == "CALL" and m1_closes[-1] > m1_ema and was_away:
                    if check_v3_momentum(last_c, prev_3, "CALL", m1_ema):
                        signal = "CALL"
                elif m5_bias == "PUT" and m1_closes[-1] < m1_ema and was_away:
                    if check_v3_momentum(last_c, prev_3, "PUT", m1_ema):
                        signal = "PUT"

                if signal:
                    # RULE 8: No re-entry at same EMA level
                    if abs(m1_ema - self.last_ema_level) > (m1_ema * 0.0002):
                        await self.execute_trade(signal)
                        self.last_ema_level = m1_ema
                        # Cooldown while trade is running
                        await asyncio.sleep(DURATION * 60 + 5) 

                await asyncio.sleep(5) 
            except Exception as e:
                logging.error(f"Scanner Loop Error: {e}")
                await asyncio.sleep(10)

    async def execute_trade(self, side):
        try:
            self.active_trade_msg = f"⏳ Buying {side}..."
            req = {
                "buy": 1, "price": STAKE,
                "parameters": {
                    "amount": STAKE, "basis": "stake", "contract_type": side,
                    "currency": "USD", "duration": DURATION, "duration_unit": "m", "symbol": SYMBOL
                }
            }
            resp = await self.api.buy(req)
            contract_id = resp['buy']['contract_id']
            
            self.trades_today += 1
            self.last_trade_time = datetime.datetime.now()
            self.active_trade_msg = f"📡 {side} Active (ID: {contract_id})"
            
            # Start a background task to check result and update PnL
            asyncio.create_task(self.check_result(contract_id))
            
        except Exception as e:
            self.active_trade_msg = "None"
            await self.send(f"❌ Execution failed: {e}")

    async def check_result(self, contract_id):
        """Waits for trade to end and updates stats"""
        await asyncio.sleep(DURATION * 60 + 2)
        try:
            proposal = await self.api.get_profit_table({"limit": 1, "description": 1})
            last_trade = proposal['profit_table']['transactions'][0]
            
            payout = float(last_trade['sell_price'])
            cost = float(last_trade['buy_price'])
            profit = payout - cost
            
            self.pnl_today += profit
            if profit > 0:
                self.wins_today += 1
            else:
                self.losses_today += 1
                
            self.active_trade_msg = "None"
            await self.send(f"✅ Trade Closed. Result: {'Win' if profit > 0 else 'Loss'} ({round(profit, 2)})")
        except:
            self.active_trade_msg = "None"

# ========================= 4. TELEGRAM UI =========================
bot = DerivSniperBot()

async def start_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    kb = [
        [InlineKeyboardButton("▶️ RUN SNIPER", callback_data="RUN"),
         InlineKeyboardButton("🛑 STOP", callback_data="STOP")],
        [InlineKeyboardButton("📊 VIEW STATUS", callback_data="STATUS")]
    ]
    await u.message.reply_text(
        f"💎 **Deriv Sniper v3**\n`Balance Target: $5` \n\n"
        f"Strict rules: 2 trades/day max.\nM5 Bias + M1 Momentum logic.",
        reply_markup=InlineKeyboardMarkup(kb), parse_mode="Markdown"
    )

async def btn_handler(u: Update, c: ContextTypes.DEFAULT_TYPE):
    q = u.callback_query
    await q.answer()
    
    if q.data == "RUN":
        if not bot.running:
            bot.running = True
            if await bot.connect():
                asyncio.create_task(bot.run_scanner())
                await q.edit_message_text("✅ Sniper Online! Market scan in progress...")
            else:
                await q.edit_message_text("❌ Connection Failed. Check Token.")
    
    elif q.data == "STOP":
        bot.running = False
        bot.current_status = "🛑 Stopped"
        await q.edit_message_text("🛑 Bot Stopped Manually.")

    elif q.data == "STATUS":
        pnl_str = f"+${round(bot.pnl_today, 2)}" if bot.pnl_today >= 0 else f"-${round(abs(bot.pnl_today), 2)}"
        stat_msg = (
            f"📊 **LIVE STATUS REPORT**\n"
            f"----------------------------\n"
            f"🤖 State: `{bot.current_status}`\n"
            f"🎯 Active: `{bot.active_trade_msg}`\n\n"
            f"📈 **Today's Stats:**\n"
            f"Bullets Used: {bot.trades_today}/2\n"
            f"Wins: {bot.wins_today} | Losses: {bot.losses_today}\n"
            f"Net PnL: `{pnl_str}`\n"
            f"----------------------------"
        )
        kb = [[InlineKeyboardButton("🔄 Refresh", callback_data="STATUS")],
              [InlineKeyboardButton("⬅️ Back", callback_data="BACK")]]
        await q.edit_message_text(stat_msg, reply_markup=InlineKeyboardMarkup(kb), parse_mode="Markdown")

    elif q.data == "BACK":
        bot_running_label = "Online" if bot.running else "Offline"
        kb = [[InlineKeyboardButton("▶️ RUN", callback_data="RUN"), InlineKeyboardButton("🛑 STOP", callback_data="STOP")],
              [InlineKeyboardButton("📊 VIEW STATUS", callback_data="STATUS")]]
        await q.edit_message_text(f"Bot is {bot_running_label}. Choose an action:", reply_markup=InlineKeyboardMarkup(kb))

def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    bot.app = app
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CallbackQueryHandler(btn_handler))
    print("Deriv Sniper v3 is LIVE...")
    app.run_polling()

if __name__ == "__main__":
    main()
