import asyncio
import logging
import time
from datetime import datetime
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
MAX_TRADES_PER_DAY = 100          # ✅ CHANGED (was 40)
MAX_CONSEC_LOSSES = 10            # ✅ stays 10 (stop after 10 losses in a row)

BASE_STAKE = 1.00  # Initial trade amount

TELEGRAM_TOKEN = "8253450930:AAHUhPk9TML-8kZlA9UaHZZvTUGdurN9MVY"
TELEGRAM_CHAT_ID = "7634818949"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ========================= STRATEGY MATH =========================
def calculate_ema(data, period):
    if len(data) < period:
        return np.array([])
    values = np.array(data, dtype=float)
    ema = np.zeros_like(values)
    k = 2 / (period + 1)
    ema[0] = values[0]
    for i in range(1, len(values)):
        ema[i] = values[i] * k + ema[i - 1] * (1 - k)
    return ema

def calculate_rsi(data, period=14):
    data = np.array(data, dtype=float)
    if len(data) < period + 2:
        return np.array([])
    delta = np.diff(data)
    gain = (delta.clip(min=0))
    loss = (-delta.clip(max=0))
    avg_gain = np.convolve(gain, np.ones(period), 'valid') / period
    avg_loss = np.convolve(loss, np.ones(period), 'valid') / period
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))

def calculate_indicators(candles):
    c = np.array([x["c"] for x in candles], dtype=float)
    o = np.array([x["o"] for x in candles], dtype=float)

    ema9 = calculate_ema(c, 9)
    ema21 = calculate_ema(c, 21)
    if len(ema21) < 2 or len(ema9) < 1:
        return None

    rsi_vals = calculate_rsi(c, 14)
    if len(rsi_vals) < 2:
        return None

    ema21_slope = ema21[-1] - ema21[-2]

    return {
        "ema9": float(ema9[-1]),
        "ema21": float(ema21[-1]),
        "ema21_slope": float(ema21_slope),
        "rsi": float(rsi_vals[-1]),
        "rsi_prev": float(rsi_vals[-2]),
        "o": float(o[-1]),
        "c": float(c[-1]),
    }

def build_m1_candles_from_ticks(times, prices):
    if not times or not prices:
        return []
    candles = []
    curr_t0 = int(times[0] - (times[0] % 60))
    o = h = l = c = float(prices[0])

    for t, p in zip(times, prices):
        t = int(t)
        t0 = int(t - (t % 60))
        p = float(p)
        if t0 != curr_t0:
            candles.append({"t0": curr_t0, "o": o, "h": h, "l": l, "c": c})
            curr_t0, o, h, l, c = t0, p, p, p, p
        else:
            h, l, c = max(h, p), min(l, p), p

    candles.append({"t0": curr_t0, "o": o, "h": h, "l": l, "c": c})
    return candles

# ========================= BOT CORE =========================
class DerivSniperBot:
    def __init__(self):
        self.api = None
        self.app = None
        self.active_token = None
        self.account_type = "None"

        self.is_scanning = False
        self.scanner_task = None
        self.market_tasks = {}

        self.active_trade_info = None
        self.active_market = "None"
        self.trade_start_time = 0.0

        self.cooldown_until = 0.0

        self.trades_today = 0
        self.total_losses_today = 0
        self.consecutive_losses = 0
        self.total_profit_today = 0.0
        self.balance = "0.00"

        self.current_stake = BASE_STAKE
        self.trade_lock = asyncio.Lock()

        # ✅ detailed live scan tracking (per market)
        self.market_debug = {m: {} for m in MARKETS}
        self.last_closed_candle_t0 = {m: 0 for m in MARKETS}  # anti-overtrade per closed candle

        # existing
        self.last_scan_symbol = "None"
        self.last_signal_reason = "None"
        self.last_block_reason = "None"
        self.last_trade_side = "None"
        self.last_trade_source = "None"

    async def connect(self) -> bool:
        try:
            if not self.active_token:
                return False
            self.api = DerivAPI(app_id=APP_ID)
            await self.api.authorize(self.active_token)
            await self.fetch_balance()
            return True
        except Exception as e:
            logger.error(f"Connect error: {e}")
            return False

    async def fetch_balance(self):
        if not self.api:
            return
        try:
            bal = await self.api.balance({"balance": 1})
            self.balance = f"{float(bal['balance']['balance']):.2f} {bal['balance']['currency']}"
        except:
            pass

    def can_auto_trade(self) -> tuple[bool, str]:
        if self.consecutive_losses >= MAX_CONSEC_LOSSES:
            return False, "Stopped: 10 losses in a row"
        if self.trades_today >= MAX_TRADES_PER_DAY:
            return False, "Stopped: daily trade limit reached"
        if time.time() < self.cooldown_until:
            return False, f"Cooldown ({int(self.cooldown_until - time.time())}s left)"
        if self.active_trade_info:
            return False, "Trade in progress"
        if not self.api:
            return False, "Not connected"
        return True, "OK"

    async def background_scanner(self):
        if not self.api:
            return
        self.market_tasks = {sym: asyncio.create_task(self.scan_market(sym)) for sym in MARKETS}
        try:
            while self.is_scanning:
                # safety clear if stuck (duration is 5m, give extra buffer)
                if self.active_trade_info and (time.time() - self.trade_start_time > 330):
                    self.active_trade_info = None
                await asyncio.sleep(1)
        finally:
            for t in self.market_tasks.values():
                t.cancel()
            self.market_tasks.clear()

    async def scan_market(self, symbol: str):
        while self.is_scanning:
            self.last_scan_symbol = symbol
            try:
                # hard stop conditions
                if self.consecutive_losses >= MAX_CONSEC_LOSSES or self.trades_today >= MAX_TRADES_PER_DAY:
                    self.is_scanning = False
                    break

                # pull ticks -> build M1 candles
                data = await self.api.ticks_history({
                    "ticks_history": symbol,
                    "end": "latest",
                    "count": 1000,
                    "style": "ticks"
                })
                candles = build_m1_candles_from_ticks(data["history"]["times"], data["history"]["prices"])

                # need enough candles
                if len(candles) < 30:
                    self.market_debug[symbol] = {
                        "time": time.time(),
                        "status": "WAIT",
                        "reason": "Not enough candle data yet",
                    }
                    await asyncio.sleep(5)
                    continue

                # ✅ use only CLOSED candle for decisions (second last)
                closed = candles[-2]
                closed_t0 = int(closed.get("t0", 0))

                # ✅ anti-overtrade: only evaluate once per closed candle
                if closed_t0 and self.last_closed_candle_t0[symbol] == closed_t0:
                    self.market_debug[symbol] = {
                        "time": time.time(),
                        "status": "WAIT",
                        "reason": "Waiting for next M1 candle to close",
                        "closed_t0": closed_t0,
                    }
                    await asyncio.sleep(2)
                    continue

                self.last_closed_candle_t0[symbol] = closed_t0

                ind = calculate_indicators(candles[:-1])  # exclude the forming candle for safer indicator calc
                if not ind:
                    self.market_debug[symbol] = {
                        "time": time.time(),
                        "status": "WAIT",
                        "reason": "Indicators not ready yet",
                        "closed_t0": closed_t0,
                    }
                    await asyncio.sleep(3)
                    continue

                ok, gate = self.can_auto_trade()
                self.last_block_reason = gate

                # build readable conditions
                ema9 = ind["ema9"]
                ema21 = ind["ema21"]
                slope = ind["ema21_slope"]
                rsi = ind["rsi"]
                rsi_prev = ind["rsi_prev"]
                close = ind["c"]

                trend_up = (ema9 > ema21) and (slope > 0)
                trend_down = (ema9 < ema21) and (slope < 0)

                conds = {
                    "trend_up": trend_up,
                    "trend_down": trend_down,
                    "price_above_ema21": close > ema21,
                    "price_below_ema21": close < ema21,
                    "rsi_call_band": (45 <= rsi <= 60) and (rsi > rsi_prev),
                    "rsi_put_band": (40 <= rsi <= 55) and (rsi < rsi_prev),
                    "price_above_ema9": close > ema9,
                    "price_below_ema9": close < ema9,
                    "rsi_safety_band": (35 <= rsi <= 65),
                }

                # update live scan debug ALWAYS (even if gate blocks)
                bias = "UP" if trend_up else "DOWN" if trend_down else "RANGE"
                self.market_debug[symbol] = {
                    "time": time.time(),
                    "status": "READY" if ok else "BLOCKED",
                    "reason": gate if not ok else "Checking setup conditions",
                    "bias": bias,
                    "closed_t0": closed_t0,
                    "ema9": ema9,
                    "ema21": ema21,
                    "ema21_slope": slope,
                    "rsi": rsi,
                    "rsi_prev": rsi_prev,
                    "close": close,
                    "conds": conds,
                }

                if not ok:
                    await asyncio.sleep(3)
                    continue

                # ========================= ENTRY LOGIC (UNCHANGED) =========================
                if conds["trend_up"]:
                    if conds["price_above_ema21"]:
                        if conds["rsi_call_band"]:
                            if conds["price_above_ema9"]:
                                if conds["rsi_safety_band"]:
                                    await self.execute_trade("CALL", symbol, "EMA Cross + RSI Rising", source="AUTO")

                elif conds["trend_down"]:
                    if conds["price_below_ema21"]:
                        if conds["rsi_put_band"]:
                            if conds["price_below_ema9"]:
                                if conds["rsi_safety_band"]:
                                    await self.execute_trade("PUT", symbol, "EMA Cross + RSI Falling", source="AUTO")

                else:
                    # range/no trade
                    self.market_debug[symbol]["status"] = "WAIT"
                    self.market_debug[symbol]["reason"] = "Trend not clear (EMA9/EMA21 + slope not aligned)"

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scanner Error ({symbol}): {e}")
                self.market_debug[symbol] = {
                    "time": time.time(),
                    "status": "ERROR",
                    "reason": str(e)[:140],
                }
            await asyncio.sleep(2)

    async def execute_trade(self, side: str, symbol: str, reason="MANUAL", source="MANUAL"):
        if not self.api or self.active_trade_info:
            return

        async with self.trade_lock:
            try:
                stake = float(self.current_stake if source == "AUTO" else BASE_STAKE)

                prop = await self.api.proposal({
                    "proposal": 1,
                    "amount": stake,
                    "basis": "stake",
                    "contract_type": side,
                    "currency": "USD",
                    "duration": 5,
                    "duration_unit": "m",
                    "symbol": symbol
                })
                buy = await self.api.buy({
                    "buy": prop["proposal"]["id"],
                    "price": float(prop["proposal"]["ask_price"])
                })

                self.active_trade_info = int(buy["buy"]["contract_id"])
                self.active_market = symbol
                self.trade_start_time = time.time()

                self.last_signal_reason = reason
                self.last_trade_side = side
                self.last_trade_source = source

                if source == "AUTO":
                    self.trades_today += 1

                safe_symbol = str(symbol).replace("_", " ")
                msg = (
                    f"🚀 {side} TRADE OPENED (${stake:.2f})\n"
                    f"🛒 Market: {safe_symbol}\n"
                    f"🧠 Reason: {reason}\n"
                    f"🤖 Source: {source}"
                )
                await self.app.bot.send_message(TELEGRAM_CHAT_ID, msg)

                asyncio.create_task(self.check_result(self.active_trade_info, source))
            except Exception as e:
                logger.error(f"Trade error: {e}")

    async def check_result(self, cid: int, source: str):
        await asyncio.sleep(305)  # 5m + buffer
        try:
            res = await self.api.proposal_open_contract({"proposal_open_contract": 1, "contract_id": cid})
            profit = float(res["proposal_open_contract"].get("profit", 0))

            if source == "AUTO":
                self.total_profit_today += profit
                if profit <= 0:
                    self.consecutive_losses += 1
                    self.total_losses_today += 1
                    self.current_stake *= 2  # martingale
                else:
                    self.consecutive_losses = 0
                    self.current_stake = BASE_STAKE  # reset

            await self.fetch_balance()

            await self.app.bot.send_message(
                TELEGRAM_CHAT_ID,
                f"🏁 FINISH: {'WIN' if profit > 0 else 'LOSS'} (${profit:.2f})\n"
                f"📊 Today: {self.trades_today}/{MAX_TRADES_PER_DAY} | ❌ Streak: {self.consecutive_losses}/{MAX_CONSEC_LOSSES}\n"
                f"💵 Total Profit: ${self.total_profit_today:.2f}\n"
                f"💰 Balance: {self.balance}"
            )
        finally:
            self.active_trade_info = None
            self.cooldown_until = time.time() + COOLDOWN_SEC

# ========================= UI =========================
bot_logic = DerivSniperBot()

def main_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("▶️ START", callback_data="START_SCAN"),
         InlineKeyboardButton("⏹️ STOP", callback_data="STOP_SCAN")],
        [InlineKeyboardButton("📊 STATUS", callback_data="STATUS"),
         InlineKeyboardButton("🔄 REFRESH", callback_data="STATUS")],
        [InlineKeyboardButton("🧪 TEST BUY", callback_data="TEST_BUY")],
        [InlineKeyboardButton("🧪 DEMO", callback_data="SET_DEMO"),
         InlineKeyboardButton("💰 LIVE", callback_data="SET_REAL")]
    ])

def _yn(b: bool) -> str:
    return "✅ YES" if b else "❌ NO"

def format_market_debug(sym: str, d: dict) -> str:
    if not d:
        return f"📌 {sym.replace('_',' ')}\n⏳ No scan data yet."

    age = int(time.time() - d.get("time", time.time()))
    status = d.get("status", "WAIT")
    reason = d.get("reason", "—")
    bias = d.get("bias", "—")

    # Basic header
    lines = []
    lines.append(f"📌 {sym.replace('_',' ')}  ({age}s ago)")
    lines.append(f"Status: {status} | Trend: {bias}")
    lines.append(f"Why: {reason}")

    # Details
    if "ema9" in d and "ema21" in d:
        lines.append(f"EMA9: {d['ema9']:.2f} | EMA21: {d['ema21']:.2f} | EMA21 slope: {d['ema21_slope']:.4f}")
    if "rsi" in d:
        lines.append(f"RSI: {d['rsi']:.1f} (prev {d.get('rsi_prev', 0):.1f}) | Close: {d.get('close', 0):.2f}")

    conds = d.get("conds", {})
    if conds:
        lines.append("Checks:")
        if bias == "UP":
            lines.append(f" - EMA9 > EMA21 & slope>0: {_yn(conds.get('trend_up', False))}")
            lines.append(f" - Close > EMA21: {_yn(conds.get('price_above_ema21', False))}")
            lines.append(f" - RSI 45-60 AND rising: {_yn(conds.get('rsi_call_band', False))}")
            lines.append(f" - Close > EMA9: {_yn(conds.get('price_above_ema9', False))}")
            lines.append(f" - RSI safety 35-65: {_yn(conds.get('rsi_safety_band', False))}")
        elif bias == "DOWN":
            lines.append(f" - EMA9 < EMA21 & slope<0: {_yn(conds.get('trend_down', False))}")
            lines.append(f" - Close < EMA21: {_yn(conds.get('price_below_ema21', False))}")
            lines.append(f" - RSI 40-55 AND falling: {_yn(conds.get('rsi_put_band', False))}")
            lines.append(f" - Close < EMA9: {_yn(conds.get('price_below_ema9', False))}")
            lines.append(f" - RSI safety 35-65: {_yn(conds.get('rsi_safety_band', False))}")
        else:
            lines.append(" - Trend alignment is not clear (range).")

    # Candle time
    if d.get("closed_t0"):
        t0 = datetime.fromtimestamp(d["closed_t0"], tz=ZoneInfo("Africa/Lagos")).strftime("%H:%M:%S")
        lines.append(f"Last closed M1 candle: {t0}")

    return "\n".join(lines)

async def btn_handler(u: Update, c: ContextTypes.DEFAULT_TYPE):
    q = u.callback_query
    await q.answer()

    if q.data == "SET_DEMO":
        bot_logic.active_token, bot_logic.account_type = DEMO_TOKEN, "DEMO"
        ok = await bot_logic.connect()
        await q.edit_message_text("✅ Connected to DEMO" if ok else "❌ DEMO Failed", reply_markup=main_keyboard())

    elif q.data == "SET_REAL":
        bot_logic.active_token, bot_logic.account_type = REAL_TOKEN, "LIVE"
        ok = await bot_logic.connect()
        await q.edit_message_text("⚠️ LIVE CONNECTED" if ok else "❌ LIVE Failed", reply_markup=main_keyboard())

    elif q.data == "START_SCAN":
        if not bot_logic.api:
            await q.edit_message_text("❌ Connect first.", reply_markup=main_keyboard())
            return
        bot_logic.is_scanning = True
        bot_logic.scanner_task = asyncio.create_task(bot_logic.background_scanner())
        await q.edit_message_text("🔍 SCANNER ACTIVE\n✅ Live scan will update every new M1 candle close.", reply_markup=main_keyboard())

    elif q.data == "STOP_SCAN":
        bot_logic.is_scanning = False
        await q.edit_message_text("⏹️ Scanner stopped.", reply_markup=main_keyboard())

    elif q.data == "TEST_BUY":
        await bot_logic.execute_trade("CALL", "R_10", "Manual Test", source="MANUAL")

    elif q.data == "STATUS":
        await bot_logic.fetch_balance()

        now_time = datetime.now(ZoneInfo("Africa/Lagos")).strftime("%Y-%m-%d %H:%M:%S")
        ok, gate = bot_logic.can_auto_trade()

        # cooldown info
        cd_left = max(0, int(bot_logic.cooldown_until - time.time()))
        cd_txt = f"{cd_left}s" if cd_left > 0 else "0s"

        # active trade status
        trade_status = "No Active Trade"
        if bot_logic.active_trade_info and bot_logic.api:
            try:
                res = await bot_logic.api.proposal_open_contract({
                    "proposal_open_contract": 1,
                    "contract_id": bot_logic.active_trade_info
                })
                pnl = float(res["proposal_open_contract"].get("profit", 0))
                rem = max(0, 300 - int(time.time() - bot_logic.trade_start_time))
                icon = "🟢 PROFIT" if pnl > 0 else "🔴 LOSS"
                mkt_clean = str(bot_logic.active_market).replace("_", " ")
                trade_status = f"🚀 Active Trade ({mkt_clean})\n📈 PnL: {icon} (${pnl:.2f})\n⏳ Left: {rem}s"
            except:
                trade_status = "🚀 Active Trade: Syncing..."

        # detailed live scan
        live_scan_blocks = []
        for sym in MARKETS:
            live_scan_blocks.append(format_market_debug(sym, bot_logic.market_debug.get(sym, {})))

        status_msg = (
            f"🕒 Time (WAT): {now_time}\n"
            f"🤖 Bot: {'ACTIVE' if bot_logic.is_scanning else 'OFFLINE'} ({bot_logic.account_type})\n"
            f"📡 Markets: {', '.join(MARKETS).replace('_', ' ')}\n"
            f"🚦 Gate: {gate}\n"
            f"⏸ Cooldown: {cd_txt}\n"
            f"━━━━━━━━━━━━━━━\n{trade_status}\n━━━━━━━━━━━━━━━\n"
            f"💵 Total Profit: ${bot_logic.total_profit_today:.2f}\n"
            f"🎯 Today: {bot_logic.trades_today}/{MAX_TRADES_PER_DAY} | ❌ Losses: {bot_logic.total_losses_today}\n"
            f"📉 Loss Streak: {bot_logic.consecutive_losses}/{MAX_CONSEC_LOSSES} | 🧪 Next Stake: ${bot_logic.current_stake:.2f}\n"
            f"💰 Balance: {bot_logic.balance}\n"
            f"\n📌 LIVE SCAN (Detailed)\n\n" + "\n\n".join(live_scan_blocks)
        )

        await q.edit_message_text(status_msg, reply_markup=main_keyboard())

async def start_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    await u.message.reply_text("💎 Sniper Survival M1 (EMA 9/21 Edition)", reply_markup=main_keyboard())

if __name__ == "__main__":
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    bot_logic.app = app
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CallbackQueryHandler(btn_handler))
    app.run_polling(drop_pending_updates=True)
