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
MAX_TRADES_PER_DAY = 60          # ✅ requested: 60 trades/day
MAX_CONSEC_LOSSES = 10
BASE_STAKE = 1.00

TELEGRAM_TOKEN = "8253450930:AAHUhPk9TML-8kZlA9UaHZZvTUGdurN9MVY"
TELEGRAM_CHAT_ID = "7634818949"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ========================= STRATEGY SETTINGS =========================
TF_SEC = 60                # M1 candles (REAL candles from Deriv)

# ✅ Stability change: reduce API pressure
CANDLES_COUNT = 200        # was 300 (still enough for EMA50 + RSI)
SCAN_SLEEP_SEC = 5         # was 2 (reduces throttling / disconnects)

EMA_PERIOD = 50

# Parabolic SAR defaults (kept as-is; strategy no longer uses it)
PSAR_AF_STEP = 0.02
PSAR_AF_MAX = 0.2

# RSI
RSI_PERIOD = 14

# ✅ UPDATED (your new idea needs 1-minute expiry)
DURATION_MIN = 1

# ===== CROSS STRATEGY SETTINGS (kept as-is; strategy no longer uses it) =====
PSAR_MIN_DIST_MULT = 0.50

# ========================= INDICATOR MATH =========================
def calculate_ema(values, period):
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
    """
    Wilder's RSI. Returns array same length as values.
    First (period) values will be np.nan.
    """
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


def calculate_psar(candles, af_step=0.02, af_max=0.2):
    """
    Standard Parabolic SAR implementation.
    Returns PSAR list (len == len(candles)).
    """
    n = len(candles)
    if n < 3:
        return []

    highs = [c["h"] for c in candles]
    lows = [c["l"] for c in candles]

    psar = [0.0] * n

    uptrend = highs[1] > highs[0]
    ep = highs[1] if uptrend else lows[1]
    af = af_step
    psar[1] = lows[0] if uptrend else highs[0]

    for i in range(2, n):
        prev_psar = psar[i - 1]
        psar_i = prev_psar + af * (ep - prev_psar)

        if uptrend:
            psar_i = min(psar_i, lows[i - 1], lows[i - 2])
        else:
            psar_i = max(psar_i, highs[i - 1], highs[i - 2])

        if uptrend:
            if lows[i] < psar_i:
                uptrend = False
                psar_i = ep
                ep = lows[i]
                af = af_step
            else:
                if highs[i] > ep:
                    ep = highs[i]
                    af = min(af + af_step, af_max)
        else:
            if highs[i] > psar_i:
                uptrend = True
                psar_i = ep
                ep = highs[i]
                af = af_step
            else:
                if lows[i] < ep:
                    ep = lows[i]
                    af = min(af + af_step, af_max)

        psar[i] = psar_i

    psar[0] = psar[1]
    return psar


def build_candles_from_deriv(candles_raw):
    out = []
    for x in candles_raw:
        out.append({
            "t0": int(x.get("epoch", 0)),
            "o": float(x.get("open", 0)),
            "h": float(x.get("high", 0)),
            "l": float(x.get("low", 0)),
            "c": float(x.get("close", 0)),
        })
    return out


def fmt_time_hhmmss(epoch):
    try:
        return datetime.fromtimestamp(epoch, ZoneInfo("Africa/Lagos")).strftime("%H:%M:%S")
    except:
        return "—"


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

        self.market_debug = {m: {} for m in MARKETS}
        self.last_processed_closed_t0 = {m: 0 for m in MARKETS}

        # Cross-state per symbol (kept as-is; strategy no longer uses it)
        self.last_cross_dir = {m: None for m in MARKETS}
        self.last_cross_t0 = {m: 0 for m in MARKETS}
        self.cross_traded = {m: False for m in MARKETS}

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

    # ========================= SELF-HEALING (ADDED) =========================
    async def safe_reconnect(self) -> bool:
        """Reconnect Deriv API safely after disconnect/throttle/auth issues."""
        try:
            if self.api:
                try:
                    await self.api.disconnect()
                except:
                    pass
        except:
            pass
        self.api = None
        return await self.connect()

    async def safe_ticks_history(self, payload: dict, retries: int = 3):
        """
        Retry wrapper for ticks_history to handle throttling / temporary disconnects.
        Does NOT change strategy logic, only stability.
        """
        last_err = None
        for attempt in range(1, retries + 1):
            try:
                if not self.api:
                    ok = await self.safe_reconnect()
                    if not ok:
                        raise RuntimeError("Reconnect failed")
                return await self.api.ticks_history(payload)
            except Exception as e:
                last_err = e
                msg = str(e).lower()

                # reconnect on common connection/auth failures
                if any(k in msg for k in ["disconnect", "connection", "websocket", "not connected", "authorize", "auth"]):
                    await self.safe_reconnect()

                # small backoff for rate limiting / server hiccups
                await asyncio.sleep(1.25 * attempt)

        raise last_err
    # ========================= END SELF-HEALING =========================

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
            return False, "Stopped: max loss streak reached"
        if self.trades_today >= MAX_TRADES_PER_DAY:
            return False, "Stopped: daily trade limit reached"
        if time.time() < self.cooldown_until:
            return False, f"Cooldown {int(self.cooldown_until - time.time())}s"
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
                if self.active_trade_info and (time.time() - self.trade_start_time > (DURATION_MIN * 60 + 90)):
                    self.active_trade_info = None
                await asyncio.sleep(1)
        finally:
            for t in self.market_tasks.values():
                t.cancel()
            self.market_tasks.clear()

    async def fetch_real_m1_candles(self, symbol: str):
        payload = {
            "ticks_history": symbol,
            "end": "latest",
            "count": CANDLES_COUNT,
            "style": "candles",
            "granularity": TF_SEC
        }
        data = await self.safe_ticks_history(payload, retries=3)
        return build_candles_from_deriv(data.get("candles", []))

    async def _sync_to_next_candle_open(self, last_closed_t0: int):
        """
        last_closed_t0 is the OPEN time of the last closed candle.
        Next candle open = last_closed_t0 + TF_SEC.
        We sleep a tiny bit so we enter near the new candle open.
        """
        next_open = int(last_closed_t0) + int(TF_SEC)
        now = time.time()
        wait = (next_open - now) + 0.15  # small buffer
        if wait > 0:
            await asyncio.sleep(min(wait, 2.0))  # never sleep too long
        else:
            await asyncio.sleep(0.15)

    async def scan_market(self, symbol: str):
        while self.is_scanning:
            try:
                if self.consecutive_losses >= MAX_CONSEC_LOSSES or self.trades_today >= MAX_TRADES_PER_DAY:
                    self.is_scanning = False
                    break

                candles = await self.fetch_real_m1_candles(symbol)
                need = max(60, RSI_PERIOD + 10, 60)  # enough for EMA50 & RSI
                if len(candles) < need:
                    self.market_debug[symbol] = {
                        "time": time.time(),
                        "gate": "Waiting for more candles",
                        "why": [f"Not enough candle history yet (need {need}, have {len(candles)})."],
                    }
                    await asyncio.sleep(SCAN_SLEEP_SEC)
                    continue

                ok_gate, gate = self.can_auto_trade()

                # ========================= STRATEGY (YOUR NEW IDEA) =========================
                # We use TWO closed candles:
                # - pullback_candle = candles[-3] (must touch EMA20)
                # - confirm_candle  = candles[-2] (must close green/red)
                pullback = candles[-3]
                confirm = candles[-2]

                pullback_t0 = int(pullback["t0"])
                confirm_t0 = int(confirm["t0"])

                closes = [x["c"] for x in candles]

                # EMA20 + EMA50
                ema20_arr = calculate_ema(closes, 20)
                ema50_arr = calculate_ema(closes, 50)
                if len(ema20_arr) < 10 or len(ema50_arr) < 10:
                    self.market_debug[symbol] = {"time": time.time(), "gate": "Indicators", "why": ["EMA20/EMA50 not ready yet."]}
                    await asyncio.sleep(SCAN_SLEEP_SEC)
                    continue

                ema20 = float(ema20_arr[-2])  # EMA at confirm candle close
                ema50 = float(ema50_arr[-2])

                # RSI14 on confirm candle close
                rsi_arr = calculate_rsi(closes, RSI_PERIOD)
                if len(rsi_arr) < 10 or np.isnan(rsi_arr[-2]):
                    self.market_debug[symbol] = {"time": time.time(), "gate": "Indicators", "why": ["RSI not ready yet."]}
                    await asyncio.sleep(SCAN_SLEEP_SEC)
                    continue
                rsi_now = float(rsi_arr[-2])

                # ---- Pullback candle touches EMA20 ----
                pb_high = float(pullback["h"])
                pb_low = float(pullback["l"])
                touched_ema20 = (pb_low <= ema20 <= pb_high)

                # ---- Confirmation candle color (forms green/red then we enter next candle) ----
                c_open = float(confirm["o"])
                c_close = float(confirm["c"])
                bull_confirm = c_close > c_open
                bear_confirm = c_close < c_open

                # Spike filter on CONFIRM candle body
                bodies = [abs(float(candles[i]["c"]) - float(candles[i]["o"])) for i in range(-22, -2)]
                if len(bodies) >= 10:
                    avg_body = float(np.mean(bodies))
                else:
                    avg_body = float(np.mean([abs(float(c["c"]) - float(c["o"])) for c in candles[-60:-2]]))
                last_body = abs(c_close - c_open)

                # Per-market tuning (R_50 stricter)
                if symbol == "R_50":
                    spike_mult = 1.2
                    rsi_call_min, rsi_call_max = 53.0, 62.0
                    rsi_put_min, rsi_put_max = 38.0, 47.0
                    ema_diff_min = 0.30
                else:
                    spike_mult = 1.5
                    rsi_call_min, rsi_call_max = 52.0, 60.0
                    rsi_put_min, rsi_put_max = 40.0, 48.0
                    ema_diff_min = 0.20

                spike_block = (avg_body > 0 and last_body > spike_mult * avg_body)

                # Trend strength (separation)
                ema_diff = abs(ema20 - ema50)
                flat_block = ema_diff < ema_diff_min

                # Trend direction
                uptrend = ema20 > ema50
                downtrend = ema20 < ema50

                # RSI zones
                call_rsi_ok = (rsi_call_min <= rsi_now <= rsi_call_max)
                put_rsi_ok = (rsi_put_min <= rsi_now <= rsi_put_max)

                # ✅ Entry is based on:
                # Trend + pullback touches EMA20 (previous candle) + confirm candle color + RSI + filters
                call_ready = (
                    uptrend
                    and touched_ema20
                    and bull_confirm
                    and call_rsi_ok
                    and not spike_block
                    and not flat_block
                )

                put_ready = (
                    downtrend
                    and touched_ema20
                    and bear_confirm
                    and put_rsi_ok
                    and not spike_block
                    and not flat_block
                )

                signal = "CALL" if call_ready else "PUT" if put_ready else None

                # ---------------- SIMPLE STATUS LABELS ----------------
                if uptrend:
                    trend_label = "UPTREND"
                elif downtrend:
                    trend_label = "DOWNTREND"
                else:
                    trend_label = "SIDEWAYS"

                if ema20 > ema50:
                    ema_label = "EMA20 ABOVE EMA50"
                elif ema20 < ema50:
                    ema_label = "EMA20 BELOW EMA50"
                else:
                    ema_label = "EMA20 = EMA50"

                trend_strength = "STRONG" if not flat_block else "WEAK"
                pullback_label = "PULLBACK TOUCHED ✅" if touched_ema20 else "WAITING PULLBACK…"

                block_label = []
                if spike_block:
                    block_label.append("SPIKE BLOCK")
                if flat_block:
                    block_label.append("WEAK/FLAT TREND")
                block_label = " | ".join(block_label) if block_label else "OK"

                why = []
                if not ok_gate:
                    why.append(f"Gate blocked: {gate}")
                if signal:
                    why.append(f"READY: {signal} (enter next candle)")
                else:
                    if trend_label == "SIDEWAYS":
                        why.append("Waiting: trend direction")
                    elif trend_strength != "STRONG":
                        why.append("Waiting: strong trend")
                    elif not touched_ema20:
                        why.append("Waiting: pullback candle touch EMA20")
                    else:
                        why.append("Waiting: confirm candle color/RSI")

                self.market_debug[symbol] = {
                    "time": time.time(),
                    "gate": gate,
                    "last_closed": confirm_t0,
                    "signal": signal,

                    "trend_label": trend_label,
                    "ema_label": ema_label,
                    "trend_strength": trend_strength,
                    "pullback_label": pullback_label,
                    "block_label": block_label,

                    "ema20": ema20,
                    "ema50": ema50,
                    "ema_diff": ema_diff,
                    "rsi_now": rsi_now,
                    "avg_body": avg_body,
                    "last_body": last_body,
                    "spike_block": spike_block,
                    "flat_block": flat_block,

                    "why": why[:10],
                }

                if not ok_gate:
                    await asyncio.sleep(SCAN_SLEEP_SEC)
                    continue

                # One trade per closed CONFIRM candle per symbol
                if self.last_processed_closed_t0[symbol] == confirm_t0:
                    await asyncio.sleep(SCAN_SLEEP_SEC)
                    continue
                self.last_processed_closed_t0[symbol] = confirm_t0

                # ✅ Enter on NEXT candle open (sync)
                if call_ready or put_ready:
                    await self._sync_to_next_candle_open(confirm_t0)

                if call_ready:
                    await self.execute_trade("CALL", symbol, reason="Trend + PullbackTouch(Prev) + ConfirmGreen + RSI + Filters", source="AUTO")
                elif put_ready:
                    await self.execute_trade("PUT", symbol, reason="Trend + PullbackTouch(Prev) + ConfirmRed + RSI + Filters", source="AUTO")
                # ========================= END STRATEGY =========================

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scanner Error ({symbol}): {e}")
                self.market_debug[symbol] = {"time": time.time(), "gate": "Error", "why": [str(e)[:140]]}

            await asyncio.sleep(SCAN_SLEEP_SEC)

    async def execute_trade(self, side: str, symbol: str, reason="MANUAL", source="MANUAL"):
        if not self.api or self.active_trade_info:
            return

        async with self.trade_lock:
            ok, _gate = self.can_auto_trade()
            if not ok:
                return

            try:
                stake = float(self.current_stake if source == "AUTO" else BASE_STAKE)

                prop = await self.api.proposal({
                    "proposal": 1,
                    "amount": stake,
                    "basis": "stake",
                    "contract_type": side,
                    "currency": "USD",
                    "duration": int(DURATION_MIN),
                    "duration_unit": "m",
                    "symbol": symbol
                })
                buy = await self.api.buy({"buy": prop["proposal"]["id"], "price": float(prop["proposal"]["ask_price"])})

                self.active_trade_info = int(buy["buy"]["contract_id"])
                self.active_market = symbol
                self.trade_start_time = time.time()

                if source == "AUTO":
                    self.trades_today += 1

                safe_symbol = str(symbol).replace("_", " ")
                msg = (
                    f"🚀 {side} TRADE OPENED (${stake:.2f})\n"
                    f"🛒 Market: {safe_symbol}\n"
                    f"⏱ Expiry: {DURATION_MIN}m\n"
                    f"🧠 Reason: {reason}\n"
                    f"🤖 Source: {source}"
                )
                await self.app.bot.send_message(TELEGRAM_CHAT_ID, msg)
                asyncio.create_task(self.check_result(self.active_trade_info, source))

            except Exception as e:
                logger.error(f"Trade error: {e}")

    async def check_result(self, cid: int, source: str):
        await asyncio.sleep(int(DURATION_MIN) * 60 + 5)
        try:
            res = await self.api.proposal_open_contract({"proposal_open_contract": 1, "contract_id": cid})
            profit = float(res["proposal_open_contract"].get("profit", 0))

            if source == "AUTO":
                self.total_profit_today += profit
                if profit <= 0:
                    self.consecutive_losses += 1
                    self.total_losses_today += 1
                else:
                    self.consecutive_losses = 0

                # ✅ remove martingale (always fixed stake)
                self.current_stake = BASE_STAKE

            await self.fetch_balance()
            await self.app.bot.send_message(
                TELEGRAM_CHAT_ID,
                (
                    f"🏁 FINISH: {'WIN' if profit > 0 else 'LOSS'} ({profit:+.2f})\n"
                    f"📊 Today: {self.trades_today}/{MAX_TRADES_PER_DAY} | ❌ Losses: {self.total_losses_today} | Streak: {self.consecutive_losses}/{MAX_CONSEC_LOSSES}\n"
                    f"🧪 Next Stake: ${self.current_stake:.2f}\n"
                    f"💰 Balance: {self.balance}"
                )
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

def format_market_detail(sym: str, d: dict) -> str:
    if not d:
        return f"📍 {sym.replace('_',' ')}\n⏳ No scan data yet"

    age = int(time.time() - d.get("time", time.time()))
    gate = d.get("gate", "—")
    last_closed = d.get("last_closed", 0)
    signal = d.get("signal") or "—"

    trend_label = d.get("trend_label", "—")
    ema_label = d.get("ema_label", "—")
    trend_strength = d.get("trend_strength", "—")
    pullback_label = d.get("pullback_label", "—")
    block_label = d.get("block_label", "—")

    return (
        f"📍 {sym.replace('_',' ')} ({age}s)\n"
        f"Gate: {gate}\n"
        f"Last closed: {fmt_time_hhmmss(last_closed)}\n"
        f"────────────────\n"
        f"Trend: {trend_label} ({trend_strength})\n"
        f"{ema_label}\n"
        f"{pullback_label}\n"
        f"Filters: {block_label}\n"
        f"Signal: {signal}\n"
    )

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
        await q.edit_message_text(
            f"🔍 SCANNER ACTIVE\n"
            f"📌 Strategy: Trend (EMA20/EMA50) + Pullback touch EMA20 (prev candle) + Confirm candle green/red + RSI (M1)\n"
            f"🕯 Timeframe: M1\n"
            f"⏱ Expiry: {DURATION_MIN}m",
            reply_markup=main_keyboard()
        )

    elif q.data == "STOP_SCAN":
        bot_logic.is_scanning = False
        await q.edit_message_text("⏹️ Scanner stopped.", reply_markup=main_keyboard())

    elif q.data == "TEST_BUY":
        await bot_logic.execute_trade("CALL", "R_10", "Manual Test", source="MANUAL")

    elif q.data == "STATUS":
        await bot_logic.fetch_balance()
        now_time = datetime.now(ZoneInfo("Africa/Lagos")).strftime("%Y-%m-%d %H:%M:%S")
        _ok, gate = bot_logic.can_auto_trade()

        trade_status = "No Active Trade"
        if bot_logic.active_trade_info and bot_logic.api:
            try:
                res = await bot_logic.api.proposal_open_contract({"proposal_open_contract": 1, "contract_id": bot_logic.active_trade_info})
                pnl = float(res["proposal_open_contract"].get("profit", 0))
                rem = max(0, int(DURATION_MIN * 60) - int(time.time() - bot_logic.trade_start_time))
                icon = "✅ PROFIT" if pnl > 0 else "❌ LOSS" if pnl < 0 else "➖ FLAT"
                mkt_clean = str(bot_logic.active_market).replace("_", " ")
                trade_status = f"🚀 Active Trade ({mkt_clean})\n📈 PnL: {icon} ({pnl:+.2f})\n⏳ Left: {rem}s"
            except:
                trade_status = "🚀 Active Trade: Syncing..."

        header = (
            f"🕒 Time (WAT): {now_time}\n"
            f"🤖 Bot: {'ACTIVE' if bot_logic.is_scanning else 'OFFLINE'} ({bot_logic.account_type})\n"
            f"📌 Strategy: Trend + Pullback(Prev) + ConfirmColor + RSI (M1)\n"
            f"⏱ Expiry: {DURATION_MIN}m | Cooldown: {COOLDOWN_SEC}s\n"
            f"📡 Markets: {', '.join(MARKETS).replace('_',' ')}\n"
            f"━━━━━━━━━━━━━━━\n{trade_status}\n━━━━━━━━━━━━━━━\n"
            f"💵 Total Profit Today: {bot_logic.total_profit_today:+.2f}\n"
            f"🎯 Trades: {bot_logic.trades_today}/{MAX_TRADES_PER_DAY} | ❌ Losses: {bot_logic.total_losses_today}\n"
            f"📉 Loss Streak: {bot_logic.consecutive_losses}/{MAX_CONSEC_LOSSES} | 🧪 Next Stake: ${bot_logic.current_stake:.2f}\n"
            f"🚦 Gate: {gate}\n"
            f"💰 Balance: {bot_logic.balance}\n"
        )

        details = "\n\n📌 LIVE SCAN (Simple)\n\n" + "\n\n".join(
            [format_market_detail(sym, bot_logic.market_debug.get(sym, {})) for sym in MARKETS]
        )

        await q.edit_message_text(header + details, reply_markup=main_keyboard())

async def start_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    await u.message.reply_text(
        "💎 Deriv Bot\n"
        "📌 Strategy: Trend + Pullback(Prev) + ConfirmColor + RSI (M1)\n"
        "🕯 Timeframe: M1\n"
        f"⏱ Expiry: {DURATION_MIN}m\n",
        reply_markup=main_keyboard()
    )

if __name__ == "__main__":
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    bot_logic.app = app
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CallbackQueryHandler(btn_handler))
    app.run_polling(drop_pending_updates=True)
