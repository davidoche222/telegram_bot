# ⚠️ SECURITY NOTE:
# You posted your Deriv + Telegram tokens publicly in this chat.
# Regenerate/revoke them in Deriv and BotFather, then replace below.

import asyncio
import logging
import random
import time
from collections import deque
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
from deriv_api import DerivAPI
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

# ========================= CONFIG =========================
DEMO_TOKEN = "tIrfitLjqeBxCOM"
REAL_TOKEN = "ZkOFWOlPtwnjqTS"
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
TF_SEC = 60          # M1 candles
CANDLES_COUNT = 150  # enough for EMA50 + RSI
SCAN_SLEEP_SEC = 2   # small idle (main pacing is candle-sync)

RSI_PERIOD = 14
DURATION_MIN = 1     # ✅ 1-minute expiry

# ========================= EMA50 SLOPE + DAILY TARGET =========================
EMA_SLOPE_LOOKBACK = 10
EMA_SLOPE_MIN = 0.0
DAILY_PROFIT_TARGET = 5.0

# ========================= PAYOUT MODE (FIXED PAYOUT) =========================
USE_PAYOUT_MODE = True
PAYOUT_TARGET = 1.00          # fixed payout per trade
MAX_STAKE_ALLOWED = 5.00      # skip trade if required stake > this
BUY_PRICE_BUFFER = 0.00       # keep 0.00 (we use fixed cap below)

# ========================= MARTINGALE (PAYOUT MODE) =========================
# In payout mode, martingale increases PAYOUT after losses.
MARTINGALE_MULT = 1.8
MARTINGALE_MAX_STEPS = 4
MARTINGALE_MAX_STAKE = 16.0   # display only

# ========================= 4 SECTIONS PER DAY =========================
SECTIONS_PER_DAY = 4
SECTION_PROFIT_TARGET = 1.50  # ✅ stop section once +$1.50 profit is reached
SECTION_LENGTH_SEC = int(24 * 60 * 60 / SECTIONS_PER_DAY)

# ========================= ANTI RATE-LIMIT (THE FIX) =========================
# 1) Global min interval between ANY ticks_history calls (across markets)
TICKS_GLOBAL_MIN_INTERVAL = 0.35
# 2) Per-market next-poll scheduling aligned to candle close
RATE_LIMIT_BACKOFF_BASE = 20  # backoff grows with strikes

# ========================= UI REFRESH COOLDOWN =========================
STATUS_REFRESH_COOLDOWN_SEC = 10

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
    out = []
    for x in candles_raw:
        out.append(
            {
                "t0": int(x.get("epoch", 0)),
                "o": float(x.get("open", 0)),
                "h": float(x.get("high", 0)),
                "l": float(x.get("low", 0)),
                "c": float(x.get("close", 0)),
            }
        )
    return out


def fmt_time_hhmmss(epoch):
    try:
        return datetime.fromtimestamp(epoch, ZoneInfo("Africa/Lagos")).strftime("%H:%M:%S")
    except Exception:
        return "—"


def fmt_hhmm(epoch):
    try:
        return datetime.fromtimestamp(epoch, ZoneInfo("Africa/Lagos")).strftime("%H:%M")
    except Exception:
        return "—"


def money2(x: float) -> float:
    # round UP to 2dp (helps avoid buy cap rounding down)
    import math
    return math.ceil(float(x) * 100.0) / 100.0


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

        # payout-mode state
        self.current_stake = 0.0        # display (ask_price)
        self.martingale_step = 0

        # 4 sections/day state
        self.section_profit = 0.0
        self.sections_won_today = 0
        self.section_index = 1
        self.section_pause_until = 0.0  # epoch

        self.trade_lock = asyncio.Lock()

        self.market_debug = {m: {} for m in MARKETS}
        self.last_processed_closed_t0 = {m: 0 for m in MARKETS}

        self.tz = ZoneInfo("Africa/Lagos")
        self.current_day = datetime.now(self.tz).date()
        self.pause_until = 0.0

        # anti rate-limit state
        self._ticks_lock = asyncio.Lock()
        self._last_ticks_ts = 0.0
        self._next_poll_epoch = {m: 0.0 for m in MARKETS}
        self._rate_limit_strikes = {m: 0 for m in MARKETS}

        # refresh cooldown
        self.status_cooldown_until = 0.0

        # stats (optional)
        self.stats = {m: {"trades": 0, "wins": 0, "losses": 0, "pnl": 0.0} for m in MARKETS}
        self.last_results = {m: deque(maxlen=20) for m in MARKETS}

    # ---------- error helpers ----------
    @staticmethod
    def _is_gatewayish_error(msg: str) -> bool:
        m = (msg or "").lower()
        return any(
            k in m
            for k in [
                "gateway", "bad gateway", "502", "503", "504",
                "timeout", "timed out",
                "temporarily unavailable",
                "connection", "websocket", "not connected", "disconnect",
                "internal server error", "service unavailable",
            ]
        )

    @staticmethod
    def _is_rate_limit_error(msg: str) -> bool:
        m = (msg or "").lower()
        return ("rate limit" in m) or ("reached the rate limit" in m) or ("too many requests" in m) or ("429" in m)

    # ---------- telegram safe send ----------
    async def safe_send_tg(self, text: str, retries: int = 5):
        if not self.app:
            return
        last_err = None
        for i in range(1, retries + 1):
            try:
                await self.app.bot.send_message(TELEGRAM_CHAT_ID, text)
                return
            except Exception as e:
                last_err = e
                msg = str(e)
                if self._is_gatewayish_error(msg):
                    await asyncio.sleep(0.8 * i + random.random() * 0.4)
                else:
                    await asyncio.sleep(0.4 * i)
        logger.warning(f"Telegram send failed after retries: {last_err}")

    # ---------- Sections helpers ----------
    def _today_midnight_epoch(self) -> float:
        now = datetime.now(self.tz)
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return midnight.timestamp()

    def _get_section_index_for_epoch(self, epoch_ts: float) -> int:
        midnight = self._today_midnight_epoch()
        sec_into_day = max(0, int(epoch_ts - midnight))
        idx0 = min(SECTIONS_PER_DAY - 1, sec_into_day // SECTION_LENGTH_SEC)
        return int(idx0 + 1)

    def _next_section_start_epoch(self, epoch_ts: float) -> float:
        midnight = self._today_midnight_epoch()
        sec_into_day = max(0, int(epoch_ts - midnight))
        idx0 = min(SECTIONS_PER_DAY - 1, sec_into_day // SECTION_LENGTH_SEC)
        next_start = midnight + (idx0 + 1) * SECTION_LENGTH_SEC
        if idx0 + 1 >= SECTIONS_PER_DAY:
            next_midnight = (datetime.fromtimestamp(midnight, self.tz) + timedelta(days=1)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            return next_midnight.timestamp()
        return float(next_start)

    def _sync_section_if_needed(self):
        now = time.time()
        today = datetime.now(self.tz).date()
        if today != self.current_day:
            return
        new_idx = self._get_section_index_for_epoch(now)
        if new_idx != self.section_index:
            self.section_index = new_idx
            self.section_profit = 0.0
            self.section_pause_until = 0.0

    # ---------- Deriv connection ----------
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

    async def safe_reconnect(self) -> bool:
        try:
            if self.api:
                try:
                    await self.api.disconnect()
                except Exception:
                    pass
        except Exception:
            pass
        self.api = None
        return await self.connect()

    async def safe_deriv_call(self, fn_name: str, payload: dict, retries: int = 6):
        last_err = None
        for attempt in range(1, retries + 1):
            try:
                if not self.api:
                    ok = await self.safe_reconnect()
                    if not ok:
                        raise RuntimeError("Reconnect failed")
                fn = getattr(self.api, fn_name)
                return await fn(payload)
            except Exception as e:
                last_err = e
                msg = str(e)
                if self._is_gatewayish_error(msg):
                    await self.safe_reconnect()
                if self._is_rate_limit_error(msg):
                    await asyncio.sleep(min(20.0, 2.5 * attempt + random.random()))
                else:
                    await asyncio.sleep(min(8.0, 0.6 * attempt + random.random() * 0.5))
        raise last_err

    async def safe_ticks_history(self, payload: dict, retries: int = 4):
        async with self._ticks_lock:
            now = time.time()
            gap = (self._last_ticks_ts + TICKS_GLOBAL_MIN_INTERVAL) - now
            if gap > 0:
                await asyncio.sleep(gap)
            self._last_ticks_ts = time.time()
        return await self.safe_deriv_call("ticks_history", payload, retries=retries)

    async def fetch_balance(self):
        if not self.api:
            return
        try:
            bal = await self.safe_deriv_call("balance", {"balance": 1}, retries=4)
            self.balance = f"{float(bal['balance']['balance']):.2f} {bal['balance']['currency']}"
        except Exception:
            pass

    # ---------- Daily reset ----------
    def _next_midnight_epoch(self) -> float:
        now = datetime.now(self.tz)
        next_midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        return next_midnight.timestamp()

    def _daily_reset_if_needed(self):
        today = datetime.now(self.tz).date()
        if today != self.current_day:
            self.current_day = today

            self.trades_today = 0
            self.total_losses_today = 0
            self.consecutive_losses = 0
            self.total_profit_today = 0.0
            self.cooldown_until = 0.0
            self.pause_until = 0.0

            self.martingale_step = 0
            self.current_stake = 0.0

            self.section_profit = 0.0
            self.sections_won_today = 0
            self.section_index = self._get_section_index_for_epoch(time.time())
            self.section_pause_until = 0.0

            self.stats = {m: {"trades": 0, "wins": 0, "losses": 0, "pnl": 0.0} for m in MARKETS}
            self.last_results = {m: deque(maxlen=20) for m in MARKETS}

        self._sync_section_if_needed()

    def can_auto_trade(self) -> tuple[bool, str]:
        self._daily_reset_if_needed()

        if time.time() < self.section_pause_until:
            left = int(self.section_pause_until - time.time())
            return False, f"Section paused. Resumes {fmt_hhmm(self.section_pause_until)} ({left}s)"

        if time.time() < self.pause_until:
            left = int(self.pause_until - time.time())
            return False, f"Paused until 12:00am WAT ({left}s)"

        if self.total_profit_today >= DAILY_PROFIT_TARGET:
            self.pause_until = self._next_midnight_epoch()
            return False, f"Daily target reached (+${self.total_profit_today:.2f})"

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

    # ---------- Scanner loop ----------
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
            "granularity": TF_SEC,
        }
        data = await self.safe_ticks_history(payload, retries=4)
        return build_candles_from_deriv(data.get("candles", []))

    async def _sleep_until(self, epoch_ts: float, buffer_s: float = 0.15):
        wait = (epoch_ts - time.time()) + buffer_s
        if wait > 0:
            await asyncio.sleep(wait)
        else:
            await asyncio.sleep(0.10)

    async def _sync_to_next_candle_open(self, last_closed_t0: int):
        next_open = int(last_closed_t0) + int(TF_SEC)
        await self._sleep_until(next_open, buffer_s=0.20)

    async def scan_market(self, symbol: str):
        self._next_poll_epoch[symbol] = time.time() + random.random() * 0.5

        while self.is_scanning:
            try:
                now = time.time()
                nxt = float(self._next_poll_epoch.get(symbol, 0.0))
                if now < nxt:
                    await asyncio.sleep(min(1.0, nxt - now))
                    continue

                ok_gate, gate = self.can_auto_trade()

                candles = await self.fetch_real_m1_candles(symbol)
                if len(candles) < 70:
                    self.market_debug[symbol] = {
                        "time": time.time(),
                        "gate": "Waiting for more candles",
                        "why": [f"Not enough candle history yet (need ~70, have {len(candles)})."],
                    }
                    self._next_poll_epoch[symbol] = time.time() + 10
                    continue

                pullback = candles[-3]
                confirm = candles[-2]
                confirm_t0 = int(confirm["t0"])

                # schedule next poll for after next candle close (rate-limit fix)
                next_closed_epoch = confirm_t0 + TF_SEC
                self._next_poll_epoch[symbol] = float(next_closed_epoch + 0.35)

                if self.last_processed_closed_t0[symbol] == confirm_t0:
                    continue

                closes = [x["c"] for x in candles]

                ema20_arr = calculate_ema(closes, 20)
                ema50_arr = calculate_ema(closes, 50)
                if len(ema20_arr) < 60 or len(ema50_arr) < 60:
                    self.market_debug[symbol] = {"time": time.time(), "gate": "Indicators", "why": ["EMA20/EMA50 not ready yet."]}
                    self.last_processed_closed_t0[symbol] = confirm_t0
                    continue

                ema20_pullback = float(ema20_arr[-3])
                ema20_confirm = float(ema20_arr[-2])
                ema50_confirm = float(ema50_arr[-2])

                # EMA50 slope filter
                slope_ok = False
                ema50_slope = 0.0
                ema50_rising = False
                ema50_falling = False
                if len(ema50_arr) >= (EMA_SLOPE_LOOKBACK + 3):
                    ema50_prev = float(ema50_arr[-(EMA_SLOPE_LOOKBACK + 2)])
                    ema50_slope = ema50_confirm - ema50_prev
                    ema50_rising = ema50_slope > EMA_SLOPE_MIN
                    ema50_falling = ema50_slope < -EMA_SLOPE_MIN
                    slope_ok = True

                rsi_arr = calculate_rsi(closes, RSI_PERIOD)
                if len(rsi_arr) < 60 or np.isnan(rsi_arr[-2]):
                    self.market_debug[symbol] = {"time": time.time(), "gate": "Indicators", "why": ["RSI not ready yet."]}
                    self.last_processed_closed_t0[symbol] = confirm_t0
                    continue
                rsi_now = float(rsi_arr[-2])

                pb_high = float(pullback["h"])
                pb_low = float(pullback["l"])
                touched_ema20 = (pb_low <= ema20_pullback <= pb_high)

                # candle colour confirmation (no doji)
                c_open = float(confirm["o"])
                c_close = float(confirm["c"])
                is_doji = (c_close == c_open)
                bull_confirm = (c_close > c_open) and (not is_doji)
                bear_confirm = (c_close < c_open) and (not is_doji)

                close_above_ema20 = c_close > ema20_confirm
                close_below_ema20 = c_close < ema20_confirm

                bodies = [abs(float(candles[i]["c"]) - float(candles[i]["o"])) for i in range(-22, -2)]
                avg_body = float(np.mean(bodies)) if len(bodies) >= 10 else float(
                    np.mean([abs(float(c["c"]) - float(c["o"])) for c in candles[-60:-2]])
                )
                last_body = abs(c_close - c_open)

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

                ema_diff = abs(ema20_confirm - ema50_confirm)
                flat_block = ema_diff < ema_diff_min

                uptrend = ema20_confirm > ema50_confirm
                downtrend = ema20_confirm < ema50_confirm

                call_rsi_ok = (rsi_call_min <= rsi_now <= rsi_call_max)
                put_rsi_ok = (rsi_put_min <= rsi_now <= rsi_put_max)

                call_ready = (
                    uptrend and slope_ok and ema50_rising and touched_ema20 and bull_confirm
                    and close_above_ema20 and call_rsi_ok and not spike_block and not flat_block
                )

                put_ready = (
                    downtrend and slope_ok and ema50_falling and touched_ema20 and bear_confirm
                    and close_below_ema20 and put_rsi_ok and not spike_block and not flat_block
                )

                signal = "CALL" if call_ready else "PUT" if put_ready else None

                trend_label = "UPTREND" if uptrend else "DOWNTREND" if downtrend else "SIDEWAYS"
                ema_label = "EMA20 ABOVE EMA50" if uptrend else "EMA20 BELOW EMA50" if downtrend else "EMA20 = EMA50"
                trend_strength = "STRONG" if not flat_block else "WEAK"
                pullback_label = "PULLBACK TOUCHED ✅" if touched_ema20 else "WAITING PULLBACK…"
                confirm_close_label = (
                    "CONFIRM CLOSE > EMA20 ✅" if close_above_ema20 else
                    "CONFIRM CLOSE < EMA20 ✅" if close_below_ema20 else
                    "CONFIRM CLOSE ON EMA20"
                )
                slope_label = "EMA50 SLOPE ↑" if ema50_rising else "EMA50 SLOPE ↓" if ema50_falling else "EMA50 SLOPE FLAT"

                block_label_parts = []
                if spike_block:
                    block_label_parts.append("SPIKE BLOCK")
                if flat_block:
                    block_label_parts.append("WEAK/FLAT TREND")
                if not slope_ok:
                    block_label_parts.append("SLOPE N/A")
                if is_doji:
                    block_label_parts.append("DOJI BLOCK")
                block_label = " | ".join(block_label_parts) if block_label_parts else "OK"

                why = []
                if not ok_gate:
                    why.append(f"Gate blocked: {gate}")
                if signal:
                    why.append(f"READY: {signal} (enter next candle)")

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
                    "confirm_close_label": confirm_close_label,
                    "slope_label": slope_label,
                    "ema50_slope": ema50_slope,
                    "rsi_now": rsi_now,
                    "avg_body": avg_body,
                    "last_body": last_body,
                    "spike_block": spike_block,
                    "flat_block": flat_block,
                    "why": why[:10],
                }

                self.last_processed_closed_t0[symbol] = confirm_t0

                if not ok_gate:
                    continue

                if call_ready or put_ready:
                    await self._sync_to_next_candle_open(confirm_t0)

                if call_ready:
                    await self.execute_trade("CALL", symbol, reason="AUTO", source="AUTO")
                elif put_ready:
                    await self.execute_trade("PUT", symbol, reason="AUTO", source="AUTO")

            except asyncio.CancelledError:
                break
            except Exception as e:
                msg = str(e)
                logger.error(f"Scanner Error ({symbol}): {msg}")
                self.market_debug[symbol] = {"time": time.time(), "gate": "Error", "why": [msg[:160]]}

                if self._is_rate_limit_error(msg):
                    self._rate_limit_strikes[symbol] = int(self._rate_limit_strikes.get(symbol, 0)) + 1
                    backoff = RATE_LIMIT_BACKOFF_BASE * self._rate_limit_strikes[symbol]
                    backoff = min(180, backoff)
                    self._next_poll_epoch[symbol] = time.time() + backoff
                else:
                    await asyncio.sleep(2 if not self._is_gatewayish_error(msg) else 5)

            await asyncio.sleep(0.05)

    # ========================= TRADE EXECUTION (PAYOUT MODE) =========================
    async def execute_trade(self, side: str, symbol: str, reason="MANUAL", source="MANUAL"):
        if not self.api or self.active_trade_info:
            return

        async with self.trade_lock:
            ok, _gate = self.can_auto_trade()
            if not ok:
                return

            try:
                payout = money2(float(PAYOUT_TARGET) * (float(MARTINGALE_MULT) ** int(self.martingale_step)))
                payout = max(0.01, float(payout))

                proposal_req = {
                    "proposal": 1,
                    "amount": payout,
                    "basis": "payout",
                    "contract_type": side,
                    "currency": "USD",
                    "duration": int(DURATION_MIN),
                    "duration_unit": "m",
                    "symbol": symbol,
                }

                prop = await self.safe_deriv_call("proposal", proposal_req, retries=6)
                if "error" in prop:
                    err = prop["error"].get("message", "Proposal error")
                    await self.safe_send_tg(f"❌ Proposal Error:\n{err}")
                    return

                p = prop["proposal"]
                proposal_id = p["id"]
                ask_price = float(p.get("ask_price", 0.0))
                if ask_price <= 0:
                    await self.safe_send_tg("❌ Proposal returned invalid ask_price.")
                    return

                if ask_price > float(MAX_STAKE_ALLOWED):
                    await self.safe_send_tg(
                        f"⛔️ Skipped trade: payout=${payout:.2f} needs stake=${ask_price:.2f} > max ${MAX_STAKE_ALLOWED:.2f}"
                    )
                    self.cooldown_until = time.time() + COOLDOWN_SEC
                    return

                buy_price_cap = float(MAX_STAKE_ALLOWED) + float(BUY_PRICE_BUFFER)

                buy = await self.safe_deriv_call("buy", {"buy": proposal_id, "price": buy_price_cap}, retries=6)
                if "error" in buy:
                    err_msg = str(buy["error"].get("message", "Buy error"))
                    low = err_msg.lower()
                    if ("moved too much" in low) or ("price has changed" in low):
                        await asyncio.sleep(0.25)
                        prop2 = await self.safe_deriv_call("proposal", proposal_req, retries=6)
                        if "error" in prop2:
                            err2 = prop2["error"].get("message", "Proposal error")
                            await self.safe_send_tg(f"❌ Proposal Error (retry):\n{err2}")
                            return

                        p2 = prop2["proposal"]
                        proposal_id2 = p2["id"]
                        ask_price2 = float(p2.get("ask_price", 0.0))
                        if ask_price2 <= 0:
                            await self.safe_send_tg("❌ Proposal retry returned invalid ask_price.")
                            return
                        if ask_price2 > float(MAX_STAKE_ALLOWED):
                            await self.safe_send_tg(
                                f"⛔️ Skipped (retry): payout=${payout:.2f} needs stake=${ask_price2:.2f} > max ${MAX_STAKE_ALLOWED:.2f}"
                            )
                            self.cooldown_until = time.time() + COOLDOWN_SEC
                            return

                        buy = await self.safe_deriv_call("buy", {"buy": proposal_id2, "price": buy_price_cap}, retries=6)
                        if "error" in buy:
                            err3 = buy["error"].get("message", "Buy error")
                            await self.safe_send_tg(f"❌ Trade Refused (retry):\n{err3}")
                            return

                        ask_price = ask_price2
                    else:
                        await self.safe_send_tg(f"❌ Trade Refused:\n{err_msg}")
                        return

                self.active_trade_info = int(buy["buy"]["contract_id"])
                self.active_market = symbol
                self.trade_start_time = time.time()
                self.current_stake = ask_price

                if source == "AUTO":
                    self.trades_today += 1

                safe_symbol = str(symbol).replace("_", " ")
                await self.safe_send_tg(
                    f"🚀 {side} TRADE OPENED\n"
                    f"🛒 Market: {safe_symbol}\n"
                    f"⏱ Expiry: {DURATION_MIN}m\n"
                    f"🎁 Payout: ${payout:.2f}\n"
                    f"💵 Stake (Deriv): ${ask_price:.2f}\n"
                    f"🧾 Buy cap: ${buy_price_cap:.2f}\n"
                    f"🧩 Section: {self.section_index}/{SECTIONS_PER_DAY} | Section PnL: {self.section_profit:+.2f} / +{SECTION_PROFIT_TARGET:.2f}\n"
                    f"🎯 Today PnL: {self.total_profit_today:+.2f} / +{DAILY_PROFIT_TARGET:.2f}\n"
                    f"🧪 Martingale step: {self.martingale_step}/{MARTINGALE_MAX_STEPS}"
                )

                asyncio.create_task(self.check_result(self.active_trade_info, source, symbol))

            except Exception as e:
                logger.error(f"Trade error: {e}")
                await self.safe_send_tg(f"⚠️ Trade error:\n{e}")

    async def check_result(self, cid: int, source: str, symbol: str):
        await asyncio.sleep(int(DURATION_MIN) * 60 + 5)
        try:
            res = await self.safe_deriv_call(
                "proposal_open_contract",
                {"proposal_open_contract": 1, "contract_id": cid},
                retries=6,
            )
            profit = float(res["proposal_open_contract"].get("profit", 0.0))

            if source == "AUTO":
                self.total_profit_today += profit
                self.section_profit += profit

                if symbol in self.stats:
                    self.stats[symbol]["trades"] += 1
                    self.stats[symbol]["pnl"] += profit
                    if profit > 0:
                        self.stats[symbol]["wins"] += 1
                        self.last_results[symbol].append(1)
                    else:
                        self.stats[symbol]["losses"] += 1
                        self.last_results[symbol].append(0)

                # ✅ section rule
                if self.section_profit >= float(SECTION_PROFIT_TARGET):
                    self.sections_won_today += 1
                    self.section_pause_until = self._next_section_start_epoch(time.time())

                # ✅ martingale on payout
                if profit <= 0:
                    self.consecutive_losses += 1
                    self.total_losses_today += 1
                    self.martingale_step = min(self.martingale_step + 1, MARTINGALE_MAX_STEPS)
                else:
                    self.consecutive_losses = 0
                    self.martingale_step = 0

                if self.total_profit_today >= DAILY_PROFIT_TARGET:
                    self.pause_until = self._next_midnight_epoch()

            await self.fetch_balance()

            pause_note = "\n⏸ Paused until 12:00am WAT" if time.time() < self.pause_until else ""
            section_note = (
                f"\n🧩 Section paused until {fmt_hhmm(self.section_pause_until)}"
                if time.time() < self.section_pause_until
                else ""
            )
            next_payout = money2(float(PAYOUT_TARGET) * (float(MARTINGALE_MULT) ** int(self.martingale_step)))

            await self.safe_send_tg(
                f"🏁 FINISH: {'WIN' if profit > 0 else 'LOSS'} ({profit:+.2f})\n"
                f"🧩 Section: {self.section_index}/{SECTIONS_PER_DAY} | Section PnL: {self.section_profit:+.2f} / +{SECTION_PROFIT_TARGET:.2f} | Sections won: {self.sections_won_today}\n"
                f"📊 Today: {self.trades_today}/{MAX_TRADES_PER_DAY} | ❌ Losses: {self.total_losses_today} | Streak: {self.consecutive_losses}/{MAX_CONSEC_LOSSES}\n"
                f"💵 Today PnL: {self.total_profit_today:+.2f} / +{DAILY_PROFIT_TARGET:.2f}\n"
                f"🎁 Next payout: ${next_payout:.2f} (step {self.martingale_step}/{MARTINGALE_MAX_STEPS})\n"
                f"💰 Balance: {self.balance}"
                f"{pause_note}{section_note}"
            )
        finally:
            self.active_trade_info = None
            self.cooldown_until = time.time() + COOLDOWN_SEC


# ========================= UI =========================
bot_logic = DerivSniperBot()


def main_keyboard():
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("▶️ START", callback_data="START_SCAN"),
                InlineKeyboardButton("⏹️ STOP", callback_data="STOP_SCAN"),
            ],
            [
                InlineKeyboardButton("📊 STATUS", callback_data="STATUS"),
                InlineKeyboardButton("🔄 REFRESH", callback_data="STATUS"),
            ],
            [InlineKeyboardButton("🧩 NEXT SECTION", callback_data="NEXT_SECTION")],
            [InlineKeyboardButton("🧪 TEST BUY", callback_data="TEST_BUY")],
            [
                InlineKeyboardButton("🧪 DEMO", callback_data="SET_DEMO"),
                InlineKeyboardButton("💰 LIVE", callback_data="SET_REAL"),
            ],
        ]
    )


def _winrate_line_for(mkt: str) -> str:
    s = bot_logic.stats.get(mkt, {})
    trades = int(s.get("trades", 0))
    wins = int(s.get("wins", 0))
    pnl = float(s.get("pnl", 0.0))
    wr = (wins / trades * 100.0) if trades > 0 else 0.0
    last = list(bot_logic.last_results.get(mkt, []))
    last_wr = (sum(last) / len(last) * 100.0) if last else 0.0
    return f"{mkt.replace('_',' ')}: {wins}/{trades} ({wr:.1f}%) | last{len(last)}: {last_wr:.1f}% | pnl {pnl:+.2f}"


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
    confirm_close_label = d.get("confirm_close_label", "—")
    slope_label = d.get("slope_label", "—")

    rsi_now = d.get("rsi_now", None)
    ema50_slope = d.get("ema50_slope", None)

    extra = []
    if isinstance(rsi_now, (int, float)) and not np.isnan(rsi_now):
        extra.append(f"RSI: {rsi_now:.2f}")
    if isinstance(ema50_slope, (int, float)) and not np.isnan(ema50_slope):
        extra.append(f"EMA50 slope: {ema50_slope:.3f}")
    extra_line = " | ".join(extra) if extra else "—"

    why = d.get("why", [])
    why_line = "Why: " + (str(why[0]) if why else "—")

    return (
        f"📍 {sym.replace('_',' ')} ({age}s)\n"
        f"Gate: {gate}\n"
        f"Last closed: {fmt_time_hhmmss(last_closed)}\n"
        f"────────────────\n"
        f"Trend: {trend_label} ({trend_strength})\n"
        f"{ema_label}\n"
        f"{slope_label}\n"
        f"{pullback_label}\n"
        f"{confirm_close_label}\n"
        f"Stats: {extra_line}\n"
        f"Filters: {block_label}\n"
        f"Signal: {signal}\n"
        f"{why_line}\n"
    )


# ========================= CALLBACK SAFETY =========================
async def _safe_answer(q, text: str | None = None, show_alert: bool = False):
    try:
        await q.answer(text=text, show_alert=show_alert)
    except Exception as e:
        logger.warning(f"Callback answer ignored: {e}")


async def _safe_edit(q, text: str, reply_markup=None):
    try:
        await q.edit_message_text(text, reply_markup=reply_markup)
    except Exception as e:
        logger.warning(f"Edit failed: {e}")


async def btn_handler(u: Update, c: ContextTypes.DEFAULT_TYPE):
    q = u.callback_query
    await _safe_answer(q)

    # quick "working" message to avoid long waits
    await _safe_edit(q, "⏳ Working...", reply_markup=main_keyboard())

    if q.data == "SET_DEMO":
        bot_logic.active_token, bot_logic.account_type = DEMO_TOKEN, "DEMO"
        ok = await bot_logic.connect()
        await _safe_edit(q, "✅ Connected to DEMO" if ok else "❌ DEMO Failed", reply_markup=main_keyboard())

    elif q.data == "SET_REAL":
        bot_logic.active_token, bot_logic.account_type = REAL_TOKEN, "LIVE"
        ok = await bot_logic.connect()
        await _safe_edit(q, "✅ LIVE CONNECTED" if ok else "❌ LIVE Failed", reply_markup=main_keyboard())

    elif q.data == "START_SCAN":
        if not bot_logic.api:
            await _safe_edit(q, "❌ Connect first.", reply_markup=main_keyboard())
            return
        bot_logic.is_scanning = True
        bot_logic.scanner_task = asyncio.create_task(bot_logic.background_scanner())
        await _safe_edit(
            q,
            (
                "🔍 SCANNER ACTIVE\n"
                f"🕯 Timeframe: M1 | ⏱ Expiry: {DURATION_MIN}m\n"
                f"🎁 PAYOUT MODE: ${PAYOUT_TARGET:.2f} per trade (martingale affects payout)\n"
                f"🧯 Max stake allowed: ${MAX_STAKE_ALLOWED:.2f}\n"
                f"🧩 Sections/day: {SECTIONS_PER_DAY} | Stop each section at +${SECTION_PROFIT_TARGET:.2f}\n"
                f"🎯 Daily target: +${DAILY_PROFIT_TARGET:.2f} (pause till 12am WAT)\n"
                "✅ Anti-rate-limit enabled (global pacing + candle-aligned polling)\n"
            ),
            reply_markup=main_keyboard(),
        )

    elif q.data == "STOP_SCAN":
        bot_logic.is_scanning = False
        if bot_logic.scanner_task and not bot_logic.scanner_task.done():
            bot_logic.scanner_task.cancel()
        await _safe_edit(q, "⏹️ Scanner stopped.", reply_markup=main_keyboard())

    elif q.data == "NEXT_SECTION":
        bot_logic._daily_reset_if_needed()
        now = time.time()
        nxt = bot_logic._next_section_start_epoch(now)
        if nxt <= now + 1:
            nxt = now + 1
        forced_idx = bot_logic._get_section_index_for_epoch(nxt + 1)
        bot_logic.section_index = forced_idx
        bot_logic.section_profit = 0.0
        bot_logic.section_pause_until = 0.0
        await _safe_edit(
            q,
            f"🧩 Moved to Section {bot_logic.section_index}/{SECTIONS_PER_DAY}. Reset section PnL to 0.00.",
            reply_markup=main_keyboard(),
        )

    elif q.data == "TEST_BUY":
        asyncio.create_task(bot_logic.execute_trade("CALL", "R_10", "Manual Test", source="MANUAL"))
        await _safe_edit(q, "🧪 Test trade triggered (CALL R 10).", reply_markup=main_keyboard())

    elif q.data == "STATUS":
        now = time.time()
        if now < bot_logic.status_cooldown_until:
            left = int(bot_logic.status_cooldown_until - now)
            await _safe_edit(q, f"⏳ Refresh cooldown: {left}s\n\nPress again after cooldown.", reply_markup=main_keyboard())
            return
        bot_logic.status_cooldown_until = now + STATUS_REFRESH_COOLDOWN_SEC

        await bot_logic.fetch_balance()
        now_time = datetime.now(ZoneInfo("Africa/Lagos")).strftime("%Y-%m-%d %H:%M:%S")
        _ok, gate = bot_logic.can_auto_trade()

        trade_status = "No Active Trade"
        if bot_logic.active_trade_info and bot_logic.api:
            try:
                res = await bot_logic.safe_deriv_call(
                    "proposal_open_contract",
                    {"proposal_open_contract": 1, "contract_id": bot_logic.active_trade_info},
                    retries=4,
                )
                pnl = float(res["proposal_open_contract"].get("profit", 0))
                rem = max(0, int(DURATION_MIN * 60) - int(time.time() - bot_logic.trade_start_time))
                icon = "✅ PROFIT" if pnl > 0 else "❌ LOSS" if pnl < 0 else "➖ FLAT"
                mkt_clean = str(bot_logic.active_market).replace("_", " ")
                trade_status = f"🚀 Active Trade ({mkt_clean})\n📈 PnL: {icon} ({pnl:+.2f})\n⏳ Left: {rem}s"
            except Exception:
                trade_status = "🚀 Active Trade: Syncing..."

        pause_line = "⏸ Paused until 12:00am WAT\n" if time.time() < bot_logic.pause_until else ""
        section_line = f"🧩 Section paused until {fmt_hhmm(bot_logic.section_pause_until)}\n" if time.time() < bot_logic.section_pause_until else ""

        next_payout = money2(float(PAYOUT_TARGET) * (float(MARTINGALE_MULT) ** int(bot_logic.martingale_step)))
        winrate_lines = "\n".join(_winrate_line_for(m) for m in MARKETS)

        header = (
            f"🕒 Time (WAT): {now_time}\n"
            f"🤖 Bot: {'ACTIVE' if bot_logic.is_scanning else 'OFFLINE'} ({bot_logic.account_type})\n"
            f"{pause_line}{section_line}"
            f"🎁 Payout: ${PAYOUT_TARGET:.2f} | Next payout: ${next_payout:.2f} | Step: {bot_logic.martingale_step}/{MARTINGALE_MAX_STEPS}\n"
            f"🧯 Max stake allowed: ${MAX_STAKE_ALLOWED:.2f}\n"
            f"🧩 Section: {bot_logic.section_index}/{SECTIONS_PER_DAY} | Section PnL: {bot_logic.section_profit:+.2f} / +{SECTION_PROFIT_TARGET:.2f} | Sections won: {bot_logic.sections_won_today}\n"
            f"⏱ Expiry: {DURATION_MIN}m | Cooldown: {COOLDOWN_SEC}s\n"
            f"🎯 Daily Target: +${DAILY_PROFIT_TARGET:.2f}\n"
            f"📡 Markets: {', '.join(MARKETS).replace('_',' ')}\n"
            f"📈 Winrate:\n{winrate_lines}\n"
            f"━━━━━━━━━━━━━━━\n{trade_status}\n━━━━━━━━━━━━━━━\n"
            f"💵 Total Profit Today: {bot_logic.total_profit_today:+.2f}\n"
            f"🎯 Trades: {bot_logic.trades_today}/{MAX_TRADES_PER_DAY} | ❌ Losses: {bot_logic.total_losses_today}\n"
            f"📉 Loss Streak: {bot_logic.consecutive_losses}/{MAX_CONSEC_LOSSES}\n"
            f"🚦 Gate: {gate}\n"
            f"💰 Balance: {bot_logic.balance}\n"
        )

        details = "\n\n📌 LIVE SCAN (FULL)\n\n" + "\n\n".join(
            [format_market_detail(sym, bot_logic.market_debug.get(sym, {})) for sym in MARKETS]
        )

        await _safe_edit(q, header + details, reply_markup=main_keyboard())


async def start_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    await u.message.reply_text(
        "💎 Deriv Bot\n"
        f"🕯 Timeframe: M1 | ⏱ Expiry: {DURATION_MIN}m\n"
        f"🎁 PAYOUT MODE: ${PAYOUT_TARGET:.2f} per trade\n"
        f"🧩 Sections/day: {SECTIONS_PER_DAY} | stop each section at +${SECTION_PROFIT_TARGET:.2f}\n"
        "✅ Anti-rate-limit enabled (global pacing + candle-aligned polling)\n",
        reply_markup=main_keyboard(),
    )


if __name__ == "__main__":
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    bot_logic.app = app
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CallbackQueryHandler(btn_handler))
    app.run_polling(drop_pending_updates=True)
