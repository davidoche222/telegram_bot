"""
Deriv Forex Binary Options Bot
Strategy: Structure Break + EMA50 Confirmation
Markets: EURUSD, GBPUSD, USDJPY, AUDUSD, GBPJPY

Entry Logic:
- M5: detect swing highs/lows (structure levels)
- M5: price closes beyond structure AND crosses EMA50 same candle
- M1: RSI genuinely crosses 50 in break direction (prev below, now above)
- M1: next candle body confirms direction
- ATR expansion confirms real momentum
- Body ratio blocks spike candles
"""

import asyncio, time, logging, json, os
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
from deriv_api import DerivAPI
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ===== CREDENTIALS =====
APP_ID       = 1089
DEMO_TOKEN   = "tIrfitLjqeBxCOM"
REAL_TOKEN   = "ZkOFWOlPtwnjqTS"
TELEGRAM_TOKEN = "8589420556:AAHmB6YE9KIEu0tBIgWdd9baBDt0eDh5FY8"

# ===== MARKETS =====
MARKETS = ["frxEURUSD", "frxGBPUSD", "frxUSDJPY", "frxAUDUSD", "frxGBPJPY"]

# ===== LIMITS =====
MAX_TRADES_PER_DAY      = 30
MAX_TRADES_PER_MARKET   = 6
MAX_LOSSES_PER_MARKET   = 3
CONSEC_LOSS_PAUSE_SEC   = 1800   # 30 min after 2 consecutive losses
MAX_CONSEC_LOSSES       = 5      # global stop
COOLDOWN_SEC            = 180    # 3 min after every trade

# ===== STRATEGY =====
TF_M1_SEC           = 60         # M1 only — all logic on 1 min candles
CANDLES_M1          = 150        # enough for swing detection + EMA50 + RSI
EXPIRY_MIN          = 5          # 5 min expiry — aligned with M5 structure
EMA_PERIOD          = 50         # EMA50 on M1 (cross detection)
ATR_PERIOD          = 14         # ATR on M5
ATR_SMA_PERIOD      = 20         # ATR average for volatility filter
RSI_PERIOD          = 14         # RSI on M1
SWING_LOOKBACK      = 5          # candles each side to confirm swing point
BODY_RATIO_MIN      = 0.35       # M5 break candle minimum body strength

# Weekend block only — trading all sessions to collect data
FOREX_WEEKEND_BLOCK = False  # trading all days including weekends

# ===== MARTINGALE =====
MARTINGALE_MULT      = 2
MARTINGALE_MAX_STEPS = 5
PAYOUT_TARGET        = 1.00
MIN_PAYOUT           = 0.35
MAX_STAKE_ALLOWED    = 10.00

# ===== RISK =====
DAILY_PROFIT_TARGET = 10.0
DAILY_LOSS_LIMIT    = -20.0
EQUITY_STOP_PCT     = 0.60
PROFIT_LOCK_TRIGGER = 5.0
PROFIT_LOCK_FLOOR   = 2.0

# ===== MISC =====
STATS_DAYS               = 30
STATUS_REFRESH_COOLDOWN_SEC = 10
TRADE_LOG_FILE           = "structure_trades.json"
SCAN_INTERVAL_SEC        = 20

# ============================================================
# INDICATORS
# ============================================================

def calculate_ema(closes, period):
    closes = np.array(closes, dtype=float)
    if len(closes) < period: return None
    k = 2.0 / (period + 1)
    ema = float(np.mean(closes[:period]))
    for c in closes[period:]:
        ema = c * k + ema * (1 - k)
    return ema

def calculate_ema_series(closes, period):
    """Returns full EMA array"""
    closes = np.array(closes, dtype=float)
    if len(closes) < period: return None
    k = 2.0 / (period + 1)
    ema = [float(np.mean(closes[:period]))]
    for c in closes[period:]:
        ema.append(c * k + ema[-1] * (1 - k))
    # pad front with None
    result = [None] * (period - 1) + ema
    return result

def calculate_rsi(closes, period=14):
    closes = np.array(closes, dtype=float)
    if len(closes) < period + 1: return None
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0: return 100.0
    rs = avg_gain / avg_loss
    return round(100.0 - (100.0 / (1 + rs)), 2)

def calculate_atr(highs, lows, closes, period=14):
    highs = np.array(highs, dtype=float)
    lows = np.array(lows, dtype=float)
    closes = np.array(closes, dtype=float)
    if len(closes) < period + 1: return None
    tr = np.maximum(highs[1:] - lows[1:],
         np.maximum(np.abs(highs[1:] - closes[:-1]),
                    np.abs(lows[1:] - closes[:-1])))
    atrs = [float(np.mean(tr[:period]))]
    for t in tr[period:]:
        atrs.append((atrs[-1] * (period - 1) + t) / period)
    return atrs


def find_swing_highs(highs, lookback=5):
    """Returns list of (index, value) for swing highs"""
    swings = []
    for i in range(lookback, len(highs) - lookback):
        if all(highs[i] >= highs[i-j] for j in range(1, lookback+1)) and \
           all(highs[i] >= highs[i+j] for j in range(1, lookback+1)):
            swings.append((i, highs[i]))
    return swings

def find_swing_lows(lows, lookback=5):
    """Returns list of (index, value) for swing lows"""
    swings = []
    for i in range(lookback, len(lows) - lookback):
        if all(lows[i] <= lows[i-j] for j in range(1, lookback+1)) and \
           all(lows[i] <= lows[i+j] for j in range(1, lookback+1)):
            swings.append((i, lows[i]))
    return swings

def get_structure_levels(highs, lows, lookback=5):
    """
    Returns last significant swing high and swing low
    These are the structure levels to watch for breaks
    """
    swing_highs = find_swing_highs(list(highs), lookback)
    swing_lows = find_swing_lows(list(lows), lookback)
    last_high = swing_highs[-1][1] if swing_highs else None
    last_low = swing_lows[-1][1] if swing_lows else None
    return last_high, last_low

def body_ratio(open_, close, high, low):
    candle_range = abs(high - low)
    if candle_range == 0: return 0
    return abs(close - open_) / candle_range

def is_forex_market_open():
    now_utc = datetime.now(ZoneInfo("UTC"))
    weekday = now_utc.weekday()
    hour_utc = now_utc.hour
    if FOREX_WEEKEND_BLOCK:
        if weekday == 5: return False, "Weekend — market closed (Saturday)"
        if weekday == 6: return False, "Weekend — market closed (Sunday)"
        if weekday == 4 and hour_utc >= 21: return False, "Weekend — market closing (Friday 9pm+ UTC)"
    return True, "OK"

def fmt_time_hhmmss(epoch):
    try: return datetime.fromtimestamp(epoch, ZoneInfo("Africa/Lagos")).strftime("%H:%M:%S")
    except: return "—"

def money2(x):
    return round(float(x), 2)

def session_bucket(ts):
    h = datetime.fromtimestamp(ts, ZoneInfo("UTC")).hour
    if 7 <= h < 12: return "London"
    elif 12 <= h < 17: return "NY Overlap"
    elif 17 <= h < 21: return "Late NY"
    else: return "Asian"

def build_candles_from_deriv(raw):
    candles = []
    for c in raw:
        candles.append({
            "epoch": int(c.get("epoch", 0)),
            "open":  float(c.get("open", 0)),
            "high":  float(c.get("high", 0)),
            "low":   float(c.get("low", 0)),
            "close": float(c.get("close", 0)),
        })
    candles.sort(key=lambda x: x["epoch"])
    return candles

# ============================================================
# BOT LOGIC
# ============================================================

class StructureBreakBot:
    def __init__(self):
        self.api = None
        self.app = None
        self.account_type = "DEMO"
        self.balance = "0.00 USD"
        self.starting_balance = 0.0
        self.is_scanning = False

        # martingale
        self.martingale_step = 0
        self.martingale_halt = False

        # daily tracking
        self.trades_today = 0
        self.total_profit_today = 0.0
        self.total_losses_today = 0
        self.consecutive_losses = 0
        self.max_loss_streak_today = 0
        self.last_reset_date = None

        # per market
        self.market_trades_today = {m: 0 for m in MARKETS}
        self.market_losses_today = {m: 0 for m in MARKETS}
        self.market_blocked = {m: False for m in MARKETS}
        self.market_pause_until = {m: 0 for m in MARKETS}

        # active trade
        self.active_trade_info = None
        self.active_market = None
        self.trade_start_time = 0

        # cooldown
        self.cooldown_until = 0
        self.pause_until = 0
        self.status_cooldown_until = 0

        # profit lock
        self.profit_lock_active = False

        # debug
        self.market_debug = {}
        self.last_processed_m5_t0 = {}

        # trade history
        self.trade_history = []
        self._load_trade_history()

    def _load_trade_history(self):
        try:
            if os.path.exists(TRADE_LOG_FILE):
                with open(TRADE_LOG_FILE, "r") as f:
                    self.trade_history = json.load(f)
        except: self.trade_history = []

    def _save_trade(self, record):
        self.trade_history.append(record)
        try:
            with open(TRADE_LOG_FILE, "w") as f:
                json.dump(self.trade_history[-500:], f)
        except: pass

    def _reset_daily_if_needed(self):
        today = datetime.now(ZoneInfo("Africa/Lagos")).date()
        if self.last_reset_date != today:
            self.last_reset_date = today
            self.trades_today = 0
            self.total_profit_today = 0.0
            self.total_losses_today = 0
            self.consecutive_losses = 0
            self.max_loss_streak_today = 0
            self.martingale_step = 0
            self.martingale_halt = False
            self.profit_lock_active = False
            self.market_trades_today = {m: 0 for m in MARKETS}
            self.market_losses_today = {m: 0 for m in MARKETS}
            self.market_blocked = {m: False for m in MARKETS}
            self.market_pause_until = {m: 0 for m in MARKETS}

    def _get_current_balance_float(self):
        try: return float(str(self.balance).replace(" USD", "").replace(",", ""))
        except: return self.starting_balance

    def _equity_ok(self):
        if self.starting_balance <= 0: return True, "OK", 1.0
        ratio = self._get_current_balance_float() / self.starting_balance
        if ratio <= EQUITY_STOP_PCT: return False, f"Equity stop — {ratio:.0%} of starting", 0.0
        if ratio >= 1.0: return True, "OK", 1.0
        if ratio >= 0.90: return True, "OK", 0.75
        if ratio >= 0.80: return True, "OK", 0.50
        return True, "OK", 0.25

    def _is_market_available(self, symbol):
        if self.market_blocked.get(symbol): return False, "Blocked for day"
        if time.time() < self.market_pause_until.get(symbol, 0):
            rem = int(self.market_pause_until[symbol] - time.time())
            return False, f"Paused {rem}s"
        if self.market_trades_today.get(symbol, 0) >= MAX_TRADES_PER_MARKET:
            return False, f"Max {MAX_TRADES_PER_MARKET} trades today"
        return True, "OK"

    def can_auto_trade(self):
        self._reset_daily_if_needed()
        if self.active_trade_info: return False, "Trade active"
        if time.time() < self.cooldown_until:
            return False, f"Cooldown {int(self.cooldown_until - time.time())}s"
        if time.time() < self.pause_until: return False, "Paused"
        eq_ok, eq_msg, _ = self._equity_ok()
        if not eq_ok: return False, eq_msg
        if self.martingale_halt: return False, f"Martingale {MARTINGALE_MAX_STEPS} steps done"
        if self.total_profit_today >= DAILY_PROFIT_TARGET:
            return False, f"Daily target +${DAILY_PROFIT_TARGET:.2f} reached"
        if self.total_profit_today <= DAILY_LOSS_LIMIT:
            return False, f"Daily loss limit ${DAILY_LOSS_LIMIT:.2f} hit"
        if self.profit_lock_active and self.total_profit_today <= PROFIT_LOCK_FLOOR:
            return False, f"Profit lock protecting +${PROFIT_LOCK_FLOOR:.2f}"
        if self.consecutive_losses >= MAX_CONSEC_LOSSES:
            return False, f"Stopped: {MAX_CONSEC_LOSSES} consecutive losses"
        if self.trades_today >= MAX_TRADES_PER_DAY:
            return False, f"Max {MAX_TRADES_PER_DAY} trades today"
        return True, "OK"

    def stats_30d(self):
        cutoff = time.time() - (STATS_DAYS * 24 * 3600)
        by_mkt = {}; by_sess = {}
        for t in self.trade_history:
            if t.get("time", 0) < cutoff: continue
            m = t.get("market", "?"); s = t.get("session", "?")
            w = 1 if t.get("result") == "WIN" else 0
            for d, k in [(by_mkt, m), (by_sess, s)]:
                if k not in d: d[k] = {"trades": 0, "wins": 0}
                d[k]["trades"] += 1; d[k]["wins"] += w
        def wr(v): return round(100 * v["wins"] / v["trades"], 1) if v["trades"] > 0 else 0.0
        return by_mkt, by_sess, wr

    async def connect(self):
        try:
            token = DEMO_TOKEN if self.account_type == "DEMO" else REAL_TOKEN
            if self.api:
                try: await self.api.disconnect()
                except: pass
            self.api = DerivAPI(app_id=APP_ID)
            await self.api.authorize(token)
            await self.fetch_balance()
            try:
                bal_val = float(str(self.balance).split()[0])
                if self.starting_balance == 0.0: self.starting_balance = bal_val
            except: pass
            logger.info(f"Connected — {self.account_type} | {self.balance}")
            return True
        except Exception as e:
            logger.error(f"Connect error: {e}")
            self.api = None
            return False

    async def fetch_balance(self):
        if not self.api: return
        try:
            bal = await self.safe_deriv_call("balance", {"balance": 1}, retries=4)
            self.balance = "{:.2f} {}".format(float(bal["balance"]["balance"]), bal["balance"]["currency"])
        except: pass

    # start_scanning() removed — scanning controlled by Telegram buttons

    async def safe_deriv_call(self, method, params, retries=3):
        for attempt in range(retries):
            try:
                fn = getattr(self.api, method)
                return await asyncio.wait_for(fn(params), timeout=15.0)
            except Exception as e:
                if attempt < retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise

    async def safe_send_tg(self, text, parse_mode=None):
        if not self.app: return
        try:
            for uid in getattr(self, "_known_chat_ids", []):
                try:
                    kwargs = {"chat_id": uid, "text": text[:4000]}
                    if parse_mode: kwargs["parse_mode"] = parse_mode
                    await self.app.bot.send_message(**kwargs)
                    return
                except: pass
        except Exception as e:
            logger.warning(f"TG send failed: {e}")

    async def fetch_candles_with_timeout(self, symbol, tf_sec, count):
        try:
            res = await asyncio.wait_for(
                self.safe_deriv_call("ticks_history", {
                    "ticks_history": symbol, "style": "candles",
                    "granularity": tf_sec, "count": count, "end": "latest"
                }), timeout=15.0)
            return build_candles_from_deriv(res.get("candles", []))
        except Exception as e:
            logger.warning(f"Candles failed {symbol} tf={tf_sec}: {e}")
            return []

    async def execute_trade(self, side, symbol, **kwargs):
        _, _, stake_mult = self._equity_ok()
        base_payout = money2(max(float(MIN_PAYOUT),
            money2(max(0.01, float(PAYOUT_TARGET) * (float(MARTINGALE_MULT) ** int(self.martingale_step))))))
        try:
            prop = await self.safe_deriv_call("proposal", {
                "proposal": 1, "amount": base_payout, "basis": "payout",
                "contract_type": side, "currency": "USD",
                "duration": int(EXPIRY_MIN), "duration_unit": "m",
                "symbol": symbol
            }, retries=6)
            if "error" in prop:
                await self.safe_send_tg(f"⚠️ Proposal error {symbol}: {prop['error'].get('message','?')}")
                self.cooldown_until = time.time() + COOLDOWN_SEC; return
            ask_price = float(prop.get("proposal", {}).get("ask_price", 0))
            proposal_id = prop.get("proposal", {}).get("id")
            if ask_price > float(MAX_STAKE_ALLOWED):
                await self.safe_send_tg(f"⛔️ Skipped: stake=${ask_price:.2f} > max ${MAX_STAKE_ALLOWED:.2f}")
                self.cooldown_until = time.time() + COOLDOWN_SEC; return
            buy = await self.safe_deriv_call("buy", {"buy": proposal_id, "price": ask_price}, retries=1)
            if "error" in buy:
                await self.safe_send_tg(f"⚠️ Buy error {symbol}: {buy['error'].get('message','?')}")
                self.cooldown_until = time.time() + COOLDOWN_SEC; return
            contract_id = buy.get("buy", {}).get("contract_id")
            self.active_trade_info = contract_id
            self.active_market = symbol
            self.trade_start_time = time.time()
            pair = symbol.replace("frx", "")
            icon = "📈" if side == "CALL" else "📉"
            await self.safe_send_tg(
                f"{icon} STRUCTURE BREAK — {side}\n"
                f"Pair: {pair}\n"
                f"Stake: ${ask_price:.2f} → Payout: ${base_payout:.2f}\n"
                f"Step: {self.martingale_step}/{MARTINGALE_MAX_STEPS}\n"
                f"Expiry: {EXPIRY_MIN}m\n"
                f"Session: {session_bucket(time.time())}\n"
                f"RSI: {kwargs.get('rsi','?')} | Body: {kwargs.get('body','?')}\n"
                f"Structure level: {kwargs.get('level','?')}\n"
                f"EMA50: {kwargs.get('ema50','?')}"
            )
            # wait for result
            await asyncio.sleep(EXPIRY_MIN * 60 + 10)
            await self._check_result(contract_id, symbol, side, ask_price, base_payout, kwargs)
        except Exception as e:
            logger.error(f"Trade execution error: {e}")
            self.active_trade_info = None
            self.cooldown_until = time.time() + COOLDOWN_SEC

    async def _check_result(self, contract_id, symbol, side, stake, payout, kwargs):
        try:
            res = await self.safe_deriv_call("profit_table", {
                "profit_table": 1, "description": 1, "limit": 5
            }, retries=3)
            contracts = res.get("profit_table", {}).get("transactions", [])
            profit = None
            for c in contracts:
                if c.get("contract_id") == contract_id:
                    profit = float(c.get("sell_price", 0)) - float(c.get("buy_price", 0))
                    break
            if profit is None:
                logger.warning(f"Contract {contract_id} not found in profit_table — marking UNKNOWN")
                await self.safe_send_tg(f"⚠️ Could not verify result for contract {contract_id}. Check manually.")
                return
            won = profit > 0
            self.total_profit_today = round(self.total_profit_today + profit, 2)
            self.trades_today += 1
            self.market_trades_today[symbol] = self.market_trades_today.get(symbol, 0) + 1

            if won:
                self.consecutive_losses = 0
                self.martingale_step = 0
                result_str = "WIN"
                icon = "✅"
                if self.total_profit_today >= PROFIT_LOCK_TRIGGER:
                    self.profit_lock_active = True
            else:
                self.consecutive_losses += 1
                self.total_losses_today += 1
                self.max_loss_streak_today = max(self.max_loss_streak_today, self.consecutive_losses)
                self.market_losses_today[symbol] = self.market_losses_today.get(symbol, 0) + 1
                result_str = "LOSS"
                icon = "❌"
                if self.martingale_step < MARTINGALE_MAX_STEPS:
                    self.martingale_step += 1
                else:
                    self.martingale_halt = True
                if self.market_losses_today.get(symbol, 0) >= MAX_LOSSES_PER_MARKET:
                    self.market_blocked[symbol] = True
                if self.consecutive_losses >= 2:
                    self.market_pause_until[symbol] = time.time() + CONSEC_LOSS_PAUSE_SEC

            pair = symbol.replace("frx", "")
            self._save_trade({
                "time": time.time(), "market": pair, "side": side,
                "stake": stake, "payout": payout, "profit": profit,
                "result": result_str, "session": session_bucket(time.time()),
                "rsi": kwargs.get("rsi"), "body": kwargs.get("body"),
            })

            await self.safe_send_tg(
                f"{icon} {result_str} — {pair} {side}\n"
                f"P/L: {profit:+.2f} | Today: {self.total_profit_today:+.2f}\n"
                f"Streak: {self.consecutive_losses} | Step: {self.martingale_step}/{MARTINGALE_MAX_STEPS}\n"
                f"Balance: {self.balance}"
            )
        except Exception as e:
            logger.error(f"Result check error: {e}")
        finally:
            self.active_trade_info = None
            self.active_market = None
            self.cooldown_until = time.time() + COOLDOWN_SEC

    async def scan_market(self, symbol):
        while self.is_scanning:
            try:
                await asyncio.sleep(SCAN_INTERVAL_SEC)
                self._reset_daily_if_needed()
                if self.active_trade_info:
                    continue

                # Weekend/session gate
                forex_open, forex_reason = is_forex_market_open()
                if not forex_open:
                    self.market_debug[symbol] = {"time": time.time(), "why": [forex_reason]}
                    continue

                ok_gate, gate = self.can_auto_trade()
                mkt_ok, mkt_msg = self._is_market_available(symbol)

                # ── FETCH M1 CANDLES ONLY ────────────────────────────
                candles_m1 = await self.fetch_candles_with_timeout(symbol, TF_M1_SEC, CANDLES_M1)

                if len(candles_m1) < EMA_PERIOD + SWING_LOOKBACK + 10:
                    self.market_debug[symbol] = {"time": time.time(), "gate": gate,
                        "why": [f"Warming up M1:{len(candles_m1)}"]}
                    continue

                # Confirmed closed candles only
                m1_confirmed = candles_m1[:-1] if len(candles_m1) > 1 else []

                if len(m1_confirmed) < EMA_PERIOD + SWING_LOOKBACK + 5:
                    continue

                # ── M1 OHLC ──────────────────────────────────────────
                m1_highs  = [c["high"]  for c in m1_confirmed]
                m1_lows   = [c["low"]   for c in m1_confirmed]
                m1_opens  = [c["open"]  for c in m1_confirmed]
                m1_closes = [c["close"] for c in m1_confirmed]

                # ── ALL INDICATORS ON M1 ──────────────────────────────
                m1_atrs    = calculate_atr(m1_highs, m1_lows, m1_closes, ATR_PERIOD)
                m1_atr     = float(m1_atrs[-1]) if m1_atrs else None
                m1_atr_sma = float(np.mean([x for x in m1_atrs[-ATR_SMA_PERIOD-1:-1] if x])) if m1_atrs else None

                m1_ema50_series = calculate_ema_series(m1_closes, EMA_PERIOD)
                m1_ema50        = m1_ema50_series[-1] if m1_ema50_series else None

                m1_rsi_now  = calculate_rsi(m1_closes, RSI_PERIOD)
                m1_rsi_prev = calculate_rsi(m1_closes[:-1], RSI_PERIOD)

                if any(v is None for v in [m1_ema50, m1_atr, m1_atr_sma,
                                           m1_rsi_now, m1_rsi_prev]):
                    self.market_debug[symbol] = {"time": time.time(), "gate": gate,
                        "why": ["Indicators warming up"]}
                    continue

                # ── STRUCTURE LEVELS (M1) ─────────────────────────────
                last_swing_high, last_swing_low = get_structure_levels(
                    m1_highs, m1_lows, SWING_LOOKBACK)

                if last_swing_high is None or last_swing_low is None:
                    self.market_debug[symbol] = {"time": time.time(), "gate": gate,
                        "why": ["Not enough M1 swing points yet"]}
                    continue

                # Current M1 candle (last confirmed)
                cur_open  = m1_opens[-1]
                cur_close = m1_closes[-1]
                cur_high  = m1_highs[-1]
                cur_low   = m1_lows[-1]

                # ── VOLATILITY (M1) ───────────────────────────────────
                vol_ok = m1_atr > m1_atr_sma

                # ── BODY RATIO (M1 break candle) ─────────────────────
                candle_body = body_ratio(cur_open, cur_close, cur_high, cur_low)
                body_ok = candle_body >= BODY_RATIO_MIN

                # ── EMA50 CROSS WITHIN 3 M1 CANDLES ──────────────────
                EMA_CROSS_WINDOW = 3
                ema50_bull_cross = False
                ema50_bear_cross = False
                ema50_cross_candle = None

                for lookback_i in range(min(EMA_CROSS_WINDOW + 1, len(m1_closes) - 1)):
                    idx_cur  = -(1 + lookback_i)
                    idx_prev = -(2 + lookback_i)
                    if abs(idx_prev) > len(m1_closes) or abs(idx_prev) > len(m1_ema50_series): break
                    c_cur  = m1_closes[idx_cur]
                    c_prev = m1_closes[idx_prev]
                    e_cur  = m1_ema50_series[idx_cur]
                    e_prev = m1_ema50_series[idx_prev]
                    if e_cur is None or e_prev is None: break
                    if c_prev < e_prev and c_cur > e_cur:
                        ema50_bull_cross = True
                        ema50_cross_candle = lookback_i
                        break
                    if c_prev > e_prev and c_cur < e_cur:
                        ema50_bear_cross = True
                        ema50_cross_candle = lookback_i
                        break

                # ── STRUCTURE BREAK (M1) ──────────────────────────────
                bullish_break = (
                    cur_close > last_swing_high and
                    cur_open  < last_swing_high and
                    ema50_bull_cross
                )
                bearish_break = (
                    cur_close < last_swing_low and
                    cur_open  > last_swing_low and
                    ema50_bear_cross
                )

                # ── RSI GENUINE CROSS (M1) ────────────────────────────
                rsi_bull_cross = m1_rsi_prev < 50 and m1_rsi_now >= 50
                rsi_bear_cross = m1_rsi_prev > 50 and m1_rsi_now <= 50

                # ── NEXT M1 CANDLE CONFIRMATION ───────────────────────
                m1_last = m1_confirmed[-1]
                m1_bullish = m1_last["close"] > m1_last["open"]
                m1_bearish = m1_last["close"] < m1_last["open"]

                # ── SIGNAL ────────────────────────────────────────────
                signal = None
                reason = "Scanning..."

                if not vol_ok:
                    reason = f"Volatility low — M1 ATR({m1_atr:.5f}) < avg({m1_atr_sma:.5f})"
                elif bullish_break:
                    if not body_ok:
                        reason = f"Weak candle — body {candle_body:.2f} < {BODY_RATIO_MIN}"
                    elif not rsi_bull_cross:
                        reason = f"RSI no cross — prev {m1_rsi_prev:.1f} now {m1_rsi_now:.1f} (need cross above 50)"
                    elif not m1_bullish:
                        reason = "M1 candle not bullish — waiting"
                    else:
                        signal = "CALL"
                        reason = f"Bullish break above {last_swing_high:.5f} | EMA50 cross {ema50_cross_candle} M1 ago | RSI cross | confirmed"
                elif bearish_break:
                    if not body_ok:
                        reason = f"Weak candle — body {candle_body:.2f} < {BODY_RATIO_MIN}"
                    elif not rsi_bear_cross:
                        reason = f"RSI no cross — prev {m1_rsi_prev:.1f} now {m1_rsi_now:.1f} (need cross below 50)"
                    elif not m1_bearish:
                        reason = "M1 candle not bearish — waiting"
                    else:
                        signal = "PUT"
                        reason = f"Bearish break below {last_swing_low:.5f} | EMA50 cross {ema50_cross_candle} M1 ago | RSI cross | confirmed"
                else:
                    reason = f"No break — watching High:{last_swing_high:.5f} Low:{last_swing_low:.5f}"

                # ── DEBUG ─────────────────────────────────────────────
                self.market_debug[symbol] = {
                    "time": time.time(), "gate": gate, "mkt_msg": mkt_msg,
                    "last_m5": m1_confirmed[-1]["epoch"], "signal": signal,
                    "m1_ema50": round(m1_ema50, 5) if m1_ema50 else None,
                    "swing_high": round(last_swing_high, 5),
                    "swing_low": round(last_swing_low, 5),
                    "vol_ok": vol_ok,
                    "ema50_bull_cross": ema50_bull_cross,
                    "ema50_bear_cross": ema50_bear_cross,
                    "ema50_cross_candle": ema50_cross_candle,
                    "body": round(candle_body, 2), "body_ok": body_ok,
                    "m1_rsi": round(m1_rsi_now, 1),
                    "m1_rsi_prev": round(m1_rsi_prev, 1),
                    "rsi_bull_cross": rsi_bull_cross,
                    "rsi_bear_cross": rsi_bear_cross,
                    "bullish_break": bullish_break, "bearish_break": bearish_break,
                    "m1_bullish": m1_bullish, "m1_bearish": m1_bearish,
                    "mkt_losses": self.market_losses_today.get(symbol, 0),
                    "mkt_trades": self.market_trades_today.get(symbol, 0),
                    "why": [reason]
                }

                if not ok_gate or not mkt_ok: continue
                if signal is None: continue

                if signal == "CALL":
                    await self.execute_trade("CALL", symbol,
                        rsi=round(m1_rsi_now, 1), body=round(candle_body, 2),
                        level=round(last_swing_high, 5), ema50=round(m1_ema50, 5))
                elif signal == "PUT":
                    await self.execute_trade("PUT", symbol,
                        rsi=round(m1_rsi_now, 1), body=round(candle_body, 2),
                        level=round(last_swing_low, 5), ema50=round(m1_ema50, 5))

            except Exception as e:
                logger.error(f"Scan error {symbol}: {e}")
                await asyncio.sleep(10)

    async def run(self):
        """Simple keepalive — connect and scan controlled by Telegram buttons"""
        logger.info("Bot started — press DEMO or LIVE in Telegram to connect, then START to scan")
        while True:
            await asyncio.sleep(60)


bot_logic = StructureBreakBot()

# ============================================================
# TELEGRAM UI
# ============================================================

def main_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("▶️ START", callback_data="START_SCAN"),
         InlineKeyboardButton("⏹️ STOP", callback_data="STOP_SCAN")],
        [InlineKeyboardButton("📊 STATUS", callback_data="STATUS"),
         InlineKeyboardButton("🔄 REFRESH", callback_data="STATUS")],
        [InlineKeyboardButton("🧪 TEST BUY", callback_data="TEST_BUY")],
        [InlineKeyboardButton("🧪 DEMO", callback_data="SET_DEMO"),
         InlineKeyboardButton("💰 LIVE", callback_data="SET_REAL")],
        [InlineKeyboardButton("⏸ PAUSE", callback_data="PAUSE"),
         InlineKeyboardButton("▶️ RESUME", callback_data="RESUME"),
         InlineKeyboardButton("🔓 UNBLOCK", callback_data="UNBLOCK")],
    ])

def format_market_detail(sym, d):
    if not d: return f"\U0001f4cd {sym.replace('frx','')}\n\u23f3 No scan data yet"
    age = int(time.time() - d.get("time", time.time()))
    signal = d.get("signal") or "\u2014"
    why = d.get("why", [])
    vol_ok = d.get("vol_ok", False)
    body = d.get("body", "\u2014"); body_ok = d.get("body_ok", False)
    m1_rsi = d.get("m1_rsi", "\u2014"); m1_rsi_prev = d.get("m1_rsi_prev", "\u2014")
    rsi_bull = d.get("rsi_bull_cross", False); rsi_bear = d.get("rsi_bear_cross", False)
    swing_high = d.get("swing_high", "\u2014"); swing_low = d.get("swing_low", "\u2014")
    ema50 = d.get("m1_ema50", "\u2014")
    bullish = d.get("bullish_break", False); bearish = d.get("bearish_break", False)
    ema50_bull = d.get("ema50_bull_cross", False); ema50_bear = d.get("ema50_bear_cross", False)
    m1_bull = d.get("m1_bullish", False); m1_bear = d.get("m1_bearish", False)
    mkt_losses = d.get("mkt_losses", 0); mkt_trades = d.get("mkt_trades", 0)
    last_m5 = d.get("last_m5", 0)
    vol_str = "OK" if vol_ok else "Low"
    body_str = "OK" if body_ok else "weak"
    ema50_str = "UP OK" if ema50_bull else ("DOWN OK" if ema50_bear else "-- No cross")
    rsi_dir = "UP" if rsi_bull else "DOWN"
    rsi_str = ("crossed " + rsi_dir + " OK") if (rsi_bull or rsi_bear) else ("-- " + str(m1_rsi_prev) + "->" + str(m1_rsi))
    break_str = "BULLISH BREAK" if bullish else ("BEARISH BREAK" if bearish else "-- No break")
    m1_str = "Bullish OK" if m1_bull else ("Bearish OK" if m1_bear else "-- Doji")
    return (
        f"\U0001f4cd {sym.replace('frx','')} ({age}s ago)\n"
        f"Market: {d.get('mkt_msg','OK')} | {mkt_trades}/{MAX_TRADES_PER_MARKET} | {mkt_losses}/{MAX_LOSSES_PER_MARKET} losses\n"
        f"Last M5: {fmt_time_hhmmss(last_m5)}\n"
        f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        f"Structure: High {swing_high} | Low {swing_low}\n"
        f"M5 EMA50: {ema50} | Cross: {ema50_str}\n"
        f"M5 Volatility: {vol_str}\n"
        f"M5 Body: {body} {body_str}\n"
        f"M1 RSI cross: {rsi_str}\n"
        f"M1 candle: {m1_str}\n"
        f"{break_str}\n"
        f"Signal: {signal}\n"
        f"Why: {why[0] if why else '--'}\n"
    )

async def _safe_answer(q, text=None, show_alert=False):
    try: await q.answer(text=text, show_alert=show_alert)
    except Exception as e: logger.warning(f"Callback answer: {e}")

async def _safe_edit(q, text, reply_markup=None):
    try:
        kwargs = {"text": text[:4000]}
        if reply_markup: kwargs["reply_markup"] = reply_markup
        await q.edit_message_text(**kwargs)
    except Exception as e: logger.warning(f"Edit failed: {e}")

async def btn_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    if not q:
        return

    # Register chat ID
    if not hasattr(bot_logic, "_known_chat_ids"):
        bot_logic._known_chat_ids = set()
    bot_logic._known_chat_ids.add(q.message.chat_id)

    # Always answer immediately — stops button spinner
    try:
        await q.answer()
    except Exception:
        pass

    # Show working indicator
    try:
        await q.edit_message_text("⏳ Working...", reply_markup=main_keyboard())
    except Exception:
        pass

    try:
        if q.data == "SET_DEMO":
            bot_logic.account_type = "DEMO"
            ok = await bot_logic.connect()
            await _safe_edit(q, "✅ Connected to DEMO" if ok else "❌ DEMO connection failed", reply_markup=main_keyboard())

        elif q.data == "SET_REAL":
            bot_logic.account_type = "LIVE"
            ok = await bot_logic.connect()
            await _safe_edit(q, "✅ LIVE CONNECTED" if ok else "❌ LIVE connection failed", reply_markup=main_keyboard())

        elif q.data == "START_SCAN":
            if not bot_logic.api:
                await _safe_edit(q, "❌ Connect first — press DEMO or LIVE.", reply_markup=main_keyboard())
                return
            # Stop any existing tasks then restart fresh
            bot_logic.is_scanning = False
            await asyncio.sleep(0.3)
            bot_logic.is_scanning = True
            for sym in MARKETS:
                asyncio.create_task(bot_logic.scan_market(sym))
            await _safe_edit(q,
                "🔍 SCANNER ACTIVE\n"
                "✅ M5 structure | M1 EMA50 cross + RSI + confirm\n"
                "📡 EURUSD GBPUSD USDJPY AUDUSD GBPJPY",
                reply_markup=main_keyboard())

        elif q.data == "STOP_SCAN":
            bot_logic.is_scanning = False
            await _safe_edit(q, "⏹️ Scanner stopped. Press START to begin again.", reply_markup=main_keyboard())

        elif q.data == "TEST_BUY":
            if not bot_logic.api:
                await _safe_edit(q, "❌ Connect first.", reply_markup=main_keyboard())
                return
            test_symbol = MARKETS[0]
            asyncio.create_task(bot_logic.execute_trade("CALL", test_symbol,
                rsi=55.0, body=0.45, level=0.0, ema50=0.0))
            await _safe_edit(q, f"🧪 Test CALL on {test_symbol.replace('frx','')}.", reply_markup=main_keyboard())

        elif q.data == "PAUSE":
            bot_logic.pause_until = time.time() + 86400
            await _safe_edit(q, "⏸ Bot paused for 24 hours.", reply_markup=main_keyboard())

        elif q.data == "RESUME":
            bot_logic.pause_until = 0
            await _safe_edit(q, "▶️ Bot resumed.", reply_markup=main_keyboard())

        elif q.data == "UNBLOCK":
            for m in MARKETS:
                bot_logic.market_blocked[m] = False
                bot_logic.market_pause_until[m] = 0
                bot_logic.market_losses_today[m] = 0
            await _safe_edit(q, "🔓 All pairs unblocked.", reply_markup=main_keyboard())

        elif q.data == "STATUS":
            now = time.time()
            if now < bot_logic.status_cooldown_until:
                await _safe_edit(q, f"⏳ Cooldown: {int(bot_logic.status_cooldown_until - now)}s", reply_markup=main_keyboard())
                return
            bot_logic.status_cooldown_until = now + STATUS_REFRESH_COOLDOWN_SEC
            try:
                await asyncio.wait_for(bot_logic.fetch_balance(), timeout=3.0)
            except Exception:
                pass
            now_time = datetime.now(ZoneInfo("Africa/Lagos")).strftime("%Y-%m-%d %H:%M:%S")
            _, gate = bot_logic.can_auto_trade()
            trade_status = "No Active Trade"
            if bot_logic.active_trade_info:
                rem = max(0, int(EXPIRY_MIN * 60) - int(time.time() - bot_logic.trade_start_time))
                trade_status = f"🚀 Active Trade ({bot_logic.active_market.replace('frx','')})\n⏳ Left: ~{rem}s"
            next_payout = money2(float(PAYOUT_TARGET) * (float(MARTINGALE_MULT) ** int(bot_logic.martingale_step)))
            by_mkt, by_sess, wr = bot_logic.stats_30d()
            def fmt_stats(title, items):
                rows = [(k, wr(v), v["trades"], v["wins"]) for k, v in items.items()]
                rows.sort(key=lambda x: (x[1], x[2]), reverse=True)
                lines_ = [f"{title} (last {STATS_DAYS}d):"]
                if not rows:
                    lines_.append("— No trades yet")
                    return "\n".join(lines_)
                for k, wrr, t, w in rows:
                    lines_.append(f"- {k}: {wrr:.1f}% ({w}/{t})")
                return "\n".join(lines_)
            stats_block = "📈 PERFORMANCE\n" + fmt_stats("Pairs", by_mkt) + "\n" + fmt_stats("Sessions", by_sess) + "\n"
            mkt_lines = ["🛡 Pair Status:"]
            for m in MARKETS:
                ml = bot_logic.market_losses_today.get(m, 0)
                mt = bot_logic.market_trades_today.get(m, 0)
                mb = bot_logic.market_blocked.get(m, False)
                mp = bot_logic.market_pause_until.get(m, 0)
                status = "🚫 BLOCKED" if mb else ("⏸ PAUSED" if time.time() < mp else "✅")
                mkt_lines.append(f"{status} {m.replace('frx','')}: {mt}/{MAX_TRADES_PER_MARKET} | {ml}/{MAX_LOSSES_PER_MARKET} losses")
            mkt_block = "\n".join(mkt_lines) + "\n"
            _, _, stake_mult = bot_logic._equity_ok()
            eq_ratio = bot_logic._get_current_balance_float() / bot_logic.starting_balance if bot_logic.starting_balance > 0 else 1.0
            pause_line = "Paused" if time.time() < bot_logic.pause_until else ""
            pause_line_nl = ("⏸ Paused\n") if time.time() < bot_logic.pause_until else ""
            scan_status = "ACTIVE" if bot_logic.is_scanning else "OFFLINE"
            lock_str = ("ON +$" + f"{PROFIT_LOCK_FLOOR:.2f}") if bot_logic.profit_lock_active else "OFF"
            header = (
                f"🕒 {now_time}\n"
                f"🤖 {scan_status} ({bot_logic.account_type})\n"
                f"{pause_line_nl}"
                f"🎁 Next payout: ${next_payout:.2f} | Step: {bot_logic.martingale_step}/{MARTINGALE_MAX_STEPS}\n"
                f"🧯 Max stake: ${MAX_STAKE_ALLOWED:.2f} | Mult: {stake_mult:.1f}x\n"
                f"💰 Equity: {eq_ratio:.0%} | 🔒 Lock: {lock_str}\n"
                f"🎯 Target: +${DAILY_PROFIT_TARGET:.2f} | Limit: ${DAILY_LOSS_LIMIT:.2f}\n"
                f"📡 Pairs: EURUSD GBPUSD USDJPY AUDUSD GBPJPY\n"
                f"🧭 M1 only | Structure + EMA50 cross + RSI + confirm | {EXPIRY_MIN}m expiry\n"
                f"━━━━━━━━━━━━━━━\n{trade_status}\n━━━━━━━━━━━━━━━\n"
                f"{stats_block}{mkt_block}"
                f"💵 PnL: {bot_logic.total_profit_today:+.2f} | Trades: {bot_logic.trades_today}/{MAX_TRADES_PER_DAY}\n"
                f"📉 Streak: {bot_logic.consecutive_losses}/{MAX_CONSEC_LOSSES} | Losses: {bot_logic.total_losses_today}\n"
                f"🚦 Gate: {gate}\n"
                f"💰 Balance: {bot_logic.balance}\n"
                f"\n/pause /resume /unblock /stats"
            )
            details = "\n\n📌 LIVE SCAN\n\n" + "\n\n".join([
                format_market_detail(sym, bot_logic.market_debug.get(sym, {})) for sym in MARKETS
            ])
            await _safe_edit(q, header + details, reply_markup=main_keyboard())

    except Exception as e:
        logger.error(f"btn_handler error: {e}")
        try:
            await q.answer("⚠️ Error occurred. Try again.")
        except Exception:
            pass

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    bot_logic._known_chat_ids = getattr(bot_logic, "_known_chat_ids", set())
    bot_logic._known_chat_ids.add(update.message.chat_id)
    await update.message.reply_text(
        "💎 Deriv Forex Structure Break Bot\n"
        f"🧭 Strategy: M5 Structure Break + M1 Entry Confirmation\n"
        f"🏗 M1 only — swing levels, EMA50 cross, RSI, body all on 1min candles\n"
        f"📊 ATR volatility | RSI momentum | Body ratio | M1 confirmation\n"
        f"🌍 Pairs: EURUSD | GBPUSD | USDJPY | AUDUSD | GBPJPY\n"
        f"📅 Trades Mon-Fri all sessions (data collection mode)\n"
        f"🎲 Martingale: {MARTINGALE_MAX_STEPS} steps × {MARTINGALE_MULT}x\n"
        f"⏱ Expiry: {EXPIRY_MIN}m | Cooldown: {COOLDOWN_SEC//60}m\n"
        f"🛡 Max {MAX_TRADES_PER_MARKET} trades/pair | Stop after {MAX_LOSSES_PER_MARKET} losses/pair\n",
        reply_markup=main_keyboard()
    )

async def cmd_pause(update: Update, context: ContextTypes.DEFAULT_TYPE):
    bot_logic.pause_until = time.time() + 86400
    await update.message.reply_text("⏸ Bot paused for 24 hours.", reply_markup=main_keyboard())

async def cmd_resume(update: Update, context: ContextTypes.DEFAULT_TYPE):
    bot_logic.pause_until = 0
    await update.message.reply_text("▶️ Bot resumed.", reply_markup=main_keyboard())

async def cmd_unblock(update: Update, context: ContextTypes.DEFAULT_TYPE):
    for m in MARKETS:
        bot_logic.market_blocked[m] = False
        bot_logic.market_pause_until[m] = 0
        bot_logic.market_losses_today[m] = 0
    await update.message.reply_text("🔓 All pairs unblocked.", reply_markup=main_keyboard())

async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    by_mkt, by_sess, wr = bot_logic.stats_30d()
    lines = [f"📈 STATS (last {STATS_DAYS}d)\n"]
    for k, v in sorted(by_mkt.items(), key=lambda x: -x[1]["trades"]):
        wins = v["wins"]; trades = v["trades"]
        lines.append(f"  {k}: {wr(v):.1f}% ({wins}/{trades})")
    lines.append("")
    for k, v in sorted(by_sess.items(), key=lambda x: -x[1]["trades"]):
        wins = v["wins"]; trades = v["trades"]
        lines.append(f"  {k}: {wr(v):.1f}% ({wins}/{trades})")
    lines.append(f"\nToday: {bot_logic.total_profit_today:+.2f} | Trades: {bot_logic.trades_today}")
    await update.message.reply_text("\n".join(lines), reply_markup=main_keyboard())

if __name__ == "__main__":
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    bot_logic.app = app
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CallbackQueryHandler(btn_handler))
    app.add_handler(CommandHandler("pause", cmd_pause))
    app.add_handler(CommandHandler("resume", cmd_resume))
    app.add_handler(CommandHandler("unblock", cmd_unblock))
    app.add_handler(CommandHandler("stats", cmd_stats))

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.create_task(bot_logic.run())
    app.run_polling(close_loop=False)
