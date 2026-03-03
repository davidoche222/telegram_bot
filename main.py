# ⚠️ SECURITY NOTE: Do NOT post tokens publicly. Paste them only on your local machine.

import asyncio, logging, random, time, json, os, math
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import numpy as np
from deriv_api import DerivAPI
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

# ===== CONFIG =====
DEMO_TOKEN = "tIrfitLjqeBxCOM"
REAL_TOKEN = "ZkOFWOlPtwnjqTS"
APP_ID = 1089
MARKETS = ["R_50", "R_75", "R_100", "R_10"]
COOLDOWN_SEC = 3
MAX_TRADES_PER_DAY = 80
MAX_CONSEC_LOSSES = 5
TELEGRAM_TOKEN = "8589420556:AAHmB6YE9KIEu0tBIgWdd9baBDt0eDh5FY8"
TELEGRAM_CHAT_ID = "7634818949"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ===== STRATEGY =====
TF_SEC = 60
CANDLES_COUNT = 210
DURATION_MIN = 3
MA_PERIOD = 200
STD_PERIOD = 20
ATR_SHORT = 14
ATR_LONG = 50
ATR_EXPANSION_MULT = 1.3
CANDLE_COOLDOWN = 2

# ===== IMPROVEMENT 3: Dynamic Z threshold per market =====
Z_THRESHOLDS = {
    "R_10":  {"min": 2.0, "max": 3.0},
    "R_25":  {"min": 2.0, "max": 3.0},
    "R_50":  {"min": 2.2, "max": 3.2},
    "R_75":  {"min": 2.5, "max": 3.5},
    "R_100": {"min": 2.8, "max": 3.8},
}
DEFAULT_Z_THRESHOLD = {"min": 2.0, "max": 3.0}

# ===== IMPROVEMENT 4: Time-based filter (UTC hours) =====
ALLOWED_HOURS_UTC = (7, 20)  # Only trade 07:00 - 20:00 UTC

# ===== IMPROVEMENT 6: Adaptive expiry =====
EXPIRY_NORMAL_MIN = 3
EXPIRY_STRONG_MIN = 2
EXPIRY_STRONG_Z_THRESHOLD = 2.5

# ===== IMPROVEMENT 11: Profit lock =====
PROFIT_LOCK_TRIGGER = 5.0   # Activate lock once profit hits +$5
PROFIT_LOCK_FLOOR = 2.0     # Never let profit fall below +$2 after lock activates

# ===== IMPROVEMENT 13: Per-session loss limit =====
SESSION_LOSS_LIMIT = -3.0

# ===== IMPROVEMENT 17: Spike detection =====
SPIKE_MULTIPLIER = 3.0      # Skip if candle body > 3x average body

# ===== IMPROVEMENT 18: Consecutive Z confirmation =====
REQUIRE_CONSEC_Z = True     # Require Z beyond threshold for 2 consecutive candles

# ===== IMPROVEMENT 21: Performance-based parameter tuning =====
PERF_TUNE_LOOKBACK = 15     # Check last 15 trades
PERF_TUNE_MIN_WINRATE = 0.50
PERF_TUNE_Z_BOOST = 0.3     # Tighten Z by this amount if win rate drops

# ===== SESSION FILTER =====
ALLOWED_SESSIONS_UTC = None

# ===== DAILY TARGETS =====
DAILY_PROFIT_TARGET = 10.0
DAILY_LOSS_LIMIT = -8.0

# ===== SECTIONS =====
SECTIONS_PER_DAY = 1
SECTION_PROFIT_TARGET = 3.0
SECTION_LENGTH_SEC = int(24 * 60 * 60 / SECTIONS_PER_DAY)

# ===== PAYOUT / STAKE =====
PAYOUT_TARGET = 1
MIN_PAYOUT = 0.35
MAX_STAKE_ALLOWED = 10.00

# ===== IMPROVEMENT 10: Equity protection =====
EQUITY_HALF_STAKE_PCT = 0.80   # Halve stake if balance < 80% of starting balance
EQUITY_STOP_PCT = 0.60         # Stop if balance < 60% of starting balance

# ===== MARTINGALE =====
MARTINGALE_MULT = 2
MARTINGALE_MAX_STEPS = 4

# ===== MISC =====
TICKS_GLOBAL_MIN_INTERVAL = 0.35
RATE_LIMIT_BACKOFF_BASE = 20
STATUS_REFRESH_COOLDOWN_SEC = 10
STATS_DAYS = 30
TRADE_LOG_FILE = "trade_log.jsonl"
SESSION_BUCKETS = [("ASIA",0,6),("LONDON",7,11),("OVERLAP",12,15),("NEWYORK",16,20),("LATE_NY",21,23)]
MARKET_CONSEC_LOSS_BLOCK = 2


# ===== INDICATORS =====
def calculate_zscore(closes, ma_period=200, std_period=20):
    closes = np.array(closes, dtype=float)
    if len(closes) < ma_period + 1:
        return None, None
    ma_now   = np.mean(closes[-ma_period:])
    ma_prev  = np.mean(closes[-ma_period-1:-1])
    std_now  = np.std(closes[-std_period:], ddof=0)
    std_prev = np.std(closes[-std_period-1:-1], ddof=0)
    if std_now == 0 or std_prev == 0:
        return None, None
    return float((closes[-1] - ma_now) / std_now), float((closes[-2] - ma_prev) / std_prev)

def calculate_atr(highs, lows, closes, period=14):
    highs=np.array(highs,dtype=float); lows=np.array(lows,dtype=float); closes=np.array(closes,dtype=float)
    n=len(closes)
    if n < period+2: return np.array([])
    pc=np.roll(closes,1); pc[0]=closes[0]
    tr=np.maximum(highs-lows,np.maximum(np.abs(highs-pc),np.abs(lows-pc)))
    atr=np.full(n,np.nan,dtype=float)
    atr[period]=np.mean(tr[1:period+1])
    for i in range(period+1,n):
        atr[i]=(atr[i-1]*(period-1)+tr[i])/period
    return atr

def volatility_expanding(highs, lows, closes):
    a_s=calculate_atr(highs,lows,closes,ATR_SHORT)
    a_l=calculate_atr(highs,lows,closes,ATR_LONG)
    if len(a_s)<2 or len(a_l)<2: return True
    a14=a_s[-2]; a50=a_l[-2]
    if np.isnan(a14) or np.isnan(a50) or a50==0: return True
    return bool(a14 > ATR_EXPANSION_MULT * a50)

def ma200_slope(closes, ma_period=200, lookback=5):
    """Returns 'up', 'down', or 'flat' based on MA200 direction."""
    closes = np.array(closes, dtype=float)
    if len(closes) < ma_period + lookback:
        return "flat"
    ma_now  = np.mean(closes[-ma_period:])
    ma_back = np.mean(closes[-ma_period-lookback:-lookback])
    diff = ma_now - ma_back
    threshold = np.std(closes[-ma_period:], ddof=0) * 0.05
    if diff > threshold: return "up"
    if diff < -threshold: return "down"
    return "flat"

def is_spike_candle(candles, lookback=20):
    """Returns True if latest closed candle is a spike (body > SPIKE_MULTIPLIER * avg body)."""
    if len(candles) < lookback + 1:
        return False
    bodies = [abs(float(candles[i]["c"]) - float(candles[i]["o"])) for i in range(-lookback-1, -1)]
    avg_body = np.mean(bodies) if bodies else 0
    last_body = abs(float(candles[-2]["c"]) - float(candles[-2]["o"]))
    return avg_body > 0 and last_body > SPIKE_MULTIPLIER * avg_body

def candle_confirms_direction(candle, direction):
    """Checks if candle closes in the expected reversal direction."""
    c = float(candle["c"]); o = float(candle["o"])
    if direction == "CALL": return c > o   # green candle for buy
    if direction == "PUT":  return c < o   # red candle for sell
    return False

def build_candles_from_deriv(raw):
    return [{"t0":int(x.get("epoch",0)),"o":float(x.get("open",0)),"h":float(x.get("high",0)),"l":float(x.get("low",0)),"c":float(x.get("close",0))} for x in raw]

def fmt_time_hhmmss(epoch):
    try: return datetime.fromtimestamp(epoch,ZoneInfo("Africa/Lagos")).strftime("%H:%M:%S")
    except: return "—"

def fmt_hhmm(epoch):
    try: return datetime.fromtimestamp(epoch,ZoneInfo("Africa/Lagos")).strftime("%H:%M")
    except: return "—"

def money2(x):
    return math.ceil(float(x)*100.0)/100.0

def session_bucket(ts):
    h=datetime.fromtimestamp(ts,ZoneInfo("UTC")).hour
    for name,s,e in SESSION_BUCKETS:
        if s<=h<=e: return name
    return "UNKNOWN"

def is_allowed_hour():
    """IMPROVEMENT 4: Time-based filter."""
    h = datetime.utcnow().hour
    return ALLOWED_HOURS_UTC[0] <= h <= ALLOWED_HOURS_UTC[1]

def get_adaptive_expiry(z_abs):
    """IMPROVEMENT 6: Adaptive expiry based on Z strength."""
    if z_abs >= EXPIRY_STRONG_Z_THRESHOLD:
        return EXPIRY_STRONG_MIN
    return EXPIRY_NORMAL_MIN


# ===== BOT =====
class DerivSniperBot:
    def __init__(self):
        self.api=None; self.app=None; self.active_token=None; self.account_type="None"
        self.is_scanning=False; self.scanner_task=None; self.market_tasks={}
        self.active_trade_info=None; self.active_market="None"
        self.trade_start_time=0.0; self.active_trade_meta=None
        self.cooldown_until=0.0; self.trades_today=0; self.total_losses_today=0
        self.consecutive_losses=0; self.total_profit_today=0.0; self.balance="0.00"
        self.starting_balance=0.0  # IMPROVEMENT 10: equity protection
        self.max_loss_streak_today=0; self.hit_5_losses_today=False
        self.current_stake=0.0; self.martingale_step=0; self.martingale_halt=False
        self.section_profit=0.0; self.sections_won_today=0
        self.section_index=1; self.section_pause_until=0.0
        self.trade_lock=asyncio.Lock(); self._pending_buy=False
        self.market_debug={m:{} for m in MARKETS}
        self.last_processed_closed_t0={m:0 for m in MARKETS}
        self.candle_cooldown_counter={m:0 for m in MARKETS}
        self.prev_z={m:None for m in MARKETS}  # IMPROVEMENT 18: consecutive Z tracking

        # IMPROVEMENT 11: Profit lock
        self.profit_lock_active=False

        # IMPROVEMENT 13: Per-session loss tracking
        self.session_losses={}

        # IMPROVEMENT 15: Correlation filter — track last signal time per market
        self.last_signal_time={m:0.0 for m in MARKETS}

        self.tz=ZoneInfo("Africa/Lagos")
        self.current_day=datetime.now(self.tz).date()
        self.pause_until=0.0
        self._ticks_lock=asyncio.Lock(); self._last_ticks_ts=0.0
        self._next_poll_epoch={m:0.0 for m in MARKETS}
        self._rate_limit_strikes={m:0 for m in MARKETS}
        self.status_cooldown_until=0.0
        self.trade_log_path=os.path.abspath(TRADE_LOG_FILE)
        self.trade_records=[]; self._load_trade_log()
        self.market_consec_losses={m:0 for m in MARKETS}
        self.market_blocked={m:False for m in MARKETS}

    def _load_trade_log(self):
        self.trade_records=[]
        if not os.path.exists(self.trade_log_path): return
        try:
            with open(self.trade_log_path,"r",encoding="utf-8") as f:
                for line in f:
                    line=line.strip()
                    if not line: continue
                    try:
                        rec=json.loads(line)
                        if isinstance(rec,dict) and "t" in rec: self.trade_records.append(rec)
                    except: continue
            self._prune_trade_records()
        except Exception as e: logger.warning(f"Failed to load trade log: {e}")

    def _append_trade_log(self,rec):
        try:
            with open(self.trade_log_path,"a",encoding="utf-8") as f:
                f.write(json.dumps(rec,ensure_ascii=False)+"\n")
        except Exception as e: logger.warning(f"Failed to write trade log: {e}")

    def _prune_trade_records(self):
        cutoff=time.time()-(STATS_DAYS*24*3600)
        self.trade_records=[r for r in self.trade_records if float(r.get("t",0))>=cutoff]

    def record_trade_result(self,symbol,open_epoch,profit):
        sess=session_bucket(open_epoch); win=1 if profit>0 else 0
        rec={"t":float(open_epoch),"symbol":str(symbol),"session":str(sess),"win":int(win),"profit":float(profit)}
        self.trade_records.append(rec); self._append_trade_log(rec); self._prune_trade_records()

    def stats_30d(self):
        self._prune_trade_records()
        by_market={}; by_session={}
        for r in self.trade_records:
            sym=r.get("symbol","—"); sess=r.get("session","—"); win=int(r.get("win",0))
            by_market.setdefault(sym,{"wins":0,"losses":0,"trades":0})
            by_session.setdefault(sess,{"wins":0,"losses":0,"trades":0})
            by_market[sym]["trades"]+=1; by_session[sess]["trades"]+=1
            if win==1: by_market[sym]["wins"]+=1; by_session[sess]["wins"]+=1
            else: by_market[sym]["losses"]+=1; by_session[sess]["losses"]+=1
        def wr(d): t=d["trades"]; return (100.0*d["wins"]/t) if t>0 else 0.0
        return by_market,by_session,wr

    # IMPROVEMENT 21: Performance-based Z threshold tuning
    def get_tuned_z_threshold(self, symbol):
        thresh = Z_THRESHOLDS.get(symbol, DEFAULT_Z_THRESHOLD).copy()
        recent = [r for r in self.trade_records if r.get("symbol") == symbol][-PERF_TUNE_LOOKBACK:]
        if len(recent) >= PERF_TUNE_LOOKBACK:
            wins = sum(1 for r in recent if r.get("win") == 1)
            wr = wins / len(recent)
            if wr < PERF_TUNE_MIN_WINRATE:
                thresh["min"] += PERF_TUNE_Z_BOOST
                thresh["max"] += PERF_TUNE_Z_BOOST
                logger.info(f"⚠️ {symbol} win rate {wr:.0%} < {PERF_TUNE_MIN_WINRATE:.0%} — Z threshold tightened to {thresh['min']:.1f}")
        return thresh

    # IMPROVEMENT 14: Auto market rotation — best performing market
    def get_best_markets(self):
        by_mkt, _, wr = self.stats_30d()
        scored = []
        for m in MARKETS:
            if m in by_mkt and by_mkt[m]["trades"] >= 5:
                scored.append((m, wr(by_mkt[m])))
            else:
                scored.append((m, 50.0))  # default if no data
        scored.sort(key=lambda x: x[1], reverse=True)
        return [m for m, _ in scored]

    # IMPROVEMENT 16: Market warm-up check
    def market_recently_losing(self, symbol, min_losses=3, lookback=5):
        recent = [r for r in self.trade_records if r.get("symbol") == symbol][-lookback:]
        if len(recent) < lookback:
            return False
        losses = sum(1 for r in recent if r.get("win") == 0)
        return losses >= min_losses

    def _update_market_loss_state(self,symbol,profit):
        if profit<=0:
            self.market_consec_losses[symbol]=self.market_consec_losses.get(symbol,0)+1
            if self.market_consec_losses[symbol]>=MARKET_CONSEC_LOSS_BLOCK:
                if not self.market_blocked.get(symbol,False):
                    self.market_blocked[symbol]=True
                    logger.info(f"🚫 {symbol} BLOCKED after {self.market_consec_losses[symbol]} losses.")
        else:
            self.market_consec_losses[symbol]=0; self.market_blocked[symbol]=False
            for m in MARKETS:
                if m!=symbol and self.market_blocked.get(m,False):
                    self.market_blocked[m]=False; self.market_consec_losses[m]=0
                    logger.info(f"✅ {m} UNBLOCKED after win on {symbol}.")

    def _is_market_blocked(self,symbol):
        if self.market_blocked.get(symbol,False):
            return True,f"Market blocked: {self.market_consec_losses.get(symbol,0)} consecutive losses"
        return False,"OK"

    def _reset_market_loss_state(self):
        self.market_consec_losses={m:0 for m in MARKETS}
        self.market_blocked={m:False for m in MARKETS}

    @staticmethod
    def _is_gatewayish_error(msg):
        m=(msg or "").lower()
        return any(k in m for k in ["gateway","bad gateway","502","503","504","timeout","timed out","temporarily unavailable","connection","websocket","not connected","disconnect","internal server error","service unavailable"])

    @staticmethod
    def _is_rate_limit_error(msg):
        m=(msg or "").lower()
        return ("rate limit" in m) or ("reached the rate limit" in m) or ("too many requests" in m) or ("429" in m)

    async def safe_send_tg(self,text,retries=5):
        if not self.app: return
        for i in range(1,retries+1):
            try: await self.app.bot.send_message(TELEGRAM_CHAT_ID,text); return
            except Exception as e:
                if self._is_gatewayish_error(str(e)): await asyncio.sleep(0.8*i+random.random()*0.4)
                else: await asyncio.sleep(0.4*i)

    def _today_midnight_epoch(self):
        return datetime.now(self.tz).replace(hour=0,minute=0,second=0,microsecond=0).timestamp()

    def _get_section_index_for_epoch(self,epoch_ts):
        midnight=self._today_midnight_epoch()
        sec_into_day=max(0,int(epoch_ts-midnight))
        return int(min(SECTIONS_PER_DAY-1,sec_into_day//SECTION_LENGTH_SEC)+1)

    def _next_section_start_epoch(self,epoch_ts):
        midnight=self._today_midnight_epoch()
        sec_into_day=max(0,int(epoch_ts-midnight))
        idx0=min(SECTIONS_PER_DAY-1,sec_into_day//SECTION_LENGTH_SEC)
        if idx0+1>=SECTIONS_PER_DAY:
            return (datetime.fromtimestamp(midnight,self.tz)+timedelta(days=1)).replace(hour=0,minute=0,second=0,microsecond=0).timestamp()
        return float(midnight+(idx0+1)*SECTION_LENGTH_SEC)

    def _sync_section_if_needed(self):
        today=datetime.now(self.tz).date()
        if today!=self.current_day: return
        new_idx=self._get_section_index_for_epoch(time.time())
        if new_idx!=self.section_index:
            self.section_index=new_idx; self.section_profit=0.0; self.section_pause_until=0.0

    async def connect(self):
        try:
            if not self.active_token: return False
            self.api=DerivAPI(app_id=APP_ID)
            await self.api.authorize(self.active_token)
            await self.fetch_balance()
            # IMPROVEMENT 10: Record starting balance
            try:
                bal_val = float(self.balance.split()[0])
                if self.starting_balance == 0.0:
                    self.starting_balance = bal_val
            except: pass
            return True
        except Exception as e: logger.error(f"Connect error: {e}"); return False

    async def safe_reconnect(self):
        try:
            if self.api:
                try: await self.api.disconnect()
                except: pass
        except: pass
        self.api=None; return await self.connect()

    async def safe_deriv_call(self,fn_name,payload,retries=6):
        last_err=None
        for attempt in range(1,retries+1):
            try:
                if not self.api:
                    ok=await self.safe_reconnect()
                    if not ok: raise RuntimeError("Reconnect failed")
                return await getattr(self.api,fn_name)(payload)
            except Exception as e:
                last_err=e; msg=str(e)
                if self._is_gatewayish_error(msg): await self.safe_reconnect()
                if self._is_rate_limit_error(msg): await asyncio.sleep(min(20.0,2.5*attempt+random.random()))
                else: await asyncio.sleep(min(8.0,0.6*attempt+random.random()*0.5))
        raise last_err

    async def safe_ticks_history(self,payload,retries=4):
        async with self._ticks_lock:
            now=time.time(); gap=(self._last_ticks_ts+TICKS_GLOBAL_MIN_INTERVAL)-now
            if gap>0: await asyncio.sleep(gap)
            self._last_ticks_ts=time.time()
        return await self.safe_deriv_call("ticks_history",payload,retries=retries)

    async def fetch_candles(self,symbol,granularity_sec,count):
        data=await self.safe_ticks_history({"ticks_history":symbol,"end":"latest","count":int(count),"style":"candles","granularity":int(granularity_sec)},retries=4)
        return build_candles_from_deriv(data.get("candles",[]))

    async def fetch_candles_with_timeout(self,symbol,granularity_sec,count,timeout_sec=12.0):
        try:
            return await asyncio.wait_for(self.fetch_candles(symbol,granularity_sec,count),timeout=timeout_sec)
        except asyncio.TimeoutError:
            logger.warning(f"fetch_candles TIMEOUT ({symbol}) — forcing reconnect")
            asyncio.create_task(self.safe_reconnect()); return []
        except Exception as e:
            logger.warning(f"fetch_candles ERROR ({symbol}): {e}")
            if self._is_gatewayish_error(str(e)): asyncio.create_task(self.safe_reconnect())
            return []

    async def fetch_balance(self):
        if not self.api: return
        try:
            bal=await self.safe_deriv_call("balance",{"balance":1},retries=4)
            self.balance=f"{float(bal['balance']['balance']):.2f} {bal['balance']['currency']}"
        except: pass

    def _get_current_balance_float(self):
        try: return float(self.balance.split()[0])
        except: return 0.0

    def _next_midnight_epoch(self):
        return (datetime.now(self.tz)+timedelta(days=1)).replace(hour=0,minute=0,second=0,microsecond=0).timestamp()

    def _daily_reset_if_needed(self):
        today=datetime.now(self.tz).date()
        if today!=self.current_day:
            self.current_day=today; self.trades_today=0; self.total_losses_today=0
            self.consecutive_losses=0; self.total_profit_today=0.0
            self.cooldown_until=0.0; self.pause_until=0.0
            self.martingale_step=0; self.current_stake=0.0; self.martingale_halt=False
            self.section_profit=0.0; self.sections_won_today=0
            self.section_index=self._get_section_index_for_epoch(time.time())
            self.section_pause_until=0.0; self.max_loss_streak_today=0
            self.hit_5_losses_today=False
            self.candle_cooldown_counter={m:0 for m in MARKETS}
            self.profit_lock_active=False
            self.session_losses={}
            self.prev_z={m:None for m in MARKETS}
            self._reset_market_loss_state()
            # Reset starting balance daily
            self.starting_balance=self._get_current_balance_float()
        self._sync_section_if_needed()

    def _session_gate_ok(self):
        if not ALLOWED_SESSIONS_UTC: return True,"OK"
        sess=session_bucket(time.time())
        if sess in ALLOWED_SESSIONS_UTC: return True,"OK"
        return False,f"Session blocked: {sess}"

    # IMPROVEMENT 13: Check per-session loss limit
    def _session_loss_ok(self):
        sess=session_bucket(time.time())
        sess_loss=self.session_losses.get(sess,0.0)
        if sess_loss<=SESSION_LOSS_LIMIT:
            return False,f"Session {sess} loss limit reached (${sess_loss:.2f})"
        return True,"OK"

    # IMPROVEMENT 10: Equity protection check
    def _equity_ok(self):
        if self.starting_balance<=0: return True,"OK",1.0
        current=self._get_current_balance_float()
        ratio=current/self.starting_balance
        if ratio<=EQUITY_STOP_PCT:
            return False,f"⛔ Equity stop: balance at {ratio:.0%} of starting balance",1.0
        if ratio<=EQUITY_HALF_STAKE_PCT:
            return True,f"⚠️ Equity warning: balance at {ratio:.0%} — stake halved",0.5
        return True,"OK",1.0

    def can_auto_trade(self):
        self._daily_reset_if_needed()
        ok_sess,msg_sess=self._session_gate_ok()
        if not ok_sess: return False,msg_sess
        # IMPROVEMENT 4: Time filter
        if not is_allowed_hour():
            h=datetime.utcnow().hour
            return False,f"Outside trading hours (UTC {h}:00 — allowed {ALLOWED_HOURS_UTC[0]}:00-{ALLOWED_HOURS_UTC[1]}:00)"
        # IMPROVEMENT 13: Session loss limit
        ok_sess_loss,msg_sess_loss=self._session_loss_ok()
        if not ok_sess_loss: return False,msg_sess_loss
        # IMPROVEMENT 10: Equity check
        eq_ok,eq_msg,_=self._equity_ok()
        if not eq_ok: return False,eq_msg
        if self.martingale_halt: return False,f"Stopped: Martingale {MARTINGALE_MAX_STEPS} steps completed"
        if time.time()<self.section_pause_until:
            return False,f"Section paused. Resumes {fmt_hhmm(self.section_pause_until)} ({int(self.section_pause_until-time.time())}s)"
        if time.time()<self.pause_until:
            return False,f"Paused until 12:00am WAT ({int(self.pause_until-time.time())}s)"
        if self.total_profit_today>=DAILY_PROFIT_TARGET:
            self.pause_until=self._next_midnight_epoch()
            return False,f"Daily target reached (+${self.total_profit_today:.2f})"
        if self.total_profit_today<=DAILY_LOSS_LIMIT:
            self.pause_until=self._next_midnight_epoch()
            return False,f"Stopped: Daily loss limit (${DAILY_LOSS_LIMIT:.2f}) reached"
        # IMPROVEMENT 11: Profit lock
        if self.profit_lock_active and self.total_profit_today<=PROFIT_LOCK_FLOOR:
            return False,f"⛔ Profit lock: protecting +${PROFIT_LOCK_FLOOR:.2f} floor"
        if self.consecutive_losses>=MAX_CONSEC_LOSSES:
            return False,f"⛔ Stopped: {MAX_CONSEC_LOSSES} consecutive losses reached"
        if self.trades_today>=MAX_TRADES_PER_DAY: return False,"Stopped: daily trade limit reached"
        if time.time()<self.cooldown_until: return False,f"Cooldown {int(self.cooldown_until-time.time())}s"
        if self.active_trade_info: return False,"Trade in progress"
        if self._pending_buy: return False,"Trade in progress (pending buy)"
        if not self.api: return False,"Not connected"
        return True,"OK"

    def can_trade_market(self,symbol):
        ok,gate_msg=self.can_auto_trade()
        if not ok: return False,gate_msg
        blocked,block_msg=self._is_market_blocked(symbol)
        if blocked: return False,block_msg
        # IMPROVEMENT 16: Market warm-up check
        if self.market_recently_losing(symbol):
            return False,f"Market warm-up: {symbol} had 3+ losses in last 5 trades — waiting"
        return True,"OK"

    async def background_scanner(self):
        if not self.api: return
        # IMPROVEMENT 14: Sort markets by performance
        ordered_markets=self.get_best_markets()
        self.market_tasks={sym:asyncio.create_task(self.scan_market(sym)) for sym in ordered_markets}
        try:
            while self.is_scanning:
                if self.active_trade_info and (time.time()-self.trade_start_time>(DURATION_MIN*60+90)):
                    self.active_trade_info=None; self.active_trade_meta=None
                await asyncio.sleep(1)
        finally:
            for t in self.market_tasks.values(): t.cancel()
            self.market_tasks.clear()

    async def scan_market(self,symbol):
        self._next_poll_epoch[symbol]=time.time()+random.random()*0.5

        while self.is_scanning:
            try:
                now=time.time(); nxt=float(self._next_poll_epoch.get(symbol,0.0))
                if now<nxt:
                    await asyncio.sleep(min(1.0,nxt-now)); continue

                if self.consecutive_losses>=MAX_CONSEC_LOSSES or self.trades_today>=MAX_TRADES_PER_DAY:
                    self.is_scanning=False; break

                ok_gate,gate=self.can_trade_market(symbol)
                candles_1m=await self.fetch_candles_with_timeout(symbol,60,CANDLES_COUNT)

                if not candles_1m:
                    self.market_debug[symbol]={"time":time.time(),"gate":"Candle fetch failed","why":["M1 candles empty — retrying in 8s"]}
                    self._next_poll_epoch[symbol]=time.time()+8; continue

                self._rate_limit_strikes[symbol]=0

                if len(candles_1m)<MA_PERIOD+5:
                    self.market_debug[symbol]={"time":time.time(),"gate":f"Warming up ({len(candles_1m)}/{MA_PERIOD+5})","why":[f"Need {MA_PERIOD+5} candles"]}
                    self._next_poll_epoch[symbol]=time.time()+12; continue

                confirm=candles_1m[-2]; confirm_t0=int(confirm["t0"])
                self._next_poll_epoch[symbol]=float(confirm_t0+TF_SEC+2.5)

                if self.last_processed_closed_t0[symbol]==confirm_t0: continue

                closes=[x["c"] for x in candles_1m[:-1]]
                highs=[x["h"] for x in candles_1m[:-1]]
                lows=[x["l"] for x in candles_1m[:-1]]

                z_now,z_prev=calculate_zscore(closes,ma_period=MA_PERIOD,std_period=STD_PERIOD)

                if z_now is None or z_prev is None:
                    self.market_debug[symbol]={"time":time.time(),"gate":gate,"why":["Z-score not ready"]}
                    self.last_processed_closed_t0[symbol]=confirm_t0; continue

                # IMPROVEMENT 21: Get tuned Z threshold for this market
                z_thresh=self.get_tuned_z_threshold(symbol)
                z_min=z_thresh["min"]; z_max=z_thresh["max"]

                vol_exp=volatility_expanding(highs,lows,closes)

                # Spike detection
                spike=is_spike_candle(candles_1m)

                # Candle cooldown
                if self.candle_cooldown_counter.get(symbol,0)>0:
                    self.candle_cooldown_counter[symbol]-=1
                    self.market_debug[symbol]={"time":time.time(),"gate":gate,"z_now":round(z_now,4),"vol_expanding":vol_exp,"why":[f"Candle cooldown: {self.candle_cooldown_counter[symbol]+1} left"]}
                    self.last_processed_closed_t0[symbol]=confirm_t0; continue

                signal=None
                reason="No entry: conditions not met"
                z_abs=abs(z_now)

                # ===== 6 CORE CONDITIONS =====
                # 1. ATR not expanding
                if vol_exp:
                    reason=f"Blocked: Volatility expanding | Z={z_now:.3f}"
                # 2. No spike candle
                elif spike:
                    reason=f"Blocked: Spike candle detected | Z={z_now:.3f}"
                # 3. Z below cap (not a trend)
                elif z_abs > z_max:
                    reason=f"Blocked: Z={z_now:.3f} beyond cap +-{z_max} (likely trend)"
                else:
                    # 4. Z crosses threshold
                    buy_cross  = z_prev > -z_min and z_now <= -z_min
                    sell_cross = z_prev < z_min  and z_now >= z_min
                    # 5. Z momentum reversing back toward 0
                    buy_momentum  = z_now > z_prev
                    sell_momentum = z_now < z_prev
                    # 6. Trading hours handled by gate check above

                    if buy_cross and buy_momentum:
                        signal="CALL"
                        reason=f"BUY: Z={z_now:.3f} crossed below -{z_min} | momentum up"
                    elif sell_cross and sell_momentum:
                        signal="PUT"
                        reason=f"SELL: Z={z_now:.3f} crossed above +{z_min} | momentum down"
                    else:
                        parts=[]
                        if not buy_cross and not sell_cross:
                            parts.append(f"Z={z_now:.3f} not crossing threshold +-{z_min}")
                        if not buy_momentum and not sell_momentum:
                            parts.append("Z not yet reversing toward 0")
                        reason="Waiting: "+", ".join(parts) if parts else "No signal"

                mkt_blocked,_=self._is_market_blocked(symbol)
                mkt_losses=self.market_consec_losses.get(symbol,0)
                _,_,stake_mult=self._equity_ok()

                self.market_debug[symbol]={
                    "time":time.time(),"gate":gate,"last_closed":confirm_t0,"signal":signal,
                    "z_now":round(z_now,4),"z_prev":round(z_prev,4),
                    "z_min":z_min,"z_max":z_max,
                    "vol_expanding":vol_exp,"spike":spike,
                    "mkt_blocked":mkt_blocked,"mkt_losses":mkt_losses,
                    "stake_mult":stake_mult,"why":[reason]
                }

                self.last_processed_closed_t0[symbol]=confirm_t0
                if not ok_gate: continue

                if signal=="CALL":
                    await self.execute_trade("CALL",symbol,source="AUTO",z_now=z_now,stake_mult=stake_mult)
                    self.candle_cooldown_counter[symbol]=CANDLE_COOLDOWN
                elif signal=="PUT":
                    await self.execute_trade("PUT",symbol,source="AUTO",z_now=z_now,stake_mult=stake_mult)
                    self.candle_cooldown_counter[symbol]=CANDLE_COOLDOWN
                elif signal=="PUT":
                    self.last_signal_time[symbol]=time.time()
                    await self.execute_trade("PUT",symbol,source="AUTO",z_now=z_now,stake_mult=stake_mult)
                    self.candle_cooldown_counter[symbol]=CANDLE_COOLDOWN

            except asyncio.CancelledError: break
            except Exception as e:
                msg=str(e); logger.error(f"Scanner Error ({symbol}): {msg}")
                if self._is_rate_limit_error(msg):
                    self._rate_limit_strikes[symbol]=int(self._rate_limit_strikes.get(symbol,0))+1
                    self._next_poll_epoch[symbol]=time.time()+min(180,RATE_LIMIT_BACKOFF_BASE*self._rate_limit_strikes[symbol])
                else:
                    await asyncio.sleep(2 if not self._is_gatewayish_error(msg) else 5)
            await asyncio.sleep(0.05)

    async def execute_trade(self,side,symbol,reason="MANUAL",source="MANUAL",z_now=0.0,stake_mult=1.0):
        if not self.api or self.active_trade_info: return
        async with self.trade_lock:
            ok,_gate=(self.can_trade_market(symbol) if source=="AUTO" else self.can_auto_trade())
            if not ok or self._pending_buy: return
            self._pending_buy=True
            try:
                # IMPROVEMENT 6: Adaptive expiry
                expiry=get_adaptive_expiry(abs(z_now))
                base_payout=money2(max(float(MIN_PAYOUT),money2(max(0.01,float(PAYOUT_TARGET)*(float(MARTINGALE_MULT)**int(self.martingale_step))))))
                # IMPROVEMENT 10: Apply stake multiplier for equity protection
                payout=money2(base_payout*stake_mult)
                payout=max(float(MIN_PAYOUT),payout)

                prop=await self.safe_deriv_call("proposal",{"proposal":1,"amount":payout,"basis":"payout","contract_type":side,"currency":"USD","duration":int(expiry),"duration_unit":"m","symbol":symbol},retries=6)
                if "error" in prop:
                    await self.safe_send_tg(f"❌ Proposal Error:\n{prop['error'].get('message','')}"); return
                p=prop["proposal"]; proposal_id=p["id"]; ask_price=float(p.get("ask_price",0))
                if ask_price<=0:
                    await self.safe_send_tg("❌ Invalid ask_price."); return
                if ask_price>float(MAX_STAKE_ALLOWED):
                    await self.safe_send_tg(f"⛔️ Skipped: stake=${ask_price:.2f} > max ${MAX_STAKE_ALLOWED:.2f}")
                    self.cooldown_until=time.time()+COOLDOWN_SEC; return
                buy=await self.safe_deriv_call("buy",{"buy":proposal_id,"price":float(MAX_STAKE_ALLOWED)},retries=1)
                if "error" in buy:
                    await self.safe_send_tg(f"❌ Trade Refused:\n{buy['error'].get('message','')}"); return
                self.active_trade_info=int(buy["buy"]["contract_id"])
                self.active_market=symbol; self.trade_start_time=time.time(); self.current_stake=ask_price
                self.active_trade_meta={"symbol":symbol,"side":side,"open_epoch":float(self.trade_start_time),"source":source,"expiry":expiry}
                if source=="AUTO": self.trades_today+=1
                z_thresh=self.get_tuned_z_threshold(symbol)
                await self.safe_send_tg(
                    f"🚀 {side} TRADE OPENED\n"
                    f"🛒 Market: {symbol.replace('_',' ')}\n"
                    f"⏱ Expiry: {expiry}m {'(adaptive)' if expiry!=EXPIRY_NORMAL_MIN else ''}\n"
                    f"🎁 Payout: ${payout:.2f}\n"
                    f"📐 Z-Score: {z_now:.4f} (threshold ±{z_thresh['min']:.1f}/cap ±{z_thresh['max']:.1f})\n"
                    f"🎲 Martingale step: {self.martingale_step}/{MARTINGALE_MAX_STEPS}\n"
                    f"💵 Stake: ${ask_price:.2f}{' (halved)' if stake_mult<1 else ''}\n"
                    f"🕓 Session(UTC): {session_bucket(self.trade_start_time)}\n"
                    f"🤖 Source: {source}\n"
                    f"📉 {symbol.replace('_',' ')} loss streak: {self.market_consec_losses.get(symbol,0)}\n"
                    f"🔒 Profit lock: {'ACTIVE' if self.profit_lock_active else 'OFF'}\n"
                    f"🎯 Today PnL: {self.total_profit_today:+.2f}/+{DAILY_PROFIT_TARGET:.2f}"
                )
                asyncio.create_task(self.check_result(self.active_trade_info,source,side,z_now,expiry))
            except Exception as e:
                logger.error(f"Trade error: {e}")
                await self.safe_send_tg(f"⚠️ Trade error:\n{e}")
            finally:
                self._pending_buy=False

    async def check_result(self,cid,source,side,z_now,expiry=3):
        await asyncio.sleep(int(expiry)*60+5)
        try:
            res=await self.safe_deriv_call("proposal_open_contract",{"proposal_open_contract":1,"contract_id":cid},retries=6)
            profit=float(res["proposal_open_contract"].get("profit",0))
            traded_symbol="—"
            if source=="AUTO" and self.active_trade_meta:
                traded_symbol=self.active_trade_meta.get("symbol","—")
                self.record_trade_result(traded_symbol,float(self.active_trade_meta.get("open_epoch",time.time())),profit)
            if source=="AUTO":
                self.total_profit_today+=profit; self.section_profit+=profit
                # IMPROVEMENT 13: Track per-session loss
                sess=session_bucket(time.time())
                self.session_losses[sess]=self.session_losses.get(sess,0.0)+profit
                if traded_symbol in MARKETS: self._update_market_loss_state(traded_symbol,profit)
                if self.section_profit>=float(SECTION_PROFIT_TARGET):
                    self.sections_won_today+=1; self.section_pause_until=self._next_section_start_epoch(time.time())
                if profit<=0:
                    self.consecutive_losses+=1; self.total_losses_today+=1
                    if self.consecutive_losses>self.max_loss_streak_today:
                        self.max_loss_streak_today=self.consecutive_losses
                    if self.consecutive_losses>=5 and not self.hit_5_losses_today:
                        self.hit_5_losses_today=True; self.is_scanning=False
                        await self.safe_send_tg(f"⛔ STOPPED: 5 consecutive losses reached. Bot paused until tomorrow.")
                    if self.martingale_step<MARTINGALE_MAX_STEPS: self.martingale_step+=1
                    else: self.martingale_halt=True; self.is_scanning=False
                else:
                    self.consecutive_losses=0; self.martingale_step=0; self.martingale_halt=False
                # IMPROVEMENT 11: Activate profit lock
                if self.total_profit_today>=PROFIT_LOCK_TRIGGER and not self.profit_lock_active:
                    self.profit_lock_active=True
                    await self.safe_send_tg(f"🔒 PROFIT LOCK ACTIVATED at +${self.total_profit_today:.2f} — floor set at +${PROFIT_LOCK_FLOOR:.2f}")
                if self.total_profit_today>=DAILY_PROFIT_TARGET:
                    self.pause_until=self._next_midnight_epoch()
            await self.fetch_balance()
            next_payout=money2(float(PAYOUT_TARGET)*(float(MARTINGALE_MULT)**int(self.martingale_step)))
            mkt_losses_after=self.market_consec_losses.get(traded_symbol,0)
            mkt_blocked_after=self.market_blocked.get(traded_symbol,False)
            mkt_note=f"\n🚫 {traded_symbol.replace('_',' ')} BLOCKED ({mkt_losses_after}L)" if mkt_blocked_after else f"\n📊 {traded_symbol.replace('_',' ')} streak: {mkt_losses_after}"
            pause_note="\n⏸ Paused until 12:00am WAT" if time.time()<self.pause_until else ""
            halt_note=f"\n🛑 Martingale stopped" if self.martingale_halt else ""
            section_note=f"\n🧩 Section paused until {fmt_hhmm(self.section_pause_until)}" if time.time()<self.section_pause_until else ""
            lock_note=f"\n🔒 Profit lock active (floor +${PROFIT_LOCK_FLOOR:.2f})" if self.profit_lock_active else ""
            await self.safe_send_tg(
                f"🏁 FINISH: {'WIN ✅' if profit>0 else 'LOSS ❌'} ({profit:+.2f})\n"
                f"📐 Z-Score was: {z_now:.4f}\n"
                f"🧩 Section PnL: {self.section_profit:+.2f}/+{SECTION_PROFIT_TARGET:.2f}\n"
                f"📊 Today: {self.trades_today}/{MAX_TRADES_PER_DAY} | Losses: {self.total_losses_today} | Streak: {self.consecutive_losses}/{MAX_CONSEC_LOSSES}\n"
                f"📌 Max streak: {self.max_loss_streak_today} | 5-loss hit: {'YES ⛔' if self.hit_5_losses_today else 'NO'}\n"
                f"💵 Today PnL: {self.total_profit_today:+.2f}/+{DAILY_PROFIT_TARGET:.2f}\n"
                f"🎁 Next payout: ${next_payout:.2f} (step {self.martingale_step}/{MARTINGALE_MAX_STEPS})\n"
                f"💰 Balance: {self.balance}"
                f"{mkt_note}{lock_note}{pause_note}{section_note}{halt_note}"
            )

            # IMPROVEMENT 23: Daily performance report at midnight
            now_dt=datetime.now(self.tz)
            if now_dt.hour==23 and now_dt.minute>=55:
                await self._send_daily_report()

        finally:
            self.active_trade_info=None; self.active_trade_meta=None
            self.cooldown_until=time.time()+COOLDOWN_SEC

    # IMPROVEMENT 23: Daily report
    async def _send_daily_report(self):
        by_mkt,by_sess,wr=self.stats_30d()
        lines=["📊 DAILY PERFORMANCE REPORT\n"]
        lines.append(f"Date: {datetime.now(self.tz).strftime('%Y-%m-%d')}")
        lines.append(f"Total trades: {self.trades_today}")
        lines.append(f"Total PnL: {self.total_profit_today:+.2f}")
        lines.append(f"Max loss streak: {self.max_loss_streak_today}")
        lines.append(f"Balance: {self.balance}\n")
        lines.append("📈 30-Day Market Win Rates:")
        for m,d in sorted(by_mkt.items(),key=lambda x:wr(x[1]),reverse=True):
            lines.append(f"  {m.replace('_',' ')}: {wr(d):.1f}% ({d['wins']}/{d['trades']})")
        lines.append("\n🕓 30-Day Session Win Rates:")
        for s,d in sorted(by_sess.items(),key=lambda x:wr(x[1]),reverse=True):
            lines.append(f"  {s}: {wr(d):.1f}% ({d['wins']}/{d['trades']})")
        await self.safe_send_tg("\n".join(lines))


# ===== TELEGRAM COMMANDS (IMPROVEMENT 22) =====
async def cmd_setstake(u:Update,c:ContextTypes.DEFAULT_TYPE):
    global PAYOUT_TARGET
    try:
        val=float(c.args[0])
        PAYOUT_TARGET=val
        await u.message.reply_text(f"✅ Stake set to ${val:.2f}")
    except:
        await u.message.reply_text("Usage: /setstake 0.50")

async def cmd_setmarkets(u:Update,c:ContextTypes.DEFAULT_TYPE):
    global MARKETS
    valid=["R_10","R_25","R_50","R_75","R_100"]
    try:
        new_markets=[m.upper().replace("R","R_") if not m.startswith("R_") else m.upper() for m in c.args]
        new_markets=[m for m in new_markets if m in valid]
        if not new_markets:
            await u.message.reply_text(f"No valid markets. Use: {', '.join(valid)}"); return
        MARKETS=new_markets
        await u.message.reply_text(f"✅ Markets set to: {', '.join(MARKETS)}")
    except:
        await u.message.reply_text("Usage: /setmarkets R_50 R_25")

async def cmd_pause(u:Update,c:ContextTypes.DEFAULT_TYPE):
    try:
        minutes=int(c.args[0])
        bot_logic.pause_until=time.time()+minutes*60
        await u.message.reply_text(f"⏸ Bot paused for {minutes} minutes.")
    except:
        await u.message.reply_text("Usage: /pause 30")

async def cmd_stats(u:Update,c:ContextTypes.DEFAULT_TYPE):
    await bot_logic._send_daily_report()

async def cmd_resume(u:Update,c:ContextTypes.DEFAULT_TYPE):
    bot_logic.pause_until=0.0
    bot_logic.section_pause_until=0.0
    await u.message.reply_text("▶️ Bot resumed.")


# ===== UI =====
bot_logic=DerivSniperBot()

def main_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("▶️ START",callback_data="START_SCAN"),InlineKeyboardButton("⏹️ STOP",callback_data="STOP_SCAN")],
        [InlineKeyboardButton("📊 STATUS",callback_data="STATUS"),InlineKeyboardButton("🔄 REFRESH",callback_data="STATUS")],
        [InlineKeyboardButton("🧩 SECTION",callback_data="NEXT_SECTION")],
        [InlineKeyboardButton("🧪 TEST BUY",callback_data="TEST_BUY")],
        [InlineKeyboardButton("🧪 DEMO",callback_data="SET_DEMO"),InlineKeyboardButton("💰 LIVE",callback_data="SET_REAL")],
    ])

def format_market_detail(sym,d):
    if not d: return f"📍 {sym.replace('_',' ')}\n⏳ No scan data yet"
    age=int(time.time()-d.get("time",time.time()))
    z_now=d.get("z_now","—"); z_prev=d.get("z_prev","—")
    z_min=d.get("z_min","—"); z_max=d.get("z_max","—")
    vol_exp=d.get("vol_expanding",False); spike=d.get("spike",False)
    signal=d.get("signal") or "—"; trend=d.get("trend","—")
    mkt_blocked=d.get("mkt_blocked",False); mkt_losses=d.get("mkt_losses",0)
    stake_mult=d.get("stake_mult",1.0); why=d.get("why",[]); gate=d.get("gate","—")
    last_closed=d.get("last_closed",0)
    return (
        f"📍 {sym.replace('_',' ')} ({age}s ago)\n"
        f"Gate: {gate}\n"
        f"Market: {'🚫 BLOCKED' if mkt_blocked else '✅ Active'} ({mkt_losses}L streak)\n"
        f"Last closed: {fmt_time_hhmmss(last_closed)}\n"
        f"────────────────\n"
        f"📐 Z Now: {z_now} | Z Prev: {z_prev}\n"
        f"📐 Threshold: ±{z_min} | Cap: ±{z_max}\n"
        f"📈 Trend (MA200): {trend}\n"
        f"📊 Volatility: {'🔴 EXPANDING' if vol_exp else '🟢 Normal'}\n"
        f"⚡ Spike: {'🔴 YES' if spike else '🟢 NO'}\n"
        f"💵 Stake mult: {stake_mult:.1f}x\n"
        f"Signal: {signal}\n"
        f"Why: {why[0] if why else '—'}\n"
    )

async def _safe_answer(q,text=None,show_alert=False):
    try: await q.answer(text=text,show_alert=show_alert)
    except Exception as e: logger.warning(f"Callback answer: {e}")

async def _safe_edit(q,text,reply_markup=None):
    try: await q.edit_message_text(text,reply_markup=reply_markup)
    except Exception as e: logger.warning(f"Edit failed: {e}")

async def btn_handler(u:Update,c:ContextTypes.DEFAULT_TYPE):
    q=u.callback_query; await _safe_answer(q)
    await _safe_edit(q,"⏳ Working...",reply_markup=main_keyboard())

    if q.data=="SET_DEMO":
        bot_logic.active_token,bot_logic.account_type=DEMO_TOKEN,"DEMO"
        ok=await bot_logic.connect()
        await _safe_edit(q,"✅ Connected to DEMO" if ok else "❌ DEMO Failed",reply_markup=main_keyboard())
    elif q.data=="SET_REAL":
        bot_logic.active_token,bot_logic.account_type=REAL_TOKEN,"LIVE"
        ok=await bot_logic.connect()
        await _safe_edit(q,"✅ LIVE CONNECTED" if ok else "❌ LIVE Failed",reply_markup=main_keyboard())
    elif q.data=="START_SCAN":
        if not bot_logic.api:
            await _safe_edit(q,"❌ Connect first.",reply_markup=main_keyboard()); return
        bot_logic.is_scanning=True
        bot_logic.scanner_task=asyncio.create_task(bot_logic.background_scanner())
        await _safe_edit(q,"🔍 SCANNER ACTIVE\n✅ Z-Score strategy v2 running.",reply_markup=main_keyboard())
    elif q.data=="STOP_SCAN":
        bot_logic.is_scanning=False
        if bot_logic.scanner_task and not bot_logic.scanner_task.done(): bot_logic.scanner_task.cancel()
        await _safe_edit(q,"⏹️ Scanner stopped.",reply_markup=main_keyboard())
    elif q.data=="NEXT_SECTION":
        bot_logic._daily_reset_if_needed()
        now=time.time(); nxt=bot_logic._next_section_start_epoch(now)
        if nxt<=now+1: nxt=now+1
        bot_logic.section_index=bot_logic._get_section_index_for_epoch(nxt+1)
        bot_logic.section_profit=0.0; bot_logic.section_pause_until=0.0
        await _safe_edit(q,f"🧩 Moved to Section {bot_logic.section_index}/{SECTIONS_PER_DAY}.",reply_markup=main_keyboard())
    elif q.data=="TEST_BUY":
        test_symbol=MARKETS[0] if MARKETS else "R_10"
        asyncio.create_task(bot_logic.execute_trade("CALL",test_symbol,"Manual Test",source="MANUAL"))
        await _safe_edit(q,f"🧪 Test trade triggered (CALL {test_symbol.replace('_',' ')}).",reply_markup=main_keyboard())
    elif q.data=="STATUS":
        now=time.time()
        if now<bot_logic.status_cooldown_until:
            await _safe_edit(q,f"⏳ Refresh cooldown: {int(bot_logic.status_cooldown_until-now)}s",reply_markup=main_keyboard()); return
        bot_logic.status_cooldown_until=now+STATUS_REFRESH_COOLDOWN_SEC
        await bot_logic.fetch_balance()
        now_time=datetime.now(ZoneInfo("Africa/Lagos")).strftime("%Y-%m-%d %H:%M:%S")
        _ok,gate=bot_logic.can_auto_trade()
        trade_status="No Active Trade"
        if bot_logic.active_trade_info and bot_logic.api:
            try:
                res=await bot_logic.safe_deriv_call("proposal_open_contract",{"proposal_open_contract":1,"contract_id":bot_logic.active_trade_info},retries=4)
                pnl=float(res["proposal_open_contract"].get("profit",0))
                expiry=bot_logic.active_trade_meta.get("expiry",DURATION_MIN) if bot_logic.active_trade_meta else DURATION_MIN
                rem=max(0,int(expiry*60)-int(time.time()-bot_logic.trade_start_time))
                icon="✅ PROFIT" if pnl>0 else "❌ LOSS" if pnl<0 else "➖ FLAT"
                trade_status=f"🚀 Active Trade ({bot_logic.active_market.replace('_',' ')})\n🕓 Session(UTC): {session_bucket(bot_logic.trade_start_time)}\n📈 PnL: {icon} ({pnl:+.2f})\n⏳ Left: {rem}s"
            except: trade_status="🚀 Active Trade: Syncing..."
        pause_line="⏸ Paused until 12:00am WAT\n" if time.time()<bot_logic.pause_until else ""
        section_line=f"🧩 Section paused until {fmt_hhmm(bot_logic.section_pause_until)}\n" if time.time()<bot_logic.section_pause_until else ""
        next_payout=money2(float(PAYOUT_TARGET)*(float(MARTINGALE_MULT)**int(bot_logic.martingale_step)))
        by_mkt,by_sess,wr=bot_logic.stats_30d()
        def fmt_stats(title,items):
            rows=[(k,wr(v),v["trades"],v["wins"]) for k,v in items.items()]
            rows.sort(key=lambda x:(x[1],x[2]),reverse=True)
            lines=[f"{title} (last {STATS_DAYS}d):"]
            if not rows: lines.append("— No trades yet"); return "\n".join(lines)
            for k,wrr,t,w in rows: lines.append(f"- {k.replace('_',' ')}: {wrr:.1f}% ({w}/{t})")
            return "\n".join(lines)
        stats_block="📈 PERFORMANCE\n"+fmt_stats("Markets",by_mkt)+"\n"+fmt_stats("Sessions",by_sess)+"\n"
        mkt_lines=["🛡 Market Status:"]
        for m in MARKETS:
            ml=bot_logic.market_consec_losses.get(m,0); mb=bot_logic.market_blocked.get(m,False)
            z_thresh=bot_logic.get_tuned_z_threshold(m)
            mkt_lines.append(f"{'🚫' if mb else '✅'} {m.replace('_',' ')}: {ml}L | Z±{z_thresh['min']:.1f}/cap±{z_thresh['max']:.1f}{'  BLOCKED' if mb else ''}")
        mkt_block="\n".join(mkt_lines)+"\n"
        sess_loss_lines=["📉 Session Losses:"]
        for s,l in bot_logic.session_losses.items():
            sess_loss_lines.append(f"  {s}: {l:+.2f} (limit ${SESSION_LOSS_LIMIT:.2f})")
        sess_loss_block=("\n".join(sess_loss_lines)+"\n") if bot_logic.session_losses else ""
        _,_,stake_mult=bot_logic._equity_ok()
        eq_ratio=bot_logic._get_current_balance_float()/bot_logic.starting_balance if bot_logic.starting_balance>0 else 1.0
        header=(
            f"🕒 Time (WAT): {now_time}\n"
            f"🤖 Bot: {'ACTIVE' if bot_logic.is_scanning else 'OFFLINE'} ({bot_logic.account_type})\n"
            f"{pause_line}{section_line}"
            f"🧩 Section: {bot_logic.section_index}/{SECTIONS_PER_DAY} | PnL: {bot_logic.section_profit:+.2f}/+{SECTION_PROFIT_TARGET:.2f}\n"
            f"🎁 Next payout: ${next_payout:.2f} | Step: {bot_logic.martingale_step}/{MARTINGALE_MAX_STEPS}\n"
            f"🧯 Max stake: ${MAX_STAKE_ALLOWED:.2f} | Stake mult: {stake_mult:.1f}x\n"
            f"💰 Equity: {eq_ratio:.0%} of starting balance\n"
            f"🔒 Profit lock: {'ACTIVE (floor +${:.2f})'.format(PROFIT_LOCK_FLOOR) if bot_logic.profit_lock_active else 'OFF'}\n"
            f"🎯 Daily Target: +${DAILY_PROFIT_TARGET:.2f} | Loss Limit: ${DAILY_LOSS_LIMIT:.2f}\n"
            f"🕐 Trading hours (UTC): {ALLOWED_HOURS_UTC[0]}:00 — {ALLOWED_HOURS_UTC[1]}:00\n"
            f"📡 Markets: {', '.join(m.replace('_',' ') for m in MARKETS)}\n"
            f"🧭 Strategy: Z-Score v2 | MA{MA_PERIOD}/STD{STD_PERIOD} | ATR{ATR_SHORT}/{ATR_LONG}x{ATR_EXPANSION_MULT}\n"
            f"⛔ Stop after: {MAX_CONSEC_LOSSES} consecutive losses\n"
            f"━━━━━━━━━━━━━━━\n{trade_status}\n━━━━━━━━━━━━━━━\n"
            f"{stats_block}{mkt_block}{sess_loss_block}"
            f"💵 Today PnL: {bot_logic.total_profit_today:+.2f}\n"
            f"🎯 Trades: {bot_logic.trades_today}/{MAX_TRADES_PER_DAY} | ❌ Losses: {bot_logic.total_losses_today}\n"
            f"📉 Streak: {bot_logic.consecutive_losses}/{MAX_CONSEC_LOSSES} | Max today: {bot_logic.max_loss_streak_today}\n"
            f"⚠️ Hit 5-loss today: {'YES ⛔' if bot_logic.hit_5_losses_today else 'NO'}\n"
            f"🚦 Gate: {gate}\n"
            f"💰 Balance: {bot_logic.balance}\n"
            f"\nCommands: /setstake /setmarkets /pause /resume /stats"
        )
        details="\n\n📌 LIVE SCAN\n\n"+"\n\n".join([format_market_detail(sym,bot_logic.market_debug.get(sym,{})) for sym in MARKETS])
        await _safe_edit(q,header+details,reply_markup=main_keyboard())

async def start_cmd(u:Update,c:ContextTypes.DEFAULT_TYPE):
    await u.message.reply_text(
        "💎 Deriv Z-Score Bot v2\n"
        f"🧭 Strategy: Z-Score Mean Reversion + Full Filter Suite\n"
        f"📐 Dynamic Z thresholds | ATR filter | Trend filter\n"
        f"🛡 Spike block | Candle confirm | Z momentum | Correlation filter\n"
        f"💰 Equity protection | Profit lock | Session loss limits\n"
        f"📊 Auto market rotation | Performance tuning\n"
        f"⛔ Stops after {MAX_CONSEC_LOSSES} consecutive losses\n"
        f"⏱ Adaptive expiry: {EXPIRY_NORMAL_MIN}m/{EXPIRY_STRONG_MIN}m\n"
        f"🕐 Trading hours (UTC): {ALLOWED_HOURS_UTC[0]}:00—{ALLOWED_HOURS_UTC[1]}:00\n"
        f"📲 Commands: /setstake /setmarkets /pause /resume /stats\n",
        reply_markup=main_keyboard()
    )

if __name__=="__main__":
    app=Application.builder().token(TELEGRAM_TOKEN).build()
    bot_logic.app=app
    app.add_handler(CommandHandler("start",start_cmd))
    app.add_handler(CallbackQueryHandler(btn_handler))
    app.add_handler(CommandHandler("setstake",cmd_setstake))
    app.add_handler(CommandHandler("setmarkets",cmd_setmarkets))
    app.add_handler(CommandHandler("pause",cmd_pause))
    app.add_handler(CommandHandler("resume",cmd_resume))
    app.add_handler(CommandHandler("stats",cmd_stats))
    app.run_polling(drop_pending_updates=True)
