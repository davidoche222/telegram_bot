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
MARKETS = ["R_10", "R_25", "R_50", "R_75", "R_100"]
COOLDOWN_SEC = 180          # 3 minutes after every trade
MAX_TRADES_PER_DAY = 30     # 6 per index × 5 indices
MAX_TRADES_PER_MARKET = 6   # max per index per day
MAX_LOSSES_PER_MARKET = 3   # stop index for day after 3 losses
CONSEC_LOSS_PAUSE_SEC = 1800  # 30 min pause after 2 consecutive losses per market
CHOP_PAUSE_SEC = 1200       # 20 min pause after chop detected
MAX_CONSEC_LOSSES = 5       # global stop
TELEGRAM_TOKEN = "8589420556:AAHmB6YE9KIEu0tBIgWdd9baBDt0eDh5FY8"
TELEGRAM_CHAT_ID = "7634818949"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ===== STRATEGY =====
TF_M5_SEC = 300             # M5 candles
TF_M1_SEC = 60              # M1 candles
CANDLES_M5 = 100            # enough for EMA50 + ADX warmup on M5
CANDLES_M1 = 80             # enough for EMA50 + ADX warmup on M1
EXPIRY_MIN = 2              # 2 minute expiry

# EMA settings
EMA20_PERIOD = 20
EMA50_PERIOD = 50
EMA_SLOPE_LOOKBACK = 10
EMA_SLOPE_MIN = 0.05        # EMA50 must be at least gently rising/falling

# ADX settings
ADX_PERIOD = 14
ADX_MIN = 20.0
ADX_GRACE_CANDLES = 3       # ADX valid if above 20 within last 3 candles

# ATR settings
ATR_PERIOD = 14
ATR_SMA_PERIOD = 20

# VWAP pullback zone
PULLBACK_ZONE_MULT = 0.30   # relaxed — price within 0.30*ATR of VWAP

# RSI settings — relaxed single tier
RSI_PERIOD = 14
RSI_CALL_MIN = 40
RSI_CALL_MAX = 72
RSI_PUT_MIN = 28
RSI_PUT_MAX = 60

# Body ratio — relaxed
BODY_RATIO_MIN = 0.40

# Chop filter
CHOP_CROSS_LIMIT = 4

# ===== MARTINGALE =====
MARTINGALE_MULT = 2
MARTINGALE_MAX_STEPS = 5

# ===== PAYOUT / STAKE =====
PAYOUT_TARGET = 1.00
MIN_PAYOUT = 0.35
MAX_STAKE_ALLOWED = 10.00

# ===== DAILY TARGETS =====
DAILY_PROFIT_TARGET = 10.0
DAILY_LOSS_LIMIT = -20.0    # covers 5-step martingale worst case ($15.50)

# ===== EQUITY PROTECTION =====
EQUITY_HALF_STAKE_PCT = 0.80
EQUITY_STOP_PCT = 0.60

# ===== PROFIT LOCK =====
PROFIT_LOCK_TRIGGER = 5.0
PROFIT_LOCK_FLOOR = 2.0

# ===== MISC =====
TICKS_GLOBAL_MIN_INTERVAL = 0.35
RATE_LIMIT_BACKOFF_BASE = 20
STATUS_REFRESH_COOLDOWN_SEC = 10
STATS_DAYS = 30
TRADE_LOG_FILE = "vwap_trade_log.jsonl"
SESSION_BUCKETS = [("ASIA",0,6),("LONDON",7,11),("OVERLAP",12,15),("NEWYORK",16,20),("LATE_NY",21,23)]


# ===== INDICATORS =====

def calculate_ema(closes, period):
    """Returns single EMA value (latest)"""
    closes = np.array(closes, dtype=float)
    if len(closes) < period: return None
    k = 2.0 / (period + 1)
    ema = np.mean(closes[:period])
    for price in closes[period:]:
        ema = price * k + ema * (1 - k)
    return float(ema)

def calculate_ema_array(closes, period):
    """Returns full EMA array — needed for slope and pullback detection"""
    closes = np.array(closes, dtype=float)
    n = len(closes)
    if n < period: return np.full(n, np.nan)
    k = 2.0 / (period + 1)
    ema = np.full(n, np.nan, dtype=float)
    ema[period-1] = np.mean(closes[:period])
    for i in range(period, n):
        ema[i] = closes[i] * k + ema[i-1] * (1 - k)
    return ema

def calculate_adx(highs, lows, closes, period=14):
    """Full ADX array"""
    highs=np.array(highs,dtype=float); lows=np.array(lows,dtype=float); closes=np.array(closes,dtype=float)
    n=len(closes)
    if n < period*2+2: return np.full(n,np.nan)
    up_move=highs[1:]-highs[:-1]; down_move=lows[:-1]-lows[1:]
    plus_dm=np.where((up_move>down_move)&(up_move>0),up_move,0.0)
    minus_dm=np.where((down_move>up_move)&(down_move>0),down_move,0.0)
    prev_c=closes[:-1]
    tr=np.maximum(highs[1:]-lows[1:],np.maximum(np.abs(highs[1:]-prev_c),np.abs(lows[1:]-prev_c)))
    tr_s=np.zeros_like(tr); plus_s=np.zeros_like(plus_dm); minus_s=np.zeros_like(minus_dm)
    tr_s[period-1]=np.sum(tr[:period]); plus_s[period-1]=np.sum(plus_dm[:period]); minus_s[period-1]=np.sum(minus_dm[:period])
    for i in range(period,len(tr)):
        tr_s[i]=tr_s[i-1]-(tr_s[i-1]/period)+tr[i]
        plus_s[i]=plus_s[i-1]-(plus_s[i-1]/period)+plus_dm[i]
        minus_s[i]=minus_s[i-1]-(minus_s[i-1]/period)+minus_dm[i]
    plus_di=100.0*(plus_s/(tr_s+1e-12)); minus_di=100.0*(minus_s/(tr_s+1e-12))
    dx=100.0*(np.abs(plus_di-minus_di)/(plus_di+minus_di+1e-12))
    adx=np.full(n,np.nan,dtype=float); dx_full=np.full(n,np.nan,dtype=float); dx_full[1:]=dx
    start=period*2
    if start<n: adx[start]=np.nanmean(dx_full[period:start+1])
    for i in range(start+1,n):
        if not np.isnan(adx[i-1]) and not np.isnan(dx_full[i]):
            adx[i]=(adx[i-1]*(period-1)+dx_full[i])/period
    return adx

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

def calculate_vwap(highs, lows, closes):
    """VWAP without real volume — uses equal weight (price average)"""
    highs=np.array(highs,dtype=float)
    lows=np.array(lows,dtype=float)
    closes=np.array(closes,dtype=float)
    typical = (highs + lows + closes) / 3.0
    # cumulative average as proxy for VWAP on synthetic indices
    return float(np.mean(typical))

def calculate_rsi(closes, period=14):
    closes = np.array(closes, dtype=float)
    if len(closes) < period+1: return None
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    for i in range(period, len(gains)):
        avg_gain = (avg_gain*(period-1)+gains[i])/period
        avg_loss = (avg_loss*(period-1)+losses[i])/period
    if avg_loss == 0: return 100.0
    rs = avg_gain / avg_loss
    return float(100 - (100/(1+rs)))

def count_vwap_crosses(m1_closes, m1_highs, m1_lows, lookback=10):
    """Count how many times price crossed VWAP in last N candles"""
    if len(m1_closes) < lookback+1: return 0
    recent_c = m1_closes[-lookback:]
    recent_h = m1_highs[-lookback:]
    recent_l = m1_lows[-lookback:]
    vwap = calculate_vwap(recent_h, recent_l, recent_c)
    crosses = 0
    for i in range(1, len(recent_c)):
        prev_above = recent_c[i-1] > vwap
        curr_above = recent_c[i] > vwap
        if prev_above != curr_above:
            crosses += 1
    return crosses

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


# ===== BOT =====
class DerivVWAPBot:
    def __init__(self):
        self.api=None; self.app=None; self.active_token=None; self.account_type="None"
        self.is_scanning=False; self.scanner_task=None; self.market_tasks={}
        self.active_trade_info=None; self.active_market="None"
        self.trade_start_time=0.0; self.active_trade_meta=None
        self.cooldown_until=0.0
        self.trades_today=0; self.total_losses_today=0
        self.consecutive_losses=0; self.total_profit_today=0.0
        self.balance="0.00"; self.starting_balance=0.0
        self.max_loss_streak_today=0; self.hit_max_losses_today=False
        self.current_stake=0.0; self.martingale_step=0; self.martingale_halt=False
        self.trade_lock=asyncio.Lock(); self._pending_buy=False
        self.profit_lock_active=False
        self.tz=ZoneInfo("Africa/Lagos")
        self.current_day=datetime.now(self.tz).date()
        self.pause_until=0.0

        # Per-market state
        self.market_trades_today={m:0 for m in MARKETS}
        self.market_losses_today={m:0 for m in MARKETS}
        self.market_consec_losses={m:0 for m in MARKETS}
        self.market_blocked={m:False for m in MARKETS}       # blocked for the day (3 losses)
        self.market_pause_until={m:0.0 for m in MARKETS}    # paused 30min (2 consec losses)
        self.market_chop_until={m:0.0 for m in MARKETS}     # paused 20min (chop detected)
        self.market_debug={m:{} for m in MARKETS}
        self.last_processed_m1_t0={m:0 for m in MARKETS}

        self._ticks_lock=asyncio.Lock(); self._last_ticks_ts=0.0
        self._next_poll_epoch={m:0.0 for m in MARKETS}
        self._rate_limit_strikes={m:0 for m in MARKETS}
        self.status_cooldown_until=0.0
        self.trade_log_path=os.path.abspath(TRADE_LOG_FILE)
        self.trade_records=[]; self._load_trade_log()
        self.session_losses={}

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

    def record_trade_result(self,symbol,open_epoch,profit,direction,rsi,atr,vwap_dist,m5_trend):
        sess=session_bucket(open_epoch); win=1 if profit>0 else 0
        rec={
            "t":float(open_epoch),"symbol":str(symbol),"session":str(sess),
            "win":int(win),"profit":float(profit),"direction":direction,
            "rsi":round(float(rsi),2) if rsi else 0,
            "atr":round(float(atr),5) if atr else 0,
            "vwap_dist":round(float(vwap_dist),5) if vwap_dist else 0,
            "m5_trend":str(m5_trend)
        }
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

    def _update_market_loss_state(self,symbol,profit):
        if profit<=0:
            self.market_losses_today[symbol]=self.market_losses_today.get(symbol,0)+1
            self.market_consec_losses[symbol]=self.market_consec_losses.get(symbol,0)+1
            # Block market for day after 3 losses
            if self.market_losses_today[symbol]>=MAX_LOSSES_PER_MARKET:
                self.market_blocked[symbol]=True
                logger.info(f"🚫 {symbol} BLOCKED for today — {self.market_losses_today[symbol]} losses")
            # Pause market 30min after 2 consecutive losses
            if self.market_consec_losses[symbol]>=2:
                self.market_pause_until[symbol]=time.time()+CONSEC_LOSS_PAUSE_SEC
                logger.info(f"⏸ {symbol} paused 30min — 2 consecutive losses")
        else:
            self.market_consec_losses[symbol]=0

    def _is_market_available(self,symbol):
        now=time.time()
        if self.market_blocked.get(symbol,False):
            return False,f"Blocked for today ({self.market_losses_today.get(symbol,0)} losses)"
        if now<self.market_pause_until.get(symbol,0):
            rem=int(self.market_pause_until[symbol]-now)
            return False,f"Paused {rem}s (2 consec losses)"
        if now<self.market_chop_until.get(symbol,0):
            rem=int(self.market_chop_until[symbol]-now)
            return False,f"Chop pause {rem}s"
        if self.market_trades_today.get(symbol,0)>=MAX_TRADES_PER_MARKET:
            return False,f"Max {MAX_TRADES_PER_MARKET} trades reached today"
        return True,"OK"

    def _reset_daily(self):
        self.trades_today=0; self.total_losses_today=0
        self.consecutive_losses=0; self.total_profit_today=0.0
        self.cooldown_until=0.0; self.pause_until=0.0
        self.martingale_step=0; self.current_stake=0.0; self.martingale_halt=False
        self.max_loss_streak_today=0; self.hit_max_losses_today=False
        self.profit_lock_active=False; self.session_losses={}
        self.market_trades_today={m:0 for m in MARKETS}
        self.market_losses_today={m:0 for m in MARKETS}
        self.market_consec_losses={m:0 for m in MARKETS}
        self.market_blocked={m:False for m in MARKETS}
        self.market_pause_until={m:0.0 for m in MARKETS}
        self.market_chop_until={m:0.0 for m in MARKETS}
        self.last_processed_m1_t0={m:0 for m in MARKETS}
        self.starting_balance=self._get_current_balance_float()

    def _daily_reset_if_needed(self):
        today=datetime.now(self.tz).date()
        if today!=self.current_day:
            self.current_day=today
            self._reset_daily()

    def _equity_ok(self):
        if self.starting_balance<=0: return True,"OK",1.0
        current=self._get_current_balance_float()
        ratio=current/self.starting_balance
        if ratio<=EQUITY_STOP_PCT:
            return False,f"Equity stop: {ratio:.0%} of starting balance",1.0
        if ratio<=EQUITY_HALF_STAKE_PCT:
            return True,f"Equity warning: {ratio:.0%} — stake halved",0.5
        return True,"OK",1.0

    def can_auto_trade(self):
        self._daily_reset_if_needed()
        eq_ok,eq_msg,_=self._equity_ok()
        if not eq_ok: return False,eq_msg
        if self.martingale_halt: return False,f"Stopped: Martingale {MARTINGALE_MAX_STEPS} steps completed"
        if time.time()<self.pause_until:
            return False,f"Paused until 12:00am WAT ({int(self.pause_until-time.time())}s)"
        if self.total_profit_today>=DAILY_PROFIT_TARGET:
            self.pause_until=self._next_midnight_epoch()
            return False,f"Daily target reached (+${self.total_profit_today:.2f})"
        if self.total_profit_today<=DAILY_LOSS_LIMIT:
            self.pause_until=self._next_midnight_epoch()
            return False,f"Daily loss limit reached (${self.total_profit_today:.2f})"
        if self.profit_lock_active and self.total_profit_today<=PROFIT_LOCK_FLOOR:
            return False,f"Profit lock: protecting +${PROFIT_LOCK_FLOOR:.2f}"
        if self.consecutive_losses>=MAX_CONSEC_LOSSES:
            return False,f"Stopped: {MAX_CONSEC_LOSSES} consecutive losses"
        if self.trades_today>=MAX_TRADES_PER_DAY:
            return False,"Daily trade limit reached"
        if time.time()<self.cooldown_until:
            return False,f"Cooldown {int(self.cooldown_until-time.time())}s"
        if self.active_trade_info: return False,"Trade in progress"
        if self._pending_buy: return False,"Trade in progress (pending)"
        if not self.api: return False,"Not connected"
        return True,"OK"

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

    def _next_midnight_epoch(self):
        return (datetime.now(self.tz)+timedelta(days=1)).replace(hour=0,minute=0,second=0,microsecond=0).timestamp()

    def _get_current_balance_float(self):
        try: return float(self.balance.split()[0])
        except: return 0.0

    async def connect(self):
        try:
            if not self.active_token: return False
            self.api=DerivAPI(app_id=APP_ID)
            await self.api.authorize(self.active_token)
            await self.fetch_balance()
            try:
                bal_val=float(self.balance.split()[0])
                if self.starting_balance==0.0: self.starting_balance=bal_val
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

    async def fetch_candles_with_timeout(self,symbol,granularity_sec,count,timeout_sec=15.0):
        try:
            return await asyncio.wait_for(self.fetch_candles(symbol,granularity_sec,count),timeout=timeout_sec)
        except asyncio.TimeoutError:
            logger.warning(f"fetch_candles TIMEOUT ({symbol} {granularity_sec}s)")
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

    async def background_scanner(self):
        if not self.api: return
        self.market_tasks={sym:asyncio.create_task(self.scan_market(sym)) for sym in MARKETS}
        try:
            while self.is_scanning:
                if self.active_trade_info and (time.time()-self.trade_start_time>(EXPIRY_MIN*60+90)):
                    self.active_trade_info=None; self.active_trade_meta=None
                await asyncio.sleep(1)
        finally:
            for t in self.market_tasks.values(): t.cancel()
            self.market_tasks.clear()

    async def scan_market(self,symbol):
        self._next_poll_epoch[symbol]=time.time()+random.random()*2.0

        while self.is_scanning:
            try:
                now=time.time(); nxt=float(self._next_poll_epoch.get(symbol,0.0))
                if now<nxt:
                    await asyncio.sleep(min(1.0,nxt-now)); continue

                if self.consecutive_losses>=MAX_CONSEC_LOSSES or self.trades_today>=MAX_TRADES_PER_DAY:
                    self.is_scanning=False; break

                ok_gate,gate=self.can_auto_trade()
                mkt_ok,mkt_msg=self._is_market_available(symbol)

                # Fetch M5 and M1 candles simultaneously
                m5_task=asyncio.create_task(self.fetch_candles_with_timeout(symbol,TF_M5_SEC,CANDLES_M5))
                m1_task=asyncio.create_task(self.fetch_candles_with_timeout(symbol,TF_M1_SEC,CANDLES_M1))
                candles_m5,candles_m1=await asyncio.gather(m5_task,m1_task)

                if not candles_m5 or not candles_m1:
                    self.market_debug[symbol]={"time":time.time(),"gate":gate,"why":["Candle fetch failed — retrying in 10s"]}
                    self._next_poll_epoch[symbol]=time.time()+10; continue

                self._rate_limit_strikes[symbol]=0

                if len(candles_m5)<EMA50_PERIOD+5 or len(candles_m1)<EMA50_PERIOD+5:
                    self.market_debug[symbol]={"time":time.time(),"gate":gate,"why":[f"Warming up — M5:{len(candles_m5)} M1:{len(candles_m1)} candles"]}
                    self._next_poll_epoch[symbol]=time.time()+15; continue

                # Poll on M1 closed candle
                confirm_m1=candles_m1[-2]; confirm_t0=int(confirm_m1["t0"])
                self._next_poll_epoch[symbol]=float(confirm_t0+TF_M1_SEC+2.5)
                if self.last_processed_m1_t0[symbol]==confirm_t0: continue

                # ===== EXTRACT DATA =====
                m5_closes=[x["c"] for x in candles_m5[:-1]]
                m5_highs=[x["h"] for x in candles_m5[:-1]]
                m5_lows=[x["l"] for x in candles_m5[:-1]]
                m1_closes=[x["c"] for x in candles_m1[:-1]]
                m1_highs=[x["h"] for x in candles_m1[:-1]]
                m1_lows=[x["l"] for x in candles_m1[:-1]]

                m5_close_now=m5_closes[-1]
                m1_close_now=m1_closes[-1]
                m1_open_now=candles_m1[-2]["o"]
                m1_prev_high=candles_m1[-3]["h"] if len(candles_m1)>=3 else None
                m1_prev_low=candles_m1[-3]["l"] if len(candles_m1)>=3 else None

                # ===== CALCULATE INDICATORS =====
                # M5: VWAP, EMA50, ATR
                m5_vwap=calculate_vwap(m5_highs,m5_lows,m5_closes)
                m5_ema50=calculate_ema(m5_closes,EMA50_PERIOD)
                m5_atrs=calculate_atr(m5_highs,m5_lows,m5_closes,ATR_PERIOD)
                m5_atr=float(m5_atrs[-2]) if len(m5_atrs)>=2 and not np.isnan(m5_atrs[-2]) else None
                m5_atr_sma=float(np.nanmean(m5_atrs[-ATR_SMA_PERIOD-1:-1])) if m5_atr else None

                # M1: EMA20 array, EMA50 array, ADX, VWAP, RSI, chop
                m1_ema20_arr=calculate_ema_array(m1_closes,EMA20_PERIOD)
                m1_ema50_arr=calculate_ema_array(m1_closes,EMA50_PERIOD)
                m1_adx_arr=calculate_adx(m1_highs,m1_lows,m1_closes,ADX_PERIOD)
                m1_vwap=calculate_vwap(m1_highs,m1_lows,m1_closes)
                m1_rsi=calculate_rsi(m1_closes,RSI_PERIOD)
                vwap_crosses=count_vwap_crosses(m1_closes,m1_highs,m1_lows,lookback=10)

                # Check all indicators ready
                if (m5_ema50 is None or m5_atr is None or m5_atr_sma is None or m1_rsi is None
                        or np.all(np.isnan(m1_ema20_arr)) or np.all(np.isnan(m1_ema50_arr))):
                    self.market_debug[symbol]={"time":time.time(),"gate":gate,"why":["Indicators warming up"]}
                    self.last_processed_m1_t0[symbol]=confirm_t0; continue

                # Get latest valid values
                m1_ema20=float(m1_ema20_arr[-2]) if not np.isnan(m1_ema20_arr[-2]) else None
                m1_ema50=float(m1_ema50_arr[-2]) if not np.isnan(m1_ema50_arr[-2]) else None

                # EMA50 slope — compare current vs EMA_SLOPE_LOOKBACK candles ago
                slope_idx=-(EMA_SLOPE_LOOKBACK+2)
                m1_ema50_old=float(m1_ema50_arr[slope_idx]) if (len(m1_ema50_arr)>abs(slope_idx) and not np.isnan(m1_ema50_arr[slope_idx])) else None
                ema50_slope=float(m1_ema50-m1_ema50_old) if (m1_ema50 and m1_ema50_old) else 0.0
                ema50_rising=ema50_slope>EMA_SLOPE_MIN
                ema50_falling=ema50_slope<-EMA_SLOPE_MIN

                # ADX — check current and grace period
                adx_now=float(m1_adx_arr[-2]) if (len(m1_adx_arr)>=2 and not np.isnan(m1_adx_arr[-2])) else None
                adx_recent=[float(v) for v in m1_adx_arr[-(ADX_GRACE_CANDLES+2):-2] if not np.isnan(v)]
                adx_ok=(adx_now is not None and adx_now>=ADX_MIN) or (len(adx_recent)>0 and max(adx_recent)>=ADX_MIN)
                adx_val=adx_now if adx_now else (max(adx_recent) if adx_recent else 0.0)

                # ATR from M1 for body sizing
                m1_atrs=calculate_atr(m1_highs,m1_lows,m1_closes,ATR_PERIOD)
                m1_atr=float(m1_atrs[-2]) if len(m1_atrs)>=2 and not np.isnan(m1_atrs[-2]) else m5_atr

                # Candle metrics
                c_body=abs(m1_close_now-m1_open_now)
                c_range=max(1e-10,float(confirm_m1["h"])-float(confirm_m1["l"]))
                body_ratio=c_body/c_range
                bull_candle=m1_close_now>m1_open_now
                bear_candle=m1_close_now<m1_open_now

                # Chop detection
                chop=vwap_crosses>=CHOP_CROSS_LIMIT
                if chop and mkt_ok:
                    self.market_chop_until[symbol]=time.time()+CHOP_PAUSE_SEC
                    logger.info(f"⚠️ {symbol} chop — pausing {CHOP_PAUSE_SEC//60}min")

                # Pullback zone — EMA20 only
                near_ema20=(m1_ema20 is not None and abs(m1_close_now-m1_ema20)<=(0.25*m5_atr))
                in_pullback=near_ema20

                # ===== M5 TREND DIRECTION =====
                if m5_close_now>m5_vwap and m5_close_now>m5_ema50:
                    m5_trend="CALL"
                elif m5_close_now<m5_vwap and m5_close_now<m5_ema50:
                    m5_trend="PUT"
                else:
                    m5_trend=None

                # M1 EMA trend
                m1_uptrend=(m1_ema20 and m1_ema50 and m1_ema20>m1_ema50)
                m1_downtrend=(m1_ema20 and m1_ema50 and m1_ema20<m1_ema50)

                # ===== SINGLE RELAXED SIGNAL =====
                signal=None; reason="No entry"

                if chop:
                    reason=f"Chop blocked — {vwap_crosses} VWAP crosses"
                elif m5_trend is None:
                    reason="No M5 trend — price between VWAP and EMA50"
                elif not adx_ok:
                    reason=f"ADX too low — {adx_val:.1f} < {ADX_MIN}"
                elif not in_pullback:
                    reason=f"No pullback — price not near EMA20 (within 0.25×ATR)"
                elif m5_trend=="CALL" and m1_uptrend and bull_candle:
                    rsi_ok=RSI_CALL_MIN<=m1_rsi<=RSI_CALL_MAX
                    body_ok=body_ratio>=BODY_RATIO_MIN
                    slope_ok_call=ema50_slope>EMA_SLOPE_MIN
                    if slope_ok_call and rsi_ok and body_ok:
                        signal="CALL"
                        reason=f"CALL ✅ | ADX={adx_val:.1f} | RSI={m1_rsi:.1f} | slope={ema50_slope:.4f} | body={body_ratio:.2f}"
                    else:
                        parts=[]
                        if not slope_ok_call: parts.append(f"slope flat ({ema50_slope:.4f})")
                        if not rsi_ok: parts.append(f"RSI={m1_rsi:.1f} out of 40-72")
                        if not body_ok: parts.append(f"weak candle ({body_ratio:.2f})")
                        reason="CALL waiting: "+", ".join(parts)
                elif m5_trend=="PUT" and m1_downtrend and bear_candle:
                    rsi_ok=RSI_PUT_MIN<=m1_rsi<=RSI_PUT_MAX
                    body_ok=body_ratio>=BODY_RATIO_MIN
                    slope_ok_put=ema50_slope<-EMA_SLOPE_MIN
                    if slope_ok_put and rsi_ok and body_ok:
                        signal="PUT"
                        reason=f"PUT ✅ | ADX={adx_val:.1f} | RSI={m1_rsi:.1f} | slope={ema50_slope:.4f} | body={body_ratio:.2f}"
                    else:
                        parts=[]
                        if not slope_ok_put: parts.append(f"slope flat ({ema50_slope:.4f})")
                        if not rsi_ok: parts.append(f"RSI={m1_rsi:.1f} out of 28-60")
                        if not body_ok: parts.append(f"weak candle ({body_ratio:.2f})")
                        reason="PUT waiting: "+", ".join(parts)
                else:
                    reason=f"M5={m5_trend} but M1 trend or candle not aligned"

                _,_,stake_mult=self._equity_ok()

                self.market_debug[symbol]={
                    "time":time.time(),"gate":gate,"mkt_msg":mkt_msg,
                    "last_closed":confirm_t0,"signal":signal,
                    "m5_trend":m5_trend,"m1_uptrend":m1_uptrend,"m1_downtrend":m1_downtrend,
                    "m5_vwap":round(m5_vwap,5),"m5_ema50":round(m5_ema50,5) if m5_ema50 else None,
                    "m1_ema20":round(m1_ema20,5) if m1_ema20 else None,
                    "m1_ema50":round(m1_ema50,5) if m1_ema50 else None,
                    "ema50_slope":round(ema50_slope,4),"ema50_rising":ema50_rising,"ema50_falling":ema50_falling,
                    "adx":round(adx_val,1),"adx_ok":adx_ok,
                    "m5_atr":round(m5_atr,5),
                    "m1_vwap":round(m1_vwap,5),"m1_rsi":round(m1_rsi,1) if m1_rsi else None,
                    "near_ema20":near_ema20,
                    "body_ratio":round(body_ratio,2),"chop":chop,
                    "vwap_crosses":vwap_crosses,
                    "mkt_losses":self.market_losses_today.get(symbol,0),
                    "mkt_trades":self.market_trades_today.get(symbol,0),
                    "stake_mult":stake_mult,"why":[reason]
                }

                self.last_processed_m1_t0[symbol]=confirm_t0
                if not ok_gate or not mkt_ok: continue

                if signal=="CALL":
                    await self.execute_trade("CALL",symbol,source="AUTO",
                        rsi=m1_rsi,atr=m5_atr,vwap_dist=vwap_dist,
                        m5_trend=m5_trend,stake_mult=stake_mult)
                elif signal=="PUT":
                    await self.execute_trade("PUT",symbol,source="AUTO",
                        rsi=m1_rsi,atr=m5_atr,vwap_dist=vwap_dist,
                        m5_trend=m5_trend,stake_mult=stake_mult)

            except asyncio.CancelledError: break
            except Exception as e:
                msg=str(e); logger.error(f"Scanner Error ({symbol}): {msg}")
                if self._is_rate_limit_error(msg):
                    self._rate_limit_strikes[symbol]=int(self._rate_limit_strikes.get(symbol,0))+1
                    self._next_poll_epoch[symbol]=time.time()+min(180,RATE_LIMIT_BACKOFF_BASE*self._rate_limit_strikes[symbol])
                else:
                    await asyncio.sleep(2 if not self._is_gatewayish_error(msg) else 5)
            await asyncio.sleep(0.05)

    async def execute_trade(self,side,symbol,reason="MANUAL",source="MANUAL",
                             rsi=0,atr=0,vwap_dist=0,m5_trend="—",stake_mult=1.0):
        if not self.api or self.active_trade_info: return
        async with self.trade_lock:
            ok,_gate=self.can_auto_trade()
            mkt_ok,_=self._is_market_available(symbol)
            if source=="AUTO" and (not ok or not mkt_ok or self._pending_buy): return
            if source=="MANUAL" and (not ok or self._pending_buy): return

            self._pending_buy=True
            try:
                base_payout=money2(max(float(MIN_PAYOUT),money2(max(0.01,float(PAYOUT_TARGET)*(float(MARTINGALE_MULT)**int(self.martingale_step))))))
                payout=money2(base_payout*stake_mult)
                payout=max(float(MIN_PAYOUT),payout)
                prop=await self.safe_deriv_call("proposal",{"proposal":1,"amount":payout,"basis":"payout","contract_type":side,"currency":"USD","duration":int(EXPIRY_MIN),"duration_unit":"m","symbol":symbol},retries=6)
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
                self.active_trade_meta={
                    "symbol":symbol,"side":side,"open_epoch":float(self.trade_start_time),
                    "source":source,"rsi":rsi,"atr":atr,"vwap_dist":vwap_dist,"m5_trend":m5_trend
                }
                if source=="AUTO":
                    self.trades_today+=1
                    self.market_trades_today[symbol]=self.market_trades_today.get(symbol,0)+1
                await self.safe_send_tg(
                    f"🚀 {side} TRADE OPENED\n"
                    f"🛒 Market: {symbol.replace('_',' ')}\n"
                    f"⏱ Expiry: {EXPIRY_MIN}m\n"
                    f"🎁 Payout: ${payout:.2f} | Stake: ${ask_price:.2f}{' (halved)' if stake_mult<1 else ''}\n"
                    f"📈 M5 Trend: {m5_trend}\n"
                    f"📊 RSI: {rsi:.1f} | ATR: {atr:.5f}\n"
                    f"📐 VWAP dist: {vwap_dist:.5f}\n"
                    f"🎲 Martingale step: {self.martingale_step}/{MARTINGALE_MAX_STEPS}\n"
                    f"🕓 Session: {session_bucket(self.trade_start_time)}\n"
                    f"🤖 Source: {source}\n"
                    f"📉 {symbol.replace('_',' ')} losses: {self.market_losses_today.get(symbol,0)}/{MAX_LOSSES_PER_MARKET}\n"
                    f"🔒 Profit lock: {'ACTIVE' if self.profit_lock_active else 'OFF'}\n"
                    f"🎯 Today PnL: {self.total_profit_today:+.2f}/+{DAILY_PROFIT_TARGET:.2f}"
                )
                asyncio.create_task(self.check_result(self.active_trade_info,source,side,rsi,atr,vwap_dist,m5_trend))
            except Exception as e:
                logger.error(f"Trade error: {e}")
                await self.safe_send_tg(f"⚠️ Trade error:\n{e}")
            finally:
                self._pending_buy=False

    async def check_result(self,cid,source,side,rsi,atr,vwap_dist,m5_trend):
        await asyncio.sleep(EXPIRY_MIN*60+5)
        try:
            res=await self.safe_deriv_call("proposal_open_contract",{"proposal_open_contract":1,"contract_id":cid},retries=6)
            profit=float(res["proposal_open_contract"].get("profit",0))
            traded_symbol="—"
            if source=="AUTO" and self.active_trade_meta:
                traded_symbol=self.active_trade_meta.get("symbol","—")
                self.record_trade_result(traded_symbol,float(self.active_trade_meta.get("open_epoch",time.time())),profit,side,rsi,atr,vwap_dist,m5_trend)
            if source=="AUTO":
                self.total_profit_today+=profit
                sess=session_bucket(time.time())
                self.session_losses[sess]=self.session_losses.get(sess,0.0)+profit
                if traded_symbol in MARKETS:
                    self._update_market_loss_state(traded_symbol,profit)
                if profit<=0:
                    self.consecutive_losses+=1; self.total_losses_today+=1
                    if self.consecutive_losses>self.max_loss_streak_today:
                        self.max_loss_streak_today=self.consecutive_losses
                    if self.consecutive_losses>=MAX_CONSEC_LOSSES and not self.hit_max_losses_today:
                        self.hit_max_losses_today=True; self.is_scanning=False
                        await self.safe_send_tg(f"⛔ STOPPED: {MAX_CONSEC_LOSSES} consecutive losses. Bot paused until tomorrow.")
                    if self.martingale_step<MARTINGALE_MAX_STEPS: self.martingale_step+=1
                    else: self.martingale_halt=True; self.is_scanning=False
                else:
                    self.consecutive_losses=0; self.martingale_step=0; self.martingale_halt=False
                if self.total_profit_today>=PROFIT_LOCK_TRIGGER and not self.profit_lock_active:
                    self.profit_lock_active=True
                    await self.safe_send_tg(f"🔒 PROFIT LOCK at +${self.total_profit_today:.2f} — floor +${PROFIT_LOCK_FLOOR:.2f}")
                if self.total_profit_today>=DAILY_PROFIT_TARGET:
                    self.pause_until=self._next_midnight_epoch()
            await self.fetch_balance()
            next_payout=money2(float(PAYOUT_TARGET)*(float(MARTINGALE_MULT)**int(self.martingale_step)))
            mkt_losses_after=self.market_losses_today.get(traded_symbol,0)
            mkt_blocked_after=self.market_blocked.get(traded_symbol,False)
            mkt_note=f"\n🚫 {traded_symbol.replace('_',' ')} BLOCKED for today" if mkt_blocked_after else f"\n📊 {traded_symbol.replace('_',' ')} losses today: {mkt_losses_after}/{MAX_LOSSES_PER_MARKET}"
            pause_note="\n⏸ Paused until 12:00am WAT" if time.time()<self.pause_until else ""
            halt_note="\n🛑 Martingale stopped" if self.martingale_halt else ""
            lock_note=f"\n🔒 Profit lock active (floor +${PROFIT_LOCK_FLOOR:.2f})" if self.profit_lock_active else ""
            await self.safe_send_tg(
                f"🏁 FINISH: {'WIN ✅' if profit>0 else 'LOSS ❌'} ({profit:+.2f})\n"
                f"📈 M5 Trend was: {m5_trend} | RSI: {rsi:.1f}\n"
                f"📊 Today: {self.trades_today}/{MAX_TRADES_PER_DAY} | Losses: {self.total_losses_today} | Streak: {self.consecutive_losses}/{MAX_CONSEC_LOSSES}\n"
                f"📌 Max streak: {self.max_loss_streak_today}\n"
                f"💵 Today PnL: {self.total_profit_today:+.2f}/+{DAILY_PROFIT_TARGET:.2f}\n"
                f"🎁 Next payout: ${next_payout:.2f} (step {self.martingale_step}/{MARTINGALE_MAX_STEPS})\n"
                f"💰 Balance: {self.balance}"
                f"{mkt_note}{lock_note}{pause_note}{halt_note}"
            )
            now_dt=datetime.now(self.tz)
            if now_dt.hour==23 and now_dt.minute>=55:
                await self._send_daily_report()
        finally:
            self.active_trade_info=None; self.active_trade_meta=None
            self.cooldown_until=time.time()+COOLDOWN_SEC

    async def _send_daily_report(self):
        by_mkt,by_sess,wr=self.stats_30d()
        lines=["📊 DAILY REPORT — VWAP Bot\n"]
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


# ===== TELEGRAM COMMANDS =====
async def cmd_setstake(u:Update,c:ContextTypes.DEFAULT_TYPE):
    global PAYOUT_TARGET
    try:
        val=float(c.args[0]); PAYOUT_TARGET=val
        await u.message.reply_text(f"✅ Base payout set to ${val:.2f}")
    except: await u.message.reply_text("Usage: /setstake 1.00")

async def cmd_setmarkets(u:Update,c:ContextTypes.DEFAULT_TYPE):
    global MARKETS
    valid=["R_10","R_25","R_50","R_75","R_100"]
    try:
        new_markets=[m.upper().replace("R","R_") if not m.startswith("R_") else m.upper() for m in c.args]
        new_markets=[m for m in new_markets if m in valid]
        if not new_markets:
            await u.message.reply_text(f"No valid markets. Use: {', '.join(valid)}"); return
        MARKETS=new_markets
        await u.message.reply_text(f"✅ Markets: {', '.join(MARKETS)}")
    except: await u.message.reply_text("Usage: /setmarkets R_50 R_75")

async def cmd_pause(u:Update,c:ContextTypes.DEFAULT_TYPE):
    try:
        minutes=int(c.args[0]); bot_logic.pause_until=time.time()+minutes*60
        await u.message.reply_text(f"⏸ Bot paused for {minutes} minutes.")
    except: await u.message.reply_text("Usage: /pause 30")

async def cmd_resume(u:Update,c:ContextTypes.DEFAULT_TYPE):
    bot_logic.pause_until=0.0
    for m in MARKETS:
        bot_logic.market_pause_until[m]=0.0
        bot_logic.market_chop_until[m]=0.0
    await u.message.reply_text("▶️ Bot and all markets resumed.")

async def cmd_stats(u:Update,c:ContextTypes.DEFAULT_TYPE):
    await bot_logic._send_daily_report()

async def cmd_unblock(u:Update,c:ContextTypes.DEFAULT_TYPE):
    for m in MARKETS:
        bot_logic.market_blocked[m]=False
        bot_logic.market_losses_today[m]=0
        bot_logic.market_pause_until[m]=0.0
    await u.message.reply_text("✅ All markets unblocked.")


# ===== UI =====
bot_logic=DerivVWAPBot()

def main_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("▶️ START",callback_data="START_SCAN"),InlineKeyboardButton("⏹️ STOP",callback_data="STOP_SCAN")],
        [InlineKeyboardButton("📊 STATUS",callback_data="STATUS"),InlineKeyboardButton("🔄 REFRESH",callback_data="STATUS")],
        [InlineKeyboardButton("🧪 TEST BUY",callback_data="TEST_BUY")],
        [InlineKeyboardButton("🧪 DEMO",callback_data="SET_DEMO"),InlineKeyboardButton("💰 LIVE",callback_data="SET_REAL")],
    ])

def format_market_detail(sym,d):
    if not d: return f"📍 {sym.replace('_',' ')}\n⏳ No scan data yet"
    age=int(time.time()-d.get("time",time.time()))
    signal=d.get("signal") or "—"
    why=d.get("why",[]); gate=d.get("gate","—"); mkt_msg=d.get("mkt_msg","OK")
    last_closed=d.get("last_closed",0)
    m5_trend=d.get("m5_trend","—")
    m1_uptrend=d.get("m1_uptrend",False); m1_downtrend=d.get("m1_downtrend",False)
    m1_trend_str="↑ UP" if m1_uptrend else ("↓ DOWN" if m1_downtrend else "— FLAT")
    adx=d.get("adx",0); adx_ok=d.get("adx_ok",False)
    ema50_slope=d.get("ema50_slope",0)
    ema50_rising=d.get("ema50_rising",False); ema50_falling=d.get("ema50_falling",False)
    slope_str="↑ Rising" if ema50_rising else ("↓ Falling" if ema50_falling else "→ Flat")
    near_ema20=d.get("near_ema20",False)
    body_ratio=d.get("body_ratio",0); chop=d.get("chop",False)
    m1_rsi=d.get("m1_rsi","—")
    mkt_losses=d.get("mkt_losses",0); mkt_trades=d.get("mkt_trades",0)
    return (
        f"📍 {sym.replace('_',' ')} ({age}s ago)\n"
        f"Market: {mkt_msg} | {mkt_trades}/{MAX_TRADES_PER_MARKET} | {mkt_losses}/{MAX_LOSSES_PER_MARKET} losses\n"
        f"Last M1: {fmt_time_hhmmss(last_closed)}\n"
        f"────────────────\n"
        f"📈 M5: {m5_trend} | M1: {m1_trend_str}\n"
        f"📊 ADX: {adx:.1f} {'✅' if adx_ok else '❌'}\n"
        f"📉 EMA50 slope: {slope_str} ({ema50_slope:.4f})\n"
        f"📐 EMA20 pullback: {'✅' if near_ema20 else '❌'}\n"
        f"🕯 Body: {body_ratio:.2f} | Chop: {'⚠️' if chop else '✅'}\n"
        f"📉 RSI: {m1_rsi}\n"
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
        await _safe_edit(q,"🔍 SCANNER ACTIVE\n✅ VWAP Pullback strategy running.\n📡 All 5 markets active.",reply_markup=main_keyboard())
    elif q.data=="STOP_SCAN":
        bot_logic.is_scanning=False
        if bot_logic.scanner_task and not bot_logic.scanner_task.done(): bot_logic.scanner_task.cancel()
        await _safe_edit(q,"⏹️ Scanner stopped.",reply_markup=main_keyboard())
    elif q.data=="TEST_BUY":
        test_symbol=MARKETS[0] if MARKETS else "R_50"
        asyncio.create_task(bot_logic.execute_trade("CALL",test_symbol,"Manual Test",source="MANUAL"))
        await _safe_edit(q,f"🧪 Test CALL on {test_symbol.replace('_',' ')}.",reply_markup=main_keyboard())
    elif q.data=="STATUS":
        now=time.time()
        if now<bot_logic.status_cooldown_until:
            await _safe_edit(q,f"⏳ Cooldown {int(bot_logic.status_cooldown_until-now)}s",reply_markup=main_keyboard()); return
        bot_logic.status_cooldown_until=now+STATUS_REFRESH_COOLDOWN_SEC

        # Fast balance fetch with short timeout
        try:
            await asyncio.wait_for(bot_logic.fetch_balance(),timeout=3.0)
        except: pass

        now_time=datetime.now(ZoneInfo("Africa/Lagos")).strftime("%Y-%m-%d %H:%M:%S")
        _ok,gate=bot_logic.can_auto_trade()

        # Use cached trade info — no live API call to avoid timeout
        trade_status="No Active Trade"
        if bot_logic.active_trade_info:
            rem=max(0,int(EXPIRY_MIN*60)-int(time.time()-bot_logic.trade_start_time))
            trade_status=f"🚀 Active Trade ({bot_logic.active_market.replace('_',' ')})\n🕓 Session: {session_bucket(bot_logic.trade_start_time)}\n⏳ Left: ~{rem}s"

        pause_line="⏸ Paused until 12:00am WAT\n" if time.time()<bot_logic.pause_until else ""
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
            ml=bot_logic.market_losses_today.get(m,0)
            mt=bot_logic.market_trades_today.get(m,0)
            mb=bot_logic.market_blocked.get(m,False)
            mp=bot_logic.market_pause_until.get(m,0)
            mc=bot_logic.market_chop_until.get(m,0)
            status="🚫 BLOCKED" if mb else ("⏸ PAUSED" if time.time()<mp else ("🌀 CHOP" if time.time()<mc else "✅"))
            mkt_lines.append(f"{status} {m.replace('_',' ')}: {mt}/{MAX_TRADES_PER_MARKET} | {ml}/{MAX_LOSSES_PER_MARKET} losses")
        mkt_block="\n".join(mkt_lines)+"\n"

        _,_,stake_mult=bot_logic._equity_ok()
        eq_ratio=bot_logic._get_current_balance_float()/bot_logic.starting_balance if bot_logic.starting_balance>0 else 1.0

        header=(
            f"🕒 {now_time}\n"
            f"🤖 {'ACTIVE' if bot_logic.is_scanning else 'OFFLINE'} ({bot_logic.account_type})\n"
            f"{pause_line}"
            f"🎁 Next payout: ${next_payout:.2f} | Step: {bot_logic.martingale_step}/{MARTINGALE_MAX_STEPS}\n"
            f"🧯 Max stake: ${MAX_STAKE_ALLOWED:.2f} | Mult: {stake_mult:.1f}x\n"
            f"💰 Equity: {eq_ratio:.0%} | 🔒 Lock: {'ON +${:.2f}'.format(PROFIT_LOCK_FLOOR) if bot_logic.profit_lock_active else 'OFF'}\n"
            f"🎯 Target: +${DAILY_PROFIT_TARGET:.2f} | Limit: ${DAILY_LOSS_LIMIT:.2f}\n"
            f"📡 Markets: {', '.join(m.replace('_',' ') for m in MARKETS)}\n"
            f"🧭 EMA Pullback+VWAP | ADX | M5+M1 | {EXPIRY_MIN}m expiry\n"
            f"━━━━━━━━━━━━━━━\n{trade_status}\n━━━━━━━━━━━━━━━\n"
            f"{stats_block}{mkt_block}"
            f"💵 PnL: {bot_logic.total_profit_today:+.2f} | Trades: {bot_logic.trades_today}/{MAX_TRADES_PER_DAY}\n"
            f"📉 Streak: {bot_logic.consecutive_losses}/{MAX_CONSEC_LOSSES} | Losses: {bot_logic.total_losses_today}\n"
            f"🚦 Gate: {gate}\n"
            f"💰 Balance: {bot_logic.balance}\n"
            f"\n/setstake /setmarkets /pause /resume /stats /unblock"
        )
        details="\n\n📌 LIVE SCAN\n\n"+"\n\n".join([format_market_detail(sym,bot_logic.market_debug.get(sym,{})) for sym in MARKETS])
        await _safe_edit(q,header+details,reply_markup=main_keyboard())

async def start_cmd(u:Update,c:ContextTypes.DEFAULT_TYPE):
    await u.message.reply_text(
        "💎 Deriv EMA + VWAP Strategy Bot\n"
        f"🧭 EMA Pullback + VWAP Confluence (Relaxed)\n"
        f"📐 M5 trend (VWAP+EMA50) + M1 entry (EMA20+EMA50+ADX+RSI)\n"
        f"✅ Never relaxed: M5 trend | EMA crossover | ADX | Pullback zone\n"
        f"⚡ Relaxed: EMA slope | RSI range | Body ratio\n"
        f"🛡 ADX≥20 | Spike blocker | Chop filter | Per-market blocking\n"
        f"💰 Equity protection | Profit lock\n"
        f"🎲 Martingale: {MARTINGALE_MAX_STEPS} steps × {MARTINGALE_MULT}x\n"
        f"⏱ Expiry: {EXPIRY_MIN}m | Cooldown: {COOLDOWN_SEC//60}m\n"
        f"📡 Markets: R10, R25, R50, R75, R100\n"
        f"📲 /setstake /setmarkets /pause /resume /stats /unblock\n",
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
    app.add_handler(CommandHandler("unblock",cmd_unblock))
    app.run_polling(drop_pending_updates=True)
