# ⚠️ SECURITY NOTE:
# Do NOT hardcode tokens in public code.
# Paste tokens only on your local machine.
# If you ever leaked tokens, ROTATE them immediately.

import asyncio
import logging
import random
import time
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

MARKETS = ["R_10", "R_25"]

TELEGRAM_TOKEN = "8253450930:AAHUhPk9TML-8kZlA9UaHZZvTUGdurN9MVY"
TELEGRAM_CHAT_ID = "7634818949"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ========================= MODES =========================
MODE_CONFIG = {
    "M5": {"TF_SEC": 300, "DURATION_MIN": 20},
    "M1": {"TF_SEC": 60,  "DURATION_MIN": 5},
}
DEFAULT_MODE = "M5"
CANDLES_COUNT = 220

# ========================= RISK =========================
COOLDOWN_SEC = 180
MAX_TRADES_PER_DAY = 20
MAX_CONSEC_LOSSES = 6
STOP_AFTER_LOSSES = 3
DAILY_LOSS_LIMIT = -2.0
DAILY_PROFIT_TARGET = 2.0

# ========================= PAYOUT =========================
PAYOUT_TARGET = 1.0
MIN_PAYOUT = 0.35
MAX_STAKE_ALLOWED = 10.0

# ========================= MARTINGALE =========================
MARTINGALE_MULT = 1.80
MARTINGALE_MAX_STEPS = 3
MARTINGALE_HALT_ON_MAX = True

# ========================= BREAKOUT =========================
ATR_PERIOD = 14
DONCHIAN_LEN = 20
ATR_BREAKOUT_K = 0.15

USE_EMA_BIAS = True
EMA_FAST = 20
EMA_SLOW = 50

MIN_CANDLE_RANGE = 1e-6
SPIKE_RANGE_ATR = 2.5
SPIKE_BODY_ATR = 1.8

# ========================= UI =========================
STATUS_REFRESH_COOLDOWN_SEC = 8

# ========================= HELPERS =========================
def calculate_ema(values, period):
    values = np.array(values, dtype=float)
    if len(values) < period:
        return np.array([])
    k = 2 / (period + 1)
    ema = np.zeros_like(values)
    ema[0] = values[0]
    for i in range(1, len(values)):
        ema[i] = values[i] * k + ema[i - 1] * (1 - k)
    return ema

def calculate_atr(highs, lows, closes, period):
    highs, lows, closes = map(lambda x: np.array(x, dtype=float), (highs, lows, closes))
    tr = np.maximum(highs - lows, np.maximum(abs(highs - np.roll(closes, 1)), abs(lows - np.roll(closes, 1))))
    atr = np.full_like(closes, np.nan)
    atr[period] = np.mean(tr[1:period+1])
    for i in range(period+1, len(closes)):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    return atr

def build_candles_from_deriv(raw):
    return [{"t0": int(x["epoch"]), "o": float(x["open"]), "h": float(x["high"]), "l": float(x["low"]), "c": float(x["close"])} for x in raw]

def fmt_time(epoch):
    return datetime.fromtimestamp(epoch, ZoneInfo("Africa/Lagos")).strftime("%H:%M:%S") if epoch else "—"

def is_finite(x):
    return isinstance(x, (int, float)) and np.isfinite(x)

# ========================= BOT =========================
class DerivBreakoutBot:
    def __init__(self):
        self.api = None
        self.app = None
        self.active_token = None
        self.account_type = "None"
        self.mode = DEFAULT_MODE
        self.is_scanning = False
        self.scanner_task = None
        self.active_trade = None
        self.market_debug = {(m, md): {} for m in MARKETS for md in MODE_CONFIG}
        self.last_processed_closed_t0 = {(m, md): 0 for m in MARKETS for md in MODE_CONFIG}
        self.cooldown_until = 0
        self.martingale_step = 0

    # ---------- MODE ----------
    def toggle_mode(self):
        self.mode = "M1" if self.mode == "M5" else "M5"

    def get_mode_cfg(self):
        return MODE_CONFIG[self.mode]

    # ---------- STATUS ----------
    def can_auto_trade(self):
        if self.active_trade:
            return False, "Trade running"
        if time.time() < self.cooldown_until:
            return False, "Cooldown"
        return True, "OK"

    # ---------- SCAN ----------
    async def scan_market(self, symbol):
        while self.is_scanning:
            for md, cfg in MODE_CONFIG.items():
                candles = await self.fetch_candles(symbol, cfg["TF_SEC"])
                if len(candles) < DONCHIAN_LEN + 5:
                    continue

                confirm = candles[-2]
                closes = [x["c"] for x in candles]
                highs = [x["h"] for x in candles]
                lows = [x["l"] for x in candles]

                atr = calculate_atr(highs, lows, closes, ATR_PERIOD)
                atr_now = atr[-2] if is_finite(atr[-2]) else None

                window_high = max(highs[-(DONCHIAN_LEN+2):-2])
                window_low = min(lows[-(DONCHIAN_LEN+2):-2])

                buf = atr_now * ATR_BREAKOUT_K if atr_now else 0
                call_lvl = window_high + buf
                put_lvl = window_low - buf

                breakout_call = confirm["c"] > call_lvl
                breakout_put = confirm["c"] < put_lvl

                last_price = await self.fetch_last_price(symbol)

                checks = {
                    "BREAKOUT": breakout_call or breakout_put,
                    "BIAS": True,
                    "SPIKE": True,
                }

                waiting = "Ready to trade"
                if checks["BREAKOUT"]:
                    if USE_EMA_BIAS and not checks["BIAS"]:
                        waiting = "Waiting for EMA bias"
                else:
                    waiting = "Waiting for breakout"

                self.market_debug[(symbol, md)] = {
                    "breakout": checks["BREAKOUT"],
                    "approaching": abs(last_price - call_lvl) < atr_now*0.2 if atr_now else False,
                    "waiting": waiting,
                    "mode": md,
                }

            await asyncio.sleep(1)

    async def fetch_candles(self, symbol, tf):
        data = await self.api.ticks_history({"ticks_history": symbol, "count": CANDLES_COUNT, "granularity": tf, "style": "candles"})
        return build_candles_from_deriv(data["candles"])

    async def fetch_last_price(self, symbol):
        data = await self.api.ticks_history({"ticks_history": symbol, "count": 1})
        return float(data["history"]["prices"][-1])

# ========================= UI HELPERS =========================
bot_logic = DerivBreakoutBot()

def format_market_detail(sym, d):
    if not d:
        return f"{sym}: No data\n"

    return (
        f"📍 {sym.replace('_',' ')} | {d['mode']}\n"
        f"Breakout confirmed: {'YES' if d['breakout'] else 'NO'}\n"
        f"Approaching breakout: {'YES' if d['approaching'] else 'NO'}\n"
        f"Waiting for: {d['waiting']}\n"
    )

# ========================= TELEGRAM =========================
def main_keyboard():
    return InlineKeyboardMarkup([[InlineKeyboardButton("📊 STATUS", callback_data="STATUS")]])

async def btn_handler(u: Update, c: ContextTypes.DEFAULT_TYPE):
    q = u.callback_query
    await q.answer()
    text = "📌 LIVE STATUS\n\n"
    for md in ("M1", "M5"):
        text += f"===== {md} =====\n"
        for sym in MARKETS:
            text += format_market_detail(sym, bot_logic.market_debug.get((sym, md)))
        text += "\n"
    await q.edit_message_text(text, reply_markup=main_keyboard())

async def start_cmd(u: Update, c: ContextTypes.DEFAULT_TYPE):
    await u.message.reply_text("Breakout bot running.", reply_markup=main_keyboard())

if __name__ == "__main__":
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    bot_logic.app = app
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CallbackQueryHandler(btn_handler))
    app.run_polling()
