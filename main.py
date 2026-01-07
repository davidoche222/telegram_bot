import asyncio
import datetime
import logging
import numpy as np
from scipy.signal import find_peaks

from pocketoptionapi_async import AsyncPocketOptionClient, OrderDirection

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# ========================= CONFIG =========================
SSID = "n5jlYj9XTDjpMpoH/A2DysApxaK_JCsxpm"

TELEGRAM_TOKEN = "8276370676:AAGh5VqkG7b4cvpfRIVwY_rtaBlIiNwCTDM"
TELEGRAM_CHAT_ID = "7634818949"  # your personal chat id

# 8 OTC assets
ASSETS = [
    "EURUSD_otc",
    "GBPUSD_otc",
    "AUDUSD_otc",
    "USDJPY_otc",
    "EURGBP_otc",
    "EURJPY_otc",
    "GBPJPY_otc",
    "AUDJPY_otc",
]

DEFAULT_TRADE_AMOUNT = 1.0   # can be changed from Telegram
MIN_EXPIRY = 180             # 3 minutes
MAX_EXPIRY = 300             # 5 minutes

MAX_TRADES_PER_DAY = 5
MAX_CONSECUTIVE_LOSSES = 2
# =========================================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ================ PURE PYTHON INDICATORS ==================
def ema_last(values: np.ndarray, period: int):
    if values is None or len(values) < period:
        return None
    k = 2 / (period + 1)
    ema_val = float(values[0])
    for price in values[1:]:
        ema_val = float(price) * k + ema_val * (1 - k)
    return ema_val


def rsi_last(values: np.ndarray, period: int = 14):
    if values is None or len(values) < period + 1:
        return None
    v = values[-(period + 1):]
    deltas = np.diff(v)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = float(np.mean(gains))
    avg_loss = float(np.mean(losses))
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def atr_last(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14):
    if closes is None or len(closes) < period + 1:
        return None
    h = highs[-(period + 1):]
    l = lows[-(period + 1):]
    c = closes[-(period + 1):]
    trs = []
    for i in range(1, len(c)):
        tr = max(
            float(h[i] - l[i]),
            abs(float(h[i] - c[i - 1])),
            abs(float(l[i] - c[i - 1])),
        )
        trs.append(tr)
    return float(np.mean(trs)) if trs else None
# ==========================================================


def _is_auth_error(e: Exception) -> bool:
    s = str(e).lower()
    return ("auth" in s) or ("unauthorized" in s) or ("forbidden" in s) or ("session" in s)


class TradingBot:
    def init(self):
        self.client: AsyncPocketOptionClient | None = None
        self.app: Application | None = None

        self.is_demo: bool | None = True  # default DEMO (safer)
        self.trade_amount: float = float(DEFAULT_TRADE_AMOUNT)

        self.daily_trades = 0
        self.consec_losses = 0
        self.daily_pnl = 0.0
        self.current_date = datetime.date.today()

        self.running = False
        self.paused = False

        self.last_trade_info = None
        self._ssid_alert_sent = False

        # One trade at a time -> keeps WIN/LOSS balance-delta accurate
        self.trade_lock = asyncio.Lock()

        # avoid duplicate triggers per asset at the same time
        self.asset_locks = {a: asyncio.Lock() for a in ASSETS}

        # background task handle
        self._task: asyncio.Task | None = None

        # ====== NEW: status info for "searching" ======
        self.last_scan_time: datetime.datetime | None = None
        self.last_signal_time: datetime.datetime | None = None
        self.active_trade: bool = False
        self.active_trade_asset: str | None = None
        # ============================================

+2349045188578, [12/25/2025 3:33 AM]
async def send(self, message: str):
        try:
            await self.app.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
        except Exception as e:
            logging.error(f"Telegram send failed: {e}")

    def reset_daily(self):
        today = datetime.date.today()
        if today != self.current_date:
            self.daily_trades = 0
            self.consec_losses = 0
            self.daily_pnl = 0.0
            self.current_date = today

    async def can_trade(self) -> bool:
        self.reset_daily()

        if not self.running or self.paused:
            return False

        if self.daily_trades >= MAX_TRADES_PER_DAY:
            await self.send("🛑 Max trades/day reached. Bot paused until tomorrow.")
            self.paused = True
            return False

        if self.consec_losses >= MAX_CONSECUTIVE_LOSSES:
            await self.send("🛑 Max consecutive losses reached. Bot paused.")
            self.paused = True
            return False

        return True

    def status_line(self) -> str:
        if not self.running:
            return "🛑 Stopped"
        if self.paused:
            return "⏸ Paused"
        if self.active_trade:
            return f"🎯 In trade… ({self.active_trade_asset})"
        return "🔎 Searching for signal…"

    async def connect(self) -> bool:
        global SSID
        try:
            if not SSID or "PASTE_" in SSID:
                await self.send("❌ SSID missing. Paste your SSID in main.py and redeploy.")
                return False

            self.client = AsyncPocketOptionClient(ssid=SSID, is_demo=self.is_demo)
            await self.client.connect()

            if not getattr(self.client, "connected", False):
                await self.send("❌ Connection failed. Check SSID and try again.")
                return False

            bal = await self.client.get_balance()
            mode = "DEMO" if self.is_demo else "LIVE"
            await self.send(f"🚀 Connected!\nMode: {mode}\nBalance: ${bal}\nMonitoring {len(ASSETS)} assets...")

            for asset in ASSETS:
                await self.client.subscribe_candles(asset, 300)

            self._ssid_alert_sent = False
            return True

        except Exception as e:
            if _is_auth_error(e):
                if not self._ssid_alert_sent:
                    await self.send("⚠️ SESSION (SSID) EXPIRED!\nUpdate SSID in main.py and redeploy.")
                    self._ssid_alert_sent = True
                self.paused = True
                self.running = False
                return False

            await self.send(f"⚠️ Connection error: {e}")
            return False

    async def get_candles(self, asset: str, timeframe: int, count: int = 220):
        try:
            candles = await self.client.get_candles(asset, timeframe, count)
            if not candles or len(candles) < 60:
                return None

            o = np.array([c["open"] for c in candles], dtype=float)
            h = np.array([c["high"] for c in candles], dtype=float)
            l = np.array([c["low"] for c in candles], dtype=float)
            c_ = np.array([c["close"] for c in candles], dtype=float)
            return {"o": o, "h": h, "l": l, "c": c_}
        except Exception:
            return None

    async def get_1h_bias(self, asset: str):
        data = await self.get_candles(asset, 3600, 260)
        if not data:
            return None

        o, h, l, c = data["o"], data["h"], data["l"], data["c"]

        ema200_now = ema_last(c, 200)
        ema200_prev = ema_last(c[:-1], 200) if len(c) > 201 else None
        if ema200_now is None or ema200_prev is None:
            return None

        slope = ema200_now - ema200_prev
        is_up_trend = slope > 0.0001
        is_down_trend = slope < -0.0001

        peaks, _ = find_peaks(h, distance=4)
        valleys, _ = find_peaks(-l, distance=4)
        if len(peaks) < 2 or len(valleys) < 2:
            return None

+2349045188578, [12/25/2025 3:33 AM]
bull_structure = h[peaks[-1]] > h[peaks[-2]] and l[valleys[-1]] > l[valleys[-2]]
        bear_structure = h[peaks[-1]] < h[peaks[-2]] and l[valleys[-1]] < l[valleys[-2]]

        rng = h[-1] - l[-1]
        if rng <= 0:
            return None

        last_bullish = c[-1] > o[-1] and abs(c[-1] - o[-1]) > 0.5 * rng
        last_bearish = c[-1] < o[-1] and abs(o[-1] - c[-1]) > 0.5 * rng

        if bull_structure and c[-1] > ema200_now and is_up_trend and last_bullish:
            return "buy"
        if bear_structure and c[-1] < ema200_now and is_down_trend and last_bearish:
            return "sell"
        return None

    async def check_entry(self, asset: str):
        # ===== NEW: record scan time for "searching" =====
        self.last_scan_time = datetime.datetime.utcnow()
        # =================================================

        bias = await self.get_1h_bias(asset)
        if not bias:
            return

        data_5m = await self.get_candles(asset, 300, 220)
        if not data_5m:
            return

        o, h, l, c = data_5m["o"], data_5m["h"], data_5m["l"], data_5m["c"]

        # zone detection (last 30 candles)
        valleys, _ = find_peaks(-l[-30:], distance=3)
        peaks, _ = find_peaks(h[-30:], distance=3)

        zone = None
        if bias == "buy" and len(valleys) > 0:
            zone = l[-30:][valleys[-1]]
        elif bias == "sell" and len(peaks) > 0:
            zone = h[-30:][peaks[-1]]
        if zone is None:
            return

        atr14 = atr_last(h, l, c, 14)
        if atr14 is None or atr14 <= 0:
            return

        tolerance = atr14 * 0.5
        near_zone = (min(l[-3:]) <= zone + tolerance) and (max(h[-3:]) >= zone - tolerance)
        if not near_zone:
            return

        # triggers
        strength = "moderate"
        body = abs(c[-1] - o[-1])
        prev_body = abs(c[-2] - o[-2])

        is_engulfing = is_pinbar = is_micro_bos = False

        if bias == "buy":
            if c[-1] > o[-1] and c[-1] > h[-2] and o[-1] < c[-2]:
                is_engulfing = True
                strength = "strong"

            lower_wick = min(o[-1], c[-1]) - l[-1]
            if body > 0 and lower_wick >= 2 * body and c[-1] > (h[-1] + l[-1]) / 2:
                is_pinbar = True

            if len(c) >= 3 and c[-2] < o[-2] and c[-1] > h[-2]:
                is_micro_bos = True
                strength = "strong"
        else:
            if c[-1] < o[-1] and c[-1] < l[-2] and o[-1] > c[-2]:
                is_engulfing = True
                strength = "strong"

            upper_wick = h[-1] - max(o[-1], c[-1])
            if body > 0 and upper_wick >= 2 * body and c[-1] < (h[-1] + l[-1]) / 2:
                is_pinbar = True

            if len(c) >= 3 and c[-2] > o[-2] and c[-1] < l[-2]:
                is_micro_bos = True
                strength = "strong"

        if not (is_engulfing or is_pinbar or is_micro_bos):
            return

        ema200_5m = ema_last(c, 200)
        rsi14 = rsi_last(c, 14)
        if ema200_5m is None or rsi14 is None:
            return

        if bias == "buy":
            if c[-1] <= ema200_5m or not (45 <= rsi14 <= 65):
                return
        else:
            if c[-1] >= ema200_5m or not (35 <= rsi14 <= 55):
                return

        if body < 0.7 * atr14 or body <= prev_body:
            return

        # ===== NEW: record that a valid signal happened =====
        self.last_signal_time = datetime.datetime.utcnow()
        # ====================================================

        # place trade (one at a time for correct win/loss)
        async with self.trade_lock:
            if not await self.can_trade():
                return

            try:
                balance_before = await self.client.get_balance()

+2349045188578, [12/25/2025 3:33 AM]
except Exception as e:
                if _is_auth_error(e):
                    if not self._ssid_alert_sent:
                        await self.send("⚠️ SESSION (SSID) EXPIRED!\nUpdate SSID and redeploy.")
                        self._ssid_alert_sent = True
                    self.paused = True
                    self.running = False
                return

            direction = OrderDirection.CALL if bias == "buy" else OrderDirection.PUT
            expiry = MIN_EXPIRY if strength == "strong" else MAX_EXPIRY

            order = await self.client.place_order(
                asset=asset,
                amount=float(self.trade_amount),
                direction=direction,
                duration=expiry,
            )

            if order and order.get("status") == "success":
                self.daily_trades += 1
                self.last_trade_info = f"{asset} | {'CALL' if bias=='buy' else 'PUT'} | {expiry//60}m | ${self.trade_amount:.2f}"

                # ===== NEW: mark active trade for status =====
                self.active_trade = True
                self.active_trade_asset = asset
                # ===========================================

                await self.send(
                    f"{'🟢 CALL' if bias=='buy' else '🔴 PUT'} on {asset}\n"
                    f"Stake: ${self.trade_amount:.2f} | Expiry: {expiry//60} min\n"
                    f"Strength: {strength.upper()}\n"
                    f"📌 Checking result after expiry..."
                )

                asyncio.create_task(self._check_trade_result(asset, expiry, balance_before))
            else:
                await self.send(f"❌ Trade failed on {asset}")

    async def _check_trade_result(self, asset: str, expiry: int, balance_before: float):
        try:
            await asyncio.sleep(expiry + 5)
            balance_after = await self.client.get_balance()
            diff = float(balance_after) - float(balance_before)

            self.daily_pnl += diff

            if diff > 0:
                self.consec_losses = 0
                await self.send(f"✅ WIN on {asset}\nProfit: ${diff:.2f}\nBalance: ${balance_after:.2f}")
            else:
                self.consec_losses += 1
                await self.send(f"❌ LOSS on {asset}\nPnL: ${diff:.2f}\nBalance: ${balance_after:.2f}")

        except Exception as e:
            if _is_auth_error(e):
                if not self._ssid_alert_sent:
                    await self.send("⚠️ SESSION (SSID) EXPIRED!\nUpdate SSID and redeploy.")
                    self._ssid_alert_sent = True
                self.paused = True
                self.running = False
            else:
                await self.send(f"⚠️ Could not confirm result for {asset}: {e}")
        finally:
            # ===== NEW: clear active trade for status =====
            self.active_trade = False
            self.active_trade_asset = None
            # ===========================================

    async def on_candle_update(self, candle, asset: str):
        try:
            if not self.running or self.paused:
                return
            if self.client and getattr(self.client, "connected", False):
                async with self.asset_locks[asset]:
                    await self.check_entry(asset)
        except Exception as e:
            logging.error(f"Runtime error ({asset}): {e}")

    async def _runner(self):
        ok = await self.connect()
        if not ok:
            self.running = False
            return

        for asset in ASSETS:
            self.client.add_candle_handler(
                asset,
                lambda c, a=asset: asyncio.create_task(self.on_candle_update(c, a))
            )

        await self.send("🤖 Trading started. Monitoring market...")

        while self.running:
            await asyncio.sleep(30)

        await self.send("🛑 Trading stopped.")

    def start(self):
        if self._task and not self._task.done():
            return
        self.running = True
        self.paused = False
        self._task = asyncio.create_task(self._runner())

+2349045188578, [12/25/2025 3:33 AM]
def stop(self):
        self.running = False
        self.paused = True


# ==================== TELEGRAM UI ====================
bot_instance = TradingBot()


def settings_keyboard() -> InlineKeyboardMarkup:
    buttons = [
        [InlineKeyboardButton("▶️ Start Trading", callback_data="SET_START")],
        [InlineKeyboardButton("⏸ Stop Trading", callback_data="SET_STOP")],
        [InlineKeyboardButton("💰 Set Stake", callback_data="SET_STAKE")],
        [
            InlineKeyboardButton("✅ Demo", callback_data="SET_DEMO"),
            InlineKeyboardButton("⚠️ Live", callback_data="SET_LIVE"),
        ],
        [InlineKeyboardButton("📊 Status", callback_data="SET_STATUS")],
    ]
    return InlineKeyboardMarkup(buttons)


async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 Bot Menu\n\n"
        "Commands:\n"
        "/run - start trading\n"
        "/stop - stop trading\n"
        "/settings - settings menu\n"
        "/status - show status\n"
        "/stake 2 - set stake (quick)\n",
        reply_markup=settings_keyboard(),
    )


async def run_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    bot_instance.start()
    await update.message.reply_text("▶️ Trading started.", reply_markup=settings_keyboard())


async def stop_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    bot_instance.stop()
    await update.message.reply_text("⏸ Trading stopped.", reply_markup=settings_keyboard())


def _fmt_time(dt: datetime.datetime | None) -> str:
    if not dt:
        return "None"
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")


async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    mode = "DEMO" if bot_instance.is_demo else "LIVE"
    await update.message.reply_text(
        f"📊 Status\n"
        f"{bot_instance.status_line()}\n\n"
        f"Mode: {mode}\n"
        f"Pairs: {len(ASSETS)}\n"
        f"Stake: ${bot_instance.trade_amount:.2f}\n"
        f"Trades today: {bot_instance.daily_trades}/{MAX_TRADES_PER_DAY}\n"
        f"Consecutive losses: {bot_instance.consec_losses}/{MAX_CONSECUTIVE_LOSSES}\n"
        f"Daily PnL: ${bot_instance.daily_pnl:.2f}\n"
        f"Last scan: {_fmt_time(bot_instance.last_scan_time)}\n"
        f"Last signal: {_fmt_time(bot_instance.last_signal_time)}\n"
        f"Last trade: {bot_instance.last_trade_info or 'None'}"
    )


async def settings_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("⚙️ Settings", reply_markup=settings_keyboard())


async def stake_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if not context.args:
            await update.message.reply_text("Usage: /stake 2  (example: /stake 1.5)")
            return

        amt = float(context.args[0])
        if amt <= 0:
            raise ValueError

        bot_instance.trade_amount = amt
        await update.message.reply_text(f"✅ Stake set to ${amt:.2f}", reply_markup=settings_keyboard())
    except Exception:
        await update.message.reply_text("❌ Invalid amount. Example: /stake 2")


async def on_settings_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data

    if data == "SET_START":
        bot_instance.start()
        await query.edit_message_text("▶️ Trading started.", reply_markup=settings_keyboard())

    elif data == "SET_STOP":
        bot_instance.stop()
        await query.edit_message_text("⏸ Trading stopped.", reply_markup=settings_keyboard())

    elif data == "SET_DEMO":
        bot_instance.is_demo = True
        await query.edit_message_text("✅ Mode set to DEMO.", reply_markup=settings_keyboard())

    elif data == "SET_LIVE":
        bot_instance.is_demo = False
        await query.edit_message_text("⚠️ Mode set to LIVE (real money).", reply_markup=settings_keyboard())

+2349045188578, [12/25/2025 3:33 AM]
elif data == "SET_STATUS":
        mode = "DEMO" if bot_instance.is_demo else "LIVE"
        await query.edit_message_text(
            f"📊 Status\n"
            f"{bot_instance.status_line()}\n\n"
            f"Mode: {mode}\n"
            f"Pairs: {len(ASSETS)}\n"
            f"Stake: ${bot_instance.trade_amount:.2f}\n"
            f"Trades today: {bot_instance.daily_trades}/{MAX_TRADES_PER_DAY}\n"
            f"Consecutive losses: {bot_instance.consec_losses}/{MAX_CONSECUTIVE_LOSSES}\n"
            f"Daily PnL: ${bot_instance.daily_pnl:.2f}\n"
            f"Last scan: {_fmt_time(bot_instance.last_scan_time)}\n"
            f"Last signal: {_fmt_time(bot_instance.last_signal_time)}\n"
            f"Last trade: {bot_instance.last_trade_info or 'None'}",
            reply_markup=settings_keyboard(),
        )

    elif data == "SET_STAKE":
        context.user_data["awaiting_stake"] = True
        await query.edit_message_text(
            "💰 Send the stake amount as a number (example: 1 or 2.5).\n\n"
            "Tip: You can also use /stake 2",
            reply_markup=settings_keyboard(),
        )


async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.user_data.get("awaiting_stake"):
        return

    text = (update.message.text or "").strip()
    try:
        amt = float(text)
        if amt <= 0:
            raise ValueError

        bot_instance.trade_amount = amt
        context.user_data["awaiting_stake"] = False
        await update.message.reply_text(f"✅ Stake set to ${amt:.2f}", reply_markup=settings_keyboard())
    except Exception:
        await update.message.reply_text("❌ Invalid amount. Send a number like 1 or 2.5")


def main():
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    bot_instance.app = application

    application.add_handler(CommandHandler("start", start_cmd))
    application.add_handler(CommandHandler("run", run_cmd))
    application.add_handler(CommandHandler("stop", stop_cmd))
    application.add_handler(CommandHandler("settings", settings_cmd))
    application.add_handler(CommandHandler("status", status_cmd))
    application.add_handler(CommandHandler("stake", stake_cmd))

    application.add_handler(CallbackQueryHandler(on_settings_button))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))

    application.run_polling()


if name == "main":
    main()
