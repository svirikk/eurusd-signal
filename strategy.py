"""
Trading Strategy Implementation
Asia Range Sweep → CHOCH → FVG Entry Strategy
"""
import logging
from typing import List, Optional, Dict
from datetime import datetime, timedelta
from bingx_client import Candle, BingXClient
from config_bingx import Config
import utils


logger = logging.getLogger(__name__)


class AsiaRange:
    """Represents Asia session range"""
    
    def __init__(self, high: float, low: float, date: datetime):
        self.high = high
        self.low = low
        self.mid = (high + low) / 2
        self.date = date
        
    def __repr__(self):
        return f"AsiaRange(H:{self.high:.5f}, L:{self.low:.5f}, Mid:{self.mid:.5f})"


class TradingSignal:
    """Represents a trading signal"""
    
    def __init__(self, signal_type: str, entry: float, stop_loss: float, 
                 take_profit: float, context: Dict):
        self.type = signal_type  # 'BUY' or 'SELL'
        self.entry = entry
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.context = context
        self.timestamp = datetime.utcnow()
        
        # Calculate risk/reward
        self.risk_pips = abs(entry - stop_loss) * 10000
        self.reward_pips = abs(take_profit - entry) * 10000
        self.risk_reward = self.reward_pips / self.risk_pips if self.risk_pips > 0 else 0
        
        # Generate unique hash
        self.hash = utils.generate_signal_hash(signal_type, entry, self.timestamp)
    
    def is_valid(self) -> bool:
        """Check if signal meets minimum requirements"""
        if self.risk_reward < Config.MIN_RISK_REWARD:
            logger.warning(f"❌ Signal RR too low: {self.risk_reward:.2f}")
            return False
        
        if self.risk_pips <= 0 or self.reward_pips <= 0:
            logger.warning(f"❌ Invalid risk/reward: R={self.risk_pips:.1f}, RW={self.reward_pips:.1f}")
            return False
        
        return True


class StrategyEngine:
    """Main strategy engine"""
    
    def __init__(self, bingx_client: BingXClient):
        self.client = bingx_client
        self.m5_candles: List[Candle] = []
        self.m15_candles: List[Candle] = []
        self.asia_range: Optional[AsiaRange] = None
        self.last_signal_time: Optional[datetime] = None
        self.sent_signals: set = set()  # Track sent signal hashes
        
    def update_candles(self):
        """Fetch and update candle data"""
        logger.info("📊 Updating candles...")
        
        # Fetch M5 candles
        m5 = self.client.get_latest_candles(Config.TF_ENTRY, Config.MAX_CANDLES)
        if m5:
            self.m5_candles = m5
            logger.info(f"✅ M5 candles updated: {len(self.m5_candles)} candles")
        
        # Fetch M15 candles
        m15 = self.client.get_latest_candles(Config.TF_CONTEXT, Config.MAX_CANDLES)
        if m15:
            self.m15_candles = m15
            logger.info(f"✅ M15 candles updated: {len(self.m15_candles)} candles")
    
    def compute_asia_range(self) -> Optional[AsiaRange]:
        """
        Compute Asia range (23:00 UTC prev day - 07:00 UTC today)
        ICT Asia session is OVERNIGHT!
        """
        if not self.m5_candles:
            return None
        
        now = datetime.utcnow()
        yesterday = now - timedelta(days=1)
        
        # Cache check
        if self.asia_range and self.asia_range.date.date() == now.date():
            logger.debug(f"📊 Using cached Asia Range: {self.asia_range}")
            return self.asia_range
        
        logger.debug(f"🔍 Computing Asia Range (23:00 UTC {yesterday.strftime('%m-%d')} → 07:00 UTC {now.strftime('%m-%d')})...")
        
        # Asia is OVERNIGHT: 23:00 yesterday + 00:00-07:00 today
        asia_candles = [
            c for c in self.m5_candles
            if (
                # Вчора з 23:00 UTC
                (c.time.date() == yesterday.date() and c.time.hour >= Config.ASIA_SESSION_START) or
                # Сьогодні до 07:00 UTC
                (c.time.date() == now.date() and c.time.hour < Config.ASIA_SESSION_END)
            )
        ]
        
        if not asia_candles:
            logger.warning("⚠️ No Asia candles found (23:00-07:00 UTC)")
            return None
        
        logger.debug(f"✅ Found {len(asia_candles)} Asia candles: {asia_candles[0].time.strftime('%H:%M')} → {asia_candles[-1].time.strftime('%H:%M')} UTC")
        
        # Calculate high and low
        asia_high = max(c.high for c in asia_candles)
        asia_low = min(c.low for c in asia_candles)
        range_pips = (asia_high - asia_low) * 10000
        
        # ⚠️ МІНІМАЛЬНИЙ РОЗМІР RANGE - якщо менше порогу, дані погані або holiday
        if range_pips < Config.ASIA_MIN_RANGE_PIPS:
            logger.warning(f"⚠️ Asia Range занадто малий: {range_pips:.1f} pips (мін: {Config.ASIA_MIN_RANGE_PIPS} pips)")
            logger.warning(f"   Мабуть holiday session або погані дані BingX. Пропускаємо!")
            return None
        
        self.asia_range = AsiaRange(asia_high, asia_low, asia_candles[0].time)
        logger.info(f"✅ Asia Range computed: {self.asia_range}")
        logger.info(f"   Range size: {range_pips:.1f} pips")
        
        return self.asia_range
    
    def detect_sweep(self) -> Optional[Dict]:
        """
        Detect liquidity sweep of Asia range during Frankfurt/London session
        NOW LOGS ALL SWEEPS FOUND!
        
        Returns:
            Dict with sweep details or None
        """
        if not self.asia_range or not self.m5_candles:
            return None
        
        # ✅ Перевіряємо Франкфурт + Лондон (07:00-13:00 UTC)
        recent_candles = [
            c for c in self.m5_candles[-80:]
            if utils.is_time_in_session(c.time, Config.SWEEP_WINDOW_START, Config.SWEEP_WINDOW_END)
        ]
        
        if not recent_candles:
            logger.debug("⏳ Outside Frankfurt/London window (07:00-13:00 UTC)")
            return None
        
        # Логуємо поточну сесію
        current_hour = datetime.utcnow().hour
        if Config.FRANKFURT_SESSION_START <= current_hour < Config.FRANKFURT_SESSION_END:
            logger.debug(f"📍 FRANKFURT session active (07:00-08:00 UTC)")
        elif Config.LONDON_SESSION_START <= current_hour < Config.LONDON_SESSION_END:
            logger.debug(f"📍 LONDON session active (08:00-13:00 UTC)")
        
        sweep_threshold = self.client.pips_to_price(Config.SWEEP_PIPS_THRESHOLD)
        bearish_target = self.asia_range.high + sweep_threshold
        bullish_target = self.asia_range.low - sweep_threshold
        
        # 🔍 COLLECT ALL SWEEPS (both bearish and bullish)
        all_sweeps = []
        
        for i in range(len(recent_candles) - 1, -1, -1):
            candle = recent_candles[i]
            
            # Check for bearish sweep (SELL)
            if candle.high > bearish_target and candle.close < self.asia_range.high:
                pips_swept = (candle.high - self.asia_range.high) * 10000
                all_sweeps.append({
                    'type': 'bearish',
                    'candle': candle,
                    'swept_level': self.asia_range.high,
                    'direction': 'SELL',
                    'pips': pips_swept,
                    'time': candle.time
                })
            
            # Check for bullish sweep (BUY)
            if candle.low < bullish_target and candle.close > self.asia_range.low:
                pips_swept = (self.asia_range.low - candle.low) * 10000
                all_sweeps.append({
                    'type': 'bullish',
                    'candle': candle,
                    'swept_level': self.asia_range.low,
                    'direction': 'BUY',
                    'pips': pips_swept,
                    'time': candle.time
                })
        
        # Log all found sweeps
        if all_sweeps:
            logger.info(f"🔍 Found {len(all_sweeps)} sweep(s) in Frankfurt/London window:")
            for sweep in all_sweeps:
                emoji = "🔴" if sweep['type'] == 'bearish' else "🟢"
                logger.info(f"   {emoji} {sweep['type'].upper()} @ {sweep['time'].strftime('%H:%M')} - swept {sweep['pips']:.1f} pips")
            
            # Return the MOST RECENT sweep (last in list = newest)
            chosen_sweep = all_sweeps[0]  # Since we iterate backwards, first = newest
            logger.info(f"✅ Using MOST RECENT sweep: {chosen_sweep['type'].upper()} @ {chosen_sweep['time'].strftime('%H:%M')}")
            
            if chosen_sweep['type'] == 'bearish':
                logger.info(f"🔴 BEARISH SWEEP detected: High={chosen_sweep['candle'].high:.5f}, Close={chosen_sweep['candle'].close:.5f}")
            else:
                logger.info(f"🟢 BULLISH SWEEP detected: Low={chosen_sweep['candle'].low:.5f}, Close={chosen_sweep['candle'].close:.5f}")
            
            return chosen_sweep
        
        return None
    
    def detect_choch(self, direction: str, after_index: int) -> Optional[Dict]:
        """
        Detect Change of Character (market structure shift)
        WITH DETAILED DEBUG LOGGING
        
        Args:
            direction: 'BUY' or 'SELL'
            after_index: Start looking after this index
            
        Returns:
            Dict with CHOCH details or None
        """
        if not self.m5_candles or after_index >= len(self.m5_candles):
            logger.debug("❌ CHOCH: Not enough candles or invalid index")
            return None
        
        # Find swing points after sweep
        candles_after_sweep = self.m5_candles[after_index:]
        
        if len(candles_after_sweep) < Config.SWING_LOOKBACK * 2 + 1:
            logger.debug(f"❌ CHOCH: Only {len(candles_after_sweep)} candles after sweep (need {Config.SWING_LOOKBACK * 2 + 1})")
            return None
        
        swing_highs = utils.find_swing_highs(candles_after_sweep, Config.SWING_LOOKBACK)
        swing_lows = utils.find_swing_lows(candles_after_sweep, Config.SWING_LOOKBACK)
        
        logger.debug(f"🔍 CHOCH check for {direction}:")
        logger.debug(f"   Candles after sweep: {len(candles_after_sweep)}")
        logger.debug(f"   Swing Highs found: {len(swing_highs)}")
        logger.debug(f"   Swing Lows found: {len(swing_lows)}")
        
        if direction == 'SELL':
            # For SELL: need lower-low (break of bullish structure)
            logger.debug(f"   → Looking for LOWER-LOW (bearish CHOCH)")
            
            if len(swing_lows) < 2:
                logger.debug(f"   ❌ Not enough swing lows: {len(swing_lows)} (need 2+)")
                return None
            
            latest_low = swing_lows[-1]
            previous_low = swing_lows[-2]
            
            logger.debug(f"   Previous Low: {previous_low['price']:.5f} @ {previous_low['time'].strftime('%H:%M')}")
            logger.debug(f"   Latest Low:   {latest_low['price']:.5f} @ {latest_low['time'].strftime('%H:%M')}")
            
            if latest_low['price'] < previous_low['price']:
                diff_pips = (previous_low['price'] - latest_low['price']) * 10000
                logger.info(f"🔻 CHOCH (SELL) detected!")
                logger.info(f"   Broke {previous_low['price']:.5f} with {latest_low['price']:.5f} ({diff_pips:.1f} pips lower)")
                return {
                    'type': 'bearish_choch',
                    'broken_level': previous_low['price'],
                    'new_low': latest_low['price'],
                    'time': latest_low['time']
                }
            else:
                diff_pips = (latest_low['price'] - previous_low['price']) * 10000
                logger.debug(f"   ❌ Latest low is HIGHER by {diff_pips:.1f} pips → NO CHOCH yet")
                return None
        
        elif direction == 'BUY':
            # For BUY: need higher-high (break of bearish structure)
            logger.debug(f"   → Looking for HIGHER-HIGH (bullish CHOCH)")
            
            if len(swing_highs) < 2:
                logger.debug(f"   ❌ Not enough swing highs: {len(swing_highs)} (need 2+)")
                return None
            
            latest_high = swing_highs[-1]
            previous_high = swing_highs[-2]
            
            logger.debug(f"   Previous High: {previous_high['price']:.5f} @ {previous_high['time'].strftime('%H:%M')}")
            logger.debug(f"   Latest High:   {latest_high['price']:.5f} @ {latest_high['time'].strftime('%H:%M')}")
            
            if latest_high['price'] > previous_high['price']:
                diff_pips = (latest_high['price'] - previous_high['price']) * 10000
                logger.info(f"🔺 CHOCH (BUY) detected!")
                logger.info(f"   Broke {previous_high['price']:.5f} with {latest_high['price']:.5f} ({diff_pips:.1f} pips higher)")
                return {
                    'type': 'bullish_choch',
                    'broken_level': previous_high['price'],
                    'new_high': latest_high['price'],
                    'time': latest_high['time']
                }
            else:
                diff_pips = (previous_high['price'] - latest_high['price']) * 10000
                logger.debug(f"   ❌ Latest high is LOWER by {diff_pips:.1f} pips → NO CHOCH yet")
                return None
        
        return None
    
    def find_entry_fvg(self, direction: str, after_index: int) -> Optional[Dict]:
        """
        Find FVG for entry after CHOCH
        
        Args:
            direction: 'BUY' or 'SELL'
            after_index: Start looking after this index
            
        Returns:
            FVG dict or None
        """
        if not self.m5_candles or after_index >= len(self.m5_candles) - 3:
            return None
        
        candles_after_choch = self.m5_candles[after_index:]
        
        if direction == 'SELL':
            fvgs = utils.find_fvgs(candles_after_choch, 'bearish')
        else:
            fvgs = utils.find_fvgs(candles_after_choch, 'bullish')
        
        # Return the most recent FVG
        if fvgs:
            fvg = fvgs[-1]
            logger.info(f"📍 FVG found: {fvg['type']} | {fvg['low']:.5f} - {fvg['high']:.5f}")
            return fvg
        
        return None
    
    def generate_signal(self) -> Optional[TradingSignal]:
        """
        Main signal generation logic
        Implements: Asia Range → Sweep → CHOCH → FVG Entry
        
        Returns:
            TradingSignal or None
        """
        # Check cooldown
        if self.last_signal_time:
            time_since_last = (datetime.utcnow() - self.last_signal_time).total_seconds()
            if time_since_last < Config.SIGNAL_COOLDOWN:
                return None
        
        # Step 1: Compute Asia Range
        asia_range = self.compute_asia_range()
        if not asia_range:
            return None
        
        # Step 2: Detect Sweep
        sweep = self.detect_sweep()
        if not sweep:
            return None
        
        # Find sweep candle index in m5_candles
        sweep_candle_index = None
        for i, candle in enumerate(self.m5_candles):
            if candle.time == sweep['candle'].time:
                sweep_candle_index = i
                break
        
        if sweep_candle_index is None:
            return None
        
        # Step 3: Detect CHOCH
        choch = self.detect_choch(sweep['direction'], sweep_candle_index)
        if not choch:
            logger.info("⏳ Waiting for CHOCH...")
            return None
        
        # Step 4: Find Entry FVG
        fvg = self.find_entry_fvg(sweep['direction'], sweep_candle_index)
        if not fvg:
            logger.info("⏳ Waiting for FVG...")
            return None
        
        # Step 5: Calculate Entry, SL, TP
        entry_price = fvg['mid']
        
        if sweep['direction'] == 'SELL':
            # Stop loss above recent swing high
            swing_highs = utils.find_swing_highs(self.m5_candles, Config.SWING_LOOKBACK)
            if not swing_highs:
                return None
            
            sl_level = swing_highs[-1]['price']
            stop_loss = sl_level + self.client.pips_to_price(Config.SL_BUFFER_PIPS)
            take_profit = asia_range.low  # TP1: opposite side of Asia range
            
        else:  # BUY
            # Stop loss below recent swing low
            swing_lows = utils.find_swing_lows(self.m5_candles, Config.SWING_LOOKBACK)
            if not swing_lows:
                return None
            
            sl_level = swing_lows[-1]['price']
            stop_loss = sl_level - self.client.pips_to_price(Config.SL_BUFFER_PIPS)
            take_profit = asia_range.high  # TP1: opposite side of Asia range
        
        # Step 6: Build context
        context = {
            'asia_range': {
                'high': asia_range.high,
                'low': asia_range.low,
                'mid': asia_range.mid
            },
            'sweep': {
                'type': sweep['type'],
                'candle_time': sweep['candle'].time,
                'candle_high': sweep['candle'].high,
                'candle_low': sweep['candle'].low,
                'candle_close': sweep['candle'].close,
                'swept_level': sweep['swept_level']
            },
            'choch': choch,
            'fvg': {
                'low': fvg['low'],
                'high': fvg['high'],
                'mid': fvg['mid'],
                'size_pips': fvg['size_pips']
            }
        }
        
        # Step 7: Create signal
        signal = TradingSignal(
            signal_type=sweep['direction'],
            entry=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            context=context
        )
        
        # Step 8: Validate signal
        if not signal.is_valid():
            return None
        
        # Check if already sent
        if signal.hash in self.sent_signals:
            logger.info("⚠️ Signal already sent (duplicate)")
            return None
        
        # Check spread
        current_price = self.client.get_current_price()
        if current_price and current_price['spread_pips'] > Config.MAX_SPREAD_PIPS:
            logger.warning(f"❌ Spread too high: {current_price['spread_pips']:.1f} pips")
            return None
        
        logger.info(f"✅ Valid signal generated: {signal.type} @ {signal.entry:.5f}")
        
        # Mark as sent
        self.sent_signals.add(signal.hash)
        self.last_signal_time = datetime.utcnow()
        
        return signal
