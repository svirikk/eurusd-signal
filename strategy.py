"""
Trading Strategy Implementation
Asia Range Sweep → CHOCH → FVG Entry Strategy
"""
import logging
from typing import List, Optional, Dict
from datetime import datetime, timedelta
from oanda_client import Candle, OandaClient
from config import Config
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
    
    def __init__(self, oanda_client: OandaClient):
        self.client = oanda_client
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
        Compute Asia range (00:00 - 05:00 UTC)
        Only recalculates once per day
        """
        if not self.m5_candles:
            return None
        
        now = datetime.utcnow()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Check if we already have today's range
        if self.asia_range and self.asia_range.date.date() == now.date():
            return self.asia_range
        
        # Find candles in Asia session
        asia_candles = [
            c for c in self.m5_candles
            if c.time.date() == now.date() and 
            utils.is_time_in_session(c.time, Config.ASIA_SESSION_START, Config.ASIA_SESSION_END)
        ]
        
        if not asia_candles:
            # Try yesterday's Asia range if today's not ready yet
            yesterday = now - timedelta(days=1)
            asia_candles = [
                c for c in self.m5_candles
                if c.time.date() == yesterday.date() and 
                utils.is_time_in_session(c.time, Config.ASIA_SESSION_START, Config.ASIA_SESSION_END)
            ]
            
            if not asia_candles:
                logger.warning("⚠️ No Asia session candles found")
                return None
        
        # Calculate high and low
        asia_high = max(c.high for c in asia_candles)
        asia_low = min(c.low for c in asia_candles)
        
        self.asia_range = AsiaRange(asia_high, asia_low, asia_candles[0].time)
        logger.info(f"✅ Asia Range computed: {self.asia_range}")
        
        return self.asia_range
    
    def detect_sweep(self) -> Optional[Dict]:
        """
        Detect liquidity sweep of Asia range during London session
        
        Returns:
            Dict with sweep details or None
        """
        if not self.asia_range or not self.m5_candles:
            return None
        
        # Check only recent candles in London session
        recent_candles = [
            c for c in self.m5_candles[-50:]
            if utils.is_time_in_session(c.time, Config.LONDON_SESSION_START, Config.LONDON_SESSION_END)
        ]
        
        if not recent_candles:
            return None
        
        sweep_threshold = self.client.pips_to_price(Config.SWEEP_PIPS_THRESHOLD)
        
        # Check for bearish sweep (sweep high for SELL setup)
        for i in range(len(recent_candles) - 1, -1, -1):
            candle = recent_candles[i]
            
            # Sweep Asia High (for SELL)
            if candle.high > self.asia_range.high + sweep_threshold:
                if candle.close < self.asia_range.high:
                    logger.info(f"🔴 BEARISH SWEEP detected: High={candle.high:.5f}, Close={candle.close:.5f}")
                    return {
                        'type': 'bearish',
                        'candle': candle,
                        'swept_level': self.asia_range.high,
                        'direction': 'SELL'
                    }
            
            # Sweep Asia Low (for BUY)
            if candle.low < self.asia_range.low - sweep_threshold:
                if candle.close > self.asia_range.low:
                    logger.info(f"🟢 BULLISH SWEEP detected: Low={candle.low:.5f}, Close={candle.close:.5f}")
                    return {
                        'type': 'bullish',
                        'candle': candle,
                        'swept_level': self.asia_range.low,
                        'direction': 'BUY'
                    }
        
        return None
    
    def detect_choch(self, direction: str, after_index: int) -> Optional[Dict]:
        """
        Detect Change of Character (market structure shift)
        
        Args:
            direction: 'BUY' or 'SELL'
            after_index: Start looking after this index
            
        Returns:
            Dict with CHOCH details or None
        """
        if not self.m5_candles or after_index >= len(self.m5_candles):
            return None
        
        # Find swing points after sweep
        candles_after_sweep = self.m5_candles[after_index:]
        
        if len(candles_after_sweep) < Config.SWING_LOOKBACK * 2 + 1:
            return None
        
        swing_highs = utils.find_swing_highs(candles_after_sweep, Config.SWING_LOOKBACK)
        swing_lows = utils.find_swing_lows(candles_after_sweep, Config.SWING_LOOKBACK)
        
        if direction == 'SELL':
            # For SELL: need lower-low (break of bullish structure)
            if len(swing_lows) >= 2:
                # Check if latest low is lower than previous
                latest_low = swing_lows[-1]
                previous_low = swing_lows[-2]
                
                if latest_low['price'] < previous_low['price']:
                    logger.info(f"🔻 CHOCH (SELL) detected: Broken level {previous_low['price']:.5f}")
                    return {
                        'type': 'bearish_choch',
                        'broken_level': previous_low['price'],
                        'new_low': latest_low['price'],
                        'time': latest_low['time']
                    }
        
        elif direction == 'BUY':
            # For BUY: need higher-high (break of bearish structure)
            if len(swing_highs) >= 2:
                # Check if latest high is higher than previous
                latest_high = swing_highs[-1]
                previous_high = swing_highs[-2]
                
                if latest_high['price'] > previous_high['price']:
                    logger.info(f"🔺 CHOCH (BUY) detected: Broken level {previous_high['price']:.5f}")
                    return {
                        'type': 'bullish_choch',
                        'broken_level': previous_high['price'],
                        'new_high': latest_high['price'],
                        'time': latest_high['time']
                    }
        
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
