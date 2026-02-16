"""
Trading Strategy Implementation (WITH ENHANCED DEBUG LOGGING)
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
            logger.debug(f"   First M5: {m5[0].time} | Last M5: {m5[-1].time}")
            logger.debug(f"   Latest M5: O:{m5[-1].open:.5f} H:{m5[-1].high:.5f} L:{m5[-1].low:.5f} C:{m5[-1].close:.5f}")
        
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
            logger.debug("⚠️ No M5 candles available for Asia Range")
            return None
        
        now = datetime.utcnow()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Check if we already have today's range
        if self.asia_range and self.asia_range.date.date() == now.date():
            logger.debug(f"📊 Using cached Asia Range: {self.asia_range}")
            return self.asia_range
        
        logger.debug(f"🔍 Computing Asia Range for {now.date()}...")
        
        # Find candles in Asia session
        asia_candles = [
            c for c in self.m5_candles
            if c.time.date() == now.date() and 
            utils.is_time_in_session(c.time, Config.ASIA_SESSION_START, Config.ASIA_SESSION_END)
        ]
        
        if not asia_candles:
            logger.debug(f"⚠️ No Asia candles for today. Trying yesterday...")
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
            logger.debug(f"✅ Using yesterday's Asia Range ({len(asia_candles)} candles)")
        else:
            logger.debug(f"✅ Found {len(asia_candles)} Asia candles for today")
        
        # Calculate high and low
        asia_high = max(c.high for c in asia_candles)
        asia_low = min(c.low for c in asia_candles)
        
        self.asia_range = AsiaRange(asia_high, asia_low, asia_candles[0].time)
        logger.info(f"✅ Asia Range computed: {self.asia_range}")
        logger.debug(f"   Range size: {(asia_high - asia_low) * 10000:.1f} pips")
        
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
            logger.debug("⏳ No London session candles in recent 50 candles")
            return None
        
        logger.debug(f"🔍 Checking {len(recent_candles)} recent London candles for sweep...")
        logger.debug(f"   Asia High: {self.asia_range.high:.5f} | Asia Low: {self.asia_range.low:.5f}")
        
        sweep_threshold = self.client.pips_to_price(Config.SWEEP_PIPS_THRESHOLD)
        logger.debug(f"   Sweep threshold: {Config.SWEEP_PIPS_THRESHOLD} pips ({sweep_threshold:.5f})")
        
        bearish_target = self.asia_range.high + sweep_threshold
        bullish_target = self.asia_range.low - sweep_threshold
        logger.debug(f"   Bearish sweep target: > {bearish_target:.5f}")
        logger.debug(f"   Bullish sweep target: < {bullish_target:.5f}")
        
        # Check for bearish sweep (sweep high for SELL setup)
        for i in range(len(recent_candles) - 1, -1, -1):
            candle = recent_candles[i]
            
            # Debug each candle check
            if candle.high > self.asia_range.high:
                logger.debug(f"   📍 Candle {candle.time.strftime('%H:%M')}: high={candle.high:.5f} (above Asia), close={candle.close:.5f}")
            
            # Sweep Asia High (for SELL)
            if candle.high > bearish_target:
                logger.debug(f"   🔥 BEARISH sweep candidate: high={candle.high:.5f} > {bearish_target:.5f}")
                if candle.close < self.asia_range.high:
                    logger.info(f"🔴 BEARISH SWEEP detected: High={candle.high:.5f}, Close={candle.close:.5f}")
                    logger.info(f"   Swept {(candle.high - self.asia_range.high) * 10000:.1f} pips above Asia High")
                    return {
                        'type': 'bearish',
                        'candle': candle,
                        'swept_level': self.asia_range.high,
                        'direction': 'SELL'
                    }
                else:
                    logger.debug(f"   ❌ But close={candle.close:.5f} NOT back in range (needs < {self.asia_range.high:.5f})")
            
            # Debug each candle check for bullish
            if candle.low < self.asia_range.low:
                logger.debug(f"   📍 Candle {candle.time.strftime('%H:%M')}: low={candle.low:.5f} (below Asia), close={candle.close:.5f}")
            
            # Sweep Asia Low (for BUY)
            if candle.low < bullish_target:
                logger.debug(f"   🔥 BULLISH sweep candidate: low={candle.low:.5f} < {bullish_target:.5f}")
                if candle.close > self.asia_range.low:
                    logger.info(f"🟢 BULLISH SWEEP detected: Low={candle.low:.5f}, Close={candle.close:.5f}")
                    logger.info(f"   Swept {(self.asia_range.low - candle.low) * 10000:.1f} pips below Asia Low")
                    return {
                        'type': 'bullish',
                        'candle': candle,
                        'swept_level': self.asia_range.low,
                        'direction': 'BUY'
                    }
                else:
                    logger.debug(f"   ❌ But close={candle.close:.5f} NOT back in range (needs > {self.asia_range.low:.5f})")
        
        logger.debug("   ⏳ No valid sweep detected in recent candles")
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
            logger.debug(f"⏳ Not enough candles after sweep for CHOCH detection ({len(candles_after_sweep)} < {Config.SWING_LOOKBACK * 2 + 1})")
            return None
        
        swing_highs = utils.find_swing_highs(candles_after_sweep, Config.SWING_LOOKBACK)
        swing_lows = utils.find_swing_lows(candles_after_sweep, Config.SWING_LOOKBACK)
        
        logger.debug(f"🔍 CHOCH check: {len(swing_highs)} swing highs, {len(swing_lows)} swing lows")
        
        if direction == 'SELL':
            # For SELL: need lower-low (break of bullish structure)
            if len(swing_lows) >= 2:
                # Check if latest low is lower than previous
                latest_low = swing_lows[-1]
                previous_low = swing_lows[-2]
                
                logger.debug(f"   Latest low: {latest_low['price']:.5f} vs Previous low: {previous_low['price']:.5f}")
                
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
                
                logger.debug(f"   Latest high: {latest_high['price']:.5f} vs Previous high: {previous_high['price']:.5f}")
                
                if latest_high['price'] > previous_high['price']:
                    logger.info(f"🔺 CHOCH (BUY) detected: Broken level {previous_high['price']:.5f}")
                    return {
                        'type': 'bullish_choch',
                        'broken_level': previous_high['price'],
                        'new_high': latest_high['price'],
                        'time': latest_high['time']
                    }
        
        logger.debug("   ⏳ No CHOCH structure break detected")
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
        
        logger.debug(f"🔍 FVG search: found {len(fvgs)} {direction} FVGs")
        
        # Return the most recent FVG
        if fvgs:
            fvg = fvgs[-1]
            logger.info(f"📍 FVG found: {fvg['type']} | {fvg['low']:.5f} - {fvg['high']:.5f} ({fvg['size_pips']:.1f} pips)")
            return fvg
        
        logger.debug("   ⏳ No valid FVG found")
        return None
    
    def generate_signal(self) -> Optional[TradingSignal]:
        """
        Main signal generation logic
        Implements: Asia Range → Sweep → CHOCH → FVG Entry
        
        Returns:
            TradingSignal or None
        """
        logger.debug("=" * 60)
        logger.debug("🎯 SIGNAL GENERATION CYCLE START")
        logger.debug("=" * 60)
        
        # Check cooldown
        if self.last_signal_time:
            time_since_last = (datetime.utcnow() - self.last_signal_time).total_seconds()
            if time_since_last < Config.SIGNAL_COOLDOWN:
                logger.debug(f"⏳ Signal cooldown active: {time_since_last:.0f}s / {Config.SIGNAL_COOLDOWN}s")
                return None
        
        # Step 1: Compute Asia Range
        logger.debug("📊 Step 1: Computing Asia Range...")
        asia_range = self.compute_asia_range()
        if not asia_range:
            logger.debug("❌ No Asia Range available")
            return None
        
        # Step 2: Detect Sweep
        logger.debug("📊 Step 2: Detecting Sweep...")
        sweep = self.detect_sweep()
        if not sweep:
            logger.debug("⏳ No sweep detected, waiting...")
            return None
        
        # Find sweep candle index in m5_candles
        sweep_candle_index = None
        for i, candle in enumerate(self.m5_candles):
            if candle.time == sweep['candle'].time:
                sweep_candle_index = i
                break
        
        if sweep_candle_index is None:
            logger.warning("❌ Sweep candle not found in m5_candles list")
            return None
        
        logger.debug(f"✅ Sweep candle found at index {sweep_candle_index}/{len(self.m5_candles)}")
        
        # Step 3: Detect CHOCH
        logger.debug("📊 Step 3: Detecting CHOCH...")
        choch = self.detect_choch(sweep['direction'], sweep_candle_index)
        if not choch:
            logger.info("⏳ Waiting for CHOCH...")
            return None
        
        # Step 4: Find Entry FVG
        logger.debug("📊 Step 4: Finding Entry FVG...")
        fvg = self.find_entry_fvg(sweep['direction'], sweep_candle_index)
        if not fvg:
            logger.info("⏳ Waiting for FVG...")
            return None
        
        # Step 5: Calculate Entry, SL, TP
        logger.debug("📊 Step 5: Calculating Entry, SL, TP...")
        entry_price = fvg['mid']
        
        if sweep['direction'] == 'SELL':
            # Stop loss above recent swing high
            swing_highs = utils.find_swing_highs(self.m5_candles, Config.SWING_LOOKBACK)
            if not swing_highs:
                logger.warning("❌ No swing highs found for SL calculation")
                return None
            
            sl_level = swing_highs[-1]['price']
            stop_loss = sl_level + self.client.pips_to_price(Config.SL_BUFFER_PIPS)
            take_profit = asia_range.low  # TP1: opposite side of Asia range
            
        else:  # BUY
            # Stop loss below recent swing low
            swing_lows = utils.find_swing_lows(self.m5_candles, Config.SWING_LOOKBACK)
            if not swing_lows:
                logger.warning("❌ No swing lows found for SL calculation")
                return None
            
            sl_level = swing_lows[-1]['price']
            stop_loss = sl_level - self.client.pips_to_price(Config.SL_BUFFER_PIPS)
            take_profit = asia_range.high  # TP1: opposite side of Asia range
        
        logger.debug(f"   Entry: {entry_price:.5f}")
        logger.debug(f"   SL: {stop_loss:.5f}")
        logger.debug(f"   TP: {take_profit:.5f}")
        
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
        logger.debug("📊 Step 7: Creating Signal...")
        signal = TradingSignal(
            signal_type=sweep['direction'],
            entry=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            context=context
        )
        
        logger.debug(f"   Risk: {signal.risk_pips:.1f} pips")
        logger.debug(f"   Reward: {signal.reward_pips:.1f} pips")
        logger.debug(f"   RR: 1:{signal.risk_reward:.2f}")
        
        # Step 8: Validate signal
        logger.debug("📊 Step 8: Validating Signal...")
        if not signal.is_valid():
            logger.warning("❌ Signal validation failed")
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
        logger.info(f"   Risk/Reward: 1:{signal.risk_reward:.2f} | Risk: {signal.risk_pips:.1f} pips")
        
        # Mark as sent
        self.sent_signals.add(signal.hash)
        self.last_signal_time = datetime.utcnow()
        
        logger.debug("=" * 60)
        logger.debug("🎉 SIGNAL GENERATION COMPLETE")
        logger.debug("=" * 60)
        
        return signal
