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
        # ✅ ЗАВЖДИ валідний - user хоче всі сигнали!
        # Просто логуємо RR для інформації
        if self.risk_reward < Config.MIN_RISK_REWARD:
            logger.warning(f"⚠️ Low RR: {self.risk_reward:.2f} (recommended min: {Config.MIN_RISK_REWARD})")
        
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
        NOW RETURNS ALL SWEEPS for CHOCH checking!
        
        Returns:
            Dict with 'all_sweeps' and 'chosen_sweep' or None
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
                    'time': candle.time,
                    'high': candle.high,
                    'low': candle.low,
                    'close': candle.close
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
                    'time': candle.time,
                    'high': candle.high,
                    'low': candle.low,
                    'close': candle.close
                })
        
        # Log all found sweeps
        if all_sweeps:
            logger.info(f"🔍 Found {len(all_sweeps)} sweep(s) in Frankfurt/London window:")
            for sweep in all_sweeps:
                emoji = "🔴" if sweep['type'] == 'bearish' else "🟢"
                logger.info(f"   {emoji} {sweep['type'].upper()} @ {sweep['time'].strftime('%H:%M')} - swept {sweep['pips']:.1f} pips")
            
            # Return ALL sweeps + the most recent one as default
            chosen_sweep = all_sweeps[0]  # Since we iterate backwards, first = newest
            logger.info(f"✅ Using MOST RECENT sweep: {chosen_sweep['type'].upper()} @ {chosen_sweep['time'].strftime('%H:%M')}")
            
            if chosen_sweep['type'] == 'bearish':
                logger.info(f"🔴 BEARISH SWEEP detected: High={chosen_sweep['high']:.5f}, Close={chosen_sweep['close']:.5f}")
            else:
                logger.info(f"🟢 BULLISH SWEEP detected: Low={chosen_sweep['low']:.5f}, Close={chosen_sweep['close']:.5f}")
            
            # Return format with ALL sweeps
            return {
                'all_sweeps': all_sweeps,
                'chosen_sweep': chosen_sweep,
                # Keep old format fields for compatibility
                'type': chosen_sweep['type'],
                'candle': chosen_sweep['candle'],
                'swept_level': chosen_sweep['swept_level'],
                'direction': chosen_sweep['direction']
            }
        
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
        NOW CHECKS CHOCH FOR ALL SWEEPS!
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
        
        # Step 2: Detect ALL Sweeps
        sweep_data = self.detect_sweep()
        if not sweep_data:
            return None
        
        # NEW: If detect_sweep returned multiple sweeps, check CHOCH for each
        # and choose the one that ALREADY HAS CHOCH
        if 'all_sweeps' in sweep_data:
            logger.info(f"🔍 Checking CHOCH for all {len(sweep_data['all_sweeps'])} sweep(s)...")
            
            sweeps_with_choch = []
            
            for sweep in sweep_data['all_sweeps']:
                # Find sweep candle index
                sweep_candle_index = None
                for i, candle in enumerate(self.m5_candles):
                    if candle.time == sweep['time']:
                        sweep_candle_index = i
                        break
                
                if sweep_candle_index is None:
                    continue
                
                # Check if this sweep has CHOCH
                choch = self.detect_choch(sweep['direction'], sweep_candle_index)
                
                if choch:
                    logger.info(f"   ✅ {sweep['type'].upper()} sweep @ {sweep['time'].strftime('%H:%M')} HAS CHOCH!")
                    sweeps_with_choch.append({
                        'sweep': sweep,
                        'choch': choch,
                        'sweep_index': sweep_candle_index
                    })
                else:
                    logger.debug(f"   ⏳ {sweep['type'].upper()} sweep @ {sweep['time'].strftime('%H:%M')} - no CHOCH yet")
            
            # If we found sweeps with CHOCH, use the most recent one
            if sweeps_with_choch:
                # Sort by time (most recent first)
                sweeps_with_choch.sort(key=lambda x: x['sweep']['time'], reverse=True)
                chosen = sweeps_with_choch[0]
                
                logger.info(f"✅ FOUND {len(sweeps_with_choch)} sweep(s) with CHOCH!")
                logger.info(f"✅ Using: {chosen['sweep']['type'].upper()} @ {chosen['sweep']['time'].strftime('%H:%M')} (has CHOCH)")
                
                sweep = chosen['sweep']
                choch = chosen['choch']
                sweep_candle_index = chosen['sweep_index']
            else:
                # No sweep has CHOCH yet - use most recent and wait
                logger.info(f"⏳ None of the {len(sweep_data['all_sweeps'])} sweep(s) have CHOCH yet. Using most recent.")
                sweep = sweep_data['chosen_sweep']
                
                # Find sweep candle index
                sweep_candle_index = None
                for i, candle in enumerate(self.m5_candles):
                    if candle.time == sweep['time']:
                        sweep_candle_index = i
                        break
                
                if sweep_candle_index is None:
                    return None
                
                choch = self.detect_choch(sweep['direction'], sweep_candle_index)
                if not choch:
                    logger.info("⏳ Waiting for CHOCH...")
                    return None
        else:
            # Old format (single sweep) - fallback
            sweep = sweep_data
            
            # Find sweep candle index
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
            # ✅ SELL SETUP (bearish sweep above Asia High)
            # Entry: FVG after CHOCH down
            # SL: Above Asia High (protect from move back up)
            # TP: Asia Low (opposite side of range)
            
            stop_loss = asia_range.high + self.client.pips_to_price(Config.SL_BUFFER_PIPS)
            take_profit = asia_range.low
            
            logger.debug(f"📊 SELL setup:")
            logger.debug(f"   Entry: {entry_price:.5f} (FVG mid)")
            logger.debug(f"   SL: {stop_loss:.5f} (Asia High + {Config.SL_BUFFER_PIPS} pips)")
            logger.debug(f"   TP: {take_profit:.5f} (Asia Low)")
            
            # Validate: SL must be ABOVE entry, TP must be BELOW entry
            if stop_loss <= entry_price:
                logger.warning(f"❌ Invalid SELL setup: SL {stop_loss:.5f} not above entry {entry_price:.5f}")
                return None
            
            if take_profit >= entry_price:
                logger.warning(f"❌ Invalid SELL setup: TP {take_profit:.5f} not below entry {entry_price:.5f}")
                logger.warning(f"   Entry is already below Asia Low - setup invalid!")
                return None
            
        else:  # BUY
            # ✅ BUY SETUP (bullish sweep below Asia Low)
            # Entry: FVG after CHOCH up
            # SL: Below Asia Low (protect from move back down)
            # TP: Asia High (opposite side of range)
            
            stop_loss = asia_range.low - self.client.pips_to_price(Config.SL_BUFFER_PIPS)
            take_profit = asia_range.high
            
            logger.debug(f"📊 BUY setup:")
            logger.debug(f"   Entry: {entry_price:.5f} (FVG mid)")
            logger.debug(f"   SL: {stop_loss:.5f} (Asia Low - {Config.SL_BUFFER_PIPS} pips)")
            logger.debug(f"   TP: {take_profit:.5f} (Asia High)")
            
            # Validate: SL must be BELOW entry, TP must be ABOVE entry
            if stop_loss >= entry_price:
                logger.warning(f"❌ Invalid BUY setup: SL {stop_loss:.5f} not below entry {entry_price:.5f}")
                return None
            
            if take_profit <= entry_price:
                logger.warning(f"❌ Invalid BUY setup: TP {take_profit:.5f} not above entry {entry_price:.5f}")
                logger.warning(f"   Entry is already above Asia High - setup invalid!")
                return None
        
        # Step 6: Build context
        context = {
            'asia_range': {
                'high': asia_range.high,
                'low': asia_range.low,
                'mid': asia_range.mid
            },
            'sweep': {
                'type': sweep['type'],
                'candle_time': sweep.get('candle', sweep).time if hasattr(sweep.get('candle', sweep), 'time') else sweep['time'],
                'candle_high': sweep.get('candle', sweep).high if hasattr(sweep.get('candle', sweep), 'high') else sweep.get('high', 0),
                'candle_low': sweep.get('candle', sweep).low if hasattr(sweep.get('candle', sweep), 'low') else sweep.get('low', 0),
                'candle_close': sweep.get('candle', sweep).close if hasattr(sweep.get('candle', sweep), 'close') else sweep.get('close', 0),
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
