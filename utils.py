"""
Utility functions for trading bot
Swing detection, FVG detection, etc.
"""
import hashlib
from typing import List, Optional, Dict, Tuple
from datetime import datetime
from bingx_client import Candle
from config_bingx import Config


def is_swing_high(candles: List[Candle], index: int, lookback: int = 3) -> bool:
    """
    Check if candle at index is a swing high
    
    Args:
        candles: List of candles
        index: Index to check
        lookback: Number of candles to check on each side
        
    Returns:
        True if it's a swing high
    """
    if index < lookback or index >= len(candles) - lookback:
        return False
    
    high = candles[index].high
    
    # Check left side
    for i in range(index - lookback, index):
        if candles[i].high >= high:
            return False
    
    # Check right side
    for i in range(index + 1, index + lookback + 1):
        if candles[i].high >= high:
            return False
    
    return True


def is_swing_low(candles: List[Candle], index: int, lookback: int = 3) -> bool:
    """
    Check if candle at index is a swing low
    
    Args:
        candles: List of candles
        index: Index to check
        lookback: Number of candles to check on each side
        
    Returns:
        True if it's a swing low
    """
    if index < lookback or index >= len(candles) - lookback:
        return False
    
    low = candles[index].low
    
    # Check left side
    for i in range(index - lookback, index):
        if candles[i].low <= low:
            return False
    
    # Check right side
    for i in range(index + 1, index + lookback + 1):
        if candles[i].low <= low:
            return False
    
    return True


def find_swing_highs(candles: List[Candle], lookback: int = 3) -> List[Dict]:
    """
    Find all swing highs in candle list
    
    Returns:
        List of dicts with 'index', 'price', 'time'
    """
    swing_highs = []
    
    for i in range(lookback, len(candles) - lookback):
        if is_swing_high(candles, i, lookback):
            swing_highs.append({
                'index': i,
                'price': candles[i].high,
                'time': candles[i].time
            })
    
    return swing_highs


def find_swing_lows(candles: List[Candle], lookback: int = 3) -> List[Dict]:
    """
    Find all swing lows in candle list
    
    Returns:
        List of dicts with 'index', 'price', 'time'
    """
    swing_lows = []
    
    for i in range(lookback, len(candles) - lookback):
        if is_swing_low(candles, i, lookback):
            swing_lows.append({
                'index': i,
                'price': candles[i].low,
                'time': candles[i].time
            })
    
    return swing_lows


def detect_bullish_fvg(candle1: Candle, candle2: Candle, candle3: Candle) -> Optional[Dict]:
    """
    Detect bullish Fair Value Gap (FVG)
    FVG exists when low of candle3 > high of candle1
    
    Returns:
        Dict with 'low', 'high', 'mid' or None
    """
    if candle3.low > candle1.high:
        gap_low = candle1.high
        gap_high = candle3.low
        
        # Check minimum size
        gap_size_pips = (gap_high - gap_low) * 10000
        if gap_size_pips >= Config.FVG_MIN_PIPS:
            return {
                'low': gap_low,
                'high': gap_high,
                'mid': (gap_low + gap_high) / 2,
                'size_pips': gap_size_pips,
                'type': 'bullish',
                'time': candle2.time
            }
    
    return None


def detect_bearish_fvg(candle1: Candle, candle2: Candle, candle3: Candle) -> Optional[Dict]:
    """
    Detect bearish Fair Value Gap (FVG)
    FVG exists when high of candle3 < low of candle1
    
    Returns:
        Dict with 'low', 'high', 'mid' or None
    """
    if candle3.high < candle1.low:
        gap_low = candle3.high
        gap_high = candle1.low
        
        # Check minimum size
        gap_size_pips = (gap_high - gap_low) * 10000
        if gap_size_pips >= Config.FVG_MIN_PIPS:
            return {
                'low': gap_low,
                'high': gap_high,
                'mid': (gap_low + gap_high) / 2,
                'size_pips': gap_size_pips,
                'type': 'bearish',
                'time': candle2.time
            }
    
    return None


def find_fvgs(candles: List[Candle], direction: str = 'both') -> List[Dict]:
    """
    Find all FVGs in candle list
    
    Args:
        candles: List of candles
        direction: 'bullish', 'bearish', or 'both'
        
    Returns:
        List of FVG dicts
    """
    fvgs = []
    
    for i in range(len(candles) - 2):
        if direction in ['bullish', 'both']:
            bullish_fvg = detect_bullish_fvg(candles[i], candles[i+1], candles[i+2])
            if bullish_fvg:
                bullish_fvg['index'] = i + 2
                fvgs.append(bullish_fvg)
        
        if direction in ['bearish', 'both']:
            bearish_fvg = detect_bearish_fvg(candles[i], candles[i+1], candles[i+2])
            if bearish_fvg:
                bearish_fvg['index'] = i + 2
                fvgs.append(bearish_fvg)
    
    return fvgs


def is_time_in_session(time: datetime, session_start: int, session_end: int) -> bool:
    """
    Check if time is within session hours (UTC)
    
    Args:
        time: datetime object
        session_start: Start hour (0-23)
        session_end: End hour (0-23)
        
    Returns:
        True if time is in session
    """
    hour = time.hour
    
    if session_start < session_end:
        return session_start <= hour < session_end
    else:
        # Handle overnight sessions
        return hour >= session_start or hour < session_end


def generate_signal_hash(signal_type: str, entry_price: float, timestamp: datetime) -> str:
    """
    Generate unique hash for signal to prevent duplicates
    
    Args:
        signal_type: 'BUY' or 'SELL'
        entry_price: Entry price
        timestamp: Signal timestamp
        
    Returns:
        Hash string
    """
    data = f"{signal_type}_{entry_price:.5f}_{timestamp.strftime('%Y%m%d%H%M')}"
    return hashlib.md5(data.encode()).hexdigest()


def format_price(price: float, decimals: int = 5) -> str:
    """Format price to specified decimals"""
    return f"{price:.{decimals}f}"


def format_pips(pips: float, decimals: int = 1) -> str:
    """Format pips value"""
    return f"{pips:.{decimals}f}"


def calculate_lot_size(account_balance: float, risk_percent: float, sl_pips: float) -> float:
    """
    Calculate lot size based on risk management
    (Optional - not used in MVP but useful for reference)
    
    Args:
        account_balance: Account balance in USD
        risk_percent: Risk percentage (e.g., 1.0 for 1%)
        sl_pips: Stop loss in pips
        
    Returns:
        Lot size
    """
    risk_amount = account_balance * (risk_percent / 100)
    pip_value = 10  # For 1 standard lot EURUSD, 1 pip = $10
    lot_size = risk_amount / (sl_pips * pip_value)
    
    # Round to 2 decimals (0.01 lot increments)
    return round(lot_size, 2)
