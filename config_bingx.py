"""
Configuration module for EURUSD Trading Signal Bot (BingX TradFi Forex Perpetual)
Loads settings from environment variables
"""
import os
from typing import Optional


class Config:
    """Configuration class for bot settings"""
    
    # BingX Perpetual Swap API Configuration
    # Public endpoints - NO AUTH needed for market data!
    BINGX_BASE_URL: str = "https://open-api.bingx.com"
    
    # Telegram Configuration
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")
    
    # Trading Parameters
    # EURUSD on BingX TradFi Forex Perpetual Futures
    SYMBOL: str = "EURUSD-USDT"  # BingX TradFi format (NOT EUR-USDT!)
    
    # Timeframes (BingX format)
    TF_CONTEXT: str = "15m"   # Context timeframe (was M15)
    TF_ENTRY: str = "5m"      # Entry timeframe (was M5)
    
    # Session Times (UTC)
    ASIA_SESSION_START: int = 0   # 00:00 UTC
    ASIA_SESSION_END: int = 5     # 05:00 UTC
    LONDON_SESSION_START: int = 5  # 05:00 UTC
    LONDON_SESSION_END: int = 10   # 10:00 UTC
    
    # Strategy Parameters (can be overridden via environment variables)
    SWEEP_PIPS_THRESHOLD: float = float(os.getenv("SWEEP_PIPS_THRESHOLD", "1.5"))  # Pips beyond Asia range to confirm sweep
    SWING_LOOKBACK: int = int(os.getenv("SWING_LOOKBACK", "3"))  # Number of candles for swing high/low detection
    FVG_MIN_PIPS: float = float(os.getenv("FVG_MIN_PIPS", "1.5"))  # Minimum FVG size in pips
    
    # Risk Management
    MIN_RISK_REWARD: float = 1.5  # Minimum RR ratio
    MAX_SPREAD_PIPS: float = 2.0  # Maximum allowed spread
    SL_BUFFER_PIPS: float = 1.5   # Stop loss buffer
    
    # Bot Behavior
    UPDATE_INTERVAL: int = 45  # Seconds between updates
    SIGNAL_COOLDOWN: int = 1800  # Seconds (30 min) between signals
    MAX_CANDLES: int = 500  # Number of historical candles to keep
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    @classmethod
    def validate(cls) -> bool:
        """Validate required configuration"""
        required = [
            cls.TELEGRAM_BOT_TOKEN,
            cls.TELEGRAM_CHAT_ID
        ]
        
        if not all(required):
            missing = []
            if not cls.TELEGRAM_BOT_TOKEN:
                missing.append("TELEGRAM_BOT_TOKEN")
            if not cls.TELEGRAM_CHAT_ID:
                missing.append("TELEGRAM_CHAT_ID")
            
            print(f"❌ Missing required environment variables: {', '.join(missing)}")
            return False
        
        return True
