"""
Configuration module for EURUSD Trading Signal Bot (BingX TradFi Forex Perpetual)
Loads settings from environment variables
"""
import os


class Config:
    """Configuration class for bot settings"""
    
    # BingX API
    BINGX_BASE_URL: str = "https://open-api.bingx.com"
    
    # Telegram
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")
    
    # Symbol
    SYMBOL: str = "EURUSD-USDT"  # Fallback, auto-detected at runtime
    
    # Timeframes
    TF_CONTEXT: str = "15m"
    TF_ENTRY: str = "5m"
    
    # ✅ ПРАВИЛЬНІ ICT/ТРЕЙДЕРСЬКІ СЕСІЇ (UTC)
    # Азія: 23:00 попереднього дня - 07:00
    ASIA_SESSION_START: int = 23   # 23:00 UTC (починається ВЧОРА!)
    ASIA_SESSION_END: int = 7      # 07:00 UTC
    
    # Франкфурт: 07:00 - 08:00 (Kill Zone, може вибити Asia range!)
    FRANKFURT_SESSION_START: int = 7   # 07:00 UTC
    FRANKFURT_SESSION_END: int = 8     # 08:00 UTC
    
    # Лондон: 08:00 - 13:00
    LONDON_SESSION_START: int = 8      # 08:00 UTC
    LONDON_SESSION_END: int = 13       # 13:00 UTC
    
    # Нью-Йорк: 13:00 - 22:00
    NY_SESSION_START: int = 13     # 13:00 UTC
    NY_SESSION_END: int = 22       # 22:00 UTC
    
    # ✅ SWEEP DETECTION WINDOW = Франкфурт + Лондон (07:00 - 13:00)
    # Sweep може відбутись і в Франкфурті, і в Лондоні!
    SWEEP_WINDOW_START: int = 7    # 07:00 UTC (з початку Франкфурту)
    SWEEP_WINDOW_END: int = 13     # 13:00 UTC (кінець Лондону)
    
    # Strategy Parameters (env variables)
    ASIA_MIN_RANGE_PIPS: float = float(os.getenv("ASIA_MIN_RANGE_PIPS", "3.0"))  # Мін розмір Asia Range
    SWEEP_PIPS_THRESHOLD: float = float(os.getenv("SWEEP_PIPS_THRESHOLD", "1.5"))
    SWING_LOOKBACK: int = int(os.getenv("SWING_LOOKBACK", "3"))
    FVG_MIN_PIPS: float = float(os.getenv("FVG_MIN_PIPS", "1.5"))
    
    # Risk Management
    MIN_RISK_REWARD: float = 1.5
    MAX_SPREAD_PIPS: float = 2.0
    SL_BUFFER_PIPS: float = 1.5
    
    # Bot Behavior
    UPDATE_INTERVAL: int = 45
    SIGNAL_COOLDOWN: int = 1800
    MAX_CANDLES: int = 500
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "DEBUG")
    
    @classmethod
    def validate(cls) -> bool:
        missing = []
        if not cls.TELEGRAM_BOT_TOKEN:
            missing.append("TELEGRAM_BOT_TOKEN")
        if not cls.TELEGRAM_CHAT_ID:
            missing.append("TELEGRAM_CHAT_ID")
        if missing:
            print(f"❌ Missing env variables: {', '.join(missing)}")
            return False
        return True
