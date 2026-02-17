"""
BingX TradFi Forex Perpetual API Client
Handles all interactions with BingX TradFi Forex Perpetual Futures (EURUSD-USDT)
Note: Uses same Perpetual Swap endpoints but with TradFi symbols
"""
import requests
import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from config_bingx import Config


logger = logging.getLogger(__name__)


class Candle:
    """Represents a single candlestick"""
    
    def __init__(self, data: Dict):
        """
        Initialize candle from BingX Perpetual Swap kline data
        
        BingX Perpetual Swap format: {
            "time": 1699934400000,
            "open": "1.08500",
            "high": "1.08600",
            "low": "1.08400",
            "close": "1.08550",
            "volume": "12345.67"
        }
        """
        self.time = datetime.utcfromtimestamp(int(data['time']) / 1000)  # UTC! НЕ локальний час!
        self.open = float(data['open'])
        self.high = float(data['high'])
        self.low = float(data['low'])
        self.close = float(data['close'])
        self.volume = float(data['volume'])
        self.complete = True  # BingX returns completed candles
    
    def __repr__(self):
        return f"Candle({self.time.strftime('%Y-%m-%d %H:%M')}, O:{self.open:.5f}, H:{self.high:.5f}, L:{self.low:.5f}, C:{self.close:.5f})"
    
    def is_bullish(self) -> bool:
        return self.close > self.open
    
    def is_bearish(self) -> bool:
        return self.close < self.open
    
    def body_size(self) -> float:
        return abs(self.close - self.open)
    
    def range_size(self) -> float:
        return self.high - self.low


class BingXClient:
    """Client for BingX Perpetual Swap API"""
    
    def __init__(self):
        self.base_url = Config.BINGX_BASE_URL
        self.symbol = Config.SYMBOL
        
    def get_candles(self, interval: str, limit: int = 500) -> List[Candle]:
        """
        Fetch candles from BingX Perpetual Swap
        
        Args:
            interval: 5m, 15m, 1h, etc. (BingX format)
            limit: Number of candles to fetch (max 1440)
            
        Returns:
            List of Candle objects
        """
        endpoint = f"{self.base_url}/openApi/swap/v2/quote/klines"
        
        params = {
            "symbol": self.symbol,
            "interval": interval,
            "limit": min(limit, 1440)  # BingX Perpetual max limit
        }
        
        try:
            response = requests.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Check for API error
            if data.get('code') != 0:
                logger.error(f"❌ BingX Perpetual API error: {data.get('msg', 'Unknown error')}")
                return []
            
            # Parse candles
            kline_data = data.get('data', [])
            
            if not kline_data:
                logger.warning(f"⚠️ No candle data received for {self.symbol}")
                return []
            
            candles = [Candle(k) for k in kline_data]
            
            logger.info(f"✅ Fetched {len(candles)} {interval} candles from BingX Perpetual")
            return candles
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Error fetching candles from BingX Perpetual: {e}")
            return []
        except Exception as e:
            logger.error(f"❌ Error parsing BingX Perpetual candles: {e}")
            return []
    
    def get_latest_candles(self, interval: str, count: int = 100) -> List[Candle]:
        """Get latest candles"""
        return self.get_candles(interval, count)
    
    def get_current_price(self) -> Optional[Dict[str, float]]:
        """
        Get current ticker price from BingX Perpetual Swap
        
        Returns:
            Dict with 'bid', 'ask', 'mid', 'spread_pips'
        """
        endpoint = f"{self.base_url}/openApi/swap/v2/quote/ticker"
        
        params = {
            "symbol": self.symbol
        }
        
        try:
            response = requests.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Check for API error
            if data.get('code') != 0:
                logger.error(f"❌ BingX Perpetual API error: {data.get('msg', 'Unknown error')}")
                return None
            
            ticker_data = data.get('data', {})
            
            # BingX Perpetual ticker returns lastPrice, we'll approximate bid/ask
            last_price = float(ticker_data.get('lastPrice', 0))
            
            if last_price <= 0:
                logger.warning(f"⚠️ Invalid last price: {last_price}")
                return None
            
            # Approximate spread (0.1 pip for EURUSD on perpetual)
            spread_estimate = 0.00001  # ~0.1 pips
            bid = last_price - (spread_estimate / 2)
            ask = last_price + (spread_estimate / 2)
            mid = last_price
            spread_pips = (ask - bid) * 10000  # For EURUSD
            
            return {
                'bid': bid,
                'ask': ask,
                'mid': mid,
                'spread_pips': spread_pips,
                'time': datetime.utcnow()
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Error fetching current price from BingX Perpetual: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Error parsing BingX Perpetual price: {e}")
            return None
    
    def pips_to_price(self, pips: float) -> float:
        """Convert pips to price difference for EURUSD"""
        return pips * 0.0001
    
    def price_to_pips(self, price_diff: float) -> float:
        """Convert price difference to pips for EURUSD"""
        return price_diff * 10000
