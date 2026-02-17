"""
BingX TradFi Forex Perpetual API Client
Handles all interactions with BingX TradFi Forex Perpetual Futures (EURUSD)
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
        # ✅ ФІКС: utcfromtimestamp замість fromtimestamp (UTC завжди!)
        self.time = datetime.utcfromtimestamp(int(data['time']) / 1000)
        self.open = float(data['open'])
        self.high = float(data['high'])
        self.low = float(data['low'])
        self.close = float(data['close'])
        self.volume = float(data['volume'])
        self.complete = True
    
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
        self.symbol = None           # ✅ Визначається динамічно!
        self._verified_symbol = False
        
    def verify_symbol(self) -> bool:
        """
        ✅ Автоматично знаходить правильний символ EURUSD на BingX
        Реальний символ: NCFXEUR2USD-USDT (не EURUSD-USDT!)
        """
        if self._verified_symbol and self.symbol:
            return True
            
        logger.info("🔍 Verifying EURUSD symbol format on BingX...")
        
        endpoint = f"{self.base_url}/openApi/swap/v2/quote/contracts"
        
        try:
            response = requests.get(endpoint, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get('code') != 0:
                logger.error(f"❌ API error getting contracts: {data.get('msg')}")
                return False
            
            contracts = data.get('data', [])
            
            # Шукаємо EURUSD серед всіх контрактів
            for contract in contracts:
                symbol = contract.get('symbol', '')
                if ('EUR' in symbol and 'USD' in symbol and 
                    'JPY' not in symbol and 'BTC' not in symbol and 
                    'ETH' not in symbol):
                    self.symbol = symbol
                    self._verified_symbol = True
                    logger.info(f"✅ Found EURUSD symbol: {self.symbol}")
                    return True
            
            # Якщо не знайшли в contracts - пробуємо напряму
            logger.warning("⚠️ EURUSD not found in contracts. Trying direct test...")
            for test_symbol in ['NCFXEUR2USD-USDT', 'EURUSD-USDT', 'EUR-USD', 'EURUSD']:
                if self._test_symbol(test_symbol):
                    self.symbol = test_symbol
                    self._verified_symbol = True
                    logger.info(f"✅ Found working symbol: {self.symbol}")
                    return True
            
            logger.error("❌ Could not find EURUSD symbol. Market might be closed!")
            return False
            
        except Exception as e:
            logger.error(f"❌ Error verifying symbol: {e}")
            self.symbol = Config.SYMBOL
            return False
    
    def _test_symbol(self, test_symbol: str) -> bool:
        """Тестує чи працює символ"""
        try:
            response = requests.get(
                f"{self.base_url}/openApi/swap/v2/quote/ticker",
                params={"symbol": test_symbol},
                timeout=5
            )
            return response.json().get('code') == 0
        except:
            return False
    
    def get_candles(self, interval: str, limit: int = 500) -> List[Candle]:
        """Fetch candles from BingX"""
        
        # ✅ Перевіряємо символ перед кожним запитом
        if not self.symbol or not self._verified_symbol:
            if not self.verify_symbol():
                logger.error("❌ Cannot fetch candles: symbol not found")
                return []
        
        endpoint = f"{self.base_url}/openApi/swap/v2/quote/klines"
        params = {
            "symbol": self.symbol,
            "interval": interval,
            "limit": min(limit, 1440)
        }
        
        try:
            response = requests.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get('code') != 0:
                error_msg = data.get('msg', 'Unknown error')
                logger.error(f"❌ BingX API error: {error_msg}")
                
                if 'not exist' in error_msg.lower():
                    logger.warning("⏰ Market might be closed (weekends/holidays)")
                    # Скидаємо щоб знайти знову наступного разу
                    self._verified_symbol = False
                    self.symbol = None
                
                return []
            
            kline_data = data.get('data', [])
            
            if not kline_data:
                logger.warning(f"⚠️ No candle data - market might be closed")
                return []
            
            candles = [Candle(k) for k in kline_data]
            logger.info(f"✅ Fetched {len(candles)} {interval} candles from BingX Perpetual")
            return candles
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Network error: {e}")
            return []
        except Exception as e:
            logger.error(f"❌ Error parsing candles: {e}")
            return []
    
    def get_latest_candles(self, interval: str, count: int = 100) -> List[Candle]:
        """Get latest candles"""
        return self.get_candles(interval, count)
    
    def get_current_price(self) -> Optional[Dict[str, float]]:
        """Get current price"""
        
        if not self.symbol or not self._verified_symbol:
            if not self.verify_symbol():
                return None
        
        try:
            response = requests.get(
                f"{self.base_url}/openApi/swap/v2/quote/ticker",
                params={"symbol": self.symbol},
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            if data.get('code') != 0:
                logger.error(f"❌ BingX API error: {data.get('msg', 'Unknown error')}")
                return None
            
            last_price = float(data.get('data', {}).get('lastPrice', 0))
            if last_price <= 0:
                return None
            
            spread_estimate = 0.00001
            bid = last_price - (spread_estimate / 2)
            ask = last_price + (spread_estimate / 2)
            
            return {
                'bid': bid,
                'ask': ask,
                'mid': last_price,
                'spread_pips': (ask - bid) * 10000,
                'time': datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"❌ Error fetching price: {e}")
            return None
    
    def pips_to_price(self, pips: float) -> float:
        return pips * 0.0001
    
    def price_to_pips(self, price_diff: float) -> float:
        return price_diff * 10000
