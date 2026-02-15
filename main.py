"""
EURUSD Trading Signal Bot - Main Entry Point (BingX TradFi Forex Perpetual)
Monitors market 24/7 and sends Telegram signals
"""
import logging
import time
import sys
from datetime import datetime
from config_bingx import Config
from bingx_client import BingXClient
from strategy import StrategyEngine
from notifier import TelegramNotifier


# Configure logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)


class TradingBot:
    """Main trading bot coordinator"""
    
    def __init__(self):
        logger.info("🚀 Initializing EURUSD Trading Bot (BingX)...")
        
        # Validate configuration
        if not Config.validate():
            logger.error("❌ Configuration validation failed. Exiting.")
            sys.exit(1)
        
        # Initialize components
        self.bingx = BingXClient()
        self.strategy = StrategyEngine(self.bingx)
        self.notifier = TelegramNotifier()
        
        # Bot state
        self.is_running = False
        self.error_count = 0
        self.max_errors = 10
        
        logger.info("✅ Bot initialized successfully")
    
    def run_once(self):
        """Execute one iteration of the bot logic"""
        try:
            # Update market data
            self.strategy.update_candles()
            
            # Generate signal if conditions are met
            signal = self.strategy.generate_signal()
            
            if signal:
                logger.info(f"📢 New signal generated: {signal.type}")
                
                # Send to Telegram
                success = self.notifier.send_signal(signal)
                
                if success:
                    logger.info(f"✅ Signal sent successfully: {signal.type} @ {signal.entry:.5f}")
                else:
                    logger.error("❌ Failed to send signal to Telegram")
            
            # Reset error count on successful run
            self.error_count = 0
            
        except KeyboardInterrupt:
            raise
        
        except Exception as e:
            self.error_count += 1
            logger.error(f"❌ Error in run_once: {e}", exc_info=True)
            
            if self.error_count >= self.max_errors:
                logger.critical(f"❌ Too many errors ({self.error_count}). Notifying and exiting.")
                self.notifier.send_error_message(f"Critical: {self.error_count} consecutive errors. Bot stopped.")
                raise
            
            # Notify on first error
            if self.error_count == 1:
                self.notifier.send_error_message(str(e))
    
    def run(self):
        """Main bot loop - runs 24/7"""
        self.is_running = True
        
        logger.info("=" * 60)
        logger.info("🤖 EURUSD TRADING SIGNAL BOT (BingX TradFi)")
        logger.info("=" * 60)
        logger.info(f"📊 Symbol: {Config.SYMBOL} (TradFi Forex Perpetual)")
        logger.info(f"🌐 Exchange: BingX TradFi Forex Perpetual")
        logger.info(f"⏰ Update Interval: {Config.UPDATE_INTERVAL}s")
        logger.info(f"📈 Strategy: Asia Range Sweep → CHOCH → FVG Entry")
        logger.info("=" * 60)
        
        # Send startup notification
        self.notifier.send_startup_message()
        
        logger.info("🟢 Bot is now running. Press Ctrl+C to stop.")
        logger.info("")
        
        try:
            while self.is_running:
                cycle_start = time.time()
                
                # Run bot logic
                self.run_once()
                
                # Calculate sleep time
                elapsed = time.time() - cycle_start
                sleep_time = max(0, Config.UPDATE_INTERVAL - elapsed)
                
                if sleep_time > 0:
                    logger.debug(f"💤 Sleeping for {sleep_time:.1f}s...")
                    time.sleep(sleep_time)
        
        except KeyboardInterrupt:
            logger.info("\n🛑 Received shutdown signal (Ctrl+C)")
            self.shutdown()
        
        except Exception as e:
            logger.critical(f"❌ Fatal error in main loop: {e}", exc_info=True)
            self.shutdown()
            sys.exit(1)
    
    def shutdown(self):
        """Graceful shutdown"""
        logger.info("🛑 Shutting down bot...")
        self.is_running = False
        logger.info("✅ Bot stopped successfully")


def main():
    """Main entry point"""
    try:
        bot = TradingBot()
        bot.run()
    except Exception as e:
        logger.critical(f"❌ Failed to start bot: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
