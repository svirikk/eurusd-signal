"""
Telegram Notifier
Sends formatted trading signals to Telegram
"""
import requests
import logging
from typing import Optional
from datetime import datetime
from strategy import TradingSignal
from config import Config


logger = logging.getLogger(__name__)


class TelegramNotifier:
    """Handles Telegram notifications"""
    
    def __init__(self):
        self.bot_token = Config.TELEGRAM_BOT_TOKEN
        self.chat_id = Config.TELEGRAM_CHAT_ID
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
    
    def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """
        Send message to Telegram
        
        Args:
            text: Message text
            parse_mode: HTML or Markdown
            
        Returns:
            True if sent successfully
        """
        url = f"{self.base_url}/sendMessage"
        
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True
        }
        
        try:
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            logger.info("✅ Telegram message sent successfully")
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Failed to send Telegram message: {e}")
            return False
    
    def format_signal_message(self, signal: TradingSignal) -> str:
        """
        Format trading signal as detailed message
        
        Args:
            signal: TradingSignal object
            
        Returns:
            Formatted message string
        """
        ctx = signal.context
        
        # Emoji based on direction
        emoji = "🔴" if signal.type == "SELL" else "🟢"
        
        message = f"""
{emoji} <b>EURUSD — {signal.type}</b> (M5)
━━━━━━━━━━━━━━━━━━━━━━

🕒 <b>Time:</b> {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')} UTC

<b>1️⃣ Session Context:</b>
   • Asia Range High: {ctx['asia_range']['high']:.5f}
   • Asia Range Low: {ctx['asia_range']['low']:.5f}
   • Asia Range Mid: {ctx['asia_range']['mid']:.5f}
   • London Session: ACTIVE ✅

<b>2️⃣ Liquidity Sweep:</b>
   • Sweep Type: {ctx['sweep']['type'].upper()}
   • Swept Level: {ctx['sweep']['swept_level']:.5f}
   • Sweep Candle: {ctx['sweep']['candle_time'].strftime('%H:%M')}
   • High: {ctx['sweep']['candle_high']:.5f}
   • Low: {ctx['sweep']['candle_low']:.5f}
   • Close: {ctx['sweep']['candle_close']:.5f}

<b>3️⃣ Market Structure Shift (CHOCH):</b>
   • CHOCH Detected: YES ✅
   • Type: {ctx['choch']['type'].replace('_', ' ').title()}
   • Broken Level: {ctx['choch']['broken_level']:.5f}
   • Confirmation Time: {ctx['choch']['time'].strftime('%H:%M')}

<b>4️⃣ Entry Model (FVG):</b>
   • FVG Zone: {ctx['fvg']['low']:.5f} - {ctx['fvg']['high']:.5f}
   • FVG Size: {ctx['fvg']['size_pips']:.1f} pips
   • Entry Price: <b>{signal.entry:.5f}</b> (midpoint)

<b>5️⃣ Risk Management:</b>
   • Stop Loss: {signal.stop_loss:.5f}
   • Take Profit 1: {signal.take_profit:.5f}
   • Risk: <b>{signal.risk_pips:.1f} pips</b>
   • Reward: <b>{signal.reward_pips:.1f} pips</b>
   • Risk/Reward: <b>1:{signal.risk_reward:.2f}</b> 📊

<b>6️⃣ Trade Reasons:</b>
   ✓ Asia liquidity swept + close back in range
   ✓ CHOCH confirms market structure shift
   ✓ FVG provides optimal entry zone
   ✓ TP targets opposite side of Asia range
   ✓ London session volatility supports move

<b>⚠️ DISCLAIMER:</b>
<i>This is a signal alert only. Manual execution required.
Always verify setup on your charts before entering.
Never risk more than you can afford to lose.</i>

━━━━━━━━━━━━━━━━━━━━━━
Signal ID: <code>{signal.hash[:8]}</code>
"""
        
        return message.strip()
    
    def send_signal(self, signal: TradingSignal) -> bool:
        """
        Send trading signal to Telegram
        
        Args:
            signal: TradingSignal object
            
        Returns:
            True if sent successfully
        """
        message = self.format_signal_message(signal)
        return self.send_message(message)
    
    def send_startup_message(self):
        """Send bot startup notification"""
        message = """
🤖 <b>EURUSD Trading Bot Started</b>

✅ Bot is now monitoring EURUSD
📊 Strategy: Asia Range Sweep → CHOCH → FVG
⏰ Running 24/7

Waiting for valid setups...
"""
        self.send_message(message.strip())
    
    def send_error_message(self, error: str):
        """Send error notification"""
        message = f"""
⚠️ <b>Bot Error</b>

Error: {error}

Bot will continue attempting to run.
"""
        self.send_message(message.strip())
    
    def send_daily_summary(self, signals_sent: int):
        """Send daily summary (optional)"""
        message = f"""
📊 <b>Daily Summary</b>

Signals sent today: {signals_sent}
Status: Running ✅

Next update: Tomorrow
"""
        self.send_message(message.strip())
