"""
Alert system for FinSentinel - Slack and Discord notifications
"""
import json
import logging
from typing import Dict, List, Optional
import requests
from datetime import datetime
from dataclasses import dataclass

from config import SLACK_WEBHOOK_URL, DISCORD_WEBHOOK_URL

logger = logging.getLogger(__name__)

@dataclass
class AlertData:
    asset_symbol: str
    signal: str
    strength: float
    influencer: str
    tweet_content: str
    reasoning: str
    timestamp: datetime

class AlertManager:
    def __init__(self):
        self.slack_webhook = SLACK_WEBHOOK_URL
        self.discord_webhook = DISCORD_WEBHOOK_URL
        
        # Alert thresholds
        self.min_strength_threshold = 0.7  # Only alert on high-confidence signals
        self.asset_priorities = {
            # High priority assets (always alert)
            'TSLA': 1,
            'BTC': 1, 
            'ETH': 1,
            'AAPL': 1,
            'GOOGL': 1,
            'MSFT': 1,
            'NVDA': 1,
            # Medium priority (alert if strong signal)
            'META': 2,
            'AMZN': 2,
            'NFLX': 2,
            'COIN': 2,
            # Low priority (only very strong signals)
            'DOGE': 3,
            'SHIB': 3
        }
    
    def should_send_alert(self, alert_data: AlertData) -> bool:
        """Determine if an alert should be sent based on criteria"""
        # Check minimum strength threshold
        if alert_data.strength < self.min_strength_threshold:
            return False
        
        # Check asset priority
        asset_priority = self.asset_priorities.get(alert_data.asset_symbol, 3)
        
        if asset_priority == 1:  # High priority - always alert if above threshold
            return True
        elif asset_priority == 2:  # Medium priority - need stronger signal
            return alert_data.strength >= 0.8
        else:  # Low priority - need very strong signal
            return alert_data.strength >= 0.9
    
    def format_slack_message(self, alert_data: AlertData) -> Dict:
        """Format alert for Slack"""
        # Choose emoji and color based on signal
        if alert_data.signal == 'BUY':
            emoji = "🟢"
            color = "#28a745"
        elif alert_data.signal == 'SELL':
            emoji = "🔴" 
            color = "#dc3545"
        else:
            emoji = "🟡"
            color = "#ffc107"
        
        # Strength indicators
        strength_bars = "█" * int(alert_data.strength * 5)
        strength_empty = "░" * (5 - int(alert_data.strength * 5))
        strength_display = f"{strength_bars}{strength_empty} {alert_data.strength:.2f}"
        
        # Truncate tweet content
        tweet_preview = alert_data.tweet_content[:200] + "..." if len(alert_data.tweet_content) > 200 else alert_data.tweet_content
        
        return {
            "username": "FinSentinel Bot",
            "icon_emoji": ":chart_with_upwards_trend:",
            "attachments": [
                {
                    "color": color,
                    "title": f"{emoji} {alert_data.signal} Signal: {alert_data.asset_symbol}",
                    "title_link": f"https://finance.yahoo.com/quote/{alert_data.asset_symbol}",
                    "fields": [
                        {
                            "title": "Influencer",
                            "value": f"@{alert_data.influencer}",
                            "short": True
                        },
                        {
                            "title": "Signal Strength",
                            "value": strength_display,
                            "short": True
                        },
                        {
                            "title": "Tweet",
                            "value": f"```{tweet_preview}```",
                            "short": False
                        },
                        {
                            "title": "AI Reasoning",
                            "value": alert_data.reasoning,
                            "short": False
                        }
                    ],
                    "footer": "FinSentinel AI",
                    "ts": int(alert_data.timestamp.timestamp())
                }
            ]
        }
    
    def format_discord_message(self, alert_data: AlertData) -> Dict:
        """Format alert for Discord"""
        # Choose color based on signal
        if alert_data.signal == 'BUY':
            color = 0x28a745  # Green
            emoji = "🟢"
        elif alert_data.signal == 'SELL':
            color = 0xdc3545  # Red
            emoji = "🔴"
        else:
            color = 0xffc107  # Yellow
            emoji = "🟡"
        
        # Strength display
        strength_percentage = f"{alert_data.strength * 100:.1f}%"
        
        # Truncate content
        tweet_preview = alert_data.tweet_content[:300] + "..." if len(alert_data.tweet_content) > 300 else alert_data.tweet_content
        
        return {
            "embeds": [
                {
                    "title": f"{emoji} {alert_data.signal} Signal: {alert_data.asset_symbol}",
                    "description": f"AI-detected trading signal from @{alert_data.influencer}",
                    "color": color,
                    "fields": [
                        {
                            "name": "📊 Signal Strength",
                            "value": strength_percentage,
                            "inline": True
                        },
                        {
                            "name": "👤 Influencer", 
                            "value": f"@{alert_data.influencer}",
                            "inline": True
                        },
                        {
                            "name": "💬 Tweet",
                            "value": f"```{tweet_preview}```",
                            "inline": False
                        },
                        {
                            "name": "🤖 AI Analysis",
                            "value": alert_data.reasoning,
                            "inline": False
                        }
                    ],
                    "footer": {
                        "text": "FinSentinel AI • Real-time Social Sentiment Analysis"
                    },
                    "timestamp": alert_data.timestamp.isoformat(),
                    "url": f"https://finance.yahoo.com/quote/{alert_data.asset_symbol}"
                }
            ]
        }
    
    def send_slack_alert(self, alert_data: AlertData) -> bool:
        """Send alert to Slack"""
        if not self.slack_webhook:
            logger.warning("Slack webhook URL not configured")
            return False
        
        try:
            message = self.format_slack_message(alert_data)
            
            response = requests.post(
                self.slack_webhook,
                json=message,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"Slack alert sent for {alert_data.asset_symbol} {alert_data.signal}")
                return True
            else:
                logger.error(f"Slack alert failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending Slack alert: {e}")
            return False
    
    def send_discord_alert(self, alert_data: AlertData) -> bool:
        """Send alert to Discord"""
        if not self.discord_webhook:
            logger.warning("Discord webhook URL not configured")
            return False
        
        try:
            message = self.format_discord_message(alert_data)
            
            response = requests.post(
                self.discord_webhook,
                json=message,
                timeout=10
            )
            
            if response.status_code in [200, 204]:
                logger.info(f"Discord alert sent for {alert_data.asset_symbol} {alert_data.signal}")
                return True
            else:
                logger.error(f"Discord alert failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending Discord alert: {e}")
            return False
    
    def send_alert(self, alert_data: AlertData) -> Dict[str, bool]:
        """Send alert through all configured channels"""
        if not self.should_send_alert(alert_data):
            logger.debug(f"Alert criteria not met for {alert_data.asset_symbol}")
            return {'slack': False, 'discord': False, 'sent': False}
        
        results = {
            'slack': False,
            'discord': False,
            'sent': False
        }
        
        # Send to Slack
        if self.slack_webhook:
            results['slack'] = self.send_slack_alert(alert_data)
        
        # Send to Discord
        if self.discord_webhook:
            results['discord'] = self.send_discord_alert(alert_data)
        
        # Mark as sent if any channel succeeded
        results['sent'] = results['slack'] or results['discord']
        
        if results['sent']:
            logger.info(f"Alert sent for {alert_data.asset_symbol} {alert_data.signal} (strength: {alert_data.strength:.2f})")
        else:
            logger.warning(f"Failed to send alert for {alert_data.asset_symbol} {alert_data.signal}")
        
        return results
    
    def send_summary_alert(self, signals: List[Dict], period_hours: int = 1) -> bool:
        """Send a summary of recent signals"""
        if not signals:
            return False
        
        # Group signals by type
        buy_signals = [s for s in signals if s['signal'] == 'BUY']
        sell_signals = [s for s in signals if s['signal'] == 'SELL']
        
        if not buy_signals and not sell_signals:
            return False
        
        # Create summary message
        summary_lines = [
            f"📈 **FinSentinel Summary - Last {period_hours}h**",
            "",
            f"🟢 **BUY Signals:** {len(buy_signals)}",
            f"🔴 **SELL Signals:** {len(sell_signals)}",
            ""
        ]
        
        # Add top signals
        if buy_signals:
            top_buys = sorted(buy_signals, key=lambda x: x['strength'], reverse=True)[:3]
            summary_lines.append("**Top BUY Signals:**")
            for signal in top_buys:
                summary_lines.append(f"• {signal['asset_symbol']} ({signal['strength']:.2f}) - @{signal['influencer']}")
            summary_lines.append("")
        
        if sell_signals:
            top_sells = sorted(sell_signals, key=lambda x: x['strength'], reverse=True)[:3]
            summary_lines.append("**Top SELL Signals:**")
            for signal in top_sells:
                summary_lines.append(f"• {signal['asset_symbol']} ({signal['strength']:.2f}) - @{signal['influencer']}")
        
        summary_text = "\n".join(summary_lines)
        
        # Send to Slack
        slack_success = False
        if self.slack_webhook:
            try:
                slack_message = {
                    "username": "FinSentinel Bot",
                    "icon_emoji": ":chart_with_upwards_trend:",
                    "text": summary_text
                }
                
                response = requests.post(self.slack_webhook, json=slack_message, timeout=10)
                slack_success = response.status_code == 200
                
            except Exception as e:
                logger.error(f"Error sending Slack summary: {e}")
        
        # Send to Discord
        discord_success = False
        if self.discord_webhook:
            try:
                discord_message = {
                    "embeds": [
                        {
                            "title": f"📈 FinSentinel Summary - Last {period_hours}h",
                            "description": summary_text,
                            "color": 0x1f77b4,
                            "footer": {
                                "text": "FinSentinel AI"
                            },
                            "timestamp": datetime.now().isoformat()
                        }
                    ]
                }
                
                response = requests.post(self.discord_webhook, json=discord_message, timeout=10)
                discord_success = response.status_code in [200, 204]
                
            except Exception as e:
                logger.error(f"Error sending Discord summary: {e}")
        
        return slack_success or discord_success
    
    def test_alerts(self) -> Dict[str, bool]:
        """Test alert system with sample data"""
        test_alert = AlertData(
            asset_symbol="TSLA",
            signal="BUY",
            strength=0.85,
            influencer="elonmusk",
            tweet_content="Tesla's new battery technology is a game-changer for the industry. Revolutionary energy density!",
            reasoning="Strong positive sentiment from CEO with historical 65% accuracy on Tesla-related predictions. Battery announcements typically drive 5-10% short-term gains.",
            timestamp=datetime.now()
        )
        
        return self.send_alert(test_alert)

# Global alert manager instance
alert_manager = AlertManager()
