"""
Decision Agent for generating BUY/SELL/HOLD signals with reasoning
"""
import openai
import anthropic
import json
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass

from config import OPENAI_API_KEY, ANTHROPIC_API_KEY, SIGNAL_RULES
from utils.database import db

logger = logging.getLogger(__name__)

@dataclass
class TradingSignal:
    asset_symbol: str
    signal: str  # 'BUY', 'SELL', 'HOLD'
    strength: float  # 0.0 to 1.0
    reasoning: str
    confidence: float
    risk_level: str  # 'LOW', 'MEDIUM', 'HIGH'
    time_horizon: str  # 'SHORT', 'MEDIUM', 'LONG'

class DecisionAgent:
    def __init__(self):
        self.setup_ai_clients()
        self.signal_rules = SIGNAL_RULES
        
        # Historical performance tracking (simplified)
        self.influencer_accuracy = {
            'elonmusk': 0.65,
            'cathiedwood': 0.72,
            'chamath': 0.58,
            'michael_saylor': 0.78,
            'naval': 0.70
        }
    
    def setup_ai_clients(self):
        """Initialize AI clients for decision reasoning"""
        self.openai_client = None
        self.anthropic_client = None
        
        if OPENAI_API_KEY:
            try:
                openai.api_key = OPENAI_API_KEY
                self.openai_client = openai
                logger.info("OpenAI client initialized for decision agent")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
        
        if ANTHROPIC_API_KEY:
            try:
                self.anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
                logger.info("Anthropic client initialized for decision agent")
            except Exception as e:
                logger.error(f"Failed to initialize Anthropic client: {e}")
    
    def get_decision_prompt(self, tweet_data: Dict, sentiment_data: Dict, asset_data: List[Dict]) -> str:
        """Create a detailed prompt for trading signal generation"""
        assets_str = ", ".join([f"{a['symbol']} ({a['type']})" for a in asset_data])
        
        return f"""
You are an expert financial analyst generating trading signals based on social media sentiment from influential figures.

TWEET ANALYSIS:
- Influencer: @{tweet_data['influencer']}
- Content: "{tweet_data['content']}"
- Timestamp: {tweet_data['timestamp']}

SENTIMENT ANALYSIS:
- Sentiment: {sentiment_data['label']} (Score: {sentiment_data['score']:.2f})
- Confidence: {sentiment_data['confidence']:.2f}

DETECTED ASSETS:
- {assets_str}

INFLUENCER TRACK RECORD:
- Historical accuracy: {self.influencer_accuracy.get(tweet_data['influencer'], 0.5):.0%}

TASK: Generate a trading signal for each detected asset. Consider:

1. SENTIMENT IMPACT: How does the sentiment translate to market movement?
2. INFLUENCER CREDIBILITY: Track record and market influence
3. ASSET TYPE: Stocks vs crypto react differently to social sentiment
4. MARKET CONTEXT: Current market conditions and timing
5. RISK FACTORS: Potential downside and volatility

For each asset, provide:
- Signal: BUY, SELL, or HOLD
- Strength: 0.0 (weak) to 1.0 (strong conviction)
- Risk Level: LOW, MEDIUM, HIGH
- Time Horizon: SHORT (hours-days), MEDIUM (days-weeks), LONG (weeks-months)
- Reasoning: Detailed explanation (2-3 sentences)

Format as JSON array:
[
  {{
    "asset_symbol": "TSLA",
    "signal": "BUY",
    "strength": 0.75,
    "risk_level": "MEDIUM",
    "time_horizon": "SHORT",
    "reasoning": "Elon Musk's positive announcement about Tesla technology typically drives 5-10% short-term gains. Given his 65% historical accuracy and strong positive sentiment, this represents a solid short-term opportunity with moderate risk."
  }}
]

IMPORTANT GUIDELINES:
- Be conservative with strength scores (>0.8 only for very strong signals)
- Consider that crypto is more volatile than stocks
- Factor in the influencer's specific expertise area
- Account for potential market manipulation or pump-and-dump scenarios
- Higher sentiment doesn't always mean higher signal strength
"""

    def generate_signals_with_ai(self, tweet_data: Dict, sentiment_data: Dict, asset_data: List[Dict]) -> List[TradingSignal]:
        """Generate trading signals using AI reasoning"""
        prompt = self.get_decision_prompt(tweet_data, sentiment_data, asset_data)
        
        # Try GPT-4 first
        if self.openai_client:
            try:
                response = self.openai_client.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a professional financial analyst specializing in social media sentiment analysis and trading signal generation. Always provide conservative, well-reasoned analysis."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    max_tokens=1000
                )
                
                result_text = response.choices[0].message.content
                
                try:
                    signals_data = json.loads(result_text)
                    return [self.create_trading_signal(**signal) for signal in signals_data]
                except json.JSONDecodeError:
                    logger.warning("GPT-4 response was not valid JSON")
                    
            except Exception as e:
                logger.error(f"Error using GPT-4 for signal generation: {e}")
        
        # Try Claude as backup
        if self.anthropic_client:
            try:
                response = self.anthropic_client.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=1000,
                    temperature=0.2,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                result_text = response.content[0].text
                
                try:
                    signals_data = json.loads(result_text)
                    return [self.create_trading_signal(**signal) for signal in signals_data]
                except json.JSONDecodeError:
                    logger.warning("Claude response was not valid JSON")
                    
            except Exception as e:
                logger.error(f"Error using Claude for signal generation: {e}")
        
        # Fall back to rule-based system
        return self.generate_signals_rule_based(tweet_data, sentiment_data, asset_data)
    
    def generate_signals_rule_based(self, tweet_data: Dict, sentiment_data: Dict, asset_data: List[Dict]) -> List[TradingSignal]:
        """Fallback rule-based signal generation"""
        signals = []
        
        sentiment_score = sentiment_data['score']
        sentiment_confidence = sentiment_data['confidence']
        influencer = tweet_data['influencer']
        
        # Get influencer credibility multiplier
        credibility = self.influencer_accuracy.get(influencer, 0.5)
        
        for asset in asset_data:
            asset_symbol = asset['symbol']
            asset_type = asset['type']
            asset_confidence = asset['confidence']
            
            # Calculate base signal strength
            base_strength = abs(sentiment_score) * sentiment_confidence * asset_confidence * credibility
            
            # Determine signal direction
            if sentiment_score > self.signal_rules['buy_threshold']:
                signal = 'BUY'
                strength = min(0.9, base_strength * 1.2)
            elif sentiment_score < self.signal_rules['sell_threshold']:
                signal = 'SELL'
                strength = min(0.9, base_strength * 1.2)
            else:
                signal = 'HOLD'
                strength = min(0.6, base_strength)
            
            # Adjust for asset type (crypto more volatile)
            if asset_type == 'crypto':
                strength *= 1.1  # Crypto responds more to social sentiment
                risk_level = 'HIGH' if strength > 0.6 else 'MEDIUM'
            else:
                risk_level = 'MEDIUM' if strength > 0.7 else 'LOW'
            
            # Determine time horizon based on strength and asset type
            if strength > 0.7:
                time_horizon = 'SHORT'
            elif strength > 0.4:
                time_horizon = 'MEDIUM'
            else:
                time_horizon = 'LONG'
            
            # Generate reasoning
            reasoning = self.generate_rule_based_reasoning(
                signal, strength, influencer, sentiment_data['label'], asset_symbol, asset_type
            )
            
            trading_signal = TradingSignal(
                asset_symbol=asset_symbol,
                signal=signal,
                strength=strength,
                reasoning=reasoning,
                confidence=base_strength,
                risk_level=risk_level,
                time_horizon=time_horizon
            )
            
            signals.append(trading_signal)
        
        return signals
    
    def create_trading_signal(self, **kwargs) -> TradingSignal:
        """Create TradingSignal from kwargs with validation"""
        return TradingSignal(
            asset_symbol=kwargs.get('asset_symbol', ''),
            signal=kwargs.get('signal', 'HOLD'),
            strength=max(0.0, min(1.0, kwargs.get('strength', 0.5))),
            reasoning=kwargs.get('reasoning', 'No reasoning provided'),
            confidence=max(0.0, min(1.0, kwargs.get('confidence', 0.5))),
            risk_level=kwargs.get('risk_level', 'MEDIUM'),
            time_horizon=kwargs.get('time_horizon', 'MEDIUM')
        )
    
    def generate_rule_based_reasoning(self, signal: str, strength: float, influencer: str, 
                                    sentiment: str, asset: str, asset_type: str) -> str:
        """Generate reasoning for rule-based signals"""
        accuracy = self.influencer_accuracy.get(influencer, 0.5)
        
        base_reason = f"@{influencer} expressed {sentiment.lower()} sentiment about {asset}"
        
        if signal == 'BUY':
            return f"{base_reason}. With {accuracy:.0%} historical accuracy and {strength:.1f} signal strength, this suggests potential upward movement in the {asset_type} market."
        elif signal == 'SELL':
            return f"{base_reason}. Given {accuracy:.0%} track record and negative sentiment strength of {strength:.1f}, this indicates potential downward pressure."
        else:
            return f"{base_reason}. Neutral sentiment with {strength:.1f} strength suggests waiting for clearer signals before taking action."
    
    def apply_risk_management(self, signals: List[TradingSignal]) -> List[TradingSignal]:
        """Apply risk management rules to signals"""
        adjusted_signals = []
        
        for signal in signals:
            # Reduce strength for very new/volatile assets
            if signal.asset_symbol in ['DOGE', 'SHIB']:  # Meme coins
                signal.strength *= 0.8
                signal.risk_level = 'HIGH'
            
            # Boost strength for established assets with consistent influencers
            if (signal.asset_symbol in ['BTC', 'ETH', 'TSLA', 'AAPL'] and 
                signal.strength > 0.6):
                signal.strength = min(0.95, signal.strength * 1.1)
            
            # Minimum threshold filter
            if signal.strength > 0.3:  # Only include signals with reasonable confidence
                adjusted_signals.append(signal)
        
        return adjusted_signals
    
    def process_tweet_signals(self, tweet_id: str) -> List[Dict]:
        """Process a tweet and generate trading signals"""
        try:
            # Get tweet data from database
            with db.conn if hasattr(db, 'conn') else db.init_database() as conn:
                cursor = conn.cursor()
                
                # Get tweet, sentiment, and asset data
                cursor.execute("""
                    SELECT t.*, sa.sentiment_score, sa.sentiment_label, sa.confidence, sa.model_used
                    FROM tweets t
                    JOIN sentiment_analysis sa ON t.tweet_id = sa.tweet_id
                    WHERE t.tweet_id = ?
                """, (tweet_id,))
                
                tweet_row = cursor.fetchone()
                if not tweet_row:
                    logger.warning(f"No tweet data found for {tweet_id}")
                    return []
                
                # Convert to dict
                tweet_data = {
                    'tweet_id': tweet_row[1],
                    'timestamp': tweet_row[2],
                    'influencer': tweet_row[3],
                    'content': tweet_row[4]
                }
                
                sentiment_data = {
                    'score': tweet_row[6],
                    'label': tweet_row[7],
                    'confidence': tweet_row[8]
                }
                
                # Get asset mappings
                cursor.execute("""
                    SELECT asset_symbol, asset_type, confidence, detection_method
                    FROM asset_mappings
                    WHERE tweet_id = ?
                """, (tweet_id,))
                
                asset_rows = cursor.fetchall()
                asset_data = [{
                    'symbol': row[0],
                    'type': row[1],
                    'confidence': row[2],
                    'method': row[3]
                } for row in asset_rows]
                
                if not asset_data:
                    logger.info(f"No assets detected for tweet {tweet_id}")
                    return []
                
                # Generate signals
                signals = self.generate_signals_with_ai(tweet_data, sentiment_data, asset_data)
                
                # Apply risk management
                signals = self.apply_risk_management(signals)
                
                # Save signals to database
                saved_signals = []
                for signal in signals:
                    signal_data = {
                        'asset_symbol': signal.asset_symbol,
                        'signal': signal.signal,
                        'strength': signal.strength,
                        'reasoning': signal.reasoning
                    }
                    
                    db.insert_trading_signal(tweet_id, signal_data)
                    saved_signals.append({
                        'tweet_id': tweet_id,
                        'signal': signal
                    })
                
                logger.info(f"Generated {len(saved_signals)} trading signals for tweet {tweet_id}")
                return saved_signals
                
        except Exception as e:
            logger.error(f"Error processing signals for tweet {tweet_id}: {e}")
            return []
    
    def batch_process_signals(self, tweet_ids: List[str]) -> List[Dict]:
        """Process multiple tweets for signal generation"""
        all_signals = []
        
        for tweet_id in tweet_ids:
            try:
                signals = self.process_tweet_signals(tweet_id)
                all_signals.extend(signals)
            except Exception as e:
                logger.error(f"Error processing signals for tweet {tweet_id}: {e}")
                continue
        
        logger.info(f"Batch processed {len(all_signals)} trading signals")
        return all_signals
    
    def get_signal_summary(self, hours: int = 24) -> Dict:
        """Get summary of signals generated in the last N hours"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            # This would query the database for recent signals
            # For now, return a basic summary structure
            return {
                'total_signals': 0,
                'buy_signals': 0,
                'sell_signals': 0,
                'hold_signals': 0,
                'avg_strength': 0.0,
                'top_assets': [],
                'period_hours': hours
            }
            
        except Exception as e:
            logger.error(f"Error getting signal summary: {e}")
            return {}

# Global instance
decision_agent = DecisionAgent()
