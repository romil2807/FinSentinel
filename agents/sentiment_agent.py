"""
Sentiment & Intent Analysis Agent using LLMs
"""
import openai
import anthropic
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import logging
from typing import Dict, Optional, Tuple
import json
import time
from datetime import datetime

from config import OPENAI_API_KEY, ANTHROPIC_API_KEY, SENTIMENT_THRESHOLDS
from utils.database import db

logger = logging.getLogger(__name__)

class SentimentAgent:
    def __init__(self):
        self.setup_clients()
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
    def setup_clients(self):
        """Initialize AI clients"""
        self.openai_client = None
        self.anthropic_client = None
        
        if OPENAI_API_KEY:
            try:
                openai.api_key = OPENAI_API_KEY
                self.openai_client = openai
                logger.info("OpenAI client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
        
        if ANTHROPIC_API_KEY:
            try:
                self.anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
                logger.info("Anthropic client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Anthropic client: {e}")
    
    def get_sentiment_prompt(self, tweet_content: str, influencer: str) -> str:
        """Create a detailed prompt for sentiment analysis"""
        return f"""
Analyze the following tweet from financial influencer @{influencer} for market sentiment and investment intent.

Tweet: "{tweet_content}"

Please provide:
1. Overall sentiment (Very Positive, Positive, Neutral, Negative, Very Negative)
2. Confidence score (0.0 to 1.0)
3. Investment intent (Bullish, Bearish, Neutral, Uncertain)
4. Key sentiment indicators (specific words/phrases that drove the sentiment)
5. Market context (if applicable)

Format your response as JSON:
{{
    "sentiment": "Positive",
    "confidence": 0.85,
    "intent": "Bullish", 
    "indicators": ["game-changer", "revolutionary", "excited"],
    "market_context": "Product announcement likely to drive stock price",
    "reasoning": "Strong positive language with specific product claims"
}}

Consider:
- The influencer's typical communication style
- Financial/market context of mentioned companies
- Implicit vs explicit sentiment
- Sarcasm or irony (especially common in crypto Twitter)
"""

    def analyze_with_gpt4(self, tweet_content: str, influencer: str) -> Optional[Dict]:
        """Analyze sentiment using GPT-4"""
        if not self.openai_client:
            return None
            
        try:
            prompt = self.get_sentiment_prompt(tweet_content, influencer)
            
            response = self.openai_client.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert financial sentiment analyst. Provide accurate, nuanced analysis of market-related social media posts."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            result_text = response.choices[0].message.content
            
            # Parse JSON response
            try:
                result = json.loads(result_text)
                return {
                    'sentiment': result.get('sentiment', 'Neutral'),
                    'confidence': float(result.get('confidence', 0.5)),
                    'intent': result.get('intent', 'Neutral'),
                    'indicators': result.get('indicators', []),
                    'reasoning': result.get('reasoning', ''),
                    'model': 'gpt-4'
                }
            except json.JSONDecodeError:
                logger.warning("GPT-4 response was not valid JSON, using fallback parsing")
                return self.parse_text_response(result_text, 'gpt-4')
                
        except Exception as e:
            logger.error(f"Error in GPT-4 sentiment analysis: {e}")
            return None
    
    def analyze_with_claude(self, tweet_content: str, influencer: str) -> Optional[Dict]:
        """Analyze sentiment using Claude"""
        if not self.anthropic_client:
            return None
            
        try:
            prompt = self.get_sentiment_prompt(tweet_content, influencer)
            
            response = self.anthropic_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=500,
                temperature=0.3,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            result_text = response.content[0].text
            
            # Parse JSON response
            try:
                result = json.loads(result_text)
                return {
                    'sentiment': result.get('sentiment', 'Neutral'),
                    'confidence': float(result.get('confidence', 0.5)),
                    'intent': result.get('intent', 'Neutral'),
                    'indicators': result.get('indicators', []),
                    'reasoning': result.get('reasoning', ''),
                    'model': 'claude-3-sonnet'
                }
            except json.JSONDecodeError:
                logger.warning("Claude response was not valid JSON, using fallback parsing")
                return self.parse_text_response(result_text, 'claude-3-sonnet')
                
        except Exception as e:
            logger.error(f"Error in Claude sentiment analysis: {e}")
            return None
    
    def parse_text_response(self, text: str, model: str) -> Dict:
        """Fallback parser for non-JSON responses"""
        # Simple keyword-based parsing as fallback
        text_lower = text.lower()
        
        # Extract sentiment
        if 'very positive' in text_lower or 'extremely positive' in text_lower:
            sentiment = 'Very Positive'
            confidence = 0.9
        elif 'positive' in text_lower or 'bullish' in text_lower:
            sentiment = 'Positive'
            confidence = 0.7
        elif 'negative' in text_lower or 'bearish' in text_lower:
            sentiment = 'Negative' 
            confidence = 0.7
        elif 'very negative' in text_lower or 'extremely negative' in text_lower:
            sentiment = 'Very Negative'
            confidence = 0.9
        else:
            sentiment = 'Neutral'
            confidence = 0.5
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'intent': 'Neutral',
            'indicators': [],
            'reasoning': f'Fallback parsing from {model}',
            'model': f'{model}_fallback'
        }
    
    def analyze_with_vader(self, tweet_content: str) -> Dict:
        """Analyze sentiment using VADER (fallback method)"""
        try:
            scores = self.vader_analyzer.polarity_scores(tweet_content)
            compound_score = scores['compound']
            
            # Map VADER scores to our sentiment labels
            if compound_score >= SENTIMENT_THRESHOLDS['very_positive']:
                sentiment = 'Very Positive'
            elif compound_score >= SENTIMENT_THRESHOLDS['positive']:
                sentiment = 'Positive'
            elif compound_score <= SENTIMENT_THRESHOLDS['very_negative']:
                sentiment = 'Very Negative'
            elif compound_score <= SENTIMENT_THRESHOLDS['negative']:
                sentiment = 'Negative'
            else:
                sentiment = 'Neutral'
            
            return {
                'sentiment': sentiment,
                'confidence': abs(compound_score),
                'intent': 'Bullish' if compound_score > 0 else 'Bearish' if compound_score < 0 else 'Neutral',
                'indicators': [],
                'reasoning': f'VADER compound score: {compound_score:.3f}',
                'model': 'vader'
            }
            
        except Exception as e:
            logger.error(f"Error in VADER sentiment analysis: {e}")
            return {
                'sentiment': 'Neutral',
                'confidence': 0.0,
                'intent': 'Neutral',
                'indicators': [],
                'reasoning': f'VADER analysis failed: {e}',
                'model': 'vader_error'
            }
    
    def analyze_sentiment(self, tweet_content: str, influencer: str) -> Dict:
        """Main sentiment analysis method with fallback chain"""
        logger.debug(f"Analyzing sentiment for tweet from {influencer}")
        
        # Try GPT-4 first (most sophisticated)
        result = self.analyze_with_gpt4(tweet_content, influencer)
        if result and result['confidence'] > 0.3:
            logger.debug("Used GPT-4 for sentiment analysis")
            return result
        
        # Try Claude as backup
        result = self.analyze_with_claude(tweet_content, influencer)
        if result and result['confidence'] > 0.3:
            logger.debug("Used Claude for sentiment analysis")
            return result
        
        # Fall back to VADER
        logger.debug("Using VADER as fallback for sentiment analysis")
        return self.analyze_with_vader(tweet_content)
    
    def convert_sentiment_to_score(self, sentiment: str) -> float:
        """Convert sentiment label to numerical score (-1 to 1)"""
        sentiment_mapping = {
            'Very Positive': 0.8,
            'Positive': 0.4,
            'Neutral': 0.0,
            'Negative': -0.4,
            'Very Negative': -0.8
        }
        return sentiment_mapping.get(sentiment, 0.0)
    
    def process_tweet(self, tweet_id: str, tweet_content: str, influencer: str) -> Dict:
        """Process a single tweet through sentiment analysis"""
        try:
            # Perform sentiment analysis
            sentiment_result = self.analyze_sentiment(tweet_content, influencer)
            
            # Convert to standard format for database
            sentiment_data = {
                'score': self.convert_sentiment_to_score(sentiment_result['sentiment']),
                'label': sentiment_result['sentiment'],
                'confidence': sentiment_result['confidence'],
                'model': sentiment_result['model']
            }
            
            # Save to database
            db.insert_sentiment(tweet_id, sentiment_data)
            
            logger.info(f"Processed sentiment for tweet {tweet_id}: {sentiment_result['sentiment']} ({sentiment_result['confidence']:.2f})")
            
            return {
                'tweet_id': tweet_id,
                'sentiment_data': sentiment_data,
                'analysis_details': sentiment_result
            }
            
        except Exception as e:
            logger.error(f"Error processing tweet {tweet_id}: {e}")
            raise
    
    def batch_process_tweets(self, tweet_data_list: list) -> list:
        """Process multiple tweets in batch"""
        results = []
        
        for tweet_data in tweet_data_list:
            try:
                result = self.process_tweet(
                    tweet_data['tweet_id'],
                    tweet_data['content'], 
                    tweet_data['influencer']
                )
                results.append(result)
                
                # Small delay to respect API rate limits
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error in batch processing tweet {tweet_data.get('tweet_id', 'unknown')}: {e}")
                continue
        
        logger.info(f"Batch processed {len(results)} tweets for sentiment analysis")
        return results

# Global instance
sentiment_agent = SentimentAgent()
