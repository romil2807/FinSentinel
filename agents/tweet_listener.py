"""
Tweet Listener Agent for monitoring financial influencers
"""
import tweepy
import json
import time
import logging
from datetime import datetime
from typing import List, Dict, Optional
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import requests
from bs4 import BeautifulSoup

from config import TWITTER_BEARER_TOKEN, FINANCIAL_INFLUENCERS
from utils.database import db

logger = logging.getLogger(__name__)

class TweetListenerAgent:
    def __init__(self):
        self.bearer_token = TWITTER_BEARER_TOKEN
        self.influencers = FINANCIAL_INFLUENCERS
        self.setup_twitter_client()
        
    def setup_twitter_client(self):
        """Initialize Twitter API client"""
        if self.bearer_token:
            try:
                self.client = tweepy.Client(bearer_token=self.bearer_token)
                logger.info("Twitter API client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Twitter client: {e}")
                self.client = None
        else:
            logger.warning("No Twitter bearer token provided, using fallback methods")
            self.client = None
    
    def is_financial_tweet(self, tweet_text: str) -> bool:
        """Check if a tweet is related to finance/trading"""
        financial_keywords = [
            'stock', 'stocks', 'market', 'trading', 'buy', 'sell', 'hold',
            'bitcoin', 'crypto', 'eth', 'btc', 'investment', 'portfolio',
            'earnings', 'revenue', 'profit', 'loss', 'bull', 'bear',
            'nasdaq', 'sp500', 'dow', 'price', 'valuation', 'ipo',
            'tesla', 'apple', 'microsoft', 'nvidia', 'amazon', 'google',
            'meta', 'netflix', 'paypal', 'coinbase', 'square', 'zoom'
        ]
        
        tweet_lower = tweet_text.lower()
        return any(keyword in tweet_lower for keyword in financial_keywords)
    
    def extract_tweet_data(self, tweet) -> Dict:
        """Extract relevant data from tweet object"""
        try:
            return {
                'tweet_id': str(tweet.id),
                'timestamp': tweet.created_at,
                'influencer': tweet.author.username if hasattr(tweet, 'author') else 'unknown',
                'content': tweet.text,
                'raw_data': json.dumps({
                    'public_metrics': tweet.public_metrics if hasattr(tweet, 'public_metrics') else {},
                    'context_annotations': tweet.context_annotations if hasattr(tweet, 'context_annotations') else [],
                    'entities': tweet.entities if hasattr(tweet, 'entities') else {}
                })
            }
        except Exception as e:
            logger.error(f"Error extracting tweet data: {e}")
            return None
    
    def get_user_tweets(self, username: str, max_results: int = 10) -> List[Dict]:
        """Get recent tweets from a specific user"""
        tweets_data = []
        
        if not self.client:
            return self.fallback_get_user_tweets(username, max_results)
        
        try:
            # Get user ID
            user = self.client.get_user(username=username)
            if not user.data:
                logger.warning(f"User {username} not found")
                return tweets_data
            
            user_id = user.data.id
            
            # Get user's recent tweets
            tweets = self.client.get_users_tweets(
                id=user_id,
                max_results=max_results,
                tweet_fields=['created_at', 'public_metrics', 'context_annotations', 'entities'],
                user_fields=['username']
            )
            
            if tweets.data:
                for tweet in tweets.data:
                    # Add author info
                    tweet.author = user.data
                    
                    # Check if tweet is financial
                    if self.is_financial_tweet(tweet.text):
                        tweet_data = self.extract_tweet_data(tweet)
                        if tweet_data:
                            tweets_data.append(tweet_data)
                            
        except Exception as e:
            logger.error(f"Error getting tweets for {username}: {e}")
            # Fallback to scraping method
            return self.fallback_get_user_tweets(username, max_results)
        
        return tweets_data
    
    def fallback_get_user_tweets(self, username: str, max_results: int = 10) -> List[Dict]:
        """Fallback method using web scraping when API is unavailable"""
        tweets_data = []
        
        try:
            # This is a simplified example - in production, you'd want more robust scraping
            # Note: Twitter's terms of service should be respected
            url = f"https://nitter.net/{username}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract tweets (this is a simplified extraction)
                tweet_elements = soup.find_all('div', class_='tweet-content')[:max_results]
                
                for i, tweet_element in enumerate(tweet_elements):
                    tweet_text = tweet_element.get_text(strip=True)
                    
                    if self.is_financial_tweet(tweet_text):
                        tweet_data = {
                            'tweet_id': f"{username}_{int(time.time())}_{i}",  # Synthetic ID
                            'timestamp': datetime.now(),
                            'influencer': username,
                            'content': tweet_text,
                            'raw_data': json.dumps({'source': 'scraping', 'url': url})
                        }
                        tweets_data.append(tweet_data)
                        
        except Exception as e:
            logger.error(f"Error in fallback scraping for {username}: {e}")
        
        return tweets_data
    
    def monitor_influencers(self) -> List[Dict]:
        """Monitor all configured influencers for new tweets"""
        all_tweets = []
        
        logger.info(f"Starting to monitor {len(self.influencers)} influencers")
        
        # Use ThreadPoolExecutor for concurrent API calls
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_user = {
                executor.submit(self.get_user_tweets, username): username 
                for username in self.influencers
            }
            
            for future in future_to_user:
                username = future_to_user[future]
                try:
                    tweets = future.result(timeout=30)
                    all_tweets.extend(tweets)
                    logger.info(f"Retrieved {len(tweets)} financial tweets from {username}")
                except Exception as e:
                    logger.error(f"Error monitoring {username}: {e}")
        
        return all_tweets
    
    def save_tweets_to_db(self, tweets: List[Dict]) -> List[str]:
        """Save tweets to database and return tweet IDs"""
        saved_tweet_ids = []
        
        for tweet_data in tweets:
            try:
                tweet_id = db.insert_tweet(tweet_data)
                saved_tweet_ids.append(tweet_id)
                logger.debug(f"Saved tweet {tweet_id} to database")
            except Exception as e:
                logger.error(f"Error saving tweet to database: {e}")
        
        logger.info(f"Saved {len(saved_tweet_ids)} tweets to database")
        return saved_tweet_ids
    
    def run_monitoring_cycle(self) -> List[str]:
        """Run a complete monitoring cycle"""
        try:
            # Monitor all influencers
            tweets = self.monitor_influencers()
            
            if tweets:
                # Save to database
                tweet_ids = self.save_tweets_to_db(tweets)
                logger.info(f"Monitoring cycle completed: {len(tweet_ids)} new tweets")
                return tweet_ids
            else:
                logger.info("No new financial tweets found in this cycle")
                return []
                
        except Exception as e:
            logger.error(f"Error in monitoring cycle: {e}")
            return []
    
    async def continuous_monitoring(self, interval_seconds: int = 300):
        """Run continuous monitoring with specified interval"""
        logger.info(f"Starting continuous monitoring with {interval_seconds}s intervals")
        
        while True:
            try:
                tweet_ids = self.run_monitoring_cycle()
                
                if tweet_ids:
                    logger.info(f"Found {len(tweet_ids)} new tweets, triggering analysis pipeline")
                    # Here you would trigger the sentiment analysis pipeline
                    # This will be implemented when we create the orchestrator
                
                await asyncio.sleep(interval_seconds)
                
            except KeyboardInterrupt:
                logger.info("Monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in continuous monitoring: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying

# Create global instance
tweet_listener = TweetListenerAgent()
