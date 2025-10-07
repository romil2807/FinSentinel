"""
Main orchestrator for FinSentinel - coordinates all agents
"""
import asyncio
import logging
import schedule
import time
from datetime import datetime
from typing import List, Dict
import threading

from agents.tweet_listener import tweet_listener
from agents.sentiment_agent import sentiment_agent
from agents.asset_mapping_agent import asset_mapping_agent
from agents.decision_agent import decision_agent
from utils.database import db
from config import DASHBOARD_REFRESH_INTERVAL

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('finsentinel.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class FinSentinelOrchestrator:
    def __init__(self):
        self.is_running = False
        self.last_processing_time = None
        self.stats = {
            'tweets_processed': 0,
            'signals_generated': 0,
            'errors': 0,
            'last_run': None
        }
    
    def process_pipeline(self, tweet_ids: List[str]) -> Dict:
        """Process a batch of tweets through the complete pipeline"""
        pipeline_stats = {
            'tweets_processed': 0,
            'sentiment_analyses': 0,
            'asset_mappings': 0,
            'signals_generated': 0,
            'errors': 0
        }
        
        try:
            logger.info(f"Starting pipeline processing for {len(tweet_ids)} tweets")
            
            # Get tweet data from database
            tweet_data_list = []
            for tweet_id in tweet_ids:
                try:
                    with sqlite3.connect(db.db_path) as conn:
                        cursor = conn.cursor()
                        cursor.execute("""
                            SELECT tweet_id, timestamp, influencer, content, raw_data
                            FROM tweets WHERE tweet_id = ?
                        """, (tweet_id,))
                        
                        row = cursor.fetchone()
                        if row:
                            tweet_data_list.append({
                                'tweet_id': row[0],
                                'timestamp': row[1],
                                'influencer': row[2],
                                'content': row[3],
                                'raw_data': row[4]
                            })
                            pipeline_stats['tweets_processed'] += 1
                            
                except Exception as e:
                    logger.error(f"Error fetching tweet {tweet_id}: {e}")
                    pipeline_stats['errors'] += 1
            
            if not tweet_data_list:
                logger.warning("No valid tweet data found for processing")
                return pipeline_stats
            
            # Step 1: Sentiment Analysis
            logger.info("Step 1: Running sentiment analysis...")
            sentiment_results = sentiment_agent.batch_process_tweets(tweet_data_list)
            pipeline_stats['sentiment_analyses'] = len(sentiment_results)
            
            # Step 2: Asset Mapping
            logger.info("Step 2: Running asset mapping...")
            asset_results = asset_mapping_agent.batch_process_tweets(tweet_data_list)
            pipeline_stats['asset_mappings'] = len(asset_results)
            
            # Step 3: Decision Generation (only for tweets with both sentiment and assets)
            logger.info("Step 3: Generating trading signals...")
            
            # Find tweets that have both sentiment and asset data
            processed_tweet_ids = []
            for tweet_data in tweet_data_list:
                tweet_id = tweet_data['tweet_id']
                
                has_sentiment = any(r['tweet_id'] == tweet_id for r in sentiment_results)
                has_assets = any(r['tweet_id'] == tweet_id for r in asset_results)
                
                if has_sentiment and has_assets:
                    processed_tweet_ids.append(tweet_id)
            
            if processed_tweet_ids:
                signal_results = decision_agent.batch_process_signals(processed_tweet_ids)
                pipeline_stats['signals_generated'] = len(signal_results)
            
            # Update global stats
            self.stats['tweets_processed'] += pipeline_stats['tweets_processed']
            self.stats['signals_generated'] += pipeline_stats['signals_generated']
            self.stats['errors'] += pipeline_stats['errors']
            self.stats['last_run'] = datetime.now()
            
            logger.info(f"Pipeline completed: {pipeline_stats}")
            return pipeline_stats
            
        except Exception as e:
            logger.error(f"Error in pipeline processing: {e}")
            pipeline_stats['errors'] += 1
            return pipeline_stats
    
    def run_monitoring_cycle(self):
        """Run a complete monitoring and processing cycle"""
        try:
            logger.info("=== Starting FinSentinel monitoring cycle ===")
            
            # Step 1: Monitor for new tweets
            new_tweet_ids = tweet_listener.run_monitoring_cycle()
            
            if new_tweet_ids:
                logger.info(f"Found {len(new_tweet_ids)} new tweets")
                
                # Step 2: Process through pipeline
                pipeline_stats = self.process_pipeline(new_tweet_ids)
                
                logger.info(f"Cycle completed successfully: {pipeline_stats}")
            else:
                logger.info("No new tweets found in this cycle")
            
            self.last_processing_time = datetime.now()
            
        except Exception as e:
            logger.error(f"Error in monitoring cycle: {e}")
            self.stats['errors'] += 1
    
    def setup_scheduler(self):
        """Setup scheduled tasks"""
        # Main monitoring cycle every 5 minutes
        schedule.every(5).minutes.do(self.run_monitoring_cycle)
        
        # Database cleanup daily at 2 AM
        schedule.every().day.at("02:00").do(self.cleanup_database)
        
        # Generate summary report every hour
        schedule.every().hour.do(self.generate_hourly_report)
        
        logger.info("Scheduler setup completed")
    
    def cleanup_database(self):
        """Clean up old database records"""
        try:
            logger.info("Running database cleanup...")
            db.cleanup_old_data(days_to_keep=30)
            logger.info("Database cleanup completed")
        except Exception as e:
            logger.error(f"Error in database cleanup: {e}")
    
    def generate_hourly_report(self):
        """Generate and log hourly statistics"""
        try:
            # Get recent signals summary
            signal_summary = decision_agent.get_signal_summary(hours=1)
            
            # Get asset mapping statistics
            asset_stats = asset_mapping_agent.get_asset_statistics()
            
            logger.info(f"Hourly Report - Signals: {signal_summary}, Assets: {asset_stats}")
            
        except Exception as e:
            logger.error(f"Error generating hourly report: {e}")
    
    def run_scheduler(self):
        """Run the scheduler in a separate thread"""
        while self.is_running:
            try:
                schedule.run_pending()
                time.sleep(1)
            except Exception as e:
                logger.error(f"Error in scheduler: {e}")
                time.sleep(60)  # Wait 1 minute before retrying
    
    async def run_continuous(self):
        """Run FinSentinel continuously"""
        logger.info("🚀 Starting FinSentinel continuous monitoring...")
        
        self.is_running = True
        
        # Setup and start scheduler in background thread
        self.setup_scheduler()
        scheduler_thread = threading.Thread(target=self.run_scheduler, daemon=True)
        scheduler_thread.start()
        
        # Run initial cycle
        self.run_monitoring_cycle()
        
        try:
            while self.is_running:
                # Main loop just keeps the process alive
                # Actual work is done by the scheduler
                await asyncio.sleep(60)
                
                # Periodic health check
                if self.last_processing_time:
                    time_since_last = (datetime.now() - self.last_processing_time).total_seconds()
                    if time_since_last > 900:  # 15 minutes
                        logger.warning(f"No processing in {time_since_last:.0f} seconds")
                
        except KeyboardInterrupt:
            logger.info("Stopping FinSentinel...")
            self.is_running = False
        except Exception as e:
            logger.error(f"Unexpected error in continuous run: {e}")
            self.is_running = False
    
    def run_single_cycle(self):
        """Run a single monitoring cycle (useful for testing)"""
        logger.info("Running single FinSentinel cycle...")
        self.run_monitoring_cycle()
        return self.stats
    
    def get_system_status(self) -> Dict:
        """Get current system status"""
        return {
            'is_running': self.is_running,
            'last_processing_time': self.last_processing_time,
            'stats': self.stats,
            'components': {
                'tweet_listener': tweet_listener is not None,
                'sentiment_agent': sentiment_agent is not None,
                'asset_mapping_agent': asset_mapping_agent is not None,
                'decision_agent': decision_agent is not None,
                'database': db is not None
            }
        }

# Global orchestrator instance
orchestrator = FinSentinelOrchestrator()

if __name__ == "__main__":
    import sys
    import sqlite3
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "single":
            # Run single cycle for testing
            stats = orchestrator.run_single_cycle()
            print(f"Cycle completed: {stats}")
        elif sys.argv[1] == "status":
            # Show system status
            status = orchestrator.get_system_status()
            print(f"System status: {status}")
    else:
        # Run continuously
        asyncio.run(orchestrator.run_continuous())
