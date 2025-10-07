"""
Database utilities for FinSentinel
"""
import sqlite3
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional
import logging
from config import DATABASE_PATH

logger = logging.getLogger(__name__)

class FinSentinelDB:
    def __init__(self, db_path: str = DATABASE_PATH):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Tweets table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS tweets (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        tweet_id TEXT UNIQUE,
                        timestamp DATETIME,
                        influencer TEXT,
                        content TEXT,
                        raw_data TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Sentiment analysis results
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS sentiment_analysis (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        tweet_id TEXT,
                        sentiment_score REAL,
                        sentiment_label TEXT,
                        confidence REAL,
                        model_used TEXT,
                        analyzed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (tweet_id) REFERENCES tweets (tweet_id)
                    )
                """)
                
                # Asset mappings
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS asset_mappings (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        tweet_id TEXT,
                        asset_symbol TEXT,
                        asset_type TEXT,  -- 'stock' or 'crypto'
                        confidence REAL,
                        detection_method TEXT,
                        mapped_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (tweet_id) REFERENCES tweets (tweet_id)
                    )
                """)
                
                # Trading signals
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS trading_signals (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        tweet_id TEXT,
                        asset_symbol TEXT,
                        signal TEXT,  -- 'BUY', 'SELL', 'HOLD'
                        signal_strength REAL,
                        reasoning TEXT,
                        generated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (tweet_id) REFERENCES tweets (tweet_id)
                    )
                """)
                
                # Influencer performance tracking
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS influencer_performance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        influencer TEXT,
                        total_signals INTEGER DEFAULT 0,
                        correct_predictions INTEGER DEFAULT 0,
                        accuracy_rate REAL DEFAULT 0.0,
                        last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.commit()
                logger.info("Database initialized successfully")
                
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    def insert_tweet(self, tweet_data: Dict) -> str:
        """Insert a new tweet into the database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR IGNORE INTO tweets 
                    (tweet_id, timestamp, influencer, content, raw_data)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    tweet_data['tweet_id'],
                    tweet_data['timestamp'],
                    tweet_data['influencer'],
                    tweet_data['content'],
                    tweet_data.get('raw_data', '')
                ))
                conn.commit()
                return tweet_data['tweet_id']
        except Exception as e:
            logger.error(f"Error inserting tweet: {e}")
            raise
    
    def insert_sentiment(self, tweet_id: str, sentiment_data: Dict):
        """Insert sentiment analysis results"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO sentiment_analysis
                    (tweet_id, sentiment_score, sentiment_label, confidence, model_used)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    tweet_id,
                    sentiment_data['score'],
                    sentiment_data['label'],
                    sentiment_data['confidence'],
                    sentiment_data['model']
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error inserting sentiment: {e}")
            raise
    
    def insert_asset_mapping(self, tweet_id: str, asset_data: Dict):
        """Insert asset mapping results"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO asset_mappings
                    (tweet_id, asset_symbol, asset_type, confidence, detection_method)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    tweet_id,
                    asset_data['symbol'],
                    asset_data['type'],
                    asset_data['confidence'],
                    asset_data['method']
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error inserting asset mapping: {e}")
            raise
    
    def insert_trading_signal(self, tweet_id: str, signal_data: Dict):
        """Insert trading signal"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO trading_signals
                    (tweet_id, asset_symbol, signal, signal_strength, reasoning)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    tweet_id,
                    signal_data['asset_symbol'],
                    signal_data['signal'],
                    signal_data['strength'],
                    signal_data['reasoning']
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error inserting trading signal: {e}")
            raise
    
    def get_recent_signals(self, limit: int = 50) -> pd.DataFrame:
        """Get recent trading signals for dashboard"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = """
                    SELECT 
                        t.timestamp,
                        t.influencer,
                        t.content,
                        am.asset_symbol,
                        sa.sentiment_label,
                        sa.sentiment_score,
                        ts.signal,
                        ts.signal_strength,
                        ts.reasoning,
                        ts.generated_at
                    FROM tweets t
                    JOIN sentiment_analysis sa ON t.tweet_id = sa.tweet_id
                    JOIN asset_mappings am ON t.tweet_id = am.tweet_id  
                    JOIN trading_signals ts ON t.tweet_id = ts.tweet_id
                    ORDER BY ts.generated_at DESC
                    LIMIT ?
                """
                return pd.read_sql_query(query, conn, params=(limit,))
        except Exception as e:
            logger.error(f"Error getting recent signals: {e}")
            return pd.DataFrame()
    
    def get_influencer_performance(self) -> pd.DataFrame:
        """Get influencer performance metrics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = """
                    SELECT 
                        influencer,
                        COUNT(*) as total_signals,
                        AVG(CASE WHEN signal_strength > 0.5 THEN 1 ELSE 0 END) as high_confidence_rate
                    FROM tweets t
                    JOIN trading_signals ts ON t.tweet_id = ts.tweet_id
                    GROUP BY influencer
                    ORDER BY total_signals DESC
                """
                return pd.read_sql_query(query, conn)
        except Exception as e:
            logger.error(f"Error getting influencer performance: {e}")
            return pd.DataFrame()
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """Clean up old data to prevent database bloat"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    DELETE FROM tweets 
                    WHERE created_at < datetime('now', '-{} days')
                """.format(days_to_keep))
                conn.commit()
                logger.info(f"Cleaned up data older than {days_to_keep} days")
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")

# Global database instance
db = FinSentinelDB()
