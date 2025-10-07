"""
Configuration settings for FinSentinel
"""
import os
from dotenv import load_dotenv

load_dotenv()

# API Keys (set these in your .env file)
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")

# Influencers to monitor
FINANCIAL_INFLUENCERS = [
    "elonmusk",
    "cathiedwood", 
    "chamath",
    "michael_saylor",
    "naval",
    "APompliano",
    "BitcoinMagazine",
    "CoinDesk",
    "business",
    "WSJ",
    "Reuters",
    "BloombergTV"
]

# Asset mappings for ticker detection
STOCK_TICKERS = {
    "tesla": "TSLA",
    "tsla": "TSLA",
    "apple": "AAPL",
    "aapl": "AAPL",
    "microsoft": "MSFT",
    "msft": "MSFT",
    "nvidia": "NVDA",
    "nvda": "NVDA",
    "amazon": "AMZN",
    "amzn": "AMZN",
    "google": "GOOGL",
    "googl": "GOOGL",
    "alphabet": "GOOGL",
    "meta": "META",
    "facebook": "META",
    "netflix": "NFLX",
    "nflx": "NFLX",
    "coinbase": "COIN",
    "coin": "COIN",
    "paypal": "PYPL",
    "pypl": "PYPL",
    "square": "SQ",
    "sq": "SQ",
    "shopify": "SHOP",
    "shop": "SHOP",
    "zoom": "ZM",
    "zm": "ZM",
    "slack": "WORK",
    "work": "WORK"
}

CRYPTO_TICKERS = {
    "bitcoin": "BTC",
    "btc": "BTC",
    "ethereum": "ETH",
    "eth": "ETH",
    "solana": "SOL",
    "sol": "SOL",
    "cardano": "ADA",
    "ada": "ADA",
    "polygon": "MATIC",
    "matic": "MATIC",
    "chainlink": "LINK",
    "link": "LINK",
    "dogecoin": "DOGE",
    "doge": "DOGE",
    "shiba": "SHIB",
    "shib": "SHIB",
    "avalanche": "AVAX",
    "avax": "AVAX",
    "polkadot": "DOT",
    "dot": "DOT"
}

# Sentiment thresholds
SENTIMENT_THRESHOLDS = {
    "very_positive": 0.6,
    "positive": 0.2,
    "neutral": -0.2,
    "negative": -0.6,
    "very_negative": -1.0
}

# Signal generation rules
SIGNAL_RULES = {
    "buy_threshold": 0.4,
    "sell_threshold": -0.4,
    "confidence_multiplier": 1.2  # Multiply signal strength by influencer's track record
}

# Database settings
DATABASE_PATH = "finsentinel.db"

# Dashboard settings
DASHBOARD_REFRESH_INTERVAL = 30  # seconds
MAX_TWEETS_DISPLAY = 100
