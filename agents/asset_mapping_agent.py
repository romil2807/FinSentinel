"""
Asset Mapping Agent for detecting and mapping mentions to ticker symbols
"""
import re
import spacy
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging
from typing import Dict, List, Tuple, Optional
import json

from config import STOCK_TICKERS, CRYPTO_TICKERS
from utils.database import db

logger = logging.getLogger(__name__)

class AssetMappingAgent:
    def __init__(self):
        self.stock_tickers = STOCK_TICKERS
        self.crypto_tickers = CRYPTO_TICKERS
        self.all_tickers = {**self.stock_tickers, **self.crypto_tickers}
        
        # Initialize NLP models
        self.setup_nlp_models()
        
        # Create embeddings for asset names
        self.setup_asset_embeddings()
    
    def setup_nlp_models(self):
        """Initialize NLP models for named entity recognition"""
        try:
            # Load spaCy model for NER
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("SpaCy model loaded successfully")
        except OSError:
            logger.warning("SpaCy model not found, using fallback methods")
            self.nlp = None
        
        try:
            # Load sentence transformer for semantic similarity
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Sentence transformer model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load sentence transformer: {e}")
            self.sentence_model = None
    
    def setup_asset_embeddings(self):
        """Pre-compute embeddings for all known assets"""
        if not self.sentence_model:
            self.asset_embeddings = {}
            return
            
        try:
            asset_names = list(self.all_tickers.keys()) + list(self.all_tickers.values())
            
            # Add company full names for better matching
            company_names = [
                "Tesla", "Apple", "Microsoft", "Nvidia", "Amazon", "Google", "Alphabet",
                "Meta", "Facebook", "Netflix", "PayPal", "Coinbase", "Square", "Shopify",
                "Zoom", "Bitcoin", "Ethereum", "Solana", "Cardano", "Polygon", "Chainlink",
                "Dogecoin", "Shiba Inu", "Avalanche", "Polkadot"
            ]
            
            all_names = asset_names + company_names
            embeddings = self.sentence_model.encode(all_names)
            
            self.asset_embeddings = {
                name: embedding for name, embedding in zip(all_names, embeddings)
            }
            
            logger.info(f"Pre-computed embeddings for {len(self.asset_embeddings)} asset names")
            
        except Exception as e:
            logger.error(f"Error setting up asset embeddings: {e}")
            self.asset_embeddings = {}
    
    def extract_ticker_symbols(self, text: str) -> List[Tuple[str, str, float]]:
        """Extract explicit ticker symbols from text (e.g., $TSLA, AAPL)"""
        results = []
        
        # Pattern for ticker symbols with $ prefix
        dollar_pattern = r'\$([A-Z]{1,5})'
        dollar_matches = re.findall(dollar_pattern, text.upper())
        
        for match in dollar_matches:
            if match in self.stock_tickers.values():
                asset_type = 'stock'
                results.append((match, asset_type, 0.95))
            elif match in self.crypto_tickers.values():
                asset_type = 'crypto'
                results.append((match, asset_type, 0.95))
        
        # Pattern for standalone ticker symbols (3-5 uppercase letters)
        ticker_pattern = r'\b([A-Z]{2,5})\b'
        ticker_matches = re.findall(ticker_pattern, text.upper())
        
        for match in ticker_matches:
            if match in self.stock_tickers.values() and match not in [r[0] for r in results]:
                asset_type = 'stock'
                results.append((match, asset_type, 0.8))
            elif match in self.crypto_tickers.values() and match not in [r[0] for r in results]:
                asset_type = 'crypto'
                results.append((match, asset_type, 0.8))
        
        return results
    
    def extract_company_names_ner(self, text: str) -> List[Tuple[str, str, float]]:
        """Extract company names using Named Entity Recognition"""
        results = []
        
        if not self.nlp:
            return results
        
        try:
            doc = self.nlp(text)
            
            for ent in doc.ents:
                if ent.label_ in ["ORG", "PERSON"]:  # Organizations and persons (influencers)
                    entity_text = ent.text.lower().strip()
                    
                    # Check if entity matches known companies
                    if entity_text in self.stock_tickers:
                        ticker = self.stock_tickers[entity_text]
                        results.append((ticker, 'stock', 0.85))
                    elif entity_text in self.crypto_tickers:
                        ticker = self.crypto_tickers[entity_text]
                        results.append((ticker, 'crypto', 0.85))
        
        except Exception as e:
            logger.error(f"Error in NER extraction: {e}")
        
        return results
    
    def extract_fuzzy_matches(self, text: str, threshold: float = 0.7) -> List[Tuple[str, str, float]]:
        """Extract assets using fuzzy string matching and embeddings"""
        results = []
        
        if not self.sentence_model or not self.asset_embeddings:
            return self.extract_keyword_matches(text)
        
        try:
            text_embedding = self.sentence_model.encode([text])
            
            for asset_name, asset_embedding in self.asset_embeddings.items():
                # Calculate cosine similarity
                similarity = cosine_similarity(text_embedding, [asset_embedding])[0][0]
                
                if similarity > threshold:
                    # Map asset name to ticker
                    asset_lower = asset_name.lower()
                    
                    if asset_lower in self.stock_tickers:
                        ticker = self.stock_tickers[asset_lower]
                        asset_type = 'stock'
                        results.append((ticker, asset_type, similarity))
                    elif asset_lower in self.crypto_tickers:
                        ticker = self.crypto_tickers[asset_lower]
                        asset_type = 'crypto'
                        results.append((ticker, asset_type, similarity))
                    elif asset_name in self.stock_tickers.values():
                        results.append((asset_name, 'stock', similarity))
                    elif asset_name in self.crypto_tickers.values():
                        results.append((asset_name, 'crypto', similarity))
        
        except Exception as e:
            logger.error(f"Error in fuzzy matching: {e}")
            return self.extract_keyword_matches(text)
        
        return results
    
    def extract_keyword_matches(self, text: str) -> List[Tuple[str, str, float]]:
        """Fallback method using simple keyword matching"""
        results = []
        text_lower = text.lower()
        
        # Check for stock mentions
        for keyword, ticker in self.stock_tickers.items():
            if keyword in text_lower:
                # Calculate simple confidence based on keyword length and position
                confidence = min(0.9, 0.5 + (len(keyword) * 0.1))
                results.append((ticker, 'stock', confidence))
        
        # Check for crypto mentions
        for keyword, ticker in self.crypto_tickers.items():
            if keyword in text_lower:
                confidence = min(0.9, 0.5 + (len(keyword) * 0.1))
                results.append((ticker, 'crypto', confidence))
        
        return results
    
    def detect_assets(self, text: str) -> List[Dict]:
        """Main asset detection method combining all approaches"""
        all_detections = []
        
        # Method 1: Extract explicit ticker symbols
        ticker_results = self.extract_ticker_symbols(text)
        for ticker, asset_type, confidence in ticker_results:
            all_detections.append({
                'symbol': ticker,
                'type': asset_type,
                'confidence': confidence,
                'method': 'ticker_extraction'
            })
        
        # Method 2: Named Entity Recognition
        ner_results = self.extract_company_names_ner(text)
        for ticker, asset_type, confidence in ner_results:
            # Avoid duplicates
            if not any(d['symbol'] == ticker for d in all_detections):
                all_detections.append({
                    'symbol': ticker,
                    'type': asset_type,
                    'confidence': confidence,
                    'method': 'ner'
                })
        
        # Method 3: Fuzzy/semantic matching
        fuzzy_results = self.extract_fuzzy_matches(text)
        for ticker, asset_type, confidence in fuzzy_results:
            # Avoid duplicates and low confidence matches
            if (not any(d['symbol'] == ticker for d in all_detections) and 
                confidence > 0.6):
                all_detections.append({
                    'symbol': ticker,
                    'type': asset_type,
                    'confidence': confidence,
                    'method': 'semantic_similarity'
                })
        
        # Sort by confidence and return top matches
        all_detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Return top 3 matches to avoid noise
        return all_detections[:3]
    
    def enhance_asset_context(self, text: str, detected_assets: List[Dict]) -> List[Dict]:
        """Add contextual information to detected assets"""
        enhanced_assets = []
        
        for asset in detected_assets:
            enhanced_asset = asset.copy()
            
            # Analyze context around the asset mention
            symbol_mentions = text.upper().count(asset['symbol'])
            if symbol_mentions > 1:
                enhanced_asset['confidence'] *= 1.2  # Boost confidence for repeated mentions
            
            # Check for price/performance indicators
            price_indicators = ['up', 'down', 'surge', 'drop', 'rally', 'crash', 'moon', 'dump']
            has_price_context = any(indicator in text.lower() for indicator in price_indicators)
            
            if has_price_context:
                enhanced_asset['confidence'] *= 1.1
                enhanced_asset['has_price_context'] = True
            else:
                enhanced_asset['has_price_context'] = False
            
            # Check for action words
            action_words = ['buy', 'sell', 'hold', 'invest', 'trade', 'purchase', 'acquire']
            has_action_context = any(action in text.lower() for action in action_words)
            enhanced_asset['has_action_context'] = has_action_context
            
            enhanced_assets.append(enhanced_asset)
        
        return enhanced_assets
    
    def process_tweet(self, tweet_id: str, tweet_content: str) -> List[Dict]:
        """Process a single tweet for asset mapping"""
        try:
            # Detect assets in the tweet
            detected_assets = self.detect_assets(tweet_content)
            
            if not detected_assets:
                logger.debug(f"No assets detected in tweet {tweet_id}")
                return []
            
            # Enhance with contextual information
            enhanced_assets = self.enhance_asset_context(tweet_content, detected_assets)
            
            # Save to database
            for asset in enhanced_assets:
                if asset['confidence'] > 0.5:  # Only save high-confidence matches
                    db.insert_asset_mapping(tweet_id, asset)
            
            logger.info(f"Mapped {len(enhanced_assets)} assets for tweet {tweet_id}")
            return enhanced_assets
            
        except Exception as e:
            logger.error(f"Error processing asset mapping for tweet {tweet_id}: {e}")
            return []
    
    def batch_process_tweets(self, tweet_data_list: List[Dict]) -> List[Dict]:
        """Process multiple tweets for asset mapping"""
        results = []
        
        for tweet_data in tweet_data_list:
            try:
                assets = self.process_tweet(
                    tweet_data['tweet_id'],
                    tweet_data['content']
                )
                
                if assets:
                    results.append({
                        'tweet_id': tweet_data['tweet_id'],
                        'assets': assets
                    })
                    
            except Exception as e:
                logger.error(f"Error in batch asset mapping for tweet {tweet_data.get('tweet_id', 'unknown')}: {e}")
                continue
        
        logger.info(f"Batch processed asset mapping for {len(results)} tweets")
        return results
    
    def get_asset_statistics(self) -> Dict:
        """Get statistics about detected assets"""
        try:
            # This would query the database for asset detection statistics
            # For now, return basic info about configured assets
            return {
                'total_stock_tickers': len(self.stock_tickers),
                'total_crypto_tickers': len(self.crypto_tickers),
                'total_embeddings': len(self.asset_embeddings),
                'nlp_available': self.nlp is not None,
                'embeddings_available': self.sentence_model is not None
            }
        except Exception as e:
            logger.error(f"Error getting asset statistics: {e}")
            return {}

# Global instance
asset_mapping_agent = AssetMappingAgent()
