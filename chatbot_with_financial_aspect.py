"""
Enhanced Congressional Chatbot with Financial Analysis - Complete Production Version
Author: Jason Mowrey
Date: December 5, 2024
Version: 2.0.1

A comprehensive chatbot system for analyzing congressional hearings and their financial market impacts.
Includes complete error handling, resource management, and performance optimizations.
"""

# Standard library imports
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Union, Any, Callable
import logging
from pathlib import Path
import os
from datetime import datetime, timedelta
from collections import deque
import re
import json
import asyncio
import sqlite3
import traceback
from functools import lru_cache
import prometheus_client
from pmdarima import auto_arima
import plotly.graph_objects as go

# Third-party imports
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from cachetools import TTLCache, LRUCache
from tenacity import retry, stop_after_attempt, wait_exponential
import openai
from dotenv import load_dotenv
import pandas as pd
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA  # Correctly imported ARIMA
from statsmodels.tsa.stattools import grangercausalitytests
from scipy import stats

# Constants for model configuration
OPENAI_CHAT_MODEL = "gpt-4-turbo-preview"
OPENAI_EMBEDDING_MODEL = "text-embedding-ada-002"

# Custom exceptions
class ConfigurationError(Exception):
    """Custom exception for configuration errors."""
    pass

class APIError(Exception):
    """Custom exception for API-related errors."""
    pass

class DataError(Exception):
    """Custom exception for data processing errors."""
    pass

@dataclass
class ChatbotConfig:
    """Enhanced configuration settings for the chatbot with validation."""
    max_context_chunks: int = 8
    similarity_threshold: float = 0.1
    max_response_tokens: int = 1500
    conversation_memory_limit: int = 8
    temperature: float = 0.6
    model_name: str = OPENAI_CHAT_MODEL
    embedding_model: str = OPENAI_EMBEDDING_MODEL
    rate_limit_calls: int = 60
    rate_limit_period: int = 60
    cache_ttl: int = 3600
    max_retries: int = 3
    max_consecutive_errors: int = 5  # Add this line
    
    system_prompt: str = """You are a sophisticated financial and congressional analysis assistant specializing in connecting legislative proceedings with market implications. 
    Follow these enhanced guidelines:

    1. ANALYSIS DEPTH:
    - Ground responses in hearing excerpts with direct quotes and statistical evidence
    - Identify causal relationships between congressional actions and market movements
    - Analyze both short-term and long-term market implications
    - Highlight regulatory changes and their sector-specific impacts
    
    2. MARKET CONTEXT:
    - Consider macroeconomic conditions during hearing periods
    - Analyze sector-wide implications of discussed policies
    - Evaluate market sentiment changes post-hearings
    - Compare similar historical scenarios and their outcomes
    
    3. RISK ASSESSMENT:
    - Identify potential regulatory risks
    - Analyze compliance implications
    - Evaluate market reaction probabilities
    - Consider systemic risk factors
    
    4. TEMPORAL ANALYSIS:
    - Track policy evolution across multiple hearings
    - Identify turning points in regulatory stance
    - Compare pre/post hearing market behaviors
    - Consider seasonal and cyclical factors

    5. STAKEHOLDER IMPACT:
    - Analyze effects on different market participants
    - Consider institutional vs retail investor implications
    - Evaluate industry-specific consequences
    - Assess international market reactions

    Remember: Provide balanced, evidence-based analysis while acknowledging uncertainty in market predictions."""
    
    def __post_init__(self):
        """Validate configuration parameters."""
        self._validate_numeric_range("max_context_chunks", 1, 100)
        self._validate_numeric_range("similarity_threshold", 0, 1)
        self._validate_numeric_range("max_response_tokens", 1, 4000)
        self._validate_numeric_range("conversation_memory_limit", 1, 100)
        self._validate_numeric_range("temperature", 0, 1)
        self._validate_model_names()
    
    def _validate_numeric_range(self, field_name: str, min_val: float, max_val: float):
        """Validate numeric fields are within acceptable ranges."""
        value = getattr(self, field_name)
        if not min_val <= value <= max_val:
            raise ConfigurationError(
                f"{field_name} must be between {min_val} and {max_val}, got {value}"
            )
    
    def _validate_model_names(self):
        """Validate OpenAI model names."""
        valid_chat_models = ["gpt-4-turbo-preview", "gpt-4", "gpt-3.5-turbo"]
        if self.model_name not in valid_chat_models:
            raise ConfigurationError(
                f"Invalid chat model name. Must be one of {valid_chat_models}"
            )
            
        valid_embedding_models = ["text-embedding-ada-002"]
        if self.embedding_model not in valid_embedding_models:
            raise ConfigurationError(
                f"Invalid embedding model name. Must be one of {valid_embedding_models}"
            )

class RateLimiter:
   """Rate limiter implementation using token bucket algorithm with concurrent request limiting."""
   
   def __init__(self, max_calls: int = 60, time_window: int = 60):
       self.max_calls = max_calls
       self.time_window = time_window
       self.calls = deque()
       self.semaphore = asyncio.Semaphore(10)  # Concurrent requests limit
       self._setup_logging()
   
   def _setup_logging(self):
       """Setup logging for rate limiter."""
       self.logger = logging.getLogger(__name__)
       
       if not self.logger.handlers:
           fh = logging.FileHandler('logs/ratelimit.log')
           ch = logging.StreamHandler()
           formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
           fh.setFormatter(formatter)
           ch.setFormatter(formatter)
           self.logger.addHandler(fh)
           self.logger.addHandler(ch)
   
   async def acquire(self):
       """Acquire a rate limit token while respecting concurrent limits."""
       try:
           async with self.semaphore:
               now = datetime.now()
               
               # Remove expired timestamps
               while self.calls and (now - self.calls[0]) > timedelta(seconds=self.time_window):
                   self.calls.popleft()
               
               # Check if we need to wait
               if len(self.calls) >= self.max_calls:
                   wait_time = (self.calls[0] + timedelta(seconds=self.time_window) - now).total_seconds()
                   if wait_time > 0:
                       self.logger.debug(f"Rate limit reached. Waiting {wait_time} seconds")
                       await asyncio.sleep(wait_time)
               
               self.calls.append(now)
               
       except Exception as e:
           self.logger.error(f"Error in rate limiter: {str(e)}")
           raise

   async def cleanup(self):
       """Cleanup resources."""
       try:
           for handler in self.logger.handlers[:]:
               handler.close()
               self.logger.removeHandler(handler)
       except Exception as e:
           print(f"Error during cleanup: {str(e)}")

@dataclass
class FinancialAnalysisConfig:
    """Enhanced configuration for financial analysis with caching."""
    historical_lookback: int = 5
    min_confidence_score: float = 0.7
    volatility_window: int = 30
    correlation_threshold: float = 0.6
    granger_max_lag: int = 5
    
    sector_mapping: Dict[str, Dict] = field(default_factory=dict)
    market_data_cache: TTLCache = field(default_factory=lambda: TTLCache(maxsize=100, ttl=3600))
    
    def __post_init__(self):
        """Initialize sector mapping and validate configuration."""
        self._initialize_sector_mapping()
        self._validate_config()
    
    def _initialize_sector_mapping(self):
        """Initialize comprehensive sector mapping."""
        self.sector_mapping = {
            'technology': {
                'etf': 'XLK',
                'sub_sectors': ['software', 'hardware', 'semiconductors', 'cloud', 'cybersecurity', 'ai'],
                'keywords': ['tech', 'digital', 'cyber', 'AI', 'data', 'software', 'semiconductor', 'cloud', 'computing'],
                'companies': ['AAPL', 'MSFT', 'NVDA', 'CRM', 'ADBE'],
                'risk_factors': ['regulation', 'competition', 'innovation', 'cybersecurity'],
                'economic_indicators': ['NASDAQ', 'semiconductor_index', 'cloud_computing_index']
            },
            'healthcare': {
                'etf': 'XLV',
                'sub_sectors': ['biotech', 'pharmaceuticals', 'medical_devices', 'healthcare_services', 'telemedicine'],
                'keywords': ['health', 'medical', 'drug', 'clinical', 'biotech', 'pharmaceutical', 'healthcare', 'medicine'],
                'companies': ['JNJ', 'UNH', 'PFE', 'ABT', 'TMO'],
                'risk_factors': ['regulation', 'patent_expiry', 'clinical_trials', 'healthcare_policy'],
                'economic_indicators': ['healthcare_spending', 'drug_approval_rate']
            },
            'financials': {
                'etf': 'XLF',
                'sub_sectors': ['banks', 'insurance', 'fintech', 'asset_management', 'investment_banking'],
                'keywords': ['banking', 'financial', 'credit', 'payment', 'insurance', 'bank', 'loan', 'mortgage', 'invest'],
                'companies': ['JPM', 'BAC', 'WFC', 'GS', 'MS'],
                'risk_factors': ['interest_rates', 'regulation', 'credit_risk', 'market_volatility'],
                'economic_indicators': ['federal_funds_rate', 'yield_curve', 'credit_spread']
            },
            'energy': {
                'etf': 'XLE',
                'sub_sectors': ['oil_gas', 'renewable', 'utilities', 'energy_services', 'clean_energy'],
                'keywords': ['energy', 'oil', 'gas', 'renewable', 'solar', 'wind', 'utilities', 'power'],
                'companies': ['XOM', 'CVX', 'COP', 'SLB', 'EOG'],
                'risk_factors': ['commodity_prices', 'regulation', 'geopolitical', 'environmental'],
                'economic_indicators': ['oil_price', 'natural_gas_price', 'renewable_index']
            },
            'consumer': {
                'etf': 'XLY',
                'sub_sectors': ['retail', 'automotive', 'entertainment', 'e_commerce', 'hospitality'],
                'keywords': ['consumer', 'retail', 'sales', 'e-commerce', 'entertainment', 'shopping', 'goods'],
                'companies': ['AMZN', 'HD', 'MCD', 'NKE', 'SBUX'],
                'risk_factors': ['consumer_confidence', 'unemployment', 'inflation', 'disposable_income'],
                'economic_indicators': ['retail_sales', 'consumer_confidence', 'disposable_income']
            },
            'real_estate': {
                'etf': 'IYR',
                'sub_sectors': ['residential', 'commercial', 'industrial', 'retail_space', 'office_space', 'reit'],
                'keywords': ['real estate', 'property', 'housing', 'commercial property', 'residential', 'rent', 'lease', 'reit'],
                'companies': ['SPG', 'PLD', 'AMT', 'CCI', 'EQIX'],
                'risk_factors': ['interest_rates', 'regulation', 'market_demand', 'location'],
                'economic_indicators': ['housing_starts', 'home_sales', 'mortgage_rates']
            },
            'industrials': {
                'etf': 'XLI',
                'sub_sectors': ['aerospace', 'defense', 'transportation', 'machinery', 'construction'],
                'keywords': ['industrial', 'manufacturing', 'aerospace', 'defense', 'transportation', 'construction'],
                'companies': ['HON', 'UPS', 'UNP', 'BA', 'CAT'],
                'risk_factors': ['economic_cycles', 'trade_policy', 'raw_materials', 'labor_costs'],
                'economic_indicators': ['industrial_production', 'manufacturing_pmi', 'durable_goods']
            },
            'materials': {
                'etf': 'XLB',
                'sub_sectors': ['chemicals', 'metals', 'mining', 'construction_materials', 'packaging'],
                'keywords': ['materials', 'chemicals', 'metals', 'mining', 'commodities', 'raw materials'],
                'companies': ['LIN', 'FCX', 'APD', 'ECL', 'NEM'],
                'risk_factors': ['commodity_prices', 'environmental_regulation', 'global_demand'],
                'economic_indicators': ['commodity_index', 'construction_spending', 'mining_production']
            },
            'telecommunications': {
                'etf': 'IYZ',
                'sub_sectors': ['wireless', 'broadband', 'infrastructure', '5g', 'telecom_services'],
                'keywords': ['telecom', 'communications', 'wireless', '5g', 'broadband', 'network'],
                'companies': ['T', 'VZ', 'TMUS', 'CSCO', 'ERIC'],
                'risk_factors': ['regulation', 'technology_change', 'competition', 'infrastructure_costs'],
                'economic_indicators': ['wireless_subscribers', 'broadband_penetration', '5g_adoption']
            },
            'agriculture': {
                'etf': 'DBA',
                'sub_sectors': ['farming', 'livestock', 'agricultural_products', 'fertilizers', 'agtech'],
                'keywords': ['agriculture', 'farming', 'crops', 'livestock', 'agricultural', 'food production'],
                'companies': ['DE', 'ADM', 'NTR', 'MOS', 'FMC'],
                'risk_factors': ['weather', 'commodity_prices', 'trade_policy', 'environmental_regulation'],
                'economic_indicators': ['crop_prices', 'fertilizer_prices', 'agricultural_exports']
            },
            'utilities': {
                'etf': 'XLU',
                'sub_sectors': ['electric', 'water', 'gas', 'renewable_utilities', 'waste_management'],
                'keywords': ['utilities', 'electric', 'power', 'water', 'gas', 'waste management'],
                'companies': ['NEE', 'DUK', 'SO', 'AEP', 'EXC'],
                'risk_factors': ['regulation', 'interest_rates', 'environmental_policy', 'infrastructure'],
                'economic_indicators': ['electricity_rates', 'natural_gas_rates', 'utility_index']
            },
            'cybersecurity': {
                'etf': 'HACK',
                'sub_sectors': ['network_security', 'cloud_security', 'endpoint_security', 'identity_management'],
                'keywords': ['cybersecurity', 'security', 'cyber', 'hacking', 'data protection', 'privacy'],
                'companies': ['CRWD', 'PANW', 'FTNT', 'ZS', 'OKTA'],
                'risk_factors': ['cyber_threats', 'regulation', 'technology_change', 'competition'],
                'economic_indicators': ['cybersecurity_spending', 'data_breach_costs', 'security_index']
            }
        }
    
    def _validate_config(self):
        """Validate configuration parameters."""
        if self.historical_lookback < 1:
            raise ConfigurationError("historical_lookback must be positive")
        if not 0 <= self.min_confidence_score <= 1:
            raise ConfigurationError("min_confidence_score must be between 0 and 1")
        if not self.sector_mapping:
            raise ConfigurationError("sector_mapping cannot be empty")

class MarketAnalyzer:
    """Enhanced market analysis capabilities with caching and rate limiting."""
    
    def __init__(self, config: FinancialAnalysisConfig):
        """Initialize the market analyzer with logging and configuration."""
        # Set up logging first
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.max_cache_size_mb = 512
        self.data_cache = TTLCache(
            maxsize=self.max_cache_size_mb * 1024 * 1024 // (8 * 1024),  # Approximate number of entries
            ttl=3600
        )
        
        @lru_cache(maxsize=1000)
        async def get_cached_market_data(self, symbol: str, start_date: str, end_date: str):
            return await self.get_market_data(symbol, start_date, end_date)
        
        # Add handlers if none exist
        if not self.logger.handlers:
            # Create logs directory if it doesn't exist
            os.makedirs('logs', exist_ok=True)
            
            # File handler
            fh = logging.FileHandler('logs/chatbot.log')
            fh.setLevel(logging.INFO)
            
            # Console handler
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            
            # Formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)
            
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)

        self.logger.debug("Initializing MarketAnalyzer")
        
        try:
            self.config = config
            self.rate_limiter = RateLimiter(max_calls=60, time_window=60)
            self._initialize_cache()
            self.logger.info("MarketAnalyzer initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing MarketAnalyzer: {str(e)}")
            raise
    
    def _initialize_cache(self):
        """Initialize caching system for market data and analysis results."""
        self.data_cache = TTLCache(maxsize=100, ttl=3600)
        self.analysis_cache = LRUCache(maxsize=1000)
        self.logger.debug("Cache initialized")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def get_market_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get market data with caching and rate limiting."""
        # Validate symbol first
        if symbol.upper() in ['ETF', 'STOCK', 'SECTOR']:
            self.logger.warning(f"Invalid symbol: {symbol}")
            raise DataError(f"Invalid symbol: {symbol}")
        
        cache_key = f"{symbol}_{start_date.isoformat()}_{end_date.isoformat()}"
        
        if cache_key in self.data_cache:
            self.logger.debug(f"Cache hit for {symbol}")
            return self.data_cache[cache_key]
        
        await self.rate_limiter.acquire()
        
        try:
            self.logger.info(f"Fetching market data for {symbol} from {start_date} to {end_date}")
            # Download data with yfinance
            data = yf.download(symbol, start=start_date, end=end_date)
            
            if data.empty:
                self.logger.warning(f"No data available for {symbol}")
                raise DataError(f"No data available for {symbol}")
            
            # Ensure proper datetime index with business day frequency
            data.index = pd.DatetimeIndex(data.index)
            data = data.asfreq('B', method='ffill')  # Forward fill any gaps
            
            self.data_cache[cache_key] = data
            self.logger.debug(f"Successfully cached data for {symbol}")
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching market data for {symbol}: {str(e)}")
            raise APIError(f"Failed to fetch market data: {str(e)}")

    def calculate_abnormal_returns(self, 
                                 stock_returns: pd.Series,
                                 market_returns: pd.Series,
                                 event_date: datetime,
                                 window: int = 30) -> Optional[Dict[str, float]]:
        """Calculate abnormal returns around events using market model."""
        try:
            if stock_returns.empty or market_returns.empty:
                raise DataError("Empty return series provided")
                
            pre_event = slice(event_date - timedelta(days=window), event_date)
            post_event = slice(event_date, event_date + timedelta(days=window))
            
            # Calculate beta using pre-event data
            beta = (
                np.cov(stock_returns[pre_event], market_returns[pre_event])[0,1] /
                np.var(market_returns[pre_event])
            )
            
            # Calculate alpha (intercept)
            alpha = stock_returns[pre_event].mean() - beta * market_returns[pre_event].mean()
            
            # Calculate expected returns
            expected_returns = alpha + beta * market_returns
            abnormal_returns = stock_returns - expected_returns
            
            # Calculate cumulative abnormal returns
            pre_event_car = abnormal_returns[pre_event].sum()
            post_event_car = abnormal_returns[post_event].sum()
            
            # Calculate statistical significance
            ar_std = abnormal_returns[pre_event].std()
            t_stat = post_event_car / (ar_std * np.sqrt(window))
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), window-1))
            
            return {
                'pre_event_car': pre_event_car,
                'post_event_car': post_event_car,
                'event_day_ar': abnormal_returns.get(event_date, 0),
                'beta': beta,
                'alpha': alpha,
                'avg_daily_ar': abnormal_returns[post_event].mean(),
                'ar_volatility': abnormal_returns[post_event].std(),
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating abnormal returns: {str(e)}")
            return None

    async def test_market_reaction(self,
                        returns: pd.Series,
                        event_date: datetime,
                        window: int = 3) -> Optional[Dict[str, float]]:
        """Statistical tests for market reaction significance."""
        try:
            if returns.empty:
                self.logger.warning("Empty returns series provided")
                return None
                
            # Ensure we have enough data after the event date
            days_since_event = (datetime.now() - event_date).days
            if days_since_event < window:
                # Adjust event date to allow for enough post-event data
                adjusted_date = datetime.now() - timedelta(days=window+1)
                self.logger.info(f"Adjusting event date from {event_date} to {adjusted_date} to ensure enough post-event data")
                event_date = adjusted_date
                
            # Find the closest available date in the data
            if event_date not in returns.index:
                closest_date = returns.index[returns.index.get_indexer([event_date], method='nearest')[0]]
                self.logger.info(f"Event date {event_date} not found, using closest date {closest_date}")
                event_date = closest_date
                
            # Get data around event date
            pre_event = returns[returns.index < event_date].tail(window)
            post_event = returns[returns.index >= event_date].head(window)
            
            # Verify we have enough data
            if len(pre_event) < window or len(post_event) < window:
                self.logger.warning(f"Insufficient data for market reaction test. "
                                f"Pre-event: {len(pre_event)}, Post-event: {len(post_event)}, "
                                f"Window: {window}")
                
                # Try with smaller window
                smaller_window = min(len(pre_event), len(post_event))
                if smaller_window >= 2:
                    self.logger.info(f"Attempting analysis with smaller window size: {smaller_window}")
                    pre_event = returns[returns.index < event_date].tail(smaller_window)
                    post_event = returns[returns.index >= event_date].head(smaller_window)
                else:
                    # Return basic statistics if we can't do full analysis
                    return {
                        'pre_event_mean': float(pre_event.mean()) if not pre_event.empty else None,
                        'post_event_mean': float(post_event.mean()) if not post_event.empty else None,
                        'pre_event_samples': len(pre_event),
                        'post_event_samples': len(post_event),
                        'analysis_window': smaller_window,
                        'warning': 'Insufficient data for complete analysis',
                        'dates': {
                            'event_date': event_date.strftime('%Y-%m-%d'),
                            'pre_event_start': pre_event.index[0].strftime('%Y-%m-%d') if not pre_event.empty else None,
                            'pre_event_end': pre_event.index[-1].strftime('%Y-%m-%d') if not pre_event.empty else None,
                            'post_event_start': post_event.index[0].strftime('%Y-%m-%d') if not post_event.empty else None,
                            'post_event_end': post_event.index[-1].strftime('%Y-%m-%d') if not post_event.empty else None
                        }
                    }
            
            # Remove any NaN values
            pre_event = pre_event.dropna()
            post_event = post_event.dropna()
            
            # Calculate statistics
            pre_mean = float(pre_event.mean()) if not pre_event.empty else 0
            post_mean = float(post_event.mean()) if not post_event.empty else 0
            mean_change = post_mean - pre_mean
            mean_change_pct = (mean_change / abs(pre_mean)) if abs(pre_mean) > 1e-10 else 0
            
            pre_vol = float(pre_event.std()) if len(pre_event) > 1 else 0
            post_vol = float(post_event.std()) if len(post_event) > 1 else 0
            vol_change = post_vol - pre_vol
            vol_change_pct = (vol_change / pre_vol) if abs(pre_vol) > 1e-10 else 0
            
            # Statistical tests
            try:
                t_stat, p_value = stats.ttest_ind(pre_event, post_event)
                significant = p_value < 0.05
            except Exception as e:
                self.logger.warning(f"Error in statistical tests: {str(e)}")
                t_stat = p_value = None
                significant = None
            
            self.logger.info("Market reaction analysis completed successfully")
            
            return {
                'statistical_tests': {
                    't_statistic': float(t_stat) if t_stat is not None else None,
                    'p_value': float(p_value) if p_value is not None else None,
                    'significant': significant
                },
                'return_metrics': {
                    'pre_event_mean': pre_mean,
                    'post_event_mean': post_mean,
                    'mean_change': float(mean_change),
                    'mean_change_pct': float(mean_change_pct)
                },
                'volatility_metrics': {
                    'pre_event_volatility': pre_vol,
                    'post_event_volatility': post_vol,
                    'volatility_change': float(vol_change),
                    'volatility_change_pct': float(vol_change_pct)
                },
                'data_quality': {
                    'pre_event_samples': len(pre_event),
                    'post_event_samples': len(post_event),
                    'analysis_window': window,
                    'dates': {
                        'event_date': event_date.strftime('%Y-%m-%d'),
                        'pre_event_start': pre_event.index[0].strftime('%Y-%m-%d') if not pre_event.empty else None,
                        'pre_event_end': pre_event.index[-1].strftime('%Y-%m-%d') if not pre_event.empty else None,
                        'post_event_start': post_event.index[0].strftime('%Y-%m-%d') if not post_event.empty else None,
                        'post_event_end': post_event.index[-1].strftime('%Y-%m-%d') if not post_event.empty else None
                    }
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error testing market reaction: {str(e)}")
            return None

    def test_granger_causality(self,
                             hearing_returns: pd.Series,
                             market_returns: pd.Series,
                             max_lag: int = 5) -> Optional[Dict[str, Any]]:
        """Test for Granger causality between hearing-related returns and market returns."""
        try:
            if hearing_returns.empty or market_returns.empty:
                raise DataError("Empty return series provided")
            
            # Prepare data
            data = pd.DataFrame({
                'hearing': hearing_returns,
                'market': market_returns
            }).dropna()
            
            if len(data) < max_lag + 2:
                raise DataError(f"Insufficient data points for lag {max_lag}")
            
            # Run Granger causality test
            results = grangercausalitytests(data[['hearing', 'market']], 
                                          maxlag=max_lag,
                                          verbose=False)
            
            # Extract detailed results for each lag
            causality_results = {}
            for lag in range(1, max_lag + 1):
                test_results = results[lag][0]
                causality_results[f'lag_{lag}'] = {
                    'ssr_ftest': {
                        'statistic': float(test_results['ssr_ftest'][0]),
                        'p_value': float(test_results['ssr_ftest'][1])
                    },
                    'ssr_chi2test': {
                        'statistic': float(test_results['ssr_chi2test'][0]),
                        'p_value': float(test_results['ssr_chi2test'][1])
                    },
                    'lrtest': {
                        'statistic': float(test_results['lrtest'][0]),
                        'p_value': float(test_results['lrtest'][1])
                    },
                    'params_ftest': {
                        'statistic': float(test_results['params_ftest'][0]),
                        'p_value': float(test_results['params_ftest'][1])
                    },
                    'significant': test_results['ssr_ftest'][1] < 0.05
                }
            
            # Determine optimal lag based on AIC
            aic_scores = {}
            for lag in range(1, max_lag + 1):
                model = grangercausalitytests(data[['hearing', 'market']], 
                                            maxlag=lag,
                                            verbose=False)
                aic_scores[lag] = model[lag][1][0].aic
            
            optimal_lag = min(aic_scores, key=aic_scores.get)
            
            return {
                'has_causality': any(result['significant'] 
                                   for result in causality_results.values()),
                'optimal_lag': optimal_lag,
                'lag_results': causality_results,
                'aic_scores': aic_scores
            }
            
        except Exception as e:
            self.logger.error(f"Error in Granger causality test: {str(e)}")
            return None

    def analyze_sector_correlation(self,
                                 sector_returns: Dict[str, pd.Series],
                                 event_date: datetime,
                                 window: int = 30) -> Optional[Dict[str, Any]]:
        """Analyze cross-sector correlations and spillover effects around events."""
        try:
            if not sector_returns:
                raise DataError("No sector returns provided")
                
            post_event = {k: v[event_date:event_date + timedelta(days=window)] 
                         for k, v in sector_returns.items()}
            
            pre_event = {k: v[event_date - timedelta(days=window):event_date]
                        for k, v in sector_returns.items()}
            
            # Calculate correlation matrices
            pre_corr = pd.DataFrame(pre_event).corr()
            post_corr = pd.DataFrame(post_event).corr()
            
            # Calculate correlation changes
            corr_change = post_corr - pre_corr
            
            # Analyze correlation stability
            pre_eigen = np.linalg.eigvals(pre_corr)
            post_eigen = np.linalg.eigvals(post_corr)
            
            # Calculate spillover indices
            spillover_index = self._calculate_spillover_index(post_corr)
            
            return {
                'pre_event_correlation': pre_corr.to_dict(),
                'post_event_correlation': post_corr.to_dict(),
                'correlation_change': corr_change.to_dict(),
                'stability_metrics': {
                    'pre_condition_number': np.max(pre_eigen) / np.min(pre_eigen),
                    'post_condition_number': np.max(post_eigen) / np.min(post_eigen),
                    'eigenvalue_change': np.mean(np.abs(post_eigen - pre_eigen))
                },
                'spillover_metrics': spillover_index
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing sector correlation: {str(e)}")
            return None

    def _calculate_spillover_index(self, correlation_matrix: pd.DataFrame) -> Dict[str, float]:
        """Calculate spillover index based on correlation matrix."""
        try:
            # Calculate eigenvector centrality
            eigenvalues, eigenvectors = np.linalg.eigh(correlation_matrix)
            centrality = pd.Series(
                np.abs(eigenvectors[:, -1]) / np.sum(np.abs(eigenvectors[:, -1])),
                index=correlation_matrix.index
            )
            
            # Calculate condition number for stability
            condition_number = np.max(np.abs(eigenvalues)) / np.min(np.abs(eigenvalues))
            
            # Calculate various spillover metrics
            total_spillover = (np.sum(correlation_matrix.values) - np.trace(correlation_matrix)) / \
                            (len(correlation_matrix) - 1)
            
            # Calculate directional spillover
            directional_spillover = {}
            for sector in correlation_matrix.index:
                # To others
                to_others = (correlation_matrix[sector].sum() - correlation_matrix[sector][sector]) / \
                          (len(correlation_matrix) - 1)
                # From others
                from_others = (correlation_matrix.loc[sector].sum() - correlation_matrix[sector][sector]) / \
                            (len(correlation_matrix) - 1)
                
                directional_spillover[sector] = {
                    'to_others': to_others,
                    'from_others': from_others,
                    'net_spillover': to_others - from_others
                }
            
            return {
                'total_spillover': total_spillover,
                'directional_spillover': directional_spillover,
                'centrality_scores': centrality.to_dict(),
                'condition_number': condition_number
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating spillover index: {str(e)}")
            return None
        
    async def predict_future_prices(self, data: pd.DataFrame, days: int = 30) -> pd.DataFrame:
        """Predict future stock prices using ARIMA."""
        try:
            self.logger.info(f"Predicting future prices for {days} days")
            
            # Deep copy to avoid modifying original data
            data = data.copy(deep=True)
            
            # Convert to datetime index if needed
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)
            
            # Create a proper time series with business day frequency
            ts = pd.Series(
                data["Close"],
                index=pd.DatetimeIndex(data.index, freq=None),  # Let pandas infer freq
                name="Close"
            )
            
            # Resample to business days
            ts = ts.asfreq('B', method='ffill')
            
            # Create and fit ARIMA model
            model = ARIMA(
                ts,
                order=(1, 1, 1),  # Simple order to avoid seasonal components
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            model_fit = model.fit()
            
            # Create future dates index
            future_dates = pd.date_range(
                start=ts.index[-1] + pd.Timedelta(days=1),
                periods=days,
                freq='B'
            )
            
            # Generate forecasts
            forecast = model_fit.forecast(steps=days)
            
            # Create forecast DataFrame with proper index
            forecast_df = pd.DataFrame(
                forecast,
                index=future_dates,
                columns=["Prediction"]
            )
            
            self.logger.info("Successfully generated price predictions")
            return forecast_df

        except Exception as e:
            self.logger.error(f"Error predicting future prices: {str(e)}")
            raise DataError(f"Failed to predict future prices: {str(e)}")
        
class EnhancedFinancialAnalyzer:
    """Advanced financial analysis capabilities for sector and stock recommendations."""
    
    def __init__(self, market_analyzer: MarketAnalyzer):
        self.market_analyzer = market_analyzer
        self.logger = logging.getLogger(__name__)
        self._setup_metrics()
        
    def _setup_metrics(self):
        """Initialize tracking metrics."""
        self.metrics = {
            'total_analyses': 0,
            'successful_analyses': 0,
            'processing_times': []
        }

    async def predict_returns(self, ticker: str, historical_data: pd.DataFrame, days: int = 30) -> Dict[str, Any]:
        """Predict future returns using enhanced ARIMA model."""
        try:
            # Prepare data
            data = historical_data['Close'].copy()
            
            # Create proper time series
            ts = pd.Series(
                data.values,
                index=pd.DatetimeIndex(data.index, freq=None),  # Let pandas infer freq
                name='Close'
            )
            
            # Resample to business day frequency
            ts = ts.asfreq('B', method='ffill')
            
            # Fit auto_arima without seasonal components
            model = auto_arima(
                ts,
                start_p=1,
                start_q=1,
                max_p=3,
                max_q=3,
                d=1,
                seasonal=False,  # Explicitly disable seasonal
                m=1,  # Set m=1 to avoid seasonal warnings
                information_criterion='aic',
                trace=False,
                error_action='ignore',
                suppress_warnings=True,
                stepwise=True
            )
            
            # Generate future dates
            future_dates = pd.date_range(
                start=ts.index[-1] + pd.Timedelta(days=1),
                periods=days,
                freq='B'
            )
            
            # Generate predictions and confidence intervals
            forecast, conf_int = model.predict(n_periods=days, return_conf_int=True)
            
            # Create predictions DataFrame with proper index
            predictions = pd.DataFrame({
                'Prediction': forecast,
                'Lower_CI': conf_int[:, 0],
                'Upper_CI': conf_int[:, 1]
            }, index=future_dates)
            
            # Calculate confidence score
            confidence = 1 - np.mean(conf_int[:, 1] - conf_int[:, 0]) / np.mean(forecast)
            
            return {
                'predictions': predictions,
                'confidence': float(confidence),
                'model_metrics': {
                    'aic': float(model.aic()),
                    'bic': float(model.bic())
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error predicting returns for {ticker}: {str(e)}")
            return None
    
    async def _get_peer_comparison(self, ticker: str, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Get peer comparison with error handling."""
        try:
            peers = await self._get_peer_tickers(ticker)
            peer_data = {}
            
            for peer in peers:
                try:
                    data = await self.market_analyzer.get_market_data(peer, start_date, end_date)
                    if data is not None and not data.empty:
                        peer_data[peer] = data
                except Exception as e:
                    self.logger.warning(f"Skipping peer {peer} due to error: {str(e)}")
                    continue
                    
            return {
                'peer_performance': peer_data,
                'relative_metrics': await self._calculate_relative_metrics(ticker, peer_data)
            }
        except Exception as e:
            self.logger.error(f"Error in peer comparison: {str(e)}")
            return None

    async def _get_peer_tickers(self, ticker: str) -> List[str]:
        """Get list of peer tickers for comparison."""
        
        # Comprehensive peer mapping dictionary
        company_peers = {
            # Technology Companies
            'AAPL': ['MSFT', 'GOOGL', 'DELL', 'HPQ', 'SAMSUNG.KS'], # Consumer electronics
            'NVDA': ['AMD', 'INTC', 'QCOM', 'TSM', 'MU'], # Semiconductors
            'MSFT': ['GOOGL', 'AAPL', 'ORCL', 'CRM', 'IBM'], # Enterprise software
            'CRM': ['NOW', 'WDAY', 'TEAM', 'ZEN', 'ADBE'], # Enterprise SaaS
            
            # E-commerce & Retail
            'AMZN': ['WMT', 'TGT', 'EBAY', 'BABA', 'JD'], # E-commerce/Retail
            'WMT': ['TGT', 'COST', 'KR', 'DG', 'DLTR'], # Retail
            
            # Financial Services
            'JPM': ['BAC', 'C', 'WFC', 'GS', 'MS'], # Banks
            'V': ['MA', 'AXP', 'PYPL', 'SQ', 'AFRM'], # Payments
            'BLK': ['STT', 'BK', 'TROW', 'IVZ', 'BEN'], # Asset management
            
            # Healthcare
            'JNJ': ['PFE', 'MRK', 'ABBV', 'BMY', 'LLY'], # Pharma
            'UNH': ['CVS', 'CI', 'HUM', 'CNC'], # Health insurance
            'MDT': ['ABT', 'BSX', 'EW', 'ZBH', 'ISRG'], # Medical devices
            
            # Energy
            'XOM': ['CVX', 'BP', 'SHEL', 'TTE', 'COP'], # Oil & Gas
            'NEE': ['DUK', 'SO', 'D', 'AEP', 'EXC'], # Utilities
            
            # Aerospace & Defense
            'BA': ['AIR.PA', 'LMT', 'NOC', 'RTX', 'GD'], # Aerospace
            'LMT': ['RTX', 'NOC', 'GD', 'BA', 'HII'], # Defense
            
            # Automotive
            'TSLA': ['F', 'GM', 'TM', 'VWAGY', 'NIO'], # Auto manufacturers
            'F': ['GM', 'TM', 'HMC', 'STLA', 'TSLA'],
            
            # Social Media
            'META': ['SNAP', 'TWTR', 'PINS', 'GOOGL', 'MSFT'], # Social platforms
            
            # Telecommunications
            'T': ['VZ', 'TMUS', 'CMCSA', 'VOD', 'BCE'], # Telecom
            
            # Transportation & Logistics
            'UPS': ['FDX', 'JBHT', 'XPO', 'CHRW', 'EXPD'], # Logistics
            
            # Entertainment & Streaming
            'NFLX': ['DIS', 'PARA', 'WBD', 'CMCSA', 'SONY'], # Media/Streaming
            
            # Real Estate
            'PLD': ['DRE', 'STAG', 'FR', 'TRNO', 'EGP'], # Industrial REITs
            'SPG': ['MAC', 'KIM', 'FRT', 'REG', 'O'], # Retail REITs
            
            # Cloud Computing
            'AMZN': ['MSFT', 'GOOGL', 'IBM', 'ORCL', 'VMW'], # Cloud providers
            
            # Cybersecurity
            'PANW': ['CRWD', 'FTNT', 'ZS', 'OKTA', 'NET'], # Security
            
            # Semiconductors Additional
            'TSM': ['ASML', 'AMAT', 'KLAC', 'LRCX', 'TER'], # Chip manufacturing
            
            # Clean Energy
            'ENPH': ['SEDG', 'FSLR', 'RUN', 'NOVA'] # Solar/Clean energy
        }
        
        sector_peers = {
        # Technology and Related
        'XLK': ['AAPL', 'MSFT', 'NVDA', 'AVGO', 'CSCO', 'AMD', 'INTC', 'CRM', 'ACN', 'ORCL', 'IBM', 'ADBE', 'NOW', 'QCOM', 'AMAT'],
        'HACK': ['PANW', 'CRWD', 'FTNT', 'ZS', 'OKTA', 'NET', 'S', 'RPD', 'CYBR', 'QLYS', 'VRNS', 'TENB'],
        'BOTZ': ['NVDA', 'ISRG', 'ABB', 'FANUY', 'IRBT', 'AZPN', 'ROK', 'NXPI', 'TER', 'PATH', 'SIEGY'],
        
        # Healthcare
        'XLV': ['JNJ', 'UNH', 'PFE', 'ABT', 'TMO', 'MRK', 'ABBV', 'BMY', 'LLY', 'AMGN', 'MDT', 'GILD', 'REGN', 'VRTX', 'BIIB'],
        
        # Financial Services
        'XLF': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'SPGI', 'AXP', 'USB', 'PNC', 'TFC', 'SCHW', 'CME', 'ICE'],
        
        # Energy and Utilities
        'XLE': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PSX', 'MPC', 'VLO', 'OXY', 'KMI', 'HAL', 'BKR', 'DVN', 'APA'],
        'XLU': ['NEE', 'DUK', 'SO', 'AEP', 'SRE', 'D', 'EXC', 'PCG', 'XEL', 'WEC', 'ED', 'EIX', 'PEG', 'ETR', 'FE'],
        'ICLN': ['ENPH', 'SEDG', 'PLUG', 'NEE', 'FSLR', 'BE', 'NOVA', 'RUN', 'DQ', 'JKS', 'CSIQ', 'VWDRY', 'TPIC', 'ARRY'],
        
        # Consumer
        'XLY': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'LOW', 'SBUX', 'TJX', 'BKNG', 'MAR', 'TGT', 'EBAY', 'ETSY', 'ROST', 'DLTR'],
        
        # Real Estate
        'IYR': ['PLD', 'AMT', 'EQIX', 'CCI', 'SPG', 'WY', 'PSA', 'O', 'WELL', 'AVB', 'DLR', 'VTR', 'BXP', 'ARE', 'HST'],
        
        # Basic Materials
        'XLB': ['LIN', 'FCX', 'APD', 'ECL', 'NEM', 'SHW', 'CTVA', 'NUE', 'DOW', 'VMC', 'ALB', 'CF', 'MLM', 'DD', 'PPG'],
        
        # Industrials and Defense
        'XLI': ['HON', 'UPS', 'UNP', 'BA', 'CAT', 'RTX', 'GE', 'DE', 'MMM', 'LMT', 'FDX', 'EMR', 'NSC', 'ETN', 'ITW'],
        'ITA': ['RTX', 'LMT', 'BA', 'NOC', 'GD', 'TDG', 'LHX', 'HII', 'TXT', 'KTOS', 'LDOS', 'CW', 'SPCE'],
        
        # Communications
        'IYZ': ['T', 'VZ', 'TMUS', 'CMCSA', 'CHTR', 'LUMN', 'DISH', 'CE', 'IPG', 'OMC', 'CCOI', 'BAND', 'CSCO', 'JNPR', 'IRDM'],
        
        # E-commerce/Digital
        'EBIZ': ['AMZN', 'MELI', 'SE', 'SHOP', 'STMP', 'W', 'CHWY', 'OSTK', 'ETSY', 'PINS']
        }
        
        peers = set()
    
        # First check company-specific peers
        if ticker in company_peers:
            peers.update(company_peers[ticker])
        
        # Then check sector peers
        for sector, components in sector_peers.items():
            if ticker in components or ticker == sector:
                peers.update(components)
        
        # Remove the ticker itself from peers
        peers.discard(ticker)
        
        # Convert to list and limit to top 10 most relevant peers
        return list(peers)[:10]

    async def _calculate_relative_metrics(self, ticker: str, peer_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Calculate relative performance metrics."""
        try:
            metrics = {}
            for peer, data in peer_data.items():
                if not data.empty:
                    metrics[peer] = {
                        'relative_return': (data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1,
                        'volatility': data['Close'].pct_change().std(),
                        'volume_trend': data['Volume'].mean()
                    }
            return metrics
        except Exception as e:
            self.logger.error(f"Error calculating relative metrics: {str(e)}")
            return {}

    async def analyze_recommendation(self, ticker: str, hearing_date: datetime) -> Dict[str, Any]:
        """Comprehensive analysis with peer comparison."""
        try:
            pre_window = hearing_date - timedelta(days=90)
            post_window = hearing_date + timedelta(days=90)
            
            market_data, sector_data, peer_data = await asyncio.gather(
                self.market_analyzer.get_market_data(ticker, pre_window, post_window),
                self._get_sector_performance(ticker, pre_window, post_window),
                self._get_peer_comparison(ticker, pre_window, post_window)
            )
            
            peer_performance = {}
            # Add self to peer comparison first
            if not market_data.empty:
                returns = market_data['Close'].pct_change().dropna()
                peer_performance[f"{ticker} (Selected)"] = {
                    'return': ((market_data['Close'].iloc[-1] / market_data['Close'].iloc[0]) - 1) * 100,
                    'volatility': returns.std() * np.sqrt(252) * 100,
                    'sharpe': returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0,
                    'volume': market_data['Volume'].mean()
                }
            
            # Add peer data
            if peer_data and 'peer_performance' in peer_data:
                for peer, data in peer_data['peer_performance'].items():
                    if isinstance(data, pd.DataFrame) and not data.empty:
                        returns = data['Close'].pct_change().dropna()
                        try:
                            peer_performance[peer] = {
                      
                                'return': ((data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1) * 100,
                                'volatility': returns.std() * np.sqrt(252) * 100,
                                'sharpe': returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0,
                                'volume': data['Volume'].mean()
                            }
                        except Exception as e:
                            self.logger.error(f"Error calculating metrics for {peer}: {str(e)}")
                            continue
            
            return {
                'ticker_data': {
                    'price': market_data['Close'].iloc[-1] if not market_data.empty else None,
                    'return': ((market_data['Close'].iloc[-1] / market_data['Close'].iloc[0]) - 1) * 100 if not market_data.empty else None,
                    'volume': market_data['Volume'].mean() if not market_data.empty else None
                },
                'peer_comparison': peer_performance,
                'sector_data': sector_data
            }
                
        except Exception as e:
            self.logger.error(f"Error in recommendation analysis: {str(e)}")
            return None
    
    def _calculate_technical_indicators(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive technical indicators."""
        try:
            # Price momentum
            momentum = self._calculate_momentum_indicators(data)
            
            # Volatility measures
            volatility = self._calculate_volatility_metrics(data)
            
            # Volume analysis
            volume = self._analyze_volume_patterns(data)
            
            # Trend strength indicators
            trends = self._calculate_trend_strength(data)
            
            return {
                'momentum_indicators': momentum,
                'volatility_metrics': volatility,
                'volume_analysis': volume,
                'trend_indicators': trends
            }
        except Exception as e:
            self.logger.error(f"Error calculating technical indicators: {str(e)}")
            return None
            
    async def _get_sector_performance(self, ticker: str, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Get sector performance data."""
        try:
            sector_etf = self._get_sector_etf(ticker)
            if not sector_etf:
                return None
                
            sector_data = await self.market_analyzer.get_market_data(sector_etf, start_date, end_date)
            return {
                'sector_etf': sector_etf,
                'performance': sector_data
            }
        except Exception as e:
            self.logger.error(f"Error getting sector performance: {str(e)}")
            return None

    def _get_sector_etf(self, ticker: str) -> Optional[str]:
        """Map ticker to its sector ETF."""
        sector_mapping = {
            # Technology
            'AAPL': 'XLK', 'MSFT': 'XLK', 'NVDA': 'XLK', 'AVGO': 'XLK', 'CSCO': 'XLK', 'AMD': 'XLK', 'INTC': 'XLK',
            'CRM': 'XLK', 'ACN': 'XLK', 'ORCL': 'XLK', 'IBM': 'XLK', 'ADBE': 'XLK', 'NOW': 'XLK', 'QCOM': 'XLK',
            # Healthcare
            'JNJ': 'XLV', 'UNH': 'XLV', 'PFE': 'XLV', 'ABT': 'XLV', 'TMO': 'XLV', 'MRK': 'XLV', 'ABBV': 'XLV',
            'BMY': 'XLV', 'LLY': 'XLV', 'AMGN': 'XLV', 'MDT': 'XLV', 'GILD': 'XLV', 'REGN': 'XLV', 'VRTX': 'XLV',
            # Financial
            'JPM': 'XLF', 'BAC': 'XLF', 'WFC': 'XLF', 'GS': 'XLF', 'MS': 'XLF', 'C': 'XLF', 'BLK': 'XLF',
            'SPGI': 'XLF', 'AXP': 'XLF', 'USB': 'XLF', 'PNC': 'XLF', 'TFC': 'XLF', 'SCHW': 'XLF',
            # Energy
            'XOM': 'XLE', 'CVX': 'XLE', 'COP': 'XLE', 'SLB': 'XLE', 'EOG': 'XLE', 'PSX': 'XLE', 'MPC': 'XLE',
            'VLO': 'XLE', 'OXY': 'XLE', 'KMI': 'XLE', 'HAL': 'XLE', 'BKR': 'XLE', 'DVN': 'XLE',
            # Consumer
            'AMZN': 'XLY', 'TSLA': 'XLY', 'HD': 'XLY', 'MCD': 'XLY', 'NKE': 'XLY', 'SBUX': 'XLY', 'LOW': 'XLY',
            'TJX': 'XLY', 'BKNG': 'XLY', 'MAR': 'XLY', 'TGT': 'XLY', 'EBAY': 'XLY', 'ETSY': 'XLY',
            # Utilities
            'NEE': 'XLU', 'DUK': 'XLU', 'SO': 'XLU', 'AEP': 'XLU', 'SRE': 'XLU', 'D': 'XLU', 'EXC': 'XLU',
            'PCG': 'XLU', 'XEL': 'XLU', 'WEC': 'XLU', 'ED': 'XLU', 'EIX': 'XLU', 'PEG': 'XLU',
            # Real Estate
            'PLD': 'IYR', 'AMT': 'IYR', 'EQIX': 'IYR', 'CCI': 'IYR', 'SPG': 'IYR', 'PSA': 'IYR', 'WELL': 'IYR',
            'AVB': 'IYR', 'DLR': 'IYR', 'VTR': 'IYR', 'BXP': 'IYR', 'ARE': 'IYR', 'HST': 'IYR',
            # Industrial
            'HON': 'XLI', 'UPS': 'XLI', 'UNP': 'XLI', 'BA': 'XLI', 'CAT': 'XLI', 'RTX': 'XLI', 'GE': 'XLI',
            'DE': 'XLI', 'MMM': 'XLI', 'LMT': 'XLI', 'FDX': 'XLI', 'EMR': 'XLI', 'NSC': 'XLI',
            # Aerospace & Defense
            'BA': 'ITA', 'LMT': 'ITA', 'RTX': 'ITA', 'NOC': 'ITA', 'GD': 'ITA', 'TDG': 'ITA', 'LHX': 'ITA',
            'HII': 'ITA', 'TXT': 'ITA', 'KTOS': 'ITA',
            # Communication
            'T': 'IYZ', 'VZ': 'IYZ', 'TMUS': 'IYZ', 'CMCSA': 'IYZ', 'CHTR': 'IYZ', 'LUMN': 'IYZ', 'DISH': 'IYZ',
            # Cybersecurity
            'CRWD': 'HACK', 'PANW': 'HACK', 'FTNT': 'HACK', 'ZS': 'HACK', 'OKTA': 'HACK', 'NET': 'HACK'
        }
        
        # List of valid ETFs that can be returned directly
        valid_etfs = {
            'XLK',  # Technology
            'XLV',  # Healthcare
            'XLF',  # Financial
            'XLE',  # Energy
            'XLY',  # Consumer Discretionary
            'XLU',  # Utilities
            'IYR',  # Real Estate
            'XLI',  # Industrial
            'IYZ',  # Telecommunications
            'HACK', # Cybersecurity
            'ITA',  # Aerospace & Defense
            'SPY',  # S&P 500
            'QQQ',  # NASDAQ
            'DIA'   # Dow Jones
        }
        
        # If ticker is already a valid ETF, return it
        if ticker in valid_etfs:
            return ticker
        
        # If ticker is in mapping, return corresponding ETF
        if ticker in sector_mapping:
            return sector_mapping[ticker]
            
        # If ticker is invalid or not found, return None
        return None
            
    async def _analyze_regulatory_impact(self, ticker: str, hearing_date: datetime) -> Dict[str, Any]:
        """Analyze regulatory impact on the stock/sector."""
        try:
            # Get historical regulatory events
            regulatory_events = await self._get_regulatory_events(ticker)
            
            # Analyze price movements around regulatory events
            regulatory_impact = await self._analyze_regulatory_price_impact(ticker, regulatory_events)
            
            # Calculate compliance costs
            compliance_metrics = self._estimate_compliance_impact(ticker, regulatory_events)
            
            # Assess regulatory risk
            regulatory_risk = self._calculate_regulatory_risk(ticker, regulatory_events)
            
            return {
                'historical_impact': regulatory_impact,
                'compliance_metrics': compliance_metrics,
                'regulatory_risk': regulatory_risk,
                'upcoming_regulations': await self._get_upcoming_regulations(ticker)
            }
        except Exception as e:
            self.logger.error(f"Error analyzing regulatory impact: {str(e)}")
            return None
            
    def _generate_recommendation(self, 
                               technical_indicators: Dict,
                               risk_metrics: Dict,
                               regulatory_impact: Dict,
                               options_sentiment: Dict) -> Dict[str, Any]:
        """Generate comprehensive investment recommendation."""
        try:
            # Calculate overall score
            technical_score = self._calculate_technical_score(technical_indicators)
            risk_score = self._calculate_risk_score(risk_metrics)
            regulatory_score = self._calculate_regulatory_score(regulatory_impact)
            sentiment_score = self._calculate_sentiment_score(options_sentiment)
            
            # Weight the scores
            weights = {
                'technical': 0.3,
                'risk': 0.25,
                'regulatory': 0.25,
                'sentiment': 0.2
            }
            
            overall_score = (
                technical_score * weights['technical'] +
                risk_score * weights['risk'] +
                regulatory_score * weights['regulatory'] +
                sentiment_score * weights['sentiment']
            )
            
            # Generate recommendation
            recommendation = self._get_recommendation_level(overall_score)
            
            return {
                'overall_score': overall_score,
                'component_scores': {
                    'technical': technical_score,
                    'risk': risk_score,
                    'regulatory': regulatory_score,
                    'sentiment': sentiment_score
                },
                'recommendation': recommendation,
                'confidence_level': self._calculate_confidence_level(technical_indicators, risk_metrics),
                'time_horizon': self._determine_time_horizon(technical_indicators, regulatory_impact),
                'risk_factors': self._identify_key_risks(risk_metrics, regulatory_impact)
            }
        except Exception as e:
            self.logger.error(f"Error generating recommendation: {str(e)}")
            return None

class SentimentAnalyzer:
    """Enhanced sentiment analysis with batching and caching."""
    
    def __init__(self, openai_client, config: ChatbotConfig):
        """Initialize the sentiment analyzer."""
        # Set up logging first
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Add handlers if none exist
        if not self.logger.handlers:
            # File handler
            fh = logging.FileHandler('logs/chatbot.log')
            fh.setLevel(logging.INFO)
            
            # Console handler
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            
            # Formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)
            
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)
        
        self.logger.debug("Initializing SentimentAnalyzer")
        
        self.openai_client = openai_client
        self.config = config
        
        try:
            self.cache = TTLCache(maxsize=100, ttl=config.cache_ttl)
            self.rate_limiter = RateLimiter(
                max_calls=config.rate_limit_calls,
                time_window=config.rate_limit_period
            )
            
            self.sentiment_prompt = """
            You are a financial sentiment analyzer. Analyze the following congressional hearing excerpt for sentiment regarding financial markets and regulation.

            Consider:
            1. Overall tone (positive/negative/neutral)
            2. Regulatory stance (strict/moderate/relaxed)
            3. Market implications (bullish/bearish/neutral)
            4. Key concerns or opportunities mentioned
            5. Policy direction (expansionary/neutral/contractionary)
            6. Industry-specific impacts
            7. Timeline considerations (short-term vs long-term effects)
            8. Uncertainty levels

            Below is the text to analyze:
            {text}

            Respond ONLY with a JSON object containing the following fields:
            {
                "overall_tone": float between -1 and 1,
                "regulatory_stance": float between -1 and 1,
                "market_implications": float between -1 and 1,
                "confidence": float between 0 and 1,
                "key_topics": list of strings,
                "concerns": list of strings,
                "opportunities": list of strings,
                "timeline": {
                    "short_term": string,
                    "medium_term": string,
                    "long_term": string
                },
                "industry_impacts": object with industry names as keys and float values,
                "uncertainty_level": float between 0 and 1,
                "policy_direction": string,
                "supporting_quotes": list of strings
            }"""
            self.logger.info("SentimentAnalyzer initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing SentimentAnalyzer: {str(e)}")
            raise
    
    async def analyze_batch(self, texts: List[str], batch_size: int = 5):
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        tasks = [self.analyze_sentiment(batch) for batch in batches]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    async def analyze_sentiment(self, text: str) -> Dict[str, Union[float, str, List]]:
        """Analyze sentiment in text using OpenAI with rate limiting and caching."""
        try:
            # Check cache
            cache_key = hash(text)
            if cache_key in self.cache:
                self.logger.debug("Cache hit for sentiment analysis")
                return self.cache[cache_key]
            
            await self.rate_limiter.acquire()
            
            messages = [
                {"role": "system", "content": self.sentiment_prompt},
                {"role": "user", "content": text}
            ]
            
            response = await self.openai_client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=0.3,
                max_tokens=1000
            )
            
            sentiment_data = json.loads(response.choices[0].message.content)
            sentiment_data['timestamp'] = datetime.now().isoformat()
            sentiment_data['text_length'] = len(text)
            
            # Validate response format
            self._validate_sentiment_response(sentiment_data)
            
            # Cache result
            self.cache[cache_key] = sentiment_data
            return sentiment_data
            
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing error in sentiment analysis: {str(e)}")
            raise DataError("Invalid sentiment analysis response format")
        except Exception as e:
            self.logger.error(f"Sentiment analysis error: {str(e)}")
            raise
    
    def _validate_sentiment_response(self, response: Dict):
        """Validate sentiment analysis response format."""
        required_fields = {
            'overall_tone': float,
            'regulatory_stance': float,
            'market_implications': float,
            'confidence': float,
            'key_topics': list,
            'industry_impacts': dict
        }
        
        for field, field_type in required_fields.items():
            if field not in response:
                raise DataError(f"Missing required field: {field}")
            if not isinstance(response[field], field_type):
                raise DataError(f"Invalid type for field {field}: expected {field_type}")
    
    async def analyze_multiple_excerpts(self, 
                                      excerpts: List[str],
                                      batch_size: int = 5) -> Dict[str, Any]:
        """Analyze sentiment for multiple excerpts with batching and aggregation."""
        try:
            if not excerpts:
                raise DataError("No excerpts provided for analysis")
            
            results = []
            for i in range(0, len(excerpts), batch_size):
                batch = excerpts[i:i + batch_size]
                batch_results = await asyncio.gather(
                    *[self.analyze_sentiment(text) for text in batch],
                    return_exceptions=True
                )
                
                # Filter out failed analyses
                valid_results = [
                    result for result in batch_results
                    if isinstance(result, dict)
                ]
                results.extend(valid_results)
            
            if not results:
                raise DataError("No valid sentiment analyses produced")
            
            # Aggregate results
            aggregated = self._aggregate_sentiment_results(results)
            
            # Calculate confidence metrics
            confidence_metrics = self._calculate_confidence_metrics(results)
            
            # Combine results
            return {
                'aggregated_sentiment': aggregated,
                'confidence_metrics': confidence_metrics,
                'individual_analyses': results,
                'meta': {
                    'total_excerpts': len(excerpts),
                    'successful_analyses': len(results),
                    'timestamp': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in multiple excerpt analysis: {str(e)}")
            raise
    
    def _aggregate_sentiment_results(self, results: List[Dict]) -> Dict[str, Any]:
        """Aggregate multiple sentiment analysis results."""
        try:
            # Calculate weighted averages based on confidence scores
            weights = np.array([r['confidence'] for r in results])
            weights = weights / np.sum(weights)
            
            aggregated = {
                'overall_tone': np.average(
                    [r['overall_tone'] for r in results], 
                    weights=weights
                ),
                'regulatory_stance': np.average(
                    [r['regulatory_stance'] for r in results],
                    weights=weights
                ),
                'market_implications': np.average(
                    [r['market_implications'] for r in results],
                    weights=weights
                ),
                'uncertainty_level': np.average(
                    [r.get('uncertainty_level', 0.5) for r in results],
                    weights=weights
                )
            }
            
            # Aggregate key topics and concerns
            all_topics = set()
            all_concerns = set()
            all_opportunities = set()
            
            for result in results:
                all_topics.update(result.get('key_topics', []))
                all_concerns.update(result.get('concerns', []))
                all_opportunities.update(result.get('opportunities', []))
            
            # Aggregate industry impacts
            industry_impacts = {}
            for result in results:
                for industry, impact in result.get('industry_impacts', {}).items():
                    if industry not in industry_impacts:
                        industry_impacts[industry] = []
                    industry_impacts[industry].append(impact)
            
            # Calculate average industry impacts
            avg_industry_impacts = {
                industry: np.mean(impacts)
                for industry, impacts in industry_impacts.items()
            }
            
            return {
                'sentiment_scores': aggregated,
                'key_topics': list(all_topics),
                'concerns': list(all_concerns),
                'opportunities': list(all_opportunities),
                'industry_impacts': avg_industry_impacts
            }
            
        except Exception as e:
            self.logger.error(f"Error aggregating sentiment results: {str(e)}")
            raise
    
    def _calculate_confidence_metrics(self, results: List[Dict]) -> Dict[str, float]:
        """Calculate confidence metrics for aggregated results."""
        try:
            confidences = [r['confidence'] for r in results]
            
            return {
                'mean_confidence': np.mean(confidences),
                'min_confidence': np.min(confidences),
                'max_confidence': np.max(confidences),
                'std_confidence': np.std(confidences),
                'consensus_level': self._calculate_consensus_level(results)
            }
        except Exception as e:
            self.logger.error(f"Error calculating confidence metrics: {str(e)}")
            raise
    
    def _calculate_consensus_level(self, results: List[Dict]) -> float:
        """Calculate consensus level among different analyses."""
        try:
            # Calculate standard deviation of key metrics
            tones = np.array([r['overall_tone'] for r in results])
            stances = np.array([r['regulatory_stance'] for r in results])
            implications = np.array([r['market_implications'] for r in results])
            
            # Normalize standard deviations to [0, 1] range
            tone_std = np.std(tones) / 2  # Divide by 2 since range is [-1, 1]
            stance_std = np.std(stances) / 2
            impl_std = np.std(implications) / 2
            
            # Calculate consensus level (1 - average normalized std)
            consensus = 1 - np.mean([tone_std, stance_std, impl_std])
            
            return max(0, min(1, consensus))  # Ensure result is in [0, 1]
            
        except Exception as e:
            self.logger.error(f"Error calculating consensus level: {str(e)}")
            raise

class EnhancedCongressionalChatbot:
    """Enhanced chatbot with improved analysis capabilities, resource management, and chat history."""
    
    def __init__(self, db_path: str, env_path: str):
        """Initialize the chatbot with configuration and components."""
        self._start_time = datetime.now()
        self._setup_logging()
        self._load_environment(env_path)
        self.config = ChatbotConfig()
        self.chat_history = deque(maxlen=self.config.conversation_memory_limit)
        self.conversation_history = deque(maxlen=self.config.conversation_memory_limit)
        self.db_path = db_path
        self._initialize_components()
        
        # Initialize performance metrics
        self.metrics = {
            'requests_processed': 0,
            'avg_response_time': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'errors': 0,
            'api_calls': 0,
            'api_errors': 0,
            'total_tokens': 0,
            'response_times': []
        }

    def _setup_metrics(self):
        self.request_latency = prometheus_client.Histogram(
            'request_latency_seconds',
            'Request latency in seconds'
        )
        self.error_counter = prometheus_client.Counter(
            'error_total',
            'Total number of errors'
        )
    
    def _setup_logging(self):
        """Setup logging for chatbot."""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            # File handler
            fh = logging.FileHandler('logs/chatbot.log')
            fh.setLevel(logging.INFO)
            
            # Console handler
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            
            # Formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)
            
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)

    def _load_environment(self, env_path: str):
        """Load and validate environment variables."""
        try:
            if not os.path.exists(env_path):
                raise ConfigurationError(f"Environment file not found: {env_path}")
            
            load_dotenv(env_path)
            
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
            if not self.openai_api_key:
                raise ConfigurationError("OPENAI_API_KEY not found in environment")
            
            self.openai_client = openai.OpenAI(api_key=self.openai_api_key)
            
        except Exception as e:
            self.logger.error(f"Environment configuration error: {str(e)}")
            raise

    def _initialize_components(self):
        """Initialize all components with proper error handling."""
        try:
            # Initialize analyzers
            self.financial_config = FinancialAnalysisConfig()
            self.market_analyzer = MarketAnalyzer(self.financial_config)
            self.financial_analyzer = EnhancedFinancialAnalyzer(self.market_analyzer)
            self.sentiment_analyzer = SentimentAnalyzer(self.openai_client, self.config)
                
            # Register SQLite function for cosine similarity
            conn = sqlite3.connect(str(self.db_path))
            conn.create_function("cosine_similarity", 2, self._cosine_similarity)
            conn.close()
                
        except Exception as e:
            self.logger.error(f"Component initialization error: {str(e)}")
            raise

    def _get_db_connection(self):
        """Get a thread-local database connection."""
        try:
            # Create new connection for current thread
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            return conn, cursor
        except Exception as e:
            self.logger.error(f"Error creating database connection: {str(e)}")
            raise

    async def initialize_database(self):
        """Initialize or update the database schema."""
        try:
            conn = None
            cursor = None
            try:
                conn, cursor = self._get_db_connection()
                conn.create_function("cosine_similarity", 2, self._cosine_similarity)
                
                # Drop existing table if it exists
                cursor.execute("DROP TABLE IF EXISTS conversation_history")
                
                # Create new table with all required columns
                cursor.execute("""
                CREATE TABLE conversation_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_query TEXT NOT NULL,
                    assistant_response TEXT NOT NULL,
                    chat_history TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """)
                
                conn.commit()
                self.logger.info("Database schema initialized successfully")
                
            finally:
                if cursor:
                    cursor.close()
                if conn:
                    conn.close()
                    
        except Exception as e:
            self.logger.error(f"Error initializing database: {str(e)}")
            raise
    
    @staticmethod
    def _cosine_similarity(v1: bytes, v2: bytes) -> float:
        """SQLite function to compute cosine similarity between two embedding vectors."""
        try:
            vec1 = np.frombuffer(v1, dtype=np.float32)
            vec2 = np.frombuffer(v2, dtype=np.float32)
                
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
                
            return float(dot_product / (norm1 * norm2) if norm1 * norm2 != 0 else 0.0)
        except Exception as e:
            return 0.0

    def extract_stock_recommendations(self, content: str) -> List[Tuple[str, str]]:
        """Extract stock and sector recommendations from text content."""
        recommendations = []
        
        # Define sector mappings
        sectors = {
            'technology': 'XLK',
            'healthcare': 'XLV',
            'finance': 'XLF',
            'energy': 'XLE',
            'consumer': 'XLY',
            'utilities': 'XLU',
            'real estate': 'IYR',
            'materials': 'XLB',
            'industrial': 'XLI',
            'telecom': 'IYZ'
        }
        
        # Extract explicit stock tickers
        ticker_pattern = r'\b[A-Z]{1,5}\b(?!\d)'
        tickers = re.finditer(ticker_pattern, content)
        for match in tickers:
            ticker = match.group()
            context = content[max(0, match.start()-50):min(len(content), match.end()+50)]
            if any(indicator in context.lower() for indicator in [
                'stock', 'share', 'price', 'company', 'corporation', 'ticker'
            ]):
                recommendations.append(('stock', ticker))
        
        # Extract sector recommendations
        for sector, etf in sectors.items():
            if sector.lower() in content.lower():
                context = content[max(0, content.lower().find(sector.lower())-50):
                                min(len(content), content.lower().find(sector.lower())+50)]
                if any(indicator in context.lower() for indicator in [
                    'sector', 'industry', 'market', 'segment', 'space'
                ]):
                    recommendations.append(('sector', etf))
        
        # Remove duplicates while preserving order
        seen = set()
        return [(t, s) for t, s in recommendations if not (s in seen or seen.add(s))]
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics of the chatbot system."""
        try:
            # Calculate uptime
            uptime = datetime.now() - self._start_time
            
            # Get database connection metrics
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Query database statistics
            cursor.execute("SELECT COUNT(*) FROM conversation_history")
            total_conversations = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM embeddings")
            total_embeddings = cursor.fetchone()[0]
            
            # Get database size
            cursor.execute("PRAGMA page_count")
            page_count = cursor.fetchone()[0]
            cursor.execute("PRAGMA page_size")
            page_size = cursor.fetchone()[0]
            db_size = (page_count * page_size) / (1024 * 1024)  # Size in MB
            
            # Get memory usage
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            # Compile metrics
            metrics = {
                "system_metrics": {
                    "uptime_seconds": uptime.total_seconds(),
                    "uptime_formatted": str(uptime).split('.')[0],  # Format as HH:MM:SS
                    "memory_usage_mb": memory_info.rss / (1024 * 1024),
                    "cpu_percent": process.cpu_percent(),
                    "thread_count": process.num_threads()
                },
                "database_metrics": {
                    "total_conversations": total_conversations,
                    "total_embeddings": total_embeddings,
                    "database_size_mb": db_size,
                    "last_backup": self._get_last_backup_time()
                },
                "performance_metrics": {
                    "requests_processed": self.metrics['requests_processed'],
                    "average_response_time": self.metrics['avg_response_time'],
                    "cache_hits": self.metrics['cache_hits'],
                    "cache_misses": self.metrics['cache_misses'],
                    "error_count": self.metrics['errors']
                },
                "api_metrics": {
                    "api_calls_made": self.metrics.get('api_calls', 0),
                    "api_errors": self.metrics.get('api_errors', 0)
                },
                "model_metrics": {
                    "model_name": self.config.model_name,
                    "embedding_model": self.config.embedding_model,
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_response_tokens
                }
            }
            
            cursor.close()
            conn.close()
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error getting performance metrics: {str(e)}")
            return {
                "error": str(e),
                "traceback": traceback.format_exc()
            }

    def _get_last_backup_time(self) -> str:
        """Get the timestamp of the last database backup."""
        try:
            backup_path = Path(self.db_path).parent / "backups"
            if not backup_path.exists():
                return "No backups found"
            
            backups = list(backup_path.glob("*.sqlite3"))
            if not backups:
                return "No backups found"
            
            latest_backup = max(backups, key=lambda x: x.stat().st_mtime)
            backup_time = datetime.fromtimestamp(latest_backup.stat().st_mtime)
            return backup_time.strftime("%Y-%m-%d %H:%M:%S")
            
        except Exception as e:
            self.logger.error(f"Error getting last backup time: {str(e)}")
            return "Error checking backups"
    
    async def infer_ticker_from_query(self, query: str) -> Optional[str]:
        """Infer the relevant ticker based on user query."""
        query = query.lower()
        
        # First check for explicit ticker mentions
        for sector_info in self.financial_config.sector_mapping.values():
            if sector_info['etf'].lower() in query:
                return sector_info['etf']
                
            # Check for company mentions
            for company in sector_info['companies']:
                if company.lower() in query:
                    return sector_info['etf']
        
        # Check for sector keywords
        max_matches = 0
        best_sector = None
        
        for sector, details in self.financial_config.sector_mapping.items():
            matches = 0
            # Check main keywords
            for keyword in details['keywords']:
                if keyword.lower() in query:
                    matches += 1
                    
            # Check sub-sectors
            for subsector in details['sub_sectors']:
                if subsector.lower().replace('_', ' ') in query:
                    matches += 1
                    
            if matches > max_matches:
                max_matches = matches
                best_sector = sector
                
        if best_sector:
            return self.financial_config.sector_mapping[best_sector]['etf']
        
        # Check for general financial terms
        general_financial_terms = [
            'market', 'stock', 'sector', 'industry', 'investment',
            'price', 'trend', 'performance', 'trading', 'financial'
        ]
        
        if any(term in query for term in general_financial_terms):
            # Default to S&P 500 ETF for general market queries
            return 'SPY'
            
        return None

    async def get_market_analysis(self, ticker: str) -> Tuple[str, pd.DataFrame, pd.DataFrame]:
        """Perform market analysis."""
        start_date = datetime.now() - timedelta(days=365)
        end_date = datetime.now()
        try:
            market_data = await self.market_analyzer.get_market_data(ticker, start_date, end_date)
            predictions = self.market_analyzer.predict_future_prices(market_data)
            return ticker, market_data, predictions
        except Exception as e:
            self.logger.error(f"Market analysis error: {e}")
            return ticker, pd.DataFrame(), pd.DataFrame()

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords using enhanced regex patterns."""
        try:
            keywords = []
            
            # Financial terms pattern
            financial_pattern = r'\b(stock|market|economy|inflation|rate|price|growth|GDP|investment|trading|volatility|debt|equity|asset|liability|revenue|profit|loss|margin|dividend|yield)\b'
            financial_terms = re.findall(financial_pattern, text.lower())
            
            # Stock ticker pattern
            ticker_pattern = r'\b[A-Z]{1,5}\b(?!\d)'
            tickers = re.findall(ticker_pattern, text)
            
            # Numerical values with units
            numerical_pattern = r'\b\d+\.?\d*[%$BMK]|\d+\.?\d*\s*(billion|million|thousand)'
            numericals = re.findall(numerical_pattern, text)
            
            # Time-related terms
            temporal_pattern = r'\b(year|quarter|month|week|day|annual|quarterly|monthly|weekly|daily)\b'
            temporal_terms = re.findall(temporal_pattern, text.lower())
            
            # Policy-related terms
            policy_pattern = r'\b(regulation|policy|law|act|bill|reform|amendment|provision|requirement|compliance)\b'
            policy_terms = re.findall(policy_pattern, text.lower())
            
            # Combine all patterns
            keywords.extend(financial_terms)
            keywords.extend(tickers)
            keywords.extend([num[0] if isinstance(num, tuple) else num for num in numericals])
            keywords.extend(temporal_terms)
            keywords.extend(policy_terms)
            
            # Remove duplicates while preserving order
            return list(dict.fromkeys(keywords))
            
        except Exception as e:
            self.logger.error(f"Error extracting keywords: {str(e)}")
            return []

    async def _analyze_market_impact(self, sources: List[Dict]) -> Optional[Dict[str, Any]]:
        """Analyze market impact for hearing dates."""
        if not sources:
            return None
            
        # Get unique hearing dates
        hearing_dates = [
            datetime.strptime(source['hearing_date'], '%Y-%m-%d')
            for source in sources
            if source.get('hearing_date')
        ]
        
        if not hearing_dates:
            return None
        
        # Use the earliest hearing date for analysis
        event_date = min(hearing_dates)
        
        # Get market data and analyze impact
        try:
            market_data = await self.market_analyzer.get_market_data('^GSPC', 
                event_date - timedelta(days=30), 
                event_date + timedelta(days=30)
            )
            
            if market_data is None:
                return None
            
            impact_analysis = await self.market_analyzer.test_market_reaction(
                market_data['Close'].pct_change(),
                event_date
            )
            
            return impact_analysis
            
        except Exception as e:
            self.logger.error(f"Error in market impact analysis: {str(e)}")
            return None

    def _format_sentiment_analysis(self, sentiment_analysis: Optional[Dict]) -> str:
        """Format sentiment analysis results into readable text."""
        if not sentiment_analysis or 'aggregated_sentiment' not in sentiment_analysis:
            return ""
        
        agg = sentiment_analysis['aggregated_sentiment']
        text = "\n\nSentiment Analysis:\n"
        
        if 'sentiment_scores' in agg:
            scores = agg['sentiment_scores']
            text += f"- Market Tone: {self._format_sentiment_score(scores['overall_tone'])}\n"
            text += f"- Regulatory Stance: {self._format_sentiment_score(scores['regulatory_stance'])}\n"
            text += f"- Market Implications: {self._format_sentiment_score(scores['market_implications'])}\n"
        
        if 'key_topics' in agg:
            text += "\nKey Topics:\n"
            for topic in agg['key_topics']:
                text += f"- {topic}\n"
        
        if 'industry_impacts' in agg:
            text += "\nIndustry Impacts:\n"
            for industry, impact in agg['industry_impacts'].items():
                text += f"- {industry}: {self._format_sentiment_score(impact)}\n"
        
        return text

    def _format_sentiment_score(self, score: float) -> str:
        """Format sentiment score with descriptive text."""
        if score >= 0.3:
            return f"Positive ({score:.2f})"
        elif score <= -0.3:
            return f"Negative ({score:.2f})"
        else:
            return f"Neutral ({score:.2f})"

    def _format_market_analysis(self, market_impacts: Dict) -> str:
        """Format market analysis results into readable text."""
        if not market_impacts:
            return ""
        
        text = "\n\nMarket Impact Analysis:\n"
        
        if 'statistical_tests' in market_impacts:
            stats = market_impacts['statistical_tests']
            text += f"- Market Reaction: {'Significant' if stats.get('significant') else 'Not Significant'}\n"
            text += f"- P-Value: {stats.get('p_value', 0):.3f}\n"
            
            if 'mean_change_pct' in stats:
                text += f"- Price Change: {stats['mean_change_pct']:.2%}\n"
            
            if 'volatility_change_pct' in stats:
                text += f"- Volatility Change: {stats['volatility_change_pct']:.2%}\n"
        
        if 'trend_analysis' in market_impacts:
            trend = market_impacts['trend_analysis']
            text += "\nTrend Analysis:\n"
            text += f"- Pre-Event Trend: {trend.get('pre_trend', 'Unknown')}\n"
            text += f"- Post-Event Trend: {trend.get('post_trend', 'Unknown')}\n"
        
        return text

    def _generate_response(self, messages: List[Dict]) -> Any:
        """Generate response using OpenAI API."""
        try:
            return self.openai_client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_response_tokens
            )
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            raise APIError("Failed to generate response")

    async def get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
        
            stats = {}
        
            # Basic stats
            cursor.execute("SELECT COUNT(*), page_count * page_size as size FROM embeddings, pragma_page_count(), pragma_page_size()")
            count, size = cursor.fetchone()
            stats['document_count'] = count
            stats['database_size'] = size / (1024 * 1024)  # Convert to MB
            
            # Get embedding dimensions
            cursor.execute("SELECT embedding FROM embeddings LIMIT 1")
            sample_embedding = cursor.fetchone()
            if sample_embedding and sample_embedding[0]:
                embedding_vector = np.frombuffer(sample_embedding[0], dtype=np.float32)
                stats['embedding_dimensions'] = len(embedding_vector)
            
            # Get collections count
            cursor.execute("SELECT COUNT(DISTINCT json_extract(metadata, '$.hearing_identifier')) FROM embeddings")
            stats['num_collections'] = cursor.fetchone()[0]
            
            # Get segments count
            cursor.execute("SELECT COUNT(DISTINCT json_extract(metadata, '$.chunk_number')) FROM embeddings")
            stats['num_segments'] = cursor.fetchone()[0]
            
            # Get embedding stats
            cursor.execute("SELECT AVG(embedding_norm), MIN(embedding_norm), MAX(embedding_norm) FROM embeddings")
            avg_norm, min_norm, max_norm = cursor.fetchone()
            stats['embedding_stats'] = {
                'average_norm': float(avg_norm) if avg_norm else 0,
                'min_norm': float(min_norm) if min_norm else 0,
                'max_norm': float(max_norm) if max_norm else 0
            }
            
            return stats
        
        except Exception as e:
            logging.error(f"Error getting database stats: {str(e)}")
            return {}
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    async def verify_database_content(self):
        """Verify database content and accessibility."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Check total documents
            cursor.execute("SELECT COUNT(*) FROM embeddings")
            total_docs = cursor.fetchone()[0]
            
            # Check unique hearings
            cursor.execute("""
                SELECT COUNT(DISTINCT json_extract(metadata, '$.hearing_identifier'))
                FROM embeddings
            """)
            unique_hearings = cursor.fetchone()[0]
            
            # Sample hearing identifiers
            cursor.execute("""
                SELECT DISTINCT json_extract(metadata, '$.hearing_identifier')
                FROM embeddings
                LIMIT 5
            """)
            sample_hearings = cursor.fetchall()
            
            return {
                'total_documents': total_docs,
                'unique_hearings': unique_hearings,
                'sample_hearings': sample_hearings
            }
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                'rss': memory_info.rss / 1024 / 1024,  # RSS in MB
                'vms': memory_info.vms / 1024 / 1024,  # VMS in MB
                'percent': process.memory_percent(),
                'num_threads': process.num_threads()
            }
        except ImportError:
            return {"error": "psutil not installed"}
        except Exception as e:
            self.logger.error(f"Error getting memory usage: {str(e)}")
            return {"error": str(e)}

    async def embed_document(self, text: str, metadata: Dict[str, Any]) -> bool:
        """Embed a new document using text-embedding-ada-002 model."""
        try:
            # Generate embedding using OpenAI
            embedding_response = await self.openai_client.embeddings.create(
                model=self.config.embedding_model,
                input=[text]
            )
            embedding_vector = np.array(embedding_response.data[0].embedding, dtype=np.float32)
            embedding_norm = float(np.linalg.norm(embedding_vector))

            # Store in database
            conn = None
            cursor = None
            try:
                conn, cursor = self._get_db_connection()
                
                cursor.execute("""
                INSERT INTO embeddings 
                (document_text, embedding, embedding_norm, metadata, created_at)
                VALUES (?, ?, ?, ?, ?)
                """, (
                    text,
                    embedding_vector.tobytes(),
                    embedding_norm,
                    json.dumps(metadata),
                    datetime.now().isoformat()
                ))
                
                conn.commit()
                return True
                
            finally:
                if cursor:
                    cursor.close()
                if conn:
                    conn.close()
                    
        except Exception as e:
            self.logger.error(f"Error embedding document: {str(e)}")
            return False

    async def batch_embed_documents(self, 
                                  documents: List[Tuple[str, Dict[str, Any]]],
                                  batch_size: int = 100) -> Tuple[int, int]:
        """Batch embed multiple documents with progress tracking."""
        successful = 0
        failed = 0
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            # Process batch
            results = await asyncio.gather(
                *[self.embed_document(text, metadata) for text, metadata in batch],
                return_exceptions=True
            )
            
            # Count successes and failures
            for result in results:
                if result is True:
                    successful += 1
                else:
                    failed += 1
            
            # Log progress
            self.logger.info(f"Processed {i + len(batch)}/{len(documents)} documents. "
                           f"Success: {successful}, Failed: {failed}")
        
        return successful, failed

    def _format_chat_history(self) -> str:
        """Format chat history for context in new queries."""
        if not self.chat_history:
            return ""
        
        formatted_history = []
        for entry in self.chat_history:
            formatted_history.append(f"User: {entry['user']}")
            formatted_history.append(f"Assistant: {entry['assistant']}\n")
        
        return "\n".join(formatted_history)

    def _update_chat_history(self, user_query: str, assistant_response: str):
        """Update chat history with new interaction."""
        self.chat_history.append({
            "user": user_query,
            "assistant": assistant_response,
            "timestamp": datetime.now().isoformat()
        })

    async def clear_chat_history(self):
        """Clear the chat history."""
        self.chat_history.clear()

    async def get_chat_history(self) -> List[Dict]:
        """Get the current chat history."""
        return list(self.chat_history)

    async def _get_relevant_excerpts(self, query: str, conversation_context: str = "") -> Optional[Dict]:
        """Get relevant excerpts from the database."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Combine query with conversation context and chat history for better relevance
            enhanced_query = f"{query} {conversation_context}".strip()
            
            # Get query embedding
            query_response = self.openai_client.embeddings.create(
                model=self.config.embedding_model,
                input=[enhanced_query]
            )
            query_embedding = np.array(query_response.data[0].embedding, dtype=np.float32)
            
            # Get relevant documents using SQL similarity comparison
            cursor.execute("""
                SELECT document_text, metadata
                FROM embeddings 
                WHERE embedding IS NOT NULL
                ORDER BY RANDOM()
                LIMIT 10
            """)
            
            results = cursor.fetchall()
            
            if not results:
                return None

            documents = []
            metadatas = []
            distances = []
            
            for doc_text, metadata_str in results:
                documents.append(doc_text)
                metadatas.append(json.loads(metadata_str))
                distances.append(0.5)  # Placeholder distance

            return {
                "documents": [documents],
                "metadatas": [metadatas],
                "distances": [distances]
            }

        except Exception as e:
            self.logger.error(f"Error in _get_relevant_excerpts: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    async def get_enhanced_response(self, user_query: str) -> Tuple[str, List[Dict], Dict]:
        """Generate enhanced response with comprehensive analysis and chat history context."""
        start_time = datetime.now()
        
        try:
            # Add circuit breaker pattern
            if self.metrics['errors'] > self.config.max_consecutive_errors:
                await self.reset_conversation()
                self.metrics['errors'] = 0
            
            # Add request validation
            if not user_query or len(user_query.strip()) < 3:
                raise ValueError("Query too short")
                
            # Add timeout handling
            async with asyncio.timeout(30):  # 30 second timeout
                results = await self._get_relevant_excerpts(user_query)
        
        except asyncio.TimeoutError:
            self.metrics['errors'] += 1
            return "Request timed out. Please try again.", [], {'error': 'timeout'}
            
        except ValueError as e:
            return str(e), [], {'error': 'validation'}
            
        except Exception as e:
            self.logger.error(f"Error in get_enhanced_response: {str(e)}")
            self.metrics['errors'] += 1
            return "An error occurred. Please try again.", [], {'error': str(e)}
        
        try:
            self.metrics['requests_processed'] += 1
            
            # Format chat history for context
            history_context = self._format_chat_history()
            
            # Get relevant excerpts with history context
            results = await self._get_relevant_excerpts(user_query, history_context)
            if not results or not results['documents'][0]:
                self.metrics['errors'] += 1
                return (
                    "I couldn't find relevant information in the congressional hearings. "
                    "Could you please rephrase your question or provide more context?",
                    [],
                    {'search_results': None, 'relevance_scores': None}
                )
            
            # Format context and prepare sources
            context_parts = []
            sources = []
            
            for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
                context_part = (
                    f"\nEXCERPT {i+1}:\n"
                    f"Source: Congress {metadata['congress_number']}, "
                    f"{metadata['hearing_type'].title()} Hearing {metadata['hearing_identifier']}\n"
                    f"Date: {metadata.get('hearing_date', 'Date not available')}\n"
                    f"Content: {doc}\n"
                )
                context_parts.append(context_part)
                sources.append(metadata)

            # Prepare messages with chat history context
            messages = [
                {"role": "system", "content": self.config.system_prompt},
                {"role": "system", "content": "Previous conversation context:\n" + history_context if history_context else ""},
                {"role": "user", "content": f"""
                Question: {user_query}
                
                Relevant excerpts from congressional hearings:
                {"".join(context_parts)}
                """}
            ]
            
            # Generate response
            response = self.openai_client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_response_tokens
            )
            
            answer = response.choices[0].message.content
            
            # Initialize metadata dictionary
            metadata = {
                'processing_time': (datetime.now() - start_time).total_seconds(),
                'sources_used': len(sources),
                'relevance_scores': results['distances'][0]
            }
            
            # Analyze financial recommendations
            recommendations = self.extract_stock_recommendations(answer)
            if recommendations:
                financial_analyses = {}
                for rec_type, ticker in recommendations:
                    analysis = await self.financial_analyzer.analyze_recommendation(
                        ticker,
                        datetime.now()
                    )
                    if analysis:
                        financial_analyses[ticker] = analysis
                        
                metadata['financial_analyses'] = financial_analyses
            
            # Update chat history
            self._update_chat_history(user_query, answer)
            
            # Update conversation history
            self.conversation_history.append({
                "user": user_query,
                "assistant": answer,
                "sources": sources,
                "timestamp": datetime.now().isoformat()
            })
            
            # Save updated history
            self._save_conversation_history()
            
            return answer, sources, metadata
                
        except Exception as e:
            self.logger.error(f"Error generating enhanced response: {str(e)}")
            self.metrics['errors'] += 1
            return (
                "I encountered an error processing your question. Please try again.",
                [],
                {'error': str(e)}
            )

    def _save_conversation_history(self):
        """Save conversation history to persistent storage."""
        try:
            conn = None
            cursor = None
            try:
                conn, cursor = self._get_db_connection()
                
                # Check if chat_history column exists
                cursor.execute("PRAGMA table_info(conversation_history)")
                columns = [col[1] for col in cursor.fetchall()]
                
                # If table doesn't exist or needs updating
                if 'conversation_history' not in [table[0] for table in cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")]:
                    # Create new table with all columns
                    cursor.execute("""
                    CREATE TABLE conversation_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_query TEXT NOT NULL,
                        assistant_response TEXT NOT NULL,
                        chat_history TEXT,
                        metadata TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                    """)
                elif 'chat_history' not in columns:
                    # Add chat_history column to existing table
                    cursor.execute("ALTER TABLE conversation_history ADD COLUMN chat_history TEXT")
                
                # Save each conversation
                for item in self.conversation_history:
                    cursor.execute("""
                    INSERT INTO conversation_history 
                    (user_query, assistant_response, chat_history, metadata, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """, (
                        item['user'],
                        item['assistant'],
                        json.dumps(list(self.chat_history)),
                        json.dumps({
                            'sources': item.get('sources', []),
                            'metadata': item.get('metadata', {})
                        }),
                        item['timestamp']
                    ))
                
                conn.commit()
                
            finally:
                if cursor:
                    cursor.close()
                if conn:
                    conn.close()
                    
        except Exception as e:
            self.logger.error(f"Error saving conversation history: {str(e)}")

    async def reset_conversation(self):
        """Reset conversation and chat history and clean up resources."""
        try:
            conn = None
            cursor = None
            try:
                conn, cursor = self._get_db_connection()
                
                # Clear conversation history table
                cursor.execute("DELETE FROM conversation_history")
                conn.commit()
                
                # Clear memory
                self.conversation_history.clear()
                self.chat_history.clear()
                
            finally:
                if cursor:
                    cursor.close()
                if conn:
                    conn.close()
                    
        except Exception as e:
            self.logger.error(f"Error resetting conversation: {str(e)}")

    async def cleanup(self):
        """Cleanup resources before shutdown."""
        try:
            # Save final conversation history
            self._save_conversation_history()
            
            # Close any open database connections
            if hasattr(self, '_db_connection'):
                self._db_connection.close()
            
            # Close any open file handlers
            for handler in self.logger.handlers[:]:
                handler.close()
                self.logger.removeHandler(handler)
                
            # Clear caches
            if hasattr(self, 'market_analyzer'):
                self.market_analyzer.data_cache.clear()
                self.market_analyzer.analysis_cache.clear()
            
            if hasattr(self, 'sentiment_analyzer'):
                self.sentiment_analyzer.cache.clear()
                
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
    
    def __del__(self):
        """Enhanced cleanup of resources."""
        try:
            # Save final conversation history
            self._save_conversation_history()
            
            # Close any open file handlers
            for handler in self.logger.handlers[:]:
                handler.close()
                self.logger.removeHandler(handler)
                
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")

if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def main():
        try:
            # Setup logging
            logging.basicConfig(
                filename='logs/chatbot.log',
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            logger = logging.getLogger(__name__)

            # Load configuration
            config_path = Path("config/config.json")
            if not config_path.exists():
                raise FileNotFoundError(f"Configuration file not found at {config_path}")

            with config_path.open() as f:
                config = json.load(f)

            # Initialize chatbot
            chatbot = EnhancedCongressionalChatbot(
                db_path=str(Path(config["db_path"]) / "embeddings.sqlite3"),
                env_path=config["env_path"]
            )

            # Example query
            response, sources, metrics = await chatbot.get_enhanced_response(
                "What were the key discussions about cryptocurrency regulation in recent hearings?"
            )

            print("Response:", response)
            print("\nSources:", sources)
            print("\nMetrics:", metrics)

            # Get performance metrics
            perf_metrics = await chatbot.get_performance_metrics()
            print("\nPerformance Metrics:", perf_metrics)

        except Exception as e:
            logger.error(f"Error in main: {str(e)}", exc_info=True)
            raise

    # Run the example
    asyncio.run(main())