#!/usr/bin/env python3
"""
Enhanced Price Discovery Engine
Multi-cloud price comparison with capacity and confidence scoring
"""

import asyncio
import aiohttp
import json
import sqlite3
import statistics
try:
    import numpy as np
except ImportError:
    class _NpShim:
        @staticmethod
        def mean(x): return statistics.mean(x)
        @staticmethod
        def std(x): return statistics.pstdev(x)
        @staticmethod
        def min(x): return min(x)
        @staticmethod
        def max(x): return max(x)
    np = _NpShim()
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path

@dataclass
class PriceInfo:
    provider: str
    gpu_type: str
    price: float
    instance_type: str
    region: str
    capacity: str
    confidence: float
    last_updated: datetime
    spot: bool = False

@dataclass
class CapacityInfo:
    provider: str
    gpu_type: str
    region: str
    available: bool
    estimated_wait: int
    last_check: datetime

class PriceDiscoveryEngine:
    """Enhanced price discovery with real-time capacity and confidence scoring"""
    
    def __init__(self):
        self.db_path = Path.home() / '.terradev' / 'price_cache.db'
        self.session: Optional[aiohttp.ClientSession] = None
        self.price_cache = {}
        self.capacity_cache = {}
        self._init_database()
    
    def _init_database(self):
        """Initialize price cache database"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS price_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    provider TEXT NOT NULL,
                    gpu_type TEXT NOT NULL,
                    price REAL NOT NULL,
                    instance_type TEXT NOT NULL,
                    region TEXT NOT NULL,
                    spot BOOLEAN DEFAULT FALSE,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    confidence REAL DEFAULT 0.5
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS capacity_info (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    provider TEXT NOT NULL,
                    gpu_type TEXT NOT NULL,
                    region TEXT NOT NULL,
                    available BOOLEAN NOT NULL,
                    estimated_wait INTEGER DEFAULT 0,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_price_provider_gpu 
                ON price_history(provider, gpu_type, timestamp)
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_capacity_provider_gpu 
                ON capacity_info(provider, gpu_type, timestamp)
            ''')
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_realtime_prices(self, gpu_type: str, region: Optional[str] = None) -> List[PriceInfo]:
        """Get real-time prices with capacity info and confidence scoring"""
        from terradev_cli.cli import TerradevAPI
        
        api = TerradevAPI()
        providers = ['runpod', 'aws', 'gcp', 'azure', 'vastai', 'lambda', 'coreweave', 'tensordock']
        
        prices = []
        
        # Get quotes from all providers
        for provider in providers:
            try:
                provider_prices = await self._get_provider_prices(api, provider, gpu_type, region)
                prices.extend(provider_prices)
            except Exception as e:
                print(f"Error getting prices from {provider}: {e}")
                continue
        
        # Calculate confidence scores
        for price_info in prices:
            price_info.confidence = await self._calculate_confidence(price_info)
        
        # Sort by value (price adjusted for confidence)
        prices.sort(key=lambda x: x.price / x.confidence)
        
        return prices
    
    async def _get_provider_prices(self, api, provider: str, gpu_type: str, region: Optional[str]) -> List[PriceInfo]:
        """Get prices from specific provider"""
        try:
            # Use existing quote functionality
            quotes = await self._get_quotes_from_provider(api, provider, gpu_type, region)
            
            price_infos = []
            for quote in quotes:
                # Check capacity
                capacity = await self._check_capacity(provider, gpu_type, region)
                
                price_info = PriceInfo(
                    provider=provider,
                    gpu_type=gpu_type,
                    price=quote['price_hr'],
                    instance_type=quote.get('instance_type', 'unknown'),
                    region=quote.get('region', region or 'default'),
                    capacity=capacity,
                    confidence=0.5,  # Will be calculated later
                    last_updated=datetime.now(),
                    spot=quote.get('spot', False)
                )
                price_infos.append(price_info)
            
            return price_infos
            
        except Exception as e:
            print(f"Error getting prices from {provider}: {e}")
            return []
    
    async def _get_quotes_from_provider(self, api, provider: str, gpu_type: str, region: Optional[str]) -> List[Dict]:
        """Get quotes from provider using existing functionality"""
        # This would integrate with existing provider quote methods
        # For now, return mock data to demonstrate the structure
        
        mock_quotes = {
            'runpod': [{'price_hr': 1.19, 'instance_type': 'A100', 'spot': True, 'region': 'us-east'}],
            'aws': [{'price_hr': 32.77, 'instance_type': 'p4d.24xlarge', 'spot': False, 'region': 'us-east-1'}],
            'gcp': [{'price_hr': 3.67, 'instance_type': 'a2-highgpu-1g', 'spot': False, 'region': 'us-central1'}],
            'azure': [{'price_hr': 4.50, 'instance_type': 'Standard_NC96ads_A100_v4', 'spot': False, 'region': 'eastus'}],
            'vastai': [{'price_hr': 2.50, 'instance_type': 'A100', 'spot': True, 'region': 'us-east'}],
            'lambda': [{'price_hr': 2.20, 'instance_type': 'A100', 'spot': True, 'region': 'us-east'}],
            'coreweave': [{'price_hr': 2.80, 'instance_type': 'A100', 'spot': False, 'region': 'LGA1'}],
            'tensordock': [{'price_hr': 1.80, 'instance_type': 'A100', 'spot': True, 'region': 'us-east'}]
        }
        
        return mock_quotes.get(provider, [])
    
    async def _check_capacity(self, provider: str, gpu_type: str, region: Optional[str]) -> str:
        """Check capacity availability for provider"""
        # Check cache first
        cache_key = f"{provider}_{gpu_type}_{region or 'default'}"
        
        if cache_key in self.capacity_cache:
            last_check = self.capacity_cache[cache_key]['last_check']
            if datetime.now() - last_check < timedelta(minutes=5):
                return self.capacity_cache[cache_key]['status']
        
        # Simulate capacity check
        # In real implementation, this would call provider APIs
        import random
        
        available = random.choice([True, True, True, False])  # 75% availability
        status = "Available" if available else "Limited"
        
        # Cache result
        self.capacity_cache[cache_key] = {
            'status': status,
            'last_check': datetime.now()
        }
        
        # Store in database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO capacity_info (provider, gpu_type, region, available, estimated_wait)
                VALUES (?, ?, ?, ?, ?)
            ''', (provider, gpu_type, region or 'default', available, 0))
        
        return status
    
    async def _calculate_confidence(self, price_info: PriceInfo) -> float:
        """Calculate confidence score based on historical accuracy"""
        with sqlite3.connect(self.db_path) as conn:
            # Get recent price history for this provider/gpu
            cursor = conn.execute('''
                SELECT price, timestamp FROM price_history 
                WHERE provider = ? AND gpu_type = ? 
                ORDER BY timestamp DESC 
                LIMIT 24
            ''', (price_info.provider, price_info.gpu_type))
            
            history = cursor.fetchall()
            
            if not history:
                return 0.5  # Default confidence for new data
            
            # Calculate volatility
            prices = [row[0] for row in history]
            if len(prices) < 2:
                return 0.6
            
            volatility = np.std(prices)
            avg_price = np.mean(prices)
            
            # Lower volatility = higher confidence
            confidence = max(0.1, 1.0 - (volatility / avg_price))
            
            # Factor in recency of data
            latest_timestamp = datetime.fromisoformat(history[0][1])
            hours_old = (datetime.now() - latest_timestamp).total_seconds() / 3600
            
            # Decay confidence based on age
            age_factor = max(0.5, 1.0 - (hours_old / 24))
            
            return confidence * age_factor
    
    async def get_price_trends(self, gpu_type: str, hours: int = 24) -> Dict[str, Any]:
        """Get price trends for analysis"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT provider, price, timestamp FROM price_history 
                WHERE gpu_type = ? AND timestamp > datetime('now', '-{} hours')
                ORDER BY timestamp
            '''.format(hours), (gpu_type,))
            
            data = cursor.fetchall()
        
        trends = {}
        for provider, price, timestamp in data:
            if provider not in trends:
                trends[provider] = []
            trends[provider].append({
                'price': price,
                'timestamp': timestamp
            })
        
        # Calculate trend metrics
        for provider in trends:
            prices = [p['price'] for p in trends[provider]]
            trends[provider]['metrics'] = {
                'avg_price': np.mean(prices),
                'min_price': np.min(prices),
                'max_price': np.max(prices),
                'volatility': np.std(prices),
                'trend': 'up' if prices[-1] > prices[0] else 'down'
            }
        
        return trends
    
    def store_price_data(self, price_infos: List[PriceInfo]):
        """Store price data in database"""
        with sqlite3.connect(self.db_path) as conn:
            for price_info in price_infos:
                conn.execute('''
                    INSERT INTO price_history 
                    (provider, gpu_type, price, instance_type, region, spot, confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    price_info.provider,
                    price_info.gpu_type,
                    price_info.price,
                    price_info.instance_type,
                    price_info.region,
                    price_info.spot,
                    price_info.confidence
                ))

class BudgetOptimizationEngine:
    """Budget-first optimization engine"""
    
    def __init__(self):
        self.price_engine = PriceDiscoveryEngine()
        self.cost_predictor = CostPredictorML()
        self.risk_assessor = SpotRiskAssessor()
    
    async def optimize_for_budget(self, requirements: Dict[str, Any], budget: float, constraints: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Find optimal deployment under budget constraints"""
        constraints = constraints or {}
        
        # Get all available options
        price_infos = await self.price_engine.get_realtime_prices(
            requirements['gpu_type'], 
            requirements.get('region')
        )
        
        optimized_options = []
        
        for price_info in price_infos:
            # Predict actual cost
            predicted_cost = await self.cost_predictor.predict(requirements, price_info)
            
            # Assess spot risk
            risk_score = await self.risk_assessor.assess(price_info)
            
            # Calculate total cost including risk
            risk_adjusted_cost = predicted_cost * (1 + risk_score)
            
            if risk_adjusted_cost <= budget:
                optimized_options.append({
                    'provider': price_info.provider,
                    'instance_type': price_info.instance_type,
                    'price': price_info.price,
                    'predicted_cost': predicted_cost,
                    'risk_score': risk_score,
                    'risk_adjusted_cost': risk_adjusted_cost,
                    'confidence': price_info.confidence,
                    'capacity': price_info.capacity,
                    'spot': price_info.spot,
                    'budget_utilization': (risk_adjusted_cost / budget) * 100
                })
        
        # Sort by budget utilization (prefer closer to budget for better value)
        optimized_options.sort(key=lambda x: abs(x['budget_utilization'] - 80))
        
        return optimized_options

class CostPredictorML:
    """ML-based cost prediction"""
    
    def __init__(self):
        self.models = {}
    
    async def predict(self, requirements: Dict[str, Any], price_info: PriceInfo) -> float:
        """Predict actual deployment cost"""
        # Simple prediction based on requirements and price
        base_cost = price_info.price
        
        # Adjust for GPU count
        gpu_count = requirements.get('gpu_count', 1)
        gpu_cost = base_cost * gpu_count
        
        # Adjust for estimated duration
        estimated_hours = requirements.get('estimated_hours', 1.0)
        total_cost = gpu_cost * estimated_hours
        
        # Add buffer for unknown factors
        buffer_factor = 1.1 if price_info.spot else 1.05
        
        return total_cost * buffer_factor

class SpotRiskAssessor:
    """Spot instance risk assessment"""
    
    def __init__(self):
        self.risk_factors = {
            'runpod': 0.1,  # Low risk
            'aws': 0.3,     # Medium risk
            'gcp': 0.2,     # Low-medium risk
            'azure': 0.25,   # Medium risk
            'vastai': 0.15,  # Low risk
            'lambda': 0.2,   # Low-medium risk
            'coreweave': 0.1, # Low risk
            'tensordock': 0.2 # Low-medium risk
        }
    
    async def assess(self, price_info: PriceInfo) -> float:
        """Assess spot instance risk"""
        if not price_info.spot:
            return 0.0  # No risk for on-demand
        
        base_risk = self.risk_factors.get(price_info.provider, 0.2)
        
        # Adjust based on capacity
        if price_info.capacity == "Limited":
            base_risk *= 1.5
        
        # Adjust based on confidence
        confidence_factor = 1.0 - price_info.confidence
        base_risk *= (1 + confidence_factor)
        
        return min(base_risk, 0.5)  # Cap at 50% risk
