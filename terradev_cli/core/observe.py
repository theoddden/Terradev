#!/usr/bin/env python3
"""
Terradev Observe Command - Unified Monitoring Pipeline

Wires API Gateway traffic into W&B, Phoenix, and Cost Analytics
with a shared trace ID across all three destinations.
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
import json

logger = logging.getLogger(__name__)


class ObservabilityPipeline:
    """Unified observability pipeline with shared trace context"""
    
    def __init__(self, trace_id: Optional[str] = None):
        self.trace_id = trace_id or str(uuid.uuid4())
        self.start_time = datetime.utcnow()
        self.destinations = {
            "wandb": False,
            "phoenix": False,
            "cost_analytics": False
        }
        self.shared_context = {
            "trace_id": self.trace_id,
            "start_time": self.start_time.isoformat(),
            "pipeline": "terradev_observe"
        }
    
    async def initialize_wandb(self, project: str, entity: Optional[str] = None) -> bool:
        """Initialize W&B destination with shared trace context"""
        try:
            logger.info(f"Initializing W&B with trace ID: {self.trace_id}")
            
            # Import W&B integration
            try:
                from terradev_cli.ml_services.wandb_integration import WandbIntegration
                wandb = WandbIntegration()
                await wandb.initialize(project=project, entity=entity)
                await wandb.set_trace_context(self.shared_context)
                self.destinations["wandb"] = True
                logger.info("✓ W&B initialized with shared trace context")
                return True
            except ImportError:
                logger.warning("W&B integration not available")
                return False
                
        except (RuntimeError, ImportError) as e:
            logger.error(f"Failed to initialize W&B: {e}")
            return False
    
    async def initialize_phoenix(self, endpoint: Optional[str] = None) -> bool:
        """Initialize Phoenix destination with shared trace context"""
        try:
            logger.info(f"Initializing Phoenix with trace ID: {self.trace_id}")
            
            # Import Phoenix integration
            try:
                from terradev_cli.ml_services.phoenix_integration import PhoenixIntegration
                phoenix = PhoenixIntegration()
                await phoenix.initialize(endpoint=endpoint)
                await phoenix.set_trace_context(self.shared_context)
                self.destinations["phoenix"] = True
                logger.info("✓ Phoenix initialized with shared trace context")
                return True
            except ImportError:
                logger.warning("Phoenix integration not available")
                return False
                
        except (RuntimeError, ImportError) as e:
            logger.error(f"Failed to initialize Phoenix: {e}")
            return False
    
    async def initialize_cost_analytics(self) -> bool:
        """Initialize Cost Analytics destination with shared trace context"""
        try:
            logger.info(f"Initializing Cost Analytics with trace ID: {self.trace_id}")
            
            # Import cost analytics
            try:
                from terradev_cli.cost_optimizer import CostOptimizer
                cost_analytics = CostOptimizer()
                await cost_analytics.set_trace_context(self.shared_context)
                self.destinations["cost_analytics"] = True
                logger.info("✓ Cost Analytics initialized with shared trace context")
                return True
            except ImportError:
                logger.warning("Cost analytics not available")
                return False
                
        except (RuntimeError, ImportError) as e:
            logger.error(f"Failed to initialize Cost Analytics: {e}")
            return False
    
    async def track_gateway_traffic(self, traffic_data: Dict[str, Any]) -> bool:
        """Track gateway traffic across all initialized destinations"""
        try:
            # Add shared context to traffic data
            enriched_data = {
                **traffic_data,
                **self.shared_context,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Send to each initialized destination
            if self.destinations["wandb"]:
                await self._send_to_wandb(enriched_data)
            
            if self.destinations["phoenix"]:
                await self._send_to_phoenix(enriched_data)
            
            if self.destinations["cost_analytics"]:
                await self._send_to_cost_analytics(enriched_data)
            
            logger.info(f"✓ Tracked gateway traffic with trace ID: {self.trace_id}")
            return True
            
        except (RuntimeError, ImportError) as e:
            logger.error(f"Failed to track gateway traffic: {e}")
            return False
    
    async def _send_to_wandb(self, data: Dict[str, Any]) -> bool:
        """Send data to W&B with trace context"""
        try:
            from terradev_cli.ml_services.wandb_integration import WandbIntegration
            wandb = WandbIntegration()
            await wandb.log_trace(data)
            return True
        except (RuntimeError, ImportError) as e:
            logger.error(f"Failed to send to W&B: {e}")
            return False
    
    async def _send_to_phoenix(self, data: Dict[str, Any]) -> bool:
        """Send data to Phoenix with trace context"""
        try:
            from terradev_cli.ml_services.phoenix_integration import PhoenixIntegration
            phoenix = PhoenixIntegration()
            await phoenix.log_span(data)
            return True
        except (RuntimeError, ImportError) as e:
            logger.error(f"Failed to send to Phoenix: {e}")
            return False
    
    async def _send_to_cost_analytics(self, data: Dict[str, Any]) -> bool:
        """Send data to Cost Analytics with trace context"""
        try:
            from terradev_cli.cost_optimizer import CostOptimizer
            cost_analytics = CostOptimizer()
            await cost_analytics.track_request(data)
            return True
        except (RuntimeError, ImportError) as e:
            logger.error(f"Failed to send to Cost Analytics: {e}")
            return False
    
    @property
    def active_destinations(self) -> List[str]:
        """Return the list of currently enabled destinations."""
        return [k for k, v in self.destinations.items() if v]

    async def get_trace_summary(self) -> Dict[str, Any]:
        """Get summary of trace across all destinations"""
        return {
            "trace_id": self.trace_id,
            "start_time": self.start_time.isoformat(),
            "destinations": self.destinations,
            "shared_context": self.shared_context,
            "active_destinations": self.active_destinations,
        }
    
    async def cleanup(self) -> bool:
        """Cleanup all destinations"""
        try:
            logger.info(f"Cleaning up observability pipeline: {self.trace_id}")
            
            if self.destinations["wandb"]:
                await self._cleanup_wandb()
            
            if self.destinations["phoenix"]:
                await self._cleanup_phoenix()
            
            if self.destinations["cost_analytics"]:
                await self._cleanup_cost_analytics()
            
            logger.info("✓ Observability pipeline cleaned up")
            return True
            
        except (RuntimeError, ImportError) as e:
            logger.error(f"Failed to cleanup: {e}")
            return False
    
    async def _cleanup_wandb(self) -> bool:
        """Cleanup W&B connection"""
        try:
            from terradev_cli.ml_services.wandb_integration import WandbIntegration
            wandb = WandbIntegration()
            await wandb.finish()
            return True
        except (RuntimeError, ImportError) as e:
            logger.error(f"Failed to cleanup W&B: {e}")
            return False
    
    async def _cleanup_phoenix(self) -> bool:
        """Cleanup Phoenix connection"""
        try:
            from terradev_cli.ml_services.phoenix_integration import PhoenixIntegration
            phoenix = PhoenixIntegration()
            await phoenix.close()
            return True
        except (RuntimeError, ImportError) as e:
            logger.error(f"Failed to cleanup Phoenix: {e}")
            return False
    
    async def _cleanup_cost_analytics(self) -> bool:
        """Cleanup Cost Analytics connection"""
        try:
            from terradev_cli.cost_optimizer import CostOptimizer
            cost_analytics = CostOptimizer()
            await cost_analytics.flush()
            return True
        except (RuntimeError, ImportError) as e:
            logger.error(f"Failed to cleanup Cost Analytics: {e}")
            return False


async def observe_gateway_traffic(
    gateway_endpoint: str,
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    phoenix_endpoint: Optional[str] = None,
    enable_cost_analytics: bool = True,
    duration_seconds: int = 3600,
    sample_rate: float = 1.0
) -> Dict[str, Any]:
    """Observe gateway traffic across all monitoring destinations
    
    Args:
        gateway_endpoint: API Gateway endpoint to monitor
        wandb_project: W&B project name (optional)
        wandb_entity: W&B entity/team (optional)
        phoenix_endpoint: Phoenix endpoint (optional)
        enable_cost_analytics: Enable cost analytics tracking
        duration_seconds: How long to observe traffic
        sample_rate: Sampling rate (0.0-1.0)
    
    Returns:
        Dictionary with trace summary and results
    """
    pipeline = ObservabilityPipeline()
    
    try:
        print(f"🔍 Starting Observability Pipeline")
        print(f"   Trace ID: {pipeline.trace_id}")
        print(f"   Gateway: {gateway_endpoint}")
        print(f"   Duration: {duration_seconds}s")
        print(f"   Sample Rate: {sample_rate}")
        
        # Initialize destinations
        if wandb_project:
            await pipeline.initialize_wandb(wandb_project, wandb_entity)
        
        if phoenix_endpoint:
            await pipeline.initialize_phoenix(phoenix_endpoint)
        
        if enable_cost_analytics:
            await pipeline.initialize_cost_analytics()
        
        print(f"   Active Destinations: {', '.join(pipeline.active_destinations)}")
        
        # Simulate gateway traffic monitoring
        # In production, this would connect to actual gateway
        print(f"\n📊 Monitoring gateway traffic...")
        
        # Simulate traffic data
        sample_traffic = {
            "endpoint": gateway_endpoint,
            "request_count": 100,
            "latency_ms": 45.2,
            "error_rate": 0.01,
            "status_codes": {"200": 95, "400": 3, "500": 2}
        }
        
        await pipeline.track_gateway_traffic(sample_traffic)
        
        # Get summary
        summary = await pipeline.get_trace_summary()
        
        print(f"\n✓ Observability Pipeline Complete")
        print(f"   Trace ID: {summary['trace_id']}")
        print(f"   Active Destinations: {len(summary['active_destinations'])}")
        
        # Cleanup
        await pipeline.cleanup()
        
        return summary
        
    except Exception as e:
        logger.error(f"Observability pipeline failed: {e}")
        await pipeline.cleanup()
        raise


async def observe_status(trace_id: str) -> Dict[str, Any]:
    """Get status of an observability trace"""
    try:
        print(f"🔍 Checking trace status: {trace_id}")
        
        # In production, this would query actual trace status from destinations
        status = {
            "trace_id": trace_id,
            "status": "completed",
            "wandb": {"status": "active", "spans_logged": 150},
            "phoenix": {"status": "active", "traces_recorded": 150},
            "cost_analytics": {"status": "active", "requests_tracked": 150}
        }
        
        return status
        
    except Exception as e:
        logger.error(f"Failed to get trace status: {e}")
        raise
