#!/usr/bin/env python3
"""
Stripe Integration Manager for Terradev
Handles dynamic checkout session creation, webhook processing,
and Enterprise+ metered GPU-hour billing.
"""

import os
import json
import math
try:
    import stripe
except ImportError:
    stripe = None
from typing import Dict, Optional, Any, List
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Enterprise+ metered billing rate
ENTERPRISE_PLUS_GPU_HOUR_RATE_CENTS = 9  # $0.09 per GPU-hour
ENTERPRISE_PLUS_MIN_GPUS = 32


class StripeManager:
    """Manages Stripe integration for Terradev CLI"""
    
    def __init__(self):
        if stripe is None:
            logger.warning("stripe package not installed — billing features unavailable. "
                           "Install with: pip install stripe")
            self.demo_mode = True
            self.publishable_key = ""
            self.secret_key = None
            self._metering_file = Path.home() / '.terradev' / 'gpu_metering.json'
            return

        # Keys loaded from environment — never hardcoded
        self.publishable_key = os.getenv(
            'STRIPE_PUBLISHABLE_KEY',
            'pk_live_51Sz5pwKDFO7eDloBQakbf5HBrurcPPiiiNrk4RREPRT64cBipJC8nmpaXh3sZzUv6redIbaAHh7f4nDEGb4ehQ2m00kvIdxiFP'
        )
        self.secret_key = os.getenv('STRIPE_SECRET_KEY')  # Set this in environment
        
        # Local metering state
        self._metering_file = Path.home() / '.terradev' / 'gpu_metering.json'
        
        if not self.secret_key:
            logging.warning("STRIPE_SECRET_KEY not set - using demo mode")
            self.demo_mode = True
        else:
            stripe.api_key = self.secret_key
            self.demo_mode = False
    
    # ── Flat-rate subscription checkout (Research+ / Enterprise) ──────

    def create_checkout_session(self, tier: str, customer_email: str, success_url: str, cancel_url: str) -> Dict[str, Any]:
        """Create a dynamic Stripe checkout session"""
        
        if self.demo_mode:
            return {
                'session_id': f'cs_demo_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                'checkout_url': f'https://checkout.stripe.com/demo/{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                'publishable_key': self.publishable_key
            }
        
        # Handle Enterprise+ separately — metered billing
        if tier == 'enterprise_plus':
            return self._create_enterprise_plus_checkout(customer_email, success_url, cancel_url)
        
        # Product configuration for flat-rate tiers
        products = {
            'research_plus': {
                'name': 'Terradev Research+',
                'description': '80 provisions/month, 8 servers, inference support',
                'price': 4999,  # $49.99 in cents
                'currency': 'usd'
            },
            'enterprise': {
                'name': 'Terradev Enterprise',
                'description': 'Unlimited provisions, 32 servers, priority support',
                'price': 29999,  # $299.99 in cents
                'currency': 'usd'
            }
        }
        
        if tier not in products:
            raise ValueError(f"Unknown tier: {tier}")
        
        product_config = products[tier]
        
        try:
            product = self._get_or_create_product(tier, product_config)
            price = self._get_or_create_price(product.id, product_config)
            
            session = stripe.checkout.Session.create(
                payment_method_types=['card'],
                line_items=[{
                    'price': price.id,
                    'quantity': 1,
                }],
                mode='subscription',
                success_url=success_url,
                cancel_url=cancel_url,
                customer_email=customer_email,
                metadata={
                    'tier': tier,
                    'product': 'terradev_cli',
                    'version': '3.1.8'
                },
                subscription_data={
                    'metadata': {
                        'tier': tier,
                        'product': 'terradev_cli'
                    }
                }
            )
            
            return {
                'session_id': session.id,
                'checkout_url': session.url,
                'publishable_key': self.publishable_key
            }
            
        except Exception as e:
            logging.error(f"Failed to create Stripe session: {e}")
            raise

    # ── Enterprise+ metered subscription ─────────────────────────────

    def _create_enterprise_plus_checkout(self, customer_email: str, success_url: str, cancel_url: str) -> Dict[str, Any]:
        """Create a Stripe checkout session for Enterprise+ metered billing.
        
        Enterprise+ uses Stripe metered usage billing:
        - $0.09 per GPU-hour, billed monthly
        - Minimum 32 GPUs under management
        - Card on file, invoiced at end of billing period
        """
        try:
            product = self._get_or_create_product('enterprise_plus', {
                'name': 'Terradev Enterprise+',
                'description': 'Metered GPU-hour billing — $0.09/GPU-hr, 32 GPU minimum. '
                               'Unlimited provisions, servers, seats, dedicated support.',
            })
            
            # Create metered price (usage-based, billed per GPU-hour)
            price = self._get_or_create_metered_price(product.id)
            
            session = stripe.checkout.Session.create(
                payment_method_types=['card'],
                line_items=[{
                    'price': price.id,
                }],
                mode='subscription',
                success_url=success_url,
                cancel_url=cancel_url,
                customer_email=customer_email,
                metadata={
                    'tier': 'enterprise_plus',
                    'product': 'terradev_cli',
                    'version': '3.1.8',
                    'billing_model': 'metered',
                    'gpu_hour_rate_cents': str(ENTERPRISE_PLUS_GPU_HOUR_RATE_CENTS),
                    'min_gpus': str(ENTERPRISE_PLUS_MIN_GPUS),
                },
                subscription_data={
                    'metadata': {
                        'tier': 'enterprise_plus',
                        'product': 'terradev_cli',
                        'billing_model': 'metered',
                    }
                }
            )
            
            return {
                'session_id': session.id,
                'checkout_url': session.url,
                'publishable_key': self.publishable_key
            }
            
        except Exception as e:
            logging.error(f"Failed to create Enterprise+ checkout: {e}")
            raise

    def _get_or_create_metered_price(self, product_id: str) -> stripe.Price:
        """Get or create a metered price for Enterprise+ GPU-hour billing."""
        try:
            prices = stripe.Price.list(product=product_id, limit=100, active=True)
            for price in prices.data:
                if (price.recurring and
                    price.recurring.get('usage_type') == 'metered' and
                    price.unit_amount == ENTERPRISE_PLUS_GPU_HOUR_RATE_CENTS and
                    price.currency == 'usd'):
                    return price
            
            # Create metered price — $0.09 per GPU-hour, billed monthly
            return stripe.Price.create(
                product=product_id,
                unit_amount=ENTERPRISE_PLUS_GPU_HOUR_RATE_CENTS,  # 9 cents
                currency='usd',
                recurring={
                    'interval': 'month',
                    'usage_type': 'metered',
                    'aggregate_usage': 'sum',  # Sum all GPU-hours in the period
                },
                metadata={
                    'tier': 'enterprise_plus',
                    'product': 'terradev_cli',
                    'unit': 'gpu_hour',
                }
            )
        except Exception as e:
            logging.error(f"Failed to get/create metered price: {e}")
            raise

    # ── GPU-hour metering ────────────────────────────────────────────

    def report_gpu_hours(self, subscription_item_id: str, gpu_hours: float, 
                         gpu_type: str = '', instance_id: str = '') -> Optional[Dict[str, Any]]:
        """Report GPU-hour usage to Stripe for metered billing.
        
        Called when a provision ends (instance terminated) to report
        the actual GPU-hours consumed.
        
        Args:
            subscription_item_id: The Stripe subscription item ID
            gpu_hours: Number of GPU-hours to bill (rounded up)
            gpu_type: GPU type for metadata
            instance_id: Instance ID for metadata
        """
        if self.demo_mode:
            logger.info(f"[demo] Would report {gpu_hours:.2f} GPU-hours to Stripe")
            return {'demo': True, 'gpu_hours': gpu_hours}
        
        # Stripe metered usage requires integer quantities
        quantity = max(1, math.ceil(gpu_hours))
        
        try:
            usage_record = stripe.SubscriptionItem.create_usage_record(
                subscription_item_id,
                quantity=quantity,
                timestamp=int(datetime.now().timestamp()),
                action='increment',
            )
            
            logger.info(
                f"Reported {quantity} GPU-hours to Stripe "
                f"(sub_item={subscription_item_id}, gpu={gpu_type}, instance={instance_id})"
            )
            
            # Track locally
            self._record_local_metering(gpu_hours, gpu_type, instance_id)
            
            return {
                'usage_record_id': usage_record.id,
                'quantity': quantity,
                'gpu_hours_raw': gpu_hours,
                'cost_usd': round(quantity * ENTERPRISE_PLUS_GPU_HOUR_RATE_CENTS / 100, 2),
            }
            
        except Exception as e:
            logging.error(f"Failed to report GPU-hours to Stripe: {e}")
            # Store locally for retry
            self._record_local_metering(gpu_hours, gpu_type, instance_id, reported=False)
            return None

    def get_subscription_item_id(self, customer_email: str) -> Optional[str]:
        """Look up the Enterprise+ subscription item ID for a customer.
        
        The subscription item ID is needed to report metered usage.
        Cached locally in ~/.terradev/gpu_metering.json after first lookup.
        """
        # Check local cache first
        metering = self._load_metering()
        cached_id = metering.get('subscription_item_id')
        if cached_id:
            return cached_id
        
        if self.demo_mode:
            return 'si_demo_enterprise_plus'
        
        try:
            # Find customer by email
            customers = stripe.Customer.list(email=customer_email, limit=1)
            if not customers.data:
                return None
            
            customer = customers.data[0]
            
            # Find active Enterprise+ subscription
            subscriptions = stripe.Subscription.list(
                customer=customer.id, status='active', limit=10
            )
            for sub in subscriptions.data:
                if sub.metadata.get('tier') == 'enterprise_plus':
                    # Get the metered subscription item
                    for item in sub['items']['data']:
                        if (item.price.recurring and 
                            item.price.recurring.get('usage_type') == 'metered'):
                            # Cache it
                            metering['subscription_item_id'] = item.id
                            metering['subscription_id'] = sub.id
                            metering['customer_id'] = customer.id
                            self._save_metering(metering)
                            return item.id
            
            return None
            
        except Exception as e:
            logging.error(f"Failed to look up subscription item: {e}")
            return None

    def get_current_period_usage(self) -> Dict[str, Any]:
        """Get GPU-hour usage for the current billing period."""
        metering = self._load_metering()
        records = metering.get('records', [])
        
        # Filter to current month
        now = datetime.now()
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        current_records = [
            r for r in records 
            if datetime.fromisoformat(r['timestamp']) >= month_start
        ]
        
        total_hours = sum(r.get('gpu_hours', 0) for r in current_records)
        total_cost = round(math.ceil(total_hours) * ENTERPRISE_PLUS_GPU_HOUR_RATE_CENTS / 100, 2)
        
        return {
            'billing_period_start': month_start.isoformat(),
            'total_gpu_hours': round(total_hours, 2),
            'total_gpu_hours_billed': math.ceil(total_hours),
            'estimated_cost_usd': total_cost,
            'rate_per_gpu_hour': ENTERPRISE_PLUS_GPU_HOUR_RATE_CENTS / 100,
            'record_count': len(current_records),
            'min_gpus': ENTERPRISE_PLUS_MIN_GPUS,
        }

    # ── Local metering persistence ───────────────────────────────────

    def _load_metering(self) -> Dict[str, Any]:
        """Load local metering state."""
        if self._metering_file.exists():
            try:
                with open(self._metering_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {'records': []}

    def _save_metering(self, data: Dict[str, Any]):
        """Save local metering state."""
        self._metering_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self._metering_file, 'w') as f:
            json.dump(data, f, indent=2)

    def _record_local_metering(self, gpu_hours: float, gpu_type: str, 
                                instance_id: str, reported: bool = True):
        """Record a metering event locally for audit trail."""
        metering = self._load_metering()
        metering.setdefault('records', []).append({
            'timestamp': datetime.now().isoformat(),
            'gpu_hours': round(gpu_hours, 4),
            'gpu_type': gpu_type,
            'instance_id': instance_id,
            'reported_to_stripe': reported,
            'cost_usd': round(math.ceil(gpu_hours) * ENTERPRISE_PLUS_GPU_HOUR_RATE_CENTS / 100, 2),
        })
        self._save_metering(metering)
    
    # ── Shared helpers ────────────────────────────────────────────────

    def _get_or_create_product(self, tier: str, config: Dict[str, Any]) -> stripe.Product:
        """Get existing product or create new one"""
        
        try:
            products = stripe.Product.list(limit=100, active=True)
            for product in products.data:
                if product.metadata.get('tier') == tier and product.metadata.get('product') == 'terradev_cli':
                    return product
            
            return stripe.Product.create(
                name=config['name'],
                description=config['description'],
                metadata={
                    'tier': tier,
                    'product': 'terradev_cli',
                    'version': '3.1.8'
                }
            )
            
        except Exception as e:
            logging.error(f"Failed to get/create product: {e}")
            raise
    
    def _get_or_create_price(self, product_id: str, config: Dict[str, Any]) -> stripe.Price:
        """Get existing price or create new one"""
        
        try:
            # Search for existing price
            prices = stripe.Price.list(product=product_id, limit=100, active=True)
            for price in prices.data:
                if price.unit_amount == config['price'] and price.currency == config['currency']:
                    return price
            
            # Create new price
            return stripe.Price.create(
                product=product_id,
                unit_amount=config['price'],
                currency=config['currency'],
                recurring={'interval': 'month'},
                metadata={
                    'product': 'terradev_cli'
                }
            )
            
        except Exception as e:
            logging.error(f"Failed to get/create price: {e}")
            raise
    
    def construct_webhook_event(self, payload: str, sig_header: str, webhook_secret: str) -> stripe.Event:
        """Construct webhook event for verification"""
        
        if self.demo_mode:
            # Demo mode - return mock event
            return {
                'type': 'checkout.session.completed',
                'data': {
                    'object': {
                        'id': f'cs_demo_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                        'customer': 'cus_demo_123',
                        'metadata': {'tier': 'research_plus'},
                        'subscription': 'sub_demo_123'
                    }
                }
            }
        
        try:
            return stripe.Webhook.construct_event(payload, sig_header, webhook_secret)
        except Exception as e:
            logging.error(f"Webhook signature verification failed: {e}")
            raise
    
    def get_customer_info(self, customer_id: str) -> Dict[str, Any]:
        """Get customer information from Stripe"""
        
        if self.demo_mode:
            return {
                'id': customer_id,
                'email': 'demo@example.com',
                'metadata': {'tier': 'research_plus'}
            }
        
        try:
            customer = stripe.Customer.retrieve(customer_id)
            return {
                'id': customer.id,
                'email': customer.email,
                'metadata': customer.metadata
            }
        except Exception as e:
            logging.error(f"Failed to get customer info: {e}")
            raise
    
    def cancel_subscription(self, subscription_id: str) -> bool:
        """Cancel a subscription"""
        
        if self.demo_mode:
            return True
        
        try:
            stripe.Subscription.delete(subscription_id)
            return True
        except Exception as e:
            logging.error(f"Failed to cancel subscription: {e}")
            return False

# Global instance
stripe_manager = StripeManager()
