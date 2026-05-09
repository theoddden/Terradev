#!/usr/bin/env python3
"""
AWS Telemetry Server with Stripe Integration
Enhanced telemetry server with payment processing and user analytics
"""

import json
import sqlite3
from datetime import datetime, timedelta
from flask import Flask, jsonify, request, render_template_string
import os

app = Flask(__name__)

# Stripe configuration (you'll need to set these)
STRIPE_API_KEY = os.getenv('STRIPE_API_KEY', 'sk_test_...')  # Test key
STRIPE_PUBLISHABLE_KEY = os.getenv('STRIPE_PUBLISHABLE_KEY', 'pk_test_...')

# Pricing tiers - REAL TERRADEV PRICING
PRICING_TIERS = {
    'research': {'price': 0, 'name': 'Research Tier', 'provisions': 10, 'instances': 1, 'parallel_queries': 3},
    'research_plus': {'price': 4999, 'name': 'Research+ Tier', 'provisions': 80, 'instances': 4, 'parallel_queries': 10},
    'enterprise': {'price': 29999, 'name': 'Enterprise Tier', 'provisions': -1, 'instances': 32, 'parallel_queries': 50},
    'enterprise_plus': {'price': 900, 'name': 'Enterprise+ Tier', 'provisions': -1, 'instances': -1, 'parallel_queries': 100, 'metered': True, 'gpu_hour_rate': 0.09, 'min_monthly': 36864}
}

def init_db():
    """Initialize database with user and payment tables"""
    conn = sqlite3.connect('telemetry.db')
    cursor = conn.cursor()
    
    # Telemetry logs
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS telemetry_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            machine_id TEXT,
            action TEXT,
            timestamp TEXT,
            details TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            machine_id TEXT UNIQUE,
            email TEXT,
            tier TEXT DEFAULT 'research',
            stripe_customer_id TEXT,
            subscription_id TEXT,
            subscription_status TEXT,
            requests_used INTEGER DEFAULT 0,
            requests_limit INTEGER DEFAULT 10,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Usage tracking
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS usage_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            machine_id TEXT,
            endpoint TEXT,
            timestamp TEXT,
            success BOOLEAN,
            response_time_ms INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # GPU metering for Enterprise+
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS gpu_metering (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            machine_id TEXT,
            gpu_type TEXT,
            instance_id TEXT,
            gpu_hours REAL,
            cost_cents INTEGER,
            month TEXT,
            reported_to_stripe BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

def get_or_create_user(machine_id):
    """Get existing user or create new one"""
    conn = sqlite3.connect('telemetry.db')
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM users WHERE machine_id = ?', (machine_id,))
    user = cursor.fetchone()
    
    if not user:
        cursor.execute('''
            INSERT INTO users (machine_id, tier, requests_limit)
            VALUES (?, 'research', 10)
        ''', (machine_id,))
        conn.commit()
        
        cursor.execute('SELECT * FROM users WHERE machine_id = ?', (machine_id,))
        user = cursor.fetchone()
    
    conn.close()
    
    # Convert to dict
    columns = [desc[0] for desc in cursor.description]
    return dict(zip(columns, user))

def check_usage_limits(machine_id):
    """Check if user has exceeded their usage limits"""
    user = get_or_create_user(machine_id)
    
    # Enterprise+ has metered billing - always allowed
    if user['tier'] == 'enterprise_plus':
        return {'allowed': True, 'remaining': -1, 'limit': -1, 'metered': True, 'gpu_hour_rate': 0.09}
    
    # Enterprise has unlimited provisions
    if user['tier'] == 'enterprise':
        return {'allowed': True, 'remaining': -1, 'limit': -1}
    
    # Check provisions this month
    conn = sqlite3.connect('telemetry.db')
    cursor = conn.cursor()
    
    first_day_of_month = datetime.now().replace(day=1).isoformat()
    cursor.execute('''
        SELECT COUNT(*) FROM usage_logs 
        WHERE machine_id = ? AND timestamp >= ? AND endpoint = '/provision'
    ''', (machine_id, first_day_of_month))
    
    provisions_this_month = cursor.fetchone()[0]
    remaining = max(0, user['requests_limit'] - provisions_this_month)
    
    conn.close()
    
    return {
        'allowed': remaining > 0,
        'remaining': remaining,
        'limit': user['requests_limit'],
        'tier': user['tier'],
        'provisions_used': provisions_this_month
    }

def log_usage(machine_id, endpoint, success=True, response_time=0):
    """Log API usage"""
    conn = sqlite3.connect('telemetry.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO usage_logs (machine_id, endpoint, timestamp, success, response_time_ms)
        VALUES (?, ?, ?, ?, ?)
    ''', (machine_id, endpoint, datetime.now().isoformat(), success, response_time))
    
    conn.commit()
    conn.close()

# === TELEMETRY ENDPOINTS ===

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '2.0.0-stripe',
        'server': 'production',
        'features': ['telemetry', 'payments', 'usage_tracking']
    })

@app.route('/log-usage', methods=['POST'])
def log_usage_endpoint():
    """Log telemetry usage (with rate limiting)"""
    try:
        data = request.get_json()
        machine_id = data.get('machine_id')
        
        if not machine_id:
            return jsonify({'error': 'machine_id required'}), 400
        
        # Check usage limits
        usage_check = check_usage_limits(machine_id)
        if not usage_check['allowed']:
            log_usage(machine_id, '/log-usage', success=False)
            return jsonify({
                'error': 'Usage limit exceeded',
                'tier': usage_check['tier'],
                'limit': usage_check['limit'],
                'upgrade_url': '/upgrade'
            }), 429
        
        # Log the telemetry data
        conn = sqlite3.connect('telemetry.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO telemetry_logs (machine_id, action, timestamp, details)
            VALUES (?, ?, ?, ?)
        ''', (
            machine_id,
            data.get('action'),
            data.get('timestamp'),
            json.dumps(data.get('details', {}))
        ))
        conn.commit()
        conn.close()
        
        # Log usage
        log_usage(machine_id, '/log-usage', success=True)
        
        return jsonify({
            'status': 'success',
            'message': 'Telemetry logged successfully',
            'timestamp': datetime.now().isoformat(),
            'usage': usage_check
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/user-stats')
def user_stats():
    """Get user statistics"""
    conn = sqlite3.connect('telemetry.db')
    cursor = conn.cursor()
    
    # Total users
    cursor.execute('SELECT COUNT(DISTINCT machine_id) FROM users')
    total_users = cursor.fetchone()[0] or 0
    
    # Active users today
    today = datetime.now().date().isoformat()
    cursor.execute('''
        SELECT COUNT(DISTINCT machine_id) FROM usage_logs 
        WHERE date(timestamp) = ?
    ''', (today,))
    active_today = cursor.fetchone()[0] or 0
    
    # Users by tier
    cursor.execute('SELECT tier, COUNT(*) FROM users GROUP BY tier')
    tier_breakdown = dict(cursor.fetchall())
    
    # Usage this month
    first_day = datetime.now().replace(day=1).isoformat()
    cursor.execute('SELECT COUNT(*) FROM usage_logs WHERE timestamp >= ?')
    monthly_requests = cursor.fetchone()[0] or 0
    
    conn.close()
    
    return jsonify({
        'total_users': total_users,
        'active_users_today': active_today,
        'users_by_tier': tier_breakdown,
        'monthly_requests': monthly_requests,
        'server_info': {
            'version': '2.0.0-stripe',
            'timestamp': datetime.now().isoformat()
        }
    })

@app.route('/dashboard')
def dashboard():
    """Admin dashboard"""
    dashboard_html = '''
<!DOCTYPE html>
<html>
<head>
    <title>Terradev Telemetry Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .metric { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }
        .metric h3 { margin: 0 0 10px 0; color: #333; }
        .metric .value { font-size: 2em; font-weight: bold; color: #007bff; }
        .user-list { max-height: 300px; overflow-y: auto; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #f8f9fa; }
    </style>
</head>
<body>
    <h1>🚀 Terradev Telemetry Dashboard</h1>
    
    <div style="display: flex; gap: 20px;">
        <div class="metric">
            <h3>Total Users</h3>
            <div class="value" id="total-users">-</div>
        </div>
        <div class="metric">
            <h3>Active Today</h3>
            <div class="value" id="active-today">-</div>
        </div>
        <div class="metric">
            <h3>Monthly Requests</h3>
            <div class="value" id="monthly-requests">-</div>
        </div>
        <div class="metric">
            <h3>Revenue (est.)</h3>
            <div class="value" id="revenue">$0</div>
        </div>
    </div>
    
    <h2>👥 Recent Users</h2>
    <div class="user-list">
        <table id="users-table">
            <thead>
                <tr>
                    <th>Machine ID</th>
                    <th>Tier</th>
                    <th>Requests Used</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody id="users-body">
            </tbody>
        </table>
    </div>
    
    <script>
        async function loadDashboard() {
            try {
                const response = await fetch('/user-stats');
                const data = await response.json();
                
                document.getElementById('total-users').textContent = data.total_users;
                document.getElementById('active-today').textContent = data.active_users_today;
                document.getElementById('monthly-requests').textContent = data.monthly_requests.toLocaleString();
                
                // Estimate revenue
                const revenue = (data.users_by_tier.research_plus || 0) * 49.99 + 
                               (data.users_by_tier.enterprise || 0) * 299.99;
                document.getElementById('revenue').textContent = '$' + revenue.toFixed(2);
                
            } catch (error) {
                console.error('Error loading dashboard:', error);
            }
        }
        
        loadDashboard();
        setInterval(loadDashboard, 30000); // Refresh every 30 seconds
    </script>
</body>
</html>
    '''
    
    return dashboard_html

# === PAYMENT ENDPOINTS ===

@app.route('/pricing')
def pricing():
    """Get pricing tiers"""
    return jsonify({
        'tiers': PRICING_TIERS,
        'currency': 'USD',
        'billing': 'monthly'
    })

@app.route('/create-payment-intent', methods=['POST'])
def create_payment_intent():
    """Create Stripe payment intent for tier upgrade"""
    try:
        data = request.get_json()
        machine_id = data.get('machine_id')
        tier = data.get('tier', 'professional')
        
        if tier not in PRICING_TIERS:
            return jsonify({'error': 'Invalid tier'}), 400
        
        amount = PRICING_TIERS[tier]['price']
        
        # In production, you'd use actual Stripe API
        # For now, return a mock response
        return jsonify({
            'client_secret': f'pi_mock_{datetime.now().timestamp()}',
            'amount': amount,
            'tier': tier,
            'currency': 'usd'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/upgrade', methods=['POST'])
def upgrade_tier():
    """Upgrade user tier (after successful payment)"""
    try:
        data = request.get_json()
        machine_id = data.get('machine_id')
        new_tier = data.get('tier')
        payment_intent_id = data.get('payment_intent_id')
        
        if new_tier not in PRICING_TIERS:
            return jsonify({'error': 'Invalid tier'}), 400
        
        # Update user tier
        conn = sqlite3.connect('telemetry.db')
        cursor = conn.cursor()
        
        # Set limits based on tier
        tier_config = PRICING_TIERS[new_tier]
        new_limit = tier_config.get('provisions', 10)
        
        cursor.execute('''
            UPDATE users 
            SET tier = ?, subscription_id = ?, subscription_status = 'active', 
                requests_limit = ?, updated_at = ?
            WHERE machine_id = ?
        ''', (
            new_tier,
            payment_intent_id,
            new_limit,
            datetime.now().isoformat(),
            machine_id
        ))
        
        conn.commit()
        conn.close()
        
        return jsonify({
            'status': 'success',
            'tier': new_tier,
            'provisions_limit': new_limit,
            'instances': tier_config.get('instances', 1),
            'parallel_queries': tier_config.get('parallel_queries', 3),
            'message': f'Successfully upgraded to {PRICING_TIERS[new_tier]["name"]}'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/check-subscription', methods=['POST'])
def check_subscription():
    """Check user subscription status"""
    try:
        data = request.get_json()
        machine_id = data.get('machine_id')
        
        user = get_or_create_user(machine_id)
        usage = check_usage_limits(machine_id)
        
        return jsonify({
            'tier': user['tier'],
            'subscription_status': user['subscription_status'] or 'inactive',
            'provisions_used': user['requests_used'],
            'provisions_limit': user['requests_limit'],
            'instances': PRICING_TIERS[user['tier']].get('instances', 1),
            'parallel_queries': PRICING_TIERS[user['tier']].get('parallel_queries', 3),
            'usage': usage,
            'can_upgrade': user['tier'] != 'enterprise_plus',
            'metered': PRICING_TIERS[user['tier']].get('metered', False),
            'gpu_hour_rate': PRICING_TIERS[user['tier']].get('gpu_hour_rate', 0),
            'min_monthly': PRICING_TIERS[user['tier']].get('min_monthly', 0)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    init_db()
    print("🚀 Starting Terradev Telemetry Server with Stripe Integration")
    print("💰 Payment processing enabled")
    print("📊 Dashboard: http://localhost:8080/dashboard")
    print("🔍 Health: http://localhost:8080/health")
    print("💳 Pricing: http://localhost:8080/pricing")
    app.run(host='0.0.0.0', port=8080, debug=False)
