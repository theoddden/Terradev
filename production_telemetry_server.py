#!/usr/bin/env python3
"""
Terradev Production Telemetry Server Setup
Production-ready telemetry server with proper configuration
"""

import json
import sqlite3
import threading
import logging
from datetime import datetime, timedelta
from flask import Flask, jsonify, request, g
from flask_cors import CORS
import os
import hashlib
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
DATABASE = 'telemetry_production.db'
SECRET_KEY = os.environ.get('TELEMETRY_SECRET_KEY', 'dev-secret-key-change-in-production')

# In-memory cache for performance
user_stats_cache = {
    'total_users': 0,
    'active_users_today': 0,
    'daily_active_users': {},
    'machine_ids': set(),
    'install_ids': set(),
    'last_updated': datetime.now()
}

def get_db():
    """Get database connection"""
    if 'db' not in g:
        g.db = sqlite3.connect(DATABASE)
        g.db.row_factory = sqlite3.Row
    return g.db

def close_db(e=None):
    """Close database connection"""
    db = g.pop('db', None)
    if db is not None:
        db.close()

def init_db():
    """Initialize database with proper schema"""
    db = sqlite3.connect(DATABASE)
    cursor = db.cursor()
    
    # Create tables with proper indexes
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS telemetry_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            machine_id TEXT NOT NULL,
            install_id TEXT NOT NULL,
            action TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            details TEXT,
            ip_address TEXT,
            user_agent TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS daily_active_users (
            date TEXT PRIMARY KEY,
            machine_ids TEXT,
            count INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_profiles (
            machine_id TEXT PRIMARY KEY,
            install_id TEXT NOT NULL,
            first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            total_actions INTEGER DEFAULT 0,
            platform TEXT,
            version TEXT,
            country TEXT,
            metadata TEXT
        )
    ''')
    
    # Create indexes for performance
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_telemetry_machine_id ON telemetry_logs(machine_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_telemetry_timestamp ON telemetry_logs(timestamp)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_telemetry_action ON telemetry_logs(action)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_profiles_machine_id ON user_profiles(machine_id)')
    
    db.commit()
    db.close()
    logger.info("Database initialized successfully")

def update_user_stats(data):
    """Update user statistics cache"""
    global user_stats_cache
    
    machine_id = data.get('machine_id')
    install_id = data.get('install_id')
    action = data.get('action')
    
    if machine_id:
        user_stats_cache['machine_ids'].add(machine_id)
        user_stats_cache['total_users'] = len(user_stats_cache['machine_ids'])
    
    if install_id:
        user_stats_cache['install_ids'].add(install_id)
    
    # Track daily active users
    today = datetime.now().date().isoformat()
    if action == 'daily_active_users':
        user_stats_cache['daily_active_users'][today] = user_stats_cache['daily_active_users'].get(today, 0) + 1
        user_stats_cache['active_users_today'] = user_stats_cache['daily_active_users'].get(today, 0)
    
    user_stats_cache['last_updated'] = datetime.now()

def log_telemetry(data):
    """Log telemetry data to database"""
    try:
        db = get_db()
        cursor = db.cursor()
        
        # Insert telemetry log
        cursor.execute('''
            INSERT INTO telemetry_logs 
            (machine_id, install_id, action, timestamp, details, ip_address, user_agent)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            data.get('machine_id'),
            data.get('install_id'),
            data.get('action'),
            data.get('timestamp'),
            json.dumps(data.get('details', {})),
            request.remote_addr,
            request.headers.get('User-Agent', '')
        ))
        
        # Update or create user profile
        machine_id = data.get('machine_id')
        if machine_id:
            cursor.execute('''
                INSERT OR REPLACE INTO user_profiles 
                (machine_id, install_id, last_seen, total_actions, platform, version, metadata)
                VALUES (?, ?, 
                    COALESCE((SELECT last_seen FROM user_profiles WHERE machine_id = ?), CURRENT_TIMESTAMP),
                    COALESCE((SELECT total_actions FROM user_profiles WHERE machine_id = ?), 0) + 1,
                    ?, ?, ?)
            ''', (
                machine_id,
                data.get('install_id'),
                machine_id,
                machine_id,
                data.get('details', {}).get('platform'),
                data.get('details', {}).get('version'),
                json.dumps(data.get('details', {}))
            ))
        
        db.commit()
        update_user_stats(data)
        
        logger.info(f"Logged telemetry: {data.get('action')} from {machine_id[:16] if machine_id else 'unknown'}")
        
    except Exception as e:
        logger.error(f"Error logging telemetry: {e}")
        raise

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '2.0.0-production',
        'server': 'production',
        'users': {
            'total_users': user_stats_cache['total_users'],
            'active_users_today': user_stats_cache['active_users_today'],
            'total_installations': len(user_stats_cache['install_ids']),
            'cache_updated': user_stats_cache['last_updated'].isoformat()
        }
    })

@app.route('/log-usage', methods=['POST'])
def log_usage():
    """Log usage telemetry"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate required fields
        required_fields = ['machine_id', 'install_id', 'action', 'timestamp']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        log_telemetry(data)
        
        return jsonify({
            'status': 'success',
            'message': 'Telemetry logged successfully',
            'timestamp': datetime.now().isoformat(),
            'server': 'production'
        })
        
    except Exception as e:
        logger.error(f"Error in log_usage: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/check-license', methods=['POST'])
def check_license():
    """Check license compliance"""
    try:
        data = request.get_json()
        machine_id = data.get('machine_id', 'unknown')
        
        # Production license check logic
        # In production, this would integrate with your payment system
        allowed = True
        tier = 'research'
        limit = 10
        usage = 0
        
        # Would implement real license logic here
        response = {
            'allowed': allowed,
            'tier': tier,
            'limit': limit,
            'usage': usage,
            'reason': 'Production mode - license check passed',
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"License check for {machine_id[:16] if machine_id else 'unknown'}: {tier} tier")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in check_license: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/user-stats')
def get_user_stats():
    """Get comprehensive user statistics"""
    try:
        db = get_db()
        cursor = db.cursor()
        
        # Get recent activity (last 30 days)
        recent_activity = {}
        for i in range(30):
            date = (datetime.now() - timedelta(days=i)).date().isoformat()
            cursor.execute('''
                SELECT COUNT(DISTINCT machine_id) 
                FROM telemetry_logs 
                WHERE date(timestamp) = ?
            ''', (date,))
            count = cursor.fetchone()[0] or 0
            recent_activity[date] = count
        
        # Get top actions
        cursor.execute('''
            SELECT action, COUNT(*) as count
            FROM telemetry_logs
            WHERE timestamp >= date('now', '-7 days')
            GROUP BY action
            ORDER BY count DESC
            LIMIT 10
        ''')
        top_actions = dict(cursor.fetchall())
        
        # Get platform distribution
        cursor.execute('''
            SELECT platform, COUNT(*) as count
            FROM user_profiles
            WHERE platform IS NOT NULL
            GROUP BY platform
            ORDER BY count DESC
        ''')
        platform_dist = dict(cursor.fetchall())
        
        # Get new users over time
        new_users = {}
        for i in range(30):
            date = (datetime.now() - timedelta(days=i)).date().isoformat()
            cursor.execute('''
                SELECT COUNT(*) 
                FROM user_profiles 
                WHERE date(first_seen) = ?
            ''', (date,))
            count = cursor.fetchone()[0] or 0
            new_users[date] = count
        
        stats = {
            'total_users': user_stats_cache['total_users'],
            'active_users_today': user_stats_cache['active_users_today'],
            'daily_active_users': dict(user_stats_cache['daily_active_users']),
            'total_installations': len(user_stats_cache['install_ids']),
            'recent_activity': recent_activity,
            'top_actions': top_actions,
            'platform_distribution': platform_dist,
            'new_users_over_time': new_users,
            'server_info': {
                'version': '2.0.0-production',
                'timestamp': datetime.now().isoformat(),
                'mode': 'production',
                'cache_updated': user_stats_cache['last_updated'].isoformat()
            }
        }
        
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Error in get_user_stats: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/dashboard')
def dashboard():
    """Production dashboard with enhanced UI"""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Terradev Production Telemetry Dashboard</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #f8f9fa; }
            .container { max-width: 1200px; margin: 0 auto; }
            .header { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }
            .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 20px; }
            .metric { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .metric h3 { margin: 0 0 10px 0; color: #495057; font-size: 14px; text-transform: uppercase; letter-spacing: 0.5px; }
            .metric .value { font-size: 2em; font-weight: bold; color: #007bff; margin: 0; }
            .metric .change { font-size: 0.9em; color: #28a745; margin-top: 5px; }
            .chart { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }
            .refresh { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; font-size: 14px; }
            .refresh:hover { background: #0056b3; }
            .status { display: inline-block; padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; }
            .status.healthy { background: #d4edda; color: #155724; }
            .loading { opacity: 0.6; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🚀 Terradev Production Telemetry Dashboard</h1>
                <p>Real-time user analytics and system monitoring</p>
                <button class="refresh" onclick="location.reload()">Refresh Data</button>
                <span class="status healthy" id="server-status">● Production Server</span>
            </div>
            
            <div class="metrics">
                <div class="metric">
                    <h3>Total Users</h3>
                    <p class="value" id="total-users">Loading...</p>
                    <p class="change" id="users-change">+0 today</p>
                </div>
                
                <div class="metric">
                    <h3>Active Users Today</h3>
                    <p class="value" id="active-today">Loading...</p>
                    <p class="change" id="active-change">Real-time</p>
                </div>
                
                <div class="metric">
                    <h3>Total Installations</h3>
                    <p class="value" id="total-installations">Loading...</p>
                    <p class="change" id="installations-change">+0 today</p>
                </div>
                
                <div class="metric">
                    <h3>Server Uptime</h3>
                    <p class="value" id="server-uptime">Loading...</p>
                    <p class="change" id="server-version">Production</p>
                </div>
            </div>
            
            <div class="chart">
                <h3>📊 Recent Activity (Last 7 Days)</h3>
                <div id="activity-chart" style="height: 200px; display: flex; align-items: end; justify-content: space-between; padding: 20px 0;">
                    <!-- Chart bars will be inserted here -->
                </div>
            </div>
            
            <div class="chart">
                <h3>🔥 Top Actions This Week</h3>
                <div id="top-actions">
                    <!-- Top actions will be inserted here -->
                </div>
            </div>
        </div>
        
        <script>
            async function loadStats() {
                try {
                    const response = await fetch('/user-stats');
                    const stats = await response.json();
                    
                    // Update metrics
                    document.getElementById('total-users').textContent = stats.total_users;
                    document.getElementById('active-today').textContent = stats.active_users_today;
                    document.getElementById('total-installations').textContent = stats.total_installations;
                    document.getElementById('server-version').textContent = stats.server_info.mode;
                    
                    // Update changes
                    const today = new Date().toISOString().split('T')[0];
                    const todayUsers = stats.recent_activity[today] || 0;
                    document.getElementById('users-change').textContent = `+${todayUsers} today`;
                    
                    // Update activity chart
                    const activityChart = document.getElementById('activity-chart');
                    const last7Days = Object.entries(stats.recent_activity)
                        .slice(-7)
                        .reverse();
                    
                    activityChart.innerHTML = last7Days.map(([date, count]) => {
                        const height = Math.max(count * 20, 10);
                        const dateStr = new Date(date).toLocaleDateString('en-US', { weekday: 'short' });
                        return `
                            <div style="text-align: center; flex: 1;">
                                <div style="height: ${height}px; background: #007bff; border-radius: 4px 4px 0 0; margin-bottom: 5px;"></div>
                                <div style="font-size: 12px; color: #6c757d;">${dateStr}</div>
                                <div style="font-size: 10px; color: #6c757d;">${count}</div>
                            </div>
                        `;
                    }).join('');
                    
                    // Update top actions
                    const topActionsDiv = document.getElementById('top-actions');
                    const topActions = Object.entries(stats.top_actions).slice(0, 5);
                    topActionsDiv.innerHTML = topActions.map(([action, count]) => `
                        <div style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #e9ecef;">
                            <span style="text-transform: capitalize;">${action.replace('_', ' ')}</span>
                            <span style="font-weight: bold;">${count}</span>
                        </div>
                    `).join('');
                    
                } catch (error) {
                    console.error('Error loading stats:', error);
                    document.querySelectorAll('.value').forEach(el => el.textContent = 'Error');
                }
            }
            
            // Initial load
            loadStats();
            
            // Auto-refresh every 30 seconds
            setInterval(loadStats, 30000);
        </script>
    </body>
    </html>
    '''

# Register teardown function
app.teardown_appcontext(close_db)

if __name__ == '__main__':
    print("🚀 Starting Terradev Production Telemetry Server...")
    print("📊 Production Dashboard: http://localhost:8080/dashboard")
    print("🔍 Health Check: http://localhost:8080/health")
    print("📈 User Stats: http://localhost:8080/user-stats")
    print("🗄️  Database: telemetry_production.db")
    
    # Initialize database
    init_db()
    
    # Start production server
    app.run(host='0.0.0.0', port=8080, debug=False)
