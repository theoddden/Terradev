#!/usr/bin/env python3
"""
Terradev Telemetry API Server
Simple Flask API to handle telemetry data and user statistics
"""

import json
import sqlite3
import threading
from datetime import datetime, timedelta
from flask import Flask, jsonify, request
from collections import defaultdict
import os

app = Flask(__name__)

# In-memory storage for demo purposes
telemetry_data = defaultdict(list)
user_stats = {
    'total_users': 0,
    'active_users_today': 0,
    'daily_active_users': {},
    'machine_ids': set(),
    'install_ids': set()
}

# Initialize database
def init_db():
    """Initialize SQLite database for telemetry storage"""
    conn = sqlite3.connect('telemetry.db')
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS telemetry_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            machine_id TEXT,
            install_id TEXT,
            action TEXT,
            timestamp TEXT,
            details TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS daily_active_users (
            date TEXT PRIMARY KEY,
            machine_ids TEXT,
            count INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

def log_telemetry(data):
    """Log telemetry data"""
    try:
        conn = sqlite3.connect('telemetry.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO telemetry_logs 
            (machine_id, install_id, action, timestamp, details)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            data.get('machine_id'),
            data.get('install_id'),
            data.get('action'),
            data.get('timestamp'),
            json.dumps(data.get('details', {}))
        ))
        
        conn.commit()
        conn.close()
        
        # Update in-memory stats
        machine_id = data.get('machine_id')
        install_id = data.get('install_id')
        
        if machine_id:
            user_stats['machine_ids'].add(machine_id)
            user_stats['total_users'] = len(user_stats['machine_ids'])
        
        if install_id:
            user_stats['install_ids'].add(install_id)
        
        # Track daily active users
        today = datetime.now().date().isoformat()
        if data.get('action') == 'daily_active_users':
            user_stats['daily_active_users'][today] = user_stats['daily_active_users'].get(today, 0) + 1
            user_stats['active_users_today'] = user_stats['daily_active_users'].get(today, 0)
        
        print(f"✅ Logged telemetry: {data.get('action')} from {machine_id[:16]}...")
        
    except Exception as e:
        print(f"❌ Error logging telemetry: {e}")

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0',
        'users': {
            'total_users': user_stats['total_users'],
            'active_users_today': user_stats['active_users_today'],
            'total_installations': len(user_stats['install_ids'])
        }
    })

@app.route('/log-usage', methods=['POST'])
def log_usage():
    """Log usage telemetry"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        log_telemetry(data)
        
        return jsonify({
            'status': 'success',
            'message': 'Telemetry logged successfully',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/check-license', methods=['POST'])
def check_license():
    """Check license compliance"""
    try:
        data = request.get_json()
        machine_id = data.get('machine_id', 'unknown')
        
        # Simple license check logic
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
            'reason': 'Demo mode - always allowed'
        }
        
        print(f"📋 License check for {machine_id[:16]}...: {tier} tier")
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/user-stats')
def get_user_stats():
    """Get user statistics"""
    try:
        # Calculate recent activity
        recent_activity = defaultdict(int)
        conn = sqlite3.connect('telemetry.db')
        cursor = conn.cursor()
        
        # Get last 7 days of activity
        for i in range(7):
            date = (datetime.now() - timedelta(days=i)).date().isoformat()
            cursor.execute('''
                SELECT COUNT(DISTINCT machine_id) 
                FROM telemetry_logs 
                WHERE date(timestamp) = ?
            ''', (date,))
            count = cursor.fetchone()[0] or 0
            recent_activity[date] = count
        
        conn.close()
        
        stats = {
            'total_users': user_stats['total_users'],
            'active_users_today': user_stats['active_users_today'],
            'daily_active_users': dict(user_stats['daily_active_users']),
            'recent_activity': dict(recent_activity),
            'total_installations': len(user_stats['install_ids']),
            'server_info': {
                'version': '1.0.0',
                'uptime': 'N/A',
                'timestamp': datetime.now().isoformat()
            }
        }
        
        return jsonify(stats)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/dashboard')
def dashboard():
    """Simple dashboard HTML"""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Terradev Telemetry Dashboard</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .metric { background: #f5f5f5; padding: 20px; margin: 10px 0; border-radius: 8px; }
            .metric h3 { margin: 0 0 10px 0; color: #333; }
            .metric .value { font-size: 2em; font-weight: bold; color: #007bff; }
            .refresh { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
        </style>
    </head>
    <body>
        <h1>🚀 Terradev Telemetry Dashboard</h1>
        <button class="refresh" onclick="location.reload()">Refresh</button>
        
        <div class="metric">
            <h3>Total Users</h3>
            <div class="value" id="total-users">Loading...</div>
        </div>
        
        <div class="metric">
            <h3>Active Users Today</h3>
            <div class="value" id="active-today">Loading...</div>
        </div>
        
        <div class="metric">
            <h3>Total Installations</h3>
            <div class="value" id="total-installations">Loading...</div>
        </div>
        
        <script>
            async function loadStats() {
                try {
                    const response = await fetch('/user-stats');
                    const stats = await response.json();
                    
                    document.getElementById('total-users').textContent = stats.total_users;
                    document.getElementById('active-today').textContent = stats.active_users_today;
                    document.getElementById('total-installations').textContent = stats.total_installations;
                } catch (error) {
                    console.error('Error loading stats:', error);
                }
            }
            
            loadStats();
            setInterval(loadStats, 30000); // Refresh every 30 seconds
        </script>
    </body>
    </html>
    '''

if __name__ == '__main__':
    print("🚀 Starting Terradev Telemetry API Server...")
    print("📊 Dashboard will be available at: http://localhost:8080/dashboard")
    print("🔍 Health check at: http://localhost:8080/health")
    print("📈 User stats at: http://localhost:8080/user-stats")
    
    # Initialize database
    init_db()
    
    # Start server
    app.run(host='0.0.0.0', port=8080, debug=True)
