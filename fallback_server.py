#!/usr/bin/env python3
"""
Terradev Fallback Telemetry Server
Runs on port 8080 to handle telemetry when primary server is down
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
    conn = sqlite3.connect('telemetry_fallback.db')
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
        conn = sqlite3.connect('telemetry_fallback.db')
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
        
        print(f"✅ [FALLBACK] Logged telemetry: {data.get('action')} from {machine_id[:16]}...")
        
    except Exception as e:
        print(f"❌ Error logging telemetry: {e}")

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0-fallback',
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
            'message': 'Telemetry logged successfully (fallback)',
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
        
        response = {
            'allowed': allowed,
            'tier': tier,
            'limit': limit,
            'usage': usage,
            'reason': 'Fallback mode - always allowed'
        }
        
        print(f"📋 [FALLBACK] License check for {machine_id[:16]}...: {tier} tier")
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/user-stats')
def get_user_stats():
    """Get user statistics"""
    try:
        # Calculate recent activity
        recent_activity = defaultdict(int)
        conn = sqlite3.connect('telemetry_fallback.db')
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
                'version': '1.0.0-fallback',
                'uptime': 'N/A',
                'timestamp': datetime.now().isoformat(),
                'mode': 'fallback'
            }
        }
        
        return jsonify(stats)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Terradev Fallback Telemetry API Server...")
    print("📊 This is the fallback server (port 8080)")
    print("🔍 Health check at: http://localhost:8080/health")
    print("📈 User stats at: http://localhost:8080/user-stats")
    
    # Initialize database
    init_db()
    
    # Start server
    app.run(host='0.0.0.0', port=8080, debug=False)
