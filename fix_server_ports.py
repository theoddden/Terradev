#!/usr/bin/env python3
"""
Fix Server Port Issues and Deploy Properly
"""

import subprocess
import time

def fix_server_ports():
    print('🔧 FIXING SERVER PORT ISSUES')
    print('=' * 40)
    
    # Check what's in the fallback and stripe servers
    commands = [
        "cd /home/ubuntu",
        "ls -la *.py || echo 'No Python files in /home/ubuntu'",
        "echo '=== Check fallback_server.py port ==='",
        "grep -n 'app.run' fallback_server.py || echo 'No app.run found in fallback'",
        "echo '=== Check stripe_server.py port ==='",
        "grep -n 'app.run' stripe_server.py || echo 'No app.run found in stripe'",
        "echo '=== Kill all Python processes ==='",
        "pkill -f python3 || echo 'No Python processes'",
        "sleep 3",
        "echo '=== Start Production Server (port 8080) ==='",
        "cd terradev && nohup python3 production_server.py > production.log 2>&1 &",
        "sleep 5",
        "echo '=== Start Fallback Server (port 8081) ==='",
        "cd terradev && python3 fallback_server.py &",
        "sleep 3",
        "echo '=== Check if fallback is running ==='",
        "ps aux | grep fallback_server || echo 'Fallback not running'",
        "echo '=== Start Stripe Server (port 8082) ==='",
        "cd terradev && python3 stripe_server.py &",
        "sleep 3",
        "echo '=== Check if stripe is running ==='",
        "ps aux | grep stripe_server || echo 'Stripe not running'",
        "echo '=== Check all processes ==='",
        "ps aux | grep python3 | grep -v grep || echo 'No Python processes'",
        "echo '=== Test ports ==='",
        "curl -s http://localhost:8080/health || echo 'Production not ready'",
        "curl -s http://localhost:8081/health || echo 'Fallback not ready'",
        "curl -s http://localhost:8082/pricing || echo 'Stripe not ready'"
    ]
    
    for cmd in commands:
        print(f"📤 Running: {cmd[:50]}...")
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=15)
            if result.stdout.strip():
                print(f"✅ Output: {result.stdout.strip()}")
            if result.stderr.strip():
                print(f"⚠️  Error: {result.stderr.strip()}")
        except subprocess.TimeoutExpired:
            print("⏰ Command timeout")
        except Exception as e:
            print(f"❌ Error: {e}")
        print()

def create_fixed_servers():
    """Create fixed server files with correct ports"""
    print('🔧 CREATING FIXED SERVER FILES')
    print('=' * 40)
    
    # Fixed fallback server
    fallback_fix = '''#!/usr/bin/env python3
import json
import sqlite3
from datetime import datetime
from flask import Flask, jsonify, request

app = Flask(__name__)

def init_db():
    conn = sqlite3.connect('telemetry_fallback.db')
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS telemetry_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            machine_id TEXT,
            action TEXT,
            timestamp TEXT,
            details TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0-fallback',
        'server': 'fallback'
    })

@app.route('/log-usage', methods=['POST'])
def log_usage():
    try:
        data = request.get_json()
        conn = sqlite3.connect('telemetry_fallback.db')
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO telemetry_logs (machine_id, action, timestamp, details)
            VALUES (?, ?, ?, ?)
        """, (
            data.get('machine_id'),
            data.get('action'),
            data.get('timestamp'),
            json.dumps(data.get('details', {}))
        ))
        conn.commit()
        conn.close()
        
        return jsonify({
            'status': 'success',
            'message': 'Telemetry logged successfully (fallback)',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=8081)
'''
    
    # Write fixed fallback server
    with open('/tmp/fallback_server_fixed.py', 'w') as f:
        f.write(fallback_fix)
    
    print('✅ Fixed fallback server created')
    print('📤 Will deploy to: /home/ubuntu/terradev/fallback_server_fixed.py')
    
    return True

if __name__ == '__main__':
    fix_server_ports()
    create_fixed_servers()
