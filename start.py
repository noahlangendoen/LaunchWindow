#!/usr/bin/env python3
"""
Minimal startup script for Railway deployment
"""

import os
import sys
from datetime import datetime

print(f"=== Launch Prediction API Startup ===")
print(f"Timestamp: {datetime.now().isoformat()}")
print(f"Python: {sys.version}")
print(f"Working dir: {os.getcwd()}")
print(f"Python path: {sys.path}")
print(f"Environment PORT: {os.environ.get('PORT', 'not set')}")

# List files in current directory
print("\nFiles in current directory:")
for item in os.listdir('.'):
    print(f"  {item}")

# Check if src directory exists
if os.path.exists('src'):
    print("\nFiles in src directory:")
    for item in os.listdir('src'):
        print(f"  src/{item}")

try:
    # Import and run the Flask app with database integration
    from src.api.app_db import app
    
    port = int(os.environ.get('PORT', 8000))
    print(f"\nStarting Flask app on port {port}")
    
    app.run(host='0.0.0.0', port=port, debug=False)
    
except Exception as e:
    print(f"\nERROR starting Flask app: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)