#!/usr/bin/env python3
"""
Test script to simulate Railway startup process.
This tests if your app initializes properly when imported.
"""

import os
import sys
from datetime import datetime

print("=== Railway Startup Simulation ===")
print(f"Timestamp: {datetime.now().isoformat()}")

# Add project root to path (same as start.py does)
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

try:
    print("\n--- Attempting to import app_db (simulating start.py) ---")
    
    # This simulates exactly what start.py does
    from src.api.app_db import app
    
    print("\n✅ App imported successfully!")
    print("✅ Components should have auto-initialized")
    
    # Check if initialization worked
    print("\n--- Checking initialization status ---")
    from src.api.app_db import is_initialized, scheduler, db_weather_service
    
    print(f"Is initialized: {is_initialized}")
    print(f"Scheduler exists: {scheduler is not None}")
    print(f"Scheduler running: {scheduler.running if scheduler else False}")
    print(f"Weather service exists: {db_weather_service is not None}")
    
    if is_initialized:
        print("\n✅ SUCCESS! Your app will initialize properly on Railway")
    else:
        print("\n❌ ISSUE: App failed to initialize automatically")
        
    # Test a simple endpoint
    print("\n--- Testing Flask app functionality ---")
    with app.test_client() as client:
        response = client.get('/health')
        print(f"Health endpoint status: {response.status_code}")
        if response.status_code == 200:
            print("✅ Health endpoint working")
        else:
            print(f"⚠️  Health endpoint returned: {response.status_code}")
            
except Exception as e:
    print(f"\n❌ IMPORT FAILED: {e}")
    import traceback
    traceback.print_exc()
    print("\nThis indicates there will be issues on Railway deployment.")
    print("Check the error above and fix any import/dependency issues.")

print(f"\nTest completed at {datetime.now().isoformat()}")