#!/usr/bin/env python3
"""
Test script for PostgreSQL database setup on Railway.
Run this to verify your database connection and schema.
"""

import os
import sys
from datetime import datetime

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

def test_database_connection():
    """Test database connection and setup."""
    try:
        print("=== PostgreSQL Database Connection Test ===")
        print(f"Timestamp: {datetime.now().isoformat()}")
        
        # Import database components
        from src.database.connection import db, init_database, get_database_url_from_railway
        
        # Get database URL
        database_url = get_database_url_from_railway()
        if not database_url:
            print("❌ No database URL found!")
            print("Make sure these Railway environment variables are set:")
            print("  - DATABASE_URL or DATABASE_PRIVATE_URL")
            print("  - Or: PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD")
            return False
        
        print(f"✅ Database URL found: {database_url[:50]}...")
        
        # Initialize database
        print("\n--- Initializing Database ---")
        success = init_database(database_url=database_url, create_tables=True)
        if not success:
            print("❌ Database initialization failed!")
            return False
        
        print("✅ Database initialized successfully!")
        
        # Test connection health
        print("\n--- Testing Connection Health ---")
        health = db.health_check()
        print(f"Database Status: {health['status']}")
        if health['status'] == 'healthy':
            print(f"✅ PostgreSQL Version: {health.get('database_version', 'Unknown')}")
            print(f"✅ Tables Found: {health.get('table_count', 0)}")
            print(f"✅ Connection Pool: {health.get('connection_pool', {})}")
        else:
            print(f"❌ Health Check Failed: {health.get('error', 'Unknown error')}")
            return False
        
        # Test weather service
        print("\n--- Testing Weather Service ---")
        from src.database.weather_service_db import DatabaseWeatherService
        
        weather_service = DatabaseWeatherService()
        status = weather_service.get_service_status()
        print(f"Weather Service Status:")
        print(f"  - Storage Type: {status.get('storage_type', 'unknown')}")
        print(f"  - API Mode: {status.get('api_mode', 'unknown')}")
        print(f"  - Available Sites: {status.get('available_sites', [])}")
        
        # Test a single weather update
        print("\n--- Testing Single Site Weather Update ---")
        current_weather = weather_service.get_current_weather('KSC', force_refresh=True)
        if current_weather:
            print(f"✅ Weather data retrieved for KSC:")
            print(f"  - Temperature: {current_weather.get('temperature_c', 'N/A')}°C")
            print(f"  - Wind Speed: {current_weather.get('wind_speed_ms', 'N/A')} m/s")
            print(f"  - Conditions: {current_weather.get('weather_description', 'N/A')}")
        else:
            print("⚠️  No weather data retrieved (may be using mock data)")
        
        print("\n✅ ALL TESTS PASSED! Your PostgreSQL setup is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up database connections
        try:
            if 'db' in locals():
                db.close()
        except:
            pass

def show_environment_setup():
    """Show environment setup instructions."""
    print("\n=== Railway Environment Setup ===")
    print("Make sure these environment variables are set in Railway:")
    print("")
    print("1. Database Connection (Railway usually provides these automatically):")
    print("   - DATABASE_URL or DATABASE_PRIVATE_URL")
    print("   - Or individual variables: PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD")
    print("")
    print("2. OpenWeather API (for live weather data):")
    print("   - OPENWEATHER_API_KEY=your_api_key_here")
    print("")
    print("3. Current Environment Variables Found:")
    env_vars = ['DATABASE_URL', 'DATABASE_PRIVATE_URL', 'PGHOST', 'PGUSER', 'OPENWEATHER_API_KEY']
    for var in env_vars:
        value = os.getenv(var)
        if value:
            print(f"   ✅ {var}: {'*' * min(len(value), 20)}...")
        else:
            print(f"   ❌ {var}: Not set")

if __name__ == "__main__":
    print("Starting database setup test...")
    
    # Show environment setup
    show_environment_setup()
    
    # Run connection test
    success = test_database_connection()
    
    if success:
        print("\n🎉 Your PostgreSQL setup is ready!")
        print("You can now:")
        print("1. Deploy to Railway")
        print("2. Weather data will be collected every 10 minutes automatically")
        print("3. Frontend will pull latest weather from PostgreSQL")
    else:
        print("\n⚠️  Please fix the issues above before deploying.")
    
    sys.exit(0 if success else 1)