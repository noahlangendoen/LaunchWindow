"""
Setup script for deploying Launch Window Prediction System on Railway.
Handles database initialization, data migration, and system setup.
"""

import os
import sys
import logging
import argparse
from datetime import datetime

# Add src to path
sys.path.append('src')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_database():
    """Initialize PostgreSQL database on Railway."""
    try:
        from database.connection import init_database, get_database_url_from_railway
        
        logger.info("Setting up PostgreSQL database...")
        
        # Get database URL from Railway environment
        database_url = get_database_url_from_railway()
        if not database_url:
            logger.error("No database URL found. Ensure Railway PostgreSQL service is provisioned.")
            return False
        
        logger.info(f"Database URL found: {database_url.split('@')[0]}@***")
        
        # Initialize database and create tables
        success = init_database(database_url=database_url, create_tables=True)
        
        if success:
            logger.info("Database setup completed successfully!")
            return True
        else:
            logger.error("Database setup failed!")
            return False
            
    except Exception as e:
        logger.error(f"Database setup error: {e}")
        return False

def migrate_data():
    """Migrate existing CSV data to PostgreSQL."""
    try:
        from database.migrate_data import DataMigrator
        
        logger.info("Starting data migration from CSV to PostgreSQL...")
        
        # Check if CSV files exist
        data_dir = "data/processed"
        if not os.path.exists(data_dir):
            logger.warning(f"Data directory {data_dir} not found. Skipping migration.")
            return True
        
        # Run migration
        migrator = DataMigrator(data_dir=data_dir)
        results = migrator.migrate_all_data()
        
        total_migrated = sum(results.values())
        logger.info(f"Data migration completed! Migrated {total_migrated} records total.")
        
        # Print detailed results
        for data_type, count in results.items():
            if count > 0:
                logger.info(f"  {data_type}: {count} records")
        
        return True
        
    except Exception as e:
        logger.error(f"Data migration error: {e}")
        return False

def install_dependencies():
    """Install required Python packages."""
    try:
        logger.info("Installing Python dependencies...")
        
        # Install from requirements.txt
        os.system("pip install -r requirements.txt")
        
        logger.info("Dependencies installed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Dependency installation error: {e}")
        return False

def validate_setup():
    """Validate that everything is set up correctly."""
    try:
        logger.info("Validating setup...")
        
        # Test database connection
        from database.connection import db
        health = db.health_check()
        
        if health['status'] == 'healthy':
            logger.info(f"Database validation: PASSED")
            logger.info(f"  Database version: {health.get('database_version', 'Unknown')}")
            logger.info(f"  Table count: {health.get('table_count', 0)}")
        else:
            logger.error(f"Database validation: FAILED - {health.get('error', 'Unknown error')}")
            return False
        
        # Test ML model loading
        try:
            from ml_models.success_predictor import LaunchSuccessPredictor
            predictor = LaunchSuccessPredictor()
            
            model_path = "data/models/launch_success_model.pkl"
            if os.path.exists(model_path):
                predictor.load_model(model_path)
                logger.info(f"ML model validation: PASSED")
                logger.info(f"  Model type: {predictor.model_type}")
                logger.info(f"  Is trained: {predictor.is_trained}")
            else:
                logger.warning(f"ML model file not found at {model_path}")
        except Exception as e:
            logger.warning(f"ML model validation issue: {e}")
        
        # Test weather service
        try:
            from database.weather_service_db import DatabaseWeatherService
            weather_service = DatabaseWeatherService()
            status = weather_service.get_service_status()
            logger.info(f"Weather service validation: PASSED")
            logger.info(f"  API mode: {status.get('api_mode', 'unknown')}")
        except Exception as e:
            logger.warning(f"Weather service validation issue: {e}")
        
        logger.info("Setup validation completed!")
        return True
        
    except Exception as e:
        logger.error(f"Setup validation error: {e}")
        return False

def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description='Setup Launch Window System on Railway')
    parser.add_argument('--skip-deps', action='store_true', help='Skip dependency installation')
    parser.add_argument('--skip-migration', action='store_true', help='Skip data migration')
    parser.add_argument('--validate-only', action='store_true', help='Only run validation')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Launch Window Prediction System - Railway Setup")
    print(f"Started at: {datetime.now().isoformat()}")
    print("=" * 60)
    
    success = True
    
    if not args.validate_only:
        # Step 1: Install dependencies
        if not args.skip_deps:
            print("\n1. Installing dependencies...")
            if not install_dependencies():
                success = False
        else:
            print("\n1. Skipping dependency installation")
        
        # Step 2: Setup database
        print("\n2. Setting up database...")
        if not setup_database():
            success = False
        
        # Step 3: Migrate data
        if not args.skip_migration:
            print("\n3. Migrating data...")
            if not migrate_data():
                success = False
        else:
            print("\n3. Skipping data migration")
    
    # Step 4: Validate setup
    print("\n4. Validating setup...")
    if not validate_setup():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("✅ SETUP COMPLETED SUCCESSFULLY!")
        print("\nNext steps:")
        print("1. Your Railway app is ready to deploy")
        print("2. Use the database-integrated API: python src/api/app_db.py")
        print("3. Access the health endpoint: /health")
        print("4. Monitor database stats: /database/stats")
    else:
        print("❌ SETUP FAILED!")
        print("\nPlease check the error messages above and fix any issues.")
    
    print(f"Completed at: {datetime.now().isoformat()}")
    print("=" * 60)
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)