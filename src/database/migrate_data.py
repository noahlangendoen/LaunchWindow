"""
Data migration script to move existing CSV data to PostgreSQL database.
Run this script after setting up your Railway PostgreSQL database.
"""

import os
import sys
import pandas as pd
import logging
from datetime import datetime
from typing import Dict, List

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from database.connection import init_database, db
from database.dal import LaunchDataAccess, WeatherDataAccess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataMigrator:
    """Handles migration of CSV data to PostgreSQL."""
    
    def __init__(self, data_dir: str = "data/processed"):
        self.data_dir = data_dir
        self.launch_dal = LaunchDataAccess()
        self.weather_dal = WeatherDataAccess()
        
    def migrate_all_data(self) -> Dict[str, int]:
        """
        Migrate all data from CSV files to database.
        
        Returns:
            Dictionary with migration counts for each data type
        """
        results = {}
        
        try:
            # Migrate launch data
            logger.info("Starting launch data migration...")
            results['spacex_launches'] = self.migrate_spacex_launches()
            results['other_launches'] = self.migrate_other_launches()
            
            # Migrate weather data
            logger.info("Starting weather data migration...")
            results['current_weather'] = self.migrate_current_weather()
            results['weather_forecast'] = self.migrate_weather_forecast()
            
            logger.info("Data migration completed successfully")
            logger.info(f"Migration results: {results}")
            
        except Exception as e:
            logger.error(f"Data migration failed: {e}")
            raise
            
        return results
    
    def migrate_spacex_launches(self) -> int:
        """Migrate SpaceX launch data from CSV."""
        csv_path = os.path.join(self.data_dir, "spacex_launches_processed.csv")
        
        if not os.path.exists(csv_path):
            logger.warning(f"SpaceX launches CSV not found: {csv_path}")
            return 0
            
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"Loading {len(df)} SpaceX launch records...")
            
            launches_data = []
            for _, row in df.iterrows():
                try:
                    # Parse dates
                    date_utc = pd.to_datetime(row['date_utc'])
                    date_local = pd.to_datetime(row['date_local']) if pd.notna(row['date_local']) else date_utc
                    
                    # Parse success (handle various formats)
                    success = None
                    if pd.notna(row.get('success')):
                        success_val = str(row['success']).lower()
                        if success_val in ['true', '1', 'success']:
                            success = True
                        elif success_val in ['false', '0', 'failure']:
                            success = False
                    
                    # Parse payload types and failures from JSON strings
                    payload_types = self._parse_json_field(row.get('payload_types'))
                    failures = self._parse_json_field(row.get('failures'))
                    
                    # Map launchpad to site code (simplified mapping)
                    site_code = self._map_launchpad_to_site(row.get('launchpad_name', ''))
                    
                    launch_data = {
                        'flight_number': int(row['flight_number']) if pd.notna(row.get('flight_number')) else None,
                        'name': row['name'],
                        'date_utc': date_utc,
                        'date_local': date_local,
                        'success': success,
                        'upcoming': bool(row.get('upcoming', False)),
                        'rocket_name': row.get('rocket_name', 'Unknown'),
                        'rocket_type': row.get('rocket_type', 'rocket'),
                        'site_code': site_code,
                        'payload_mass_kg': float(row['payload_mass_kg']) if pd.notna(row.get('payload_mass_kg')) else None,
                        'payload_types': payload_types,
                        'failures': failures
                    }
                    
                    launches_data.append(launch_data)
                    
                except Exception as e:
                    logger.warning(f"Error processing launch record: {e}, row: {row.to_dict()}")
                    continue
            
            # Store in database
            stored_count = self.launch_dal.store_launch_data(launches_data)
            logger.info(f"Successfully migrated {stored_count} SpaceX launches")
            return stored_count
            
        except Exception as e:
            logger.error(f"Error migrating SpaceX launches: {e}")
            return 0
    
    def migrate_other_launches(self) -> int:
        """Migrate other launch data from CSV."""
        csv_path = os.path.join(self.data_dir, "launches_processed.csv")
        
        if not os.path.exists(csv_path):
            logger.warning(f"Other launches CSV not found: {csv_path}")
            return 0
            
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"Loading {len(df)} other launch records...")
            
            launches_data = []
            for _, row in df.iterrows():
                try:
                    # Similar processing as SpaceX data
                    date_utc = pd.to_datetime(row['date_utc']) if pd.notna(row.get('date_utc')) else datetime.utcnow()
                    
                    launch_data = {
                        'name': row.get('name', 'Unknown Mission'),
                        'date_utc': date_utc,
                        'date_local': date_utc,  # Use UTC if local not available
                        'success': bool(row.get('success', False)) if pd.notna(row.get('success')) else None,
                        'rocket_name': row.get('rocket_name', 'Unknown'),
                        'rocket_type': row.get('rocket_type', 'rocket'),
                        'site_code': row.get('site_code', 'KSC'),
                        'payload_mass_kg': float(row['payload_mass_kg']) if pd.notna(row.get('payload_mass_kg')) else None
                    }
                    
                    launches_data.append(launch_data)
                    
                except Exception as e:
                    logger.warning(f"Error processing other launch record: {e}")
                    continue
            
            stored_count = self.launch_dal.store_launch_data(launches_data)
            logger.info(f"Successfully migrated {stored_count} other launches")
            return stored_count
            
        except Exception as e:
            logger.error(f"Error migrating other launches: {e}")
            return 0
    
    def migrate_current_weather(self) -> int:
        """Migrate current weather data from CSV."""
        csv_path = os.path.join(self.data_dir, "weather_current_processed.csv")
        
        if not os.path.exists(csv_path):
            logger.warning(f"Current weather CSV not found: {csv_path}")
            return 0
            
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"Loading {len(df)} current weather records...")
            
            stored_count = 0
            for _, row in df.iterrows():
                try:
                    # Parse timestamp
                    timestamp = pd.to_datetime(row['dt']) if pd.notna(row.get('dt')) else datetime.utcnow()
                    
                    weather_data = {
                        'site_code': row['site_code'],
                        'timestamp': timestamp,
                        'temperature_c': float(row['temmperature_c']),  # Note: typo in original CSV
                        'feels_like_c': float(row['feels_like_c']) if pd.notna(row.get('feels_like_c')) else None,
                        'temp_min_c': float(row['temp_min_c']) if pd.notna(row.get('temp_min_c')) else None,
                        'temp_max_c': float(row['temp_max_c']) if pd.notna(row.get('temp_max_c')) else None,
                        'humidity_percent': float(row['humidity_percent']) if pd.notna(row.get('humidity_percent')) else None,
                        'pressure_hpa': float(row['pressure_hpa']) if pd.notna(row.get('pressure_hpa')) else None,
                        'wind_speed_ms': float(row['wind_speed_ms']) if pd.notna(row.get('wind_speed_ms')) else None,
                        'wind_direction_deg': float(row['wind_direction_deg']) if pd.notna(row.get('wind_direction_deg')) else None,
                        'wind_gust_ms': float(row['wind_gust_ms']) if pd.notna(row.get('wind_gust_ms')) else None,
                        'weather_main': row.get('weather_main'),
                        'weather_description': row.get('weather_description'),
                        'visibility_m': float(row['visibility_m']) if pd.notna(row.get('visibility_m')) else None,
                        'cloud_cover_percent': float(row['cloud_cover_percent']) if pd.notna(row.get('cloud_cover_percent')) else None,
                        'rain_1h_mm': float(row['rain_1h_mm']) if pd.notna(row.get('rain_1h_mm')) else 0,
                        'rain_3h_mm': float(row['rain_3h_mm']) if pd.notna(row.get('rain_3h_mm')) else 0,
                        'snow_1h_mm': float(row['snow_1h_mm']) if pd.notna(row.get('snow_1h_mm')) else 0,
                        'snow_3h_mm': float(row['snow_3h_mm']) if pd.notna(row.get('snow_3h_mm')) else 0,
                        'data_source': 'migrated'
                    }
                    
                    if self.weather_dal.store_current_weather(weather_data):
                        stored_count += 1
                        
                except Exception as e:
                    logger.warning(f"Error processing weather record: {e}")
                    continue
            
            logger.info(f"Successfully migrated {stored_count} current weather records")
            return stored_count
            
        except Exception as e:
            logger.error(f"Error migrating current weather: {e}")
            return 0
    
    def migrate_weather_forecast(self) -> int:
        """Migrate weather forecast data from CSV."""
        csv_path = os.path.join(self.data_dir, "weather_forecast_processed.csv")
        
        if not os.path.exists(csv_path):
            logger.warning(f"Weather forecast CSV not found: {csv_path}")
            return 0
            
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"Loading {len(df)} weather forecast records...")
            
            forecast_data = []
            for _, row in df.iterrows():
                try:
                    # Parse forecast time
                    forecast_time = pd.to_datetime(row['forecast_time'])
                    retrieved_at = pd.to_datetime(row.get('retrieved_at', datetime.utcnow()))
                    
                    forecast = {
                        'site_code': row['site_code'],
                        'forecast_time': forecast_time,
                        'retrieved_at': retrieved_at,
                        'temperature_c': float(row['temperature_c']),
                        'feels_like_c': float(row['feels_like_c']) if pd.notna(row.get('feels_like_c')) else None,
                        'humidity_percent': float(row['humidity_percent']) if pd.notna(row.get('humidity_percent')) else None,
                        'pressure_hpa': float(row['pressure_hpa']) if pd.notna(row.get('pressure_hpa')) else None,
                        'wind_speed_ms': float(row['wind_speed_ms']) if pd.notna(row.get('wind_speed_ms')) else None,
                        'wind_direction_deg': float(row['wind_direction_deg']) if pd.notna(row.get('wind_direction_deg')) else None,
                        'weather_main': row.get('weather_main'),
                        'weather_description': row.get('weather_description'),
                        'visibility_m': float(row['visibility_m']) if pd.notna(row.get('visibility_m')) else None,
                        'cloud_cover_percent': float(row['cloud_cover_percent']) if pd.notna(row.get('cloud_cover_percent')) else None,
                        'rain_1h_mm': float(row['rain_1h_mm']) if pd.notna(row.get('rain_1h_mm')) else 0,
                        'rain_3h_mm': float(row['rain_3h_mm']) if pd.notna(row.get('rain_3h_mm')) else 0,
                        'pop': float(row['pop']) if pd.notna(row.get('pop')) else None,
                        'data_source': 'migrated'
                    }
                    
                    forecast_data.append(forecast)
                    
                except Exception as e:
                    logger.warning(f"Error processing forecast record: {e}")
                    continue
            
            stored_count = self.weather_dal.store_forecast_data(forecast_data)
            logger.info(f"Successfully migrated {stored_count} weather forecast records")
            return stored_count
            
        except Exception as e:
            logger.error(f"Error migrating weather forecast: {e}")
            return 0
    
    def _parse_json_field(self, field_value) -> List:
        """Parse JSON-like string fields from CSV."""
        if pd.isna(field_value) or field_value == '':
            return []
            
        try:
            # Remove brackets and quotes, split by comma
            if isinstance(field_value, str):
                # Handle formats like '["Satellite"]' or '[{"time": 33, "reason": "failure"}]'
                import json
                import ast
                
                # Try JSON first
                try:
                    return json.loads(field_value)
                except:
                    # Try literal_eval for Python literals
                    try:
                        return ast.literal_eval(field_value)
                    except:
                        # Fallback: treat as simple string list
                        cleaned = field_value.strip('[]"\'')
                        if cleaned:
                            return [item.strip().strip('"\'') for item in cleaned.split(',')]
                        return []
            return []
            
        except Exception as e:
            logger.warning(f"Error parsing JSON field '{field_value}': {e}")
            return []
    
    def _map_launchpad_to_site(self, launchpad_name: str) -> str:
        """Map launchpad name to site code."""
        launchpad_lower = launchpad_name.lower()
        
        if 'kennedy' in launchpad_lower or 'ksc' in launchpad_lower:
            return 'KSC'
        elif 'vandenberg' in launchpad_lower or 'vsfb' in launchpad_lower:
            return 'VSFB'
        elif 'canaveral' in launchpad_lower or 'ccafs' in launchpad_lower:
            return 'CCAFS'
        else:
            # Default to KSC for unknown launchpads
            return 'KSC'

def main():
    """Main migration function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Migrate CSV data to PostgreSQL')
    parser.add_argument('--database-url', help='PostgreSQL database URL')
    parser.add_argument('--data-dir', default='data/processed', help='Directory containing CSV files')
    parser.add_argument('--create-tables', action='store_true', help='Create tables before migration')
    
    args = parser.parse_args()
    
    try:
        # Initialize database
        logger.info("Initializing database connection...")
        success = init_database(
            database_url=args.database_url,
            create_tables=args.create_tables
        )
        
        if not success:
            logger.error("Failed to initialize database")
            return False
        
        # Perform migration
        migrator = DataMigrator(data_dir=args.data_dir)
        results = migrator.migrate_all_data()
        
        logger.info("Migration completed successfully!")
        logger.info(f"Final results: {results}")
        
        # Print summary
        total_records = sum(results.values())
        print(f"\n=== Migration Summary ===")
        print(f"SpaceX Launches: {results.get('spacex_launches', 0)}")
        print(f"Other Launches: {results.get('other_launches', 0)}")
        print(f"Current Weather: {results.get('current_weather', 0)}")
        print(f"Weather Forecast: {results.get('weather_forecast', 0)}")
        print(f"Total Records: {total_records}")
        
        return True
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        return False
    finally:
        # Close database connections
        db.close()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)