"""
Database-integrated Weather Service for Launch Window Prediction System.
Replaces file-based caching with PostgreSQL storage.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from .dal import WeatherDataAccess
from ..services.weather_service import WeatherService as BaseWeatherService

logger = logging.getLogger(__name__)

class DatabaseWeatherService(BaseWeatherService):
    """Weather service with PostgreSQL database integration."""
    
    def __init__(self):
        super().__init__()
        self.weather_dal = WeatherDataAccess()
        logger.info("DatabaseWeatherService initialized")
    
    def get_current_weather(self, site_code: str, force_refresh: bool = False) -> Optional[Dict]:
        """
        Get current weather for a site, using database cache.
        
        Args:
            site_code: Launch site code (KSC, VSFB, CCAFS)
            force_refresh: If True, bypass cache and fetch fresh data
            
        Returns:
            Current weather data or None if unavailable
        """
        if not force_refresh:
            # Try to get from database cache first
            cached_weather = self.weather_dal.get_current_weather(site_code, max_age_minutes=10)
            if cached_weather:
                logger.info(f"Using cached weather data for {site_code}")
                return cached_weather
        
        # Fetch fresh data from API
        fresh_weather = super().get_current_weather(site_code)
        if fresh_weather:
            # Store in database
            try:
                # Add metadata for database storage
                db_weather_data = fresh_weather.copy()
                db_weather_data.update({
                    'timestamp': datetime.utcnow(),
                    'data_source': 'live',
                    'expires_at': datetime.utcnow() + timedelta(minutes=10)
                })
                
                success = self.weather_dal.store_current_weather(db_weather_data)
                if success:
                    logger.info(f"Stored fresh weather data for {site_code} in database")
                else:
                    logger.warning(f"Failed to store weather data for {site_code}")
                    
            except Exception as e:
                logger.error(f"Error storing weather data for {site_code}: {e}")
        
        return fresh_weather
    
    def get_forecast_weather(self, site_code: str, days: int = 5, force_refresh: bool = False) -> Optional[List[Dict]]:
        """
        Get weather forecast for a site, using database cache.
        
        Args:
            site_code: Launch site code
            days: Number of days of forecast data
            force_refresh: If True, bypass cache and fetch fresh data
            
        Returns:
            List of forecast data or None if unavailable
        """
        if not force_refresh:
            # Try to get from database cache first
            cached_forecast = self.weather_dal.get_forecast_data(site_code, hours_ahead=days * 24)
            if cached_forecast:
                logger.info(f"Using cached forecast data for {site_code} ({len(cached_forecast)} records)")
                return cached_forecast
        
        # Fetch fresh data from API
        fresh_forecast = super().get_forecast_weather(site_code, days)
        if fresh_forecast:
            # Convert to database format and store
            try:
                db_forecast_data = []
                retrieved_at = datetime.utcnow()
                expires_at = retrieved_at + timedelta(minutes=30)
                
                for forecast_item in fresh_forecast:
                    db_item = forecast_item.copy()
                    db_item.update({
                        'retrieved_at': retrieved_at,
                        'data_source': 'live',
                        'expires_at': expires_at
                    })
                    
                    # Parse forecast_time if it's a string
                    if isinstance(db_item.get('forecast_time'), str):
                        db_item['forecast_time'] = datetime.fromisoformat(
                            db_item['forecast_time'].replace('Z', '+00:00')
                        )
                    
                    db_forecast_data.append(db_item)
                
                stored_count = self.weather_dal.store_forecast_data(db_forecast_data)
                logger.info(f"Stored {stored_count} forecast records for {site_code} in database")
                
            except Exception as e:
                logger.error(f"Error storing forecast data for {site_code}: {e}")
        
        return fresh_forecast
    
    def update_all_weather_data(self) -> Dict:
        """
        Update weather data for all sites and store in database.
        
        Returns:
            Update results summary
        """
        logger.info("=== Starting scheduled weather update (Database) ===")
        
        sites = ['KSC', 'VSFB', 'CCAFS']
        successful_updates = 0
        update_details = {}
        
        for site_code in sites:
            logger.info(f"Updating weather data for {site_code}...")
            site_success = True
            
            try:
                # Update current weather
                current_weather = self.get_current_weather(site_code, force_refresh=True)
                if current_weather:
                    logger.info(f"[OK] Current weather updated for {site_code}")
                else:
                    logger.warning(f"[FAIL] Current weather update failed for {site_code}")
                    site_success = False
                
                # Update forecast
                forecast_data = self.get_forecast_weather(site_code, days=5, force_refresh=True)
                if forecast_data:
                    logger.info(f"[OK] Forecast data updated for {site_code} ({len(forecast_data)} records)")
                else:
                    logger.warning(f"[FAIL] Forecast update failed for {site_code}")
                    site_success = False
                
                if site_success:
                    successful_updates += 1
                    
                update_details[site_code] = {
                    'success': site_success,
                    'current_weather': current_weather is not None,
                    'forecast_count': len(forecast_data) if forecast_data else 0
                }
                
            except Exception as e:
                logger.error(f"Error updating weather for {site_code}: {e}")
                update_details[site_code] = {
                    'success': False,
                    'error': str(e)
                }
        
        # Clean up old data
        try:
            cleanup_results = self.weather_dal.cleanup_old_data(days_old=7)
            logger.info(f"Cleaned up old weather data: {cleanup_results}")
        except Exception as e:
            logger.warning(f"Failed to clean up old data: {e}")
        
        success_rate = successful_updates / len(sites) if sites else 0
        
        result = {
            'timestamp': datetime.utcnow().isoformat(),
            'total_sites': len(sites),
            'successful_updates': successful_updates,
            'success_rate': success_rate,
            'update_details': update_details
        }
        
        logger.info(f"=== Weather update completed: {successful_updates}/{len(sites)} sites updated ===")
        return result
    
    def get_service_status(self) -> Dict:
        """
        Get weather service status with database information.
        
        Returns:
            Service status information
        """
        base_status = super().get_service_status()
        
        # Add database-specific information
        try:
            # Get recent weather data counts
            sites = ['KSC', 'VSFB', 'CCAFS']
            cache_status = {}
            
            for site_code in sites:
                current_weather = self.weather_dal.get_current_weather(site_code, max_age_minutes=60)
                forecast_data = self.weather_dal.get_forecast_data(site_code, hours_ahead=48)
                
                cache_status[site_code] = {
                    'has_current': current_weather is not None,
                    'current_age_minutes': self._calculate_age_minutes(current_weather),
                    'forecast_count': len(forecast_data) if forecast_data else 0,
                    'data_source': current_weather.get('data_source', 'unknown') if current_weather else 'none'
                }
            
            base_status.update({
                'storage_type': 'postgresql',
                'database_cache_status': cache_status
            })
            
        except Exception as e:
            logger.error(f"Error getting database status: {e}")
            base_status['database_error'] = str(e)
        
        return base_status
    
    def force_refresh_site(self, site_code: str) -> bool:
        """
        Force refresh weather data for a specific site.
        
        Args:
            site_code: Site to refresh
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Force refreshing weather data for {site_code}")
            
            # Refresh current weather
            current = self.get_current_weather(site_code, force_refresh=True)
            # Refresh forecast
            forecast = self.get_forecast_weather(site_code, force_refresh=True)
            
            success = current is not None and forecast is not None
            logger.info(f"Force refresh for {site_code}: {'success' if success else 'failed'}")
            return success
            
        except Exception as e:
            logger.error(f"Error during force refresh for {site_code}: {e}")
            return False
    
    def _calculate_age_minutes(self, weather_data: Optional[Dict]) -> Optional[int]:
        """Calculate age of weather data in minutes."""
        if not weather_data or 'timestamp' not in weather_data:
            return None
        
        try:
            if isinstance(weather_data['timestamp'], str):
                timestamp = datetime.fromisoformat(weather_data['timestamp'].replace('Z', '+00:00'))
            else:
                timestamp = weather_data['timestamp']
            
            age = datetime.utcnow() - timestamp.replace(tzinfo=None)
            return int(age.total_seconds() / 60)
            
        except Exception as e:
            logger.warning(f"Error calculating weather data age: {e}")
            return None