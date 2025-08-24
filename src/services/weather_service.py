from datetime import datetime
from typing import Optional, Dict, List
import sys
import os

# Add project root to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_root)

from src.data_storage.weather_cache import WeatherCache
from src.data_ingestion.collect_weather import WeatherCollector

class WeatherService:
    """Service class that provides weather data with caching layer"""
    
    def __init__(self):
        self.cache = WeatherCache()
        self.collector = WeatherCollector()
        print("WeatherService initialized with caching enabled")
    
    def get_current_weather(self, site_code: str, force_refresh: bool = False) -> Optional[Dict]:
        """
        Get current weather with cache-first approach
        
        Args:
            site_code: Launch site code (KSC, VSFB, CCAFS)
            force_refresh: If True, bypass cache and fetch fresh data
        
        Returns:
            Weather data dictionary or None if unavailable
        """
        if not force_refresh:
            # Try to get from cache first
            cached_data = self.cache.get_current_weather(site_code)
            if cached_data:
                print(f"Using cached weather data for {site_code}")
                return cached_data
        
        # Cache miss or force refresh - fetch fresh data
        print(f"Fetching fresh weather data for {site_code}")
        fresh_data = self.collector.get_current_weather(site_code)
        
        if fresh_data:
            # Cache the fresh data for 10 minutes
            self.cache.cache_current_weather(site_code, fresh_data, cache_duration_minutes=10)
            return fresh_data
        
        # If fresh fetch fails, try to return stale cache data as fallback
        if not force_refresh:
            print(f"Fresh data fetch failed for {site_code}, checking for stale cache data")
            # Get any cached data, even if expired (emergency fallback)
            try:
                with self.cache._lock:
                    import sqlite3
                    with sqlite3.connect(self.cache.db_path) as conn:
                        cursor = conn.execute("""
                            SELECT data FROM weather_current 
                            WHERE site_code = ?
                            ORDER BY cached_at DESC LIMIT 1
                        """, (site_code,))
                        
                        row = cursor.fetchone()
                        if row:
                            import json
                            stale_data = json.loads(row[0])
                            print(f"Using stale cached data for {site_code} as emergency fallback")
                            return stale_data
            except Exception as e:
                print(f"Error retrieving stale cache data: {e}")
        
        return None
    
    def get_forecast_weather(self, site_code: str, days: int = 5, force_refresh: bool = False) -> Optional[List[Dict]]:
        """
        Get weather forecast with cache-first approach
        
        Args:
            site_code: Launch site code
            days: Number of days to forecast
            force_refresh: If True, bypass cache and fetch fresh data
        
        Returns:
            List of forecast data dictionaries or None if unavailable
        """
        if not force_refresh:
            # Try to get from cache first
            cached_data = self.cache.get_forecast_weather(site_code)
            if cached_data:
                print(f"Using cached forecast data for {site_code}")
                return cached_data
        
        # Cache miss or force refresh - fetch fresh data
        print(f"Fetching fresh forecast data for {site_code}")
        fresh_data = self.collector.get_forecast(site_code, days=days)
        
        if fresh_data:
            # Cache the fresh data for 30 minutes (forecasts change less frequently)
            self.cache.cache_forecast_weather(site_code, fresh_data, cache_duration_minutes=30)
            return fresh_data
        
        return None
    
    def update_all_weather_data(self):
        """
        Update weather data for all launch sites
        Called by the background scheduler every 10 minutes
        """
        print("=== Starting scheduled weather update ===")
        
        # Get all available sites from the collector
        sites = self.collector.sites.keys()
        
        success_count = 0
        total_sites = len(sites)
        
        for site_code in sites:
            try:
                print(f"Updating weather data for {site_code}...")
                
                # Update current weather
                current_data = self.collector.get_current_weather(site_code)
                if current_data:
                    self.cache.cache_current_weather(site_code, current_data, cache_duration_minutes=10)
                    print(f"[OK] Current weather updated for {site_code}")
                else:
                    print(f"[FAIL] Failed to fetch current weather for {site_code}")
                
                # Update forecast data
                forecast_data = self.collector.get_forecast(site_code, days=5)
                if forecast_data:
                    self.cache.cache_forecast_weather(site_code, forecast_data, cache_duration_minutes=30)
                    print(f"[OK] Forecast data updated for {site_code}")
                else:
                    print(f"[FAIL] Failed to fetch forecast data for {site_code}")
                
                if current_data or forecast_data:
                    success_count += 1
                    
            except Exception as e:
                print(f"[ERROR] Error updating weather data for {site_code}: {e}")
        
        # Clean up expired cache entries
        self.cache.cleanup_expired_data()
        
        print(f"=== Weather update completed: {success_count}/{total_sites} sites updated ===")
        
        return {
            'timestamp': datetime.now().isoformat(),
            'total_sites': total_sites,
            'successful_updates': success_count,
            'success_rate': success_count / total_sites if total_sites > 0 else 0
        }
    
    def get_service_status(self) -> Dict:
        """Get service status including cache information"""
        cache_status = self.cache.get_cache_status()
        
        return {
            'service_name': 'WeatherService',
            'cache_enabled': True,
            'api_mode': 'live' if not self.collector.mock_mode else 'mock',
            'available_sites': list(self.collector.sites.keys()),
            'cache_status': cache_status,
            'last_status_check': datetime.now().isoformat()
        }
    
    def force_refresh_site(self, site_code: str) -> bool:
        """Force refresh weather data for a specific site"""
        try:
            # Clear cache for this site
            self.cache.force_refresh_site(site_code)
            
            # Fetch fresh data
            current_data = self.get_current_weather(site_code, force_refresh=True)
            forecast_data = self.get_forecast_weather(site_code, force_refresh=True)
            
            return current_data is not None or forecast_data is not None
        except Exception as e:
            print(f"Error forcing refresh for {site_code}: {e}")
            return False