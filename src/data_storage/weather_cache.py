import sqlite3
import json
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import threading

class WeatherCache:
    """SQLite-based weather data cache for launch prediction system"""
    
    def __init__(self, db_path: str = "data/weather_cache.db"):
        self.db_path = db_path
        self._lock = threading.Lock()
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize the database schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS weather_current (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    site_code TEXT NOT NULL,
                    data TEXT NOT NULL,
                    cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL,
                    UNIQUE(site_code)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS weather_forecast (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    site_code TEXT NOT NULL,
                    forecast_time TEXT NOT NULL,
                    data TEXT NOT NULL,
                    cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL,
                    UNIQUE(site_code, forecast_time)
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_weather_current_expires 
                ON weather_current(expires_at)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_weather_forecast_expires 
                ON weather_forecast(expires_at)
            """)
            
            conn.commit()
    
    def get_current_weather(self, site_code: str) -> Optional[Dict]:
        """Get cached current weather data if not expired"""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute("""
                        SELECT data FROM weather_current 
                        WHERE site_code = ? AND expires_at > datetime('now')
                    """, (site_code,))
                    
                    row = cursor.fetchone()
                    if row:
                        return json.loads(row[0])
                    return None
            except Exception as e:
                print(f"Error retrieving cached weather for {site_code}: {e}")
                return None
    
    def cache_current_weather(self, site_code: str, weather_data: Dict, cache_duration_minutes: int = 10):
        """Cache current weather data"""
        with self._lock:
            try:
                expires_at = datetime.now() + timedelta(minutes=cache_duration_minutes)
                
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO weather_current 
                        (site_code, data, expires_at) 
                        VALUES (?, ?, ?)
                    """, (site_code, json.dumps(weather_data), expires_at))
                    
                    conn.commit()
                    print(f"Cached weather data for {site_code} (expires: {expires_at})")
            except Exception as e:
                print(f"Error caching weather data for {site_code}: {e}")
    
    def get_forecast_weather(self, site_code: str) -> Optional[List[Dict]]:
        """Get cached forecast weather data if not expired"""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute("""
                        SELECT data FROM weather_forecast 
                        WHERE site_code = ? AND expires_at > datetime('now')
                        ORDER BY forecast_time
                    """, (site_code,))
                    
                    rows = cursor.fetchall()
                    if rows:
                        return [json.loads(row[0]) for row in rows]
                    return None
            except Exception as e:
                print(f"Error retrieving cached forecast for {site_code}: {e}")
                return None
    
    def cache_forecast_weather(self, site_code: str, forecast_data: List[Dict], cache_duration_minutes: int = 30):
        """Cache forecast weather data"""
        with self._lock:
            try:
                expires_at = datetime.now() + timedelta(minutes=cache_duration_minutes)
                
                with sqlite3.connect(self.db_path) as conn:
                    # Clear existing forecast data for this site
                    conn.execute("""
                        DELETE FROM weather_forecast WHERE site_code = ?
                    """, (site_code,))
                    
                    # Insert new forecast data
                    for forecast_item in forecast_data:
                        forecast_time = forecast_item.get('forecast_time', datetime.now().isoformat())
                        conn.execute("""
                            INSERT INTO weather_forecast 
                            (site_code, forecast_time, data, expires_at) 
                            VALUES (?, ?, ?, ?)
                        """, (site_code, forecast_time, json.dumps(forecast_item), expires_at))
                    
                    conn.commit()
                    print(f"Cached {len(forecast_data)} forecast items for {site_code} (expires: {expires_at})")
            except Exception as e:
                print(f"Error caching forecast data for {site_code}: {e}")
    
    def cleanup_expired_data(self):
        """Remove expired cache entries"""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    # Clean up expired current weather
                    cursor = conn.execute("""
                        DELETE FROM weather_current 
                        WHERE expires_at <= datetime('now')
                    """)
                    current_deleted = cursor.rowcount
                    
                    # Clean up expired forecast weather
                    cursor = conn.execute("""
                        DELETE FROM weather_forecast 
                        WHERE expires_at <= datetime('now')
                    """)
                    forecast_deleted = cursor.rowcount
                    
                    conn.commit()
                    
                    if current_deleted > 0 or forecast_deleted > 0:
                        print(f"Cleaned up {current_deleted} expired current weather entries, {forecast_deleted} expired forecast entries")
            except Exception as e:
                print(f"Error cleaning up expired data: {e}")
    
    def get_cache_status(self) -> Dict:
        """Get cache status information"""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    # Count current weather entries
                    cursor = conn.execute("""
                        SELECT COUNT(*) FROM weather_current 
                        WHERE expires_at > datetime('now')
                    """)
                    current_count = cursor.fetchone()[0]
                    
                    # Count forecast entries
                    cursor = conn.execute("""
                        SELECT COUNT(*) FROM weather_forecast 
                        WHERE expires_at > datetime('now')
                    """)
                    forecast_count = cursor.fetchone()[0]
                    
                    # Get sites with cached data
                    cursor = conn.execute("""
                        SELECT DISTINCT site_code FROM weather_current 
                        WHERE expires_at > datetime('now')
                    """)
                    cached_sites = [row[0] for row in cursor.fetchall()]
                    
                    return {
                        'current_weather_entries': current_count,
                        'forecast_entries': forecast_count,
                        'cached_sites': cached_sites,
                        'last_updated': datetime.now().isoformat()
                    }
            except Exception as e:
                print(f"Error getting cache status: {e}")
                return {'error': str(e)}
    
    def force_refresh_site(self, site_code: str):
        """Force refresh of cached data for a specific site"""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        DELETE FROM weather_current WHERE site_code = ?
                    """, (site_code,))
                    
                    conn.execute("""
                        DELETE FROM weather_forecast WHERE site_code = ?
                    """, (site_code,))
                    
                    conn.commit()
                    print(f"Forced refresh for site {site_code} - cache cleared")
            except Exception as e:
                print(f"Error forcing refresh for {site_code}: {e}")