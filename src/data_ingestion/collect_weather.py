import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import os
from dotenv import load_dotenv

load_dotenv()

class WeatherCollector:
    def __init__(self):
        self.api_key = os.getenv('OPENWEATHER_API_KEY')
        self.mock_mode = not self.api_key
        
        if self.mock_mode:
            print("WARNING: OpenWeather API key not found. Using mock weather data.")
        else:
            print("OpenWeather API key found. Using live weather data.")
        
        self.base_url = "https://api.openweathermap.org/data/2.5"

        self.session = requests.Session()

        self.sites = {
            "KSC": {"name": "Kennedy Space Center", "lat": 28.5721, "lon": -80.6480, 'timezone': 'America/New_York'},
            "VSFB": {"name": "Vandenberg Space Force Base", "lat": 34.7420, "lon": -120.5724, 'timezone': 'America/Los_Angeles'},
            "CCAFS": {"name": "Cape Canaveral Space Force Station", "lat": 28.3922, "lon": -80.6077, 'timezone': 'America/New_York'},
        }

    def get_current_weather(self, site_key):
        """Gather current weather data for a specific site."""
        if self.mock_mode:
            return self._get_mock_weather(site_key)
            
        site = self.sites[site_key]

        try:
            params = {
                'lat': site['lat'],
                'lon': site['lon'],
                'appid': self.api_key,
                'units': 'metric'
            }

            response = self.session.get(f"{self.base_url}/weather", params=params)
            response.raise_for_status()

            data = response.json()

            weather_data = {
                'site_code': site_key,
                'site_name': site['name'],
                'dt': datetime.fromtimestamp(data['dt']).isoformat(),

                # Temperature Data
                'temperature_c': data['main']['temp'],
                'feels_like_c': data['main']['feels_like'],
                'temp_min_c': data['main']['temp_min'],
                'temp_max_c': data['main']['temp_max'],

                # Humidity and Pressure
                'humidity_percent': data['main']['humidity'],
                'pressure_hpa': data['main']['pressure'],

                # Wind Data
                'wind_speed_ms': data.get('wind', {}).get('speed', 0),
                'wind_direction_deg': data.get('wind', {}).get('deg', 0),
                'wind_gust_ms': data.get('wind', {}).get('gust', 0),

                # Weather Conditions
                'weather_main': data['weather'][0]['main'],
                'weather_description': data['weather'][0]['description'],

                # Visibility and Cloudiness
                'visibility_m': data.get('visibility', 0),
                'cloud_cover_percent': data.get('clouds', {}).get('all', 0),

                # Precipitation Data
                'rain_1h_mm': data.get('rain', {}).get('1h', 0),
                'rain_3h_mm': data.get('rain', {}).get('3h', 0),
                'snow_1h_mm': data.get('snow', {}).get('1h', 0),
                'snow_3h_mm': data.get('snow', {}).get('3h', 0),

                # Location and Timezone
                'latitude': site['lat'],
                'longitude': site['lon'],
                'timezone': site['timezone']
            }

            return weather_data
        except requests.exceptions.RequestException as e:
            print(f"Error fetching current weather for {site_key}: {e}")
            return None
        
    def get_forecast(self, site_key, days=5):
        """Gather a 5-day weather forecast for a specific site."""
        if self.mock_mode:
            return self._get_mock_forecast(site_key, days)
            
        site = self.sites[site_key]

        try:
            params = {
                'lat': site['lat'],
                'lon': site['lon'],
                'appid': self.api_key,
                'units': 'metric'
            }

            response = self.session.get(f"{self.base_url}/forecast", params=params)
            response.raise_for_status()

            data = response.json()
            forecast_data = []

            for item in data['list']:
                forecast_item = {
                    'site_code': site_key,
                    'site_name': site['name'],
                    'forecast_time': datetime.fromtimestamp(item['dt']).isoformat(),
                    'collected_at': datetime.now().isoformat(),

                    # Temperature Forecast Data
                    'temperature_c': item['main']['temp'],
                    'feels_like_c': item['main']['feels_like'],
                    'temp_min_c': item['main']['temp_min'],
                    'temp_max_c': item['main']['temp_max'],

                    # Humidity and Pressure
                    'humidity_percent': item['main']['humidity'],
                    'pressure_hpa': item['main']['pressure'],

                    # Wind Forecast Data
                    'wind_speed_ms': item.get('wind', {}).get('speed', 0),
                    'wind_direction_deg': item.get('wind', {}).get('deg', 0),
                    'wind_gust_ms': item.get('wind', {}).get('gust', 0),

                    # Weather Conditions
                    'weather_main': item['weather'][0]['main'],
                    'weather_description': item['weather'][0]['description'],

                    # Precipitation Data
                    'rain_3h_mm': item.get('rain', {}).get('3h', 0),
                    'snow_3h_mm': item.get('snow', {}).get('3h', 0),
                    'precipitation_probability_percent': item.get('pop', 0) * 100,

                    # Cloudiness
                    'cloud_cover_percent': item.get('clouds', {}).get('all', 0),

                    # Location and Timezone
                    'latitude': site['lat'],
                    'longitude': site['lon'],
                    'timezone': site['timezone']
                }

                forecast_data.append(forecast_item)

            return forecast_data
        
        except requests.exceptions.RequestException as e:
            print(f"Error fetching forecast for {site_key}: {e}")
            return []
        
    def collect_weather_data(self):
        """Collect current weather data for all sites."""
        weather_data = []
        print("Collecting current weather data...")

        for key in self.sites.keys():
            print(f"Fetching data for {self.sites[key]['name']}...")
            data = self.get_current_weather(key)
            if data:
                weather_data.append(data)
                print(f"Data for {self.sites[key]['name']} collected successfully.")
            else:
                print(f"Failed to fetch data for {self.sites[key]['name']}")

            time.sleep(1)
        return weather_data
    
    def collect_forecast_data(self):
        """Collect 5-day weather forecast data for all sites."""
        forecast_data = []
        print("Collecting 5-day weather forecast data for all sites...")

        for key in self.sites.keys():
            print(f"Fetching forecast data for {self.sites[key]['name']}...")
            data = self.get_forecast(key)
            if data:
                forecast_data.extend(data)
                print(f"Forecast data for {self.sites[key]['name']} collected successfully.")
            else:
                print(f"Failed to fetch forecast data for {self.sites[key]['name']}")

            time.sleep(1)

        return forecast_data
    
    def save_to_csv(self, data, filename):
        """Save collected data to a CSV file."""
        if not data:
            print("No data to save.")
            return
        
        # Ensure the data directory exists
        os.makedirs("data/raw", exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"data/raw/{filename}_{timestamp}.csv"

        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)

        print(f"{len(data)} records saved to {filepath}.")

        if 'site_code' in df.columns:
            print(F"Sites covered: {', '.join(df['site_code'].unique())}")

    def _get_mock_weather(self, site_key):
        """Return mock weather data when API key is not available."""
        import random
        
        site = self.sites.get(site_key, {"name": "Unknown Site"})
        
        return {
            'site_code': site_key,
            'site_name': site['name'],
            'dt': datetime.now().isoformat(),
            'temperature_c': 20 + random.uniform(-5, 10),
            'feels_like_c': 20 + random.uniform(-5, 10),
            'temp_min_c': 15 + random.uniform(-3, 5),
            'temp_max_c': 25 + random.uniform(-5, 10),
            'humidity_percent': 60 + random.uniform(-20, 30),
            'pressure_hpa': 1013 + random.uniform(-20, 20),
            'visibility_m': 10000 + random.uniform(-2000, 0),
            'wind_speed_ms': 5 + random.uniform(-3, 10),
            'wind_direction_deg': random.uniform(0, 360),
            'wind_gust_ms': 7 + random.uniform(-3, 15),
            'cloud_cover_percent': random.uniform(0, 100),
            'weather_main': 'Clear',
            'weather_description': 'clear sky',
            'rain_1h_mm': 0,
            'rain_3h_mm': 0,
            'snow_1h_mm': 0,
            'snow_3h_mm': 0,
            'latitude': site.get('lat', 0),
            'longitude': site.get('lon', 0),
            'timezone': site.get('timezone', 'UTC'),
            'collected_at': datetime.now().isoformat()
        }


    def _get_mock_forecast(self, site_key, days=5):
        """Return mock forecast data."""
        import random
        
        site = self.sites.get(site_key, {"name": "Unknown Site"})
        forecast_data = []
        
        for i in range(days * 8):  # 8 forecasts per day (3-hour intervals)
            forecast_time = datetime.now() + timedelta(hours=i * 3)
            
            forecast_item = {
                'site_code': site_key,
                'site_name': site['name'],
                'forecast_time': forecast_time.isoformat(),
                'collected_at': datetime.now().isoformat(),
                'temperature_c': 20 + random.uniform(-5, 10),
                'feels_like_c': 20 + random.uniform(-5, 10),
                'temp_min_c': 15 + random.uniform(-3, 5),
                'temp_max_c': 25 + random.uniform(-5, 10),
                'humidity_percent': 60 + random.uniform(-20, 30),
                'pressure_hpa': 1013 + random.uniform(-20, 20),
                'wind_speed_ms': 5 + random.uniform(-3, 10),
                'wind_direction_deg': random.uniform(0, 360),
                'wind_gust_ms': 7 + random.uniform(-3, 15),
                'cloud_cover_percent': random.uniform(0, 100),
                'weather_main': 'Clear',
                'weather_description': 'clear sky',
                'visibility_m': 10000 + random.uniform(-2000, 0),
                'pop': random.uniform(0, 0.3),
                'rain_3h_mm': 0,
                'snow_3h_mm': 0,
                'go_for_launch': True
            }
            forecast_data.append(forecast_item)
            
        return forecast_data

    def _get_real_forecast(self, site_key, days=5):
        """Original forecast method renamed."""

def main():
    try:
        collector = WeatherCollector()

        # Collect current weather data
        current_weather_data = collector.collect_weather_data()
        if current_weather_data:
            collector.save_to_csv(current_weather_data, "current_weather")

        # Collect 5-day weather forecast data
        forecast_data = collector.collect_forecast_data()
        if forecast_data:
            collector.save_to_csv(forecast_data, "weather_forecast")

        return True

    except Exception as e:
        print(f"An error occurred in weather collection: {e}")
        return False
    
if __name__ == "__main__":
    main()