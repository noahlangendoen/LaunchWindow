import requests
import pandas as pd
import time
import os
import re
from datetime import datetime
from urllib.parse import urljoin, urlparse
import json

class NASADataCollector:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

        self.sources = {
            'nasa_launches': 'https://www.nasa.gov/launchschedule',
            'launch_library': 'https://ll.thespacedevs.com/2.2.0/launch/',
            'nasa_lsp': 'https://www.nasa.gov/content/launch-services-program'
        }

    
    def scrape_launch_library_api(self):
        """Scrape launch data from the Launch Library API."""
        launches = []

        try:
            # Get recent launches
            for status in ['3', '4']:
                response = self.session.get(self.sources['launch_library'], params={'mode': 'detailed', 'status': status, 'limit': 100})
                response.raise_for_status()

                data = response.json()

                for launch in data.get('results', []):
                    launch_data = self.parse_launch_data(launch)
                    if launch_data:
                        launches.append(launch_data)
                
                time.sleep(1)  # Respect API rate limits

        except Exception as e:
            print(f"Error scraping Launch Library API: {e}")
            
        return launches
    
    def parse_launch_data(self, launch):
        """Parse Launch Library API data."""
        try:
            # Add null check for mission
            mission = launch.get('mission') or {}
            weather_keywords = self.extract_weather_keywords(mission.get('description', ''))
            status = launch.get('status') or {}
            rocket_config = launch.get('rocket', {}).get('configuration') or {}
            pad = launch.get('pad') or {}
            pad_location = pad.get('location') or {}
            launch_provider = launch.get('launch_service_provider') or {}

            launch_data = {
                'source': 'launch_library',
                'collection_date': datetime.now().isoformat(),

                # Basic launch details
                'name': launch.get('name', ''),
                'status': status.get('name', ''),
                'net_time': launch.get('net', ''),
                'window_start': launch.get('window_start', ''),
                'window_end': launch.get('window_end', ''),

                # Launch provider and rocket
                'launch_provider': launch_provider.get('name', ''),
                'rocket_name': rocket_config.get('full_name', ''),
                'rocket_family': rocket_config.get('family', ''),

                # Launch location
                'pad_name': pad.get('name', ''),
                'pad_location': pad_location.get('name', ''),
                'pad_latitude': pad.get('latitude', ''),
                'pad_longitude': pad.get('longitude', ''),

                # Mission information
                'mission_name': mission.get('name', ''),
                'mission_description': mission.get('description', ''),
                'mission_type': mission.get('type', ''),

                # Weather information
                'weather_keywords': ', '.join(weather_keywords) if weather_keywords else '',
                'has_weather_delay': 'weather' in status.get('name', '').lower(),

                # Payload information
                'payload_count': len(rocket_config.get('payloads', [])),

                # URLs and references
                'launch_library_url': launch.get('url', ''),
                'video_url': launch.get('vid_urls', [{}])[0].get('url', '') if launch.get('vid_urls') else '',

                # Additional metadata
                'launch_library_id': launch.get('id', ''),
                'last_updated': launch.get('updated', '')
            }

            return launch_data
        
        except Exception as e:
            print(f"Error parsing launch data: {e}")
            return None
    
    def extract_weather_keywords(self, description):
        """Extract weather-related keywords from the mission description."""
        if not description:
            return []
        
        weather_keywords = [
            'weather', 'wind', 'rain', 'storm', 'cloud', 'fog', 'lightning',
            'delay', 'postpone', 'scrub', 'abort', 'hold', 'atmospheric',
            'precipitation', 'visibility', 'conditions', 'forecast'
        ]

        lower_desc = description.lower()
        found_keywords = []

        for keyword in weather_keywords:
            if keyword in lower_desc:
                found_keywords.append(keyword)

        return found_keywords
    
    def scrape_nasa_launch_schedule(self):
        """Scrape NASA's official launch schedule page."""
        launches = []

        try:
            print("Scraping NASA Launch Schedule...")

            response = self.session.get(self.sources['nasa_launches'])
            response.raise_for_status()

        except Exception as e:
            print(f"Error scraping NASA Launch Schedule: {e}")
        
        return launches
    
    def get_historical_nasa_missions(self):
        """Get historical NASA missions data from Launch Library"""
        missions = []

        try:
            print("Fetching historical NASA missions...")

            params = {
                'mode': 'detailed',
                'lsp_name_icontains': 'nasa',
                'ordering': '-net',
                'limit': 50
            }

            response = self.session.get(self.sources['launch_library'], params=params)
            response.raise_for_status()

            data = response.json()

            for launch in data.get('results', []):
                launch_data = self.parse_nasa_mission_data(launch)
                if launch_data:
                    missions.append(launch_data)
            
            next_url = data.get('next')
            page_count = 1

            while next_url and page_count < 5:
                response = self.session.get(next_url)
                response.raise_for_status()

                data = response.json()

                for launch in data.get('results', []):
                    launch_data = self.parse_nasa_mission_data(launch)
                    if launch_data:
                        missions.append(launch_data)

                next_url = data.get('next')
                page_count += 1
                time.sleep(1) # Rate limiting

        except Exception as e:
            print(f"Error fetching historical NASA missions: {e}")

        return missions
    
    def parse_nasa_mission_data(self, launch):
        """Parse specific NASA mission data."""
        try:
            provider = launch.get('launch_service_provider', {}).get('name', '').lower()
            if 'nasa' not in provider:
                return None
            
            mission_data = {
                'source': 'nasa_historical',
                'collection_date': datetime.now().isoformat(),

                # Mission identification
                'mission_name': launch.get('mission', {}).get('name', ''),
                'flight_number': launch.get('flight_number', ''),
                'launch_library_id': launch.get('id', ''),

                # Timing and status
                'launch_date': launch.get('net', ''),
                'window_start': launch.get('window_start', ''),
                'window_end': launch.get('window_end', ''),
                'status': launch.get('status', {}).get('name', ''),
                'status_abbr': launch.get('status', {}).get('abbrev', ''),
                'success': self.determine_success(launch.get('status', {}).get('abbrev', '')),

                # Rocket information
                'rocket_name': launch.get('rocket', {}).get('configuration', {}).get('full_name', ''),
                'rocket_family': launch.get('rocket', {}).get('configuration', {}).get('family', ''),
                'rocket_variant': launch.get('rocket', {}).get('configuration', {}).get('variant', ''),

                # Launch site
                'pad_name': launch.get('pad', {}).get('name', ''),
                'location_name': launch.get('pad', {}).get('location', {}).get('name', ''),
                'country_code': launch.get('pad', {}).get('location', {}).get('country_code', ''),

                # Mission details
                'mission_description': launch.get('mission', {}).get('description', ''),
                'mission_type': launch.get('mission', {}).get('type', ''),
                'orbit': launch.get('mission', {}).get('orbit', {}).get('name', '') if launch.get('mission', {}).get('orbit') else '',

                # Agency information
                'launch_service_provider': launch.get('launch_service_provider', {}).get('name', ''),
                'agency_type': launch.get('launch_service_provider', {}).get('type', ''),

                # Weather analysis
                'weather_mentioned': self.check_weather_mentions(launch),
                'hold_reason': launch.get('hold_reason', ''),
                'fail_reason': launch.get('fail_reason', ''),

                # References and URLs
                'info_urls': json.dumps([url.get('url', '') for url in launch.get('infoURLs', [])]),
                'video_urls': json.dumps([url.get('url', '') for url in launch.get('vidURLs', [])]),

                # Image data
                'image_urls': launch.get('image', ''),

                # Program information
                'programs': json.dumps([program.get('name', '') for program in launch.get('programs', [])] if launch.get('programs') else [])
            }

            return mission_data
        
        except Exception as e:
            print(f"Error parsing NASA mission data: {e}")
            return None
        
    def determine_success(self, status_abbr):
        """Determine if a launch was successful."""
        if not status_abbr:
            return None
        status_lower = status_abbr.lower()

        if 'success' in status_lower:
            return True
        elif any(term in status_lower for term in ['failure', 'failed']):
            return False
        else:
            return None
        
    def check_weather_mentions(self, launch):
        """Check if weather is mentioned in launch information."""
        text_fields = [
            launch.get('mission', {}).get('description', ''),
            launch.get('hold_reason', ''),
            launch.get('fail_reason', '')
        ]

        weather_terms = ['weather', 'wind', 'rain', 'storm', 'cloud', 'atmospheric']

        for text in text_fields:
            if text:
                lower_text = text.lower()
                if any(term in lower_text for term in weather_terms):
                    return True
        
        return False
    
    def collect_all_launch_data(self):
        """Collect launch data from all sources."""
        all_launches = []

        print("Starting comprehensive launch data collection...")

        library_launches = self.scrape_launch_library_api()
        all_launches.extend(library_launches)

        nasa_missions = self.get_historical_nasa_missions()
        all_launches.extend(nasa_missions)

        return all_launches
    
    def save_launch_data_to_csv(self, launches, filename='comprehensive_launch_data'):
        """Save collected launch data to a CSV file."""
        if not launches:
            print("No launch data to save.")
            return
        
        os.makedirs('data/raw', exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'data/raw/{filename}_{timestamp}.csv'

        df = pd.DataFrame(launches)
        df.to_csv(filename, index=False)
        
        print(f"Launch data saved to {filename}")

def main():
    try:
        collector = NASADataCollector()
        all_launches = collector.collect_all_launch_data()

        if all_launches:
            collector.save_launch_data_to_csv(all_launches)

            nasa_only = [launch for launch in all_launches if 'nasa' in launch.get('source', '').lower()]
            if nasa_only:
                collector.save_launch_data_to_csv(nasa_only, filename='nasa_only_launch_data')
            
            weather_launches = [launch for launch in all_launches if launch.get('weather_keywords', False)]
            if weather_launches:
                collector.save_launch_data_to_csv(weather_launches, filename='weather_related_launches')
            
        return True

    except Exception as e:
        print(f"An error occurred: {e}")
        return False
    

if __name__ == "__main__":
    main()