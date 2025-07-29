import requests
import os
import time
import pandas as pd
from datetime import datetime
import json

class spaceXLaunchCollector:
    def __init__(self):
        self.base_url = "https://api.spacexdata.com/v4"
        self.session = requests.Session()

    def get_all_launches(self):
        """Fetch all SpaceX launches."""
        try:
            print("Fetching all SpaceX launches...")

            launches_url = f"{self.base_url}/launches/query"

            query = {
                "query": {},
                "options": {
                    "populate": [
                        "rocket",
                        "payloads",
                        "launchpad",
                        "crew",
                        "capsules"
                    ],
                    "pagination": False
                }
            }

            response = self.session.post(launches_url, json=query)
            response.raise_for_status()

            data = response.json()
            launches = data.get('docs', [])

            print(f"Total launches fetched: {len(launches)}")
            return self.process_launches(launches)
        
        except requests.RequestException as e:
            print(f"Error fetching launches: {e}")
            return []
        
    def process_launches(self, launches):
        """Process and format the fetched launches data."""
        processed_launches = []
        
        
        for launch in launches:
            try:
                launch_data = {
                    'flight_number': launch.get('flight_number'),
                    'name': launch.get('name'),
                    'date_utc': launch.get('date_utc'),
                    'date_local': launch.get('date_local'),
                    'success': launch.get('success'),
                    'upcoming': launch.get('upcoming'),
                    'details': launch.get('details'),

                    # Rocket information
                    'rocket_name': launch.get('rocket', {}).get('name') if launch.get('rocket') else '',
                    'rocket_type': launch.get('rocket', {}).get('type') if launch.get('rocket') else '',

                    # Launchpad information
                    'launchpad_name': launch.get('launchpad', {}).get('full_name') if launch.get('launchpad') else '',
                    'launchpad_locality': launch.get('launchpad', {}).get('locality') if launch.get('launchpad') else '',
                    'launchpad_region': launch.get('launchpad', {}).get('region') if launch.get('launchpad') else '',

                    # Failure information
                    'failures': launch.get('failures', []),

                    # Links
                    'webcast': launch.get('links', {}).get('webcast'),
                    'wikipedia': launch.get('links', {}).get('wikipedia'),

                    # Collect information
                    'collected_at': datetime.now().isoformat()
                }

                payloads = launch.get('payloads', [])
                if payloads:
                    payload_masses = []
                    payload_types = []
                    for payload in payloads:
                        if isinstance(payload, dict):
                            if payload.get('mass_kg'):
                                payload_masses.append(payload['mass_kg'])
                            if payload.get('type'):
                                payload_types.append(payload['type'])

                    launch_data['payload_mass_kg'] = sum(payload_masses) if payload_masses else None
                    launch_data['payload_types'] = json.dumps(list(set(payload_types)))

                else:
                    launch_data['payload_mass_kg'] = None
                    launch_data['payload_types'] = '[]'

                processed_launches.append(launch_data)
            
            except Exception as e:
                print(f"Error processing launch {launch.get('name', 'Unkown')}: {e}")
                continue

        return processed_launches
    
    def save_launches_to_csv(self, launches, filename='spacex_launches.csv'):
        """Save the processed launches data to a CSV file."""
        if not launches:
            print("No launches data to save.")
            return
        
        df = pd.DataFrame(launches)

        # Ensure the data directory exists
        os.makedirs("data/raw", exist_ok=True)
        
        filepath = f"data/raw/{filename}"

        df.to_csv(filepath, index=False)
        print(f"Launches data saved to {filepath}")

def main():
    collector = spaceXLaunchCollector()
    launches = collector.get_all_launches()

    if launches:
        collector.save_launches_to_csv(launches)
        return True
    else:
        print("No launches data collected.")
        return False
    
if __name__ == "__main__":
    main()