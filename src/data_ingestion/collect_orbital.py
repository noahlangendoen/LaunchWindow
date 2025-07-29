import requests
from datetime import datetime
import pandas as pd
import os
import re
import time

class TLECollector:
    def __init__(self):
        self.session = requests.Session()

        # User Agent
        self.session.headers.update({
            'User-Agent': 'RocketLaunchPredictor/1.0 (Educational Project)'
        })

        # Common TLE Data Sources

        self.sources = {
            'celestrak_stations': 'https://celestrak.org/NORAD/elements/gp.php?GROUP=stations&FORMAT=tle',
            'celestrak_active': 'https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle',
            'celestrak_weather': 'https://celestrak.org/NORAD/elements/gp.php?GROUP=weather&FORMAT=tle',
            'celestrak_navigation': 'https://celestrak.org/NORAD/elements/gp.php?GROUP=gps-ops&FORMAT=tle',
        }
    
    def parse_tle(self, tle_data, source_name):
        """Parses TLE data and returns the data."""
        lines = tle_data.strip().split('\n')
        tle_records = []

        i = 0

        while i < len(lines):
            
            # Skip empty lines
            if not lines[i].strip():
                i += 1
                continue

            # Look for TLE Format
            if i + 2 < len(lines):
                name_line = lines[i].strip()
                line1 = lines[i + 1].strip()
                line2 = lines[i + 2].strip()

                # Validate TLE Format
                if (line1.startswith('1 ') and len(line1) == 69 and
                    line2.startswith('2 ') and len(line2) == 69):

                    try:
                        tle_record = self.parse_tle_record(name_line, line1, line2, source_name)
                        if tle_record:
                            tle_records.append(tle_record)
                    except Exception as e:
                        print(f"Error parsing TLE record: {e}")
                    i += 3
                else:
                    i += 1
            else:
                break
        
        return tle_records
    
    def parse_tle_record(self, name, line1, line2, source_name):
        """Parse a single TLE record."""
        try:
            # Parse line1
            sat_num = int(line1[2:7].strip())
            classification = line1[7]
            launch_year = int(line1[9:11]) + (2000 if int(line1[9:11]) < 57 else 1900) # Adjust for 21st century
            launch_num = int(line1[11:14])
            launch_piece = line1[14:17].strip()
            epoch_year = int(line1[18:20]) + (2000 if int(line1[18:20]) < 57 else 1900) # Adjust for 21st century
            epoch_day = float(line1[20:32].strip())
            mean_motion_derivative = float(line1[33:43].strip()) if line1[33:43].strip() else 0.0
            mean_motion_second_derivative = float(line1[44:52].strip()) if line1[44:52].strip() else 0.0
            drag_term = float(line1[53:61].strip()) if line1[53:61].strip() else 0.0
            element_set_num = int(line1[64:68])

            # Parse line2
            inclination = float(line2[8:16].strip())
            right_ascension = float(line2[17:25])
            eccentricity = float('0.' + line2[26:33])
            argument_of_perigee = float(line2[34:42])
            mean_anomaly = float(line2[43:51])
            mean_motion = float(line2[52:63])
            revolution_number = int(line2[63:68])

            # Calculate approximate orbital period in minutes
            day_len_mins = 1440.0
            earth_gravity_constant = 398600.4418  # km^3/s^2
            orbital_period = day_len_mins / mean_motion if mean_motion else None

            # Roughly estimate altitude using Kelper's Third Law
            earth_radius_km = 6371.0
            semi_major_axis = ((day_len_mins / (2 * 3.14159 * mean_motion)) ** (2/3)) * (earth_gravity_constant ** (1/3)) if mean_motion > 0 else 0
            altitude_km = semi_major_axis - earth_radius_km if semi_major_axis > earth_radius_km else 0

            tle_record = {
                'name': name,
                'source': source_name,
                'collection_date': datetime.now().isoformat(),

                # Satellite Information
                'satellite_number': sat_num,
                'classification': classification,

                # Launch Information
                'launch_year': launch_year,
                'launch_number': launch_num,
                'launch_piece': launch_piece,

                # Epoch Information
                'epoch_year': epoch_year,
                'epoch_day': epoch_day,

                # Orbital Elements
                'inclination_deg': inclination,
                'right_ascension_deg': right_ascension,
                'eccentricity': eccentricity,
                'argument_of_perigee_deg': argument_of_perigee,
                'mean_anomaly_deg': mean_anomaly,
                'mean_motion': mean_motion,

                # Calculated Values
                'orbital_period_minutes': orbital_period,
                'approximate_altitude_km': altitude_km,

                # Derivatives
                'mean_motion_derivative': mean_motion_derivative,
                'mean_motion_second_derivative': mean_motion_second_derivative,
                'drag_term': drag_term,

                # Numbers
                'element_set_number': element_set_num,
                'revolution_number': revolution_number,

                # Raw Elements
                'line1': line1,
                'line2': line2
            }

            return tle_record
        
        except (ValueError, IndexError) as e:
            print(f"Error parsing TLE record: {e}")
            return None
        
    def fetch_tle_data(self, source_name, url):
        """Fetch TLE data from a specific source."""
        try:
            print(f"Fetching TLE data from {source_name}...")

            response = self.session.get(url, timeout=60)
            response.raise_for_status()

            tle_records = self.parse_tle(response.text, source_name)
            print(f"Fetched {len(tle_records)} TLE records from {source_name}.")

            return tle_records
        
        except requests.exceptions.RequestException as e:
            print(f"Error fetching TLE data from {source_name}: {e}")
            return []
        
    def collect_all_tle_data(self):
        """Collect TLE data from all sources."""
        all_tle_records = []

        print("Starting TLE data collection...")

        for source_name, url in self.sources.items():
            tle_data = self.fetch_tle_data(source_name, url)
            all_tle_records.extend(tle_data)

            # Sleep to avoid overwhelming the server
            time.sleep(2)

        print(f"Collected {len(all_tle_records)} rows of TLE data.")

        return all_tle_records
    
    def filter_most_important_satellites(self, tle_records):
        """Filter most important satellites for launch operations."""
        satellites = [
            'ISS', 'SPACE STATION', 'DRAGON', 'FALCON', 'STARSHIP', 
            'CREW', 'CARGO', 'STARLINK', 'GPS', 'WEATHER', 'HUBBLE',
            'TELESCOPE', 'LANDSAT'
        ]

        filtered_records = []

        for record in tle_records:
            name = record['name'].upper()

            # Check if satellite meets importance criteria
            if any(keyword in name for keyword in satellites):
                record['category'] = 'important'
                filtered_records.append(record)
            elif 'WEATHER' in record['source'] or record['approximate_altitude_km'] > 35000:
                record['category'] = 'operational'
                filtered_records.append(record)

        return filtered_records
    
    def save_tle_data_to_csv(self, tle_records, output_file='tle_data'):
        """Save TLE data to a CSV file."""
        if not tle_records:
            print("No TLE records to save.")
            return
        
        # Ensure the output directory exists
        os.makedirs("data/raw", exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"data/raw/{output_file}_{timestamp}.csv"

        df = pd.DataFrame(tle_records)
        df.to_csv(filepath, index=False)

        print(f"TLE data saved to {filepath}")


def main():
    try:
        collector = TLECollector()

        # Collect TLE data from all sources
        tle_records = collector.collect_all_tle_data()

        if tle_records:
            collector.save_tle_data_to_csv(tle_records)
            print("TLE data collection completed successfully.")

            # Filter important satellites
            important_satellites = collector.filter_most_important_satellites(tle_records)
            if important_satellites:
                collector.save_tle_data_to_csv(important_satellites, output_file='important_tle_data')
                print("Important TLE data saved successfully.")
        
        return True
    
    except Exception as e:
        print(f"An error occurred during TLE data collection: {e}")
        return False


if __name__ == "__main__":
    main()