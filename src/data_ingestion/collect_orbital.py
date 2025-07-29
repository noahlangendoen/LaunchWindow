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
    
    