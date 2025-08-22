import pandas as pd
import numpy as np
import os
import re
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

@dataclass
class DataQuality:
    """Class used to measure the quality of a dataset."""
    total_records: int
    missing_values: Dict[str, int]
    duplicate_records: int
    outliers: int
    completeness_score: float
    consistency_score: float
    overall_quality: str

class DataPreprocessor:
    """Class to do the data preprocessing for rocket launch project."""
    def __init__(self, data_dir="data/raw"):
        self.data_dir = data_dir
        self.processed_data = {}
        self.feature_mappings = {}
        self.quality_reports = {}

        # Standard feature engineering configurations

        self.launch_features = [
            # Rocket characteristics
            'rocket_name', 'rocket_family', 'rocket_type', 'flight_number',

            # Launch provider
            'launch_provider', 'launch_service_provider', 'agency_type',

            # Launch location
            'pad_name', 'pad_location', 'launchpad_name', 'launchpad_locality',
            'launchpad_region', 'country_code',
            'pad_latitude', 'pad_longitude',

            # Mission characteristics
            'mission_type', 'mission_name',
            'payload_mass_kg', 'payload_count', 'payload_types',
            'orbit',

            # Temporal features (to be engineered)
            'launch_year', 'launch_month', 'launch_hour', 
            'launch_day_of_week', 'launch_season'

            # Weather-related indicators from launch data
            'weather_keywords', 'has_weather_delay', 'weather_metnioned',
            'hold_reason', 'fail_reason',

            # Target variable
            'success'
        ]

        self.weather_features = [
            'temperature_c', 'pressure_hpa', 'humidity_percent',
            'wind_speed_ms', 'wind_direction_deg', 'wind_gust_ms',
            'cloud_cover_percent', 'visibility_m', 'rain_1h_mm',
            'rain_3h_mm', 'snow_1h_mm', 'precipitation_probability_percent',
            'weather_main', 'weather_description'
        ]

        self.orbital_features = [
            'orbital_object_count', 'leo_objects_count', 'meo_objects_count',
            'geo_objects_count', 'orbital_congestion_score'
        ]

        self.engineered_features = [
            'provider_experience', # Cumulative launches by provider
            'is_commerical', # Government vs commercial
            'is_reusable', # Reusable rocket technology
            'mission_complexity', # Derived from mission type
            'payload_cateogory', # Light/Medium/Heavy/Super Heavy
            'weather_wind_category', # Calm/Light/Moderate/Strong
            'days_since_last_launch', # Launch cadence
            'previous_success_rate' # Historical success rate for provider
        ]

    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """Load all available data files from the data directory"""
        data_files = {}

        if not os.path.exists(self.data_dir):
            print(f"The directory {self.data_dir} does not exist.")
            return data_files
        
        # Find all csv files
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.csv'):
                filepath = os.path.join(self.data_dir, filename)

                try:
                    df = pd.read_csv(filepath)
                    low_term = filename.lower()

                    if 'launch' in low_term:
                        if 'nasa' in low_term:
                            data_files['nasa_launches'] = df
                        elif 'spacex' in low_term:
                            data_files['spacex_launches'] = df
                        else:
                            data_files['launches'] = df
                    elif 'weather' in low_term:
                        if 'forecast' in low_term:
                            data_files['weather_forecast'] = df
                        else:
                            data_files['weather_current'] = df
                    elif 'tle' in low_term:
                        data_files['orbital_data'] = df
                    else:
                        # Generic categorization just in case
                        base_name = filename.replace('.csv', '').split('_')[0]
                        data_files[base_name] = df
                    
                    print(f"Loaded {filename}: {len(df)} records")
                
                except Exception as e:
                    print(f"Error loading {filename}: {e}")

        self.processed_data = data_files

        return data_files
    
    def analyze_data_quality(self, df: pd.DataFrame, dataset_name: str) -> DataQuality:
        """Analyze the quality of a dataset."""
        total_records = len(df)

        # Missing values
        missing_values = df.isnull().sum().to_dict()
        missing_values = {k: v for k, v in missing_values.items() if v > 0}

        # Duplicate records
        duplicate_records = df.duplicated().sum()

        # Outliers detection (using IQR for numeric columns)
        outliers_count = 0
        numeric_columns = df.select_dtypes(include=[np.number]).columns

        for column in numeric_columns:
            if df[column].notna().sum() > 0:
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers_count += ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()
        
        # Completeness score as percentage
        completeness_score = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100

        # Consistency score as int in range 0 to 100 (approx percentage)
        consistency_issues = 0
        for column in df.columns:
            if 'date' in column.lower() or 'time' in column.lower():
                # Check valid date formats
                try:
                    pd.to_datetime(df[column], errors='coerce')
                except:
                    consistency_issues += 1
        
        consistency_score = max(0, 100 - consistency_issues * 10)

        # Overall quality assessment
        overall_score = (completeness_score + consistency_score) / 2
        if overall_score >= 90:
            overall_quality = 'EXCELLENT'
        elif overall_score >= 75:
            overall_quality = 'GOOD'
        elif overall_score >= 60:
            overall_quality = 'FAIR'
        else:
            overall_quality = 'POOR'

        quality_report = DataQuality(
            total_records=total_records,
            missing_values=missing_values,
            duplicate_records=duplicate_records,
            outliers=outliers_count,
            completeness_score=completeness_score,
            consistency_score=consistency_score,
            overall_quality=overall_quality
        )
    
        self.quality_reports[dataset_name] = quality_report
        return quality_report
    
    def clean_launch_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize launch data."""
        df_cleaned = df.copy()

        # Standardize date columns
        date_columns = [col for col in df_cleaned.columns if 'date' in col.lower() or 'time' in col.lower()]
        for col in date_columns:
            df_cleaned[col] = pd.to_datetime(df_cleaned[col], errors='coerce')
        
        # Standardize success indicators
        if 'success' in df_cleaned.columns:
            # Convert various success labels to boolean
            df_cleaned['success'] = df_cleaned['success'].map({
                True: True, False: False, 'true': True, 'false': False,
                'True': True, 'False': False, 1: True, 0: False,
                'SUCCESS': True, 'FAILURE': False, 'SUCCESS COMPLETE': True
            })

        # Clean launch provider names
        if 'launch_provider' in df_cleaned.columns:
            df_cleaned['launch_provider'] = df_cleaned['launch_provider'].str.strip().str.title()
            # Standardize common providers
            provider_mapping = {
                'Spacex': 'SpaceX',
                'Space X': 'SpaceX',
                'National Aeronautics and Space Administration': 'NASA',
                'Nasa': 'NASA',
                'United Launch Alliance': 'ULA'
            }
        
            df_cleaned['launch_provider'] = df_cleaned['launch_provider'].replace(provider_mapping)

        # Clean mission types
        if 'mission_type' in df_cleaned.columns:
            df_cleaned['mission_type'] = df_cleaned['mission_type'].str.strip().str.title()
            df_cleaned['mission_type'] = df_cleaned['mission_type'].fillna('Unknown')
        
        # Convert payload mass to numeric
        if 'payload_mass_kg' in df_cleaned.columns:
            df_cleaned['payload_mass_kg'] = pd.to_numeric(df_cleaned['payload_mass_kg'], errors='coerce')

        # Remove duplicates
        df_cleaned = df_cleaned.drop_duplicates()

        return df_cleaned
    
    def clean_weather_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize weather data."""
        df_cleaned = df.copy()

        # Convert temperature columns
        temp_columns = [col for col in df_cleaned.columns if 'temp' in col.lower()]
        for col in temp_columns:
            df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
        
        # Clean wind data
        wind_columns = [col for col in df_cleaned.columns if 'wind' in col.lower()]
        for col in wind_columns:
            df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
            # Remove negative wind speeds
            if 'speed' in col.lower():
                df_cleaned[col] = df_cleaned[col].clip(lower=0)
            # Normalize wind direciton to 0 to 360 degrees
            if 'direction' in col.lower():
                df_cleaned[col] = df_cleaned[col] % 360
        
        # Clean pressure data
        if 'pressure_hpa' in df_cleaned.columns:
            df_cleaned['pressure_hpa'] = pd.to_numeric(df_cleaned['pressure_hpa'], errors='coerce')
            # Remove unrealistic pressure values
            df_cleaned['pressure_hpa'] = df_cleaned['pressure_hpa'].clip(lower=800, upper=1100)
        
        # Clean humidity data
        if 'humidity_percent' in df_cleaned.columns:
            df_cleaned['humidity_percent'] = pd.to_numeric(df_cleaned['humidity_percent'], errors='coerce')
            df_cleaned['humidity_percent'] = df_cleaned['humidity_percent'].clip(lower=0, upper=100)
        
        # Standardize site codes
        if 'site_code' in df_cleaned.columns:
            df_cleaned['site_code'] = df_cleaned['site_code'].str.strip().str.upper()

        # Handle datetime columns
        datetime_columns = [col for col in df_cleaned.columns if any(keyword in col.lower() for keyword in ['dt', 'time', 'date'])]
        for col in datetime_columns:
            df_cleaned[col] = pd.to_datetime(df_cleaned[col], errors='coerce')

        return df_cleaned
    
    def clean_orbital_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize orbital/TLE data"""
        df_cleaned = df.copy()

        # Clean satellite names
        if 'name' in df_cleaned.columns:
            df_cleaned['name'] = df_cleaned['name'].str.strip().str.upper()

        # Convert orbital elements to numeric
        orbital_numeric_columns = [
            'inclination_deg', 'eccentricity', 'mean_motion', 'orbital_period_minutes',
            'approximate_altitude_km', 'right_ascension_deg', 'argument_of_perigee_deg'
        ]

        for col in orbital_numeric_columns:
            if col in df_cleaned.columns:
                df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')

        # Validate orbital elements ranges
        if 'inclination_deg' in df_cleaned.columns:
            df_cleaned['inclination_deg'] = df_cleaned['inclination_deg'].clip(lower=0, upper=180)

        if 'eccentricity' in df_cleaned.columns:
            df_cleaned['eccentricity'] = df_cleaned['eccentricity'].clip(lower=0, upper=0.99)

        if 'approximate_altitude_km' in df_cleaned.columns:
            df_cleaned['approximate_altitude_km'] = df_cleaned['approximate_altitude_km'].clip(lower=150, upper=50000)

        # Handle epoch data
        if 'epoch_year' in df_cleaned.columns and 'epoch_day' in df_cleaned.columns:
            try:
                df_cleaned['epoch_datetime'] = pd.to_datetime(
                    df_cleaned['epoch_year'].astype(str) + df_cleaned['epoch_day'].astype(str),
                    format='%Y%j', errors='coerce'
                )
            except:
                pass
        
        return df_cleaned
    
    def engineer_features(self, launch_df: pd.DataFrame, weather_df: pd.DataFrame = None, orbital_df: pd.DataFrame = None) -> pd.DataFrame:
        """Create engineered features for machine learning."""
        df_features = launch_df.copy()

        # Time-based features
        if 'date_utc' in df_features.columns:
            df_features['launch_datetime'] = pd.to_datetime(df_features['date_utc'])
            df_features['launch_year'] = df_features['launch_datetime'].dt.year
            df_features['launch_month'] = df_features['launch_datetime'].dt.month
            df_features['launch_day_of_year'] = df_features['launch_datetime'].dt.dayofyear
            df_features['launch_hour'] = df_features['launch_datetime'].dt.hour
            df_features['launch_day_of_week'] = df_features['launch_datetime'].dt.dayofweek
            df_features['launch_season'] = df_features['launch_month'].map({
                3: 'Spring', 4: 'Spring', 5: 'Spring',
                6: 'Summer', 7: 'Summer', 8: 'Summer',
                9: 'Fall', 10: 'Fall', 11: 'Fall',
                12: 'Winter', 1: 'Winter', 2: 'Winter'
            })

        # Rocket family features
        if 'rocket_name' in df_features.columns:
            df_features['rocket_family'] = df_features['rocket_name'].apply(self._extract_rocket_family)
            df_features['rocket_version'] = df_features['rocket_name'].apply(self._extract_rocket_version)
            df_features['is_reusable'] = df_features['rocket_name'].apply(self._is_reusable_rocket)

        # Launch provider features
        if 'launch_provider' in df_features.columns:
            df_features['is_commercial'] = df_features['launch_provider'].apply(self._is_commercial_provider)
            df_features['provider_experience'] = df_features.groupby('launch_provider').cumcount() + 1
        
        # Mission complexity features
        if 'mission_type' in df_features.columns:
            complexity_mapping = {
                'Communication': 2, 'Earth Observation': 2, 'Navigation': 3,
                'Scientific': 4, 'Planetary': 5, 'Crewed': 5,
                'Cargo': 2, 'Satellite Deployment': 2, 'Unknown': 3
            }
            df_features['mission_complexity'] = df_features['mission_type'].map(complexity_mapping).fillna(3)
        
        # Payload features (adding bins)
        if 'payload_mass_kg' in df_features.columns:
            df_features['payload_mass_kg'] = df_features['payload_mass_kg'].fillna(0)
            df_features['payload_category'] = pd.cut(
                df_features['payload_mass_kg'],
                bins = [0, 1000, 5000, 15000, float('inf')],
                labels=['Light', 'Medium', 'Heavy', 'Super Heavy'],
                include_lowest=True
            )
        
        # Weather integration
        if weather_df is not None:
            df_features = self._integrate_weather_features(df_features, weather_df)
        
        # Orbital traffic integration
        if orbital_df is not None:
            df_features = self._integrate_orbital_features(df_features, orbital_df)

        return df_features
    
    def _extract_rocket_family(self, rocket_name: str) -> str:
        """Extract rocket family from rocket name."""
        if pd.isna(rocket_name):
            return "Unknown"
        
        name = str(rocket_name).upper()

        if 'FALCON' in name:
            return 'Falcon'
        elif 'ATLAS' in name:
            return 'Atlas'
        elif 'DELTA' in name:
            return 'Delta'
        elif 'ANTARES' in name:
            return 'Antares'
        elif 'ELECTRON' in name:
            return 'Electron'
        elif 'SOYUZ' in name:
            return 'Soyuz'
        elif 'ARIANE' in name:
            return 'Ariane'
        elif 'SLS' in name:
            return 'SLS'
        else:
            return 'Other'
        
    def _extract_rocket_version(self, rocket_name: str) -> str:
        """Extract version number of rocket"""
        if pd.isna(rocket_name):
            return 'Unkown'
        
        # Look for version patterns like integers, weights, versions (V), etc.
        version_patterns = [
            r'\d+', # Numbers
            r'(HEAVY|LIGHT)', # Weight classes
            r'V\d*' # Version indicators
        ]

        for pattern in version_patterns:
            match = re.search(pattern, str(rocket_name).upper())
            if match:
                return match.group(0)
        
        return 'Base'
    
    def _is_reusable_rocket(self, rocket_name: str) -> bool:
        """Determine if rocket is reusable based on name."""
        if pd.isna(rocket_name):
            return None
        
        reusable_rockets = ['FALCON 9', 'FALCON HEAVY', 'ELECTRON']
        return any(rocket in str(rocket_name).upper() for rocket in reusable_rockets)

    def _is_commercial_provider(self, provider: str) -> bool:
        """Determine if it is a commercial launch provider."""
        if pd.isna(provider):
            return False
        
        commercial_providers = ['SPACEX', 'BLUE ORIGIN', 'ROCKET LAB', 'VIRGIN ORBIT', 'ULA']
        return any(commercial in str(provider).upper() for commercial in commercial_providers)
    
    def _integrate_weather_features(self, df_launches: pd.DataFrame, df_weather: pd.DataFrame) -> pd.DataFrame:
        """Integrate weather features with launch data."""
        if 'launch_datetime' not in df_launches.columns:
            return df_launches
        
        df_integrated = df_launches.copy()

        # Initialize weather columns
        for col in self.weather_features:
            df_integrated[f'weather_{col}'] = np.nan
        
        # Match weather data to launches
        if 'forecast_time' in df_weather.columns:
            df_weather['weather_datetime'] = pd.to_datetime(df_weather['forecast_time'])
        elif 'dt' in df_weather.columns:
            df_weather['weather_datetime'] = pd.to_datetime(df_weather['dt'])
        else:
            return df_integrated
        
        df_integrated['launch_datetime'] = pd.to_datetime(df_integrated['launch_datetime'], utc=True)
        df_weather['weather_datetime'] = pd.to_datetime(df_weather['weather_datetime'], utc=True)    

        for idx, launch_row in df_integrated.iterrows():
            launch_time = launch_row['launch_datetime']

            # Skip if launch_time is not a time
            if pd.isna(launch_time):
                continue

            launch_site = launch_row.get('launch_site', 'Unknown')

            # Find closest weather data in time and location
            site_weather = df_weather[df_weather.get('site_code', 'Unknown') == launch_site]
            if len(site_weather) == 0:
                site_weather = df_weather
            
            if len(site_weather) > 0:
                # Find closest time match
                site_weather = site_weather.copy()
                site_weather['time_diff'] = abs(site_weather['weather_datetime'] - launch_time)

                if not site_weather['time_diff'].empty and not site_weather['time_diff'].isna().all():
                    try:
                        closest_idx = site_weather['time_diff'].idxmin()
                        closest_weather = site_weather.loc[closest_idx]

                        # Copy weather features
                        for col in self.weather_features:
                            if col in closest_weather:
                                df_integrated.loc[idx, f'weather_{col}'] = closest_weather[col]
                    except (ValueError, KeyError):
                        continue # Skip this launch of no matching weather.

        # Engineer additional weather features
        if 'weather_wind_speed_ms' in df_integrated.columns:
            df_integrated['weather_wind_category'] = pd.cut(
                df_integrated['weather_wind_speed_ms'],
                bins = [0, 5, 10, 15, float('inf')],
                labels=['Calm', 'Light', 'Moderate', 'Strong'],
                include_lowest=True
            )

        return df_integrated
    
    def _integrate_orbital_features(self, df_launches: pd.DataFrame, df_orbital: pd.DataFrame) -> pd.DataFrame:
        df_integrated = df_launches.copy()

        if 'launch_datetime' not in df_launches.columns:
            return df_integrated
        
        # Calculate orbital congestion features
        df_integrated['orbital_object_count'] = len(df_orbital)

        # Categorize orbital object by altitude
        if 'approximate_altitude_km' in df_orbital.columns:
            leo_objects = len(df_orbital[df_orbital['approximate_altitude_km'] < 2000])
            meo_objects = len(df_orbital[df_orbital['approximate_altitude_km'] >= 2000 & 
                              (df_orbital['approximate_altitude_km'] < 35786)])
            geo_objects = len(df_orbital[df_orbital['approximate_altitude_km'] >= 35786])

            df_integrated['leo_objects_count'] = leo_objects
            df_integrated['meo_objects_count'] = meo_objects
            df_integrated['geo_objects_count'] = geo_objects

            df_integrated['orbital_congestion_score'] = leo_objects * 0.6 + meo_objects * 0.3 + geo_objects * 0.1

        return df_integrated
    
    def create_training_dataset(self, target_column: str = 'success', exclude_weather_features: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """Create a complete training dataset for machine learning."""
        if not self.processed_data:
            print("No data loaded. Please run load_all_data() first.")
            return None, None
        
        # Start with launch data as the base
        launch_data = []
        for key, df in self.processed_data.items():
            if 'launch' in key.lower() or 'spacex' in key.lower():
                df_copy = df.copy()
                # Ensure there is a source column
                df_copy['data_source'] = key
                launch_data.append(df_copy)

        if not launch_data:
            print("No launch data found.")
            return None, None
        
        # Combine all launch data
        combined_launches = pd.concat(launch_data, ignore_index=True, sort=False)

        # Clean the launch data
        combined_launches = self.clean_launch_data(combined_launches)

        # Filter with only valid success markers
        combined_launches = combined_launches[combined_launches['success'].notna()]
        combined_launches = combined_launches[combined_launches['success'].isin([True, False, 0, 1])]
    
        weather_data = self.processed_data.get('weather_forecast')
        if weather_data is None:
            weather_data = self.processed_data.get('weather_current')
        orbital_data = self.processed_data.get('orbital_data')

        # Clean supplementary data
        if weather_data is not None:
            weather_data = self.clean_weather_data(weather_data)
        if orbital_data is not None:
            orbital_data = self.clean_orbital_data(orbital_data)

        # Engineer features (conditionally exclude weather for historical training)
        if exclude_weather_features:
            df_features = self.engineer_features(combined_launches, None, orbital_data)
            print("Weather features excluded from training dataset to avoid imputation artifacts.")
        else:
            df_features = self.engineer_features(combined_launches, weather_data, orbital_data)
            print("Weather features included in training dataset.")

        # Ensure target column exists and is clean
        if target_column not in df_features.columns:
            print(f"Target column {target_column} not found in data.")
            return None, None
        
        # Separate features and target
        y = df_features[target_column].copy()
        
        # Clean target variable
        y = y.map({True: 1, False: 0, 1:1, 0:0})

        # Remove any remaining NaN values
        valid_indices = y.notna()
        y = y[valid_indices]
        df_features = df_features[valid_indices]

        # Select feature columns (Exclude target and metadata)
        exclude_columns = [
            target_column, 'data_source', 'collection_date',
            'launch_library_id', 'launch_library_url', 'video_url',
            'wikipedia', 'webcast', 'details', 'mission_description',
            'info_urls', 'video_urls', 'image_urls', 'programs', 'last_updated'
        ]

        feature_columns = [col for col in df_features.columns if col not in exclude_columns]
        
        # Additional filtering for weather features if requested
        if exclude_weather_features:
            weather_related_columns = [col for col in feature_columns if 'weather_' in col.lower()]
            if weather_related_columns:
                print(f"Excluding {len(weather_related_columns)} weather-related columns: {weather_related_columns[:5]}...")
                feature_columns = [col for col in feature_columns if 'weather_' not in col.lower()]

        X = df_features[feature_columns]

        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns

        # One-hot encode categorical variables
        if len(categorical_cols) > 0:
            # Convert categorical columns to string type first to avoid the category issue
            for col in categorical_cols:
                # Convert to string type to allow modification
                X[col] = X[col].astype(str)
                
                # Handle missing values
                X[col] = X[col].fillna('Unknown')
                
                # Get top categories
                value_counts = X[col].value_counts()
                if len(value_counts) > 10:
                    top_categories = value_counts.head(10).index.tolist()
                    # Use numpy where for safer assignment
                    X[col] = np.where(X[col].isin(top_categories), X[col], 'Other')

            # Now do one-hot encoding
            X_encoded = pd.get_dummies(X[categorical_cols], prefix_sep='_', drop_first=True)
            
            # Clean feature names for XGBoost compatibility
            X_encoded.columns = [col.replace('[', '_').replace(']', '_').replace('<', '_').replace('>', '_').replace(',', '_') 
                               for col in X_encoded.columns]
            
            X_final = pd.concat([X[numeric_cols], X_encoded], axis=1)
        else:
            X_final = X[numeric_cols].copy()

        # Fill missing values for numeric columns
        for col in X_final.columns:
            if X_final[col].dtype in [np.float64, np.int64, float, int]:
                median_val = X_final[col].median()
                if pd.notna(median_val):
                    X_final[col] = X_final[col].fillna(median_val)
                else:
                    X_final[col] = X_final[col].fillna(0)
            else:
                X_final[col] = X_final[col].fillna(0)

        # Remove any columns that are all NaN or have zero variance
        X_final = X_final.loc[:, X_final.notna().any()]
        X_final = X_final.loc[:, X_final.std() > 0]

        self.feature_columns = X_final.columns.tolist()

        return X_final, y

    def generate_data_report(self) -> Dict:
        """Generate a comprehensive data quality and preprocessing report."""
        if not self.processed_data:
            return {'error': 'No data loaded.'}
        
        report = {
            'datasets_loaded': list(self.processed_data.keys()),
            'total_records': sum(len(df) for df in self.processed_data.values()),
            'quality_reports': {},
            'feature_summary': {},
            'recommendations': []
        }

        # Analyze each dataset
        for name, df in self.processed_data.items():
            quality_report = self.analyze_data_quality(df, name)
            report['quality_reports'][name] = {
                'total_records': quality_report.total_records,
                'missing_values': quality_report.missing_values,
                'duplicate_records': quality_report.duplicate_records,
                'completeness_score': quality_report.completeness_score,
                'overall_quality': quality_report.overall_quality
            }

            # Feature summary
            report['feature_summary'][name] = {
                'columns': list(df.columns),
                'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
                'categorical_columns': list(df.select_dtypes(include=['object', 'category']).columns),
                'datetime_columns': list(df.select_dtypes(include=['datetime64']).columns)
            }

        # Generate recommendations
        recommendations = []

        for name, quality in report['quality_reports'].items():
            if quality['overall_quality'] == 'POOR':
                recommendations.append(f"Dataset {name} has poor data quality. Consider data validation and cleaning.")
            
            if quality['duplicate_records'] > 0:
                recommendations.append(f"Dataset {name} contains {quality['duplicate_records']} duplicate records.")
            
            if len(quality['missing_values']) > 0:
                high_missing = [col for col, count in quality['missing_values'].items()
                                if count > quality['total_records'] * 0.3]
                if high_missing:
                    recommendations.append(f"Dataset {name} has high missing values in columns: {high_missing}")

        # Data integration recommendations
        has_launch_data = any('launch' in key.lower() for key in self.processed_data.keys())
        has_weather_data = any('weather' in key.lower() for key in self.processed_data.keys())
        has_orbital_data = any('orbital' in key.lower() or 'tle' in key.lower() for key in self.processed_data.keys())

        if has_launch_data:
            if has_weather_data:
                recommendations.append("Weather and launch data available - integrate for weather-based predictions.")
            if has_orbital_data:
                recommendations.append("Orbital and launch data available - analyze orbital congestion effects.")
        else:
            recommendations.append("No launch data found - required for prediction models.")

        report['recommendations'] = recommendations

        return report
    
    def remove_url_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove URL and unnecessary metadata columns from dataframe."""
        url_patterns = ['url', 'link', 'webcast', 'wikipedia']
        columns_to_remove = []

        for col in df.columns:
            # Check if column is in the patterns
            if any(pattern in col.lower() for pattern in url_patterns):
                columns_to_remove.append(col)
        
        # Also remove other non-feature columns
        other_exclude = ['collection_date', 'last_updated', 'collected_at',
                         'mission_description', 'details', 'programs']
        
        for col in other_exclude:
            if col in df.columns:
                columns_to_remove.append(col)

        return df.drop(columns=columns_to_remove, errors='ignore')

    def save_processed_data(self, output_dir: str="data/processed"):
        """Save all processed datasets."""
        os.makedirs(output_dir, exist_ok=True)

        for name, df in self.processed_data.items():
            # Clean df beforehand
            df_clean = self.remove_url_columns(df)

            filename = f"{name}_processed.csv"
            filepath = os.path.join(output_dir, filename)
            df_clean.to_csv(filepath, index=False)
            print(f"Saved {filename}.")
        try:
            metadata = {
                'feature_mappings': self.feature_mappings,
                'quality_reports': {name: {
                    'total_records': int(report.total_records),
                    'missing_values': {k: int(v) for k, v in report.missing_values.items()},
                    'duplicate_records': int(report.duplicate_records),
                    'completeness_score': report.completeness_score,
                    'consistency_score': report.consistency_score,
                    'overall_quality': report.overall_quality
                } for name, report in self.quality_reports.items()}
            }

            metadata_path = os.path.join(output_dir, 'preprocessing_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            print(f"Preprocessing metadata saved to {metadata_path}.")
        except Exception as e:
            print(f"Could not save metadata: {e}.")

    def load_processed_data(self, input_dir: str="data/processed") -> Dict[str, pd.DataFrame]:
        """Load previously processed datasets."""
        processed_data = {}

        if not os.path.exists(input_dir):
            print(f"Processed data not found in {input_dir}.")

        for filename in os.listdir(input_dir):
            if filename.endswith('_processed.csv'):
                filepath = os.path.join(input_dir, filename)
                dataset_name = filename.replace('_processed.csv', '')

                try:
                    df = pd.read_csv(filepath)
                    processed_data[dataset_name] = df
                except Exception as e:
                    print(f"Error loading {filename}: e")

        # Load metadata if available
        metadata_path = os.path.join(input_dir, 'preprocessing_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.feature_mappings = metadata.get('feature_mappings', {})
        
        self.processed_data = processed_data
        return processed_data
    
def main():
    preprocessor = DataPreprocessor()
    
    # Load all data
    datasets = preprocessor.load_all_data()

    if datasets:
        # Generate quality report
        report = preprocessor.generate_data_report()
        print("Data Quality Report:")
        for dataset, quality in report['quality_reports'].items():
            print(f" {dataset}: {quality['total_records']} records, {quality['overall_quality']} quality.")

        # Create training dataset
        X, y = preprocessor.create_training_dataset()

        if X is not None:
            print(f"\nTraining dataset: {len(X)} samples, {len(X.columns)} features.")
            print(f"Success rate: {y.mean():.1%}" if y is not None else "No target available.")
        
        preprocessor.save_processed_data()
    else:
        print("No data found. Please run your data collectors first.")


if __name__ == "__main__":
    main()