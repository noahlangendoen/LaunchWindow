"""
Demo script that shows the 4 core objectives without databases or web application development
"""

import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

import pandas as pd
import numpy
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

from src.data_ingestion.collect_weather import WeatherCollector
from src.data_ingestion.collect_spacex import spaceXLaunchCollector
from src.ml_models.data_preprocessor import DataPreprocessor
from src.ml_models.success_predictor import LaunchSuccessPredictor
from src.physics_engine.trajectory_calc import TrajectoryCalculator, VehicleSpecs
from src.physics_engine.atmospheric_models import AtmosphericModels

class LaunchPredictionDemo:
    """Complete demonstration of the launch prediction system."""

    def __init__(self):
        print("Initializing Launch Prediciton System...")

        # Initiate all components
        self.weather_collector = WeatherCollector()
        self.spacex_collector = spaceXLaunchCollector()
        self.preprocessor = DataPreprocessor()
        self.predictor = LaunchSuccessPredictor()
        self.trajectory_calculator = TrajectoryCalculator()
        self.atmospheric_models = AtmosphericModels()

        # Launch sites
        self.launch_sites = {
            'KSC': 'Kennedy Space Center',
            'VSFB': 'Vandenberg Space Force Base',
            'CCAFS': 'Cape Canaveral Space Force Station'
        }

        # Demonstration vehicle
        self.demo_vehicle = VehicleSpecs(
            name='Falcon 9',
            dry_mass_kg=22200,
            fuel_mass_kg=411000,
            thrust_n=7607000,
            specific_impulse_s=282,
            drag_coefficient=0.3,
            cross_sectional_area_m2=10.5,
            max_dynamic_pressure_pa=35000,
            max_acceleration_g=4.0,
            stage_count=2
        )

        print("System Initialized Successfully.")

    def remove_site(self, site_code, reason):
        """Remove launch site from consideration"""
        if site_code in self.launch_sites:
            site_name = self.launch_sites[site_code]
            print(f"REMOVING SITE: {site_name} - {reason}")
            del self.launch_sites[site_code]

        # Check if any sites are left
        if not self.launch_sites:
            print("CRITICAL ERROR: No launch sites remain.")
            print("HALTING PROGRAM - cannot continue without sites.")
            sys.exit(1)

    def check_minimum_sites(self, required_count=1):
        """Check if we have minimum required sites"""
        if len(self.launch_sites) < required_count:
            print(f"INSUFFICIENT SITES - Halting Program")
            sys.exit(1)
    
    def objective_1_launch_success_prediction(self):
        """Objective 1: Launch Success Prediction with ML Models."""

        try:
            """
            COLLECT SPACEX LAUNCH DATA
            """
            print("\nStep 1: Collecting Training Data...")

            # Get SpaceX Data
            spacex_launches = self.spacex_collector.get_all_launches()
            if not spacex_launches:
                print("No SpaceX Data Available -- Killing Run.")
                sys.exit(1)

            print(f"Collected {len(spacex_launches)} Historical SpaceX Launches.")

            """
            COLLECT WEATHER DATA
            """
            print("\nStep 2: Getting Current Weather Conditions...")
            current_weather = {}
            for site_code in self.launch_sites.keys():
                try:
                    weather = self.weather_collector.get_current_weather(site_code)
                    if weather:
                        current_weather[site_code] = weather
                        print(f"Successfully Collected Weather At: {self.launch_sites[site_code]}.")
                    else:
                        self.remove_site(site_code, "No weather for site.")
                except:
                    print(f"Failed To Collect Weather At: {self.launch_sites[site_code]}. Removing As Possible Launch Site.")
                    self.remove_site(site_code, "No weather for site.")
            
            """
            TRAIN THE MODEL
            """
            print("\nStep 3: Training ML Model...")

            # Convert to df and start preprocessing
            launches_df = pd.DataFrame(spacex_launches)

            for col in launches_df.columns:
                if launches_df[col].dtype == 'object':
                    # Check if column contains lists or dicts (getting list object not hashable error)
                    try:
                        if isinstance(launches_df[col].iloc[0], (list, dict)):
                            launches_df[col] = launches_df[col].apply(str)
                    except (IndexError, TypeError):
                        pass

            self.preprocessor.processed_data = {'spacex_launches': launches_df}

            # Create training dataset
            X, y = self.preprocessor.create_training_dataset(target_column='success')

            if X is not None and len(X) > 10:
                # Train Model
                metrics = self.predictor.train_model(X, y, model_type='random_forest')
                print(f"Model Trained! Accuracy: {metrics['accuracy']:.1%}%.")

                """
                MAKE PREDICTIONS ON EACH LAUNCH SITE
                """
                print("\nStep 4: Making Launch Success Predictions For Each Site...")

                for site_code, weather in current_weather.items():
                    site_name = self.launch_sites[site_code]

                    # Create prediction features
                    features = pd.DataFrame([{
                        'rocket_name': self.demo_vehicle.name,
                        'mission_type': 'Commercial',
                        'payload_mass_kg': 15000,
                        'launch_site': site_name,
                        'weather_temperature_c': weather.get('temperature_c', 20),
                        'weather_pressure_hpa': weather.get('pressure_hpa', 1013),
                        'weather_humidity_percent': weather.get('humidity_percent', 60),
                        'weather_wind_speed_ms': weather.get('wind_speed_ms', 5),
                        'weather_cloud_cover_percent': weather.get('cloud_cover_percent', 20),
                        'weather_visibility_m': weather.get('visibility_m', 10000)
                    }])

                    # Make prediction
                    prediction = self.predictor.predict_launch_success(features)

                    # Weather constraints check
                    constraints = self.atmospheric_models.assess_weather_launch_constraints(
                        weather, {
                            'max_wind_speed_ms': 15,
                            'max_wind_gust_ms': 20,
                            'min_temperature_c': -10,
                            'max_temperature_c': 35,
                            'max_precipitation_mm': 0,
                            'min_visibility_m': 5000,
                            'max_cloud_cover_percent': 50
                        }
                    )

                    print(f"\nSite Name: {site_name}:")
                    print(f"    Success Probability: {prediction['success_probability']:.1%}")
                    print(f"    Risk Level: {prediction['risk_assessment']}")
                    print(f"    Recommendation: {prediction['recommendation']}")
                    print(f"    Weather Status: {'GO' if constraints['go_for_launch'] else 'NO GO'}")
                    if constraints['violations']:
                        for violation in constraints['violations']:
                            print(f"WARNING: {violation}\n")
            else:
                print("Insufficient Training Data For ML Model.")
        
        except Exception as e:
            print(f"Error in Objective 1: {e}")

    def objective_2_trajectory_optimization(self):
        """Objective 2: Physics-based Trajectory Calculation"""

        try:
            """
            DEFINE TARGET ORBIT
            """
            target_orbit = {
                'inclination_deg': 28.5,
                'altitude_km': 400,
                'orbit_type': 'LEO'
            }

            print(f"\nStep 1: Target Orbit = {target_orbit['altitude_km']}")

            """
            GET WEATHER FOR TRAJECTORY CALCULATIONS
            """
            print("\nStep 2: Gathering Atmospheric Conditions...")
            site_trajectories = {}

            for site_code, site_name in self.launch_sites.items():
                try:
                    print(f"Analayzing For {site_name}")
                    try:
                    # Get Weather For Site
                        weather_data = self.weather_collector.get_current_weather(site_code)
                        if not weather_data:
                            print(f"No Weather Data Available For {site_name}.")
                            self.remove_site(site_code, "No weather for site.")

                        else:
                            print(f"Collection Successful For: {site_name}.")
                    except:
                        print("Removing Site.")
                        self.remove_site(site_code, "No weather for site.")
                    
                    # Adjust target inclination based on launch site
                    site_target_orbit = target_orbit.copy()
                    if site_code == 'VSFB':
                        site_target_orbit['inclination_deg'] = 97.5
                    elif site_code == 'KSC':
                        site_target_orbit['inclination_deg'] = 28.5
                    elif site_code == 'CCAFS':
                        site_target_orbit['inclination_deg'] = 28.5
                    
                    print(f"Calculating Launch Trajectory For {site_name}...")
                    launch_time = datetime.now()

                    trajectory = self.trajectory_calculator.calculate_launch_trajectory(
                        launch_site=site_code,
                        target_orbit=site_target_orbit,
                        vehicle_specs=self.demo_vehicle,
                        weather_data=weather_data,
                        launch_time=launch_time
                    )

                    site_trajectories[site_code] = {
                        'trajectory': trajectory,
                        'site_name': site_name,
                        'weather': weather_data,
                        'target_orbit': site_target_orbit
                    }

                    print(f"\nTrajectory Results For: {site_name}")
                    print(f"    Success Probability: {trajectory.success_probability:.1%}")
                    print(f"    Mission Objectives Met: {'Yes' if trajectory.mission_objectives_met else 'No'}")
                    print(f"    Max Dynamic Pressure: {trajectory.max_dynamic_pressure / 1000:.1f} kPa")
                    print(f"    Max G-Force: {trajectory.max_g_force:.1f} g")
                    print(f"    Fuel Remaining: {trajectory.fuel_remaining_kg} kg")
                    print(f"    Total Delta V Used: {trajectory.total_delta_v} m/s")

                    if trajectory.final_orbit_elements:
                        orbit = trajectory.final_orbit_elements
                        print("\nFinal Orbit Achieved:")
                        print(f"    Apogee: {orbit.get('apogee_km', 0):.1f} km")
                        print(f"    Perigee: {orbit.get('perigee_km', 0):.1f} km")
                        print(f"    Inclination: {orbit.get('inclination_deg', 0):.1f}°")
                    
                    print(f"\nSafety Analysis For {site_name}")
                    abort_scenarios = self.trajectory_calculator.calculate_abort_scenarios(
                        trajectory, [50, 100, 200]
                    )

                    for altitude, scenario in abort_scenarios.items():
                        print(f"    Abort at {altitude} km: {scenario['abort_type']} (Survival: {scenario['survival_probability']:.1%})")

                except Exception as e:
                    print(f"Error Analyzing {site_name}: {e}")
                    site_trajectories[site_code] = None
            
            """
            COMPARATIVE TRAJECTORY ANALYSIS
            """
            print("Step 3: Comparative Trajectory Analysis")

            successful_sites = {k: v for k, v in site_trajectories.items() if v is not None}

            if successful_sites:
                # Find best site for this mission
                best_site = max(successful_sites.items(),
                                key=lambda x: x[1]['trajectory'].success_probability)
                
                print(f"\nOptimal Launch Site Recommendation:")
                print(f"    Best Site: {best_site[1]['site_name']}")
                print(f"    Success Probability: {best_site[1]['trajectory'].success_probability:.1%}")
                print(f"    Target Inclination: {best_site[1]['target_orbit']['inclination_deg']}°")

                for site_code, data in successful_sites.items():
                    if data:
                        traj = data['trajectory']
                        weather_status = 'GOOD' if data['weather'].get('wind_speed_ms', 5) < 10 else "POOR"
                        print(f"{data['site_name']:<25} {traj.success_probability:.1%}"
                              f"{traj.fuel_remaining_kg:>8.0f}kg   {traj.max_g_force:>5.1f}g  {weather_status}")
                        
                print(f"\nMission-Specific Recommendations:")
                for site_code, data in successful_sites.items():
                    if data:
                        traj = data['trajectory']
                        site_name = data['site_name']

                        if 'VSFB' in site_code and data['target_orbit']['inclination_deg'] > 90:
                            print(f"    {site_name}: EXCELLENT For Polar/Sun-Synchronous Missions")
                        elif site_code in ['KSC', 'CCAFS'] and data['target_orbit']['inclination_deg'] < 35:
                            print(f"    {site_name}: EXCELLENT For Equatorial/ISS Missions")
                        
                        if traj.success_probability >= 0.9:
                            print(f"    {site_name}: HIGH Confidence Mission Success")
                        elif traj.success_probability < 0.9:
                            print(f"    {site_name}: Consider Alternative Launch Window")
                
            return site_trajectories

        except Exception as e:
            print(f"Error In Objective 2: {e}")
            return None 
        
    def objective_3_launch_window_optimization(self):
        """Objective 3: Find optimal launch windows for all sites"""

        try:
            """
            GET WEATHER DATA FOR ALL SITES
            """
            print("\nStep 1: Getting 5-Day Weather Forecasts For All Sites...")
            
            site_forecasts = {}
            for site_code, site_name in self.launch_sites.items():
                try:
                    forecast_data = self.weather_collector.get_forecast(site_code, days=5)
                    if not forecast_data:
                        print(f"No Forecast Collected For: {site_name} - Removing Site")
                        self.remove_site(site_code, "No forecast for site.")

                    site_forecasts[site_code] = forecast_data
                    print(f"{site_name}: {len(forecast_data)} Forecast Data Points")
                except:
                    print(f"No Forecast Collected For: {site_name} - Removing Site")
                    self.remove_site(site_code, "No forecast for site.")

            
            """
            DEFINE OPTIMIZATION PARAMETERS
            """
            start_time = datetime.now()
            print(f"Step 2: Optimizing Launch Windows From {start_time.strftime('%Y-%m-%d %H:%M')}")

            all_site_windows = {}

            for site_code, site_name in self.launch_sites.items():
                print(f"Analyzing: {site_name}")

                # Adjust target orbit for site capabilities
                target_orbit = {'altitude_km': 400}
                if site_code == 'VSFB':
                    target_orbit['inclination_deg'] = 97.5
                else:
                    target_orbit['inclination_deg'] = 28.5
                
                print(f"    Target: {target_orbit['altitude_km']}km, {target_orbit['inclination_deg']}° inclination")

                try:
                    # Find all optimal windows for this site
                    optimal_windows = self.trajectory_calculator.optimize_launch_window(
                        launch_site=site_code,
                        target_orbit=target_orbit,
                        vehicle_specs=self.demo_vehicle,
                        weather_forecast=site_forecasts[site_code],
                        start_time=start_time,
                        duration_hours=48
                    )

                    all_site_windows[site_code] = {
                        'windows': optimal_windows,
                        'site_name': site_name,
                        'target_orbit': target_orbit
                    }

                    # Display top 3 windows for this site
                    print(f"\nTop 3 Launch Windows For {site_name}")

                    for i, window in enumerate(optimal_windows[:3]):
                        launch_time = window['launch_time']
                        score = window['window_score']
                        success_prob = window['success_probability']
                        weather = window['weather_conditions']

                    print(f"\n   #{i+1} - {launch_time.strftime('%Y-%m-%d %H:%M UTC')}")
                    print(f"        Window Score: {score:.3f}")
                    print(f"        Success Probability: {success_prob:.1%}")
                    print(f"        Weather: {weather.get('temperature_c', 20):.1f}°C, Wind {weather.get('wind_speed_ms', 5):.1f}m/s")
                    print(f"        Status: {'EXCELLENT' if score > 0.8 else 'GOOD' if score > 0.6 else 'MARGINAL'}")
                    
                    # Site-specific analysis
                    print(f"\n Risk Assessment For: {site_name}")
                    high_risk_windows = [w for w in optimal_windows if w['window_score'] < 0.4]
                    weather_delays = [w for w in optimal_windows if not w['weather_conditions'].get('go_for_launch', True)]

                    print(f"    High Risk Windows: {len(high_risk_windows)}/{len(optimal_windows)}")
                    print(f"    Weather-Constrained: {len(weather_delays)/{len(optimal_windows)}}")

                    # Best window
                    if optimal_windows:
                        best_window = optimal_windows[0]
                        print(f"    Best Window: {best_window['launch_time'].strftime('%m/%d %H:%M')} (Score: {best_window['window_score']:.3f})")\
                    
                except Exception as e:
                    print(f"Error Analyzing Windows For {site_name}")
                    all_site_windows[site_code] = None
                
            """
            CROSS-SITE COMPARISON AND RECOMMENDATIONS
            """
            print("Step 4: Cross-Site Window Comparison")

            # Find absolute best window across all sites
            all_windows = []
            for site_code, data in all_site_windows.items():
                if data and data['windows']:
                    for window in data['windows']:
                        window['site_code'] = site_code
                        window['site_name'] = site_name
                        window['target_inclination'] = data['target_orbit']['inclination_deg']
                        all_windows.append(window)

            if all_windows:
                # Sort by window score
                all_windows.sort(key=lambda x: x['window_score'], reverse=True)

                print("\nTop 5 Windows Across All Sites:")
                for i, window in enumerate(all_windows[:5]):
                    rank=f"{i + 1}"
                    site=window['site_code']
                    dt = window['launch_time'].strftime('%m/%d %H:%M')
                    score = f"{window['window_score']:.3f}"
                    success = f"{window['success_probability']:.1%}"
                    incl = f"{window['target_inclination']:.1f}°"

                    print(f"{rank:<6} {site:<8} {dt:<16} {score:<8} {success:<8} {incl:<6}")

                # Mission planning recommendations
                print("\nMission Planning Recommendations:")
                best_window = all_windows[0]
                print(f"    Optimal Choice: {best_window['site_name']}")
                print(f"    Launch Time: {best_window['launch_time'].strftime('%Y-%m-%d %H:%M UTC')}")
                print(f"    Success Probability: {best_window['success_probability']:.1%}")
                print(f"    Target Inclination: {best_window['target_inclination']}°")

                # Site-specific recommendations
                print("Site-Specific Recommendations")
                for site_code, data in all_site_windows.items():
                    if data and data['windows']:
                        site_best = data['windows'][0]
                        site_name = data['site_name']

                        if site_code == 'VSFB':
                            mission_type = 'Polar/Earth Observation'
                        else:
                            mission_type = 'Equatorial/Commercial/ISS'
                        
                        print(f"    {site_name}: Best for {mission_type} missions")
                        print(f"    Optimal Window: {site_best['launch_time'].strftime('%m/%d %H:%M')} (Score: {site_best['window_score']:.3f})")
                
                print("\nBackup Options:")
                backup_windows = [w for w in all_windows[1:6] if w['window_score'] > 0.6]
            
            else:
                print("No Suitable Launch Windows Found")
            
            return all_site_windows
    
        except Exception as e:
            print(f"Error In Objective 3: {e}")
            return {}
        
    def objective_4_monitoring_dashboard(self):
        """Objective 4: Real-time monitoring and analysis"""

        try:
            """
            LIVE WEATHER MONITORING
            """
            print(f"Step 1: Monitoring Live Weather")
            current_conditions = {}

            for site_code in self.launch_sites.keys():
                try:
                    weather = self.weather_collector.get_current_weather(site_code)
                    if not weather:
                        self.remove_site(site_code, "No weather for site.")
                        print(f"Removing: {site_code}")

                    current_conditions[site_code] = weather
                    site_name = self.launch_sites[site_code]
                    temp = weather.get('temperature_c', 20)
                    wind = weather.get('wind_speed_ms', 5)
                    humidity = weather.get('humidity_percent', 60)
                    pressure = weather.get('pressure_hpa', 1013)

                    # Weather Status
                    wind_status = "GOOD" if wind < 10 else "OKAY" if wind < 15 else "BAD"
                    temp_status = "GOOD" if -5 <= temp <= 30 else "OKAY" if -10 <= temp <= 35 else "BAD"

                    print(f"\nLive Status For: {site_name}:")
                    print(f"    Temperature: {temp_status} {temp:.1f}°C")
                    print(f"    Wind Speed: {wind_status} {wind:.1f} m/s")
                    print(f"    Humidity: {humidity:.0f}%")
                    print(f"    Pressure: {pressure:.1f} hPa")
                
                except Exception as e:
                    print(f"Live Monitoring For {self.launch_sites[site_code]} Unavailable")
                    self.remove_site(site_code, "Can't monitor site.")

            
            """
            HISTORICAL ANALYSIS
            """
            print("Step 2: Historical Success Analysis")

            try:
                # Get historical data
                spacex_data = self.spacex_collector.get_all_launches()
                if spacex_data:
                    df = pd.DataFrame(spacex_data)

                    # Success rate by rocket
                    if 'rocket_name' in df.columns and 'success' in df.columns:
                        success_by_rocket = df.groupby('rocket_name')['success'].agg(['count', 'sum', 'mean'])
                        success_by_rocket['success_rate'] = success_by_rocket['mean']

                        print("\nSuccess Rate By Rocket Type:")
                        for rocket, data in success_by_rocket.iterrows():
                            if data['count'] >= 5:
                                rate = data['success_rate']
                                count = int(data['count'])
                                status = "GOOD" if rate > 0.9 else "OKAY" if rate > 0.8 else "BAD"
                                print(f"    {status} {rocket}: {rate:.1%} ({count} launches)")


                    if 'date_utc' in df.columns:
                        df['date_utc'] = pd.to_datetime(df['date_utc'])
                        recent_launches = df[df['date_utc'] > datetime.now() - timedelta(days=365)]

                        if len(recent_launches) > 0:
                            recent_success_rate = recent_launches['success'].mean() if 'success' in recent_launches.columns else 0
                            print(f"\nRecent Performance (Last 12 Months):")
                            print(f"    Total Launches: {len(recent_launches)}")
                            print(f"    Success Rate: {recent_success_rate:.1%}")

            except Exception as e:
                print("Historical Analysis Unavailable.")
            
            """
            ALERT SYSTEM
            """
            print("\nStep 3: Alert System")
            alerts = []

            # Weather alerts
            for site_code, weather in current_conditions.items():
                site_name = self.launch_sites[site_code]

                if weather.get('wind_speed_ms', 0) > 15:
                    alerts.append(f"HIGH WIND at {site_name}: {weather['wind_speed_ms']:.1f} m/s")
                
                if weather.get('temperature_c', 20) < -5 or weather.get('temperature_c', 20) > 35:
                    alerts.append(f"EXTREME TEMPERATURE at {site_name}: {weather.get['temperature_c']:.1f}°C")

                if weather.get('visibility_m', 10000) < 5000:
                    alerts.append(f"LOW VISIBILITY at {site_name}: {weather['visibility_m'] / 1000:.1f} km")
            
            if alerts:
                for alert in alerts:
                    print(f"    {alert}")
            else:
                print("No Active Alerts")
            
            """
            SYSTEM STATUS
            """
            print(f"\nSystem Status (as of {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}):")
            print(f"    ML Model: {'Trained' if self.predictor.is_trained else 'Model Requires Training'}")
            print(f"    Weather Data: {'Live' if len(current_conditions) > 0 else 'Unavailable'}")
            print(f"    Physics Engine: Ready")
            print(f"    Historical Data: Available")

        except Exception as e:
            print(f"Error In Objective 4: {e}")
    
    def run_complete_demo(self):
        """Run the complete demonstration of all objectives"""

        start_time = datetime.now()

        # Run all objectives
        self.objective_1_launch_success_prediction()
        self.objective_2_trajectory_optimization()
        self.objective_3_launch_window_optimization()
        self.objective_4_monitoring_dashboard()

        # Summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        print("DEMONSTRATION COMPLETE")

def main():
    """Main function to run the demonstration"""
    try:
        demo = LaunchPredictionDemo()
        demo.run_complete_demo()
    except KeyboardInterrupt:
        print(f"\n\nDemo Interrupted By User")
    except Exception as e:
        print(f"\n\nDemo Failed: {e}")

if __name__ == "__main__":
    main()