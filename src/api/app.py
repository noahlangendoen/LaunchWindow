from flask import Flask, jsonify, request
from flask_cors import CORS
# Updated: Fix weather integration for production deployment
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
import sys
import os
from datetime import datetime, timedelta
import traceback
import atexit

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables for components
demo = None
weather_service = None
is_initialized = False
scheduler = None

def initialize_components():
    """Initialize components with error handling"""
    global demo, weather_service, scheduler, is_initialized
    try:
        # Add project root to path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        sys.path.insert(0, project_root)

        # Import the existing demo classes
        from src.master_demo import LaunchPredictionDemo
        from src.services.weather_service import WeatherService
        
        print("Initializing LaunchPredictionDemo...")
        demo = LaunchPredictionDemo()
        
        print("Initializing WeatherService with caching...")
        weather_service = WeatherService()
        
        # Load pre-trained model if it exists
        try:
            model_path = "data/models/launch_success_model.pkl"
            if os.path.exists(model_path):
                demo.predictor.load_model(model_path)
                print("Pre-trained ML model loaded successfully!")
            else:
                print(f"Model file not found at {model_path}")
                print("Available files in data/models/:")
                models_dir = "data/models"
                if os.path.exists(models_dir):
                    for f in os.listdir(models_dir):
                        print(f"  {f}")
                else:
                    print("  data/models/ directory does not exist")
        except Exception as e:
            print(f"Error loading ML model: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            print("Will use fallback calculations for predictions.")
        
        # Initialize background scheduler for weather updates
        if scheduler is None:
            print("Initializing background scheduler...")
            scheduler = BackgroundScheduler()
            
            # Add job to update weather data every 10 minutes
            scheduler.add_job(
                func=weather_service.update_all_weather_data,
                trigger=IntervalTrigger(minutes=10),
                id='weather_update',
                name='Update weather data cache',
                replace_existing=True
            )
            
            scheduler.start()
            
            # Ensure scheduler shuts down when app exits
            atexit.register(lambda: scheduler.shutdown())
            
            print("Background scheduler started - weather updates every 10 minutes")
            
            # Run an initial weather update
            try:
                print("Running initial weather update...")
                initial_result = weather_service.update_all_weather_data()
                print(f"Initial update completed: {initial_result}")
            except Exception as e:
                print(f"Initial weather update failed: {e}")
        
        is_initialized = True
        print("Components initialized successfully!")
        
    except Exception as e:
        print(f"Error initializing components: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        is_initialized = False

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        model_status = 'unknown'
        scheduler_status = 'stopped'
        weather_cache_status = {}
        
        if is_initialized and demo:
            model_status = 'trained' if demo.predictor.is_trained else 'not_trained'
        
        if scheduler and scheduler.running:
            scheduler_status = 'running'
        
        if weather_service:
            weather_cache_status = weather_service.get_service_status()
        
        return jsonify({
            'status': 'healthy', 
            'timestamp': datetime.now().isoformat(),
            'components_initialized': is_initialized,
            'ml_model_status': model_status,
            'scheduler_status': scheduler_status,
            'weather_cache_status': weather_cache_status,
            'python_version': sys.version,
            'working_directory': os.getcwd()
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/prediction', methods=['POST'])
def run_prediction():
    """Run launch success prediction for a specific site"""
    try:
        if not is_initialized:
            initialize_components()
            
        if not is_initialized or demo is None:
            return jsonify({'error': 'System components not initialized'}), 503
            
        data = request.get_json()
        site_code = data.get('site_code', 'KSC')
        
        # Get current weather for the site using cached weather service
        if weather_service:
            weather = weather_service.get_current_weather(site_code)
        else:
            # Fallback to direct weather collector
            weather = demo.weather_collector.get_current_weather(site_code)
        
        if not weather:
            return jsonify({'error': 'No weather data available'}), 400
        
        # Create prediction features (weather-independent for ML model)
        import pandas as pd
        launch_features = pd.DataFrame([{
            'rocket_name': demo.demo_vehicle.name,
            'mission_type': 'Commercial',
            'payload_mass_kg': 15000,
            'launch_provider': 'SpaceX',  # Example provider
            'is_commercial': True,
            'is_reusable': True,
            'mission_complexity': 2,  # Commercial satellite deployment
            'payload_category': 'Heavy',  # 15000 kg
            'launch_year': datetime.now().year,
            'launch_month': datetime.now().month,
            'launch_hour': datetime.now().hour,
            'launch_day_of_week': datetime.now().weekday()
        }])
        
        # Use combined ML + weather constraint prediction
        try:
            combined_result = demo.predictor.predict_with_weather_constraints(
                launch_features, weather, site_code
            )
            
            return jsonify({
                'success_probability': combined_result['combined_assessment']['combined_success_probability'],
                'risk_assessment': combined_result['combined_assessment']['overall_risk_level'],
                'recommendation': combined_result['combined_assessment']['overall_recommendation'],
                'weather_status': 'GO' if combined_result['weather_assessment']['go_for_launch'] else 'NO GO',
                'site_name': demo.launch_sites.get(site_code, site_code),
                'detailed_analysis': {
                    'ml_prediction': combined_result['ml_prediction'],
                    'weather_assessment': combined_result['weather_assessment'],
                    'confidence': combined_result['combined_assessment']['confidence_level']
                },
                'methodology': 'Combined ML (weather-independent) + Physics-based weather constraints'
            })
            
        except Exception as e:
            print(f"Combined prediction failed: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            
            # Fallback to simpler prediction if new method fails
            try:
                # Try weather-independent ML prediction only
                ml_result = demo.predictor.predict_launch_success(launch_features)
                success_prob = ml_result['success_probability']
                
                # Simple weather check
                weather_go = (weather.get('wind_speed_ms', 5) < 15 and 
                             weather.get('temperature_c', 20) > -5 and
                             weather.get('rain_1h_mm', 0) < 0.1)
                
                if not weather_go:
                    success_prob *= 0.3  # Reduce probability if weather is poor
                
                risk = 'Low' if success_prob >= 0.8 else 'Medium' if success_prob >= 0.6 else 'High'
                
                return jsonify({
                    'success_probability': success_prob,
                    'risk_assessment': risk,
                    'recommendation': ml_result['recommendation'],
                    'weather_status': 'GO' if weather_go else 'NO GO',
                    'site_name': demo.launch_sites.get(site_code, site_code),
                    'methodology': 'Fallback: ML prediction with basic weather check'
                })
                
            except Exception as e2:
                print(f"Fallback prediction also failed: {e2}")
                # Ultimate fallback
                success_prob = 0.75
                return jsonify({
                    'success_probability': success_prob,
                    'risk_assessment': 'Medium',
                    'recommendation': 'Unable to generate detailed prediction',
                    'weather_status': 'UNKNOWN',
                    'site_name': demo.launch_sites.get(site_code, site_code),
                    'error': 'Prediction system unavailable',
                    'methodology': 'Static fallback'
                })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/weather/current/<site_code>', methods=['GET'])
def get_current_weather(site_code):
    """Get current weather for a launch site"""
    try:
        force_refresh = request.args.get('force_refresh', 'false').lower() == 'true'
        
        if weather_service:
            weather = weather_service.get_current_weather(site_code, force_refresh=force_refresh)
        else:
            weather = demo.weather_collector.get_current_weather(site_code)
        
        if not weather:
            return jsonify({'error': 'Weather data not available'}), 404
        return jsonify(weather)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/weather/forecast/<site_code>', methods=['GET'])
def get_weather_forecast(site_code):
    """Get weather forecast for a launch site"""
    try:
        days = request.args.get('days', 5, type=int)
        force_refresh = request.args.get('force_refresh', 'false').lower() == 'true'
        
        if weather_service:
            forecast = weather_service.get_forecast_weather(site_code, days=days, force_refresh=force_refresh)
        else:
            forecast = demo.weather_collector.get_forecast(site_code, days=days)
        
        if not forecast:
            return jsonify({'error': 'Forecast data not available'}), 404
        return jsonify(forecast)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/trajectory', methods=['POST'])
def calculate_trajectory():
    """Calculate launch trajectory"""
    try:
        data = request.get_json()
        site_code = data.get('site_code', 'KSC')
        target_orbit = data.get('target_orbit', {
            'altitude_km': 400,
            'inclination_deg': 28.5,
            'orbit_type': 'LEO'
        })
        
        # Get weather data using cached service
        if weather_service:
            weather_data = weather_service.get_current_weather(site_code)
        else:
            weather_data = demo.weather_collector.get_current_weather(site_code)
        
        # Calculate trajectory
        trajectory = demo.trajectory_calculator.calculate_launch_trajectory(
            launch_site=site_code,
            target_orbit=target_orbit,
            vehicle_specs=demo.demo_vehicle,
            weather_data=weather_data,
            launch_time=datetime.now()
        )
        
        # Convert trajectory to JSON-serializable format
        trajectory_data = {
            'trajectory_points': [{
                'time_s': point.time_s,
                'position': point.position.tolist() if hasattr(point.position, 'tolist') else list(point.position),
                'velocity': point.velocity.tolist() if hasattr(point.velocity, 'tolist') else list(point.velocity),
                'altitude_km': point.altitude_km,
                'velocity_magnitude_ms': point.velocity_magnitude_ms,
                'acceleration': point.acceleration.tolist() if hasattr(point.acceleration, 'tolist') else list(point.acceleration),
                'mass_kg': point.mass_kg,
                'thrust_n': point.thrust_n,
                'drag_n': point.drag_n,
                'dynamic_pressure_pa': point.dynamic_pressure_pa,
                'mach_number': point.mach_number,
                'g_force': point.g_force,
                'timestamp': point.timestamp.isoformat()
            } for point in trajectory.trajectory_points],
            'success_probability': trajectory.success_probability,
            'mission_objectives_met': trajectory.mission_objectives_met,
            'max_dynamic_pressure': trajectory.max_dynamic_pressure,
            'max_g_force': trajectory.max_g_force,
            'fuel_remaining_kg': trajectory.fuel_remaining_kg,
            'final_orbit_elements': trajectory.final_orbit_elements,
            'total_delta_v': getattr(trajectory, 'total_delta_v', 9200)
        }
        
        return jsonify(trajectory_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/windows', methods=['POST'])
def find_launch_windows():
    """Find optimal launch windows"""
    try:
        data = request.get_json()
        site_code = data.get('site_code', 'KSC')
        duration_hours = data.get('duration_hours', 48)
        
        # Get weather forecast using cached service
        if weather_service:
            forecast_data = weather_service.get_forecast_weather(site_code, days=5)
        else:
            forecast_data = demo.weather_collector.get_forecast(site_code, days=5)
        
        # Determine target orbit based on site
        target_orbit = {'altitude_km': 400}
        if site_code == 'VSFB':
            target_orbit['inclination_deg'] = 97.5
        else:
            target_orbit['inclination_deg'] = 28.5
        
        # Find optimal windows
        optimal_windows = demo.trajectory_calculator.optimize_launch_window(
            launch_site=site_code,
            target_orbit=target_orbit,
            vehicle_specs=demo.demo_vehicle,
            weather_forecast=forecast_data,
            start_time=datetime.now(),
            duration_hours=duration_hours
        )
        
        # Convert to JSON-serializable format
        windows_data = []
        for window in optimal_windows[:10]:  # Limit to top 10
            windows_data.append({
                'launch_time': window['launch_time'].isoformat(),
                'window_score': window['window_score'],
                'success_probability': window['success_probability'],
                'weather_conditions': window['weather_conditions'],
                'go_for_launch': window.get('go_for_launch', True)
            })
        
        return jsonify(windows_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/status', methods=['GET'])
def get_system_status():
    """Get system status"""
    try:
        weather_status = 'Offline'
        cache_info = {}
        
        if weather_service:
            service_status = weather_service.get_service_status()
            weather_status = 'Live (Cached)' if service_status.get('api_mode') == 'live' else 'Mock (Cached)'
            cache_info = service_status.get('cache_status', {})
        
        return jsonify({
            'ml_model': 'Ready' if demo.predictor.is_trained else 'Training Required',
            'weather_data': weather_status,
            'weather_cache': cache_info,
            'scheduler_running': scheduler.running if scheduler else False,
            'physics_engine': 'Ready',
            'trajectory_calculator': 'Ready',
            'last_update': datetime.now().isoformat(),
            'active_sites': len(demo.launch_sites),
            'system_health': 'Operational'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/weather/cache/status', methods=['GET'])
def get_cache_status():
    """Get detailed weather cache status"""
    try:
        if not weather_service:
            return jsonify({'error': 'Weather service not initialized'}), 503
        
        return jsonify(weather_service.get_service_status())
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/weather/cache/refresh', methods=['POST'])
def refresh_weather_cache():
    """Manually trigger weather cache refresh"""
    try:
        if not weather_service:
            return jsonify({'error': 'Weather service not initialized'}), 503
        
        data = request.get_json() or {}
        site_code = data.get('site_code')
        
        if site_code:
            # Refresh specific site
            success = weather_service.force_refresh_site(site_code)
            return jsonify({
                'status': 'success' if success else 'failed',
                'site_code': site_code,
                'timestamp': datetime.now().isoformat()
            })
        else:
            # Refresh all sites
            result = weather_service.update_all_weather_data()
            return jsonify({
                'status': 'success',
                'update_result': result,
                'timestamp': datetime.now().isoformat()
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Launch Prediction API Server...")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    
    # Try to initialize components but don't fail if it doesn't work
    print("Attempting to initialize components...")
    initialize_components()
    
    port = int(os.environ.get('PORT', 8000))
    print(f"Starting server on port {port}")
    print(f"Health check: http://localhost:{port}/health")
    
    # Start the Flask app
    app.run(debug=False, host='0.0.0.0', port=port)