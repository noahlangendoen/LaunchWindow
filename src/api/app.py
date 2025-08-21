from flask import Flask, jsonify, request
from flask_cors import CORS
import sys
import os
from datetime import datetime, timedelta
import traceback

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables for components
demo = None
is_initialized = False

def initialize_components():
    """Initialize components with error handling"""
    global demo, is_initialized
    try:
        # Add project root to path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        sys.path.insert(0, project_root)

        # Import the existing demo classes
        from src.master_demo import LaunchPredictionDemo
        
        print("Initializing LaunchPredictionDemo...")
        demo = LaunchPredictionDemo()
        
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
        if is_initialized and demo:
            model_status = 'trained' if demo.predictor.is_trained else 'not_trained'
        
        return jsonify({
            'status': 'healthy', 
            'timestamp': datetime.now().isoformat(),
            'components_initialized': is_initialized,
            'ml_model_status': model_status,
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
        
        # Get current weather for the site
        weather = demo.weather_collector.get_current_weather(site_code)
        
        if not weather:
            return jsonify({'error': 'No weather data available'}), 400
        
        # Create prediction features
        import pandas as pd
        features = pd.DataFrame([{
            'rocket_name': demo.demo_vehicle.name,
            'mission_type': 'Commercial',
            'payload_mass_kg': 15000,
            'launch_site': demo.launch_sites.get(site_code, 'Unknown'),
            'weather_temperature_c': weather.get('temperature_c', 20),
            'weather_pressure_hpa': weather.get('pressure_hpa', 1013),
            'weather_humidity_percent': weather.get('humidity_percent', 60),
            'weather_wind_speed_ms': weather.get('wind_speed_ms', 5),
            'weather_cloud_cover_percent': weather.get('cloud_cover_percent', 20),
            'weather_visibility_m': weather.get('visibility_m', 10000)
        }])
        
        # Use actual ML model for prediction
        try:
            prediction_result = demo.predictor.predict_launch_success(features)
            success_prob = prediction_result['success_probability']
        except Exception as e:
            print(f"ML model prediction failed: {e}")
            # Fallback calculation only if ML model fails
            success_prob = 0.85 - (weather.get('wind_speed_ms', 5) * 0.02)
            success_prob = max(0.5, min(0.95, success_prob))
        
        # Assess risk
        if success_prob >= 0.8:
            risk = 'Low'
            recommendation = 'Proceed with launch. All conditions are favorable.'
        elif success_prob >= 0.6:
            risk = 'Medium'
            recommendation = 'Launch conditions are acceptable. Monitor weather conditions closely.'
        else:
            risk = 'High'
            recommendation = 'Consider delaying launch. Weather conditions present elevated risks.'
        
        # Weather status
        weather_go = weather.get('wind_speed_ms', 5) < 15 and weather.get('temperature_c', 20) > -5
        
        return jsonify({
            'success_probability': success_prob,
            'risk_assessment': risk,
            'recommendation': recommendation,
            'weather_status': 'GO' if weather_go else 'NO GO',
            'site_name': demo.launch_sites.get(site_code, site_code)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/weather/current/<site_code>', methods=['GET'])
def get_current_weather(site_code):
    """Get current weather for a launch site"""
    try:
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
        
        # Get weather data
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
        
        # Get weather forecast
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
                'go_for_launch': window.get('go_for_launch', window['weather_conditions'].get('go_for_launch', True))
            })
        
        return jsonify(windows_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/status', methods=['GET'])
def get_system_status():
    """Get system status"""
    try:
        return jsonify({
            'ml_model': 'Ready' if demo.predictor.is_trained else 'Training Required',
            'weather_data': 'Live',
            'physics_engine': 'Ready',
            'trajectory_calculator': 'Ready',
            'last_update': datetime.now().isoformat(),
            'active_sites': len(demo.launch_sites),
            'system_health': 'Operational'
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