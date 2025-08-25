"""
Database-integrated Flask API for Launch Window Prediction System.
Uses PostgreSQL on Railway for data persistence instead of CSV files.
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
import sys
import os
import logging
from datetime import datetime, timedelta
import traceback
import atexit

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global components
demo = None
db_weather_service = None
is_initialized = False
scheduler = None

def initialize_components():
    """Initialize components with database integration."""
    global demo, db_weather_service, scheduler, is_initialized
    
    try:
        logger.info("Initializing database-integrated components...")
        
        # Initialize database
        from src.database.connection import init_database, get_database_url_from_railway
        
        database_url = get_database_url_from_railway()
        if not database_url:
            logger.error("No database URL found. Check Railway environment variables.")
            return False
        
        logger.info("Initializing PostgreSQL database...")
        db_success = init_database(database_url=database_url, create_tables=True)
        if not db_success:
            logger.error("Failed to initialize database")
            return False
        logger.info("Database initialized successfully")
        
        # Initialize demo system
        from src.master_demo import LaunchPredictionDemo
        logger.info("Initializing LaunchPredictionDemo...")
        demo = LaunchPredictionDemo()
        logger.info("Demo system initialized")
        
        # Initialize database-integrated weather service
        from src.database.weather_service_db import DatabaseWeatherService
        logger.info("Initializing DatabaseWeatherService...")
        db_weather_service = DatabaseWeatherService()
        logger.info("Database weather service initialized")
        
        # Load pre-trained ML model
        try:
            model_path = "data/models/launch_success_model.pkl"
            if os.path.exists(model_path):
                demo.predictor.load_model(model_path)
                logger.info("Pre-trained ML model loaded successfully!")
            else:
                logger.warning(f"Model file not found at {model_path}")
        except Exception as e:
            logger.error(f"Error loading ML model: {e}")
        
        # Initialize background scheduler for weather updates
        if scheduler is None:
            logger.info("Initializing background scheduler...")
            scheduler = BackgroundScheduler()
            
            # Add job to update weather data every 10 minutes
            scheduler.add_job(
                func=db_weather_service.update_all_weather_data,
                trigger=IntervalTrigger(minutes=10),
                id='weather_update_db',
                name='Update weather data cache (Database)',
                replace_existing=True
            )
            
            scheduler.start()
            atexit.register(lambda: scheduler.shutdown())
            logger.info("Background scheduler started - weather updates every 10 minutes")
            
            # Run initial weather update
            try:
                logger.info("Running initial weather update...")
                initial_result = db_weather_service.update_all_weather_data()
                logger.info(f"Initial update completed: {initial_result}")
            except Exception as e:
                logger.error(f"Initial weather update failed: {e}")
        
        is_initialized = True
        logger.info("All components initialized successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing components: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        is_initialized = False
        return False

@app.route('/health', methods=['GET'])
def health_check():
    """Enhanced health check with database status."""
    try:
        from src.database.connection import db
        
        # Get database health
        db_health = db.health_check() if db._initialized else {'status': 'not_initialized'}
        
        model_status = 'unknown'
        scheduler_status = 'stopped'
        weather_cache_status = {}
        
        if is_initialized and demo:
            model_status = 'trained' if demo.predictor.is_trained else 'not_trained'
        
        if scheduler and scheduler.running:
            scheduler_status = 'running'
        
        if db_weather_service:
            weather_cache_status = db_weather_service.get_service_status()
        
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'components_initialized': is_initialized,
            'ml_model_status': model_status,
            'scheduler_status': scheduler_status,
            'weather_cache_status': weather_cache_status,
            'database_status': db_health,
            'python_version': sys.version,
            'working_directory': os.getcwd(),
            'storage_type': 'postgresql'
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@app.route('/prediction', methods=['POST'])
def run_prediction():
    """Run launch success prediction with database storage."""
    try:
        if not is_initialized:
            return jsonify({'error': 'System components not initialized'}), 503
            
        data = request.get_json()
        site_code = data.get('site_code', 'KSC')
        
        # Get current weather using database service
        weather = db_weather_service.get_current_weather(site_code) if db_weather_service else None
        
        if not weather:
            return jsonify({'error': 'No weather data available'}), 400
        
        # Create prediction features
        import pandas as pd
        launch_features = pd.DataFrame([{
            'rocket_name': demo.demo_vehicle.name,
            'mission_type': 'Commercial',
            'payload_mass_kg': 15000,
            'launch_provider': 'SpaceX',
            'is_commercial': True,
            'is_reusable': True,
            'mission_complexity': 2,
            'payload_category': 'Heavy',
            'launch_year': datetime.now().year,
            'launch_month': datetime.now().month,
            'launch_hour': datetime.now().hour,
            'launch_day_of_week': datetime.now().weekday()
        }])
        
        # Run prediction with weather constraints
        combined_result = demo.predictor.predict_with_weather_constraints(
            launch_features, weather, site_code
        )
        
        # Store prediction in database
        try:
            from src.database.dal import PredictionDataAccess
            prediction_dal = PredictionDataAccess()
            
            prediction_data = {
                'site_code': site_code,
                'ml_success_probability': combined_result['ml_prediction']['success_probability'],
                'ml_model_type': 'random_forest',
                'ml_confidence_level': combined_result['ml_prediction']['confidence_level'],
                'weather_score': combined_result['weather_assessment']['weather_score'],
                'weather_go_for_launch': combined_result['weather_assessment']['go_for_launch'],
                'weather_risk_level': combined_result['weather_assessment']['risk_level'],
                'weather_violations': combined_result['weather_assessment']['violated_constraints'],
                'combined_success_probability': combined_result['combined_assessment']['combined_success_probability'],
                'overall_recommendation': combined_result['combined_assessment']['overall_recommendation'],
                'overall_risk_level': combined_result['combined_assessment']['overall_risk_level'],
                'rocket_specs': {
                    'name': demo.demo_vehicle.name,
                    'mass_kg': demo.demo_vehicle.dry_mass_kg + demo.demo_vehicle.fuel_mass_kg
                },
                'weather_data': weather,
                'methodology': 'Combined ML (weather-independent) + Physics-based weather constraints'
            }
            
            prediction_id = prediction_dal.store_prediction(prediction_data)
            logger.info(f"Stored prediction {prediction_id} for site {site_code}")
            
        except Exception as e:
            logger.warning(f"Failed to store prediction in database: {e}")
        
        # Get site name from database
        from src.database.dal import LaunchDataAccess
        site_info = LaunchDataAccess.get_site_by_code(site_code)
        site_name = site_info['site_name'] if site_info else site_code
        
        return jsonify({
            'success_probability': combined_result['combined_assessment']['combined_success_probability'],
            'risk_assessment': combined_result['combined_assessment']['overall_risk_level'],
            'recommendation': combined_result['combined_assessment']['overall_recommendation'],
            'weather_status': 'GO' if combined_result['weather_assessment']['go_for_launch'] else 'NO GO',
            'site_name': site_name,
            'detailed_analysis': {
                'ml_prediction': combined_result['ml_prediction'],
                'weather_assessment': combined_result['weather_assessment'],
                'confidence': combined_result['combined_assessment']['confidence_level']
            },
            'methodology': 'Combined ML (weather-independent) + Physics-based weather constraints',
            'storage': 'postgresql'
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/weather/current/<site_code>', methods=['GET'])
def get_current_weather(site_code):
    """Get current weather from database cache."""
    try:
        force_refresh = request.args.get('force_refresh', 'false').lower() == 'true'
        
        if not db_weather_service:
            return jsonify({'error': 'Weather service not available'}), 503
        
        weather = db_weather_service.get_current_weather(site_code, force_refresh=force_refresh)
        
        if not weather:
            return jsonify({'error': 'Weather data not available'}), 404
            
        return jsonify(weather)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/weather/forecast/<site_code>', methods=['GET'])
def get_weather_forecast(site_code):
    """Get weather forecast from database cache."""
    try:
        days = request.args.get('days', 5, type=int)
        force_refresh = request.args.get('force_refresh', 'false').lower() == 'true'
        
        if not db_weather_service:
            return jsonify({'error': 'Weather service not available'}), 503
        
        forecast = db_weather_service.get_forecast_weather(site_code, days=days, force_refresh=force_refresh)
        
        if not forecast:
            return jsonify({'error': 'Forecast data not available'}), 404
            
        return jsonify(forecast)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/trajectory', methods=['POST'])
def calculate_trajectory():
    """Calculate launch trajectory with database storage."""
    try:
        data = request.get_json()
        site_code = data.get('site_code', 'KSC')
        target_orbit = data.get('target_orbit', {
            'altitude_km': 400,
            'inclination_deg': 28.5,
            'orbit_type': 'LEO'
        })
        
        # Get weather data from database
        weather_data = db_weather_service.get_current_weather(site_code) if db_weather_service else None
        if not weather_data:
            return jsonify({'error': 'Weather data not available'}), 400
        
        # Calculate trajectory
        trajectory = demo.trajectory_calculator.calculate_launch_trajectory(
            launch_site=site_code,
            target_orbit=target_orbit,
            vehicle_specs=demo.demo_vehicle,
            weather_data=weather_data,
            launch_time=datetime.utcnow()
        )
        
        # Store trajectory analysis in database
        try:
            from src.database.dal import PredictionDataAccess
            trajectory_dal = PredictionDataAccess()
            
            trajectory_data = {
                'site_code': site_code,
                'success_probability': trajectory.success_probability,
                'mission_objectives_met': trajectory.mission_objectives_met,
                'max_dynamic_pressure': trajectory.max_dynamic_pressure,
                'max_g_force': trajectory.max_g_force,
                'fuel_remaining_kg': trajectory.fuel_remaining_kg,
                'total_delta_v': getattr(trajectory, 'total_delta_v', 9200),
                'vehicle_name': demo.demo_vehicle.name,
                'vehicle_specs': {
                    'name': demo.demo_vehicle.name,
                    'dry_mass_kg': demo.demo_vehicle.dry_mass_kg,
                    'fuel_mass_kg': demo.demo_vehicle.fuel_mass_kg,
                    'thrust_n': demo.demo_vehicle.thrust_n
                },
                'target_orbit': target_orbit,
                'final_orbit_elements': trajectory.final_orbit_elements,
                'weather_data': weather_data
            }
            
            analysis_id = trajectory_dal.store_trajectory_analysis(trajectory_data)
            logger.info(f"Stored trajectory analysis {analysis_id} for site {site_code}")
            
        except Exception as e:
            logger.warning(f"Failed to store trajectory analysis: {e}")
        
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
            'total_delta_v': getattr(trajectory, 'total_delta_v', 9200),
            'storage': 'postgresql'
        }
        
        return jsonify(trajectory_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/windows', methods=['POST'])
def find_launch_windows():
    """Find optimal launch windows with database storage."""
    try:
        data = request.get_json()
        site_code = data.get('site_code', 'KSC')
        duration_hours = data.get('duration_hours', 48)
        
        # Get weather forecast from database
        forecast_data = db_weather_service.get_forecast_weather(site_code, days=5) if db_weather_service else None
        if not forecast_data:
            return jsonify({'error': 'Forecast data not available'}), 400
        
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
            start_time=datetime.utcnow(),
            duration_hours=duration_hours
        )
        
        # Store launch windows in database
        try:
            from src.database.dal import PredictionDataAccess
            windows_dal = PredictionDataAccess()
            
            windows_for_db = []
            for window in optimal_windows[:10]:
                window_data = {
                    'site_code': site_code,
                    'launch_time': window['launch_time'],
                    'window_score': window['window_score'],
                    'success_probability': window['success_probability'],
                    'weather_score': window['weather_conditions'].get('weather_score', 0.8),
                    'go_for_launch': window.get('go_for_launch', True),
                    'risk_level': 'LOW' if window['window_score'] > 0.8 else 'MEDIUM' if window['window_score'] > 0.6 else 'HIGH',
                    'target_orbit': target_orbit,
                    'vehicle_specs': {
                        'name': demo.demo_vehicle.name,
                        'mass_kg': demo.demo_vehicle.dry_mass_kg + demo.demo_vehicle.fuel_mass_kg
                    }
                }
                windows_for_db.append(window_data)
            
            stored_count = windows_dal.store_launch_windows(windows_for_db)
            logger.info(f"Stored {stored_count} launch windows for site {site_code}")
            
        except Exception as e:
            logger.warning(f"Failed to store launch windows: {e}")
        
        # Convert to JSON-serializable format
        windows_data = []
        for window in optimal_windows[:10]:
            windows_data.append({
                'launch_time': window['launch_time'].isoformat(),
                'window_score': window['window_score'],
                'success_probability': window['success_probability'],
                'weather_conditions': window['weather_conditions'],
                'go_for_launch': window.get('go_for_launch', True)
            })
        
        return jsonify({
            'windows': windows_data,
            'storage': 'postgresql'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/status', methods=['GET'])
def get_system_status():
    """Get enhanced system status with database information."""
    try:
        from src.database.dal import LaunchDataAccess, SystemDataAccess
        
        # Get launch statistics
        launch_stats = LaunchDataAccess.get_launch_statistics()
        
        # Get system metrics
        system_metrics = SystemDataAccess.get_system_metrics(hours=24)
        
        weather_status = 'Offline'
        cache_info = {}
        
        if db_weather_service:
            service_status = db_weather_service.get_service_status()
            weather_status = f"Live (Database)" if service_status.get('api_mode') == 'live' else 'Mock (Database)'
            cache_info = service_status.get('database_cache_status', {})
        
        return jsonify({
            'ml_model': 'Ready' if demo.predictor.is_trained else 'Training Required',
            'weather_data': weather_status,
            'weather_cache': cache_info,
            'scheduler_running': scheduler.running if scheduler else False,
            'physics_engine': 'Ready',
            'trajectory_calculator': 'Ready',
            'last_update': datetime.utcnow().isoformat(),
            'storage_type': 'postgresql',
            'launch_statistics': launch_stats,
            'system_metrics': system_metrics,
            'system_health': 'Operational'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/database/stats', methods=['GET'])
def get_database_stats():
    """Get database statistics and health information."""
    try:
        from src.database.connection import db
        from src.database.dal import LaunchDataAccess, SystemDataAccess
        
        # Database health check
        db_health = db.health_check()
        
        # Data statistics
        launch_stats = LaunchDataAccess.get_launch_statistics()
        system_metrics = SystemDataAccess.get_system_metrics(hours=24)
        
        return jsonify({
            'database_health': db_health,
            'launch_statistics': launch_stats,
            'system_metrics': system_metrics,
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Additional endpoints for frontend integration
@app.route('/sites', methods=['GET'])
def get_all_sites():
    """Get all launch sites."""
    try:
        from src.database.dal import LaunchDataAccess
        sites = LaunchDataAccess.get_all_sites()
        return jsonify(sites)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/sites/<site_code>', methods=['GET'])
def get_site_info(site_code):
    """Get information for a specific launch site."""
    try:
        from src.database.dal import LaunchDataAccess
        site = LaunchDataAccess.get_site_by_code(site_code)
        if not site:
            return jsonify({'error': 'Site not found'}), 404
        return jsonify(site)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predictions/history', methods=['GET'])
def get_prediction_history():
    """Get prediction history."""
    try:
        from src.database.dal import PredictionDataAccess
        
        site_code = request.args.get('site_code')
        days = request.args.get('days', 30, type=int)
        
        history = PredictionDataAccess.get_prediction_history(site_code, days)
        return jsonify(history)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/launches/statistics', methods=['GET'])
def get_launch_statistics():
    """Get launch statistics."""
    try:
        from src.database.dal import LaunchDataAccess
        stats = LaunchDataAccess.get_launch_statistics()
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/weather/cache/status', methods=['GET'])
def get_cache_status():
    """Get detailed weather cache status."""
    try:
        if not db_weather_service:
            return jsonify({'error': 'Weather service not initialized'}), 503
        
        return jsonify(db_weather_service.get_service_status())
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/weather/cache/refresh', methods=['POST'])
def refresh_weather_cache():
    """Manually trigger weather cache refresh."""
    try:
        if not db_weather_service:
            return jsonify({'error': 'Weather service not initialized'}), 503
        
        data = request.get_json() or {}
        site_code = data.get('site_code')
        
        if site_code:
            # Refresh specific site
            success = db_weather_service.force_refresh_site(site_code)
            return jsonify({
                'status': 'success' if success else 'failed',
                'site_code': site_code,
                'timestamp': datetime.utcnow().isoformat()
            })
        else:
            # Refresh all sites
            result = db_weather_service.update_all_weather_data()
            return jsonify({
                'status': 'success',
                'update_result': result,
                'timestamp': datetime.utcnow().isoformat()
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Launch Prediction API Server (Database Edition)...")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    
    # Initialize components
    print("Attempting to initialize components...")
    success = initialize_components()
    
    if not success:
        print("WARNING: Component initialization failed. API will have limited functionality.")
    
    port = int(os.environ.get('PORT', 8000))
    print(f"Starting server on port {port}")
    print(f"Health check: http://localhost:{port}/health")
    
    # Start the Flask app
    app.run(debug=False, host='0.0.0.0', port=port)

# Auto-initialize components when module is imported (for Railway deployment)
# This runs after all functions are defined
if not is_initialized:
    logger.info("Auto-initializing components on module import...")
    try:
        initialize_components()
    except Exception as e:
        logger.error(f"Failed to auto-initialize components: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")