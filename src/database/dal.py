"""
Data Access Layer (DAL) for Launch Window Prediction System.
Provides high-level database operations and business logic.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from sqlalchemy import desc, and_, or_, func
from sqlalchemy.orm import Session
from .connection import db
from .models import (
    LaunchSite, Launch, WeatherCurrent, WeatherForecast,
    Prediction, TrajectoryAnalysis, LaunchWindow, SystemHealth
)

logger = logging.getLogger(__name__)

class LaunchDataAccess:
    """Data access operations for launch data."""
    
    @staticmethod
    def get_all_sites() -> List[Dict]:
        """Get all launch sites."""
        with db.session_scope() as session:
            sites = session.query(LaunchSite).all()
            return [
                {
                    'site_code': site.site_code,
                    'site_name': site.site_name,
                    'latitude': site.latitude,
                    'longitude': site.longitude,
                    'altitude_m': site.altitude_m,
                    'timezone': site.timezone
                }
                for site in sites
            ]
    
    @staticmethod
    def get_site_by_code(site_code: str) -> Optional[Dict]:
        """Get launch site by code."""
        with db.session_scope() as session:
            site = session.query(LaunchSite).filter_by(site_code=site_code).first()
            if site:
                return {
                    'site_code': site.site_code,
                    'site_name': site.site_name,
                    'latitude': site.latitude,
                    'longitude': site.longitude,
                    'altitude_m': site.altitude_m,
                    'timezone': site.timezone
                }
            return None
    
    @staticmethod
    def store_launch_data(launches_data: List[Dict]) -> int:
        """
        Store multiple launch records.
        
        Args:
            launches_data: List of launch data dictionaries
            
        Returns:
            Number of records stored
        """
        stored_count = 0
        with db.session_scope() as session:
            for launch_data in launches_data:
                try:
                    # Check if launch already exists (by flight_number if available)
                    existing = None
                    if launch_data.get('flight_number'):
                        existing = session.query(Launch).filter_by(
                            flight_number=launch_data['flight_number']
                        ).first()
                    
                    if not existing:
                        launch = Launch(
                            flight_number=launch_data.get('flight_number'),
                            name=launch_data['name'],
                            date_utc=launch_data['date_utc'],
                            date_local=launch_data['date_local'],
                            success=launch_data.get('success'),
                            upcoming=launch_data.get('upcoming', False),
                            rocket_name=launch_data['rocket_name'],
                            rocket_type=launch_data.get('rocket_type', 'rocket'),
                            site_code=launch_data.get('site_code', 'KSC'),  # Default to KSC
                            payload_mass_kg=launch_data.get('payload_mass_kg'),
                            payload_types=launch_data.get('payload_types'),
                            failures=launch_data.get('failures'),
                            details=launch_data.get('details'),
                            webcast=launch_data.get('webcast'),
                            wikipedia=launch_data.get('wikipedia')
                        )
                        session.add(launch)
                        stored_count += 1
                        
                except Exception as e:
                    logger.error(f"Error storing launch data: {e}")
                    continue
                    
        logger.info(f"Stored {stored_count} launch records")
        return stored_count
    
    @staticmethod
    def get_launch_statistics() -> Dict:
        """Get launch statistics for dashboard."""
        with db.session_scope() as session:
            total_launches = session.query(Launch).count()
            successful_launches = session.query(Launch).filter_by(success=True).count()
            failed_launches = session.query(Launch).filter_by(success=False).count()
            
            # Success rate by site
            site_stats = session.query(
                Launch.site_code,
                func.count(Launch.id).label('total'),
                func.sum(func.cast(Launch.success, 'integer')).label('successes')
            ).filter(Launch.success.isnot(None)).group_by(Launch.site_code).all()
            
            site_success_rates = {}
            for site_code, total, successes in site_stats:
                if total > 0:
                    site_success_rates[site_code] = {
                        'total': total,
                        'successes': successes or 0,
                        'success_rate': (successes or 0) / total
                    }
            
            return {
                'total_launches': total_launches,
                'successful_launches': successful_launches,
                'failed_launches': failed_launches,
                'unknown_outcome': total_launches - successful_launches - failed_launches,
                'overall_success_rate': successful_launches / (successful_launches + failed_launches) if (successful_launches + failed_launches) > 0 else 0,
                'site_statistics': site_success_rates
            }

class WeatherDataAccess:
    """Data access operations for weather data."""
    
    @staticmethod
    def store_current_weather(weather_data: Dict) -> bool:
        """Store current weather data."""
        try:
            with db.session_scope() as session:
                weather = WeatherCurrent(
                    site_code=weather_data['site_code'],
                    timestamp=weather_data.get('timestamp', datetime.utcnow()),
                    temperature_c=weather_data['temperature_c'],
                    feels_like_c=weather_data.get('feels_like_c'),
                    temp_min_c=weather_data.get('temp_min_c'),
                    temp_max_c=weather_data.get('temp_max_c'),
                    humidity_percent=weather_data.get('humidity_percent'),
                    pressure_hpa=weather_data.get('pressure_hpa'),
                    wind_speed_ms=weather_data.get('wind_speed_ms'),
                    wind_direction_deg=weather_data.get('wind_direction_deg'),
                    wind_gust_ms=weather_data.get('wind_gust_ms'),
                    weather_main=weather_data.get('weather_main'),
                    weather_description=weather_data.get('weather_description'),
                    visibility_m=weather_data.get('visibility_m'),
                    cloud_cover_percent=weather_data.get('cloud_cover_percent'),
                    rain_1h_mm=weather_data.get('rain_1h_mm', 0),
                    rain_3h_mm=weather_data.get('rain_3h_mm', 0),
                    snow_1h_mm=weather_data.get('snow_1h_mm', 0),
                    snow_3h_mm=weather_data.get('snow_3h_mm', 0),
                    data_source=weather_data.get('data_source', 'live'),
                    expires_at=weather_data.get('expires_at')
                )
                session.add(weather)
                return True
                
        except Exception as e:
            logger.error(f"Error storing weather data: {e}")
            return False
    
    @staticmethod
    def get_current_weather(site_code: str, max_age_minutes: int = 10) -> Optional[Dict]:
        """Get latest current weather for a site."""
        cutoff_time = datetime.utcnow() - timedelta(minutes=max_age_minutes)
        
        with db.session_scope() as session:
            weather = session.query(WeatherCurrent).filter(
                and_(
                    WeatherCurrent.site_code == site_code,
                    WeatherCurrent.timestamp >= cutoff_time
                )
            ).order_by(desc(WeatherCurrent.timestamp)).first()
            
            if weather:
                return {
                    'site_code': weather.site_code,
                    'timestamp': weather.timestamp.isoformat(),
                    'temperature_c': weather.temperature_c,
                    'feels_like_c': weather.feels_like_c,
                    'humidity_percent': weather.humidity_percent,
                    'pressure_hpa': weather.pressure_hpa,
                    'wind_speed_ms': weather.wind_speed_ms,
                    'wind_direction_deg': weather.wind_direction_deg,
                    'wind_gust_ms': weather.wind_gust_ms,
                    'weather_main': weather.weather_main,
                    'weather_description': weather.weather_description,
                    'visibility_m': weather.visibility_m,
                    'cloud_cover_percent': weather.cloud_cover_percent,
                    'rain_1h_mm': weather.rain_1h_mm,
                    'rain_3h_mm': weather.rain_3h_mm,
                    'snow_1h_mm': weather.snow_1h_mm,
                    'snow_3h_mm': weather.snow_3h_mm,
                    'data_source': weather.data_source
                }
            return None
    
    @staticmethod
    def store_forecast_data(forecast_data: List[Dict]) -> int:
        """Store weather forecast data."""
        stored_count = 0
        with db.session_scope() as session:
            for forecast in forecast_data:
                try:
                    weather = WeatherForecast(
                        site_code=forecast['site_code'],
                        forecast_time=forecast['forecast_time'],
                        retrieved_at=forecast.get('retrieved_at', datetime.utcnow()),
                        temperature_c=forecast['temperature_c'],
                        feels_like_c=forecast.get('feels_like_c'),
                        temp_min_c=forecast.get('temp_min_c'),
                        temp_max_c=forecast.get('temp_max_c'),
                        humidity_percent=forecast.get('humidity_percent'),
                        pressure_hpa=forecast.get('pressure_hpa'),
                        wind_speed_ms=forecast.get('wind_speed_ms'),
                        wind_direction_deg=forecast.get('wind_direction_deg'),
                        wind_gust_ms=forecast.get('wind_gust_ms'),
                        weather_main=forecast.get('weather_main'),
                        weather_description=forecast.get('weather_description'),
                        visibility_m=forecast.get('visibility_m'),
                        cloud_cover_percent=forecast.get('cloud_cover_percent'),
                        rain_1h_mm=forecast.get('rain_1h_mm', 0),
                        rain_3h_mm=forecast.get('rain_3h_mm', 0),
                        snow_1h_mm=forecast.get('snow_1h_mm', 0),
                        snow_3h_mm=forecast.get('snow_3h_mm', 0),
                        pop=forecast.get('pop'),
                        data_source=forecast.get('data_source', 'live'),
                        expires_at=forecast.get('expires_at')
                    )
                    session.add(weather)
                    stored_count += 1
                    
                except Exception as e:
                    logger.error(f"Error storing forecast data: {e}")
                    continue
                    
        logger.info(f"Stored {stored_count} forecast records")
        return stored_count
    
    @staticmethod
    def get_forecast_data(site_code: str, hours_ahead: int = 48) -> List[Dict]:
        """Get weather forecast for a site."""
        start_time = datetime.utcnow()
        end_time = start_time + timedelta(hours=hours_ahead)
        
        with db.session_scope() as session:
            forecasts = session.query(WeatherForecast).filter(
                and_(
                    WeatherForecast.site_code == site_code,
                    WeatherForecast.forecast_time >= start_time,
                    WeatherForecast.forecast_time <= end_time
                )
            ).order_by(WeatherForecast.forecast_time).all()
            
            return [
                {
                    'site_code': f.site_code,
                    'forecast_time': f.forecast_time.isoformat(),
                    'temperature_c': f.temperature_c,
                    'feels_like_c': f.feels_like_c,
                    'humidity_percent': f.humidity_percent,
                    'pressure_hpa': f.pressure_hpa,
                    'wind_speed_ms': f.wind_speed_ms,
                    'wind_direction_deg': f.wind_direction_deg,
                    'wind_gust_ms': f.wind_gust_ms,
                    'weather_main': f.weather_main,
                    'weather_description': f.weather_description,
                    'visibility_m': f.visibility_m,
                    'cloud_cover_percent': f.cloud_cover_percent,
                    'rain_1h_mm': f.rain_1h_mm,
                    'rain_3h_mm': f.rain_3h_mm,
                    'snow_1h_mm': f.snow_1h_mm,
                    'snow_3h_mm': f.snow_3h_mm,
                    'pop': f.pop,
                    'data_source': f.data_source
                }
                for f in forecasts
            ]
    
    @staticmethod
    def cleanup_old_data(days_old: int = 7) -> Dict[str, int]:
        """Clean up old weather data."""
        cutoff_date = datetime.utcnow() - timedelta(days=days_old)
        
        with db.session_scope() as session:
            # Clean up old current weather
            current_deleted = session.query(WeatherCurrent).filter(
                WeatherCurrent.timestamp < cutoff_date
            ).delete()
            
            # Clean up old forecast data
            forecast_deleted = session.query(WeatherForecast).filter(
                WeatherForecast.retrieved_at < cutoff_date
            ).delete()
            
        return {
            'current_weather_deleted': current_deleted,
            'forecast_deleted': forecast_deleted
        }

class PredictionDataAccess:
    """Data access operations for predictions and analyses."""
    
    @staticmethod
    def store_prediction(prediction_data: Dict) -> str:
        """Store ML prediction result."""
        with db.session_scope() as session:
            prediction = Prediction(
                site_code=prediction_data['site_code'],
                ml_success_probability=prediction_data['ml_success_probability'],
                ml_model_type=prediction_data['ml_model_type'],
                ml_confidence_level=prediction_data['ml_confidence_level'],
                weather_score=prediction_data['weather_score'],
                weather_go_for_launch=prediction_data['weather_go_for_launch'],
                weather_risk_level=prediction_data['weather_risk_level'],
                weather_violations=prediction_data.get('weather_violations'),
                combined_success_probability=prediction_data['combined_success_probability'],
                overall_recommendation=prediction_data['overall_recommendation'],
                overall_risk_level=prediction_data['overall_risk_level'],
                rocket_specs=prediction_data.get('rocket_specs'),
                target_orbit=prediction_data.get('target_orbit'),
                weather_data=prediction_data.get('weather_data'),
                methodology=prediction_data['methodology']
            )
            session.add(prediction)
            session.flush()  # Get the ID
            return str(prediction.id)
    
    @staticmethod
    def get_prediction_history(site_code: str = None, days: int = 30) -> List[Dict]:
        """Get prediction history."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        with db.session_scope() as session:
            query = session.query(Prediction).filter(Prediction.created_at >= cutoff_date)
            
            if site_code:
                query = query.filter(Prediction.site_code == site_code)
                
            predictions = query.order_by(desc(Prediction.created_at)).all()
            
            return [
                {
                    'id': str(p.id),
                    'site_code': p.site_code,
                    'created_at': p.created_at.isoformat(),
                    'combined_success_probability': p.combined_success_probability,
                    'overall_recommendation': p.overall_recommendation,
                    'overall_risk_level': p.overall_risk_level,
                    'weather_go_for_launch': p.weather_go_for_launch,
                    'ml_model_type': p.ml_model_type
                }
                for p in predictions
            ]
    
    @staticmethod
    def store_trajectory_analysis(trajectory_data: Dict) -> str:
        """Store trajectory analysis result."""
        with db.session_scope() as session:
            analysis = TrajectoryAnalysis(
                prediction_id=trajectory_data.get('prediction_id'),
                site_code=trajectory_data['site_code'],
                success_probability=trajectory_data['success_probability'],
                mission_objectives_met=trajectory_data['mission_objectives_met'],
                max_dynamic_pressure=trajectory_data['max_dynamic_pressure'],
                max_g_force=trajectory_data['max_g_force'],
                fuel_remaining_kg=trajectory_data['fuel_remaining_kg'],
                total_delta_v=trajectory_data['total_delta_v'],
                vehicle_name=trajectory_data['vehicle_name'],
                vehicle_specs=trajectory_data['vehicle_specs'],
                target_orbit=trajectory_data['target_orbit'],
                final_orbit_elements=trajectory_data.get('final_orbit_elements'),
                weather_data=trajectory_data['weather_data'],
                trajectory_points=trajectory_data.get('trajectory_points')  # Optional, can be large
            )
            session.add(analysis)
            session.flush()
            return str(analysis.id)
    
    @staticmethod
    def store_launch_windows(windows_data: List[Dict]) -> int:
        """Store launch window calculations."""
        stored_count = 0
        with db.session_scope() as session:
            for window_data in windows_data:
                try:
                    window = LaunchWindow(
                        site_code=window_data['site_code'],
                        launch_time=window_data['launch_time'],
                        window_duration_hours=window_data.get('window_duration_hours', 2.0),
                        window_score=window_data['window_score'],
                        success_probability=window_data['success_probability'],
                        weather_score=window_data['weather_score'],
                        go_for_launch=window_data['go_for_launch'],
                        risk_level=window_data['risk_level'],
                        target_orbit=window_data['target_orbit'],
                        vehicle_specs=window_data['vehicle_specs'],
                        weather_forecast_data=window_data.get('weather_forecast_data'),
                        score_breakdown=window_data.get('score_breakdown')
                    )
                    session.add(window)
                    stored_count += 1
                    
                except Exception as e:
                    logger.error(f"Error storing launch window: {e}")
                    continue
                    
        logger.info(f"Stored {stored_count} launch windows")
        return stored_count

class SystemDataAccess:
    """Data access operations for system health and monitoring."""
    
    @staticmethod
    def record_system_health(health_data: Dict) -> bool:
        """Record system health snapshot."""
        try:
            with db.session_scope() as session:
                health = SystemHealth(
                    ml_model_status=health_data['ml_model_status'],
                    weather_cache_status=health_data['weather_cache_status'],
                    scheduler_status=health_data['scheduler_status'],
                    prediction_count_24h=health_data.get('prediction_count_24h', 0),
                    trajectory_count_24h=health_data.get('trajectory_count_24h', 0),
                    weather_update_success_rate=health_data.get('weather_update_success_rate'),
                    python_version=health_data.get('python_version'),
                    system_load=health_data.get('system_load')
                )
                session.add(health)
                return True
                
        except Exception as e:
            logger.error(f"Error recording system health: {e}")
            return False
    
    @staticmethod
    def get_system_metrics(hours: int = 24) -> Dict:
        """Get system metrics for the last N hours."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        with db.session_scope() as session:
            # Prediction counts
            prediction_count = session.query(Prediction).filter(
                Prediction.created_at >= cutoff_time
            ).count()
            
            # Trajectory analysis counts
            trajectory_count = session.query(TrajectoryAnalysis).filter(
                TrajectoryAnalysis.created_at >= cutoff_time
            ).count()
            
            # Latest system health
            latest_health = session.query(SystemHealth).order_by(
                desc(SystemHealth.timestamp)
            ).first()
            
            return {
                'prediction_count': prediction_count,
                'trajectory_count': trajectory_count,
                'latest_health': {
                    'timestamp': latest_health.timestamp.isoformat() if latest_health else None,
                    'ml_model_status': latest_health.ml_model_status if latest_health else 'unknown',
                    'weather_cache_status': latest_health.weather_cache_status if latest_health else 'unknown',
                    'scheduler_status': latest_health.scheduler_status if latest_health else 'unknown'
                } if latest_health else None
            }