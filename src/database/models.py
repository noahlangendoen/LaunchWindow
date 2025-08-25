"""
Database models for Launch Window Prediction System using SQLAlchemy ORM.
Designed for PostgreSQL on Railway deployment.
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, Text, JSON, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
import uuid

Base = declarative_base()

class LaunchSite(Base):
    """Launch site information and coordinates."""
    __tablename__ = 'launch_sites'
    
    site_code = Column(String(10), primary_key=True)  # KSC, VSFB, CCAFS
    site_name = Column(String(100), nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    altitude_m = Column(Float, default=0)
    timezone = Column(String(50), nullable=False)
    
    # Relationships
    launches = relationship("Launch", back_populates="launch_site")
    weather_current = relationship("WeatherCurrent", back_populates="site")
    weather_forecast = relationship("WeatherForecast", back_populates="site")

class Launch(Base):
    """Historical launch data from SpaceX and other providers."""
    __tablename__ = 'launches'
    
    id = Column(Integer, primary_key=True)
    flight_number = Column(Integer, unique=True, nullable=True)
    name = Column(String(100), nullable=False)
    date_utc = Column(DateTime, nullable=False)
    date_local = Column(DateTime, nullable=False)
    success = Column(Boolean, nullable=True)
    upcoming = Column(Boolean, default=False)
    
    # Rocket information
    rocket_name = Column(String(50), nullable=False)
    rocket_type = Column(String(50), nullable=False)
    
    # Launch site
    site_code = Column(String(10), ForeignKey('launch_sites.site_code'), nullable=False)
    
    # Mission details
    payload_mass_kg = Column(Float, nullable=True)
    payload_types = Column(JSON, nullable=True)  # Store as JSON array
    failures = Column(JSON, nullable=True)  # Store failure details as JSON
    
    # Additional metadata
    details = Column(Text, nullable=True)
    webcast = Column(String(200), nullable=True)
    wikipedia = Column(String(200), nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    launch_site = relationship("LaunchSite", back_populates="launches")
    
    # Indexes for performance
    __table_args__ = (
        Index('ix_launches_date_utc', 'date_utc'),
        Index('ix_launches_site_code', 'site_code'),
        Index('ix_launches_success', 'success'),
    )

class WeatherCurrent(Base):
    """Current weather data for launch sites."""
    __tablename__ = 'weather_current'
    
    id = Column(Integer, primary_key=True)
    site_code = Column(String(10), ForeignKey('launch_sites.site_code'), nullable=False)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # Temperature data
    temperature_c = Column(Float, nullable=False)
    feels_like_c = Column(Float, nullable=True)
    temp_min_c = Column(Float, nullable=True)
    temp_max_c = Column(Float, nullable=True)
    
    # Atmospheric conditions
    humidity_percent = Column(Float, nullable=True)
    pressure_hpa = Column(Float, nullable=True)
    
    # Wind data
    wind_speed_ms = Column(Float, nullable=True)
    wind_direction_deg = Column(Float, nullable=True)
    wind_gust_ms = Column(Float, nullable=True)
    
    # Weather description
    weather_main = Column(String(50), nullable=True)
    weather_description = Column(String(100), nullable=True)
    
    # Visibility and clouds
    visibility_m = Column(Float, nullable=True)
    cloud_cover_percent = Column(Float, nullable=True)
    
    # Precipitation
    rain_1h_mm = Column(Float, default=0)
    rain_3h_mm = Column(Float, default=0)
    snow_1h_mm = Column(Float, default=0)
    snow_3h_mm = Column(Float, default=0)
    
    # Data quality
    data_source = Column(String(50), default='live')  # 'live' or 'mock'
    expires_at = Column(DateTime, nullable=True)
    
    # Relationships
    site = relationship("LaunchSite", back_populates="weather_current")
    
    # Indexes
    __table_args__ = (
        Index('ix_weather_current_site_timestamp', 'site_code', 'timestamp'),
        Index('ix_weather_current_expires_at', 'expires_at'),
    )

class WeatherForecast(Base):
    """Weather forecast data for launch planning."""
    __tablename__ = 'weather_forecast'
    
    id = Column(Integer, primary_key=True)
    site_code = Column(String(10), ForeignKey('launch_sites.site_code'), nullable=False)
    forecast_time = Column(DateTime, nullable=False)  # When the forecast is for
    retrieved_at = Column(DateTime, nullable=False, default=datetime.utcnow)  # When we got the data
    
    # Temperature data
    temperature_c = Column(Float, nullable=False)
    feels_like_c = Column(Float, nullable=True)
    temp_min_c = Column(Float, nullable=True)
    temp_max_c = Column(Float, nullable=True)
    
    # Atmospheric conditions
    humidity_percent = Column(Float, nullable=True)
    pressure_hpa = Column(Float, nullable=True)
    
    # Wind data
    wind_speed_ms = Column(Float, nullable=True)
    wind_direction_deg = Column(Float, nullable=True)
    wind_gust_ms = Column(Float, nullable=True)
    
    # Weather description
    weather_main = Column(String(50), nullable=True)
    weather_description = Column(String(100), nullable=True)
    
    # Visibility and clouds
    visibility_m = Column(Float, nullable=True)
    cloud_cover_percent = Column(Float, nullable=True)
    
    # Precipitation
    rain_1h_mm = Column(Float, default=0)
    rain_3h_mm = Column(Float, default=0)
    snow_1h_mm = Column(Float, default=0)
    snow_3h_mm = Column(Float, default=0)
    
    # Precipitation probability
    pop = Column(Float, nullable=True)  # Probability of precipitation
    
    # Data quality
    data_source = Column(String(50), default='live')
    expires_at = Column(DateTime, nullable=True)
    
    # Relationships
    site = relationship("LaunchSite", back_populates="weather_forecast")
    
    # Indexes
    __table_args__ = (
        Index('ix_weather_forecast_site_forecast_time', 'site_code', 'forecast_time'),
        Index('ix_weather_forecast_expires_at', 'expires_at'),
    )

class Prediction(Base):
    """ML model predictions and analysis results."""
    __tablename__ = 'predictions'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    site_code = Column(String(10), ForeignKey('launch_sites.site_code'), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # ML Model Results
    ml_success_probability = Column(Float, nullable=False)
    ml_model_type = Column(String(50), nullable=False)  # 'random_forest', 'xgboost', etc.
    ml_confidence_level = Column(String(20), nullable=False)  # 'HIGH', 'MEDIUM', 'LOW'
    
    # Weather Assessment
    weather_score = Column(Float, nullable=False)
    weather_go_for_launch = Column(Boolean, nullable=False)
    weather_risk_level = Column(String(20), nullable=False)  # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    weather_violations = Column(JSON, nullable=True)  # Array of constraint violations
    
    # Combined Assessment
    combined_success_probability = Column(Float, nullable=False)
    overall_recommendation = Column(String(100), nullable=False)
    overall_risk_level = Column(String(20), nullable=False)
    
    # Input Parameters
    rocket_specs = Column(JSON, nullable=True)  # Vehicle specifications used
    target_orbit = Column(JSON, nullable=True)  # Target orbit parameters
    weather_data = Column(JSON, nullable=True)  # Weather conditions at prediction time
    
    # Methodology
    methodology = Column(String(100), nullable=False)
    
    # Indexes
    __table_args__ = (
        Index('ix_predictions_site_created', 'site_code', 'created_at'),
        Index('ix_predictions_success_prob', 'combined_success_probability'),
    )

class TrajectoryAnalysis(Base):
    """Physics-based trajectory calculation results."""
    __tablename__ = 'trajectory_analyses'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    prediction_id = Column(UUID(as_uuid=True), ForeignKey('predictions.id'), nullable=True)
    site_code = Column(String(10), ForeignKey('launch_sites.site_code'), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Trajectory Results
    success_probability = Column(Float, nullable=False)
    mission_objectives_met = Column(Boolean, nullable=False)
    max_dynamic_pressure = Column(Float, nullable=False)
    max_g_force = Column(Float, nullable=False)
    fuel_remaining_kg = Column(Float, nullable=False)
    total_delta_v = Column(Float, nullable=False)
    
    # Vehicle Specifications Used
    vehicle_name = Column(String(50), nullable=False)
    vehicle_specs = Column(JSON, nullable=False)  # Complete vehicle specification
    
    # Target Orbit
    target_orbit = Column(JSON, nullable=False)
    final_orbit_elements = Column(JSON, nullable=True)
    
    # Weather Conditions
    weather_data = Column(JSON, nullable=False)
    
    # Full trajectory data (optional, for detailed analysis)
    trajectory_points = Column(JSON, nullable=True)  # Can be large, store selectively
    
    # Indexes
    __table_args__ = (
        Index('ix_trajectory_site_created', 'site_code', 'created_at'),
        Index('ix_trajectory_success_prob', 'success_probability'),
    )

class LaunchWindow(Base):
    """Optimal launch window calculations."""
    __tablename__ = 'launch_windows'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    site_code = Column(String(10), ForeignKey('launch_sites.site_code'), nullable=False)
    calculated_at = Column(DateTime, default=datetime.utcnow)
    
    # Window Details
    launch_time = Column(DateTime, nullable=False)
    window_duration_hours = Column(Float, nullable=False)
    window_score = Column(Float, nullable=False)
    success_probability = Column(Float, nullable=False)
    
    # Conditions
    weather_score = Column(Float, nullable=False)
    go_for_launch = Column(Boolean, nullable=False)
    risk_level = Column(String(20), nullable=False)
    
    # Parameters Used
    target_orbit = Column(JSON, nullable=False)
    vehicle_specs = Column(JSON, nullable=False)
    weather_forecast_data = Column(JSON, nullable=True)
    
    # Score Breakdown
    score_breakdown = Column(JSON, nullable=True)  # Detailed scoring components
    
    # Indexes
    __table_args__ = (
        Index('ix_launch_windows_site_time', 'site_code', 'launch_time'),
        Index('ix_launch_windows_score', 'window_score'),
    )

class SystemHealth(Base):
    """System health and monitoring data."""
    __tablename__ = 'system_health'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Component Status
    ml_model_status = Column(String(20), nullable=False)  # 'trained', 'not_trained'
    weather_cache_status = Column(String(20), nullable=False)  # 'live', 'mock', 'offline'
    scheduler_status = Column(String(20), nullable=False)  # 'running', 'stopped'
    
    # Performance Metrics
    prediction_count_24h = Column(Integer, default=0)
    trajectory_count_24h = Column(Integer, default=0)
    weather_update_success_rate = Column(Float, nullable=True)
    
    # System Info
    python_version = Column(String(50), nullable=True)
    system_load = Column(JSON, nullable=True)  # CPU, memory, etc.
    
    # Indexes
    __table_args__ = (
        Index('ix_system_health_timestamp', 'timestamp'),
    )