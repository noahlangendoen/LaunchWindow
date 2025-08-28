"""
Weather constraint checking system for launch decisions.
Separates weather assessment from ML-based success prediction.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class WeatherConstraints:
    """Launch weather constraints configuration."""
    max_wind_speed_ms: float = 15.0
    max_wind_gust_ms: float = 20.0
    min_temperature_c: float = -10.0
    max_temperature_c: float = 35.0
    max_precipitation_mm: float = 0.1
    min_visibility_m: float = 5000.0
    max_cloud_cover_pct: float = 50.0
    max_humidity_pct: float = 95.0
    min_pressure_hpa: float = 1000.0
    max_pressure_hpa: float = 1030.0


@dataclass
class WeatherAssessment:
    """Weather assessment result."""
    go_for_launch: bool
    overall_risk_level: str  # LOW, MEDIUM, HIGH, CRITICAL
    violated_constraints: List[str]
    risk_factors: List[str]
    weather_score: float  # 0-1, higher is better
    assessment_time: datetime
    confidence_level: str


class WeatherConstraintChecker:
    """
    Dedicated system for assessing weather constraints for launch decisions.
    Independent of ML model training data issues.
    """
    
    def __init__(self, constraints: Optional[WeatherConstraints] = None):
        self.constraints = constraints or WeatherConstraints()
        
    def assess_launch_weather(self, weather_data: Dict, 
                             site_specific_constraints: Optional[WeatherConstraints] = None) -> WeatherAssessment:
        """
        Assess current weather conditions against launch constraints.
        
        Args:
            weather_data: Current weather conditions
            site_specific_constraints: Optional site-specific constraints
            
        Returns:
            WeatherAssessment with go/no-go decision and detailed analysis
        """
        constraints = site_specific_constraints or self.constraints
        violated_constraints = []
        risk_factors = []
        constraint_scores = []
        
        # Wind speed assessment
        wind_speed = weather_data.get('wind_speed_ms', 0)
        if wind_speed > constraints.max_wind_speed_ms:
            violated_constraints.append(f"Wind speed {wind_speed:.1f} m/s exceeds limit {constraints.max_wind_speed_ms:.1f} m/s")
            constraint_scores.append(0.0)
        else:
            # Score based on how close to limit
            wind_score = 1.0 - (wind_speed / constraints.max_wind_speed_ms)
            constraint_scores.append(max(0.0, wind_score))
            if wind_speed > constraints.max_wind_speed_ms * 0.8:
                risk_factors.append("Wind speed approaching limits")
        
        # Wind gust assessment
        wind_gust = weather_data.get('wind_gust_ms', wind_speed)
        if wind_gust > constraints.max_wind_gust_ms:
            violated_constraints.append(f"Wind gust {wind_gust:.1f} m/s exceeds limit {constraints.max_wind_gust_ms:.1f} m/s")
            constraint_scores.append(0.0)
        else:
            gust_score = 1.0 - (wind_gust / constraints.max_wind_gust_ms)
            constraint_scores.append(max(0.0, gust_score))
            if wind_gust > constraints.max_wind_gust_ms * 0.8:
                risk_factors.append("Wind gusts approaching limits")
        
        # Temperature assessment
        temperature = weather_data.get('temperature_c', 20)
        if temperature < constraints.min_temperature_c or temperature > constraints.max_temperature_c:
            violated_constraints.append(f"Temperature {temperature:.1f}°C outside range {constraints.min_temperature_c}-{constraints.max_temperature_c}°C")
            constraint_scores.append(0.0)
        else:
            # Score based on how far from optimal (20°C)
            temp_deviation = abs(temperature - 20) / 20
            temp_score = max(0.0, 1.0 - temp_deviation)
            constraint_scores.append(temp_score)
            
        # Precipitation assessment
        precipitation = weather_data.get('rain_1h_mm', 0) + weather_data.get('snow_1h_mm', 0)
        if precipitation > constraints.max_precipitation_mm:
            violated_constraints.append(f"Precipitation {precipitation:.1f} mm exceeds limit {constraints.max_precipitation_mm:.1f} mm")
            constraint_scores.append(0.0)
        else:
            constraint_scores.append(1.0)
            
        # Visibility assessment
        visibility = weather_data.get('visibility_m', 10000)
        if visibility < constraints.min_visibility_m:
            violated_constraints.append(f"Visibility {visibility:.0f} m below minimum {constraints.min_visibility_m:.0f} m")
            constraint_scores.append(0.0)
        else:
            vis_score = min(1.0, visibility / (constraints.min_visibility_m * 2))
            constraint_scores.append(vis_score)
            
        # Cloud cover assessment
        cloud_cover = weather_data.get('cloud_cover_percent', 0)
        if cloud_cover > constraints.max_cloud_cover_pct:
            violated_constraints.append(f"Cloud cover {cloud_cover:.0f}% exceeds limit {constraints.max_cloud_cover_pct:.0f}%")
            constraint_scores.append(0.0)
        else:
            cloud_score = 1.0 - (cloud_cover / 100.0)
            constraint_scores.append(cloud_score)
            if cloud_cover > constraints.max_cloud_cover_pct * 0.8:
                risk_factors.append("High cloud cover")
        
        # Humidity assessment
        humidity = weather_data.get('humidity_percent', 50)
        if humidity > constraints.max_humidity_pct:
            risk_factors.append(f"High humidity {humidity:.0f}%")
            humidity_score = 0.5
        else:
            humidity_score = 1.0 - (humidity / 100.0)
        constraint_scores.append(humidity_score)
        
        # Pressure assessment
        pressure = weather_data.get('pressure_hpa', 1013)
        if pressure < constraints.min_pressure_hpa or pressure > constraints.max_pressure_hpa:
            risk_factors.append(f"Pressure {pressure:.0f} hPa outside normal range")
            pressure_score = 0.5
        else:
            pressure_score = 1.0
        constraint_scores.append(pressure_score)
        
        # Calculate overall weather score
        weather_score = np.mean(constraint_scores) if constraint_scores else 0.0
        
        # Determine go/no-go decision
        go_for_launch = len(violated_constraints) == 0
        
        # Risk level assessment
        if len(violated_constraints) > 0:
            risk_level = "CRITICAL"
        elif weather_score < 0.6:
            risk_level = "HIGH"
        elif weather_score < 0.8 or len(risk_factors) > 2:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
            
        # Confidence level
        if weather_score > 0.9:
            confidence = "HIGH"
        elif weather_score > 0.7:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"
        
        return WeatherAssessment(
            go_for_launch=go_for_launch,
            overall_risk_level=risk_level,
            violated_constraints=violated_constraints,
            risk_factors=risk_factors,
            weather_score=weather_score,
            assessment_time=datetime.utcnow(),
            confidence_level=confidence
        )
    
    def assess_forecast_window(self, weather_forecast: List[Dict], 
                              window_hours: int = 24) -> List[Tuple[datetime, WeatherAssessment]]:
        """
        Assess weather constraints across a forecast window.
        
        Args:
            weather_forecast: List of forecast data points
            window_hours: Hours to assess
            
        Returns:
            List of (timestamp, assessment) tuples
        """
        assessments = []
        
        for forecast_point in weather_forecast:
            try:
                forecast_time = datetime.fromisoformat(
                    forecast_point['forecast_time'].replace('Z', '+00:00')
                )
                assessment = self.assess_launch_weather(forecast_point)
                assessments.append((forecast_time, assessment))
            except (KeyError, ValueError) as e:
                continue
                
        return assessments
    
    def find_optimal_launch_windows(self, weather_forecast: List[Dict], 
                                   window_duration_hours: int = 2) -> List[Dict]:
        """
        Find optimal launch windows in forecast data.
        
        Args:
            weather_forecast: Forecast data
            window_duration_hours: Duration of each launch window
            
        Returns:
            List of optimal windows with timing and weather scores
        """
        assessments = self.assess_forecast_window(weather_forecast)
        optimal_windows = []
        
        for timestamp, assessment in assessments:
            if assessment.go_for_launch and assessment.weather_score > 0.7:
                optimal_windows.append({
                    'window_start': timestamp,
                    'weather_score': assessment.weather_score,
                    'risk_level': assessment.overall_risk_level,
                    'confidence': assessment.confidence_level,
                    'assessment': assessment
                })
        
        # Sort by weather score descending
        optimal_windows.sort(key=lambda x: x['weather_score'], reverse=True)
        
        return optimal_windows
    
    def get_site_specific_constraints(self, site_code: str) -> WeatherConstraints:
        """
        Get site-specific weather constraints.
        
        Args:
            site_code: Launch site code (KSC, VSFB, CCAFS)
            
        Returns:
            Site-specific constraints
        """
        site_constraints = {
            'KSC': WeatherConstraints(
                max_wind_speed_ms=13.0,  # More restrictive due to proximity to ocean
                max_wind_gust_ms=18.0,
                max_precipitation_mm=0.0,  # No rain for KSC
                max_cloud_cover_pct=30.0   # Lower due to lightning risk
            ),
            'VSFB': WeatherConstraints(
                max_wind_speed_ms=15.0,
                max_wind_gust_ms=20.0,
                min_temperature_c=-5.0,    # Coastal California climate
                max_temperature_c=35.0,    # Adjusted to be more realistic for coastal California
                max_precipitation_mm=0.1
            ),
            'CCAFS': WeatherConstraints(
                max_wind_speed_ms=13.0,
                max_wind_gust_ms=18.0,
                max_precipitation_mm=0.0,
                max_cloud_cover_pct=30.0   # Similar to KSC
            )
        }
        
        return site_constraints.get(site_code, self.constraints)
    
    def generate_weather_report(self, assessment: WeatherAssessment, 
                               weather_data: Dict) -> Dict:
        """
        Generate a detailed weather report for launch decision.
        
        Args:
            assessment: Weather assessment result
            weather_data: Original weather data
            
        Returns:
            Detailed weather report
        """
        return {
            'decision': {
                'go_for_launch': assessment.go_for_launch,
                'risk_level': assessment.overall_risk_level,
                'confidence': assessment.confidence_level,
                'weather_score': assessment.weather_score
            },
            'current_conditions': {
                'temperature_c': weather_data.get('temperature_c', 'N/A'),
                'wind_speed_ms': weather_data.get('wind_speed_ms', 'N/A'),
                'wind_gust_ms': weather_data.get('wind_gust_ms', 'N/A'),
                'humidity_percent': weather_data.get('humidity_percent', 'N/A'),
                'pressure_hpa': weather_data.get('pressure_hpa', 'N/A'),
                'cloud_cover_percent': weather_data.get('cloud_cover_percent', 'N/A'),
                'visibility_m': weather_data.get('visibility_m', 'N/A'),
                'precipitation_mm': weather_data.get('rain_1h_mm', 0) + weather_data.get('snow_1h_mm', 0)
            },
            'constraint_violations': assessment.violated_constraints,
            'risk_factors': assessment.risk_factors,
            'assessment_time': assessment.assessment_time.isoformat(),
            'recommendations': self._generate_weather_recommendations(assessment)
        }
    
    def _generate_weather_recommendations(self, assessment: WeatherAssessment) -> List[str]:
        """Generate recommendations based on weather assessment."""
        recommendations = []
        
        if not assessment.go_for_launch:
            recommendations.append("NO GO: Weather constraints violated. Recommend delay until conditions improve.")
            
        if assessment.overall_risk_level == "HIGH":
            recommendations.append("Monitor weather closely. Consider backup launch windows.")
            
        if assessment.weather_score < 0.8:
            recommendations.append("Marginal conditions. Review forecast trends before final decision.")
            
        if "Wind" in str(assessment.risk_factors):
            recommendations.append("Monitor wind conditions continuously during countdown.")
            
        if assessment.confidence_level == "LOW":
            recommendations.append("Low confidence in weather assessment. Consider additional meteorological analysis.")
            
        if not recommendations:
            recommendations.append("Weather conditions favorable for launch.")
            
        return recommendations