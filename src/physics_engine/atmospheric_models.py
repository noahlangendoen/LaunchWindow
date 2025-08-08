import numpy as np
import math
from dataclasses import dataclass
from typing import Tuple, Dict, Optional, List
from datetime import datetime

@dataclass
class AtmosphericConditions:
    """Represent atmospheric conditions at a specific time and altitude."""
    altitude_km: float
    temperature_k: float
    pressure_pa: float
    density_kg_m3: float
    wind_speed_ms: float
    wind_direction_deg: float
    humidity_percent: float
    timestamp: datetime

@dataclass
class DragForce:
    """Represent atmospheric drag force."""
    magnitude_n: float
    direction_deg: float
    acceleration_ms2: float

class AtmosphericModels:
    """Atmospheric modeling for launch conditions and calculations."""

    # Physical constants
    G = 9.80665  # Standard gravity in m/s^2
    EARTH_RADIUS_M = 6371000  # Earth radius in meters
    R_SPECIFIC = 287.058  # Specific gas constant for dry air in J/(kg·K)
    GAMMA = 1.4  # Ratio of specific heats for air

    # Standard atmospheric conditions at sea level
    STD_TEMP_SL = 288.15  # Kelvin
    STD_PRESSURE_SL = 101325.0  # Pascals
    STD_DENSITY_SL = 1.225  # kg/m^3

    def __init__(self):
        """Initialize atmospheric models with standard atmosphere data."""
        self.atmosphere_layers = [
            (0, -0.0065, 288.15, 101325.0),      # Troposphere
            (11000, 0.0, 216.65, 22632.1),      # Tropopause
            (20000, 0.001, 216.65, 5474.89),    # Stratosphere 1
            (32000, 0.0028, 228.65, 868.02),    # Stratosphere 2
            (47000, 0.0, 270.65, 110.91),       # Stratopause
            (51000, -0.0028, 270.65, 66.94),    # Mesosphere 1
            (71000, -0.002, 214.65, 3.96),      # Mesosphere 2
            (84852, 0.0, 186.87, 0.37),         # Mesopause
        ]

    def standard_atmosphere(self, altitude_m: float) -> AtmosphericConditions:
        """Calculate standard atmospheric conditions at a given altitude.""" 
        if altitude_m < 0:
            altitude_m = 0  # Ensure altitude is non-negative
        
        # Find the appropriate layer for the given altitude
        layer_idx = 0
        for i, (layer_altitude, _, _, _) in enumerate(self.atmosphere_layers):
            if altitude_m <= layer_altitude:
                break
            layer_idx = i

        if layer_idx >= len(self.atmosphere_layers) - 1:
            return self._high_altitude_model(altitude_m)
        
        h0, lapse_rate, base_temp, base_pressure = self.atmosphere_layers[layer_idx]

        if layer_idx > 0:
            h0, _, _, _ = self.atmosphere_layers[layer_idx - 1]
        else:
            h0 = 0

        dh = altitude_m - h0

        if abs(lapse_rate) < 1e-10:
            temperature = base_temp
            pressure = base_pressure * math.exp(-self.G * dh / (self.R_SPECIFIC * temperature))
        else:
            temperature = base_temp + lapse_rate * dh
            if temperature <= 0:
                temperature = 1.0 # Prevent divison by 0
            
            exponent = -self.G / (lapse_rate * self.R_SPECIFIC)
            pressure = base_pressure * (temperature / base_temp) ** exponent

        density = pressure / (self.R_SPECIFIC * temperature)

        return AtmosphericConditions(
            altitude_km=altitude_m / 1000.0,
            temperature_k=temperature,
            pressure_pa=pressure,
            density_kg_m3=density,
            wind_speed_ms=0.0,  # Placeholder for wind speed
            wind_direction_deg=0.0,  # Placeholder for wind direction
            humidity_percent=0.0,  # Placeholder for humidity
            timestamp=datetime.now()
        )
    
    def _high_altitude_model(self, altitude_m: float) -> AtmosphericConditions:
        """Model for high altitudes above the standard atmosphere."""
        # Use exponential decay model
        scale_height = 7400
        density_base = 0.000001 # kg/m^3 at 85 km

        density = density_base * math.exp(-(altitude_m - 85000) / scale_height)
        temperature = 186.87 # Kelvin at 85 km (roughly constant in thermosphere)
        pressure = density * self.R_SPECIFIC * temperature

        return AtmosphericConditions(
            altitude_km=altitude_m / 1000.0,
            temperature_k=temperature,
            pressure_pa=pressure,
            density_kg_m3=density,
            wind_speed_ms=0.0,  # Placeholder for wind speed
            wind_direction_deg=0.0,  # Placeholder for wind direction
            humidity_percent=0.0,  # Placeholder for humidity
            timestamp=datetime.now()
        )
    
    def weather_adjusted_atmosphere(self, altitude_m: float, surface_weather: Dict) -> AtmosphericConditions:
        """Calculate atmospheric conditions adjusted for actual weather data."""

        # Get standard atmosphere as baseline
        std_atmosphere = self.standard_atmosphere(altitude_m)

        # Extract weather data
        temp_k = surface_weather.get('temperature_c', 15) + 273.15  # Convert Celsius to Kelvin
        pressure_pa = surface_weather.get('pressure_hpa', 1013.25) * 100 # Convert hPa to Pa
        humidity_percent = surface_weather.get('humidity_percent', 50)
        wind_speed = surface_weather.get('wind_speed_ms', 0.0)
        wind_direction = surface_weather.get('wind_direction_deg', 0.0)

        # Calculate temperature adjustment
        std_surface_temp = 288.15  # Standard temperature at sea level in Kelvin
        temp_offset = temp_k - std_surface_temp

        # Adjust temperature based on conditions
        if altitude_m < 11000: # Troposphere
            temp_factor = max(0, 1 - (altitude_m / 11000))
            adjusted_temp = std_atmosphere.temperature_k + temp_factor * temp_offset
        else:
            adjusted_temp = std_atmosphere.temperature_k

        # Adjust pressure based on conditions and temperature
        std_surface_pressure = 101325.0  # Standard pressure at sea level in Pa
        pressure_ratio = pressure_pa / std_surface_pressure

        # Scale altitude for pressure calculation
        if altitude_m < 11000:
            h_scale = self.R_SPECIFIC * adjusted_temp / self.G
            adjusted_pressure = pressure_pa * math.exp(-altitude_m / h_scale)
        else:
            adjusted_pressure = std_atmosphere.pressure_pa * pressure_ratio
        
        # Calculate density from adjusted pressure and temperature
        adjusted_density = adjusted_pressure / (self.R_SPECIFIC * adjusted_temp)

        # Adjust humidity (decreases with altitude)
        humidity_scale_height = 2000 
        adjusted_humidity = humidity_percent * math.exp(-altitude_m / humidity_scale_height)

        # Wind profile - tyically increases with altitude in troposphere
        if altitude_m < 11000:
            # Power law wind profile
            wind_exponent = 0.143  # Typical for open terrain
            wind_factor = (altitude_m / 10) ** wind_exponent if altitude_m > 10 else 1.0
            adjusted_wind_speed = wind_speed * wind_factor
        else:
            # Jet stream effects above troposphere
            if 9000 < altitude_m < 15000:
                # Potential jet stream influence
                adjusted_wind_speed = wind_speed * (1 + (altitude_m - 9000) / 6000 * 2)
            else:
                adjusted_wind_speed = wind_speed * 0.5  # Reduced wind speed at very high altitudes
        
        return AtmosphericConditions(
            altitude_km=altitude_m / 1000.0,
            temperature_k=adjusted_temp,
            pressure_pa=adjusted_pressure,
            density_kg_m3=adjusted_density,
            wind_speed_ms=adjusted_wind_speed,
            wind_direction_deg=wind_direction,
            humidity_percent=adjusted_humidity,
            timestamp=datetime.now()
        )
    
    def calculate_drag_force(self, velocity_ms: float, cross_sectional_area_m2: float, drag_coefficient: float,
                             atmospheric_conditions: AtmosphericConditions, vehicle_heading_deg: float = 0.0) -> DragForce:
        """Calculate the drag force acting on a vehicle in the atmosphere."""
        # Air density
        rho = atmospheric_conditions.density_kg_m3

        # Relative velocity vector (accounting for wind)
        wind_velocity_x = atmospheric_conditions.wind_speed_ms * math.cos(math.radians(atmospheric_conditions.wind_direction_deg))
        wind_velocity_y = atmospheric_conditions.wind_speed_ms * math.sin(math.radians(atmospheric_conditions.wind_direction_deg))

        vehicle_velocity_x = velocity_ms * math.cos(math.radians(vehicle_heading_deg))
        vehicle_velocity_y = velocity_ms * math.sin(math.radians(vehicle_heading_deg))

        relative_velocity_x = vehicle_velocity_x - wind_velocity_x
        relative_velocity_y = vehicle_velocity_y - wind_velocity_y
        relative_velocity = math.sqrt(relative_velocity_x ** 2 + relative_velocity_y ** 2)

        # Drag force magnitude
        drag_magnitude = 0.5 * rho * relative_velocity ** 2 * drag_coefficient * cross_sectional_area_m2

        # Drag direction (opposite to relative velocity)
        if relative_velocity > 0:
            drag_direction = math.degrees(math.atan2(-relative_velocity_y, -relative_velocity_x))
        else:
            drag_direction = 0

        # Drag acceleration assuming mass is handled elsewhere
        drag_acceleration = drag_magnitude

        return DragForce(
            magnitude_n=drag_magnitude,
            direction_deg=drag_direction,
            acceleration_ms2=drag_acceleration
        )
    
    def calculate_mach_number(self, velocity_ms: float, atmospheric_conditions: AtmosphericConditions) -> float:
        """Calculate the Mach number based on vehicle speed and atmospheric conditions."""
        # Speed of sound in the atmosphere
        speed_of_sound = math.sqrt(self.GAMMA * self.R_SPECIFIC * atmospheric_conditions.temperature_k)

        mach_number = velocity_ms / speed_of_sound
        return mach_number if mach_number >= 0 else 0.0
    
    def calculate_reynolds_number(self, velocity_ms: float, characteristic_length_m: float, atmospheric_conditions: AtmosphericConditions) -> float:
        """Calculate the Reynolds number for flow analysis."""
        # Dynamic viscosity of air (approximation via Sutherland's formula)
        T = atmospheric_conditions.temperature_k
        mu_ref = 1.716e-5
        T_ref = 273.15
        S = 110.4

        dynamic_viscosity = mu_ref * (T / T_ref) ** 1.5 * (T_ref + S) / (T + S)

        # Reynolds number calculation
        reynolds_number = (atmospheric_conditions.density_kg_m3 * velocity_ms * characteristic_length_m) / dynamic_viscosity

        return reynolds_number if reynolds_number >= 0 else 0.0
    
    def estimate_heat_flux(self, velocity_ms: float, atmospheric_conditions: AtmosphericConditions, nose_radius_m: float = 1.0) -> float:
        """Estimate heating rate on vehicle nose (Detra-Kemp-Riddell correlation)."""
        # Simplified heating estimation
        rho = atmospheric_conditions.density_kg_m3
        V = velocity_ms
        R_n = nose_radius_m

        heat_flux = 1.83e-4 * math.sqrt(rho / R_n) * V ** 3
        return heat_flux if heat_flux >= 0 else 0.0
    
    def calculate_atmospheric_density_variation(self, base_altitude_m: float, weather_data: Dict, time_hours: float = 24) -> Dict:
        """Calculate how atmospheric density varies with weather changes."""
        variations = {}

        # Base atmospheric conditions
        base_conditions = self.weather_adjusted_atmosphere(base_altitude_m, weather_data)
        base_density = base_conditions.density_kg_m3

        # Temperature variations (+- 10K on a daily basis)
        temp_variations = [-10, -5, 0, 5, 10]
        variations['temperature'] = {}

        for temp_delta in temp_variations:
            modified_weather = weather_data.copy()
            modified_weather['temperature_c'] = weather_data.get('temperature_c', 15) + temp_delta

            modified_conditions = self.weather_adjusted_atmosphere(base_altitude_m, modified_weather)
            density_changes_percent = ((modified_conditions.density_kg_m3 - base_density) / base_density) * 100

            variations['temperature'][temp_delta] = {
                'density_kg_m3': modified_conditions.density_kg_m3,
                'change_percent': density_changes_percent
            }

        # Pressure variations (+- 20hPa on a daily basis)
        pressure_variations = [-20, -10, 0, 10, 20]
        variations['pressure'] = {}

        for pressure_delta in pressure_variations:
            modified_weather = weather_data.copy()
            current_pressure = weather_data.get('pressure_hpa', 1013.25)
            modified_weather['pressure_hpa'] = current_pressure + pressure_delta

            modified_conditions = self.weather_adjusted_atmosphere(base_altitude_m, modified_weather)
            density_changes_percent = ((modified_conditions.density_kg_m3 - base_density) / base_density) * 100

            variations['pressure'][pressure_delta] = {
                'density_kg_m3': modified_conditions.density_kg_m3,
                'change_percent': density_changes_percent
            }

        return variations
    
    def get_atmospheric_profile(self, max_altitude_m: float, weather_data: Dict, step_size_m: float = 1000) -> List[AtmosphericConditions]:
        """Generate atmospheric profile from surface to max altitude."""
        profile = []

        altitude = 0.0
        while altitude <= max_altitude_m:
            conditions = self.weather_adjusted_atmosphere(altitude, weather_data)
            profile.append(conditions)
            altitude += step_size_m

        return profile
    
    def calculate_crosswind_impact(self, vehicle_velocity_ms: float, crosswind_ms: float, vehicle_mass_kg: float,
                                    cross_sectional_area_m2: float, drag_coefficient: float, atmospheric_density: float) -> Dict:
        """Calculate the impact of crosswind on vehicle trajectory."""
        # Crosswind force
        crosswind_force = 0.5 * atmospheric_density * crosswind_ms ** 2 * drag_coefficient * cross_sectional_area_m2

        # Lateral acceleration due to crosswind
        lateral_acceleration = crosswind_force / vehicle_mass_kg

        # Trajectory deviation over time (simplified)
        # This would be integrated in a real time trajectory simulation
        deviation_rate_m_per_s = lateral_acceleration

        wind_correction_angle = math.degrees(math.atan2(crosswind_ms, vehicle_velocity_ms))

        return {
            'crosswind_force_n': crosswind_force,
            'lateral_acceleration_ms2': lateral_acceleration,
            'deviation_rate_m_per_s': deviation_rate_m_per_s,
            'wind_correction_angle_deg': wind_correction_angle,
            'impact_severity': 'HIGH' if abs(wind_correction_angle) > 5 else 'MEDIUM' if abs(wind_correction_angle) > 2 else 'LOW'
        }
    
    def assess_weather_launch_constraints(self, weather_data: Dict, vehicle_specs: Dict) -> Dict:
        """Assess weather conditions against launch constraints."""
        constraints = {
            'go_for_launch': True,
            'violations': [],
            'warnings': [],
            'conditions_assessment': {}
        }

        # Extract weather data
        wind_speed = weather_data.get('wind_speed_ms', 0.0)
        wind_gust = weather_data.get('wind_gust_ms', wind_speed)
        temperature = weather_data.get('temperature_c', 15)
        precipitation = weather_data.get('rain_1h_mm', 0.0) + weather_data.get('snow_1h_mm', 0.0)
        cloud_cover = weather_data.get('cloud_cover_percent', 0)
        visibility = weather_data.get('visibility_m', 10000)

        # Vehicle specifications
        max_wind_speed = vehicle_specs.get('max_wind_speed_ms', 15)
        max_wind_gust = vehicle_specs.get('max_wind_gust_ms', 20)
        min_temperature = vehicle_specs.get('min_temperature_c', -10)
        max_temperature = vehicle_specs.get('max_temperature_c', 35)
        max_precipitation = vehicle_specs.get('max_precipitation_mm', 0)
        min_visibility = vehicle_specs.get('min_visibility_m', 5000)
        max_cloud_cover = vehicle_specs.get('max_cloud_cover_percent', 50)

        # Check constraints
        if wind_speed > max_wind_speed:
            constraints['go_for_launch'] = False
            constraints['violations'].append(f"Wind speed {wind_speed:.1f} m/s exceeds limit {max_wind_speed} m/s.")

        if wind_gust > max_wind_gust:
            constraints['go_for_launch'] = False
            constraints['violations'].append(f"Wind gust {wind_gust:.1f} m/s exceeds limit {max_wind_gust} m/s.")
        
        if temperature < min_temperature or temperature > max_temperature:
            constraints['go_for_launch'] = False
            constraints['violations'].append(f"Temperature {temperature:.1f}°C outside acceptable range {min_temperature}-{max_temperature}°C")

        if precipitation > max_precipitation:
            constraints['go_for_launch'] = False
            constraints['violations'].append(f"Precipitation {precipitation:.1f} mm exceeds limit {max_precipitation} mm.")

        if visibility < min_visibility:
            constraints['go_for_launch'] = False
            constraints['violations'].append(f"Visibility {visibility:.1f} m below minimum {min_visibility} m.")

        if cloud_cover > max_cloud_cover:
            constraints['warnings'].append(f"Cloud cover {cloud_cover:.1f}% above recommended maximum {max_cloud_cover}%.")

        # Assess individual conditions
        constraints['conditions_assessment'] = {
            'wind': 'GOOD' if wind_speed < max_wind_speed * 0.5 else 'MARGINAL' if wind_speed < max_wind_speed else 'BAD',
            'temperature': 'GOOD' if min_temperature + 5 <= temperature <= max_temperature - 5 else 'MARGINAL',
            'precipitation': 'GOOD' if precipitation == 0 else 'POOR',
            'visibility': 'GOOD' if visibility > min_visibility * 2 else 'MARGINAL' if visibility > min_visibility else 'POOR',
            'overall': 'GO' if constraints['go_for_launch'] else 'NO GO'
        }

        return constraints
    
    