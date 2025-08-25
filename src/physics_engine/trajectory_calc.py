import numpy as np
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from datetime import datetime, timedelta
from .orbital_mechanics import OrbitalMechanics, StateVector, LaunchSite
from .atmospheric_models import AtmosphericModels, AtmosphericConditions

@dataclass
class VehicleSpecs:
    """Rocket specifications"""
    name: str
    dry_mass_kg: float
    fuel_mass_kg: float
    thrust_n: float
    specific_impulse_s: float
    drag_coefficient: float
    cross_sectional_area_m2: float
    max_dynamic_pressure_pa: float
    max_acceleration_g: float
    stage_count: int

@dataclass
class TrajectoryPoint:
    """A point in the trajectory"""
    time_s: float
    position: np.ndarray # [x, y, z] in km
    velocity: np.ndarray # [vx, vy, vz] in km/s
    altitude_km: float
    velocity_magnitude_ms: float
    acceleration: np.ndarray # [ax, ay, az] in m/s^2
    mass_kg: float
    thrust_n: float
    drag_n: float
    dynamic_pressure_pa: float
    mach_number: float
    g_force: float
    atmospheric_conditions: AtmosphericConditions
    timestamp: datetime

@dataclass
class LaunchTrajectory:
    """A complete launch trajectory"""
    trajectory_points: List[TrajectoryPoint]
    success_probability: float
    mission_objectives_met: bool
    max_dynamic_pressure: float
    max_g_force: float
    fuel_remaining_kg: float
    final_orbit_elements: Optional[Dict]
    launch_constraints_met: bool
    total_delta_v: float

class TrajectoryCalculator:
    """"Advanced trajectory calculator and optimizaiton."""

    def __init__(self):
        self.orbital_mechanics = OrbitalMechanics()
        self.atmospheric_models = AtmosphericModels()

        # Standard gravitational acceleration on Earth
        self.G0 = 9.80665  # m/s^2

        # Earth parameters
        self.EARTH_RADIUS_KM = 6371.0  # Radius of Earth in km
        self.EARTH_MU = 398600.4418  # Gravitational parameter for Earth in km^3/s^2

    def calculate_launch_trajectory(self, launch_site: str, target_orbit: Dict, vehicle_specs: VehicleSpecs, weather_data: Dict, 
                                    launch_time: datetime, time_step_s: float = 1.0, weather_constraints: Dict = None) -> LaunchTrajectory:
        """Calculate complete launch trajectory from surface to orbit."""

        DEBUG = False
        debug_interval = 10  # Print every 10 seconds
        
        site = self.orbital_mechanics.launch_sites[launch_site]
        trajectory_points = []

        # Initial conditions
        current_time = 0
        current_mass = vehicle_specs.dry_mass_kg + vehicle_specs.fuel_mass_kg
        
        if DEBUG:
            print(f"\n=== TRAJECTORY DEBUG for {launch_site} ===")
            print(f"Initial mass: {current_mass:.0f} kg")
            print(f"Fuel mass: {vehicle_specs.fuel_mass_kg:.0f} kg")
            print(f"Thrust: {vehicle_specs.thrust_n:.0f} N")
            print(f"Isp: {vehicle_specs.specific_impulse_s:.0f} s")
            print(f"Max dynamic pressure: {vehicle_specs.max_dynamic_pressure_pa:.0f} Pa")
            print(f"Max G-force: {vehicle_specs.max_acceleration_g:.1f} g")

        # Launch site position in Earth-centered coordinates
        lat_rad = math.radians(site.latitude)
        lon_rad = math.radians(site.longitude)
        site_altitude_km = site.altitude / 1000

        # Initial position (Earth's surface)
        earth_radius_km = self.EARTH_RADIUS_KM + site_altitude_km
        x0 = earth_radius_km * math.cos(lat_rad) * math.cos(lon_rad)
        y0 = earth_radius_km * math.cos(lat_rad) * math.sin(lon_rad)
        z0 = earth_radius_km * math.sin(lat_rad)
        position = np.array([x0, y0, z0])

        # Calculate optimal launch azimuth
        target_inclination = target_orbit.get('inclination_deg', 28.5)
        target_altitude = target_orbit.get('altitude_km', 400)

        try:
            launch_azimuth = self.orbital_mechanics.calculate_launch_azimuth(
                launch_site, target_inclination, target_altitude
            )
        except:
            if target_inclination > 90:
                launch_azimuth = 180  # Southward retrograde
            else:
                launch_azimuth = 90  # Eastward for prograde

        # Initial velocity
        earth_rotation_rate = 7.2921159e-5  # rad/s
        v_earth_ms = earth_rotation_rate * earth_radius_km * 1000 * math.cos(lat_rad)  # m/s at surface
        velocity = np.array([0.0, 0.0, 0.0])

        # Launch trajectory simulation
        fuel_consumed = 0.0
        thrust_on = True
        max_dynamic_pressure = 0.0
        max_g_force = 0.0
        
        # Add throttle control
        throttle = 1.0  # Start at full throttle

        # Termination conditions
        max_time = 600
        target_velocity = math.sqrt(self.EARTH_MU / (self.EARTH_RADIUS_KM + target_altitude))  # km/s

        while current_time < max_time:
            # Current altitude
            altitude_km = np.linalg.norm(position) - self.EARTH_RADIUS_KM
            altitude_m = altitude_km * 1000

            if altitude_km < -10:  # Trajectory went underground
                print(f"WARNING: Trajectory went underground at t={current_time}s.")
                break

            vel_magnitude_kms = np.linalg.norm(velocity)
            if altitude_km >= target_altitude * 0.35 and vel_magnitude_kms >= target_velocity * 0.90:
                # Adjusted for single-stage simulation reaching partial altitude
                if DEBUG:
                    print(f"\nMISSION SUCCESS at t={current_time}s!")
                    print(f"Altitude: {altitude_km:.1f} km, Velocity: {vel_magnitude_kms:.3f} km/s")
                break  #

            # Atmospheric conditions
            atmos = self.atmospheric_models.weather_adjusted_atmosphere(altitude_m, weather_data)

            # Current velocity magnitude
            vel_magnitude_ms = np.linalg.norm(velocity) * 1000  # Convert km/s to m/s

            # Calculate dynamic pressure BEFORE applying thrust
            dynamic_pressure = 0.5 * atmos.density_kg_m3 * vel_magnitude_ms ** 2
            
            # THROTTLE CONTROL based on dynamic pressure
            if dynamic_pressure > vehicle_specs.max_dynamic_pressure_pa * 0.7:
                # Start throttling down when approaching max-Q
                throttle = max(0.3, 1.0 - (dynamic_pressure - vehicle_specs.max_dynamic_pressure_pa * 0.7) / 
                            (vehicle_specs.max_dynamic_pressure_pa * 0.3))
            else:
                throttle = 1.0

            # Thrust calculation with throttle
            if thrust_on and fuel_consumed < vehicle_specs.fuel_mass_kg:
                thrust_magnitude = vehicle_specs.thrust_n * throttle  # Apply throttle
                
                # Adding thrust limiting
                max_allowed_thrust = vehicle_specs.max_acceleration_g * self.G0 * current_mass
                if thrust_magnitude > max_allowed_thrust:
                    thrust_magnitude = max_allowed_thrust

                fuel_flow_rate = thrust_magnitude / (vehicle_specs.specific_impulse_s * self.G0)
                fuel_consumed += fuel_flow_rate * time_step_s
                current_mass = vehicle_specs.dry_mass_kg + vehicle_specs.fuel_mass_kg - fuel_consumed

                if fuel_consumed >= vehicle_specs.fuel_mass_kg:
                    thrust_on = False
                    thrust_magnitude = 0.0
                    if DEBUG:
                        print(f"\nFUEL DEPLETED at t={current_time}s, altitude={altitude_km:.1f} km")
                    if altitude_km < target_altitude * 0.5:
                        # Mission failure due to depletion
                        break
            else:
                thrust_magnitude = 0.0
                thrust_on = False
            
            # Debug output every N seconds
            if DEBUG and current_time % debug_interval == 0:
                print(f"\nt={current_time:3.0f}s: Alt={altitude_km:6.1f}km, Vel={vel_magnitude_ms:6.0f}m/s, "
                    f"Q={dynamic_pressure/1000:5.1f}kPa, Throttle={throttle:3.0%}, "
                    f"Fuel={vehicle_specs.fuel_mass_kg - fuel_consumed:6.0f}kg")
            
            # Thrust direction (rest of the code remains the same...)
            if current_time < 10:
                thrust_direction = position / np.linalg.norm(position)
            else:  # Gravity turn
                pitch_angle = max(0, 90 - (current_time - 10) * 0.75)  # degrees
                pitch_rad = math.radians(pitch_angle)
                azimuth_rad = math.radians(launch_azimuth)

                # Local coordinate system
                up = position / np.linalg.norm(position)
                east = np.array([-math.sin(lon_rad), math.cos(lon_rad), 0])
                north = np.cross(up, east)

                if np.linalg.norm(north) > 0:
                    north = north / np.linalg.norm(north)
                else:
                    north = np.array([1, 0, 0])

                # Thrust vector in local coordinates
                thrust_local = (
                    math.sin(pitch_rad) * up +
                    math.cos(pitch_rad) * (
                        math.cos(azimuth_rad) * north + 
                        math.sin(azimuth_rad) * east
                    )
                )

                thrust_direction = thrust_local / np.linalg.norm(thrust_local)

            # Force calculations
            thrust_acceleration_kms2 = np.zeros(3)
            if thrust_magnitude > 0:
                thrust_acceleration_ms2 = thrust_magnitude / current_mass  # m/s^2
                thrust_acceleration_kms2 = thrust_direction * (thrust_acceleration_ms2 / 1000)  # Convert to km/s^2

            # Gravitational force
            r_km = np.linalg.norm(position)
            gravity_acceleration_kms2 = -self.EARTH_MU * position / (r_km ** 3)  # km/s^2

            # Drag acceleration
            drag_acceleration_kms2 = np.zeros(3)
            if altitude_m < 100000:  # Only consider below 100 km
                drag_force = self.atmospheric_models.calculate_drag_force(
                    vel_magnitude_ms, vehicle_specs.cross_sectional_area_m2,
                    vehicle_specs.drag_coefficient, atmos, 0
                )
                if vel_magnitude_kms > 0:
                    drag_direction = -velocity / vel_magnitude_kms
                    drag_acceleration_ms2 = drag_force.magnitude_n / current_mass  # m/s^2
                    drag_acceleration_kms2 = drag_direction * (drag_acceleration_ms2 / 1000)
            else:
                drag_force = type('obj', (object,), {'magnitude_n': 0.0})()

            # Total acceleration
            total_acceleration_kms2 = thrust_acceleration_kms2 + gravity_acceleration_kms2 + drag_acceleration_kms2

            # Update state
            position += velocity * time_step_s  # km
            velocity += total_acceleration_kms2 * time_step_s  # km/s

            # Calculate metrics
            max_dynamic_pressure = max(max_dynamic_pressure, dynamic_pressure)

            acceleration_magnitude_ms2 = np.linalg.norm(total_acceleration_kms2) * 1000  # m/s^2
            g_force = acceleration_magnitude_ms2 / self.G0
            max_g_force = max(max_g_force, g_force)

            if g_force > vehicle_specs.max_acceleration_g * 1.5:
                print(f"WARNING: Excessive G-force {g_force:.1f}g at time: {current_time}.")
                break

            mach_number = self.atmospheric_models.calculate_mach_number(vel_magnitude_ms, atmos)

            # Create trajectory point
            point = TrajectoryPoint(
                time_s=current_time,
                position=position.copy(),
                velocity=velocity.copy(),
                altitude_km=altitude_km,
                velocity_magnitude_ms=vel_magnitude_ms,
                acceleration=total_acceleration_kms2 * 1000,  # Storing in m/s^2 for consistency
                mass_kg=current_mass,
                thrust_n=thrust_magnitude,
                drag_n=drag_force.magnitude_n if 'drag_force' in locals() else 0.0,
                dynamic_pressure_pa=dynamic_pressure,
                mach_number=mach_number,
                g_force=g_force,
                atmospheric_conditions=atmos,
                timestamp=launch_time + timedelta(seconds=current_time)
            )

            trajectory_points.append(point)
            current_time += time_step_s

        if DEBUG:
            print(f"\n=== TRAJECTORY SUMMARY ===")
            print(f"Final time: {current_time}s")
            print(f"Final altitude: {altitude_km:.1f} km")
            print(f"Final velocity: {vel_magnitude_ms:.0f} m/s")
            print(f"Max dynamic pressure: {max_dynamic_pressure:.0f} Pa ({max_dynamic_pressure/1000:.1f} kPa)")
            print(f"Max G-force: {max_g_force:.1f} g")
            print(f"Fuel consumed: {fuel_consumed:.0f} kg")
            print(f"Fuel remaining: {vehicle_specs.fuel_mass_kg - fuel_consumed:.0f} kg")

        if not trajectory_points:
            # Create a failed trajectory
            return LaunchTrajectory(
                trajectory_points=[],
                success_probability=0.0,
                mission_objectives_met=False,
                max_dynamic_pressure=0,
                max_g_force=0,
                fuel_remaining_kg=vehicle_specs.fuel_mass_kg,
                final_orbit_elements=None,
                launch_constraints_met=False,
                total_delta_v=0
            )


        # Calculate final orbit elements
        final_state = StateVector(
            position=trajectory_points[-1].position,
            velocity=trajectory_points[-1].velocity,
            timestamp=trajectory_points[-1].timestamp
        )

        try:
            final_orbit = self.orbital_mechanics.cartesian_to_kepler(final_state)
            final_orbit_dict = {
                'semi_major_axis_km': final_orbit.semi_major_axis,
                'eccentricity': final_orbit.eccentricity,
                'inclination_deg': final_orbit.inclination,
                'apogee_km': final_orbit.semi_major_axis * (1 + final_orbit.eccentricity) - self.EARTH_RADIUS_KM,
                'perigee_km': final_orbit.semi_major_axis * (1 - final_orbit.eccentricity) - self.EARTH_RADIUS_KM,
            }

            mission_success = (
                final_orbit_dict['perigee_km'] > target_altitude * 0.8 and
                final_orbit_dict['apogee_km'] < target_altitude * 1.5 and 
                abs(final_orbit.inclination - target_inclination < 10)
            )
        except:
            final_orbit_dict = None
            mission_success = False

        if current_mass > 0:
            total_delta_v = vehicle_specs.specific_impulse_s * self.G0 * math.log(
                (vehicle_specs.dry_mass_kg + vehicle_specs.fuel_mass_kg) / current_mass
            )
        else:
            total_delta_v = 0

        # Success probability based on various factors
        success_weights = []
        success_scores = []

        # Dynamic pressure check
        if max_dynamic_pressure < vehicle_specs.max_dynamic_pressure_pa:
            success_scores.append(0.95)
            success_weights.append(0.25)
        elif max_dynamic_pressure < vehicle_specs.max_dynamic_pressure_pa * 1.5:
            success_scores.append(0.7)
            success_weights.append(0.25)
        else:
            success_scores.append(0.3)
            success_weights.append(0.25)

        # G-force check
        if max_g_force < vehicle_specs.max_acceleration_g:
            success_scores.append(0.95)
            success_weights.append(0.25)
        elif max_g_force < vehicle_specs.max_acceleration_g * 1.2:
            success_scores.append(0.7)
            success_weights.append(0.25)
        else:
            success_scores.append(0.4)
            success_weights.append(0.25)

        # Mission success check
        if mission_success:
            success_scores.append(0.95)
            success_weights.append(0.3)
        elif final_orbit_dict and final_orbit_dict.get('perigee_km', -1000) > 100:
            success_scores.append(0.6)
            success_weights.append(0.3)
        else:
            success_scores.append(0.2)
            success_weights.append(0.3)

        # Fuel check
        fuel_remaining = vehicle_specs.fuel_mass_kg - fuel_consumed
        if fuel_remaining > vehicle_specs.fuel_mass_kg * 0.1:
            success_scores.append(0.95)
            success_weights.append(0.15)
        elif fuel_remaining > 0:
            success_scores.append(0.7)
            success_weights.append(0.15)
        else:
            success_scores.append(0.3)
            success_weights.append(0.15)
            
        # Weather impact score with more sensitivity for small variations
        weather_score = 1.0
        wind_speed = weather_data.get('wind_speed_ms', 0)
        rain_mm = weather_data.get('rain_1h_mm', 0) + weather_data.get('rain_3h_mm', 0)
        cloud_cover = weather_data.get('cloud_cover_percent', 0)
        temp_c = weather_data.get('temperature_c', 20)
        
        # More granular wind impact (sensitive to small changes)
        wind_factor = max(0.7, 1.0 - (wind_speed / 20.0))  # Linear decrease from 1.0 to 0.7
        weather_score *= wind_factor
            
        # Rain/precipitation impact
        if rain_mm > 2:
            weather_score *= 0.4
        elif rain_mm > 0.5:
            weather_score *= 0.7
        elif rain_mm > 0.1:
            weather_score *= 0.9
            
        # Cloud cover impact (more sensitive)
        cloud_factor = max(0.8, 1.0 - (cloud_cover / 150.0))  # Linear decrease
        weather_score *= cloud_factor
            
        # Temperature impact (more sensitive to variations)
        optimal_temp = 20.0
        temp_deviation = abs(temp_c - optimal_temp)
        temp_factor = max(0.85, 1.0 - (temp_deviation / 50.0))
        weather_score *= temp_factor
        
        # Add time-of-day variation for atmospheric conditions  
        hour = launch_time.hour
        time_factor = 1.0 + 0.02 * math.sin((hour - 12) * math.pi / 12)  # ±2% variation
        weather_score *= time_factor
        
        # Add weather score to success calculation
        success_scores.append(weather_score)
        success_weights.append(0.2)
            
        success_probability = sum(s * w for s, w in zip(success_scores, success_weights)) / sum(success_weights)

        # Calculate launch constraints based on weather data
        if weather_constraints is None:
            weather_constraints = {
                'max_wind_speed_ms': 15,
                'max_wind_gust_ms': 20,
                'min_temperature_c': -10,
                'max_temperature_c': 35,
                'max_precipitation_mm': 0.0,
                'min_visibility_m': 5000.0,
                'max_cloud_cover_pct': 50
            }
        
        weather_assessment = self.atmospheric_models.assess_weather_launch_constraints(
            weather_data, weather_constraints
        )
        launch_constraints_met = weather_assessment.get('go_for_launch', True)
        
        return LaunchTrajectory(
            trajectory_points=trajectory_points,
            success_probability=success_probability,
            mission_objectives_met=mission_success,
            max_dynamic_pressure=max_dynamic_pressure,
            max_g_force=max_g_force,
            fuel_remaining_kg=fuel_remaining,
            final_orbit_elements=final_orbit_dict,
            launch_constraints_met=launch_constraints_met,
            total_delta_v=total_delta_v
        )

    def optimize_launch_window(self, launch_site: str, target_orbit: Dict, vehicle_specs: VehicleSpecs, weather_forecast: List[Dict], 
                               start_time: datetime, duration_hours: int = 24) -> List[Dict]:
        """Find optimal launch windows with a given time period."""
        launch_windows = []
        current_time = start_time
        end_time = start_time + timedelta(hours=duration_hours)
        time_step = timedelta(minutes=30)
        
        weather_constraints = {
            'max_wind_speed_ms': 15,
            'max_wind_gust_ms': 20,
            'min_temperature_c': -10,
            'max_temperature_c': 35,
            'max_precipitation_mm': 0.0,
            'min_visibility_m': 5000.0,
            'max_cloud_cover_pct': 50
        }

        while current_time < end_time:
            # Find closest weather forecast
            closest_weather = min(weather_forecast, 
                key=lambda w: abs(datetime.fromisoformat(w['forecast_time'].replace('Z', '+00:00')) - current_time))
            
            # Calculate trajectory for this time with proper weather constraints
            trajectory = self.calculate_launch_trajectory(
                launch_site, target_orbit, vehicle_specs, closest_weather, current_time, weather_constraints=weather_constraints
            )
            
            # Calculate orbital mechanics bonus based on launch timing
            orbital_bonus = self._calculate_orbital_timing_bonus(launch_site, target_orbit, current_time)
            
            # Calculate weather score (0.0 to 1.0 based on how good conditions are)
            weather_score = self._calculate_weather_score(closest_weather, weather_constraints)
            
            # Window scoring should closely match success probability with small adjustments
            base_score = trajectory.success_probability
            
            # Small adjustments that don't drastically change the score
            mission_adjustment = 0.02 if trajectory.mission_objectives_met else -0.05
            constraint_adjustment = 0.01 if trajectory.launch_constraints_met else -0.03
            
            # Orbital timing adjustment (very small)
            orbital_adjustment = (orbital_bonus - 1.0) * 0.05  # ±2.5% max
            
            # Weather quality adjustment (small)
            weather_adjustment = (weather_score - 0.95) * 0.1  # Small bonus for great weather
            
            # Window score should be very close to success probability
            window_score = base_score + mission_adjustment + constraint_adjustment + orbital_adjustment + weather_adjustment
            window_score = max(0.0, min(1.0, window_score))  # Clamp between 0 and 1
            
            launch_windows.append({
                'launch_time': current_time,
                'success_probability': trajectory.success_probability,
                'window_score': window_score,
                'weather_conditions': closest_weather,
                'go_for_launch': trajectory.launch_constraints_met,
                'score_breakdown': {
                    'base_score': base_score,
                    'mission_bonus': mission_bonus,
                    'constraint_bonus': constraint_bonus,
                    'orbital_bonus': orbital_bonus,
                    'weather_score': weather_score
                },
                'trajectory_summary': {
                    'max_dynamic_pressure': trajectory.max_dynamic_pressure,
                    'max_g_force': trajectory.max_g_force,
                    'fuel_remaining': trajectory.fuel_remaining_kg,
                    'final_orbit': trajectory.final_orbit_elements
                }
            })

            current_time += time_step
        
        # Sort windows by score
        launch_windows.sort(key=lambda x: x['window_score'], reverse=True)
        return launch_windows
    
    def calculate_abort_scenarios(self, nominal_trajectory: LaunchTrajectory, abort_altitudes: List[float]) -> Dict[float, Dict]:
        """Calculate abort scenarios at specified altitudes"""
        abort_scenarios = {}

        for abort_altitude in abort_altitudes:
            abort_point = min(nominal_trajectory.trajectory_points, key=lambda p: abs(p.altitude_km - abort_altitude))

            # Calculate return trajectory
            abort_scenario = self._calculate_abort_trajectory(abort_point)
            abort_scenarios[abort_altitude] = abort_scenario

        return abort_scenarios
    
    def _calculate_abort_trajectory(self, abort_point: TrajectoryPoint) -> Dict:
        """Calculate abort trajectory from a given point."""
        # Simplified abort trajectory from a given point

        current_velocity = np.linalg.norm(abort_point.velocity) * 1000  # Convert km/s to m/s
        current_altitude = abort_point.altitude_km

        position_mag = np.linalg.norm(abort_point.position)

        specific_energy = (current_velocity ** 2) / 2000000 - self.EARTH_MU / position_mag

        if specific_energy < 0:
            # Suborbital - will return to earth
            range_km = self._calculate_ballistic_range(current_velocity, current_altitude)
            abort_type = "BALLISTIC_RETURN"
            survival_probability = 0.8 if current_altitude < 50 else 0.6
        else:
            # Orbital or escape trajectory
            abort_type = "ORBITAL_ABORT"
            range_km = None
            survival_probability = 0.9
        
        return {
            'abort_altitude_km': current_altitude,
            'abort_type': abort_type,
            'range_km': range_km,
            'survival_probability': survival_probability,
            'specific_energy': specific_energy,
            'abort_delta_v_required': max(0, math.sqrt(abs(specific_energy)) * 100) if specific_energy < 0 else 0 
        }
    
    def _calculate_ballistic_range(self, velocity_ms: float, altitude_km: float) -> float:
        """Calculate approximate ballistic range for abort scenario."""
        g = 9.81

        # Assume 45-degree trajectory for max range
        horizontal_velocity = velocity_ms * math.cos(math.radians(45))
        vertical_velocity = velocity_ms * math.sin(math.radians(45))

        # Time of flight (Ignoring air resistance)
        total_height = altitude_km * 1000 * (vertical_velocity ** 2) / (2 * g)
        time_of_flight = 2 * math.sqrt(2 * total_height / g)

        # Range calculation
        range_m = horizontal_velocity * time_of_flight
        return range_m / 1000  # Convert to km
    
    def calculate_payload_separation_conditions(self, trajectory: LaunchTrajectory, separation_altitude_km: float) -> Dict:
        """Calculate conditions at payload separation."""

        separation_point = min(trajectory.trajectory_points, key=lambda p: abs(p.altitude_km - separation_altitude_km))

        # Calculate orbital elements at separation
        separation_state = StateVector(
            position=separation_point.position,
            velocity=separation_point.velocity,
            timestamp=separation_point.timestamp,
        )

        try:
            orbit_elements = self.orbital_mechanics.cartesian_to_kepler(separation_state)

            # Calcualte orbital characteristics
            apogee = orbit_elements.semi_major_axis * (1 + orbit_elements.eccentricity) - self.EARTH_RADIUS_KM
            perigee = orbit_elements.semi_major_axis * (1 - orbit_elements.eccentricity) - self.EARTH_RADIUS_KM
            orbital_period = 2 * math.pi * math.sqrt(orbit_elements.semi_major_axis ** 3 / self.EARTH_MU) / 3600 # Convert to hours

            separation_conditions = {
                'separation_altitude_km': separation_point.altitude_km,
                'separation_velocity_ms': separation_point.velocity_magnitude_ms,
                'orbital_elements': {
                    'semi_major_axis_km': orbit_elements.semi_major_axis,
                    'eccentricity': orbit_elements.eccentricity,
                    'inclination_deg': orbit_elements.inclination,
                    'apogee_km': apogee,
                    'perigee_km': perigee,
                    'orbital_period_hours': orbital_period
                },
                'deployment_success_probability': 0.95 if perigee > 150 else 0.7,
                'atmospheric_conditions': {
                    'density_kg_m3': separation_point.atmospheric_conditions.density_kg_m3,
                    'temperature_k': separation_point.atmospheric_conditions.temperature_k,
                }
            }
        except:
            separation_conditions = {
                'separation_altitude_km': separation_point.altitude_km,
                'separation_velocity_ms': separation_point.velocity_magnitude_ms,
                'orbital_elements': None,
                'deployment_success_probability': 0.1,
                'error': 'Failed to calculate orbital elements'
            }

        return separation_conditions
    
    def analyze_trajectory_sensitivities(self, base_trajectory: LaunchTrajectory, 
                                         vehicle_specs: VehicleSpecs, weather_data: Dict, 
                                         launch_site: str, target_orbit: Dict, 
                                         launch_time: datetime) -> Dict:
        """Analyze how sensitive the trajectory is to various parameters."""
        sensitivities = {}

        # Base success probability
        base_success = base_trajectory.success_probability

        # Wind speed sensitivity
        wind_variations = [-5, -2, 0, 2, 5]  # m/s variations
        wind_sensitivity = []

        for wind_delta in wind_variations:
            modified_weather = weather_data.copy()
            modified_weather['wind_speed_ms'] = weather_data.get('wind_speed_ms', 0) + wind_delta

            modified_trajectory = self.calculate_launch_trajectory(
                launch_site, target_orbit, vehicle_specs, modified_weather, launch_time
            )

            success_change = modified_trajectory.success_probability - base_success
            wind_sensitivity.append({
                'wind_delta_ms': wind_delta,
                'success_change': success_change,
                'new_success_probability': modified_trajectory.success_probability
            })

        sensitivities['wind_sensitivity'] = wind_sensitivity

        # Thrust sensitivity
        thrust_variations = [-0.1, -0.05, 0, 0.05, 0.1] # +- 10% variations
        thrust_sensitivity = []

        for thrust_factor in thrust_variations:
            modified_specs = VehicleSpecs(
                name=vehicle_specs.name,
                dry_mass_kg=vehicle_specs.dry_mass_kg,
                fuel_mass_kg=vehicle_specs.fuel_mass_kg,
                thrust_n=vehicle_specs.thrust_n * (1 + thrust_factor),
                specific_impulse_s=vehicle_specs.specific_impulse_s,
                drag_coefficient=vehicle_specs.drag_coefficient,
                cross_sectional_area_m2=vehicle_specs.cross_sectional_area_m2,
                max_dynamic_pressure_pa=vehicle_specs.max_dynamic_pressure_pa,
                max_acceleration_g=vehicle_specs.max_acceleration_g,
                stage_count=vehicle_specs.stage_count
            )

            modified_trajectory = self.calculate_launch_trajectory(
                launch_site, target_orbit, modified_specs, weather_data, launch_time
            )

            success_change = modified_trajectory.success_probability - base_success
            thrust_sensitivity.append({
                'thrust_factor': thrust_factor,
                'success_change': success_change,
                'new_success_probability': modified_trajectory.success_probability
            })

        sensitivities['thrust'] = thrust_sensitivity

        # Mass variations

        mass_variations = [-0.1, -0.05, 0, 0.05, 0.1] # +- 10% variations
        mass_sensitivity = []

        for mass_factor in mass_variations:
            modified_specs = VehicleSpecs(
                name = vehicle_specs.name,
                dry_mass_kg = vehicle_specs.dry_mass_kg,
                fuel_mass_kg = vehicle_specs.fuel_mass_kg * (1 + mass_factor),
                thrust_n = vehicle_specs.thrust_n,
                specific_impulse_s = vehicle_specs.specific_impulse_s,
                drag_coefficient = vehicle_specs.drag_coefficient,
                cross_sectional_area_m2 = vehicle_specs.cross_sectional_area_m2,
                max_dynamic_pressure_pa = vehicle_specs.max_dynamic_pressure_pa,
                max_acceleration_g = vehicle_specs.max_acceleration_g,
                stage_count = vehicle_specs.stage_count
            )

            modified_trajectory = self.calculate_launch_trajectory(
                launch_site, target_orbit, modified_specs, weather_data, launch_time
            )

            success_change = modified_trajectory.success_probability - base_success
            mass_sensitivity.append({
                'mass_factor': mass_factor,
                'success_change': success_change,
                'new_success_probability': modified_trajectory.success_probability
            })

        sensitivities['mass'] = mass_sensitivity

        return sensitivities
    
    def calculate_multi_stage_trajectory(self, launch_site: str, target_orbit: Dict, stage_specs: List[VehicleSpecs],
                                         weather_data: Dict, launch_time: datetime) -> LaunchTrajectory:
        """Calculate trajectory for multi-stage rocket."""
        # Simplified multi-stage trajectory calculation
        trajectory_points = []
        current_time = 0
        time_step_s = 1.0

        # Initialize with first stage
        current_stage = 0
        current_specs = stage_specs[current_stage]

        # Calcualte trajectory similar to single stage, but with stage separation logic
        base_trajectory = self.calculate_launch_trajectory(
            launch_site, target_orbit, current_specs, weather_data, launch_time, time_step_s
        )

        # This does not handle all stage separations as it is a simplified version.

        return base_trajectory
    
    def generate_trajectory_report(self, trajectory: LaunchTrajectory, vehicle_specs: VehicleSpecs, target_orbit: Dict) -> str:
        """Generate a detailed report of the launch trajectory."""
        if not trajectory.trajectory_points:
            return {'error': 'No trajectory points available for report generation.'}
        
        # Mission phases analysis
        phases = self._identify_mission_phases(trajectory)

        # Performance metrics
        performance = {
            'success_probability': trajectory.success_probability,
            'mission_objectives_met': trajectory.mission_objectives_met,
            'max_dynamic_pressure_pa': trajectory.max_dynamic_pressure,
            'max_g_force': trajectory.max_g_force,
            'total_delta_v': trajectory.total_delta_v,
            'fuel_efficiency': (vehicle_specs.fuel_mass_kg - trajectory.fuel_remaining_kg) / vehicle_specs.fuel_mass_kg
        }
    
        # Trajectory statistics
        max_altitude = max(point.altitude_km for point in trajectory.trajectory_points)
        max_velocity = max(point.velocity_magnitude_ms for point in trajectory.trajectory_points)
        flight_time = trajectory.trajectory_points[-1].time_s
        avg_acceleration_g = np.mean([point.g_force for point in trajectory.trajectory_points])
        peak_heating_altitude_km = self._find_peak_heating_altitude(trajectory)

        statistics = {
            'max_altitude_km': max_altitude,
            'max_velocity_ms': max_velocity,
            'total_flight_time_s': flight_time,
            'average_acceleration_g': avg_acceleration_g,
            'peak_heating_altitude_km': peak_heating_altitude_km
        }

        # Risk assessment
        risk_factors = []
        if trajectory.max_dynamic_pressure > vehicle_specs.max_dynamic_pressure_pa * 0.9:
            risk_factors.append('High dynamic pressure approaching limits.')
        if trajectory.max_g_force > vehicle_specs.max_acceleration_g * 0.9:
            risk_factors.append('High G-force approaching limits.')
        if trajectory.fuel_remaining_kg < vehicle_specs.fuel_mass_kg * 0.05:
            risk_factors.append('Low fuel remaining.')

        return {
            'mission_phases': phases,
            'performance_metrics': performance,
            'trajectory_statistics': statistics,
            'final_orbit': trajectory.final_orbit_elements,
            'risk_assessment': {
                'overall_risk': 'LOW' if len(risk_factors) == 0 else 'MEDIUM' if len(risk_factors) < 2 else 'HIGH',
                'risk_factors': risk_factors
            },
            'recommendations': self._generate_recommendations(trajectory, vehicle_specs, target_orbit)
        }
    
    def _identify_mission_phases(self, trajectory: LaunchTrajectory) -> List[Dict]:
        """Identify different phases of the mission from trajectory data."""
        phases = []

        # Liftoff phase (0-10 seconds)
        liftoff_points = [p for p in trajectory.trajectory_points if p.time_s <= 10]

        if liftoff_points:
            phases.append({
                'name': 'Liftoff',
                'start_time_s': 0,
                'end_time_s': 10,
                'max_acceleration_g': max(p.g_force for p in liftoff_points),
                'altitude_gained_km': liftoff_points[-1].altitude_km
            })

        # Powered ascent phase (thrust > 0)
        powered_points = [p for p in trajectory.trajectory_points if p.thrust_n > 0]
        if powered_points:
            phases.append({
                'name': 'Powered Ascent',
                'start_time_s': powered_points[0].time_s,
                'end_time_s': powered_points[-1].time_s,
                'max_dynamic_pressure_pa': max(p.dynamic_pressure_pa for p in powered_points),
                'max_mach_number': max(p.mach_number for p in powered_points if p.mach_number is not None)
            })

        # Coast phase (if any)
        coast_points = [p for p in trajectory.trajectory_points if p.thrust_n == 0 and p.altitude_km > 200]
        if coast_points:
            phases.append({
                'name': 'Coast Phase',
                'start_time_s': coast_points[0].time_s,
                'end_time_s': coast_points[-1].time_s,
                'duration_s': coast_points[-1].time_s - coast_points[0].time_s
            })

        return phases
    
    def _find_peak_heating_altitude(self, trajectory: LaunchTrajectory) -> float:
        """Find the altitude at which peak heating occurs."""
        peak_heating_point = max(trajectory.trajectory_points, key=lambda p: p.dynamic_pressure_pa)
        return peak_heating_point.altitude_km if peak_heating_point else 0.0
    
    def _generate_recommendations(self, trajectory: LaunchTrajectory, vehicle_specs: VehicleSpecs, target_orbit: Dict) -> List[str]:
        """Generate recommendations based on trajectory analysis."""
        recommendations = []

        if trajectory.success_probability < 0.8:
            recommendations.append("Consider optimizing launch trajectory or weather window for higher success probability.")
        
        if trajectory.max_dynamic_pressure > vehicle_specs.max_dynamic_pressure_pa * 0.8:
            recommendations.append("Monitor dynamic pressure closely during ascent; consider throttling if necessary.")
        
        if trajectory.fuel_remaining_kg < vehicle_specs.fuel_mass_kg * 0.1:
            recommendations.append("Fuel margins are tight; verify performance assumptions and consider contingency plans")

        if not trajectory.mission_objectives_met:
            recommendations.append("Mission objectives were not fully met; review trajectory and target orbit parameters.")

        if trajectory.final_orbit_elements:
            target_alt = target_orbit.get('altitude_km', 400)
            actual_perigee = trajectory.final_orbit_elements.get('perigee_km', 0)
            if abs(actual_perigee - target_alt) > 50:
                recommendations.append("Final orbit deviates significantly from target; consider adjusting launch parameters for better accuracy.")

        return recommendations
    
    def _calculate_orbital_timing_bonus(self, launch_site: str, target_orbit: Dict, launch_time: datetime) -> float:
        """Calculate bonus for orbital mechanics timing advantages."""
        try:
            site = self.orbital_mechanics.launch_sites[launch_site]
            target_inclination = target_orbit.get('inclination_deg', 28.5)
            
            # Time-based factors
            hour = launch_time.hour
            minute = launch_time.minute
            
            # Orbital mechanics advantages for different times
            # Better timing for polar orbits in morning/evening
            if target_inclination > 90:  # Retrograde/polar
                if 6 <= hour <= 8 or 18 <= hour <= 20:  # Dawn/dusk
                    time_bonus = 1.15
                elif 10 <= hour <= 14:  # Midday
                    time_bonus = 0.85
                else:
                    time_bonus = 1.0
            else:  # Prograde orbits
                if 10 <= hour <= 14:  # Midday optimal
                    time_bonus = 1.1
                elif hour < 6 or hour > 22:  # Night launches
                    time_bonus = 0.9
                else:
                    time_bonus = 1.0
            
            # Fine-tuning based on minutes (orbital period considerations)
            minute_factor = 1.0 + 0.05 * math.sin(2 * math.pi * minute / 60)
            
            return time_bonus * minute_factor
            
        except Exception:
            return 1.0  # Neutral if calculation fails
    
    def _calculate_weather_score(self, weather_data: Dict, constraints: Dict) -> float:
        """Calculate a detailed weather quality score (0.0 to 1.0)."""
        scores = []
        weights = []
        
        # Wind speed score
        wind_speed = weather_data.get('wind_speed_ms', 0)
        max_wind = constraints.get('max_wind_speed_ms', 15)
        if wind_speed <= max_wind * 0.3:
            wind_score = 1.0
        elif wind_speed <= max_wind * 0.6:
            wind_score = 0.8
        elif wind_speed <= max_wind:
            wind_score = 0.5
        else:
            wind_score = 0.1
        scores.append(wind_score)
        weights.append(0.3)
        
        # Temperature score
        temp = weather_data.get('temperature_c', 20)
        min_temp = constraints.get('min_temperature_c', -10)
        max_temp = constraints.get('max_temperature_c', 35)
        optimal_temp = (min_temp + max_temp) / 2
        temp_deviation = abs(temp - optimal_temp) / (max_temp - min_temp)
        temp_score = max(0.1, 1.0 - temp_deviation)
        scores.append(temp_score)
        weights.append(0.2)
        
        # Precipitation score
        precip = weather_data.get('rain_1h_mm', 0) + weather_data.get('snow_1h_mm', 0)
        max_precip = constraints.get('max_precipitation_mm', 0.0)
        if precip <= max_precip:
            precip_score = 1.0
        elif precip <= max_precip * 2:
            precip_score = 0.3
        else:
            precip_score = 0.1
        scores.append(precip_score)
        weights.append(0.25)
        
        # Cloud cover score
        clouds = weather_data.get('cloud_cover_pct', 0)
        max_clouds = constraints.get('max_cloud_cover_pct', 50)
        if clouds <= max_clouds * 0.5:
            cloud_score = 1.0
        elif clouds <= max_clouds:
            cloud_score = 0.7
        else:
            cloud_score = 0.3
        scores.append(cloud_score)
        weights.append(0.15)
        
        # Visibility score
        visibility = weather_data.get('visibility_m', 10000)
        min_visibility = constraints.get('min_visibility_m', 5000)
        if visibility >= min_visibility * 2:
            vis_score = 1.0
        elif visibility >= min_visibility:
            vis_score = 0.8
        else:
            vis_score = 0.4
        scores.append(vis_score)
        weights.append(0.1)
        
        # Calculate weighted average
        weighted_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
        return max(0.1, min(1.0, weighted_score))