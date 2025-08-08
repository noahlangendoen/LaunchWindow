import math
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Tuple, List, Optional

@dataclass
class LaunchSite:
    """Represent a launch site with coordinates, name, etc."""
    name: str
    latitude: float
    longitude: float
    altitude: float
    timezone: str

@dataclass
class OrbitalElements:
    """Represents a set of orbital elements."""
    semi_major_axis: float
    eccentricity: float
    inclination: float
    longitude_of_ascending_node: float
    argument_of_periapsis: float
    true_anomaly: float
    epoch: datetime

@dataclass
class StateVector:
    """Represents position and velocity vectors."""
    position: np.ndarray
    velocity: np.ndarray
    timestamp: datetime

class OrbitalMechanics:
    """Class for performing orbital mechanics calculations for launch predictions."""

    # Constants
    EARTH_MU = 398600.4418  # Earth's gravitational parameter in km^3/s^2
    EARTH_RADIUS = 6371.0  # Earth's radius in km
    EARTH_J2 = 1.08262668e-3  # Earth's second zonal harmonic coefficient
    EARTH_ROTATION_RATE = 7.2921159e-5  # Earth's rotation rate in rad/s

    def __init__(self):
        self.launch_sites = {
            'KSC': LaunchSite('Kennedy Space Center', 28.5721, -80.6480, 0, 'America/New_York'),
            'VSFB': LaunchSite('Vandenberg Space Force Base', 34.7420, -120.5724, 0, 'America/Los_Angeles'),
            'CCAFS': LaunchSite('Cape Canaveral Space Force Station', 28.3922, -80.6077, 0, 'America/New_York')
        }

    def kepler_to_cartesian(self, elements: OrbitalElements) -> StateVector:
        """Convert Keplerian orbital elements to Cartesian state vector."""
        a = elements.semi_major_axis
        e = elements.eccentricity
        i = math.radians(elements.inclination)
        omega = math.radians(elements.longitude_of_ascending_node)
        w = math.radians(elements.argument_of_periapsis)
        nu = math.radians(elements.true_anomaly)

        # Calculating distance
        r = a * (1 - e ** 2) / (1 + e * math.cos(nu))

        # Position in the orbital plane
        x_orb = r * math.cos(nu)
        y_orb = r * math.sin(nu)
        z_orb = 0.0

        # Velocity in orbital plane
        h = math.sqrt(self.EARTH_MU * a * (1 - e ** 2))
        vx_orb = -self.EARTH_MU / h * math.sin(nu)
        vy_orb = self.EARTH_MU / h * (e + math.cos(nu))
        vz_orb = 0.0

        # Rotation matrices
        cos_omega, sin_omega = math.cos(omega), math.sin(omega)
        cos_i, sin_i = math.cos(i), math.sin(i)
        cos_w, sin_w = math.cos(w), math.sin(w)

        # Rotation matrix from orbital plane to inertial frame
        R11 = cos_omega * cos_w - sin_omega * sin_w * cos_i
        R12 = -cos_omega * sin_w - sin_omega * cos_w * cos_i
        R13 = sin_omega * sin_i
        R21 = sin_omega * cos_w + cos_omega * sin_w * cos_i
        R22 = -sin_omega * sin_w + cos_omega * cos_w * cos_i
        R23 = -cos_omega * sin_i
        R31 = sin_i * sin_w
        R32 = sin_i * cos_w
        R33 = cos_i

        # Transform to inertial frame
        x = R11 * x_orb + R12 * y_orb + R13 * z_orb
        y = R21 * x_orb + R22 * y_orb + R23 * z_orb
        z = R31 * x_orb + R32 * y_orb + R33 * z_orb

        vx = R11 * vx_orb + R12 * vy_orb + R13 * vz_orb
        vy = R21 * vx_orb + R22 * vy_orb + R23 * vz_orb
        vz = R31 * vx_orb + R32 * vy_orb + R33 * vz_orb

        return StateVector(
            position=np.array([x, y, z]),
            velocity=np.array([vx, vy, vz]),
            timestamp=elements.epoch
        )
    
    def cartesian_to_kepler(self, state: StateVector) -> OrbitalElements:
        """Convert Cartesian state vector to Keplerian orbital elements."""
        r_vec = state.position
        v_vec = state.velocity

        r = np.linalg.norm(r_vec)
        v = np.linalg.norm(v_vec)

        # Angular momentum vector
        h_vec = np.cross(r_vec, v_vec)
        h = np.linalg.norm(h_vec)

        # Eccentricity vector
        e_vec = (np.cross(v_vec, h_vec) / self.EARTH_MU) - (r_vec / r)
        e = np.linalg.norm(e_vec)

        # Semi-major axis
        energy = v**2 / 2 - self.EARTH_MU / r
        a = -self.EARTH_MU / (2 * energy)

        # Inclination
        i = math.acos(h_vec[2] / h)

        # Node vector
        n_vec = np.cross([0, 0, 1], h_vec)
        n = np.linalg.norm(n_vec)

        # Longitude of ascending node
        if n != 0:
            omega = math.acos(n_vec[0] / n)
            if n_vec[1] < 0:
                omega = 2 * math.pi - omega
        else:
            omega = 0.0

        # Argument of periapsis
        if n != 0 and e != 0:
            w = math.acos(np.dot(n_vec, e_vec) / (n * e))
            if e_vec[2] < 0:
                w = 2 * math.pi - w
        else:
            w = 0.0

        # True anomaly
        if e != 0:
            nu = math.acos(np.dot(e_vec, r_vec) / (e * r))
            if np.dot(r_vec, v_vec) < 0:
                nu = 2 * math.pi - nu
        else:
            nu = 0.0
        
        return OrbitalElements(
            semi_major_axis=a,
            eccentricity=e,
            inclination=math.degrees(i),
            longitude_of_ascending_node=math.degrees(omega),
            argument_of_periapsis=math.degrees(w),
            true_anomaly=math.degrees(nu),
            epoch=state.timestamp
        )
    
    def propagate_orbit(self, state: StateVector, time_delta: timedelta) -> StateVector:
        """Propagate an orbit forward in time usign simplified perturbations."""
        elements = self.cartesian_to_kepler(state)

        # Mean motion
        n = math.sqrt(self.EARTH_MU / elements.semi_major_axis ** 3)

        # Time in seconds
        dt = time_delta.total_seconds()

        # Mean anomaly progression
        M0 = self.true_to_mean_anomaly(math.radians(elements.true_anomaly), elements.eccentricity)
        M = M0 + n * dt

        # J2 perturbations (simplified)
        cos_i = math.cos(math.radians(elements.inclination))

        # Secular variations due to J2
        domega_dt = -1.5 * n * self.EARTH_J2 * (self.EARTH_RADIUS / elements.semi_major_axis) ** 2 * cos_i 
        dw_dt = 0.75 * n * self.EARTH_J2 * (self.EARTH_RADIUS / elements.semi_major_axis) ** 2 * (5 * cos_i ** 2 - 1)

        # Update arguments
        new_elements = OrbitalElements(
            semi_major_axis=elements.semi_major_axis,
            eccentricity=elements.eccentricity,
            inclination=elements.inclination,
            longitude_of_ascending_node=elements.longitude_of_ascending_node + math.degrees(domega_dt * dt),
            argument_of_periapsis=elements.argument_of_periapsis + math.degrees(dw_dt * dt),
            true_anomaly=math.degrees(self.mean_to_true_anomaly(M, elements.eccentricity)),
            epoch=state.timestamp + time_delta
        )

        return self.kepler_to_cartesian(new_elements)
    
    def true_to_mean_anomaly(self, nu: float, e: float) -> float:
        """Convert true anomaly to mean anomaly."""
        E = 2 * math.atan(math.sqrt((1 - e) / (1 + e)) * math.tan(nu / 2))
        M = E - e * math.sin(E)
        return M
    
    def mean_to_true_anomaly(self, M: float, e: float, tolerance: float = 1e-8) -> float:
        """Convert mean anomaly to true anomaly using Newton's method."""
        E = M # Initial guess

        for _ in range(100):  # Limit iterations to prevent infinite loop
            f = E - e * math.sin(E) - M
            f_prime = 1 - e * math.cos(E)
            E_new = E - f / f_prime
            if abs(E_new - E) < tolerance:
                break
            E = E_new
        
        # Convert Eccentric Anomaly to True Anomaly
        nu = 2 * math.atan(math.sqrt((1 + e) / (1 - e)) * math.tan(E / 2))
        return nu
    
    def calculate_ground_track(self, elements: OrbitalElements, duration_hours: float = 24) -> List[Tuple[float, float]]:
        """Calculate the ground track of the orbit over a specified duration."""
        points = []
        time_step = timedelta(minutes = 5)
        current_time = elements.epoch
        end_time = elements.epoch + timedelta(hours=duration_hours)

        state = self.kepler_to_cartesian(elements)

        while current_time <= end_time:
            # Propagate the orbit
            dt = current_time - elements.epoch
            current_state = self.propagate_orbit(state, dt)

            # Convert latitude and longitude
            lat, lon = self.cartesian_to_geodetic(current_state.position, current_time)
            points.append((lat, lon))

            current_time += time_step

        return points
    
    def cartesian_to_geodetic(self, position: np.ndarray, timestamp: datetime) -> Tuple[float, float]:
        """Convert Cartesian coordinates to geodetic coordinates (latitude, longitude)."""
        x, y, z = position
        
        gmst = self.calculate_gmst(timestamp)

        # Convert to rotating Earth frame
        cos_gmst, sin_gmst = math.cos(gmst), math.sin(gmst)
        x_earth = cos_gmst * x + sin_gmst * y
        y_earth = -sin_gmst * x + cos_gmst * y
        z_earth = z

        # Calculate latitude and longitude
        r_earth = math.sqrt(x_earth**2 + y_earth**2)
        lat = math.degrees(math.atan2(z_earth, r_earth))
        lon = math.degrees(math.atan2(y_earth, x_earth))

        # Normalize longitude to [-180, 180]
        lon = ((lon + 180) % 360) - 180

        return lat, lon
        
    def calculate_gmst(self, timestamp: datetime) -> float:
        """Calculate the Greenwich Mean Sidereal Time (GMST) for a given timestamp."""
        # Julian date calculation
        jd = timestamp.timestamp() / 86400.0 + 2440587.5

        # GMST
        t = (jd - 2451545.0) / 36525.0
        gmst0 = 24110.54841 + 8640184.812866 * t + 0.093104 * t**2 - 6.2e-6 * t**3

        # Convert to radians and add time of day
        seconds_in_day = timestamp.hour * 3600 + timestamp.minute * 60 + timestamp.second
        gmst = (gmst0 + seconds_in_day * 1.00273790935) * (math.pi / 43200.0)

        return gmst % (2 * math.pi)  # Normalize to [0, 2π]
    
    def calculate_launch_azimuth(self, site_name: str, target_inclination: float, target_altitude: float = 400):
        """Calculate the optimal launch azimuth for a given site and target inclination."""
        if site_name not in self.launch_sites:
            raise ValueError(f"Launch site '{site_name}' not recognized.")

        site = self.launch_sites[site_name]
        lat = math.radians(site.latitude)
        inc = math.radians(target_inclination)

        # Velocity required for circular orbit
        v_orbit = math.sqrt(self.EARTH_MU / (self.EARTH_RADIUS + target_altitude))

        # Earth's surface velocity at the launch site
        v_earth = self.EARTH_ROTATION_RATE * (self.EARTH_RADIUS + site.altitude / 1000) * math.cos(lat)

        # Required velocity components
        v_north = v_orbit * math.cos(inc)
        v_east = v_orbit * math.sin(inc) - v_earth

        # Calculate azimuth
        azimuth = math.degrees(math.atan2(v_east, v_north))

        return azimuth
    
    def predict_orbital_debris_encounter(self, launch_trajectory: List[StateVector], debris_elements: List[OrbitalElements], threshold_km: float = 10):
        """Predict potential encounters with orbital debris."""
        encounters = []

        for debris in debris_elements:
            for i, state in enumerate(launch_trajectory):
                # Propagate debris state
                dt = state.timestamp - debris.epoch
                debris_state = self.propagate_orbit(self.kepler_to_cartesian(debris), dt)

                # Calculate closest approach
                distance = np.linalg.norm(state.position - debris_state.position)

                if distance < threshold_km:
                    encounters.append({
                        'time': state.timestamp,
                        'distance_km': distance,
                        'debris_id': getattr(debris, 'id', 'unknown'),
                        'trajectory_point': i,
                        'risk_level': 'HIGH' if distance < 2 else 'MEDIUM' if distance < 5 else 'LOW'
                    })

        return sorted(encounters, key=lambda x: x['distance_km'])
    
    def calculate_hohmann_transfer(self, r1: float, r2: float) -> dict:
        """Calculate Hohmann transfer orbit parameters."""
        # Semi-major axis of transfer orbit
        a_transfer = (r1 + r2) / 2

        # Velocities
        v1_circular = math.sqrt(self.EARTH_MU / r1)
        v2_circular = math.sqrt(self.EARTH_MU / r2)

        # Transfer orbit velocities
        v1_transfer = math.sqrt(self.EARTH_MU * (2 / r1 - 1 / a_transfer))
        v2_transfer = math.sqrt(self.EARTH_MU * (2 / r2 - 1 / a_transfer))

        # Delta-V requriements
        dv1 = v1_transfer
        dv2 = v2_transfer
        total_dv = abs(dv1) + abs(dv2)

        # Transfer time
        transfer_time = math.pi * math.sqrt(a_transfer ** 3 / self.EARTH_MU)

        return {
            'transfer_dv1': dv1,
            'transfer_dv2': dv2,
            'total_dv': total_dv,
            'transfer_time_seconds': transfer_time,
            'transfer_time_hours': transfer_time / 3600,
            'semi_major_axis': a_transfer
        }
    
    