export interface LaunchSite {
  code: string;
  name: string;
  latitude: number;
  longitude: number;
  timezone: string;
}

export interface WeatherData {
  temperature_c: number;
  wind_speed_ms: number;
  humidity_percent: number;
  pressure_hpa: number;
  cloud_cover_percent: number;
  visibility_m: number;
}

export interface TrajectoryPoint {
  time_s: number;
  position: [number, number, number];
  velocity: [number, number, number];
  altitude_km: number;
  velocity_magnitude_ms: number;
  acceleration: [number, number, number];
  mass_kg: number;
  thrust_n: number;
  drag_n: number;
  dynamic_pressure_pa: number;
  mach_number: number;
  g_force: number;
  timestamp: string;
}

export interface LaunchTrajectory {
  trajectory_points: TrajectoryPoint[];
  success_probability: number;
  mission_objectives_met: boolean;
  max_dynamic_pressure: number;
  max_g_force: number;
  fuel_remaining_kg: number;
  final_orbit_elements: any;
  total_delta_v: number;
}

export interface LaunchWindow {
  launch_time: string;
  window_score: number;
  success_probability: number;
  weather_conditions: WeatherData;
  go_for_launch: boolean;
}

export interface PredictionResult {
  success_probability: number;
  risk_assessment: string;
  recommendation: string;
  weather_status: string;
  site_name: string;
}