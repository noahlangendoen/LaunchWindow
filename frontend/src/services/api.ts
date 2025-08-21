import axios from 'axios';
import { LaunchTrajectory, LaunchWindow, PredictionResult, WeatherData } from '../types';

const api = axios.create({
  baseURL: 'https://launchwindow-production.up.railway.app',
  timeout: 30000,
});

export const apiService = {
  // Launch prediction endpoints
  async runPrediction(siteCode: string): Promise<PredictionResult> {
    const response = await api.post('/prediction', { site_code: siteCode });
    return response.data;
  },

  // Trajectory calculation endpoints
  async calculateTrajectory(siteCode: string, targetOrbit: any): Promise<LaunchTrajectory> {
    const response = await api.post('/trajectory', {
      site_code: siteCode,
      target_orbit: targetOrbit
    });
    return response.data;
  },

  // Weather data endpoints
  async getCurrentWeather(siteCode: string): Promise<WeatherData> {
    const response = await api.get(`/weather/current/${siteCode}`);
    return response.data;
  },

  async getWeatherForecast(siteCode: string, days: number = 5): Promise<WeatherData[]> {
    const response = await api.get(`/weather/forecast/${siteCode}`, {
      params: { days }
    });
    return response.data;
  },

  // Launch window optimization
  async findOptimalWindows(siteCode: string, durationHours: number = 48): Promise<LaunchWindow[]> {
    const response = await api.post('/windows', {
      site_code: siteCode,
      duration_hours: durationHours
    });
    return response.data;
  },

  // System status
  async getSystemStatus() {
    const response = await api.get('/status');
    return response.data;
  }
};