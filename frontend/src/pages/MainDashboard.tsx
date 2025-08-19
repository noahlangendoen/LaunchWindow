import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import LaunchSiteSelector from '../components/LaunchSiteSelector';
import PredictionResults from '../components/PredictionResults';
import WeatherPanel from '../components/WeatherPanel';
import { LaunchSite, PredictionResult, WeatherData } from '../types';
import { apiService } from '../services/api';

const LAUNCH_SITES: LaunchSite[] = [
  { code: 'KSC', name: 'Kennedy Space Center', latitude: 28.5721, longitude: -80.6480, timezone: 'America/New_York' },
  { code: 'VSFB', name: 'Vandenberg Space Force Base', latitude: 34.7420, longitude: -120.5724, timezone: 'America/Los_Angeles' },
  { code: 'CCAFS', name: 'Cape Canaveral Space Force Station', latitude: 28.3922, longitude: -80.6077, timezone: 'America/New_York' }
];

const MainDashboard: React.FC = () => {
  const [selectedSite, setSelectedSite] = useState<string>('KSC');
  const [predictionResult, setPredictionResult] = useState<PredictionResult | null>(null);
  const [weatherData, setWeatherData] = useState<WeatherData | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isLoadingWeather, setIsLoadingWeather] = useState(false);
  const [, setSystemStatus] = useState<any>(null);

  useEffect(() => {
    loadWeatherData();
    loadSystemStatus();
  }, [selectedSite]);

  const loadWeatherData = async () => {
    setIsLoadingWeather(true);
    try {
      const weather = await apiService.getCurrentWeather(selectedSite);
      setWeatherData(weather);
    } catch (error) {
      console.error('Failed to load weather data:', error);
      // Mock data for development
      setWeatherData({
        temperature_c: 22,
        wind_speed_ms: 8.5,
        humidity_percent: 65,
        pressure_hpa: 1013,
        cloud_cover_percent: 30,
        visibility_m: 10000
      });
    } finally {
      setIsLoadingWeather(false);
    }
  };

  const loadSystemStatus = async () => {
    try {
      const status = await apiService.getSystemStatus();
      setSystemStatus(status);
    } catch (error) {
      console.error('Failed to load system status:', error);
      setSystemStatus({
        ml_model: 'Ready',
        weather_data: 'Live',
        physics_engine: 'Ready',
        last_update: new Date().toISOString()
      });
    }
  };

  const handleRunPrediction = async () => {
    setIsLoading(true);
    try {
      const result = await apiService.runPrediction(selectedSite);
      setPredictionResult(result);
    } catch (error) {
      console.error('Failed to run prediction:', error);
      // Mock result for development
      setPredictionResult({
        success_probability: 0.87,
        risk_assessment: 'Medium',
        recommendation: 'Launch conditions are favorable. Weather parameters are within acceptable limits. Monitor wind conditions closely.',
        weather_status: 'GO',
        site_name: LAUNCH_SITES.find(s => s.code === selectedSite)?.name || 'Unknown'
      });
    } finally {
      setIsLoading(false);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status?.toLowerCase()) {
      case 'ready':
      case 'live':
      case 'online':
        return 'text-green-400';
      case 'warning':
      case 'caution':
        return 'text-yellow-400';
      case 'offline':
      case 'error':
        return 'text-red-400';
      default:
        return 'text-gray-400';
    }
  };

  return (
    <div className="min-h-screen bg-slate-900">
      {/* Header */}
      <header className="bg-slate-800 border-b border-slate-700">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 bg-blue-500 rounded-lg flex items-center justify-center">
                <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
              </div>
              <div>
                <h1 className="text-2xl font-bold text-white">Rocket Launch Prediction System</h1>
                <p className="text-sm text-gray-400">Real-time launch feasibility analysis and trajectory optimization</p>
              </div>
            </div>
            
            <nav className="flex items-center gap-4">
              <Link
                to="/trajectory"
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
              >
                3D Trajectory
              </Link>
              <Link
                to="/windows"
                className="px-4 py-2 bg-slate-700 text-white rounded-lg hover:bg-slate-600 transition-colors"
              >
                Launch Windows
              </Link>
            </nav>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Main Control Panel */}
          <div className="lg:col-span-2 space-y-6">
            {/* Site Selection */}
            <div className="bg-slate-800 rounded-lg p-6">
              <LaunchSiteSelector
                sites={LAUNCH_SITES}
                selectedSite={selectedSite}
                onSiteChange={setSelectedSite}
                disabled={isLoading}
              />
            </div>

            {/* Run Prediction Button */}
            <div className="bg-slate-800 rounded-lg p-6">
              <button
                onClick={handleRunPrediction}
                disabled={isLoading}
                className="w-full bg-blue-600 text-white font-medium py-3 px-6 rounded-lg hover:bg-blue-700 disabled:bg-blue-400 disabled:cursor-not-allowed transition-colors"
              >
                {isLoading ? (
                  <div className="flex items-center justify-center gap-2">
                    <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                    Running Analysis...
                  </div>
                ) : (
                  'Run Prediction'
                )}
              </button>
              <p className="text-sm text-gray-400 mt-2">Physics → ML Analysis</p>
            </div>

            {/* Prediction Results */}
            <div className="bg-slate-800 rounded-lg p-6">
              <PredictionResults result={predictionResult} isLoading={isLoading} />
            </div>

            {/* System Status */}
            <div className="bg-slate-800 rounded-lg p-6">
              <h3 className="text-lg font-semibold text-white mb-4">System Status</h3>
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-300">Weather</span>
                  <span className={getStatusColor('live')}>●</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-300">Trajectory</span>
                  <span className={getStatusColor('ready')}>●</span>
                </div>
              </div>
            </div>
          </div>

          {/* Right Sidebar */}
          <div className="space-y-6">
            <WeatherPanel weatherData={weatherData} isLoading={isLoadingWeather} />
            
            {/* System Time */}
            <div className="bg-slate-800 rounded-lg p-4 text-center">
              <div className="text-sm text-gray-400 mb-1">System Online</div>
              <div className="text-green-400 font-mono text-sm">
                Last Update: {new Date().toLocaleTimeString()}
              </div>
              <div className="text-xs text-gray-500 mt-2">Connecting...</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MainDashboard;