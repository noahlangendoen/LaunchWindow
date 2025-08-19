import React from 'react';
import { WeatherData } from '../types';

interface WeatherPanelProps {
  weatherData: WeatherData | null;
  isLoading?: boolean;
}

const WeatherPanel: React.FC<WeatherPanelProps> = ({ weatherData, isLoading = false }) => {
  if (isLoading) {
    return (
      <div className="bg-slate-800 rounded-lg p-6 animate-pulse">
        <div className="flex items-center gap-2 mb-4">
          <div className="w-6 h-6 bg-slate-600 rounded"></div>
          <div className="h-6 bg-slate-600 rounded w-32"></div>
        </div>
        <div className="space-y-3">
          {Array.from({ length: 5 }).map((_, i) => (
            <div key={i} className="flex justify-between">
              <div className="h-4 bg-slate-600 rounded w-24"></div>
              <div className="h-4 bg-slate-600 rounded w-16"></div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (!weatherData) {
    return (
      <div className="bg-slate-800 rounded-lg p-6">
        <div className="flex items-center gap-2 mb-4">
          <svg className="w-6 h-6 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 15a4 4 0 004 4h9a5 5 0 10-.1-9.999 5.002 5.002 0 10-9.78 2.096A4.004 4.004 0 003 15z" />
          </svg>
          <h3 className="text-lg font-semibold text-gray-400">Weather Conditions</h3>
        </div>
        <p className="text-gray-500">No weather data available</p>
      </div>
    );
  }

  const getWindStatus = (windSpeed: number) => {
    if (windSpeed < 10) return { text: 'GOOD', color: 'text-green-400' };
    if (windSpeed < 15) return { text: 'CAUTION', color: 'text-yellow-400' };
    return { text: 'POOR', color: 'text-red-400' };
  };

  const getTempStatus = (temp: number) => {
    if (temp >= -5 && temp <= 30) return { text: 'GOOD', color: 'text-green-400' };
    if (temp >= -10 && temp <= 35) return { text: 'CAUTION', color: 'text-yellow-400' };
    return { text: 'POOR', color: 'text-red-400' };
  };

  const windStatus = getWindStatus(weatherData.wind_speed_ms);
  const tempStatus = getTempStatus(weatherData.temperature_c);

  return (
    <div className="bg-slate-800 rounded-lg p-6">
      <div className="flex items-center gap-2 mb-4">
        <svg className="w-6 h-6 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 15a4 4 0 004 4h9a5 5 0 10-.1-9.999 5.002 5.002 0 10-9.78 2.096A4.004 4.004 0 003 15z" />
        </svg>
        <h3 className="text-lg font-semibold text-white">Weather Conditions</h3>
      </div>
      
      <div className="space-y-3">
        <div className="flex justify-between items-center">
          <span className="text-gray-300">Temperature</span>
          <div className="flex items-center gap-2">
            <span className="text-white font-mono">{weatherData.temperature_c.toFixed(1)}°C</span>
            <span className={`text-xs px-2 py-1 rounded ${tempStatus.color}`}>
              {tempStatus.text}
            </span>
          </div>
        </div>
        
        <div className="flex justify-between items-center">
          <span className="text-gray-300">Wind Speed</span>
          <div className="flex items-center gap-2">
            <span className="text-white font-mono">{weatherData.wind_speed_ms.toFixed(1)} m/s</span>
            <span className={`text-xs px-2 py-1 rounded ${windStatus.color}`}>
              {windStatus.text}
            </span>
          </div>
        </div>
        
        <div className="flex justify-between items-center">
          <span className="text-gray-300">Humidity</span>
          <span className="text-white font-mono">{weatherData.humidity_percent.toFixed(0)}%</span>
        </div>
        
        <div className="flex justify-between items-center">
          <span className="text-gray-300">Pressure</span>
          <span className="text-white font-mono">{weatherData.pressure_hpa.toFixed(1)} hPa</span>
        </div>
        
        <div className="flex justify-between items-center">
          <span className="text-gray-300">Cloud Cover</span>
          <span className="text-white font-mono">{weatherData.cloud_cover_percent.toFixed(0)}%</span>
        </div>
      </div>
    </div>
  );
};

export default WeatherPanel;