import React from 'react';
import { LaunchWindow } from '../types';

interface LaunchWindowsTableProps {
  windows: LaunchWindow[];
  siteName: string;
  isLoading?: boolean;
}

const LaunchWindowsTable: React.FC<LaunchWindowsTableProps> = ({
  windows,
  siteName,
  isLoading = false
}) => {
  if (isLoading) {
    return (
      <div className="bg-slate-800 rounded-lg p-6">
        <div className="animate-pulse">
          <div className="h-6 bg-slate-600 rounded w-48 mb-4"></div>
          <div className="space-y-3">
            {Array.from({ length: 5 }).map((_, i) => (
              <div key={i} className="h-12 bg-slate-600 rounded"></div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  if (!windows.length) {
    return (
      <div className="bg-slate-800 rounded-lg p-6 text-center">
        <svg className="w-16 h-16 mx-auto mb-4 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
        <h3 className="text-lg font-medium text-white mb-2">No Launch Windows Found</h3>
        <p className="text-gray-400">Try adjusting the search parameters or select a different site.</p>
      </div>
    );
  }

  const getWindowScoreColor = (score: number) => {
    if (score >= 0.8) return 'text-green-400 bg-green-900';
    if (score >= 0.6) return 'text-yellow-400 bg-yellow-900';
    return 'text-red-400 bg-red-900';
  };

  const getWindowScoreText = (score: number) => {
    if (score >= 0.8) return 'EXCELLENT';
    if (score >= 0.6) return 'GOOD';
    return 'MARGINAL';
  };

  const formatDateTime = (dateString: string) => {
    const date = new Date(dateString);
    return {
      date: date.toLocaleDateString(),
      time: date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };
  };

  return (
    <div className="bg-slate-800 rounded-lg p-6">
      <h3 className="text-lg font-semibold text-white mb-4">
        Optimal Launch Windows - {siteName}
      </h3>
      
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-700">
              <th className="text-left py-3 px-2 text-gray-300 font-medium">Date</th>
              <th className="text-left py-3 px-2 text-gray-300 font-medium">Time</th>
              <th className="text-center py-3 px-2 text-gray-300 font-medium">Window Score</th>
              <th className="text-center py-3 px-2 text-gray-300 font-medium">Success %</th>
              <th className="text-center py-3 px-2 text-gray-300 font-medium">Temp</th>
              <th className="text-center py-3 px-2 text-gray-300 font-medium">Wind</th>
              <th className="text-center py-3 px-2 text-gray-300 font-medium">Status</th>
            </tr>
          </thead>
          <tbody>
            {windows.map((window, index) => {
              const { date, time } = formatDateTime(window.launch_time);
              const scoreColor = getWindowScoreColor(window.window_score);
              const scoreText = getWindowScoreText(window.window_score);
              
              return (
                <tr 
                  key={index} 
                  className={`border-b border-slate-700 hover:bg-slate-700 transition-colors ${
                    index === 0 ? 'bg-slate-750' : ''
                  }`}
                >
                  <td className="py-3 px-2">
                    <div className="text-white font-medium">{date}</div>
                  </td>
                  
                  <td className="py-3 px-2">
                    <div className="text-blue-400 font-mono">{time}</div>
                  </td>
                  
                  <td className="py-3 px-2 text-center">
                    <span className={`inline-block px-2 py-1 rounded text-xs font-medium ${scoreColor}`}>
                      {window.window_score.toFixed(3)}
                    </span>
                    <div className="text-xs text-gray-400 mt-1">{scoreText}</div>
                  </td>
                  
                  <td className="py-3 px-2 text-center">
                    <div className="text-white font-medium">
                      {(window.success_probability * 100).toFixed(1)}%
                    </div>
                  </td>
                  
                  <td className="py-3 px-2 text-center">
                    <div className="text-white">
                      {window.weather_conditions.temperature_c.toFixed(1)}°C
                    </div>
                  </td>
                  
                  <td className="py-3 px-2 text-center">
                    <div className="text-white">
                      {window.weather_conditions.wind_speed_ms.toFixed(1)} m/s
                    </div>
                  </td>
                  
                  <td className="py-3 px-2 text-center">
                    <div className={`inline-flex items-center gap-1 ${
                      window.go_for_launch ? 'text-green-400' : 'text-red-400'
                    }`}>
                      <div className={`w-2 h-2 rounded-full ${
                        window.go_for_launch ? 'bg-green-400' : 'bg-red-400'
                      }`} />
                      {window.go_for_launch ? 'GO' : 'NO-GO'}
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      
      {windows.length > 0 && (
        <div className="mt-4 pt-4 border-t border-slate-700">
          <div className="flex items-center justify-between text-sm text-gray-400">
            <div>
              Showing top {windows.length} launch opportunities
            </div>
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-1">
                <div className="w-2 h-2 bg-green-400 rounded-full" />
                <span>Go for Launch</span>
              </div>
              <div className="flex items-center gap-1">
                <div className="w-2 h-2 bg-red-400 rounded-full" />
                <span>Weather Hold</span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default LaunchWindowsTable;