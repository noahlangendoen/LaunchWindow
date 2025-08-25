import React, { useState } from 'react';
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
  const [expandedRow, setExpandedRow] = useState<number | null>(null);

  const toggleRow = (index: number) => {
    setExpandedRow(expandedRow === index ? null : index);
  };
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

  const getWeatherConstraints = (weather: any) => {
    const constraints = [
      {
        name: 'Wind Speed',
        value: `${weather.wind_speed_ms.toFixed(1)} m/s`,
        limit: '< 12.0 m/s',
        status: weather.wind_speed_ms < 12.0,
        severity: weather.wind_speed_ms > 15 ? 'critical' : weather.wind_speed_ms > 10 ? 'warning' : 'good'
      },
      {
        name: 'Temperature', 
        value: `${weather.temperature_c.toFixed(1)}°C`,
        limit: '-10°C to 40°C',
        status: weather.temperature_c >= -10 && weather.temperature_c <= 40,
        severity: weather.temperature_c < -5 || weather.temperature_c > 35 ? 'warning' : 'good'
      },
      {
        name: 'Humidity',
        value: `${weather.humidity_percent}%`,
        limit: '< 95%',
        status: weather.humidity_percent < 95,
        severity: weather.humidity_percent > 90 ? 'warning' : 'good'
      },
      {
        name: 'Precipitation',
        value: `${(weather.rain_1h_mm || 0).toFixed(1)} mm/h`,
        limit: '< 0.5 mm/h',
        status: (weather.rain_1h_mm || 0) < 0.5,
        severity: (weather.rain_1h_mm || 0) > 0.1 ? 'warning' : 'good'
      },
      {
        name: 'Cloud Cover',
        value: `${weather.cloud_cover_percent}%`,
        limit: '< 75%',
        status: weather.cloud_cover_percent < 75,
        severity: weather.cloud_cover_percent > 85 ? 'warning' : 'good'
      },
      {
        name: 'Visibility',
        value: `${(weather.visibility_m / 1000).toFixed(1)} km`,
        limit: '> 5 km',
        status: weather.visibility_m > 5000,
        severity: weather.visibility_m < 3000 ? 'warning' : 'good'
      }
    ];

    return constraints;
  };

  const getFailureReasons = (weather: any) => {
    const constraints = getWeatherConstraints(weather);
    const failures = constraints.filter(c => !c.status);
    
    if (failures.length === 0) return "Weather constraints satisfied";
    
    return failures.map(f => `${f.name}: ${f.value} (limit: ${f.limit})`).join('\n');
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
              <th className="text-center py-3 px-2 text-gray-300 font-medium">Details</th>
            </tr>
          </thead>
          <tbody>
            {windows.map((window, index) => {
              const { date, time } = formatDateTime(window.launch_time);
              const scoreColor = getWindowScoreColor(window.window_score);
              const scoreText = getWindowScoreText(window.window_score);
              const constraints = getWeatherConstraints(window.weather_conditions);
              const isExpanded = expandedRow === index;
              
              return (
                <React.Fragment key={index}>
                  <tr 
                    className={`border-b border-slate-700 hover:bg-slate-700 transition-colors cursor-pointer ${
                      index === 0 ? 'bg-slate-750' : ''
                    } ${isExpanded ? 'bg-slate-700' : ''}`}
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
                      {!window.go_for_launch && (
                        <div className="relative group ml-1">
                          <svg 
                            className="w-4 h-4 text-red-400 hover:text-red-300 cursor-help" 
                            fill="none" 
                            stroke="currentColor" 
                            viewBox="0 0 24 24"
                          >
                            <path 
                              strokeLinecap="round" 
                              strokeLinejoin="round" 
                              strokeWidth={2} 
                              d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" 
                            />
                          </svg>
                          <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-3 py-2 bg-slate-900 text-white text-xs rounded-lg shadow-lg border border-slate-700 whitespace-pre-line opacity-0 group-hover:opacity-100 transition-opacity duration-200 z-50 max-w-xs">
                            <div className="font-semibold mb-1 text-red-400">Launch Hold Reasons:</div>
                            {getFailureReasons(window.weather_conditions)}
                            <div className="absolute top-full left-1/2 transform -translate-x-1/2 border-4 border-transparent border-t-slate-900"></div>
                          </div>
                        </div>
                      )}
                    </div>
                  </td>
                  
                  <td className="py-3 px-2 text-center">
                    <button
                      onClick={() => toggleRow(index)}
                      className="text-blue-400 hover:text-blue-300 text-xs font-medium px-2 py-1 rounded border border-blue-600 hover:border-blue-500 transition-colors"
                    >
                      {isExpanded ? 'Hide' : 'View'}
                    </button>
                  </td>
                </tr>
                
                {isExpanded && (
                  <tr>
                    <td colSpan={8} className="bg-slate-750 p-4">
                      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                        
                        {/* Weather Constraints */}
                        <div>
                          <h4 className="text-sm font-semibold text-white mb-3">Weather Constraints Analysis</h4>
                          <div className="space-y-2">
                            {constraints.map((constraint, idx) => (
                              <div key={idx} className="flex items-center justify-between p-2 rounded bg-slate-800">
                                <div className="flex items-center gap-2">
                                  <div className={`w-2 h-2 rounded-full ${
                                    constraint.status ? 'bg-green-400' : 
                                    constraint.severity === 'critical' ? 'bg-red-500' : 'bg-yellow-500'
                                  }`} />
                                  <span className="text-sm text-gray-300">{constraint.name}</span>
                                </div>
                                <div className="text-right">
                                  <div className="text-sm text-white font-medium">{constraint.value}</div>
                                  <div className="text-xs text-gray-400">{constraint.limit}</div>
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                        
                        {/* Launch Window Analysis */}
                        <div>
                          <h4 className="text-sm font-semibold text-white mb-3">Launch Window Analysis</h4>
                          <div className="space-y-3">
                            <div className="p-3 rounded bg-slate-800">
                              <div className="flex justify-between items-center mb-2">
                                <span className="text-sm text-gray-300">Window Score</span>
                                <span className={`text-sm font-medium px-2 py-1 rounded ${scoreColor}`}>
                                  {window.window_score.toFixed(3)} ({scoreText})
                                </span>
                              </div>
                              <div className="text-xs text-gray-400">
                                {window.go_for_launch 
                                  ? "High score indicates optimal orbital mechanics and favorable weather"
                                  : "Reduced score due to weather constraints that violate launch safety limits"
                                }
                              </div>
                            </div>
                            
                            <div className="p-3 rounded bg-slate-800">
                              <div className="flex justify-between items-center mb-2">
                                <span className="text-sm text-gray-300">Success Probability</span>
                                <span className="text-sm font-medium text-white">
                                  {(window.success_probability * 100).toFixed(1)}%
                                </span>
                              </div>
                              <div className="text-xs text-gray-400">
                                Based on vehicle performance, trajectory analysis, and mission requirements
                              </div>
                            </div>
                            
                            <div className="p-3 rounded bg-slate-800">
                              <div className="flex justify-between items-center mb-2">
                                <span className="text-sm text-gray-300">Overall Assessment</span>
                                <span className={`text-sm font-medium ${
                                  window.go_for_launch ? 'text-green-400' : 'text-red-400'
                                }`}>
                                  {window.go_for_launch ? 'CLEARED FOR LAUNCH' : 'LAUNCH HOLD RECOMMENDED'}
                                </span>
                              </div>
                              <div className="text-xs text-gray-400">
                                {window.go_for_launch 
                                  ? "All weather constraints satisfied for safe launch operations"
                                  : "Weather conditions exceed acceptable risk thresholds for launch"
                                }
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>
                    </td>
                  </tr>
                )}
                </React.Fragment>
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