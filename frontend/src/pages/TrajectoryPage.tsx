import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import TrajectoryVisualization from '../components/TrajectoryVisualization';
import WeatherPanel from '../components/WeatherPanel';
import LaunchSiteSelector from '../components/LaunchSiteSelector';
import { LaunchSite, LaunchTrajectory, WeatherData } from '../types';
import { apiService } from '../services/api';

const LAUNCH_SITES: LaunchSite[] = [
  { code: 'KSC', name: 'Kennedy Space Center', latitude: 28.5721, longitude: -80.6480, timezone: 'America/New_York' },
  { code: 'VSFB', name: 'Vandenberg Space Force Base', latitude: 34.7420, longitude: -120.5724, timezone: 'America/Los_Angeles' },
  { code: 'CCAFS', name: 'Cape Canaveral Space Force Station', latitude: 28.3922, longitude: -80.6077, timezone: 'America/New_York' }
];

const TrajectoryPage: React.FC = () => {
  const [selectedSite, setSelectedSite] = useState<string>('KSC');
  const [trajectoryData, setTrajectoryData] = useState<LaunchTrajectory | null>(null);
  const [weatherData, setWeatherData] = useState<WeatherData | null>(null);
  const [isSimulating, setIsSimulating] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [targetAltitude, setTargetAltitude] = useState(400);
  const [targetInclination, setTargetInclination] = useState(28.5);
  const [visibleSites, setVisibleSites] = useState<string[]>(['KSC', 'VSFB', 'CCAFS']);

  useEffect(() => {
    loadWeatherData();
    // Adjust default inclination based on site
    if (selectedSite === 'VSFB') {
      setTargetInclination(97.5);
    } else {
      setTargetInclination(28.5);
    }
  }, [selectedSite]);

  const loadWeatherData = async () => {
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
    }
  };

  const calculateTrajectory = async () => {
    setIsLoading(true);
    setIsSimulating(true);
    
    try {
      const targetOrbit = {
        altitude_km: targetAltitude,
        inclination_deg: targetInclination,
        orbit_type: 'LEO'
      };
      
      const trajectory = await apiService.calculateTrajectory(selectedSite, targetOrbit);
      setTrajectoryData(trajectory);
    } catch (error) {
      console.error('Failed to calculate trajectory:', error);
      // Mock trajectory data for development
      const mockTrajectory: LaunchTrajectory = {
        trajectory_points: generateMockTrajectoryPoints(),
        success_probability: 0.92,
        mission_objectives_met: true,
        max_dynamic_pressure: 45000,
        max_g_force: 3.8,
        fuel_remaining_kg: 2500,
        final_orbit_elements: {
          apogee_km: targetAltitude + 10,
          perigee_km: targetAltitude - 10,
          inclination_deg: targetInclination
        },
        total_delta_v: 9200
      };
      setTrajectoryData(mockTrajectory);
    } finally {
      setIsLoading(false);
    }
  };

  const generateMockTrajectoryPoints = () => {
    const points = [];
    const site = LAUNCH_SITES.find(s => s.code === selectedSite);
    if (!site) return [];

    const earthRadius = 6371; // Earth radius in km
    
    for (let i = 0; i < 100; i++) {
      const t = i * 5; // 5 second intervals
      const progress = i / 99; // 0 to 1
      
      // Realistic altitude progression: exponential curve that levels off
      const altitude = targetAltitude * (1 - Math.exp(-3 * progress));
      
      // Launch trajectory parameters
      const downrange = progress * 1000; // Distance traveled horizontally (km)
      
      // Start at launch site position on Earth surface
      const lat0Rad = (site.latitude * Math.PI) / 180;
      const lon0Rad = (site.longitude * Math.PI) / 180;
      
      // Calculate downrange movement (simplified - eastward launch)
      const earthCircumference = 2 * Math.PI * earthRadius;
      const deltaLonRad = (downrange / earthRadius) * Math.cos(lat0Rad); // Account for latitude
      
      const lat = site.latitude;
      const lon = site.longitude + (deltaLonRad * 180) / Math.PI;
      
      // Convert to Cartesian coordinates (km from Earth center)
      const latRad = (lat * Math.PI) / 180;
      const lonRad = (lon * Math.PI) / 180;
      const totalRadius = earthRadius + altitude;
      
      const x = totalRadius * Math.cos(latRad) * Math.cos(lonRad);
      const y = totalRadius * Math.sin(latRad);
      const z = totalRadius * Math.cos(latRad) * Math.sin(lonRad);
      
      points.push({
        time_s: t,
        position: [x, y, z] as [number, number, number],
        velocity: [7.8, 0, 0] as [number, number, number],
        altitude_km: altitude,
        velocity_magnitude_ms: 1000 + (progress * 6800), // 1 km/s to 7.8 km/s
        acceleration: [0, -9.8, 0] as [number, number, number],
        mass_kg: 550000 - (i * 1000),
        thrust_n: Math.max(0, 7607000 - (i * 60000)),
        drag_n: Math.max(0, 50000 - (i * 500)),
        dynamic_pressure_pa: Math.max(0, 50000 - (i * 500)),
        mach_number: Math.max(1, 25 - (i * 0.2)),
        g_force: Math.max(1, 4 - (i * 0.03)),
        timestamp: new Date(Date.now() + t * 1000).toISOString()
      });
    }
    return points;
  };

  const handleSiteToggle = (siteCode: string) => {
    setVisibleSites(prev => 
      prev.includes(siteCode) 
        ? prev.filter(code => code !== siteCode)
        : [...prev, siteCode]
    );
  };

  return (
    <div className="min-h-screen bg-slate-900">
      {/* Header */}
      <header className="bg-slate-800 border-b border-slate-700">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Link to="/" className="text-gray-400 hover:text-white">
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
                </svg>
              </Link>
              <h1 className="text-2xl font-bold text-white">3D Trajectory Visualization</h1>
            </div>
            
            <div className="flex items-center gap-2 text-white text-sm">
              <div className={`w-2 h-2 rounded-full ${isSimulating ? 'bg-green-400' : 'bg-gray-400'}`} />
              {isSimulating ? 'Real-time Simulation' : 'Static View'}
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 py-8">
        <div className="grid grid-cols-1 xl:grid-cols-4 gap-8">
          {/* Main Visualization */}
          <div className="xl:col-span-3">
            <div className="bg-slate-800 rounded-lg p-6 mb-6">
              <TrajectoryVisualization
                trajectoryData={trajectoryData}
                isSimulating={isSimulating}
                visibleSites={visibleSites}
                onSiteToggle={handleSiteToggle}
                className="h-96"
              />
            </div>

            {/* Trajectory Controls */}
            <div className="bg-slate-800 rounded-lg p-6">
              <h3 className="text-lg font-semibold text-white mb-4">Mission Parameters</h3>
              
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
                <div>
                  <LaunchSiteSelector
                    sites={LAUNCH_SITES}
                    selectedSite={selectedSite}
                    onSiteChange={setSelectedSite}
                    disabled={isLoading}
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Target Altitude (km)
                  </label>
                  <input
                    type="number"
                    value={targetAltitude}
                    onChange={(e) => setTargetAltitude(Number(e.target.value))}
                    disabled={isLoading}
                    className="w-full bg-slate-700 text-white rounded-lg px-4 py-2 border border-slate-600 focus:border-blue-500 focus:outline-none disabled:opacity-50"
                    min="200"
                    max="2000"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Inclination (°)
                  </label>
                  <input
                    type="number"
                    value={targetInclination}
                    onChange={(e) => setTargetInclination(Number(e.target.value))}
                    disabled={isLoading}
                    className="w-full bg-slate-700 text-white rounded-lg px-4 py-2 border border-slate-600 focus:border-blue-500 focus:outline-none disabled:opacity-50"
                    min="0"
                    max="180"
                    step="0.1"
                  />
                </div>
              </div>

              <button
                onClick={calculateTrajectory}
                disabled={isLoading}
                className="w-full md:w-auto bg-blue-600 text-white font-medium py-3 px-8 rounded-lg hover:bg-blue-700 disabled:bg-blue-400 disabled:cursor-not-allowed transition-colors"
              >
                {isLoading ? (
                  <div className="flex items-center gap-2">
                    <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                    Calculating...
                  </div>
                ) : (
                  'Calculate Trajectory'
                )}
              </button>
            </div>

            {/* Trajectory Data */}
            {trajectoryData && (
              <div className="bg-slate-800 rounded-lg p-6 mt-6">
                <h3 className="text-lg font-semibold text-white mb-4">Mission Results -- Based Vehicle Specifications</h3>
                
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  <div className="text-center">
                    <div className="text-2xl font-bold text-green-400">
                      {(trajectoryData.success_probability * 100).toFixed(1)}%
                    </div>
                    <div className="text-gray-400">Success Probability</div>
                  </div>
                  
                  <div className="text-center">
                    <div className="text-2xl font-bold text-blue-400">
                      {trajectoryData.max_g_force.toFixed(1)}g
                    </div>
                    <div className="text-gray-400">Max G-Force</div>
                  </div>
                  
                  <div className="text-center">
                    <div className="text-2xl font-bold text-purple-400">
                      {(trajectoryData.max_dynamic_pressure / 1000).toFixed(1)}
                    </div>
                    <div className="text-gray-400">Max Q (kPa)</div>
                  </div>
                  
                  <div className="text-center">
                    <div className="text-2xl font-bold text-yellow-400">
                      {(trajectoryData.fuel_remaining_kg / 1000).toFixed(1)}t
                    </div>
                    <div className="text-gray-400">Fuel Remaining</div>
                  </div>
                </div>
                
                {trajectoryData.final_orbit_elements && (
                  <div className="mt-4 pt-4 border-t border-slate-700">
                    <h4 className="font-medium text-white mb-2">Final Orbit</h4>
                    <div className="grid grid-cols-3 gap-4 text-sm">
                      <div>
                        <span className="text-gray-400">Apogee: </span>
                        <span className="text-white">{trajectoryData.final_orbit_elements.apogee_km.toFixed(1)} km</span>
                      </div>
                      <div>
                        <span className="text-gray-400">Perigee: </span>
                        <span className="text-white">{trajectoryData.final_orbit_elements.perigee_km.toFixed(1)} km</span>
                      </div>
                      <div>
                        <span className="text-gray-400">Inclination: </span>
                        <span className="text-white">{trajectoryData.final_orbit_elements.inclination_deg.toFixed(1)}°</span>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Right Sidebar */}
          <div className="space-y-6">
            <WeatherPanel weatherData={weatherData} />
            
            {/* Mission Status */}
            <div className="bg-slate-800 rounded-lg p-4">
              <h4 className="font-medium text-white mb-3">Mission Status</h4>
              
              {trajectoryData ? (
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-300">Objectives Met</span>
                    <span className={trajectoryData.mission_objectives_met ? 'text-green-400' : 'text-red-400'}>
                      {trajectoryData.mission_objectives_met ? 'Yes' : 'No'}
                    </span>
                  </div>
                  
                  <div className="flex justify-between">
                    <span className="text-gray-300">Total ΔV</span>
                    <span className="text-white">{trajectoryData.total_delta_v.toFixed(0)} m/s</span>
                  </div>
                  
                  <div className="flex justify-between">
                    <span className="text-gray-300">Data Points</span>
                    <span className="text-white">{trajectoryData.trajectory_points.length}</span>
                  </div>
                </div>
              ) : (
                <p className="text-gray-400 text-sm">No trajectory calculated</p>
              )}
            </div>

            {/* System Time */}
            <div className="bg-slate-800 rounded-lg p-4 text-center">
              <div className="text-sm text-gray-400 mb-1">System Online</div>
              <div className="text-green-400 font-mono text-sm">
                {new Date().toLocaleTimeString()}
              </div>
              <div className="text-xs text-gray-500 mt-2">Real-time Data</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TrajectoryPage;