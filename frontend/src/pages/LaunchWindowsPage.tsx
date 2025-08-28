import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import LaunchSiteSelector from '../components/LaunchSiteSelector';
import LaunchWindowsTable from '../components/LaunchWindowsTable';
import { LaunchSite, LaunchWindow } from '../types';
import { apiService } from '../services/api';

const LAUNCH_SITES: LaunchSite[] = [
  { code: 'KSC', name: 'Kennedy Space Center', latitude: 28.5721, longitude: -80.6480, timezone: 'America/New_York' },
  { code: 'VSFB', name: 'Vandenberg Space Force Base', latitude: 34.7420, longitude: -120.5724, timezone: 'America/Los_Angeles' },
  { code: 'CCAFS', name: 'Cape Canaveral Space Force Station', latitude: 28.3922, longitude: -80.6077, timezone: 'America/New_York' }
];

const LaunchWindowsPage: React.FC = () => {
  const [selectedSite, setSelectedSite] = useState<string>('KSC');
  const [duration, setDuration] = useState(48);
  const [launchWindows, setLaunchWindows] = useState<{ [key: string]: LaunchWindow[] }>({});
  const [isLoading, setIsLoading] = useState(false);
  const [searchAll, setSearchAll] = useState(false);

  const findWindows = async () => {
    setIsLoading(true);
    try {
      if (searchAll) {
        // Search all sites
        const allWindows: { [key: string]: LaunchWindow[] } = {};
        
        for (const site of LAUNCH_SITES) {
          try {
            const windows = await apiService.findOptimalWindows(site.code, duration);
            allWindows[site.code] = windows;
          } catch (error) {
            console.error(`Failed to find windows for ${site.code}:`, error);
            // Generate mock data for development
            allWindows[site.code] = generateMockWindows();
          }
        }
        
        setLaunchWindows(allWindows);
      } else {
        // Search single site
        try {
          const windows = await apiService.findOptimalWindows(selectedSite, duration);
          setLaunchWindows({ [selectedSite]: windows });
        } catch (error) {
          console.error('Failed to find windows:', error);
          setLaunchWindows({ [selectedSite]: generateMockWindows() });
        }
      }
    } finally {
      setIsLoading(false);
    }
  };

  const generateMockWindows = (): LaunchWindow[] => {
    const windows = [];
    const now = new Date();
    
    for (let i = 0; i < 10; i++) {
      const launchTime = new Date(now.getTime() + (i * 6 + Math.random() * 6) * 3600000); // Every 6-12 hours
      const score = Math.max(0.3, Math.random() * 0.7 + 0.3); // Between 0.3 and 1.0
      
      windows.push({
        launch_time: launchTime.toISOString(),
        window_score: score,
        success_probability: Math.min(0.95, score + Math.random() * 0.2),
        weather_conditions: {
          temperature_c: 15 + Math.random() * 20,
          wind_speed_ms: Math.random() * 12,
          humidity_percent: 40 + Math.random() * 40,
          pressure_hpa: 1005 + Math.random() * 20,
          cloud_cover_percent: Math.random() * 80,
          visibility_m: 8000 + Math.random() * 2000
        },
        go_for_launch: score > 0.5 && Math.random() > 0.2
      });
    }
    
    // Sort by GO status first (GO windows always come before NO-GO), then by score
    return windows.sort((a, b) => {
      // If GO status is different, prioritize GO windows
      if (a.go_for_launch !== b.go_for_launch) {
        return a.go_for_launch ? -1 : 1; // GO windows come first
      }
      // If GO status is the same, sort by window score
      return b.window_score - a.window_score;
    });
  };

  const getBestOverallWindow = (): { window: LaunchWindow; siteCode: string } | null => {
    let bestWindow: LaunchWindow | null = null;
    let bestSite = '';
    let bestScore = 0;
    
    Object.entries(launchWindows).forEach(([siteCode, windows]) => {
      // Find the highest scoring window in this site, regardless of GO status
      windows.forEach(window => {
        if (window.window_score > bestScore) {
          bestWindow = window;
          bestSite = siteCode;
          bestScore = window.window_score;
        }
      });
    });
    
    return bestWindow ? { window: bestWindow, siteCode: bestSite } : null;
  };

  const getTotalWindowsCount = () => {
    return Object.values(launchWindows).reduce((total, windows) => total + windows.length, 0);
  };

  const getGoWindowsCount = () => {
    return Object.values(launchWindows).reduce(
      (total, windows) => total + windows.filter(w => w.go_for_launch).length, 
      0
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
              <div>
                <h1 className="text-2xl font-bold text-white">Launch Window Optimization</h1>
                <p className="text-sm text-gray-400">Find optimal launch opportunities across multiple sites</p>
              </div>
            </div>
            
            <nav className="flex items-center gap-4">
              <Link
                to="/trajectory"
                className="px-4 py-2 bg-slate-700 text-white rounded-lg hover:bg-slate-600 transition-colors"
              >
                3D Trajectory
              </Link>
            </nav>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 py-8">
        {/* Search Controls */}
        <div className="bg-slate-800 rounded-lg p-6 mb-8">
          <h2 className="text-lg font-semibold text-white mb-4">Search Parameters</h2>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
            {/* Launch Site Selection */}
            <div className="flex flex-col">
              <div className="mb-4">
                <label className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={searchAll}
                    onChange={(e) => setSearchAll(e.target.checked)}
                    className="rounded bg-slate-700 border-slate-600"
                  />
                  <span className="text-sm text-gray-300">Search All Sites</span>
                </label>
              </div>
              
              <div className="flex-1 flex flex-col justify-end">
                {!searchAll && (
                  <>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Launch Site
                    </label>
                    <LaunchSiteSelector
                      sites={LAUNCH_SITES}
                      selectedSite={selectedSite}
                      onSiteChange={setSelectedSite}
                      disabled={isLoading}
                    />
                  </>
                )}
              </div>
            </div>
            
            {/* Search Duration */}
            <div className="flex flex-col justify-end">
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Search Duration (hours)
              </label>
              <select
                value={duration}
                onChange={(e) => setDuration(Number(e.target.value))}
                disabled={isLoading}
                className="w-full bg-slate-700 text-white rounded-lg px-4 py-2 border border-slate-600 focus:border-blue-500 focus:outline-none disabled:opacity-50 h-10"
              >
                <option value={24}>24 hours</option>
                <option value={48}>48 hours</option>
                <option value={72}>72 hours</option>
                <option value={120}>5 days</option>
              </select>
            </div>
            
            {/* Find Windows Button */}
            <div className="flex flex-col justify-end">
              <div className="h-6 mb-2"></div> {/* Spacer to align with labels */}
              <button
                onClick={findWindows}
                disabled={isLoading}
                className="w-full bg-blue-600 text-white font-medium py-2 px-6 rounded-lg hover:bg-blue-700 disabled:bg-blue-400 disabled:cursor-not-allowed transition-colors h-10"
              >
                {isLoading ? (
                  <div className="flex items-center justify-center gap-2">
                    <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                    Searching...
                  </div>
                ) : (
                  'Find Windows'
                )}
              </button>
            </div>
          </div>
          
          {/* Summary Stats */}
          {Object.keys(launchWindows).length > 0 && (
            <div className="bg-slate-700 rounded-lg p-4 mb-6">
              <h4 className="font-medium text-white mb-3">Search Results Summary</h4>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="text-center">
                  <div className="text-2xl font-bold text-white">{getTotalWindowsCount()}</div>
                  <div className="text-sm text-gray-300">Total Windows</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-green-400">{getGoWindowsCount()}</div>
                  <div className="text-sm text-gray-300">Go Windows</div>
                </div>
                <div className="text-center">
                  {(() => {
                    const best = getBestOverallWindow();
                    return best ? (
                      <div>
                        <div className="text-2xl font-bold text-blue-400">
                          {best.window.window_score.toFixed(3)}
                        </div>
                        <div className="text-sm text-gray-300">Best Score</div>
                        <div className="text-xs text-gray-400 mt-1">
                          {LAUNCH_SITES.find(s => s.code === best.siteCode)?.name}
                        </div>
                      </div>
                    ) : (
                      <div>
                        <div className="text-2xl font-bold text-gray-500">-</div>
                        <div className="text-sm text-gray-300">Best Score</div>
                      </div>
                    );
                  })()}
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Results */}
        <div className="space-y-8">
          {Object.entries(launchWindows).map(([siteCode, windows]) => {
            const site = LAUNCH_SITES.find(s => s.code === siteCode);
            return (
              <LaunchWindowsTable
                key={siteCode}
                windows={windows}
                siteName={site?.name || siteCode}
                isLoading={isLoading && Object.keys(launchWindows).length === 0}
              />
            );
          })}
          
          {Object.keys(launchWindows).length === 0 && !isLoading && (
            <div className="bg-slate-800 rounded-lg p-12 text-center">
              <svg className="w-16 h-16 mx-auto mb-4 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
              <h3 className="text-xl font-medium text-white mb-2">Ready to Search</h3>
              <p className="text-gray-400 mb-6">Configure your search parameters and click "Find Windows" to discover optimal launch opportunities.</p>
              <button
                onClick={findWindows}
                className="bg-blue-600 text-white font-medium py-2 px-6 rounded-lg hover:bg-blue-700 transition-colors"
              >
                Find Launch Windows
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default LaunchWindowsPage;