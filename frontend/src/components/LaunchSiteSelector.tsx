import React from 'react';
import { LaunchSite } from '../types';

interface LaunchSiteSelectorProps {
  sites: LaunchSite[];
  selectedSite: string;
  onSiteChange: (siteCode: string) => void;
  disabled?: boolean;
}

const LaunchSiteSelector: React.FC<LaunchSiteSelectorProps> = ({
  sites,
  selectedSite,
  onSiteChange,
  disabled = false
}) => {
  return (
    <div className="space-y-2">
      <label className="block text-sm font-medium text-gray-300">
        Launch Site
      </label>
      
      <select
        value={selectedSite}
        onChange={(e) => onSiteChange(e.target.value)}
        disabled={disabled}
        className="w-full bg-slate-700 text-white rounded-lg px-4 py-3 border border-slate-600 focus:border-blue-500 focus:ring-2 focus:ring-blue-500 focus:outline-none disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {sites.map((site) => (
          <option key={site.code} value={site.code}>
            {site.name}
          </option>
        ))}
      </select>
      
      {selectedSite && (
        <div className="text-sm text-gray-400 mt-2">
          <div className="flex items-center gap-1">
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
            </svg>
            {sites.find(s => s.code === selectedSite)?.latitude.toFixed(3)}°, {sites.find(s => s.code === selectedSite)?.longitude.toFixed(3)}°
          </div>
        </div>
      )}
    </div>
  );
};

export default LaunchSiteSelector;