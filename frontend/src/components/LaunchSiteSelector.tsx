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
    <div>
      <label className="block text-sm font-medium text-gray-300 mb-2">
        Launch Site
      </label>
      <select
        value={selectedSite}
        onChange={(e) => onSiteChange(e.target.value)}
        disabled={disabled}
        className="w-full bg-slate-700 text-white rounded-lg px-4 py-2 border border-slate-600 focus:border-blue-500 focus:ring-2 focus:ring-blue-500 focus:outline-none disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {sites.map((site) => (
          <option key={site.code} value={site.code}>
            {site.name}
          </option>
        ))}
      </select>
    </div>
  );
};

export default LaunchSiteSelector;