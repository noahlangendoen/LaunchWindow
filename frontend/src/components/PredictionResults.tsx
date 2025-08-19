import React from 'react';
import { PredictionResult } from '../types';

interface PredictionResultsProps {
  result: PredictionResult | null;
  isLoading?: boolean;
}

const PredictionResults: React.FC<PredictionResultsProps> = ({ result, isLoading = false }) => {
  if (isLoading) {
    return (
      <div className="space-y-4">
        <div className="animate-pulse">
          <div className="h-8 bg-slate-600 rounded w-48 mb-4"></div>
          <div className="h-16 bg-slate-600 rounded mb-4"></div>
          <div className="space-y-2">
            <div className="h-4 bg-slate-600 rounded"></div>
            <div className="h-4 bg-slate-600 rounded"></div>
          </div>
        </div>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="text-center text-gray-400 py-8">
        <svg className="w-16 h-16 mx-auto mb-4 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
        </svg>
        <p className="text-lg font-medium">No prediction results</p>
        <p className="text-sm">Click "Run Prediction" to analyze launch feasibility</p>
      </div>
    );
  }

  const getSuccessColor = (probability: number) => {
    if (probability >= 0.8) return 'text-green-400';
    if (probability >= 0.6) return 'text-yellow-400';
    return 'text-red-400';
  };

  const getRiskColor = (risk: string) => {
    switch (risk.toLowerCase()) {
      case 'low': return 'bg-green-900 text-green-400';
      case 'medium': return 'bg-yellow-900 text-yellow-400';
      case 'high': return 'bg-red-900 text-red-400';
      default: return 'bg-gray-700 text-gray-300';
    }
  };

  const getWeatherStatusColor = (status: string) => {
    return status.toLowerCase() === 'go' ? 'text-green-400' : 'text-red-400';
  };

  return (
    <div className="space-y-6">
      <div className="text-center">
        <div className="text-3xl font-bold mb-2">
          <span className={`${getSuccessColor(result.success_probability)}`}>
            {(result.success_probability * 100).toFixed(1)}%
          </span>
        </div>
        <h3 className="text-lg font-medium text-gray-300">Success Probability</h3>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-slate-800 rounded-lg p-4">
          <h4 className="font-medium text-white mb-2">Risk Assessment</h4>
          <span className={`inline-block px-3 py-1 rounded text-sm font-medium ${getRiskColor(result.risk_assessment)}`}>
            {result.risk_assessment.toUpperCase()}
          </span>
        </div>

        <div className="bg-slate-800 rounded-lg p-4">
          <h4 className="font-medium text-white mb-2">Weather Status</h4>
          <div className="flex items-center gap-2">
            <div className={`w-3 h-3 rounded-full ${result.weather_status.toLowerCase() === 'go' ? 'bg-green-400' : 'bg-red-400'}`}></div>
            <span className={`font-medium ${getWeatherStatusColor(result.weather_status)}`}>
              {result.weather_status}
            </span>
          </div>
        </div>
      </div>

      <div className="bg-slate-800 rounded-lg p-4">
        <h4 className="font-medium text-white mb-2">Recommendation</h4>
        <p className="text-gray-300 text-sm leading-relaxed">
          {result.recommendation}
        </p>
      </div>

      <div className="text-xs text-gray-500 text-center">
        Analysis for {result.site_name}
      </div>
    </div>
  );
};

export default PredictionResults;