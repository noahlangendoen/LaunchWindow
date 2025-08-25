import React, { useRef } from 'react';
import { Canvas} from '@react-three/fiber';
import { OrbitControls, Sphere, Line, Stars, Text } from '@react-three/drei';
import { Vector3} from 'three';
import { LaunchTrajectory } from '../types';

interface EarthProps {
  radius?: number;
}

function Earth({ radius = 1 }: EarthProps) {
  const meshRef = useRef<any>(null);
  
  
  // Removed spinning animation for better launch visualization
  // useFrame(() => {
  //   if (meshRef.current) {
  //     meshRef.current.rotation.y += 0.005;
  //   }
  // });

  return (
    <Sphere ref={meshRef} args={[radius, 64, 32]} position={[0, 0, 0]} rotation={[0, -0.3, 0]}>
      <meshStandardMaterial
        color="#4a90e2" // Brighter, more Earth-like blue
        transparent={false}
        roughness={0.6} // Less rough for more reflection
        metalness={0.0}
        emissive="#1a3d5c" // Slight glow
        emissiveIntensity={0.1}
      />
    </Sphere>
  );
}

interface TrajectoryPathProps {
  trajectoryData: LaunchTrajectory | null;
}

// Launch site definitions
const LAUNCH_SITES = [
  { code: 'KSC', name: 'Kennedy Space Center', latitude: 28.5721, longitude: -80.6480, color: '#ff4444' },
  { code: 'VSFB', name: 'Vandenberg SFB', latitude: 34.7420, longitude: -120.5724, color: '#4444ff' },
  { code: 'CCAFS', name: 'Cape Canaveral SFS', latitude: 28.3922, longitude: -80.6077, color: '#ffff44' }
];

// Detailed US outline coordinates with more geographic accuracy
const US_OUTLINE = [
  // West Coast - California (South to North)
  { lat: 32.53, lon: -117.12 }, // San Diego
  { lat: 32.75, lon: -117.25 }, // Point Loma
  { lat: 33.19, lon: -117.38 }, // Oceanside
  { lat: 33.77, lon: -118.18 }, // Long Beach
  { lat: 34.05, lon: -118.24 }, // Los Angeles
  { lat: 34.42, lon: -119.70 }, // Santa Barbara
  { lat: 35.37, lon: -120.85 }, // San Luis Obispo
  { lat: 36.61, lon: -121.90 }, // Monterey
  { lat: 37.77, lon: -122.42 }, // San Francisco
  { lat: 38.31, lon: -123.06 }, // Point Arena
  { lat: 39.16, lon: -123.77 }, // Mendocino
  { lat: 40.18, lon: -124.21 }, // Eureka
  { lat: 41.75, lon: -124.20 }, // Crescent City
  
  // Oregon Coast
  { lat: 42.00, lon: -124.28 }, // Oregon Border
  { lat: 44.62, lon: -124.05 }, // Newport
  { lat: 45.84, lon: -123.97 }, // Lincoln City
  { lat: 46.19, lon: -123.94 }, // Astoria
  
  // Washington Coast
  { lat: 46.91, lon: -124.10 }, // Westport
  { lat: 47.91, lon: -124.73 }, // La Push
  { lat: 48.39, lon: -124.67 }, // Neah Bay
  { lat: 48.73, lon: -122.76 }, // Bellingham
  { lat: 48.85, lon: -122.48 }, // Blaine (Canada Border)
  
  // Northern Border (West to East)
  { lat: 48.99, lon: -117.03 }, // Spokane area
  { lat: 48.99, lon: -116.05 }, // Idaho Panhandle
  { lat: 48.99, lon: -104.05 }, // Montana
  { lat: 48.99, lon: -97.23 }, // North Dakota
  { lat: 49.00, lon: -95.15 }, // Minnesota
  { lat: 48.66, lon: -94.69 }, // Lake of the Woods
  { lat: 47.28, lon: -94.61 }, // Bemidji area
  
  // Great Lakes Region
  { lat: 46.77, lon: -92.10 }, // Duluth
  { lat: 46.50, lon: -84.35 }, // Sault Ste Marie
  { lat: 45.36, lon: -84.67 }, // Mackinaw
  { lat: 43.65, lon: -83.91 }, // Thumb of Michigan
  { lat: 42.98, lon: -82.43 }, // Detroit
  { lat: 41.64, lon: -81.47 }, // Cleveland
  { lat: 42.16, lon: -79.76 }, // Buffalo
  
  // Northeast Coast
  { lat: 47.46, lon: -67.78 }, // Eastport, Maine
  { lat: 44.32, lon: -68.20 }, // Bar Harbor
  { lat: 43.66, lon: -70.25 }, // Portland, Maine
  { lat: 42.96, lon: -70.88 }, // Portsmouth
  { lat: 42.36, lon: -71.06 }, // Boston
  { lat: 41.49, lon: -71.31 }, // Newport, RI
  { lat: 41.05, lon: -71.51 }, // Block Island Sound
  { lat: 40.74, lon: -74.01 }, // New York City
  { lat: 40.22, lon: -74.76 }, // Philadelphia area
  { lat: 39.29, lon: -75.52 }, // Delaware
  { lat: 38.97, lon: -76.50 }, // Annapolis
  { lat: 37.54, lon: -76.01 }, // Norfolk area
  
  // Southeast Coast
  { lat: 36.85, lon: -75.98 }, // Outer Banks start
  { lat: 35.25, lon: -75.53 }, // Cape Hatteras
  { lat: 34.72, lon: -76.67 }, // Cape Lookout
  { lat: 33.92, lon: -78.02 }, // Myrtle Beach
  { lat: 32.78, lon: -79.93 }, // Charleston
  { lat: 32.08, lon: -81.09 }, // Savannah
  { lat: 30.67, lon: -81.46 }, // Jacksonville
  { lat: 29.19, lon: -81.00 }, // Daytona
  { lat: 28.54, lon: -80.84 }, // Cape Canaveral
  { lat: 27.76, lon: -80.04 }, // West Palm Beach
  { lat: 26.12, lon: -80.14 }, // Fort Lauderdale
  { lat: 25.77, lon: -80.18 }, // Miami
  
  // Florida Keys and Southwest Florida
  { lat: 25.07, lon: -80.45 }, // Key Largo
  { lat: 24.55, lon: -81.78 }, // Key West
  { lat: 25.64, lon: -81.86 }, // Everglades
  { lat: 26.34, lon: -81.78 }, // Naples
  { lat: 27.34, lon: -82.54 }, // Tampa
  { lat: 28.96, lon: -82.46 }, // Crystal River
  { lat: 30.44, lon: -84.28 }, // Tallahassee
  
  // Gulf Coast
  { lat: 30.69, lon: -88.04 }, // Mobile
  { lat: 30.27, lon: -89.63 }, // Biloxi
  { lat: 29.95, lon: -90.08 }, // New Orleans
  { lat: 29.76, lon: -93.84 }, // Lake Charles
  { lat: 29.69, lon: -95.01 }, // Galveston
  { lat: 28.81, lon: -95.97 }, // Matagorda Bay
  { lat: 27.80, lon: -97.40 }, // Corpus Christi
  { lat: 26.07, lon: -97.15 }, // Brownsville
  
  // Texas-Mexico Border
  { lat: 25.90, lon: -97.50 }, // Rio Grande mouth
  { lat: 26.30, lon: -98.30 }, // McAllen area
  { lat: 27.51, lon: -99.51 }, // Laredo
  { lat: 29.36, lon: -100.90 }, // Del Rio
  { lat: 31.33, lon: -104.52 }, // Big Bend
  { lat: 31.79, lon: -106.43 }, // El Paso
  
  // New Mexico-Arizona Border
  { lat: 32.00, lon: -108.21 }, // New Mexico boot heel
  { lat: 31.33, lon: -109.05 }, // Arizona border
  { lat: 32.72, lon: -114.72 }, // Yuma
  
  // California-Mexico Border
  { lat: 32.54, lon: -116.97 }, // Tijuana border
  { lat: 32.53, lon: -117.12 }, // Back to San Diego
];

function USLandmass() {
  const points = US_OUTLINE.map(coord => {
    // Convert lat/lon to 3D coordinates
    // Fix mirroring: Flip longitude sign to correct east/west orientation
    const latRad = (coord.lat * Math.PI) / 180;
    const lonRad = (-coord.lon * Math.PI) / 180; // Flip longitude sign
    const radius = 1.01; // Slightly above Earth surface
    
    // Standard spherical to Cartesian conversion
    // x = r * cos(lat) * cos(lon)
    // y = r * sin(lat) 
    // z = r * cos(lat) * sin(lon)
    const x = radius * Math.cos(latRad) * Math.cos(lonRad);
    const y = radius * Math.sin(latRad);
    const z = radius * Math.cos(latRad) * Math.sin(lonRad);
    
    return new Vector3(x, y, z);
  });

  return (
    <Line
      points={points}
      color="#00ff66" // Brighter green
      lineWidth={4}
      transparent
      opacity={0.9}
    />
  );
}

function LaunchSiteMarkers({ visibleSites = ['KSC', 'VSFB', 'CCAFS'] }: { visibleSites?: string[] }) {
  return (
    <>
      {LAUNCH_SITES.filter(site => visibleSites.includes(site.code)).map((site) => {
        // Convert lat/lon to 3D position on Earth sphere (radius = 1)
        // Fix mirroring: Flip longitude sign to match US landmass
        const latRad = (site.latitude * Math.PI) / 180;
        const lonRad = (-site.longitude * Math.PI) / 180; // Flip longitude sign
        const radius = 1.02; // Slightly above Earth surface
        
        const x = radius * Math.cos(latRad) * Math.cos(lonRad);
        const y = radius * Math.sin(latRad);
        const z = radius * Math.cos(latRad) * Math.sin(lonRad);
        
        return (
          <group key={site.code}>
            {/* Launch site marker */}
            <mesh position={[x, y, z]}>
              <sphereGeometry args={[0.025, 8, 8]} />
              <meshBasicMaterial color={site.color} />
            </mesh>
            
            {/* Launch site label */}
            <Text
              position={[x * 1.15, y * 1.15, z * 1.15]}
              fontSize={0.05}
              color={site.color}
              anchorX="center"
              anchorY="middle"
            >
              {site.code}
            </Text>
          </group>
        );
      })}
    </>
  );
}

function TrajectoryPath({ trajectoryData }: TrajectoryPathProps) {
  if (!trajectoryData || !trajectoryData.trajectory_points.length) {
    return null;
  }

  const points = trajectoryData.trajectory_points.map(point => {
    // Scale coordinates to match Earth sphere (radius = 1)
    const scaleFactor = 1 / 6371; // Scale km to sphere units
    const x = point.position[0] * scaleFactor;
    const y = point.position[1] * scaleFactor; 
    const z = point.position[2] * scaleFactor;
    
    return new Vector3(x, y, z);
  });

  return (
    <Line
      points={points}
      color="#ffaa00" // Bright orange for trajectory
      lineWidth={6}
      transparent
      opacity={0.95}
    />
  );
}

interface TrajectoryVisualizationProps {
  trajectoryData: LaunchTrajectory | null;
  isSimulating: boolean;
  className?: string;
  visibleSites?: string[];
  onSiteToggle?: (siteCode: string) => void;
}

const TrajectoryVisualization: React.FC<TrajectoryVisualizationProps> = ({
  trajectoryData,
  isSimulating,
  className = "",
  visibleSites = ['KSC', 'VSFB', 'CCAFS'],
  onSiteToggle
}) => {
  return (
    <div className={`h-96 w-full bg-black rounded-lg overflow-hidden relative ${className}`}>
      <div className="absolute top-4 right-4 z-10 flex items-center gap-2 text-white text-sm">
        <div className={`w-2 h-2 rounded-full ${isSimulating ? 'bg-green-400' : 'bg-gray-400'}`} />
        {isSimulating ? 'Real-time Simulation' : 'Static View'}
      </div>
      
      {/* Launch Sites Legend */}
      <div className="absolute top-4 left-4 z-10 bg-black bg-opacity-50 rounded-lg p-3 text-white text-xs">
        <div className="font-semibold mb-2">Launch Sites</div>
        <div className="space-y-1">
          {LAUNCH_SITES.map((site) => (
            <div 
              key={site.code}
              className={`flex items-center gap-2 cursor-pointer hover:bg-gray-700 rounded px-1 py-0.5 transition-colors ${
                onSiteToggle ? 'cursor-pointer' : ''
              }`}
              onClick={() => onSiteToggle?.(site.code)}
            >
              <div 
                className={`w-3 h-3 rounded-full transition-opacity ${
                  visibleSites.includes(site.code) ? 'opacity-100' : 'opacity-30'
                }`}
                style={{ 
                  backgroundColor: site.color.replace('#ff4444', '#ef4444')
                    .replace('#4444ff', '#3b82f6')
                    .replace('#ffff44', '#eab308')
                }}
              />
              <span className={visibleSites.includes(site.code) ? '' : 'opacity-50'}>
                {site.code} - {site.name}
              </span>
              {onSiteToggle && (
                <span className="ml-auto text-gray-400">
                  {visibleSites.includes(site.code) ? '👁' : '👁‍🗨'}
                </span>
              )}
            </div>
          ))}
          <div className="flex items-center gap-2 mt-2 pt-1 border-t border-gray-600">
            <div className="w-3 h-1 bg-green-400"></div>
            <span>US Landmass</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-1 bg-orange-400"></div>
            <span>Launch Trajectory</span>
          </div>
        </div>
      </div>
      
      <Canvas
        camera={{ position: [-0.8, 0.5, 1.8], fov: 60 }}
        style={{ background: '#111827' }} // Dark blue-gray instead of pure black
      >
        <OrbitControls 
          enablePan={true} 
          enableZoom={true} 
          enableRotate={true}
          target={[0, 0.1, 0]} // Center on US latitude
          minDistance={1.5}
          maxDistance={8}
        />
        
        {/* Much brighter lighting for better visibility */}
        <ambientLight intensity={0.6} />
        <directionalLight position={[5, 5, 5]} intensity={1.2} castShadow />
        <pointLight position={[-5, 3, 5]} intensity={0.8} color="#ffffff" />
        <hemisphereLight 
          args={['#87ceeb', '#654321', 0.4]} // Sky blue top, ground brown bottom
        />
        
        <Stars radius={100} depth={50} count={1500} factor={6} saturation={0.1} fade speed={0.5} />
        
        <Earth />
        <USLandmass />
        <LaunchSiteMarkers visibleSites={visibleSites} />
        <TrajectoryPath trajectoryData={trajectoryData} />
        
        {/* Grid plane */}
        <gridHelper args={[4, 20, '#444444', '#222222']} position={[0, -1.5, 0]} />
      </Canvas>
    </div>
  );
};

export default TrajectoryVisualization;