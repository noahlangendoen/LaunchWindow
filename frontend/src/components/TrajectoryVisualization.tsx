import React, { useRef } from 'react';
import { Canvas} from '@react-three/fiber';
import { OrbitControls, Sphere, Line, Stars } from '@react-three/drei';
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
        color="#4444aa"
        transparent={false}
        roughness={0.8}
        metalness={0.1}
      />
    </Sphere>
  );
}

interface TrajectoryPathProps {
  trajectoryData: LaunchTrajectory | null;
}

function TrajectoryPath({ trajectoryData }: TrajectoryPathProps) {
  if (!trajectoryData || !trajectoryData.trajectory_points.length) {
    return null;
  }

  const points = trajectoryData.trajectory_points.map(point => {
    // Scale to Earth radius and add small offset to keep trajectory above surface
    const scaleFactor = 1 / 6371;
    const x = point.position[0] * scaleFactor;
    const y = point.position[1] * scaleFactor;
    const z = point.position[2] * scaleFactor;
    
    // Calculate distance from center and ensure minimum distance above Earth surface
    const distance = Math.sqrt(x * x + y * y + z * z);
    const minDistance = 1.05; // 5% above Earth surface (radius = 1)
    
    if (distance < minDistance) {
      const factor = minDistance / distance;
      return new Vector3(x * factor, y * factor, z * factor);
    }
    
    return new Vector3(x, y, z);
  });

  return (
    <Line
      points={points}
      color="#00ff88"
      lineWidth={3}
    />
  );
}

interface TrajectoryVisualizationProps {
  trajectoryData: LaunchTrajectory | null;
  isSimulating: boolean;
  className?: string;
}

const TrajectoryVisualization: React.FC<TrajectoryVisualizationProps> = ({
  trajectoryData,
  isSimulating,
  className = ""
}) => {
  return (
    <div className={`h-96 w-full bg-black rounded-lg overflow-hidden relative ${className}`}>
      <div className="absolute top-4 right-4 z-10 flex items-center gap-2 text-white text-sm">
        <div className={`w-2 h-2 rounded-full ${isSimulating ? 'bg-green-400' : 'bg-gray-400'}`} />
        {isSimulating ? 'Real-time Simulation' : 'Static View'}
      </div>
      
      <Canvas
        camera={{ position: [3, 3, 3], fov: 75 }}
        style={{ background: '#000000' }}
      >
        <OrbitControls enablePan={true} enableZoom={true} enableRotate={true} />
        
        <ambientLight intensity={0.3} />
        <pointLight position={[10, 10, 10]} intensity={0.8} />
        
        <Stars radius={100} depth={50} count={1000} factor={4} saturation={0} fade />
        
        <Earth />
        <TrajectoryPath trajectoryData={trajectoryData} />
        
        {/* Grid plane */}
        <gridHelper args={[4, 20, '#444444', '#222222']} position={[0, -1.5, 0]} />
      </Canvas>
    </div>
  );
};

export default TrajectoryVisualization;