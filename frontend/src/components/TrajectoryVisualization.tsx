import React, { useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Sphere, Line, Stars } from '@react-three/drei';
import { Vector3 } from 'three';
import { LaunchTrajectory } from '../types';

interface EarthProps {
  radius?: number;
}

function Earth({ radius = 1 }: EarthProps) {
  const meshRef = useRef<any>(null);
  
  useFrame(() => {
    if (meshRef.current) {
      meshRef.current.rotation.y += 0.005;
    }
  });

  return (
    <Sphere ref={meshRef} args={[radius, 32, 32]} position={[0, 0, 0]}>
      <meshStandardMaterial
        color="#2e8b57"
        wireframe={true}
        transparent={true}
        opacity={0.8}
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

  const points = trajectoryData.trajectory_points.map(point => 
    new Vector3(
      point.position[0] / 6371, // Scale to Earth radius
      point.position[1] / 6371,
      point.position[2] / 6371
    )
  );

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