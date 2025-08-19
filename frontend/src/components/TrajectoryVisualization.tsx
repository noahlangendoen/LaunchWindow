import React, { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Sphere, Line, Stars } from '@react-three/drei';
import { Vector3, CanvasTexture, MathUtils } from 'three';
import { LaunchTrajectory } from '../types';

interface EarthProps {
  radius?: number;
}

function Earth({ radius = 1 }: EarthProps) {
  const meshRef = useRef<any>(null);
  
  // Create a procedural earth texture with realistic land formations
  const earthTexture = useMemo(() => {
    const canvas = document.createElement('canvas');
    canvas.width = 1024;
    canvas.height = 512;
    const ctx = canvas.getContext('2d')!;
    
    // Fill with water color as base
    ctx.fillStyle = '#172D87';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Set land color
    ctx.fillStyle = '#298717';
    
    // Helper function to draw irregular shapes
    const drawLandmass = (points: [number, number][]) => {
      ctx.beginPath();
      ctx.moveTo(points[0][0], points[0][1]);
      for (let i = 1; i < points.length; i++) {
        ctx.lineTo(points[i][0], points[i][1]);
      }
      ctx.closePath();
      ctx.fill();
    };
    
    // North America - improved shape for launch visualization
    // Canada
    drawLandmass([
      [50, 40], [140, 30], [180, 40], [200, 55], [190, 75], 
      [170, 85], [150, 90], [120, 95], [90, 90], [60, 85], 
      [40, 75], [35, 60], [45, 45], [50, 40]
    ]);
    
    // United States - prominent for launch sites
    drawLandmass([
      [60, 90], [170, 85], [185, 95], [190, 110], [185, 125], 
      [175, 140], [160, 150], [140, 155], [120, 150], [100, 145], 
      [80, 140], [65, 130], [55, 115], [50, 100], [60, 90]
    ]);
    
    // Florida - emphasized for KSC and CCAFS
    drawLandmass([
      [165, 150], [175, 145], [180, 155], [185, 170], [180, 185], 
      [175, 190], [170, 185], [165, 175], [160, 165], [165, 150]
    ]);
    
    // California coast - for VSFB
    drawLandmass([
      [45, 120], [55, 115], [60, 130], [65, 145], [60, 160], 
      [55, 165], [50, 160], [45, 145], [40, 130], [45, 120]
    ]);
    
    // Mexico
    drawLandmass([
      [80, 155], [140, 150], [160, 160], [170, 175], [165, 190], 
      [155, 200], [140, 205], [120, 200], [100, 195], [85, 185], 
      [75, 170], [80, 155]
    ]);
    
    // Greenland
    drawLandmass([
      [200, 30], [220, 25], [235, 35], [230, 50], [210, 55], [200, 45], [200, 30]
    ]);
    
    // South America - distinctive shape
    drawLandmass([
      [140, 180], [160, 170], [180, 175], [190, 200], [185, 230], 
      [175, 260], [165, 280], [150, 290], [135, 285], [125, 270], 
      [120, 250], [115, 220], [125, 200], [140, 180]
    ]);
    
    // Africa - characteristic shape
    drawLandmass([
      [480, 90], [520, 85], [540, 100], [550, 130], [560, 160], 
      [555, 200], [550, 240], [540, 270], [520, 285], [500, 290], 
      [480, 285], [465, 270], [455, 240], [450, 200], [455, 160], 
      [465, 130], [475, 100], [480, 90]
    ]);
    
    // Europe - more detailed
    drawLandmass([
      [460, 60], [490, 50], [520, 55], [535, 70], [530, 85], 
      [515, 90], [490, 95], [470, 90], [455, 80], [450, 65], [460, 60]
    ]);
    
    // Scandinavia
    drawLandmass([
      [480, 30], [500, 25], [510, 35], [515, 50], [505, 60], 
      [490, 55], [485, 45], [480, 30]
    ]);
    
    // Asia - large landmass
    drawLandmass([
      [540, 45], [620, 40], [680, 50], [720, 60], [750, 75], 
      [770, 100], [780, 130], [775, 160], [760, 180], [740, 190], 
      [700, 185], [660, 180], [620, 175], [580, 170], [550, 160], 
      [535, 140], [530, 120], [535, 100], [540, 80], [540, 45]
    ]);
    
    // India subcontinent
    drawLandmass([
      [620, 160], [640, 155], [660, 165], [670, 185], [665, 205], 
      [650, 220], [630, 215], [615, 200], [610, 180], [620, 160]
    ]);
    
    // Southeast Asia and Indonesia
    drawLandmass([
      [680, 190], [720, 185], [740, 195], [730, 210], [710, 215], [690, 210], [680, 200], [680, 190]
    ]);
    
    // Australia - distinctive shape
    drawLandmass([
      [780, 280], [820, 275], [860, 280], [890, 290], [900, 310], 
      [895, 330], [880, 340], [850, 345], [820, 340], [790, 335], 
      [770, 325], [765, 305], [775, 290], [780, 280]
    ]);
    
    // New Zealand
    drawLandmass([
      [920, 320], [930, 315], [935, 325], [930, 335], [920, 330], [920, 320]
    ]);
    drawLandmass([
      [925, 340], [935, 335], [940, 345], [935, 355], [925, 350], [925, 340]
    ]);
    
    // Madagascar
    drawLandmass([
      [565, 270], [575, 265], [580, 280], [575, 295], [565, 290], [560, 275], [565, 270]
    ]);
    
    // Japan
    drawLandmass([
      [800, 120], [815, 115], [820, 130], [815, 145], [800, 140], [795, 125], [800, 120]
    ]);
    
    // British Isles
    drawLandmass([
      [430, 70], [440, 65], [445, 75], [440, 85], [430, 80], [425, 70], [430, 70]
    ]);
    
    // Iceland
    drawLandmass([
      [380, 50], [390, 45], [395, 55], [385, 60], [380, 55], [380, 50]
    ]);
    
    // Add some additional islands in the Pacific
    const pacificIslands = [
      [850, 200], [870, 220], [890, 240], [910, 180], [930, 200]
    ];
    pacificIslands.forEach(([x, y]) => {
      ctx.beginPath();
      ctx.ellipse(x, y, 3, 3, 0, 0, Math.PI * 2);
      ctx.fill();
    });
    
    // Add Caribbean islands
    const caribbeanIslands = [
      [200, 180], [210, 175], [220, 180], [230, 175]
    ];
    caribbeanIslands.forEach(([x, y]) => {
      ctx.beginPath();
      ctx.ellipse(x, y, 2, 2, 0, 0, Math.PI * 2);
      ctx.fill();
    });
    
    return new CanvasTexture(canvas);
  }, []);
  
  // Removed spinning animation for better launch visualization
  // useFrame(() => {
  //   if (meshRef.current) {
  //     meshRef.current.rotation.y += 0.005;
  //   }
  // });

  return (
    <Sphere ref={meshRef} args={[radius, 64, 32]} position={[0, 0, 0]} rotation={[0, -0.3, 0]}>
      <meshStandardMaterial
        map={earthTexture}
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