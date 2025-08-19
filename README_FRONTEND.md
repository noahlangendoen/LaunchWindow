# Launch Prediction System Frontend

A React-based frontend for the Rocket Launch Prediction System with 3D trajectory visualization and interactive launch window optimization.

## Features

- **3D Trajectory Visualization**: Interactive 3D visualization of launch trajectories using Three.js
- **Launch Window Optimization**: Search and display optimal launch windows across multiple sites
- **Real-time Weather Integration**: Live weather conditions and forecasts
- **ML-based Success Prediction**: Launch success probability analysis
- **Multi-site Support**: Kennedy Space Center, Vandenberg, Cape Canaveral

## Quick Start

### Option 1: Use Batch Files (Windows)
1. Run `start_backend.bat` to start the API server
2. Run `start_frontend.bat` to start the React development server
3. Open http://localhost:3000 in your browser

### Option 2: Manual Setup

#### Backend API Server
```bash
# Install Python dependencies
pip install flask flask-cors

# Start the API server
python src/api/app.py
```
The API will be available at http://localhost:8000

#### Frontend React App
```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```
The frontend will be available at http://localhost:3000

## Architecture

### Components
- **MainDashboard**: Main interface with launch site selection and predictions
- **TrajectoryVisualization**: 3D trajectory visualization with Three.js
- **TrajectoryPage**: Dedicated page for trajectory calculations
- **LaunchWindowsPage**: Search and display optimal launch windows
- **WeatherPanel**: Real-time weather conditions display

### API Integration
The frontend communicates with the Python backend through REST APIs:
- `/prediction` - Run ML-based launch success predictions
- `/trajectory` - Calculate physics-based trajectories
- `/windows` - Find optimal launch windows
- `/weather/current/<site>` - Get current weather data
- `/weather/forecast/<site>` - Get weather forecasts

### Technologies
- **React 19** with TypeScript
- **Three.js** via @react-three/fiber for 3D visualization
- **Vite** for fast development and building
- **Tailwind CSS** for styling
- **Axios** for API communication

## Development

### Project Structure
```
frontend/
├── src/
│   ├── components/          # Reusable UI components
│   ├── pages/              # Main page components
│   ├── services/           # API service layer
│   ├── types/              # TypeScript type definitions
│   └── App.tsx             # Main application component
├── public/                 # Static assets
└── package.json            # Dependencies and scripts
```

### Available Scripts
- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build

### Environment Variables
Set up your `.env` file in the project root:
```
OPENWEATHER_API_KEY=your_api_key_here
```

## Features in Detail

### 3D Trajectory Visualization
- Interactive 3D Earth model with wireframe
- Real-time trajectory path rendering
- Orbital mechanics visualization
- Mission parameter controls (altitude, inclination)
- Performance metrics display

### Launch Window Optimization
- Multi-site search capabilities
- Weather-based scoring system
- Success probability calculations
- GO/NO-GO status indicators
- Sortable results table

### Weather Integration
- Real-time weather conditions
- 5-day forecasts
- Launch constraint assessments
- Visual status indicators
- Automatic site monitoring

## Browser Support

This application works best in modern browsers with WebGL support:
- Chrome 80+
- Firefox 75+
- Safari 13+
- Edge 80+

## Troubleshooting

### Common Issues
1. **3D visualization not loading**: Check WebGL support in your browser
2. **API connection errors**: Ensure the backend server is running on port 8000
3. **Weather data unavailable**: Check your OpenWeatherMap API key configuration

### Development Tips
- Use browser dev tools to monitor API requests
- Check the console for any JavaScript errors
- Ensure both frontend and backend servers are running simultaneously