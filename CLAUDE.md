# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Backend (Python)
```bash
# Run the complete backend demonstration
python src/master_demo.py

# Start the Flask API server
python src/api/app.py
```

### Frontend (React + TypeScript + Vite)
```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

### Full Stack Development
```bash
# Terminal 1: Start backend API server
python src/api/app.py

# Terminal 2: Start frontend development server  
cd frontend && npm run dev
```

## Architecture Overview

This is a full-stack rocket launch prediction system with 4 core objectives implemented as a comprehensive web application. The system integrates machine learning, physics-based trajectory calculations, weather data, and real-time monitoring with a modern React frontend and Flask API backend.

### Backend Architecture (Python)

#### Environment & Dependencies
- Python version: 3.13.3
- Dependencies: pandas, numpy, scikit-learn, xgboost, requests, matplotlib, seaborn, joblib, dotenv, flask, flask-cors
- Virtual environment located in `venv/` directory

#### Core Components

##### 1. Data Ingestion (`src/data_ingestion/`)
- `collect_weather.py`: OpenWeatherMap API integration for current weather and forecasts
- `collect_spacex.py`: SpaceX launch data collection for training ML models  
- `collect_orbital.py`: Orbital mechanics data collection
- `collect_nasa_launch.py`: NASA launch data integration

##### 2. Machine Learning Models (`src/ml_models/`)
- `success_predictor.py`: Launch success prediction using RandomForest, GradientBoosting, and XGBoost
- `data_preprocessor.py`: Feature engineering and data preprocessing for ML models

##### 3. Physics Engine (`src/physics_engine/`)
- `trajectory_calc.py`: Physics-based trajectory calculations with atmospheric modeling
- `orbital_mechanics.py`: Orbital mechanics computations
- `atmospheric_models.py`: Atmospheric conditions and weather constraint assessments

##### 4. API Layer (`src/api/`)
- `app.py`: Flask REST API server with CORS support
- Endpoints: `/health`, `/prediction`, `/weather/*`, `/trajectory`, `/windows`, `/status`

##### 5. Demo System (`src/master_demo.py`)
The `LaunchPredictionDemo` class orchestrates all backend components

### Frontend Architecture (React + TypeScript)

#### Technology Stack
- **Framework**: React 19.1.1 with TypeScript 5.9.2
- **Build Tool**: Vite 7.1.3
- **Styling**: Tailwind CSS 3.4.17
- **3D Graphics**: Three.js 0.179.1 with React Three Fiber 9.3.0
- **HTTP Client**: Axios 1.11.0
- **Routing**: React Router DOM 7.8.1

#### Frontend Structure (`frontend/src/`)

##### Components (`components/`)
- `LaunchSiteSelector.tsx`: Interactive launch site selection UI
- `LaunchWindowsTable.tsx`: Tabular display of optimal launch windows
- `PredictionResults.tsx`: ML prediction results and risk assessment display
- `TrajectoryVisualization.tsx`: 3D trajectory visualization using Three.js
- `WeatherPanel.tsx`: Real-time weather data display and monitoring

##### Pages (`pages/`)
- `MainDashboard.tsx`: Primary dashboard with system overview
- `LaunchWindowsPage.tsx`: Dedicated page for launch window optimization
- `TrajectoryPage.tsx`: Dedicated page for trajectory analysis and visualization

##### Services (`services/`)
- `api.ts`: Centralized API client for backend communication

##### Types (`types/`)
- `index.ts`: TypeScript type definitions for API responses and data structures

### System Objectives & Features

#### 1. ML-Based Launch Success Prediction
- **Backend**: RandomForest, GradientBoosting, and XGBoost models
- **Frontend**: Interactive prediction interface with risk visualization
- **Integration**: Real-time prediction updates based on current weather conditions

#### 2. Physics-Based Trajectory Optimization
- **Backend**: Atmospheric modeling and trajectory calculations
- **Frontend**: 3D trajectory visualization with Three.js
- **Integration**: Real-time trajectory updates with weather data integration

#### 3. Launch Window Optimization
- **Backend**: Multi-site optimization across time windows
- **Frontend**: Interactive calendar and tabular window displays
- **Integration**: Weather forecast integration for optimal timing

#### 4. Real-Time Monitoring Dashboard
- **Backend**: Live weather data and system health monitoring
- **Frontend**: Real-time dashboard with status indicators and alerts
- **Integration**: WebSocket-ready architecture for live updates

### Launch Sites Configuration
The system supports three main launch sites:
- **KSC**: Kennedy Space Center (28.5° inclination missions)
- **VSFB**: Vandenberg Space Force Base (97.5° polar missions)  
- **CCAFS**: Cape Canaveral Space Force Station (28.5° equatorial missions)

### Data Flow Architecture
1. **Data Collection**: Weather data from OpenWeatherMap API, SpaceX historical data
2. **ML Processing**: Feature engineering and model training/prediction
3. **Physics Calculations**: Trajectory optimization with atmospheric constraints
4. **API Layer**: Flask REST API serves processed data to frontend
5. **Frontend Visualization**: React components render interactive dashboards
6. **Real-time Updates**: Live weather and system status monitoring

### API Endpoints
- `GET /health` - System health check
- `POST /prediction` - Launch success prediction for specific site
- `GET /weather/current/<site_code>` - Current weather data
- `GET /weather/forecast/<site_code>` - Weather forecast data
- `POST /trajectory` - Calculate launch trajectory
- `POST /windows` - Find optimal launch windows
- `GET /status` - System status and health metrics

### Configuration & Environment
- **Weather API**: Requires `OPENWEATHER_API_KEY` environment variable
- **Data Storage**: 
  - Raw data: `data/raw/`
  - Processed data: `data/processed/`
  - ML models: `data/models/launch_success_model.pkl`
- **Frontend Build**: Production builds to `frontend/dist/`
- **API Server**: Runs on `localhost:8000`
- **Frontend Dev Server**: Runs on `localhost:3000` (Vite default)

### Development Workflow

#### Frontend Development
1. Start backend API server: `python src/api/app.py`
2. Start frontend dev server: `cd frontend && npm run dev`
3. Frontend connects to API at `http://localhost:8000`
4. Hot module replacement enabled for rapid development

#### API Development
1. Modify Flask routes in `src/api/app.py`
2. Test endpoints using frontend or API tools
3. CORS enabled for cross-origin requests from frontend

#### Component Development
- Follow React functional component patterns with hooks
- Use TypeScript for type safety
- Utilize Tailwind CSS for styling consistency
- Implement responsive design principles

### Data Quality & Monitoring
- Comprehensive data quality tracking in `preprocessing_metadata.json`
- System health monitoring via `/status` endpoint
- Real-time error handling and user feedback
- Performance monitoring for 3D visualizations

### Testing & Quality Assurance
- Backend: Unit tests for ML models and physics calculations
- Frontend: Component testing and integration testing
- API: Endpoint testing and data validation
- Cross-browser compatibility for frontend components

### Deployment Considerations
- Frontend: Static build deployment (`npm run build`)
- Backend: Flask production server configuration
- Environment variables for API keys and configuration
- CORS configuration for production domains