# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Running the Application
```bash
python src/master_demo.py
```
This runs the complete demonstration showing all 4 core objectives of the launch prediction system.

### Python Environment
- Python version: 3.13.3
- Dependencies are managed via `requirements.txt` (currently appears empty but project uses pandas, numpy, scikit-learn, xgboost, requests, matplotlib, seaborn, joblib, dotenv)
- Virtual environment is located in `venv/` directory

## Architecture Overview

This is a rocket launch prediction system with 4 core objectives implemented as a comprehensive demo application. The system integrates machine learning, physics-based trajectory calculations, weather data, and real-time monitoring.

### Core Components

#### 1. Data Ingestion (`src/data_ingestion/`)
- `collect_weather.py`: OpenWeatherMap API integration for current weather and forecasts
- `collect_spacex.py`: SpaceX launch data collection for training ML models  
- `collect_orbital.py`: Orbital mechanics data collection
- `collect_nasa_launch.py`: NASA launch data integration

#### 2. Machine Learning Models (`src/ml_models/`)
- `success_predictor.py`: Launch success prediction using RandomForest, GradientBoosting, and XGBoost
- `data_preprocessor.py`: Feature engineering and data preprocessing for ML models

#### 3. Physics Engine (`src/physics_engine/`)
- `trajectory_calc.py`: Physics-based trajectory calculations with atmospheric modeling
- `orbital_mechanics.py`: Orbital mechanics computations
- `atmospheric_models.py`: Atmospheric conditions and weather constraint assessments

#### 4. Main Demo (`src/master_demo.py`)
The `LaunchPredictionDemo` class orchestrates all components and demonstrates:
- **Objective 1**: ML-based launch success prediction
- **Objective 2**: Physics-based trajectory optimization  
- **Objective 3**: Launch window optimization across multiple sites
- **Objective 4**: Real-time monitoring and alerting dashboard

### Launch Sites
The system supports three main launch sites:
- **KSC**: Kennedy Space Center (28.5° inclination missions)
- **VSFB**: Vandenberg Space Force Base (97.5° polar missions)
- **CCAFS**: Cape Canaveral Space Force Station (28.5° equatorial missions)

### Data Flow
1. Weather data collected from OpenWeatherMap API for all launch sites
2. Historical SpaceX launch data used to train ML success prediction models
3. Physics engine calculates optimal trajectories considering atmospheric conditions
4. Launch window optimization finds best times across multiple days
5. Real-time monitoring provides alerts and status updates

### Key Classes and Data Structures
- `LaunchPredictionDemo`: Main orchestrator class
- `VehicleSpecs`: Rocket specifications (Falcon 9 is the demo vehicle)
- `TrajectoryPoint`: Individual points in calculated trajectories  
- `LaunchTrajectory`: Complete trajectory with success metrics
- `WeatherCollector`: API wrapper for weather data
- `LaunchSuccessPredictor`: ML model wrapper with multiple algorithm support

### Configuration
- Weather API requires `OPENWEATHER_API_KEY` environment variable
- Processed data stored in `data/processed/` with quality metadata
- ML models saved to `data/models/launch_success_model.pkl`

### Data Quality Monitoring
The system includes comprehensive data quality tracking with metrics stored in `preprocessing_metadata.json`, monitoring completeness, consistency, and overall data quality scores.