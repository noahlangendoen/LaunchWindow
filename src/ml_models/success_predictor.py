import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
import joblib
import os
from datetime import datetime
from typing import Dict, Tuple, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns

class LaunchSuccessPredictor:
    """Machine learning model for predicting rocket launch success."""

    def __init__(self, model_path: str="data/models/launch_success_model.pkl")
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.label_encoders = {}
        self.is_trained = False
        self.model_type = 'xgboost' # Default to XGBoost model
        self.training_history = {}

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for training or prediction."""
        features = df.copy()

        # Handle missing values
        numeric_columns = features.select_dtypes(include=[np.number]).columns
        categorical_columns = features.select_dtypes(include=['object', 'category']).columns

        # Fill numeric missing values with median
        for col in numeric_columns:
            if col in features.columns:
                median_value = features[col].median()
                if len(median_value) > 0:
                    features[col] = features[col].fillna(median_value[0])
                else:
                    features[col] = features[col].fillna('Unknown')