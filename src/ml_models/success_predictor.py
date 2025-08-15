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
from .data_preprocessor import DataPreprocessor


class LaunchSuccessPredictor:
    """Machine learning model for predicting rocket launch success."""

    def __init__(self, model_path: str="data/models/launch_success_model.pkl"):
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
                features[col] = features[col].fillna(median_value)

        # Fill categorical missing values with mode or 'unkown'
        for col in categorical_columns:
            if col in features.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    unique_values = list(features[col].unique()) + ['Unknown']
                    self.label_encoders[col].fit(unique_values)
                
                features[col] = self.label_encoders[col].transform(features[col])
        
        return features
    
    def prepare_target(self, y: pd.Series) -> pd.Series:
        """Clean and prepare target variable for classification."""
        y_clean = y.copy()
        
        # Convert various formats to boolean
        y_clean = y_clean.map({
            True: 1, False: 0, 'true': 1, 'false': 0,
            'True': 1, 'False': 0, 1: 1, 0: 0, '1': 1, '0': 0,
            'SUCCESS': 1, 'FAILURE': 0, 'Success': 1, 'Failure': 0,
            'success': 1, 'failure': 0, 'SUCCESSFUL': 1, 'FAILED': 0,
            None: 0, np.nan: 0
        })
        
        # Fill any remaining NaN values with 0 (failure)
        y_clean = y_clean.fillna(0)
        
        # Ensure all values are integers (0 or 1)
        y_clean = y_clean.astype(int)
        
        return y_clean
    
    def train_model(self, X: pd.DataFrame, y: pd.Series, model_type: str='xgboost') -> Dict:
        """Train the launch success prediction model."""
        self.model_type = model_type

        # Clean target variable first
        y_clean = self.prepare_target(y)

        # Check if we have both classes
        unique_classes = y_clean.unique()
        print(f"Target classes found: {unique_classes}.")

        if len(unique_classes) < 2:
            raise ValueError("Need at least 2 difference classes.")
        
        # Prepare features
        X_processed = self.prepare_features(X)

        # Store feature columns for later use
        self.feature_columns = X_processed.columns.tolist()

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y_clean, test_size=0.2, random_state=42, stratify=y
        )

        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model based on type
        if model_type == 'xgboost':
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss'
            )
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}. Consider implementing it.")
        
        # Train the model
        self.model.fit(X_train, y_train)

        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]

        # Calcualte metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }

        # Cross validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5, scoring='accuracy')
        metrics['cv_mean'] = cv_scores.mean()
        metrics['cv_std'] = cv_scores.std()

        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            metrics['feature_importance'] = feature_importance.to_dict('records')
        
        self.training_history = metrics
        self.is_trained = True

        print("Model has trained successfully.")
        print(f"Accuracy: {metrics['accuracy']}")
        print(f"F1-score: {metrics['f1_score']}")
        print(f"Cross-validation: {metrics['cv_mean']:.3f}")

        return metrics
    
    def predict_launch_success(self, features: pd.DataFrame) -> Dict:
        """Predict launch success probability for given features."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making prediction.")
        
        # Prepare features
        X_processed = self.prepare_features(features)

        # Ensure all requried columns are present
        for col in self.feature_columns:
            if col not in X_processed.columns:
                X_processed[col] = 0

        # Reorder columns to match training data
        X_processed = X_processed[self.feature_columns]

        # Scale features
        X_scaled = self.scaler.transform(X_processed)

        # Make predictions
        success_probability = self.model.predict_proba(X_scaled)[:, 1]
        binary_prediction = self.model.predict(X_scaled)

        results = []
        for i in range(len(features)):
            result = {
                'success_probability': float(success_probability[i]),
                'binary_prediction': bool(binary_prediction[i]),
                'confidence_level': self._calculate_confidence(success_probability[i]),
                'risk_assessment': self._assess_risk(success_probability[i]),
                'recommendation': self._generate_recommendation(success_probability[i])
            }
        
            results.append(result)
        
        return results if len(results) > 1 else results[0]
    
    def _calculate_confidence(self, probability: float) -> str:
        """Calculate confidence level based on probability."""
        if probability > 0.8 or probability < 0.2:
            return 'HIGH'
        if probability > 0.6 or probability < 0.4:
            return 'MEDIUM'
        else:
            return 'LOW'
        
    def _assess_risk(self, probability: float) -> str:
        """Assess risk based on success probability."""
        if probability > 0.8:
            return 'LOW'
        elif probability > 0.6:
            return 'MEDIUM'
        elif probability > 0.4:
            return 'HIGH'
        else:
            return 'VERY_HIGH'
        
    def _generate_recommendation(self, probability: float) -> str:
        """Generate a recommendation based on the input probability."""
        if probability > 0.8:
            return 'GO for launch - high success probability.'
        elif probability > 0.6:
            return 'CAUTION - monitor conditions closely.'
        elif probability > 0.4:
            return 'CONSIDER DELAY - marginal conditions.'
        else:
            return 'NO GO - high risk of failure.'
        
    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Optimize model hyperparameters using grid search."""
        X_processed = self.prepare_features(X)
        X_scaled = self.scaler.fit_transform(X_processed) if self.scaler else StandardScaler().fit_transform(X_processed)

        if self.model_type == 'xgboost':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15],
                'learning_rate': [0.01, 0.1, 0.2]
            }
            model = xgb.XGBClassifier(random_state=42)
        elif self.model_type == 'random_forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15],
                'min_samples_split': [2, 5, 10]
            }
            model = RandomForestClassifier(random_state=42)

        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1
        )

        grid_search.fit(X_scaled, y)

        self.model = grid_search.best_estimator_

        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
    
    def save_model(self, filepath: str = None):
        """Save the trained model and preprocessing components."""
        if not self.is_trained:
            raise ValueError("No trained model to save.")
        
        filepath = filepath or self.model_path
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'label_encoders': self.label_encoders,
            'model_type': self.model_type,
            'training_history': self.training_history,
            'is_trained': self.is_trained
        }

        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}.")

    def load_model(self, filepath: str = None):
        """Load a trained model and preprocessing components."""
        filepath = filepath or self.model_path

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model fil not found: {filepath}.")
        
        model_data = joblib.load(filepath)

        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.label_encoders = model_data['label_encoders']
        self.model_type = model_data['model_type']
        self.training_history = model_data['training_history']
        self.is_trained = model_data['is_trained']

        print(f"Model loaded from {filepath}.")

    def generate_feature_importance_plot(self, save_path: str = None):
        """Generate and optionally save feature importance plot."""
        if not self.is_trained or 'feature_importance' not in self.training_history:
            raise ValueError("Model must be trained to generate feature importance plot.")
        
        importance_df = pd.DataFrame(self.training_history['feature_importance'])

        plt.figure(figsize=(10, 8))
        sns.barplot(data=importance_df.head(15), x='importance', y='feature')
        plt.title('Top 15 feature importances for launch success prediction.')
        plt.xlabel('Importance Score')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to {save_path}.")

        plt.show()

    def evaluate_model_performance(self) -> Dict:
        """Return detailed model performance metrics."""
        if not self.is_trained:
            raise ValueError("Model must be trained to evaluate performance.")
        
        return self.training_history
    

def main():
    """Example usage of the LaunchSuccessPredictor."""

    # Intialize preprocessor and predictor
    preprocessor = DataPreprocessor()
    predictor = LaunchSuccessPredictor()

    # Load and prepare data
    datasets = preprocessor.load_all_data()
    if datasets:
        X, y = preprocessor.create_training_dataset(target_column='success')
        if X is not None and len(X) > 0:
            # Train the model
            metrics = predictor.train_model(X, y, model_type='random_forest')

            # Save model
            predictor.save_model()

            # Generate feature importance plot
            try:
                predictor.generate_feature_importance_plot('data/models/feature_importance.png')
            except:
                print("Could not generate feature importance plot.")
            
            if len(X) > 0:
                sample_features = X.iloc[:1]
                prediction = predictor.predict_launch_success(sample_features)
                print(f"Example Prediction: {prediction}")
        else:
            print("No training data available.")


if __name__ == "__main__":
    main()