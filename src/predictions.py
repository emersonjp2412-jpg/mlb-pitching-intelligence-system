"""Machine learning predictions for pitch characteristics."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings

warnings.filterwarnings('ignore')

class PitchPredictionModel:
    """XGBoost-style pitch prediction models."""
    
    def __init__(self):
        self.velocity_model = None
        self.spin_model = None
        self.feature_names = None
    
    def train_velocity_model(self, df: pd.DataFrame) -> dict:
        """Train velocity prediction model."""
        # Prepare data
        X = df[[
            'release_spin_rate',
            'release_extension',
            'arm_slot_proxy',
            'movement_total'
        ]].fillna(df.mean())
        
        y = df['release_speed'].fillna(df['release_speed'].mean())
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        
        metrics = {
            'r2': r2_score(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'feature_importance': model.feature_importances_
        }
        
        self.velocity_model = model
        return metrics
    
    def train_spin_model(self, df: pd.DataFrame) -> dict:
        """Train spin rate prediction model."""
        # Prepare data
        X = df[[
            'release_speed',
            'release_extension',
            'release_pos_x',
            'release_pos_z'
        ]].fillna(df.mean())
        
        y = df['release_spin_rate'].fillna(df['release_spin_rate'].mean())
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        
        metrics = {
            'r2': r2_score(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'feature_importance': model.feature_importances_
        }
        
        self.spin_model = model
        return metrics
    
    def predict_velocity(self, features: np.ndarray) -> float:
        """Predict pitch velocity."""
        if self.velocity_model is None:
            raise ValueError("Velocity model not trained yet")
        return self.velocity_model.predict(features.reshape(1, -1))[0]
    
    def predict_spin_rate(self, features: np.ndarray) -> float:
        """Predict spin rate."""
        if self.spin_model is None:
            raise ValueError("Spin model not trained yet")
        return self.spin_model.predict(features.reshape(1, -1))[0]
