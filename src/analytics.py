"""Biomechanical analytics and calculations."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

class BiomechanicsAnalytics:
    """Biomechanical analysis and metrics calculation."""
    
    @staticmethod
    def calculate_pitcher_stats(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate aggregate pitcher statistics."""
        stats = df.groupby('pitcher_name').agg({
            'release_speed': ['mean', 'std', 'min', 'max'],
            'release_spin_rate': ['mean', 'std'],
            'release_extension': 'mean',
            'movement_total': 'mean',
            'spin_efficiency': 'mean',
            'pfx_x': 'mean',
            'pfx_z': 'mean'
        }).round(2)
        
        return stats
    
    @staticmethod
    def calculate_movement_profile(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate movement profile by pitch type."""
        movement = df.groupby(['pitcher_name', 'pitch_type']).agg({
            'pfx_x': 'mean',
            'pfx_z': 'mean',
            'release_speed': 'mean',
            'release_spin_rate': 'mean'
        }).round(2)
        
        return movement
    
    @staticmethod
    def cluster_pitchers(df: pd.DataFrame, n_clusters: int = 3) -> dict:
        """Cluster pitchers using K-Means."""
        # Aggregate by pitcher
        pitcher_features = df.groupby('pitcher_name').agg({
            'release_speed': 'mean',
            'release_spin_rate': 'mean',
            'release_extension': 'mean',
            'movement_total': 'mean',
            'spin_efficiency': 'mean'
        }).reset_index()
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(pitcher_features.iloc[:, 1:])
        
        # K-Means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(features_scaled)
        
        pitcher_features['cluster'] = clusters
        
        return {
            'pitcher_features': pitcher_features,
            'kmeans': kmeans,
            'scaler': scaler
        }
    
    @staticmethod
    def calculate_biomechanical_score(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate composite biomechanical score."""
        # Normalize metrics
        scaler = StandardScaler()
        metrics = df[[
            'release_speed',
            'release_spin_rate',
            'release_extension',
            'movement_total',
            'spin_efficiency'
        ]].copy()
        
        metrics_scaled = scaler.fit_transform(metrics)
        
        # Weighted score
        weights = np.array([0.25, 0.25, 0.15, 0.20, 0.15])
        score = np.dot(metrics_scaled, weights)
        
        df['biomechanical_score'] = score
        return df
    
    @staticmethod
    def compare_pitchers(df: pd.DataFrame, pitcher1: str, pitcher2: str) -> dict:
        """Compare two pitchers across metrics."""
        p1_data = df[df['pitcher_name'] == pitcher1]
        p2_data = df[df['pitcher_name'] == pitcher2]
        
        if len(p1_data) == 0 or len(p2_data) == 0:
            return {"error": "One or both pitchers not found"}
        
        comparison = {
            pitcher1: {
                'avg_velocity': p1_data['release_speed'].mean(),
                'avg_spin_rate': p1_data['release_spin_rate'].mean(),
                'avg_extension': p1_data['release_extension'].mean(),
                'avg_movement': p1_data['movement_total'].mean()
            },
            pitcher2: {
                'avg_velocity': p2_data['release_speed'].mean(),
                'avg_spin_rate': p2_data['release_spin_rate'].mean(),
                'avg_extension': p2_data['release_extension'].mean(),
                'avg_movement': p2_data['movement_total'].mean()
            }
        }
        
        return comparison
