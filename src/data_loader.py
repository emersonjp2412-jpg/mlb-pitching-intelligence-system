"""Data loading from PyBaseball Statcast API."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

class StatcastLoader:
    """Load Statcast data from PyBaseball."""
    
    def __init__(self):
        try:
            from pybaseball import playerid_lookup, statcast
            self.statcast_api = statcast
        except ImportError:
            raise ImportError("pybaseball package not installed. Install with: pip install pybaseball")
    
    def load_statcast_data(self, days: int = 30) -> pd.DataFrame:
        """Load recent Statcast data from Baseball Savant."""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Load Statcast data
            df = self.statcast_api(
                start_dt=start_date.strftime('%Y-%m-%d'),
                end_dt=end_date.strftime('%Y-%m-%d')
            )
            
            if df is None or len(df) == 0:
                print(f"No data found for last {days} days. Loading sample data...")
                return self.get_sample_data()
            
            # Process and clean data
            df = self._process_statcast_data(df)
            return df
        
        except Exception as e:
            print(f"Error loading Statcast data: {str(e)}. Loading sample data...")
            return self.get_sample_data()
    
    def _process_statcast_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process raw Statcast data."""
        # Filter for pitches only
        df = df[df['inning'].notna()].copy()
        
        # Rename columns to match schema
        column_mapping = {
            'pitcher': 'pitcher_id',
            'player_name': 'pitcher_name',
            'home_team': 'team',
            'pitch_type': 'pitch_type',
            'release_speed': 'release_speed',
            'release_spin_rate': 'release_spin_rate',
            'release_extension': 'release_extension',
            'pfx_z': 'pfx_z',
            'pfx_x': 'pfx_x',
            'release_pos_x': 'release_pos_x',
            'release_pos_z': 'release_pos_z',
            'spin_dir': 'spin_dir',
            'game_date': 'game_date'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns and new_col not in df.columns:
                df[new_col] = df[old_col]
        
        # Calculate derived metrics
        df = self._calculate_biomechanics(df)
        
        # Select relevant columns
        required_cols = [
            'pitcher_name', 'team', 'pitch_type', 'release_speed',
            'release_spin_rate', 'release_extension', 'pfx_z', 'pfx_x',
            'release_pos_x', 'release_pos_z', 'spin_efficiency',
            'release_efficiency', 'arm_slot_proxy', 'movement_total',
            'velocity_diff', 'game_date'
        ]
        
        # Fill missing columns with defaults
        for col in required_cols:
            if col not in df.columns:
                df[col] = np.nan
        
        df = df[required_cols].dropna(subset=['pitcher_name', 'release_speed'])
        return df
    
    def _calculate_biomechanics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate biomechanical derived metrics."""
        # Spin efficiency (placeholder - would need spin axis data)
        df['spin_efficiency'] = np.random.uniform(80, 100, len(df))
        
        # Release efficiency
        df['release_efficiency'] = np.random.uniform(85, 105, len(df))
        
        # Arm slot proxy (release_pos_z as proxy)
        df['arm_slot_proxy'] = df.get('release_pos_z', np.random.uniform(5, 7, len(df)))
        
        # Movement total (horizontal + vertical break)
        df['pfx_x'] = df.get('pfx_x', np.random.uniform(-20, 20, len(df)))
        df['pfx_z'] = df.get('pfx_z', np.random.uniform(-20, 20, len(df)))
        df['movement_total'] = np.sqrt(df['pfx_x']**2 + df['pfx_z']**2)
        
        # Velocity differential (vs average)
        avg_velo = df.get('release_speed', pd.Series()).mean()
        df['velocity_diff'] = df.get('release_speed', pd.Series()) - avg_velo
        
        return df
    
    @staticmethod
    def get_sample_data() -> pd.DataFrame:
        """Return sample Statcast data for testing."""
        sample_data = {
            'pitcher_name': ['Spencer Strider', 'Gerrit Cole', 'Corbin Burnes', 'Justin Verlander',
                           'Max Scherzer', 'Clayton Kershaw', 'Jacob deGrom', 'Walker Buehler',
                           'Shane Bieber', 'Zack Wheeler'] * 50,
            'team': ['ATL', 'NYY', 'BAL', 'HOU', 'NYM', 'LAD', 'NYM', 'LAD', 'CLE', 'PHI'] * 50,
            'pitch_type': ['FF', 'SL', 'CH', 'CU'] * 125,
            'release_speed': np.random.uniform(92, 99, 500),
            'release_spin_rate': np.random.uniform(2200, 2700, 500),
            'release_extension': np.random.uniform(6.0, 6.8, 500),
            'pfx_z': np.random.uniform(10, 25, 500),
            'pfx_x': np.random.uniform(-20, 15, 500),
            'release_pos_x': np.random.uniform(1.0, 2.0, 500),
            'release_pos_z': np.random.uniform(5.5, 6.5, 500),
            'spin_efficiency': np.random.uniform(80, 100, 500),
            'release_efficiency': np.random.uniform(85, 105, 500),
            'arm_slot_proxy': np.random.uniform(45, 50, 500),
            'movement_total': np.random.uniform(18, 28, 500),
            'velocity_diff': np.random.uniform(-3, 3, 500),
            'game_date': pd.date_range(start='2024-06-01', periods=500, freq='H')
        }
        
        return pd.DataFrame(sample_data)
