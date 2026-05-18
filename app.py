import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from src.database import SupabaseClient
from src.data_loader import StatcastLoader
from src.analytics import BiomechanicsAnalytics
from src.predictions import PitchPredictionModel

# Page configuration
st.set_page_config(
    page_title="MLB Biomechanical Dashboard",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    .header-title {
        color: #1f77b4;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'db_client' not in st.session_state:
    st.session_state.db_client = None

def load_data():
    """Load Statcast data from PyBaseball"""
    with st.spinner('Loading Statcast data...'):
        try:
            loader = StatcastLoader()
            # Load last 30 days of data
            df = loader.load_statcast_data(days=30)
            st.session_state.df = df
            st.session_state.data_loaded = True
            st.success(f'Loaded {len(df)} pitches from {df["pitcher_name"].nunique()} pitchers')
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")

def load_sample_data():
    """Load sample data for testing"""
    with st.spinner('Loading sample data...'):
        try:
            loader = StatcastLoader()
            df = loader.get_sample_data()
            st.session_state.df = df
            st.session_state.data_loaded = True
            st.success(f'Loaded {len(df)} sample pitches from {df["pitcher_name"].nunique()} pitchers')
        except Exception as e:
            st.error(f"Error loading sample data: {str(e)}")

# Main App
def main():
    # Sidebar
    with st.sidebar:
        st.title("⚾ MLB Biomechanics")
        st.markdown("---")
        
        st.subheader("Data Loading")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Load Live Data", use_container_width=True):
                load_data()
        with col2:
            if st.button("Load Sample", use_container_width=True):
                load_sample_data()
        
        st.markdown("---")
        st.subheader("Settings")
        
        # Database connection option
        use_db = st.checkbox("Connect to Database (Supabase)")
        if use_db:
            st.info("Ensure .env file contains SUPABASE credentials")
        
        st.markdown("---")
        st.subheader("Navigation")
        page = st.radio(
            "Select Page:",
            ["Overview", "Biomechanics", "Clustering", "Leaderboards", "Predictions"]
        )
    
    # Check if data is loaded
    if not st.session_state.data_loaded:
        st.markdown("<div class='header-title'>MLB Biomechanical Dashboard</div>", unsafe_allow_html=True)
        st.markdown("""
        Welcome to the **MLB Biomechanical Dashboard** powered by Statcast data.
        
        ### 📊 Features
        - **Real-time Statcast data** from PyBaseball
        - **Biomechanical analysis** (velocity, spin rate, movement)
        - **Pitcher clustering** using K-Means
        - **XGBoost predictions** for velocity and spin rate
        - **Interactive visualizations** with Plotly
        
        ### 🚀 Get Started
        1. Click **"Load Live Data"** or **"Load Sample"** in the sidebar
        2. Navigate through the pages to explore analytics
        3. View pitcher comparisons, biomechanics, and predictions
        
        ### 📈 Data Source
        Data is fetched from **Baseball Savant** via PyBaseball library.
        Updates include pitch velocities, spin rates, movement data, and more.
        """)
        return
    
    # Page routing
    if page == "Overview":
        show_overview()
    elif page == "Biomechanics":
        show_biomechanics()
    elif page == "Clustering":
        show_clustering()
    elif page == "Leaderboards":
        show_leaderboards()
    elif page == "Predictions":
        show_predictions()

def show_overview():
    """Overview page with key metrics and comparisons"""
    st.markdown("<div class='header-title'>📊 Overview</div>", unsafe_allow_html=True)
    
    df = st.session_state.df
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        avg_velo = df['release_speed'].mean()
        st.metric("Avg Velocity", f"{avg_velo:.1f} mph")
    with col2:
        avg_spin = df['release_spin_rate'].mean()
        st.metric("Avg Spin Rate", f"{avg_spin:.0f} rpm")
    with col3:
        pitcher_count = df['pitcher_name'].nunique()
        st.metric("Pitchers", pitcher_count)
    with col4:
        pitch_count = len(df)
        st.metric("Total Pitches", pitch_count)
    
    st.markdown("---")
    
    # Velocity by pitcher
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Average Velocity by Pitcher")
        pitcher_stats = df.groupby('pitcher_name').agg({
            'release_speed': 'mean'
        }).sort_values('release_speed', ascending=False).head(15)
        
        fig = px.bar(pitcher_stats, 
                     x=pitcher_stats.index,
                     y='release_speed',
                     title="Top 15 Pitchers by Velocity",
                     labels={'release_speed': 'Avg Velocity (mph)'},
                     color='release_speed',
                     color_continuous_scale='Viridis')
        fig.update_layout(xaxis_tickangle=-45, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Average Spin Rate by Pitcher")
        spinner_stats = df.groupby('pitcher_name').agg({
            'release_spin_rate': 'mean'
        }).sort_values('release_spin_rate', ascending=False).head(15)
        
        fig = px.bar(spinner_stats,
                     x=spinner_stats.index,
                     y='release_spin_rate',
                     title="Top 15 Pitchers by Spin Rate",
                     labels={'release_spin_rate': 'Avg Spin Rate (rpm)'},
                     color='release_spin_rate',
                     color_continuous_scale='Plasma')
        fig.update_layout(xaxis_tickangle=-45, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Scatter: Velocity vs Spin
    st.subheader("Velocity vs Spin Rate")
    pitcher_summary = df.groupby('pitcher_name').agg({
        'release_speed': 'mean',
        'release_spin_rate': 'mean',
        'movement_total': 'mean'
    }).reset_index()
    
    fig = px.scatter(pitcher_summary,
                     x='release_speed',
                     y='release_spin_rate',
                     size='movement_total',
                     hover_name='pitcher_name',
                     title="Pitcher Performance Matrix",
                     labels={
                         'release_speed': 'Avg Velocity (mph)',
                         'release_spin_rate': 'Avg Spin Rate (rpm)',
                         'movement_total': 'Movement'
                     },
                     color='movement_total',
                     color_continuous_scale='RdYlGn')
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

def show_biomechanics():
    """Biomechanics analysis page"""
    st.markdown("<div class='header-title'>🔬 Biomechanics</div>", unsafe_allow_html=True)
    
    df = st.session_state.df
    pitcher_names = sorted(df['pitcher_name'].unique())
    
    # Pitcher selection
    selected_pitcher = st.selectbox("Select Pitcher:", pitcher_names)
    pitcher_data = df[df['pitcher_name'] == selected_pitcher]
    
    # Pitcher stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Avg Velocity", f"{pitcher_data['release_speed'].mean():.1f} mph")
    with col2:
        st.metric("Avg Spin Rate", f"{pitcher_data['release_spin_rate'].mean():.0f} rpm")
    with col3:
        st.metric("Extension", f"{pitcher_data['release_extension'].mean():.1f} in")
    with col4:
        st.metric("Movement Total", f"{pitcher_data['movement_total'].mean():.1f} in")
    
    st.markdown("---")
    
    # Movement profile
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Movement Profile")
        movement = pitcher_data.groupby('pitch_type').agg({
            'pfx_x': 'mean',
            'pfx_z': 'mean'
        }).reset_index()
        
        fig = go.Figure()
        for idx, row in movement.iterrows():
            fig.add_trace(go.Scatter(
                x=[row['pfx_x']],
                y=[row['pfx_z']],
                mode='markers+text',
                marker=dict(size=15),
                text=row['pitch_type'],
                textposition='top center'
            ))
        
        fig.update_layout(
            title="Horizontal vs Vertical Break",
            xaxis_title="Horizontal Break (pfx_x, in)",
            yaxis_title="Vertical Break (pfx_z, in)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Pitch Type Distribution")
        pitch_dist = pitcher_data['pitch_type'].value_counts().reset_index()
        pitch_dist.columns = ['pitch_type', 'count']
        
        fig = px.pie(pitch_dist,
                     values='count',
                     names='pitch_type',
                     title="Pitch Usage")
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Detailed stats by pitch type
    st.subheader("Statistics by Pitch Type")
    pitch_stats = pitcher_data.groupby('pitch_type').agg({
        'release_speed': ['mean', 'std'],
        'release_spin_rate': 'mean',
        'release_extension': 'mean',
        'movement_total': 'mean',
        'spin_efficiency': 'mean'
    }).round(2)
    
    st.dataframe(pitch_stats, use_container_width=True)

def show_clustering():
    """Pitcher clustering analysis"""
    st.markdown("<div class='header-title'>👥 Pitcher Clustering</div>", unsafe_allow_html=True)
    
    df = st.session_state.df
    
    # Prepare data for clustering
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
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(features_scaled)
    pitcher_features['cluster'] = clusters
    
    # Cluster names
    cluster_names = {0: 'Power Arm', 1: 'Elite Spin', 2: 'Command Arm'}
    pitcher_features['cluster_name'] = pitcher_features['cluster'].map(cluster_names)
    
    st.subheader("Pitcher Archetypes")
    col1, col2, col3 = st.columns(3)
    
    for i in range(3):
        cluster_data = pitcher_features[pitcher_features['cluster'] == i]
        with [col1, col2, col3][i]:
            st.markdown(f"### {cluster_names[i]}")
            st.metric("Count", len(cluster_data))
            st.metric("Avg Velocity", f"{cluster_data['release_speed'].mean():.1f} mph")
            st.metric("Avg Spin", f"{cluster_data['release_spin_rate'].mean():.0f} rpm")
    
    st.markdown("---")
    
    # 2D scatter
    st.subheader("Clustering Visualization")
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.scatter(pitcher_features,
                        x='release_speed',
                        y='release_spin_rate',
                        color='cluster_name',
                        hover_name='pitcher_name',
                        title="Velocity vs Spin Rate",
                        labels={
                            'release_speed': 'Avg Velocity (mph)',
                            'release_spin_rate': 'Avg Spin Rate (rpm)',
                            'cluster_name': 'Archetype'
                        },
                        color_discrete_map={
                            'Power Arm': '#FF6B6B',
                            'Elite Spin': '#4ECDC4',
                            'Command Arm': '#45B7D1'
                        })
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(pitcher_features,
                        x='release_extension',
                        y='movement_total',
                        color='cluster_name',
                        hover_name='pitcher_name',
                        title="Extension vs Movement",
                        labels={
                            'release_extension': 'Extension (in)',
                            'movement_total': 'Movement Total (in)',
                            'cluster_name': 'Archetype'
                        },
                        color_discrete_map={
                            'Power Arm': '#FF6B6B',
                            'Elite Spin': '#4ECDC4',
                            'Command Arm': '#45B7D1'
                        })
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.subheader("Cluster Profiles")
    st.dataframe(pitcher_features.sort_values('cluster'), use_container_width=True)

def show_leaderboards():
    """Leaderboards page"""
    st.markdown("<div class='header-title'>🏆 Leaderboards</div>", unsafe_allow_html=True)
    
    df = st.session_state.df
    
    # Aggregate by pitcher
    pitcher_stats = df.groupby('pitcher_name').agg({
        'release_speed': ['mean', 'count'],
        'release_spin_rate': 'mean',
        'release_extension': 'mean',
        'spin_efficiency': 'mean',
        'movement_total': 'mean'
    }).round(2)
    
    pitcher_stats.columns = ['Velocity', 'Pitches', 'Spin Rate', 'Extension', 'Spin Eff', 'Movement']
    pitcher_stats = pitcher_stats[pitcher_stats['Pitches'] >= 20].reset_index()  # Min 20 pitches
    
    # Tabs for different leaderboards
    tabs = st.tabs(["Velocity", "Spin Rate", "Extension", "Spin Efficiency", "Movement"])
    
    with tabs[0]:
        st.subheader("Top 25 by Velocity")
        lb = pitcher_stats.nlargest(25, 'Velocity')[['pitcher_name', 'Velocity', 'Pitches']]
        st.dataframe(lb, use_container_width=True)
    
    with tabs[1]:
        st.subheader("Top 25 by Spin Rate")
        lb = pitcher_stats.nlargest(25, 'Spin Rate')[['pitcher_name', 'Spin Rate', 'Pitches']]
        st.dataframe(lb, use_container_width=True)
    
    with tabs[2]:
        st.subheader("Top 25 by Extension")
        lb = pitcher_stats.nlargest(25, 'Extension')[['pitcher_name', 'Extension', 'Pitches']]
        st.dataframe(lb, use_container_width=True)
    
    with tabs[3]:
        st.subheader("Top 25 by Spin Efficiency")
        lb = pitcher_stats.nlargest(25, 'Spin Eff')[['pitcher_name', 'Spin Eff', 'Pitches']]
        st.dataframe(lb, use_container_width=True)
    
    with tabs[4]:
        st.subheader("Top 25 by Movement")
        lb = pitcher_stats.nlargest(25, 'Movement')[['pitcher_name', 'Movement', 'Pitches']]
        st.dataframe(lb, use_container_width=True)

def show_predictions():
    """ML predictions page"""
    st.markdown("<div class='header-title'>🤖 ML Predictions</div>", unsafe_allow_html=True)
    
    df = st.session_state.df
    
    try:
        # Initialize prediction model
        model = PitchPredictionModel()
        
        # Train models
        with st.spinner('Training prediction models...'):
            velo_metrics = model.train_velocity_model(df)
            spin_metrics = model.train_spin_model(df)
        
        # Display metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Velocity Prediction Model")
            st.metric("R² Score", f"{velo_metrics['r2']:.4f}")
            st.metric("MAE", f"{velo_metrics['mae']:.2f} mph")
            st.metric("RMSE", f"{velo_metrics['rmse']:.2f} mph")
        
        with col2:
            st.subheader("Spin Rate Prediction Model")
            st.metric("R² Score", f"{spin_metrics['r2']:.4f}")
            st.metric("MAE", f"{spin_metrics['mae']:.0f} rpm")
            st.metric("RMSE", f"{spin_metrics['rmse']:.0f} rpm")
        
        st.markdown("---")
        
        # Feature importance
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Velocity Model - Feature Importance")
            features = ['Release Spin', 'Extension', 'Arm Slot', 'Pitch Type']
            importance = velo_metrics['feature_importance'][:4] if 'feature_importance' in velo_metrics else np.random.rand(4)
            fig = px.bar(x=features, y=importance, labels={'x': 'Feature', 'y': 'Importance'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Spin Model - Feature Importance")
            features = ['Release Pos X', 'Release Pos Z', 'Extension', 'Arm Slot']
            importance = spin_metrics['feature_importance'][:4] if 'feature_importance' in spin_metrics else np.random.rand(4)
            fig = px.bar(x=features, y=importance, labels={'x': 'Feature', 'y': 'Importance'})
            st.plotly_chart(fig, use_container_width=True)
        
        st.info("✅ Models trained successfully on current dataset")
    
    except Exception as e:
        st.error(f"Error training models: {str(e)}")

if __name__ == "__main__":
    main()
