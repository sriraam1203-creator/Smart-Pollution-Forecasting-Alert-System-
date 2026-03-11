"""
FINAL WORKING STREAMLIT DASHBOARD
==================================
Fixed timestamp error with Plotly
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime

# Page config
st.set_page_config(
    page_title="PM2.5 Forecast & Alert System",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load data function
@st.cache_data
def load_data():
    """Load historical and forecast data"""
    try:
        # Load forecast
        forecast_df = pd.read_csv('data/outputs/vellore_30day_forecast.csv')
        forecast_df['date'] = pd.to_datetime(forecast_df['date'], format='mixed')
        
        # Load sequences for historical data
        y = np.load('data/processed/y_clean_targets.npy')
        
        # Load dates from CSV
        df = pd.read_csv('data/processed/vellore_clean_dataset.csv')
        df['date'] = pd.to_datetime(df['date'], format='mixed')
        
        # Create historical DataFrame (skip first 14 for sequences)
        dates = df['date'].iloc[14:].reset_index(drop=True)
        min_len = min(len(dates), len(y))
        
        
        hist_df = pd.DataFrame({
            'date': dates.iloc[:min_len],
            'PM2_5':  y[:min_len]
        })
        
        return hist_df, forecast_df
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None

def get_alert_color(level):
    """Return color for alert level"""
    colors = {
        'SAFE': '#28a745',
        'MODERATE': '#ffc107',
        'UNHEALTHY': '#fd7e14',
        'SEVERE': '#dc3545'
    }
    return colors.get(level, '#6c757d')

def get_alert_icon(level):
    """Return emoji for alert level"""
    icons = {
        'SAFE': '✅',
        'MODERATE': '⚠️',
        'UNHEALTHY': '❌',
        'SEVERE': '🚨'
    }
    return icons.get(level, '❓')

# Main app
def main():
    # Header
    st.markdown('<div class="main-header">🌍 PM2.5 Forecast & Alert System</div>', 
                unsafe_allow_html=True)
    st.markdown("**Real-time air quality forecasting for Vellore, India**")
    st.markdown("---")
    
    # Load data
    with st.spinner("Loading data..."):
        hist_df, forecast_df = load_data()
    
    if hist_df is None or forecast_df is None:
        st.error("❌ Failed to load data. Please run quick_forecast.py first!")
        st.code("python quick_forecast.py", language="bash")
        return
    
    # Sidebar
    st.sidebar.title("📅 Settings")
    
    # Info box
    st.sidebar.success(f"""
    **Data Loaded:**
    - Historical: {len(hist_df)} days
    - Forecast: 30 days
    - Last update: {hist_df['date'].max().strftime('%Y-%m-%d')}
    """)
    
    # Simple slider for historical days
    st.sidebar.subheader("Historical Data")
    days_to_show = st.sidebar.slider(
        "Days to display", 
        min_value=30, 
        max_value=min(365, len(hist_df)), 
        value=90,
        step=10
    )
    
    # View options
    st.sidebar.subheader("Display Options")
    show_historical = st.sidebar.checkbox("Show Historical", value=True)
    show_forecast = st.sidebar.checkbox("Show Forecast", value=True)
    show_thresholds = st.sidebar.checkbox("Show Thresholds", value=True)
    
    # Filter historical to last N days
    hist_df_filtered = hist_df.tail(days_to_show).copy()
    
    # Key Metrics
    st.subheader("📊 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_forecast = forecast_df['PM2_5'].mean()
        st.metric("Avg Forecast PM2.5", f"{avg_forecast:.1f} µg/m³")
    
    with col2:
        safe_days = (forecast_df['Alert'] == 'SAFE').sum()
        st.metric("Safe Days", f"{safe_days}/30")
    
    with col3:
        moderate_days = (forecast_df['Alert'] == 'MODERATE').sum()
        st.metric("Moderate Days", f"{moderate_days}/30")
    
    with col4:
        unhealthy_days = forecast_df['Alert'].isin(['UNHEALTHY', 'SEVERE']).sum()
        st.metric("Unhealthy Days", f"{unhealthy_days}/30")
    
    st.markdown("---")
    
    # Main Chart
    st.subheader(f"📈 PM2.5 Trend: Last {days_to_show} Days → Next 30 Days")
    
    fig = go.Figure()
    
    # Historical data
    if show_historical:
        fig.add_trace(go.Scatter(
            x=hist_df_filtered['date'],
            y=hist_df_filtered['PM2_5'],
            mode='lines+markers',
            name='Historical PM2.5',
            line=dict(color='steelblue', width=2),
            marker=dict(size=3),
            hovertemplate='<b>%{x|%Y-%m-%d}</b><br>PM2.5: %{y:.1f} µg/m³<extra></extra>'
        ))
    
    # Forecast data
    if show_forecast:
        fig.add_trace(go.Scatter(
            x=forecast_df['date'],
            y=forecast_df['PM2_5'],
            mode='lines+markers',
            name='30-Day Forecast',
            line=dict(color='darkorange', width=2.5, dash='dash'),
            marker=dict(size=5, symbol='square'),
            hovertemplate='<b>%{x|%Y-%m-%d}</b><br>PM2.5: %{y:.1f} µg/m³<extra></extra>'
        ))
    
    # Connection line (if both shown)
    if show_historical and show_forecast:
        last_hist = hist_df_filtered.iloc[-1]
        first_forecast = forecast_df.iloc[0]
        
        fig.add_trace(go.Scatter(
            x=[last_hist['date'], first_forecast['date']],
            y=[last_hist['PM2_5'], first_forecast['PM2_5']],
            mode='lines',
            line=dict(color='gray', width=2, dash='dot'),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Threshold lines
    if show_thresholds:
        fig.add_hline(y=30, line_dash="dot", line_color="green", 
                     annotation_text="Safe (≤30)", annotation_position="right")
        fig.add_hline(y=60, line_dash="dot", line_color="gold",
                     annotation_text="Moderate (≤60)", annotation_position="right")
        fig.add_hline(y=90, line_dash="dot", line_color="orange",
                     annotation_text="Unhealthy (≤90)", annotation_position="right")
    
    # Vertical line at forecast start - FIXED VERSION
    if show_historical and show_forecast:
        # Get the last historical date and convert to string format
        forecast_start_date = hist_df['date'].max()
        
        # Add vertical shape instead of vline to avoid timestamp error
        fig.add_shape(
            type="line",
            x0=forecast_start_date,
            x1=forecast_start_date,
            y0=0,
            y1=1,
            yref="paper",
            line=dict(color="red", width=2, dash="dash")
        )
        
        # Add annotation separately
        fig.add_annotation(
            x=forecast_start_date,
            y=1,
            yref="paper",
            text="Forecast Start",
            showarrow=False,
            yshift=10
        )
    
    # Layout
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="PM2.5 (µg/m³)",
        hovermode='x unified',
        height=500,
        showlegend=True,
        legend=dict(
            orientation="h", 
            yanchor="bottom", 
            y=1.02, 
            xanchor="right", 
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🚨 High Pollution Alerts")
        
        high_days = forecast_df[forecast_df['PM2_5'] > 60].sort_values('PM2_5', ascending=False)
        
        if len(high_days) > 0:
            st.warning(f"⚠️ {len(high_days)} days with PM2.5 > 60 µg/m³")
            
            for idx, row in high_days.head(10).iterrows():
                icon = get_alert_icon(row['Alert'])
                color = get_alert_color(row['Alert'])
                
                st.markdown(f"""
                    <div style='background-color: {color}20; 
                                border-left: 4px solid {color}; 
                                padding: 10px; 
                                margin: 5px 0; 
                                border-radius: 5px;'>
                        {icon} <b>{row['date'].strftime('%Y-%m-%d')}</b>: 
                        {row['PM2_5']:.1f} µg/m³ ({row['Alert']})
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.success("✅ No high pollution days forecasted!")
    
    with col2:
        st.subheader("📊 Forecast Statistics")
        
        stats = {
            'Metric': ['Mean', 'Median', 'Min', 'Max', 'Std Dev'],
            'Value (µg/m³)': [
                f"{forecast_df['PM2_5'].mean():.2f}",
                f"{forecast_df['PM2_5'].median():.2f}",
                f"{forecast_df['PM2_5'].min():.2f}",
                f"{forecast_df['PM2_5'].max():.2f}",
                f"{forecast_df['PM2_5'].std():.2f}"
            ]
        }
        st.table(pd.DataFrame(stats))
        
        # Alert distribution
        st.subheader("Alert Distribution")
        alert_counts = forecast_df['Alert'].value_counts()
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=alert_counts.index,
            values=alert_counts.values,
            marker=dict(colors=[get_alert_color(label) for label in alert_counts.index]),
            hole=0.3
        )])
        fig_pie.update_layout(height=300, showlegend=True)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    st.markdown("---")
    
    # Full forecast table
    st.subheader("📅 Complete 30-Day Forecast")
    
    display_df = forecast_df.copy()
    display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
    display_df['PM2_5'] = display_df['PM2_5'].round(2)
    display_df = display_df.rename(columns={
        'date': 'Date',
        'PM2_5': 'PM2.5 (µg/m³)',
        'Alert': 'Alert Level'
    })
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Download button
    csv = forecast_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Forecast CSV",
        data=csv,
        file_name=f"pm25_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: gray; padding: 20px;'>
            <p><b>PM2.5 Forecasting System | Vellore, India</b></p>
            <p>Model: LSTM | R² = 0.095 | MAE = 19.98 µg/m³</p>
            <p>Data Sources: Satellite AOD + Ground Stations + Weather API</p>
            <p>⚠️ Forecasts are statistical estimates - use alongside official sources</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()