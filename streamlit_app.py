import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add src to path to import our modules
sys.path.append('src')

# Define our own data generation function (no external dependencies)
def generate_synthetic_data(start_date, end_date, num_products=5):
    """Generate synthetic sales data"""
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    data = []
    
    for product_id in range(1, num_products + 1):
        for date in date_range:
            # Add some seasonality and trends
            base_sales = 100 + product_id * 20
            seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * date.dayofyear / 365)
            trend_factor = 1 + 0.001 * (date - pd.Timestamp('2023-01-01')).days
            noise = np.random.normal(0, 0.1)
            
            sales = int(base_sales * seasonal_factor * trend_factor * (1 + noise))
            data.append({
                'date': date,
                'product_id': f'Product_{product_id}',
                'sales': max(0, sales)
            })
    
    return pd.DataFrame(data)

# Page configuration
st.set_page_config(
    page_title="MLOps Demand Forecasting",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stButton > button {
        width: 100%;
        border-radius: 0.5rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Main navigation
st.markdown('<h1 class="main-header">🚀 MLOps Demand Forecasting</h1>', unsafe_allow_html=True)

# Sidebar navigation
page = st.sidebar.selectbox(
    "Navigation",
    ["🏠 Overview", "📈 Data Generation", "🤖 Model Training", "🔮 Forecasting", "📊 Analytics"]
)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'forecast' not in st.session_state:
    st.session_state.forecast = None

# Overview Page
if page == "🏠 Overview":
    st.markdown("## 📊 Project Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 Objective</h3>
            <p>Build an end-to-end MLOps pipeline for demand forecasting using Azure services</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🛠️ Tech Stack</h3>
            <p>Azure ML, Databricks, Prophet, Streamlit</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>📈 Status</h3>
            <p>Live on Streamlit Cloud</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Project metrics
    st.markdown("### 📈 Key Metrics")
    
    if st.session_state.data is not None:
        total_sales = st.session_state.data['sales'].sum()
        avg_daily_sales = st.session_state.data.groupby('date')['sales'].sum().mean()
        num_products = st.session_state.data['product_id'].nunique()
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Sales", f"{total_sales:,}")
        col2.metric("Avg Daily Sales", f"{avg_daily_sales:.0f}")
        col3.metric("Products", num_products)
        col4.metric("Data Points", len(st.session_state.data))
    else:
        st.info("Generate data to see metrics")
    
    # Quick actions
    st.markdown("### ⚡ Quick Actions")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🚀 Generate Sample Data"):
            with st.spinner("Generating synthetic data..."):
                start_date = datetime(2023, 1, 1)
                end_date = datetime(2023, 12, 31)
                st.session_state.data = generate_synthetic_data(start_date, end_date)
                st.success("Data generated successfully!")
                st.rerun()
    
    with col2:
        if st.button("📊 View Sample Analytics"):
            if st.session_state.data is not None:
                st.markdown("### Sample Analytics")
                
                # Sales trend
                daily_sales = st.session_state.data.groupby('date')['sales'].sum().reset_index()
                fig = px.line(daily_sales, x='date', y='sales', title='Daily Sales Trend')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Please generate data first")

# Data Generation Page
elif page == "📈 Data Generation":
    st.markdown("## 📈 Data Generation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ⚙️ Parameters")
        start_date = st.date_input("Start Date", value=datetime(2023, 1, 1))
        end_date = st.date_input("End Date", value=datetime(2023, 12, 31))
        num_products = st.slider("Number of Products", 1, 10, 5)
        
        if st.button("🔄 Generate Data", type="primary"):
            with st.spinner("Generating synthetic data..."):
                st.session_state.data = generate_synthetic_data(
                    start_date, end_date, num_products
                )
                st.success("Data generated successfully!")
    
    with col2:
        st.markdown("### 📊 Data Preview")
        if st.session_state.data is not None:
            st.dataframe(st.session_state.data.head(10))
            
            # Data statistics
            st.markdown("#### 📈 Data Statistics")
            st.write(f"**Total Records:** {len(st.session_state.data):,}")
            st.write(f"**Date Range:** {st.session_state.data['date'].min().strftime('%Y-%m-%d')} to {st.session_state.data['date'].max().strftime('%Y-%m-%d')}")
            st.write(f"**Products:** {st.session_state.data['product_id'].nunique()}")
            st.write(f"**Total Sales:** {st.session_state.data['sales'].sum():,}")
        else:
            st.info("Generate data to see preview")
    
    # Data visualization
    if st.session_state.data is not None:
        st.markdown("### 📊 Data Visualization")
        
        tab1, tab2, tab3 = st.tabs(["Sales Trend", "Product Performance", "Seasonality"])
        
        with tab1:
            daily_sales = st.session_state.data.groupby('date')['sales'].sum().reset_index()
            fig = px.line(daily_sales, x='date', y='sales', title='Daily Sales Trend')
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            product_sales = st.session_state.data.groupby('product_id')['sales'].sum().reset_index()
            fig = px.bar(product_sales, x='product_id', y='sales', title='Total Sales by Product')
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            # Monthly seasonality
            monthly_sales = st.session_state.data.groupby(st.session_state.data['date'].dt.to_period('M'))['sales'].sum().reset_index()
            monthly_sales['date'] = monthly_sales['date'].astype(str)
            fig = px.bar(monthly_sales, x='date', y='sales', title='Monthly Sales Pattern')
            st.plotly_chart(fig, use_container_width=True)

# Model Training Page
elif page == "🤖 Model Training":
    st.markdown("## 🤖 Model Training")
    
    if st.session_state.data is None:
        st.warning("Please generate data first in the Data Generation tab")
    else:
        st.markdown("### 📊 Training Data Summary")
        st.write(f"**Training Period:** {st.session_state.data['date'].min().strftime('%Y-%m-%d')} to {st.session_state.data['date'].max().strftime('%Y-%m-%d')}")
        st.write(f"**Total Records:** {len(st.session_state.data):,}")
        
        # Model parameters
        st.markdown("### ⚙️ Model Parameters")
        col1, col2 = st.columns(2)
        
        with col1:
            changepoint_prior_scale = st.slider("Changepoint Prior Scale", 0.01, 0.5, 0.05, 0.01)
            seasonality_prior_scale = st.slider("Seasonality Prior Scale", 0.01, 10.0, 10.0, 0.01)
        
        with col2:
            seasonality_mode = st.selectbox("Seasonality Mode", ["additive", "multiplicative"])
            forecast_periods = st.number_input("Forecast Periods", 30, 365, 90)
        
        if st.button("🚀 Train Model", type="primary"):
            with st.spinner("Training Prophet model..."):
                try:
                    from prophet import Prophet
                    
                    # Prepare data for Prophet
                    daily_sales = st.session_state.data.groupby('date')['sales'].sum().reset_index()
                    daily_sales.columns = ['ds', 'y']
                    
                    # Create and train model
                    model = Prophet(
                        changepoint_prior_scale=changepoint_prior_scale,
                        seasonality_prior_scale=seasonality_prior_scale,
                        seasonality_mode=seasonality_mode,
                        yearly_seasonality=True,
                        weekly_seasonality=True,
                        daily_seasonality=False
                    )
                    
                    model.fit(daily_sales)
                    st.session_state.model = model
                    
                    # Generate forecast
                    future = model.make_future_dataframe(periods=forecast_periods)
                    forecast = model.predict(future)
                    st.session_state.forecast = forecast
                    
                    st.success("Model trained successfully!")
                    
                except Exception as e:
                    st.error(f"Error training model: {str(e)}")
        
        # Model results
        if st.session_state.model is not None:
            st.markdown("### 📈 Model Results")
            
            # Model components
            if st.session_state.forecast is not None:
                fig = st.session_state.model.plot_components(st.session_state.forecast)
                st.pyplot(fig)
            
            # Forecast plot
            if st.session_state.forecast is not None:
                fig = st.session_state.model.plot(st.session_state.forecast)
                st.pyplot(fig)

# Forecasting Page
elif page == "🔮 Forecasting":
    st.markdown("## 🔮 Demand Forecasting")
    
    if st.session_state.model is None:
        st.warning("Please train a model first in the Model Training tab")
    else:
        st.markdown("### 📊 Forecast Results")
        
        if st.session_state.forecast is not None:
            # Forecast metrics
            latest_forecast = st.session_state.forecast.tail(30)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Next 30 Days Avg", f"{latest_forecast['yhat'].mean():.0f}")
            col2.metric("Forecast Range", f"{latest_forecast['yhat_lower'].mean():.0f} - {latest_forecast['yhat_upper'].mean():.0f}")
            col3.metric("Confidence", "95%")
            
            # Interactive forecast
            st.markdown("### 🎯 Interactive Forecast")
            
            forecast_days = st.slider("Forecast Days", 7, 365, 30)
            
            if st.button("🔄 Update Forecast"):
                future = st.session_state.model.make_future_dataframe(periods=forecast_days)
                forecast = st.session_state.model.predict(future)
                
                # Plot forecast
                fig = go.Figure()
                
                # Historical data
                historical = st.session_state.forecast[st.session_state.forecast['ds'] <= st.session_state.data['date'].max()]
                fig.add_trace(go.Scatter(
                    x=historical['ds'], 
                    y=historical['yhat'],
                    mode='lines',
                    name='Historical',
                    line=dict(color='blue')
                ))
                
                # Forecast
                forecast_period = forecast[forecast['ds'] > st.session_state.data['date'].max()]
                fig.add_trace(go.Scatter(
                    x=forecast_period['ds'],
                    y=forecast_period['yhat'],
                    mode='lines',
                    name='Forecast',
                    line=dict(color='red', dash='dash')
                ))
                
                # Confidence intervals
                fig.add_trace(go.Scatter(
                    x=forecast_period['ds'],
                    y=forecast_period['yhat_upper'],
                    mode='lines',
                    name='Upper Bound',
                    line=dict(color='gray', width=0),
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter(
                    x=forecast_period['ds'],
                    y=forecast_period['yhat_lower'],
                    mode='lines',
                    fill='tonexty',
                    name='Confidence Interval',
                    line=dict(color='gray', width=0)
                ))
                
                fig.update_layout(
                    title='Demand Forecast',
                    xaxis_title='Date',
                    yaxis_title='Sales',
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Forecast table
            st.markdown("### 📋 Forecast Details")
            if st.session_state.forecast is not None:
                forecast_df = st.session_state.forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30)
                forecast_df.columns = ['Date', 'Forecast', 'Lower Bound', 'Upper Bound']
                forecast_df['Date'] = forecast_df['Date'].dt.strftime('%Y-%m-%d')
                forecast_df['Forecast'] = forecast_df['Forecast'].round(0).astype(int)
                forecast_df['Lower Bound'] = forecast_df['Lower Bound'].round(0).astype(int)
                forecast_df['Upper Bound'] = forecast_df['Upper Bound'].round(0).astype(int)
                
                st.dataframe(forecast_df, use_container_width=True)

# Analytics Page
elif page == "📊 Analytics":
    st.markdown("## 📊 Sales Analytics")
    
    if st.session_state.data is None:
        st.warning("Please generate data first in the Data Generation tab")
    else:
        st.markdown("### 📈 Key Insights")
        
        # Calculate metrics
        total_sales = st.session_state.data['sales'].sum()
        avg_daily_sales = st.session_state.data.groupby('date')['sales'].sum().mean()
        best_day = st.session_state.data.groupby('date')['sales'].sum().idxmax()
        worst_day = st.session_state.data.groupby('date')['sales'].sum().idxmin()
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Sales", f"{total_sales:,}")
        col2.metric("Avg Daily Sales", f"{avg_daily_sales:.0f}")
        col3.metric("Best Day", best_day.strftime('%Y-%m-%d'))
        col4.metric("Worst Day", worst_day.strftime('%Y-%m-%d'))
        
        # Analytics tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Trend Analysis", "Seasonality", "Product Analysis", "Time Series Decomposition"])
        
        with tab1:
            st.markdown("### 📈 Trend Analysis")
            
            # Daily trend
            daily_sales = st.session_state.data.groupby('date')['sales'].sum().reset_index()
            
            # Add moving averages
            daily_sales['7d_ma'] = daily_sales['sales'].rolling(7).mean()
            daily_sales['30d_ma'] = daily_sales['sales'].rolling(30).mean()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=daily_sales['date'], y=daily_sales['sales'], 
                                    mode='lines', name='Daily Sales'))
            fig.add_trace(go.Scatter(x=daily_sales['date'], y=daily_sales['7d_ma'], 
                                    mode='lines', name='7-Day MA'))
            fig.add_trace(go.Scatter(x=daily_sales['date'], y=daily_sales['30d_ma'], 
                                    mode='lines', name='30-Day MA'))
            
            fig.update_layout(title='Sales Trend with Moving Averages',
                            xaxis_title='Date', yaxis_title='Sales')
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.markdown("### 📅 Seasonality Analysis")
            
            # Monthly seasonality
            monthly_sales = st.session_state.data.groupby(st.session_state.data['date'].dt.to_period('M'))['sales'].sum().reset_index()
            monthly_sales['date'] = monthly_sales['date'].astype(str)
            
            fig = px.bar(monthly_sales, x='date', y='sales', title='Monthly Sales Pattern')
            st.plotly_chart(fig, use_container_width=True)
            
            # Weekly seasonality
            weekly_sales = st.session_state.data.groupby(st.session_state.data['date'].dt.dayofweek)['sales'].sum().reset_index()
            weekly_sales['day_name'] = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            
            fig = px.bar(weekly_sales, x='day_name', y='sales', title='Weekly Sales Pattern')
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.markdown("### 🏷️ Product Analysis")
            
            # Product performance
            product_sales = st.session_state.data.groupby('product_id')['sales'].sum().reset_index()
            product_sales = product_sales.sort_values('sales', ascending=False)
            
            fig = px.bar(product_sales, x='product_id', y='sales', title='Total Sales by Product')
            st.plotly_chart(fig, use_container_width=True)
            
            # Product trend over time
            product_trend = st.session_state.data.pivot_table(
                index='date', columns='product_id', values='sales', aggfunc='sum'
            ).fillna(0)
            
            fig = px.line(product_trend, title='Product Sales Trend Over Time')
            st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.markdown("### 🔍 Time Series Decomposition")
            
            try:
                from statsmodels.tsa.seasonal import seasonal_decompose
                
                # Prepare data for decomposition
                daily_sales = st.session_state.data.groupby('date')['sales'].sum()
                
                # Perform decomposition
                decomposition = seasonal_decompose(daily_sales, model='additive', period=7)
                
                # Plot decomposition
                fig = make_subplots(rows=4, cols=1, subplot_titles=('Original', 'Trend', 'Seasonal', 'Residual'))
                
                fig.add_trace(go.Scatter(x=daily_sales.index, y=daily_sales.values, name='Original'), row=1, col=1)
                fig.add_trace(go.Scatter(x=decomposition.trend.index, y=decomposition.trend.values, name='Trend'), row=2, col=1)
                fig.add_trace(go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal.values, name='Seasonal'), row=3, col=1)
                fig.add_trace(go.Scatter(x=decomposition.resid.index, y=decomposition.resid.values, name='Residual'), row=4, col=1)
                
                fig.update_layout(height=800, title_text="Time Series Decomposition")
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error in time series decomposition: {str(e)}")
                st.info("This feature requires additional statistical packages")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🚀 MLOps Demand Forecasting | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True) 