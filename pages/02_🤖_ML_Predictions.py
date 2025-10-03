import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="ML Predictions", page_icon="🤖", layout="wide")

st.title("🤖 Machine Learning Stock Price Predictions")
st.markdown("---")

# Sidebar controls
st.sidebar.header("Prediction Settings")
ticker_symbol = st.sidebar.text_input("Stock Ticker", value="AAPL").upper()

# Model selection
model_type = st.sidebar.selectbox(
    "Select ML Model",
    ["Random Forest", "Linear Regression", "Both Models"]
)

# Prediction parameters
prediction_days = st.sidebar.slider("Days to Predict", 1, 30, 7)
training_period = st.sidebar.selectbox("Training Data Period", ["1y", "2y", "3y", "5y"], index=1)

# Feature engineering options
st.sidebar.subheader("Feature Engineering")
use_technical_indicators = st.sidebar.checkbox("Use Technical Indicators", value=True)
use_volume = st.sidebar.checkbox("Use Volume Data", value=True)
lookback_window = st.sidebar.slider("Lookback Window (days)", 5, 60, 20)

@st.cache_data(ttl=300)
def fetch_training_data(symbol, period):
    """Fetch historical data for training"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        return data
    except Exception as e:
        st.error(f"Error fetching training data: {str(e)}")
        return None

def create_features(data, lookback_window=20):
    """Create features for ML model"""
    features_df = pd.DataFrame(index=data.index)
    
    # Basic price features
    features_df['Close'] = data['Close']
    features_df['Open'] = data['Open']
    features_df['High'] = data['High']
    features_df['Low'] = data['Low']
    
    # Price-based features
    features_df['Price_Change'] = data['Close'].pct_change()
    features_df['Price_Range'] = (data['High'] - data['Low']) / data['Close']
    features_df['Open_Close_Ratio'] = data['Open'] / data['Close']
    
    if use_volume:
        features_df['Volume'] = data['Volume']
        features_df['Volume_Change'] = data['Volume'].pct_change()
        features_df['Price_Volume'] = data['Close'] * data['Volume']
    
    if use_technical_indicators:
        # Moving averages
        for window in [5, 10, 20, 50]:
            if len(data) >= window:
                ma = data['Close'].rolling(window=window).mean()
                features_df[f'SMA_{window}'] = ma
                features_df[f'Price_SMA_{window}_Ratio'] = data['Close'] / ma
        
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features_df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        bb_period = 20
        bb_std = data['Close'].rolling(window=bb_period).std()
        bb_middle = data['Close'].rolling(window=bb_period).mean()
        features_df['BB_Upper'] = bb_middle + (bb_std * 2)
        features_df['BB_Lower'] = bb_middle - (bb_std * 2)
        features_df['BB_Position'] = (data['Close'] - features_df['BB_Lower']) / (features_df['BB_Upper'] - features_df['BB_Lower'])
        
        # MACD
        exp1 = data['Close'].ewm(span=12).mean()
        exp2 = data['Close'].ewm(span=26).mean()
        features_df['MACD'] = exp1 - exp2
        features_df['MACD_Signal'] = features_df['MACD'].ewm(span=9).mean()
        
        # Volatility
        features_df['Volatility'] = data['Close'].rolling(window=20).std()
    
    # Lag features
    for lag in range(1, min(lookback_window + 1, 11)):  # Limit to prevent too many features
        features_df[f'Close_Lag_{lag}'] = features_df['Close'].shift(lag)
        features_df[f'Price_Change_Lag_{lag}'] = features_df['Price_Change'].shift(lag)
    
    return features_df

def prepare_data_for_ml(features_df, target_days=1):
    """Prepare data for machine learning"""
    # Create target variable (future price)
    target = features_df['Close'].shift(-target_days)
    
    # Remove rows with NaN values
    valid_data = features_df.dropna()
    valid_target = target.dropna()
    
    # Align data and target
    common_index = valid_data.index.intersection(valid_target.index)
    X = valid_data.loc[common_index]
    y = valid_target.loc[common_index]
    
    return X, y

def train_models(X, y):
    """Train both Random Forest and Linear Regression models"""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Scale features for Linear Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {}
    predictions = {}
    metrics = {}
    
    # Random Forest
    if model_type in ["Random Forest", "Both Models"]:
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        
        models['Random Forest'] = rf_model
        predictions['Random Forest'] = rf_pred
        metrics['Random Forest'] = {
            'MSE': mean_squared_error(y_test, rf_pred),
            'MAE': mean_absolute_error(y_test, rf_pred),
            'R2': r2_score(y_test, rf_pred)
        }
    
    # Linear Regression
    if model_type in ["Linear Regression", "Both Models"]:
        lr_model = LinearRegression()
        lr_model.fit(X_train_scaled, y_train)
        lr_pred = lr_model.predict(X_test_scaled)
        
        models['Linear Regression'] = (lr_model, scaler)
        predictions['Linear Regression'] = lr_pred
        metrics['Linear Regression'] = {
            'MSE': mean_squared_error(y_test, lr_pred),
            'MAE': mean_absolute_error(y_test, lr_pred),
            'R2': r2_score(y_test, lr_pred)
        }
    
    return models, predictions, metrics, X_test, y_test

def generate_future_predictions(models, features_df, prediction_days):
    """Generate future predictions"""
    predictions = {}
    last_known_data = features_df.iloc[-1:].copy()
    
    for model_name, model_info in models.items():
        future_predictions = []
        current_data = last_known_data.copy()
        
        for day in range(prediction_days):
            if model_name == "Random Forest":
                model = model_info
                pred = model.predict(current_data)[0]
            else:  # Linear Regression
                model, scaler = model_info
                scaled_data = scaler.transform(current_data)
                pred = model.predict(scaled_data)[0]
            
            future_predictions.append(pred)
            
            # Update current_data for next prediction (simplified approach)
            # In practice, you'd want more sophisticated feature updating
            current_data['Close'] = pred
            
            # Shift lag features
            for lag in range(1, 11):
                if f'Close_Lag_{lag}' in current_data.columns:
                    if lag == 1:
                        current_data[f'Close_Lag_{lag}'] = current_data['Close']
                    elif f'Close_Lag_{lag-1}' in current_data.columns:
                        current_data[f'Close_Lag_{lag}'] = current_data[f'Close_Lag_{lag-1}']
        
        predictions[model_name] = future_predictions
    
    return predictions

# Main execution
if ticker_symbol:
    with st.spinner("Fetching data and training models..."):
        # Fetch data
        stock_data = fetch_training_data(ticker_symbol, training_period)
        
        if stock_data is not None and not stock_data.empty:
            # Display basic info
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Current Price", f"${stock_data['Close'].iloc[-1]:.2f}")
            
            with col2:
                price_change = stock_data['Close'].pct_change().iloc[-1] * 100
                st.metric("Daily Change", f"{price_change:.2f}%")
            
            with col3:
                st.metric("Training Days", len(stock_data))
            
            with col4:
                volatility = stock_data['Close'].pct_change().std() * np.sqrt(252) * 100
                st.metric("Volatility", f"{volatility:.1f}%")
            
            # Create features
            features_df = create_features(stock_data, lookback_window)
            
            # Prepare data for ML
            X, y = prepare_data_for_ml(features_df, target_days=1)
            
            if len(X) > 50:  # Ensure we have enough data
                # Train models
                models, test_predictions, metrics, X_test, y_test = train_models(X, y)
                
                # Display model performance
                st.subheader("📊 Model Performance")
                
                perf_cols = st.columns(len(metrics))
                for idx, (model_name, model_metrics) in enumerate(metrics.items()):
                    with perf_cols[idx]:
                        st.markdown(f"**{model_name}**")
                        st.write(f"R² Score: {model_metrics['R2']:.4f}")
                        st.write(f"MAE: ${model_metrics['MAE']:.2f}")
                        st.write(f"MSE: ${model_metrics['MSE']:.2f}")
                
                # Generate future predictions
                future_predictions = generate_future_predictions(models, features_df, prediction_days)
                
                # Create prediction visualization
                st.subheader("🔮 Future Price Predictions")
                
                fig = go.Figure()
                
                # Historical prices (last 60 days)
                recent_data = stock_data.tail(60)
                fig.add_trace(go.Scatter(
                    x=recent_data.index,
                    y=recent_data['Close'],
                    mode='lines',
                    name='Historical Price',
                    line=dict(color='blue', width=2)
                ))
                
                # Future dates
                last_date = stock_data.index[-1]
                future_dates = [last_date + timedelta(days=i+1) for i in range(prediction_days)]
                
                # Add predictions for each model
                colors = ['red', 'green', 'orange', 'purple']
                for idx, (model_name, predictions) in enumerate(future_predictions.items()):
                    fig.add_trace(go.Scatter(
                        x=future_dates,
                        y=predictions,
                        mode='lines+markers',
                        name=f'{model_name} Prediction',
                        line=dict(color=colors[idx % len(colors)], width=2, dash='dash'),
                        marker=dict(size=6)
                    ))
                
                fig.update_layout(
                    title=f"{ticker_symbol} Price Predictions ({prediction_days} days)",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    height=500,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Prediction summary table
                st.subheader("📋 Prediction Summary")
                
                prediction_df = pd.DataFrame(index=future_dates)
                for model_name, predictions in future_predictions.items():
                    prediction_df[f'{model_name} Price'] = predictions
                
                prediction_df.index = prediction_df.index.strftime('%Y-%m-%d')
                st.dataframe(prediction_df.round(2), use_container_width=True)
                
                # Model comparison on test data
                if len(models) > 1:
                    st.subheader("🔍 Model Comparison on Test Data")
                    
                    fig_comparison = go.Figure()
                    
                    # Actual prices
                    fig_comparison.add_trace(go.Scatter(
                        x=X_test.index,
                        y=y_test,
                        mode='lines',
                        name='Actual Price',
                        line=dict(color='black', width=2)
                    ))
                    
                    # Model predictions
                    for idx, (model_name, predictions) in enumerate(test_predictions.items()):
                        fig_comparison.add_trace(go.Scatter(
                            x=X_test.index,
                            y=predictions,
                            mode='lines',
                            name=f'{model_name} Prediction',
                            line=dict(color=colors[idx], width=1, dash='dot')
                        ))
                    
                    fig_comparison.update_layout(
                        title="Model Predictions vs Actual Prices (Test Data)",
                        xaxis_title="Date",
                        yaxis_title="Price ($)",
                        height=400
                    )
                    
                    st.plotly_chart(fig_comparison, use_container_width=True)
                
                # Feature importance (for Random Forest)
                if "Random Forest" in models:
                    st.subheader("🎯 Feature Importance (Random Forest)")
                    
                    rf_model = models["Random Forest"]
                    feature_importance = pd.DataFrame({
                        'Feature': X.columns,
                        'Importance': rf_model.feature_importances_
                    }).sort_values('Importance', ascending=False).head(15)
                    
                    fig_importance = go.Figure(go.Bar(
                        x=feature_importance['Importance'],
                        y=feature_importance['Feature'],
                        orientation='h',
                        marker_color='lightblue'
                    ))
                    
                    fig_importance.update_layout(
                        title="Top 15 Most Important Features",
                        xaxis_title="Importance",
                        height=400
                    )
                    
                    st.plotly_chart(fig_importance, use_container_width=True)
                
            else:
                st.error("Insufficient data for training. Please try a longer training period.")
        
        else:
            st.error("Unable to fetch stock data. Please check the ticker symbol.")

else:
    st.warning("Please enter a valid stock ticker symbol.")

# Disclaimer
st.markdown("---")
st.markdown("""
**⚠️ Disclaimer:** 
These predictions are generated using machine learning models trained on historical data. 
Stock market predictions are inherently uncertain and should not be used as the sole basis for investment decisions. 
Past performance does not guarantee future results. Please consult with financial professionals before making investment decisions.
""")
