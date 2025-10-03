import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Prediction Comparison", page_icon="📊", layout="wide")

st.title("📊 Prediction vs Actual Price Comparison")
st.markdown("---")

# Sidebar controls
st.sidebar.header("Comparison Settings")
ticker_symbol = st.sidebar.text_input("Stock Ticker", value="AAPL").upper()

# Backtesting parameters
st.sidebar.subheader("Backtesting Parameters")
backtest_period = st.sidebar.selectbox("Backtesting Period", ["1mo", "3mo", "6mo", "1y"], index=2)
prediction_horizon = st.sidebar.selectbox("Prediction Horizon", [1, 3, 5, 7, 10], index=2)
training_window = st.sidebar.selectbox("Training Window", ["3mo", "6mo", "1y", "2y"], index=2)

# Model selection
models_to_compare = st.sidebar.multiselect(
    "Models to Compare",
    ["Random Forest", "Linear Regression", "Simple Moving Average", "Exponential Moving Average"],
    default=["Random Forest", "Linear Regression"]
)

@st.cache_data(ttl=300)
def fetch_extended_data(symbol, period):
    """Fetch extended historical data for backtesting"""
    try:
        ticker = yf.Ticker(symbol)
        # Get more data than requested to allow for training windows
        extended_periods = {
            "1mo": "4mo",
            "3mo": "1y", 
            "6mo": "2y",
            "1y": "3y"
        }
        data = ticker.history(period=extended_periods.get(period, "2y"))
        return data
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

def create_features_for_prediction(data, lookback_window=20):
    """Create features for ML models"""
    features_df = pd.DataFrame(index=data.index)
    
    # Basic price features
    features_df['Close'] = data['Close']
    features_df['Open'] = data['Open']
    features_df['High'] = data['High']
    features_df['Low'] = data['Low']
    features_df['Volume'] = data['Volume']
    
    # Price-based features
    features_df['Price_Change'] = data['Close'].pct_change()
    features_df['Price_Range'] = (data['High'] - data['Low']) / data['Close']
    features_df['Open_Close_Ratio'] = data['Open'] / data['Close']
    features_df['Volume_Change'] = data['Volume'].pct_change()
    
    # Technical indicators
    # Moving averages
    for window in [5, 10, 20]:
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
    
    # Volatility
    features_df['Volatility'] = data['Close'].rolling(window=10).std()
    
    # Lag features
    for lag in range(1, 6):
        features_df[f'Close_Lag_{lag}'] = features_df['Close'].shift(lag)
        features_df[f'Price_Change_Lag_{lag}'] = features_df['Price_Change'].shift(lag)
    
    return features_df

def backtest_predictions(data, training_window_days, prediction_horizon, models_to_test):
    """Perform backtesting to compare model predictions with actual prices"""
    
    # Convert training window to days
    window_mapping = {"3mo": 90, "6mo": 180, "1y": 365, "2y": 730}
    training_days = window_mapping.get(training_window, 365)
    
    # Determine backtesting period
    backtest_start = len(data) - 252  # Start backtesting from 1 year ago
    if backtest_start < training_days + 50:  # Ensure enough training data
        backtest_start = training_days + 50
    
    results = {}
    predictions_data = []
    
    # Initialize models
    scaler = StandardScaler()
    
    for i in range(backtest_start, len(data) - prediction_horizon, 5):  # Step by 5 days
        current_date = data.index[i]
        actual_future_price = data['Close'].iloc[i + prediction_horizon]
        
        # Training data
        train_start = max(0, i - training_days)
        train_data = data.iloc[train_start:i]
        
        if len(train_data) < 50:  # Skip if insufficient training data
            continue
        
        # Create features
        features = create_features_for_prediction(train_data)
        features_clean = features.dropna()
        
        if len(features_clean) < 30:  # Skip if insufficient clean data
            continue
        
        # Prepare target
        target = features_clean['Close'].shift(-1).dropna()
        X_train = features_clean.iloc[:-1]
        y_train = target
        
        # Current features for prediction
        current_features = features.iloc[-1:].dropna(axis=1)
        
        prediction_results = {"date": current_date, "actual": actual_future_price}
        
        # Random Forest
        if "Random Forest" in models_to_test:
            try:
                rf_model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=1)
                rf_model.fit(X_train, y_train)
                
                # Get common columns
                common_cols = X_train.columns.intersection(current_features.columns)
                rf_pred = rf_model.predict(current_features[common_cols])[0]
                prediction_results["Random Forest"] = rf_pred
            except Exception:
                pass
        
        # Linear Regression
        if "Linear Regression" in models_to_test:
            try:
                lr_model = LinearRegression()
                X_train_scaled = scaler.fit_transform(X_train)
                lr_model.fit(X_train_scaled, y_train)
                
                common_cols = X_train.columns.intersection(current_features.columns)
                X_pred_scaled = scaler.transform(current_features[common_cols])
                lr_pred = lr_model.predict(X_pred_scaled)[0]
                prediction_results["Linear Regression"] = lr_pred
            except Exception:
                pass
        
        # Simple Moving Average
        if "Simple Moving Average" in models_to_test:
            sma_pred = train_data['Close'].tail(20).mean()
            prediction_results["Simple Moving Average"] = sma_pred
        
        # Exponential Moving Average
        if "Exponential Moving Average" in models_to_test:
            ema_pred = train_data['Close'].ewm(span=20).mean().iloc[-1]
            prediction_results["Exponential Moving Average"] = ema_pred
        
        predictions_data.append(prediction_results)
    
    return pd.DataFrame(predictions_data)

def calculate_accuracy_metrics(predictions_df, model_names):
    """Calculate accuracy metrics for each model"""
    metrics = {}
    
    for model in model_names:
        if model in predictions_df.columns:
            actual = predictions_df['actual']
            predicted = predictions_df[model]
            
            # Remove any NaN values
            mask = ~(pd.isna(actual) | pd.isna(predicted))
            actual_clean = actual[mask]
            predicted_clean = predicted[mask]
            
            if len(actual_clean) > 0:
                mse = mean_squared_error(actual_clean, predicted_clean)
                mae = mean_absolute_error(actual_clean, predicted_clean)
                rmse = np.sqrt(mse)
                
                # Calculate MAPE (Mean Absolute Percentage Error)
                mape = np.mean(np.abs((actual_clean - predicted_clean) / actual_clean)) * 100
                
                # Calculate directional accuracy (percentage of correct trend predictions)
                actual_changes = np.diff(actual_clean)
                predicted_changes = np.diff(predicted_clean)
                directional_accuracy = np.mean(np.sign(actual_changes) == np.sign(predicted_changes)) * 100
                
                metrics[model] = {
                    'MAE': mae,
                    'MSE': mse,
                    'RMSE': rmse,
                    'MAPE': mape,
                    'Directional_Accuracy': directional_accuracy,
                    'Predictions': len(actual_clean)
                }
    
    return metrics

# Main execution
if ticker_symbol and models_to_compare:
    with st.spinner("Performing backtesting analysis..."):
        # Fetch data
        stock_data = fetch_extended_data(ticker_symbol, backtest_period)
        
        if stock_data is not None and not stock_data.empty:
            # Display basic info
            st.subheader("📈 Stock Information")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Current Price", f"${stock_data['Close'].iloc[-1]:.2f}")
            
            with col2:
                price_change = stock_data['Close'].pct_change().iloc[-1] * 100
                st.metric("Daily Change", f"{price_change:.2f}%")
            
            with col3:
                st.metric("Prediction Horizon", f"{prediction_horizon} days")
            
            with col4:
                st.metric("Data Points", len(stock_data))
            
            # Perform backtesting
            predictions_df = backtest_predictions(
                stock_data, training_window, prediction_horizon, models_to_compare
            )
            
            if not predictions_df.empty and len(predictions_df) > 5:
                # Calculate accuracy metrics
                metrics = calculate_accuracy_metrics(predictions_df, models_to_compare)
                
                # Display accuracy metrics
                st.subheader("🎯 Model Accuracy Metrics")
                
                if metrics:
                    metrics_df = pd.DataFrame(metrics).T
                    metrics_df = metrics_df.round(4)
                    
                    # Sort by MAPE (lower is better)
                    metrics_df = metrics_df.sort_values('MAPE')
                    
                    st.dataframe(metrics_df, use_container_width=True)
                    
                    # Metrics explanation
                    with st.expander("📚 Metrics Explanation"):
                        st.markdown("""
                        - **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual prices
                        - **RMSE (Root Mean Square Error)**: Square root of average squared differences
                        - **MAPE (Mean Absolute Percentage Error)**: Average percentage error (lower is better)
                        - **Directional Accuracy**: Percentage of correct trend predictions (higher is better)
                        - **Predictions**: Number of predictions made during backtesting
                        """)
                
                # Visualization: Predictions vs Actual
                st.subheader("📊 Predictions vs Actual Prices")
                
                fig = go.Figure()
                
                # Actual prices
                fig.add_trace(go.Scatter(
                    x=predictions_df['date'],
                    y=predictions_df['actual'],
                    mode='lines+markers',
                    name='Actual Price',
                    line=dict(color='black', width=3),
                    marker=dict(size=4)
                ))
                
                # Model predictions
                colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
                for idx, model in enumerate(models_to_compare):
                    if model in predictions_df.columns:
                        fig.add_trace(go.Scatter(
                            x=predictions_df['date'],
                            y=predictions_df[model],
                            mode='lines+markers',
                            name=f'{model} Prediction',
                            line=dict(color=colors[idx % len(colors)], width=2, dash='dash'),
                            marker=dict(size=3)
                        ))
                
                fig.update_layout(
                    title=f"{ticker_symbol} - Predicted vs Actual Prices ({prediction_horizon}-day horizon)",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    height=600,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Error analysis
                st.subheader("📉 Error Analysis")
                
                fig_errors = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('Absolute Errors', 'Percentage Errors', 'Error Distribution', 'Cumulative Errors'),
                    specs=[[{"secondary_y": False}, {"secondary_y": False}],
                           [{"secondary_y": False}, {"secondary_y": False}]]
                )
                
                for idx, model in enumerate(models_to_compare):
                    if model in predictions_df.columns:
                        color = colors[idx % len(colors)]
                        
                        # Calculate errors
                        absolute_errors = np.abs(predictions_df['actual'] - predictions_df[model])
                        percentage_errors = (absolute_errors / predictions_df['actual']) * 100
                        
                        # Absolute errors
                        fig_errors.add_trace(
                            go.Scatter(x=predictions_df['date'], y=absolute_errors,
                                     mode='lines', name=f'{model} Abs Error',
                                     line=dict(color=color)),
                            row=1, col=1
                        )
                        
                        # Percentage errors
                        fig_errors.add_trace(
                            go.Scatter(x=predictions_df['date'], y=percentage_errors,
                                     mode='lines', name=f'{model} % Error',
                                     line=dict(color=color)),
                            row=1, col=2
                        )
                        
                        # Error distribution
                        fig_errors.add_trace(
                            go.Histogram(x=percentage_errors, name=f'{model} Error Dist',
                                       marker_color=color, opacity=0.7, nbinsx=20),
                            row=2, col=1
                        )
                        
                        # Cumulative errors
                        cumulative_errors = np.cumsum(absolute_errors)
                        fig_errors.add_trace(
                            go.Scatter(x=predictions_df['date'], y=cumulative_errors,
                                     mode='lines', name=f'{model} Cumulative',
                                     line=dict(color=color)),
                            row=2, col=2
                        )
                
                fig_errors.update_layout(height=800, showlegend=True)
                fig_errors.update_xaxes(title_text="Date", row=1, col=1)
                fig_errors.update_xaxes(title_text="Date", row=1, col=2)
                fig_errors.update_xaxes(title_text="Percentage Error", row=2, col=1)
                fig_errors.update_xaxes(title_text="Date", row=2, col=2)
                fig_errors.update_yaxes(title_text="Absolute Error ($)", row=1, col=1)
                fig_errors.update_yaxes(title_text="Percentage Error (%)", row=1, col=2)
                fig_errors.update_yaxes(title_text="Frequency", row=2, col=1)
                fig_errors.update_yaxes(title_text="Cumulative Error ($)", row=2, col=2)
                
                st.plotly_chart(fig_errors, use_container_width=True)
                
                # Best performing model
                if metrics:
                    best_model = min(metrics.keys(), key=lambda x: metrics[x]['MAPE'])
                    st.success(f"🏆 Best performing model: **{best_model}** (MAPE: {metrics[best_model]['MAPE']:.2f}%)")
                
                # Raw predictions data
                with st.expander("📋 Raw Predictions Data"):
                    display_df = predictions_df.copy()
                    display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
                    st.dataframe(display_df.round(2), use_container_width=True)
                
            else:
                st.error("Insufficient data for backtesting. Please try a shorter backtesting period or longer training window.")
        
        else:
            st.error("Unable to fetch stock data. Please check the ticker symbol.")

else:
    st.warning("Please enter a valid stock ticker symbol and select at least one model to compare.")

# Information section
st.markdown("---")
st.markdown("""
### 📚 About This Analysis

This backtesting analysis compares different prediction models by:

1. **Training models** on historical data using a sliding window approach
2. **Making predictions** for future price movements
3. **Comparing predictions** with actual historical prices
4. **Calculating accuracy metrics** to evaluate model performance

**Key Insights:**
- Lower MAPE values indicate more accurate price predictions
- Higher directional accuracy means better trend prediction
- RMSE penalizes large errors more than MAE
- Consider both price accuracy and trend accuracy for trading decisions

**⚠️ Important:** Past performance does not guarantee future results. Use this analysis as one factor among many in your investment decisions.
""")
