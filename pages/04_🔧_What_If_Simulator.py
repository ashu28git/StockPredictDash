import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="What-If Simulator", page_icon="🔧", layout="wide")

st.title("🔧 What-If Scenario Simulator")
st.markdown("Explore how different parameters affect stock price predictions")
st.markdown("---")

# Sidebar controls
st.sidebar.header("Simulation Settings")
ticker_symbol = st.sidebar.text_input("Stock Ticker", value="AAPL").upper()

st.sidebar.subheader("📊 Moving Average Settings")
short_ma_period = st.sidebar.slider("Short MA Period", 5, 50, 20, step=5)
long_ma_period = st.sidebar.slider("Long MA Period", 50, 200, 50, step=10)
rsi_period = st.sidebar.slider("RSI Period", 10, 30, 14, step=2)

st.sidebar.subheader("🤖 ML Model Settings")
prediction_days = st.sidebar.slider("Prediction Horizon", 1, 30, 7)
training_period = st.sidebar.selectbox("Training Data Period", ["6mo", "1y", "2y", "3y"], index=1)
lookback_window = st.sidebar.slider("Feature Lookback Window", 5, 30, 15)

st.sidebar.subheader("📈 Scenario Parameters")
volatility_adjustment = st.sidebar.slider("Volatility Adjustment", 0.5, 2.0, 1.0, step=0.1)
volume_boost = st.sidebar.slider("Volume Boost Factor", 0.5, 3.0, 1.0, step=0.1)
trend_strength = st.sidebar.slider("Trend Strength", 0.1, 2.0, 1.0, step=0.1)

@st.cache_data(ttl=300)
def fetch_stock_data(symbol, period):
    """Fetch stock data"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        return data
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

def calculate_technical_indicators(data, short_ma, long_ma, rsi_period):
    """Calculate technical indicators with custom parameters"""
    df = data.copy()
    
    # Moving averages
    df[f'SMA_{short_ma}'] = df['Close'].rolling(window=short_ma).mean()
    df[f'SMA_{long_ma}'] = df['Close'].rolling(window=long_ma).mean()
    
    # Golden Cross and Death Cross signals
    df['MA_Signal'] = np.where(df[f'SMA_{short_ma}'] > df[f'SMA_{long_ma}'], 1, 0)
    df['MA_CrossOver'] = df['MA_Signal'].diff()
    
    # RSI with custom period
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    df[f'RSI_{rsi_period}'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    bb_period = 20
    df['BB_Middle'] = df['Close'].rolling(window=bb_period).mean()
    bb_std = df['Close'].rolling(window=bb_period).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    
    # MACD
    exp1 = df['Close'].ewm(span=12).mean()
    exp2 = df['Close'].ewm(span=26).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
    # Price momentum
    df['Price_Change'] = df['Close'].pct_change()
    df['Price_Change_5d'] = df['Close'].pct_change(5)
    df['Price_Change_10d'] = df['Close'].pct_change(10)
    
    # Volume indicators
    df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
    
    # Volatility
    df['Volatility'] = df['Close'].rolling(window=20).std()
    df['ATR'] = calculate_atr(df)
    
    return df

def calculate_atr(df, period=14):
    """Calculate Average True Range"""
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = true_range.rolling(window=period).mean()
    
    return atr

def apply_scenario_adjustments(data, vol_adj, vol_boost, trend_str):
    """Apply what-if scenario adjustments to the data"""
    adjusted_data = data.copy()
    
    # Adjust volatility
    if vol_adj != 1.0:
        mean_price = adjusted_data['Close'].mean()
        adjusted_data['Close'] = mean_price + (adjusted_data['Close'] - mean_price) * vol_adj
        adjusted_data['High'] = adjusted_data['Close'] + (adjusted_data['High'] - data['Close']) * vol_adj
        adjusted_data['Low'] = adjusted_data['Close'] + (adjusted_data['Low'] - data['Close']) * vol_adj
        adjusted_data['Open'] = adjusted_data['Close'].shift(1).fillna(adjusted_data['Close'])
    
    # Adjust volume
    if vol_boost != 1.0:
        adjusted_data['Volume'] = adjusted_data['Volume'] * vol_boost
    
    # Apply trend strength
    if trend_str != 1.0:
        # Calculate trend
        trend = adjusted_data['Close'].rolling(window=20).mean().diff()
        trend_adjustment = trend * (trend_str - 1.0)
        adjusted_data['Close'] = adjusted_data['Close'] + trend_adjustment.fillna(0)
    
    return adjusted_data

def create_ml_features(data, lookback):
    """Create features for ML models"""
    features_df = pd.DataFrame(index=data.index)
    
    # Basic price features
    features_df['Close'] = data['Close']
    features_df['Open'] = data['Open']
    features_df['High'] = data['High']
    features_df['Low'] = data['Low']
    features_df['Volume'] = data['Volume']
    
    # Technical indicators (using available columns)
    available_columns = data.columns
    
    if 'SMA_20' in available_columns:
        features_df['SMA_20'] = data['SMA_20']
        features_df['Price_SMA_Ratio'] = data['Close'] / data['SMA_20']
    
    if 'RSI_14' in available_columns:
        features_df['RSI'] = data['RSI_14']
    
    if 'MACD' in available_columns:
        features_df['MACD'] = data['MACD']
        features_df['MACD_Signal'] = data['MACD_Signal']
    
    if 'BB_Position' in available_columns:
        features_df['BB_Position'] = data['BB_Position']
    
    if 'Volume_Ratio' in available_columns:
        features_df['Volume_Ratio'] = data['Volume_Ratio']
    
    if 'Volatility' in available_columns:
        features_df['Volatility'] = data['Volatility']
    
    # Price-based features
    features_df['Price_Change'] = data['Close'].pct_change()
    features_df['Price_Range'] = (data['High'] - data['Low']) / data['Close']
    
    # Lag features
    for lag in range(1, min(lookback + 1, 8)):
        features_df[f'Close_Lag_{lag}'] = features_df['Close'].shift(lag)
        features_df[f'Price_Change_Lag_{lag}'] = features_df['Price_Change'].shift(lag)
    
    return features_df

def train_and_predict(features_df, prediction_days):
    """Train models and make predictions"""
    # Prepare data
    target = features_df['Close'].shift(-1)
    
    # Remove NaN values
    clean_data = features_df.dropna()
    clean_target = target.dropna()
    
    # Align data
    common_index = clean_data.index.intersection(clean_target.index)
    X = clean_data.loc[common_index]
    y = clean_target.loc[common_index]
    
    if len(X) < 50:
        return None, None
    
    # Split data (use recent 20% for testing)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    predictions = {}
    
    # Random Forest
    try:
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf_model.fit(X_train, y_train)
        
        # Generate future predictions
        future_predictions = []
        last_features = X.iloc[-1:].copy()
        
        for day in range(prediction_days):
            pred = rf_model.predict(last_features)[0]
            future_predictions.append(pred)
            
            # Update features for next prediction (simplified)
            last_features['Close'] = pred
            for lag in range(1, 8):
                if f'Close_Lag_{lag}' in last_features.columns:
                    if lag == 1:
                        last_features[f'Close_Lag_{lag}'] = pred
                    elif f'Close_Lag_{lag-1}' in last_features.columns:
                        last_features[f'Close_Lag_{lag}'] = last_features[f'Close_Lag_{lag-1}']
        
        predictions['Random Forest'] = future_predictions
        
    except Exception as e:
        st.warning(f"Random Forest training failed: {str(e)}")
    
    # Linear Regression
    try:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        lr_model = LinearRegression()
        lr_model.fit(X_train_scaled, y_train)
        
        # Generate future predictions
        future_predictions = []
        last_features = X.iloc[-1:].copy()
        
        for day in range(prediction_days):
            last_features_scaled = scaler.transform(last_features)
            pred = lr_model.predict(last_features_scaled)[0]
            future_predictions.append(pred)
            
            # Update features
            last_features['Close'] = pred
            for lag in range(1, 8):
                if f'Close_Lag_{lag}' in last_features.columns:
                    if lag == 1:
                        last_features[f'Close_Lag_{lag}'] = pred
                    elif f'Close_Lag_{lag-1}' in last_features.columns:
                        last_features[f'Close_Lag_{lag}'] = last_features[f'Close_Lag_{lag-1}']
        
        predictions['Linear Regression'] = future_predictions
        
    except Exception as e:
        st.warning(f"Linear Regression training failed: {str(e)}")
    
    return predictions, (X_test, y_test)

# Main execution
if ticker_symbol:
    with st.spinner("Running what-if simulation..."):
        # Fetch data
        stock_data = fetch_stock_data(ticker_symbol, training_period)
        
        if stock_data is not None and not stock_data.empty:
            # Display current settings
            st.subheader("🎛️ Current Simulation Settings")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Technical Indicators:**")
                st.write(f"• Short MA: {short_ma_period} days")
                st.write(f"• Long MA: {long_ma_period} days")
                st.write(f"• RSI Period: {rsi_period} days")
            
            with col2:
                st.markdown("**ML Parameters:**")
                st.write(f"• Prediction Horizon: {prediction_days} days")
                st.write(f"• Lookback Window: {lookback_window} days")
                st.write(f"• Training Period: {training_period}")
            
            with col3:
                st.markdown("**Scenario Adjustments:**")
                st.write(f"• Volatility: {volatility_adjustment:.1f}x")
                st.write(f"• Volume: {volume_boost:.1f}x")
                st.write(f"• Trend Strength: {trend_strength:.1f}x")
            
            # Calculate technical indicators
            stock_data_with_indicators = calculate_technical_indicators(
                stock_data, short_ma_period, long_ma_period, rsi_period
            )
            
            # Apply scenario adjustments
            adjusted_data = apply_scenario_adjustments(
                stock_data_with_indicators, volatility_adjustment, volume_boost, trend_strength
            )
            
            # Recalculate indicators on adjusted data
            adjusted_data = calculate_technical_indicators(
                adjusted_data, short_ma_period, long_ma_period, rsi_period
            )
            
            # Create ML features
            features = create_ml_features(adjusted_data, lookback_window)
            
            # Train models and get predictions
            predictions, test_data = train_and_predict(features, prediction_days)
            
            # Comparison visualization
            st.subheader("📊 Original vs Adjusted Data Comparison")
            
            # Show last 60 days for comparison
            comparison_period = min(60, len(stock_data))
            original_recent = stock_data.tail(comparison_period)
            adjusted_recent = adjusted_data.tail(comparison_period)
            
            fig_comparison = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=('Price Comparison', 'Volume Comparison', 'Technical Indicators'),
                row_heights=[0.5, 0.25, 0.25]
            )
            
            # Price comparison
            fig_comparison.add_trace(go.Scatter(
                x=original_recent.index,
                y=original_recent['Close'],
                mode='lines',
                name='Original Price',
                line=dict(color='blue', width=2)
            ), row=1, col=1)
            
            fig_comparison.add_trace(go.Scatter(
                x=adjusted_recent.index,
                y=adjusted_recent['Close'],
                mode='lines',
                name='Adjusted Price',
                line=dict(color='red', width=2, dash='dash')
            ), row=1, col=1)
            
            # Add moving averages
            fig_comparison.add_trace(go.Scatter(
                x=adjusted_recent.index,
                y=adjusted_recent[f'SMA_{short_ma_period}'],
                mode='lines',
                name=f'SMA {short_ma_period}',
                line=dict(color='orange', width=1)
            ), row=1, col=1)
            
            fig_comparison.add_trace(go.Scatter(
                x=adjusted_recent.index,
                y=adjusted_recent[f'SMA_{long_ma_period}'],
                mode='lines',
                name=f'SMA {long_ma_period}',
                line=dict(color='green', width=1)
            ), row=1, col=1)
            
            # Volume comparison
            fig_comparison.add_trace(go.Bar(
                x=original_recent.index,
                y=original_recent['Volume'],
                name='Original Volume',
                marker_color='lightblue',
                opacity=0.7
            ), row=2, col=1)
            
            fig_comparison.add_trace(go.Bar(
                x=adjusted_recent.index,
                y=adjusted_recent['Volume'],
                name='Adjusted Volume',
                marker_color='lightcoral',
                opacity=0.7
            ), row=2, col=1)
            
            # RSI comparison
            fig_comparison.add_trace(go.Scatter(
                x=original_recent.index,
                y=calculate_technical_indicators(original_recent, short_ma_period, long_ma_period, rsi_period)[f'RSI_{rsi_period}'],
                mode='lines',
                name='Original RSI',
                line=dict(color='purple', width=2)
            ), row=3, col=1)
            
            fig_comparison.add_trace(go.Scatter(
                x=adjusted_recent.index,
                y=adjusted_recent[f'RSI_{rsi_period}'],
                mode='lines',
                name='Adjusted RSI',
                line=dict(color='orange', width=2, dash='dash')
            ), row=3, col=1)
            
            # Add RSI reference lines
            fig_comparison.add_hline(y=70, line_dash="dot", line_color="red", row=3, col=1)
            fig_comparison.add_hline(y=30, line_dash="dot", line_color="green", row=3, col=1)
            
            fig_comparison.update_layout(
                height=800,
                title="Impact of Parameter Changes",
                showlegend=True
            )
            
            st.plotly_chart(fig_comparison, use_container_width=True)
            
            # Show predictions if available
            if predictions:
                st.subheader("🔮 Predictions Under Current Scenario")
                
                # Create prediction visualization
                fig_pred = go.Figure()
                
                # Recent historical data
                recent_data = adjusted_data.tail(30)
                fig_pred.add_trace(go.Scatter(
                    x=recent_data.index,
                    y=recent_data['Close'],
                    mode='lines',
                    name='Historical (Adjusted)',
                    line=dict(color='blue', width=2)
                ))
                
                # Future dates
                last_date = adjusted_data.index[-1]
                future_dates = [last_date + timedelta(days=i+1) for i in range(prediction_days)]
                
                # Predictions
                colors = ['red', 'green']
                for idx, (model_name, pred_values) in enumerate(predictions.items()):
                    fig_pred.add_trace(go.Scatter(
                        x=future_dates,
                        y=pred_values,
                        mode='lines+markers',
                        name=f'{model_name} Prediction',
                        line=dict(color=colors[idx], width=2, dash='dash'),
                        marker=dict(size=6)
                    ))
                
                fig_pred.update_layout(
                    title=f"Price Predictions with Current Settings",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    height=500
                )
                
                st.plotly_chart(fig_pred, use_container_width=True)
                
                # Predictions table
                st.subheader("📋 Detailed Predictions")
                
                pred_df = pd.DataFrame(index=future_dates)
                for model_name, pred_values in predictions.items():
                    pred_df[model_name] = pred_values
                
                pred_df.index = pred_df.index.strftime('%Y-%m-%d')
                st.dataframe(pred_df.round(2), use_container_width=True)
            
            # Trading signals based on current settings
            st.subheader("📈 Trading Signals")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # MA Signal
                current_short_ma = adjusted_data[f'SMA_{short_ma_period}'].iloc[-1]
                current_long_ma = adjusted_data[f'SMA_{long_ma_period}'].iloc[-1]
                ma_signal = "🟢 BULLISH" if current_short_ma > current_long_ma else "🔴 BEARISH"
                st.metric("Moving Average Signal", ma_signal)
            
            with col2:
                # RSI Signal
                current_rsi = adjusted_data[f'RSI_{rsi_period}'].iloc[-1]
                if current_rsi > 70:
                    rsi_signal = "🔴 OVERBOUGHT"
                elif current_rsi < 30:
                    rsi_signal = "🟢 OVERSOLD"
                else:
                    rsi_signal = "🟡 NEUTRAL"
                st.metric(f"RSI ({rsi_period}d) Signal", rsi_signal)
                st.caption(f"Value: {current_rsi:.1f}")
            
            with col3:
                # Bollinger Band Signal
                current_bb_pos = adjusted_data['BB_Position'].iloc[-1]
                if current_bb_pos > 0.8:
                    bb_signal = "🔴 UPPER BAND"
                elif current_bb_pos < 0.2:
                    bb_signal = "🟢 LOWER BAND"
                else:
                    bb_signal = "🟡 MIDDLE RANGE"
                st.metric("Bollinger Band Signal", bb_signal)
                st.caption(f"Position: {current_bb_pos:.2f}")
            
            # Parameter impact analysis
            st.subheader("🎯 Parameter Impact Analysis")
            
            with st.expander("📊 How Parameters Affect Predictions"):
                impact_col1, impact_col2 = st.columns(2)
                
                with impact_col1:
                    st.markdown("**Moving Average Periods:**")
                    st.write(f"• Short MA ({short_ma_period}d): Faster response to price changes")
                    st.write(f"• Long MA ({long_ma_period}d): Smoother trend indication")
                    st.write("• Smaller differences = More sensitive signals")
                    
                    st.markdown("**Volatility Adjustment:**")
                    st.write(f"• Current setting: {volatility_adjustment:.1f}x")
                    if volatility_adjustment > 1:
                        st.write("• Increased volatility → Wider price swings")
                    elif volatility_adjustment < 1:
                        st.write("• Decreased volatility → Smoother price movement")
                    else:
                        st.write("• Normal volatility maintained")
                
                with impact_col2:
                    st.markdown("**Volume Boost:**")
                    st.write(f"• Current setting: {volume_boost:.1f}x")
                    if volume_boost > 1:
                        st.write("• Higher volume → Stronger price movements")
                    elif volume_boost < 1:
                        st.write("• Lower volume → Weaker price signals")
                    
                    st.markdown("**Trend Strength:**")
                    st.write(f"• Current setting: {trend_strength:.1f}x")
                    if trend_strength > 1:
                        st.write("• Enhanced trends → Stronger directional moves")
                    elif trend_strength < 1:
                        st.write("• Dampened trends → More sideways movement")
        
        else:
            st.error("Unable to fetch stock data. Please check the ticker symbol.")

else:
    st.warning("Please enter a valid stock ticker symbol.")

# Usage guide
st.markdown("---")
st.markdown("""
### 🎮 How to Use the What-If Simulator

1. **Adjust Parameters**: Use the sidebar controls to modify technical indicator periods and scenario settings
2. **Observe Changes**: Watch how the adjustments affect the historical data and predictions
3. **Compare Signals**: Review how different MA periods change the trading signals
4. **Analyze Impact**: Use the parameter impact analysis to understand the effects of your changes
5. **Test Scenarios**: Try extreme settings to see how robust your trading strategy might be

**💡 Pro Tips:**
- Shorter MA periods = More signals but more false positives
- Higher volatility settings test strategy resilience
- Volume adjustments simulate different market conditions
- Trend strength modifications help test range vs trending markets
""")
