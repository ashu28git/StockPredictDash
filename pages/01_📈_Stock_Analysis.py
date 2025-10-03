import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(page_title="Stock Analysis", page_icon="📈", layout="wide")

st.title("📈 Advanced Stock Analysis")
st.markdown("---")

# Sidebar controls
st.sidebar.header("Analysis Settings")
ticker_symbol = st.sidebar.text_input("Stock Ticker", value="AAPL").upper()
analysis_period = st.sidebar.selectbox("Analysis Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)

# Technical indicators settings
st.sidebar.subheader("Technical Indicators")
show_ma = st.sidebar.checkbox("Moving Averages", value=True)
ma_periods = st.sidebar.multiselect("MA Periods", [5, 10, 20, 50, 100, 200], default=[20, 50])
show_bollinger = st.sidebar.checkbox("Bollinger Bands", value=False)
show_rsi = st.sidebar.checkbox("RSI", value=True)
show_macd = st.sidebar.checkbox("MACD", value=False)

@st.cache_data(ttl=300)
def fetch_stock_data(symbol, period):
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        info = ticker.info
        return data, info
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None, None

def calculate_technical_indicators(data):
    """Calculate various technical indicators"""
    # Moving averages
    for period in ma_periods:
        if len(data) >= period:
            data[f'SMA_{period}'] = data['Close'].rolling(window=period).mean()
    
    # Bollinger Bands
    if show_bollinger:
        data['BB_Middle'] = data['Close'].rolling(window=20).mean()
        bb_std = data['Close'].rolling(window=20).std()
        data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
        data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
    
    # RSI
    if show_rsi:
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    if show_macd:
        exp1 = data['Close'].ewm(span=12).mean()
        exp2 = data['Close'].ewm(span=26).mean()
        data['MACD'] = exp1 - exp2
        data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
        data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
    
    return data

# Fetch data
if ticker_symbol:
    stock_data, stock_info = fetch_stock_data(ticker_symbol, analysis_period)
    
    if stock_data is not None and not stock_data.empty:
        # Calculate technical indicators
        stock_data = calculate_technical_indicators(stock_data)
        
        # Display stock info
        if stock_info:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Current Price", 
                    f"${stock_info.get('regularMarketPrice', stock_info.get('currentPrice', 0)):.2f}"
                )
            
            with col2:
                change_pct = stock_info.get('regularMarketChangePercent', 0)
                st.metric("Daily Change", f"{change_pct:.2f}%")
            
            with col3:
                volume = stock_info.get('regularMarketVolume', 0)
                st.metric("Volume", f"{volume:,}")
            
            with col4:
                market_cap = stock_info.get('marketCap', 0)
                if market_cap >= 1e12:
                    cap_display = f"${market_cap/1e12:.2f}T"
                elif market_cap >= 1e9:
                    cap_display = f"${market_cap/1e9:.2f}B"
                else:
                    cap_display = f"${market_cap/1e6:.2f}M"
                st.metric("Market Cap", cap_display)
        
        # Create subplot structure
        subplot_count = 1
        subplot_titles = [f'{ticker_symbol} Price Chart']
        
        if show_rsi:
            subplot_count += 1
            subplot_titles.append('RSI')
        
        if show_macd:
            subplot_count += 1
            subplot_titles.append('MACD')
        
        # Always add volume
        subplot_count += 1
        subplot_titles.append('Volume')
        
        fig = make_subplots(
            rows=subplot_count,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=subplot_titles,
            row_heights=[0.6] + [0.4/(subplot_count-1)]*(subplot_count-1) if subplot_count > 1 else [1]
        )
        
        # Main price chart
        fig.add_trace(go.Candlestick(
            x=stock_data.index,
            open=stock_data['Open'],
            high=stock_data['High'],
            low=stock_data['Low'],
            close=stock_data['Close'],
            name=ticker_symbol,
            showlegend=False
        ), row=1, col=1)
        
        # Add moving averages
        if show_ma:
            colors = ['orange', 'red', 'purple', 'brown', 'pink', 'gray']
            for i, period in enumerate(ma_periods):
                if f'SMA_{period}' in stock_data.columns:
                    fig.add_trace(go.Scatter(
                        x=stock_data.index,
                        y=stock_data[f'SMA_{period}'],
                        mode='lines',
                        name=f'SMA {period}',
                        line=dict(color=colors[i % len(colors)], width=1)
                    ), row=1, col=1)
        
        # Add Bollinger Bands
        if show_bollinger and 'BB_Upper' in stock_data.columns:
            fig.add_trace(go.Scatter(
                x=stock_data.index,
                y=stock_data['BB_Upper'],
                mode='lines',
                name='BB Upper',
                line=dict(color='gray', width=1, dash='dash')
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=stock_data.index,
                y=stock_data['BB_Lower'],
                mode='lines',
                name='BB Lower',
                line=dict(color='gray', width=1, dash='dash'),
                fill='tonexty',
                fillcolor='rgba(128,128,128,0.1)'
            ), row=1, col=1)
        
        current_row = 2
        
        # RSI subplot
        if show_rsi and 'RSI' in stock_data.columns:
            fig.add_trace(go.Scatter(
                x=stock_data.index,
                y=stock_data['RSI'],
                mode='lines',
                name='RSI',
                line=dict(color='purple')
            ), row=current_row, col=1)
            
            # Add RSI reference lines
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=current_row, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=current_row, col=1)
            fig.add_hline(y=50, line_dash="dot", line_color="gray", row=current_row, col=1)
            
            current_row += 1
        
        # MACD subplot
        if show_macd and 'MACD' in stock_data.columns:
            fig.add_trace(go.Scatter(
                x=stock_data.index,
                y=stock_data['MACD'],
                mode='lines',
                name='MACD',
                line=dict(color='blue')
            ), row=current_row, col=1)
            
            fig.add_trace(go.Scatter(
                x=stock_data.index,
                y=stock_data['MACD_Signal'],
                mode='lines',
                name='Signal',
                line=dict(color='red')
            ), row=current_row, col=1)
            
            fig.add_trace(go.Bar(
                x=stock_data.index,
                y=stock_data['MACD_Histogram'],
                name='Histogram',
                marker_color='gray',
                opacity=0.7
            ), row=current_row, col=1)
            
            current_row += 1
        
        # Volume subplot
        fig.add_trace(go.Bar(
            x=stock_data.index,
            y=stock_data['Volume'],
            name='Volume',
            marker_color='rgba(158,202,225,0.5)',
            showlegend=False
        ), row=current_row, col=1)
        
        fig.update_layout(
            title=f"{ticker_symbol} Technical Analysis",
            xaxis_rangeslider_visible=False,
            height=800,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Analysis summary
        st.subheader("📊 Analysis Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Technical Indicators")
            
            # Current values
            current_price = stock_data['Close'].iloc[-1]
            
            if show_ma and ma_periods:
                st.markdown("**Moving Averages:**")
                for period in ma_periods:
                    if f'SMA_{period}' in stock_data.columns:
                        ma_value = stock_data[f'SMA_{period}'].iloc[-1]
                        if not pd.isna(ma_value):
                            trend = "↗️" if current_price > ma_value else "↘️"
                            st.write(f"SMA {period}: ${ma_value:.2f} {trend}")
            
            if show_rsi and 'RSI' in stock_data.columns:
                current_rsi = stock_data['RSI'].iloc[-1]
                if not pd.isna(current_rsi):
                    rsi_signal = "Overbought" if current_rsi > 70 else "Oversold" if current_rsi < 30 else "Neutral"
                    st.write(f"**RSI:** {current_rsi:.2f} ({rsi_signal})")
        
        with col2:
            st.markdown("### Price Statistics")
            
            period_high = stock_data['High'].max()
            period_low = stock_data['Low'].min()
            price_change = ((current_price - stock_data['Close'].iloc[0]) / stock_data['Close'].iloc[0]) * 100
            
            st.write(f"**Period High:** ${period_high:.2f}")
            st.write(f"**Period Low:** ${period_low:.2f}")
            st.write(f"**Period Change:** {price_change:+.2f}%")
            st.write(f"**Average Volume:** {stock_data['Volume'].mean():,.0f}")
            
            # Volatility
            volatility = stock_data['Close'].pct_change().std() * np.sqrt(252) * 100
            st.write(f"**Annualized Volatility:** {volatility:.2f}%")
        
        # Recent data table
        st.subheader("📋 Recent Trading Data")
        
        # Select columns to display
        display_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if show_ma and ma_periods:
            display_columns.extend([f'SMA_{period}' for period in ma_periods if f'SMA_{period}' in stock_data.columns])
        
        recent_data = stock_data[display_columns].tail(10).round(2)
        recent_data.index = recent_data.index.strftime('%Y-%m-%d')
        st.dataframe(recent_data, use_container_width=True)
        
    else:
        st.error("Unable to fetch stock data. Please check the ticker symbol.")

else:
    st.warning("Please enter a valid stock ticker symbol.")

# Footer
st.markdown("---")
st.markdown("*Technical analysis is based on historical price data and should be used in conjunction with other forms of analysis.*")
