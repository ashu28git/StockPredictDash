import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Stock Market Prediction Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main dashboard title
st.title("📈 Stock Market Prediction Dashboard")
st.markdown("---")

# Sidebar for stock selection
st.sidebar.header("Stock Selection")
ticker_symbol = st.sidebar.text_input(
    "Enter Stock Ticker Symbol", 
    value="AAPL",
    help="Enter a valid stock ticker (e.g., AAPL, TSLA, MSFT, GOOGL)"
).upper()

# Helper functions
def calculate_rsi(prices, window=14):
    """Calculate Relative Strength Index (RSI)"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Validate ticker symbol
@st.cache_data(ttl=300)  # Cache for 5 minutes
def validate_and_fetch_stock_info(symbol):
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        if 'regularMarketPrice' in info or 'currentPrice' in info:
            return info
        else:
            return None
    except Exception as e:
        return None

# Main content area
col1, col2, col3 = st.columns([2, 2, 1])

if ticker_symbol:
    stock_info = validate_and_fetch_stock_info(ticker_symbol)
    
    if stock_info:
        # Display stock information
        with col1:
            st.subheader(f"{ticker_symbol} - {stock_info.get('longName', 'N/A')}")
            
        with col2:
            current_price = stock_info.get('regularMarketPrice') or stock_info.get('currentPrice')
            if current_price:
                st.metric(
                    label="Current Price",
                    value=f"${current_price:.2f}",
                    delta=f"{stock_info.get('regularMarketChangePercent', 0):.2f}%"
                )
        
        with col3:
            market_cap = stock_info.get('marketCap')
            if market_cap:
                if market_cap >= 1e12:
                    market_cap_str = f"${market_cap/1e12:.2f}T"
                elif market_cap >= 1e9:
                    market_cap_str = f"${market_cap/1e9:.2f}B"
                else:
                    market_cap_str = f"${market_cap/1e6:.2f}M"
                st.metric("Market Cap", market_cap_str)
        
        # Fetch historical data
        @st.cache_data(ttl=300)
        def get_stock_data(symbol, period="1y"):
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period)
                return data
            except Exception as e:
                st.error(f"Error fetching data: {str(e)}")
                return None
        
        # Time period selection
        st.sidebar.subheader("Chart Settings")
        period = st.sidebar.selectbox(
            "Select Time Period",
            ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
            index=3
        )
        
        # Get stock data
        stock_data = get_stock_data(ticker_symbol, period)
        
        if stock_data is not None and not stock_data.empty:
            # Create candlestick chart
            fig = go.Figure()
            
            fig.add_trace(go.Candlestick(
                x=stock_data.index,
                open=stock_data['Open'],
                high=stock_data['High'],
                low=stock_data['Low'],
                close=stock_data['Close'],
                name=ticker_symbol
            ))
            
            # Add volume subplot
            fig_subplot = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                subplot_titles=(f'{ticker_symbol} Price', 'Volume'),
                row_width=[0.2, 0.7]
            )
            
            # Add candlestick to main plot
            fig_subplot.add_trace(go.Candlestick(
                x=stock_data.index,
                open=stock_data['Open'],
                high=stock_data['High'],
                low=stock_data['Low'],
                close=stock_data['Close'],
                name=ticker_symbol
            ), row=1, col=1)
            
            # Add volume bars
            fig_subplot.add_trace(go.Bar(
                x=stock_data.index,
                y=stock_data['Volume'],
                name='Volume',
                marker_color='rgba(158,202,225,0.5)'
            ), row=2, col=1)
            
            fig_subplot.update_layout(
                title=f"{ticker_symbol} Stock Analysis",
                xaxis_rangeslider_visible=False,
                height=600
            )
            
            st.plotly_chart(fig_subplot, use_container_width=True)
            
            # Key Statistics
            st.subheader("📊 Key Statistics")
            
            # Calculate technical indicators
            stock_data['SMA_20'] = stock_data['Close'].rolling(window=20).mean()
            stock_data['SMA_50'] = stock_data['Close'].rolling(window=50).mean()
            stock_data['RSI'] = calculate_rsi(stock_data['Close'])
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                high_52w = stock_data['High'].max()
                st.metric("52W High", f"${high_52w:.2f}")
            
            with col2:
                low_52w = stock_data['Low'].min()
                st.metric("52W Low", f"${low_52w:.2f}")
            
            with col3:
                avg_volume = stock_data['Volume'].mean()
                st.metric("Avg Volume", f"{avg_volume:,.0f}")
            
            with col4:
                current_rsi = stock_data['RSI'].iloc[-1] if not stock_data['RSI'].isna().iloc[-1] else 0
                st.metric("RSI", f"{current_rsi:.2f}")
            
            # Recent data table
            st.subheader("📋 Recent Price Data")
            recent_data = stock_data.tail(10)[['Open', 'High', 'Low', 'Close', 'Volume']].round(2)
            recent_data.index = recent_data.index.strftime('%Y-%m-%d')
            st.dataframe(recent_data, use_container_width=True)
            
        else:
            st.error("Unable to fetch stock data. Please check the ticker symbol and try again.")
    
    else:
        st.error(f"Invalid ticker symbol: {ticker_symbol}. Please enter a valid stock ticker.")

# Navigation info
st.sidebar.markdown("---")
st.sidebar.markdown("### 🧭 Navigation")
st.sidebar.markdown("Use the pages in the sidebar to access:")
st.sidebar.markdown("- 📈 **Stock Analysis**: Detailed charts and indicators")
st.sidebar.markdown("- 🤖 **ML Predictions**: Machine learning forecasts")
st.sidebar.markdown("- 📊 **Prediction Comparison**: Model accuracy analysis")
st.sidebar.markdown("- 🔧 **What-If Simulator**: Scenario analysis")

# Footer
st.markdown("---")
st.markdown("*Data provided by Yahoo Finance. This is for educational purposes only and should not be considered as financial advice.*")
