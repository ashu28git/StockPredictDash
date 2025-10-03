"""
Data fetching utilities for stock market data
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_stock_data(symbol, period="1y"):
    """
    Fetch stock data from Yahoo Finance
    
    Args:
        symbol (str): Stock ticker symbol
        period (str): Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
    
    Returns:
        pd.DataFrame: Stock data with OHLCV columns
    """
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        
        if data.empty:
            st.error(f"No data found for symbol: {symbol}")
            return None
            
        return data
    
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

@st.cache_data(ttl=600)  # Cache for 10 minutes
def get_stock_info(symbol):
    """
    Get stock information and metadata
    
    Args:
        symbol (str): Stock ticker symbol
    
    Returns:
        dict: Stock information
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return info
    
    except Exception as e:
        st.error(f"Error fetching info for {symbol}: {str(e)}")
        return None

def validate_ticker(symbol):
    """
    Validate if a ticker symbol exists
    
    Args:
        symbol (str): Stock ticker symbol
    
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        # Check if basic price information exists
        return 'regularMarketPrice' in info or 'currentPrice' in info
    
    except Exception:
        return False

def get_multiple_stocks(symbols, period="1y"):
    """
    Fetch data for multiple stocks
    
    Args:
        symbols (list): List of stock ticker symbols
        period (str): Time period
    
    Returns:
        dict: Dictionary with symbol as key and DataFrame as value
    """
    stock_data = {}
    
    for symbol in symbols:
        data = get_stock_data(symbol, period)
        if data is not None:
            stock_data[symbol] = data
    
    return stock_data

def get_market_data(indices=["^GSPC", "^DJI", "^IXIC"], period="1d"):
    """
    Get major market indices data
    
    Args:
        indices (list): List of market index symbols
        period (str): Time period
    
    Returns:
        dict: Market indices data
    """
    market_data = {}
    index_names = {
        "^GSPC": "S&P 500",
        "^DJI": "Dow Jones",
        "^IXIC": "NASDAQ"
    }
    
    for index in indices:
        data = get_stock_data(index, period)
        if data is not None:
            market_data[index_names.get(index, index)] = data
    
    return market_data

def calculate_returns(data, periods=[1, 5, 10, 21, 63, 252]):
    """
    Calculate returns for different periods
    
    Args:
        data (pd.DataFrame): Stock data
        periods (list): List of periods to calculate returns for
    
    Returns:
        pd.Series: Returns for different periods
    """
    if data is None or data.empty:
        return None
    
    current_price = data['Close'].iloc[-1]
    returns = {}
    
    for period in periods:
        if len(data) > period:
            past_price = data['Close'].iloc[-period-1]
            return_pct = ((current_price - past_price) / past_price) * 100
            
            # Map periods to labels
            period_labels = {
                1: "1D",
                5: "5D", 
                10: "10D",
                21: "1M",
                63: "3M",
                252: "1Y"
            }
            
            returns[period_labels.get(period, f"{period}D")] = return_pct
    
    return pd.Series(returns)

def get_sector_performance():
    """
    Get sector ETF performance data
    
    Returns:
        pd.DataFrame: Sector performance data
    """
    sector_etfs = {
        'XLK': 'Technology',
        'XLF': 'Financial',
        'XLV': 'Healthcare',
        'XLE': 'Energy',
        'XLI': 'Industrial',
        'XLY': 'Consumer Discretionary',
        'XLP': 'Consumer Staples',
        'XLU': 'Utilities',
        'XLB': 'Materials',
        'XLRE': 'Real Estate',
        'XLC': 'Communication Services'
    }
    
    sector_data = {}
    
    for etf, sector in sector_etfs.items():
        data = get_stock_data(etf, "1d")
        if data is not None and not data.empty:
            current_price = data['Close'].iloc[-1]
            prev_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
            daily_change = ((current_price - prev_price) / prev_price) * 100
            
            sector_data[sector] = {
                'ETF': etf,
                'Price': current_price,
                'Change': daily_change
            }
    
    if sector_data:
        return pd.DataFrame(sector_data).T
    else:
        return None

def get_crypto_data(symbols=["BTC-USD", "ETH-USD"], period="1d"):
    """
    Get cryptocurrency data
    
    Args:
        symbols (list): List of crypto symbols
        period (str): Time period
    
    Returns:
        dict: Crypto data
    """
    crypto_data = {}
    crypto_names = {
        "BTC-USD": "Bitcoin",
        "ETH-USD": "Ethereum"
    }
    
    for symbol in symbols:
        data = get_stock_data(symbol, period)
        if data is not None:
            crypto_data[crypto_names.get(symbol, symbol)] = data
    
    return crypto_data
