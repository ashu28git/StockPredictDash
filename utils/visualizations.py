"""
Visualization utilities for the stock market prediction dashboard
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta

def create_candlestick_chart(data, title="Stock Price", show_volume=True):
    """
    Create a candlestick chart with optional volume
    
    Args:
        data (pd.DataFrame): Stock data with OHLCV columns
        title (str): Chart title
        show_volume (bool): Whether to show volume subplot
    
    Returns:
        plotly.graph_objects.Figure: Candlestick chart
    """
    if show_volume:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=(title, 'Volume'),
            row_heights=[0.7, 0.3]
        )
        
        # Candlestick chart
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name="OHLC"
        ), row=1, col=1)
        
        # Volume bars
        fig.add_trace(go.Bar(
            x=data.index,
            y=data['Volume'],
            name='Volume',
            marker_color='rgba(158,202,225,0.5)',
            showlegend=False
        ), row=2, col=1)
        
    else:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name="OHLC"
        ))
    
    fig.update_layout(
        title=title,
        xaxis_rangeslider_visible=False,
        height=600 if show_volume else 400
    )
    
    return fig

def create_line_chart(data, columns, title="Price Chart", colors=None):
    """
    Create a line chart for multiple time series
    
    Args:
        data (pd.DataFrame): Data to plot
        columns (list): Column names to plot
        title (str): Chart title
        colors (list): Optional list of colors for each line
    
    Returns:
        plotly.graph_objects.Figure: Line chart
    """
    fig = go.Figure()
    
    if colors is None:
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    for i, col in enumerate(columns):
        if col in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data[col],
                mode='lines',
                name=col,
                line=dict(color=colors[i % len(colors)], width=2)
            ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price",
        height=400,
        hovermode='x unified'
    )
    
    return fig

def create_prediction_chart(historical_data, predictions_dict, prediction_dates, title="Price Predictions"):
    """
    Create a chart showing historical data and future predictions
    
    Args:
        historical_data (pd.DataFrame): Historical stock data
        predictions_dict (dict): Dictionary of model predictions
        prediction_dates (list): List of future dates
        title (str): Chart title
    
    Returns:
        plotly.graph_objects.Figure: Prediction chart
    """
    fig = go.Figure()
    
    # Historical data (last 60 days)
    recent_data = historical_data.tail(60)
    fig.add_trace(go.Scatter(
        x=recent_data.index,
        y=recent_data['Close'],
        mode='lines',
        name='Historical Price',
        line=dict(color='blue', width=2)
    ))
    
    # Predictions
    colors = ['red', 'green', 'orange', 'purple', 'brown']
    for i, (model_name, predictions) in enumerate(predictions_dict.items()):
        fig.add_trace(go.Scatter(
            x=prediction_dates,
            y=predictions,
            mode='lines+markers',
            name=f'{model_name} Prediction',
            line=dict(color=colors[i % len(colors)], width=2, dash='dash'),
            marker=dict(size=6)
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=500,
        hovermode='x unified'
    )
    
    return fig

def create_technical_indicators_chart(data, indicators=None):
    """
    Create a multi-panel chart with technical indicators
    
    Args:
        data (pd.DataFrame): Stock data with technical indicators
        indicators (list): List of indicators to show
    
    Returns:
        plotly.graph_objects.Figure: Technical indicators chart
    """
    if indicators is None:
        indicators = ['RSI', 'MACD']
    
    # Count available indicators
    available_indicators = [ind for ind in indicators if ind in data.columns or any(ind in col for col in data.columns)]
    subplot_count = 1 + len(available_indicators)
    
    subplot_titles = ['Price'] + available_indicators
    
    fig = make_subplots(
        rows=subplot_count,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=subplot_titles,
        row_heights=[0.6] + [0.4/len(available_indicators)] * len(available_indicators) if len(available_indicators) > 0 else [1]
    )
    
    # Price chart
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name="Price",
        showlegend=False
    ), row=1, col=1)
    
    # Add moving averages if available
    ma_columns = [col for col in data.columns if 'SMA' in col]
    colors = ['orange', 'red', 'purple', 'brown']
    for i, ma_col in enumerate(ma_columns[:4]):  # Limit to 4 MAs
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data[ma_col],
            mode='lines',
            name=ma_col,
            line=dict(color=colors[i], width=1)
        ), row=1, col=1)
    
    current_row = 2
    
    # RSI
    if 'RSI' in available_indicators:
        rsi_col = 'RSI' if 'RSI' in data.columns else [col for col in data.columns if 'RSI' in col][0]
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data[rsi_col],
            mode='lines',
            name='RSI',
            line=dict(color='purple'),
            showlegend=False
        ), row=current_row, col=1)
        
        # Add RSI reference lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=current_row, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=current_row, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", row=current_row, col=1)
        
        current_row += 1
    
    # MACD
    if 'MACD' in available_indicators and 'MACD' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['MACD'],
            mode='lines',
            name='MACD',
            line=dict(color='blue'),
            showlegend=False
        ), row=current_row, col=1)
        
        if 'MACD_Signal' in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['MACD_Signal'],
                mode='lines',
                name='Signal',
                line=dict(color='red'),
                showlegend=False
            ), row=current_row, col=1)
        
        if 'MACD_Histogram' in data.columns:
            fig.add_trace(go.Bar(
                x=data.index,
                y=data['MACD_Histogram'],
                name='Histogram',
                marker_color='gray',
                opacity=0.7,
                showlegend=False
            ), row=current_row, col=1)
        
        current_row += 1
    
    fig.update_layout(
        height=400 + 200 * len(available_indicators),
        showlegend=True,
        xaxis_rangeslider_visible=False
    )
    
    return fig

def create_correlation_heatmap(data, columns=None, title="Correlation Matrix"):
    """
    Create a correlation heatmap
    
    Args:
        data (pd.DataFrame): Data to analyze
        columns (list): Columns to include in correlation
        title (str): Chart title
    
    Returns:
        plotly.graph_objects.Figure: Heatmap
    """
    if columns is None:
        # Use numeric columns only
        columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    corr_matrix = data[columns].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title=title,
        height=500,
        width=500
    )
    
    return fig

def create_performance_comparison_chart(actual_prices, predictions_dict, dates, title="Model Performance Comparison"):
    """
    Create a chart comparing model predictions with actual prices
    
    Args:
        actual_prices (pd.Series): Actual prices
        predictions_dict (dict): Dictionary of model predictions
        dates (pd.DatetimeIndex): Dates for the data
        title (str): Chart title
    
    Returns:
        plotly.graph_objects.Figure: Performance comparison chart
    """
    fig = go.Figure()
    
    # Actual prices
    fig.add_trace(go.Scatter(
        x=dates,
        y=actual_prices,
        mode='lines+markers',
        name='Actual Price',
        line=dict(color='black', width=3),
        marker=dict(size=4)
    ))
    
    # Model predictions
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i, (model_name, predictions) in enumerate(predictions_dict.items()):
        fig.add_trace(go.Scatter(
            x=dates,
            y=predictions,
            mode='lines+markers',
            name=f'{model_name}',
            line=dict(color=colors[i % len(colors)], width=2, dash='dash'),
            marker=dict(size=3)
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=500,
        hovermode='x unified'
    )
    
    return fig

def create_error_analysis_chart(actual, predicted_dict, dates):
    """
    Create error analysis charts
    
    Args:
        actual (pd.Series): Actual values
        predicted_dict (dict): Dictionary of predictions from different models
        dates (pd.DatetimeIndex): Dates
    
    Returns:
        plotly.graph_objects.Figure: Error analysis chart
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Absolute Errors', 'Percentage Errors', 'Error Distribution', 'Cumulative Errors'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for i, (model_name, predicted) in enumerate(predicted_dict.items()):
        color = colors[i % len(colors)]
        
        # Calculate errors
        absolute_errors = np.abs(actual - predicted)
        percentage_errors = (absolute_errors / actual) * 100
        
        # Absolute errors over time
        fig.add_trace(
            go.Scatter(x=dates, y=absolute_errors,
                     mode='lines', name=f'{model_name} Abs Error',
                     line=dict(color=color)),
            row=1, col=1
        )
        
        # Percentage errors over time
        fig.add_trace(
            go.Scatter(x=dates, y=percentage_errors,
                     mode='lines', name=f'{model_name} % Error',
                     line=dict(color=color)),
            row=1, col=2
        )
        
        # Error distribution
        fig.add_trace(
            go.Histogram(x=percentage_errors, name=f'{model_name} Error Dist',
                       marker_color=color, opacity=0.7, nbinsx=20),
            row=2, col=1
        )
        
        # Cumulative errors
        cumulative_errors = np.cumsum(absolute_errors)
        fig.add_trace(
            go.Scatter(x=dates, y=cumulative_errors,
                     mode='lines', name=f'{model_name} Cumulative',
                     line=dict(color=color)),
            row=2, col=2
        )
    
    fig.update_layout(height=600, showlegend=True)
    return fig

def create_feature_importance_chart(importance_df, title="Feature Importance", top_n=15):
    """
    Create a feature importance chart
    
    Args:
        importance_df (pd.DataFrame): DataFrame with 'feature' and 'importance' columns
        title (str): Chart title
        top_n (int): Number of top features to show
    
    Returns:
        plotly.graph_objects.Figure: Feature importance chart
    """
    # Take top N features
    top_features = importance_df.head(top_n)
    
    fig = go.Figure(go.Bar(
        x=top_features['importance'],
        y=top_features['feature'],
        orientation='h',
        marker_color='lightblue'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Importance",
        yaxis_title="Features",
        height=400 + 20 * top_n,
        yaxis=dict(autorange="reversed")  # Show most important at top
    )
    
    return fig

def create_volatility_chart(data, window=20, title="Price Volatility"):
    """
    Create a volatility chart
    
    Args:
        data (pd.DataFrame): Stock data
        window (int): Rolling window for volatility calculation
        title (str): Chart title
    
    Returns:
        plotly.graph_objects.Figure: Volatility chart
    """
    # Calculate rolling volatility
    returns = data['Close'].pct_change()
    volatility = returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('Price', f'{window}-Day Rolling Volatility'),
        row_heights=[0.7, 0.3]
    )
    
    # Price chart
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='blue')
    ), row=1, col=1)
    
    # Volatility chart
    fig.add_trace(go.Scatter(
        x=data.index,
        y=volatility,
        mode='lines',
        name='Volatility',
        line=dict(color='red'),
        fill='tonexty',
        fillcolor='rgba(255,0,0,0.1)'
    ), row=2, col=1)
    
    fig.update_layout(
        title=title,
        height=500,
        showlegend=False
    )
    
    return fig

def create_volume_analysis_chart(data, title="Volume Analysis"):
    """
    Create a volume analysis chart with price
    
    Args:
        data (pd.DataFrame): Stock data with Volume column
        title (str): Chart title
    
    Returns:
        plotly.graph_objects.Figure: Volume analysis chart
    """
    # Calculate volume moving average
    volume_ma = data['Volume'].rolling(window=20).mean()
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('Price', 'Volume'),
        row_heights=[0.6, 0.4]
    )
    
    # Price chart
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='blue')
    ), row=1, col=1)
    
    # Volume bars
    fig.add_trace(go.Bar(
        x=data.index,
        y=data['Volume'],
        name='Volume',
        marker_color='lightblue',
        opacity=0.7
    ), row=2, col=1)
    
    # Volume moving average
    fig.add_trace(go.Scatter(
        x=data.index,
        y=volume_ma,
        mode='lines',
        name='Volume MA(20)',
        line=dict(color='red', width=2)
    ), row=2, col=1)
    
    fig.update_layout(
        title=title,
        height=500,
        showlegend=True
    )
    
    return fig

def display_metrics_cards(metrics_dict, cols=4):
    """
    Display metrics in Streamlit columns
    
    Args:
        metrics_dict (dict): Dictionary of metrics to display
        cols (int): Number of columns
    """
    columns = st.columns(cols)
    
    for i, (label, value) in enumerate(metrics_dict.items()):
        with columns[i % cols]:
            if isinstance(value, float):
                if 'percentage' in label.lower() or '%' in label:
                    st.metric(label, f"{value:.2f}%")
                elif 'price' in label.lower() or '$' in label:
                    st.metric(label, f"${value:.2f}")
                else:
                    st.metric(label, f"{value:.4f}")
            else:
                st.metric(label, str(value))

def create_drawdown_chart(data, title="Drawdown Analysis"):
    """
    Create a drawdown chart showing peak-to-trough declines
    
    Args:
        data (pd.DataFrame): Stock data
        title (str): Chart title
    
    Returns:
        plotly.graph_objects.Figure: Drawdown chart
    """
    # Calculate cumulative returns and drawdown
    prices = data['Close']
    peak = prices.cummax()
    drawdown = (prices - peak) / peak * 100
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('Price vs Peak', 'Drawdown (%)'),
        row_heights=[0.6, 0.4]
    )
    
    # Price vs peak
    fig.add_trace(go.Scatter(
        x=data.index,
        y=prices,
        mode='lines',
        name='Price',
        line=dict(color='blue')
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=data.index,
        y=peak,
        mode='lines',
        name='Peak',
        line=dict(color='green', dash='dash')
    ), row=1, col=1)
    
    # Drawdown
    fig.add_trace(go.Scatter(
        x=data.index,
        y=drawdown,
        mode='lines',
        name='Drawdown',
        line=dict(color='red'),
        fill='tonexty',
        fillcolor='rgba(255,0,0,0.3)'
    ), row=2, col=1)
    
    fig.update_layout(
        title=title,
        height=500,
        showlegend=True
    )
    
    return fig
