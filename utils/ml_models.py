"""
Machine learning models for stock price prediction
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class StockPredictor:
    """
    A class for stock price prediction using machine learning models
    """
    
    def __init__(self, model_type='random_forest'):
        """
        Initialize the predictor
        
        Args:
            model_type (str): Type of model ('random_forest', 'linear_regression')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.is_trained = False
        
        # Initialize model based on type
        if model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
                max_depth=10
            )
        elif model_type == 'linear_regression':
            self.model = LinearRegression()
            self.scaler = StandardScaler()
        else:
            raise ValueError("model_type must be 'random_forest' or 'linear_regression'")
    
    def create_features(self, data, lookback_window=20):
        """
        Create features for machine learning
        
        Args:
            data (pd.DataFrame): Stock data with OHLCV columns
            lookback_window (int): Number of days to look back for lag features
        
        Returns:
            pd.DataFrame: Feature DataFrame
        """
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
        features_df['High_Close_Ratio'] = data['High'] / data['Close']
        features_df['Low_Close_Ratio'] = data['Low'] / data['Close']
        
        # Volume features
        features_df['Volume_Change'] = data['Volume'].pct_change()
        features_df['Price_Volume'] = data['Close'] * data['Volume']
        features_df['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
        features_df['Volume_Ratio'] = data['Volume'] / features_df['Volume_SMA']
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            if len(data) >= window:
                ma = data['Close'].rolling(window=window).mean()
                features_df[f'SMA_{window}'] = ma
                features_df[f'Price_SMA_{window}_Ratio'] = data['Close'] / ma
        
        # Technical indicators
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features_df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        bb_window = 20
        if len(data) >= bb_window:
            bb_middle = data['Close'].rolling(window=bb_window).mean()
            bb_std = data['Close'].rolling(window=bb_window).std()
            features_df['BB_Upper'] = bb_middle + (bb_std * 2)
            features_df['BB_Lower'] = bb_middle - (bb_std * 2)
            features_df['BB_Width'] = features_df['BB_Upper'] - features_df['BB_Lower']
            features_df['BB_Position'] = (data['Close'] - features_df['BB_Lower']) / (features_df['BB_Upper'] - features_df['BB_Lower'])
        
        # MACD
        exp1 = data['Close'].ewm(span=12).mean()
        exp2 = data['Close'].ewm(span=26).mean()
        features_df['MACD'] = exp1 - exp2
        features_df['MACD_Signal'] = features_df['MACD'].ewm(span=9).mean()
        features_df['MACD_Histogram'] = features_df['MACD'] - features_df['MACD_Signal']
        
        # Volatility features
        features_df['Volatility_10'] = data['Close'].rolling(window=10).std()
        features_df['Volatility_20'] = data['Close'].rolling(window=20).std()
        
        # Momentum features
        for period in [3, 5, 10]:
            features_df[f'Momentum_{period}'] = data['Close'] / data['Close'].shift(period) - 1
        
        # Lag features
        max_lag = min(lookback_window, 10)  # Limit lags to prevent overfitting
        for lag in range(1, max_lag + 1):
            features_df[f'Close_Lag_{lag}'] = features_df['Close'].shift(lag)
            features_df[f'Price_Change_Lag_{lag}'] = features_df['Price_Change'].shift(lag)
            features_df[f'Volume_Lag_{lag}'] = features_df['Volume'].shift(lag)
        
        # Day of week and month features
        features_df['DayOfWeek'] = data.index.dayofweek
        features_df['Month'] = data.index.month
        features_df['Quarter'] = data.index.quarter
        
        return features_df
    
    def prepare_data(self, features_df, target_days=1):
        """
        Prepare data for training
        
        Args:
            features_df (pd.DataFrame): Features DataFrame
            target_days (int): Number of days ahead to predict
        
        Returns:
            tuple: (X, y) training data
        """
        # Create target variable (future close price)
        target = features_df['Close'].shift(-target_days)
        
        # Remove rows with NaN values
        valid_data = features_df.dropna()
        valid_target = target.dropna()
        
        # Align data and target
        common_index = valid_data.index.intersection(valid_target.index)
        X = valid_data.loc[common_index]
        y = valid_target.loc[common_index]
        
        # Store feature columns for later use
        self.feature_columns = X.columns
        
        return X, y
    
    def train(self, X, y, test_size=0.2):
        """
        Train the model
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target variable
            test_size (float): Fraction of data to use for testing
        
        Returns:
            dict: Training metrics
        """
        if len(X) < 50:
            raise ValueError("Insufficient data for training. Need at least 50 samples.")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False, random_state=42
        )
        
        # Scale features if using linear regression
        if self.model_type == 'linear_regression':
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Make predictions on test set
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred),
            'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        }
        
        # Cross-validation score
        if self.model_type == 'random_forest':
            cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='r2')
            metrics['cv_r2_mean'] = cv_scores.mean()
            metrics['cv_r2_std'] = cv_scores.std()
        
        self.is_trained = True
        return metrics
    
    def predict_future(self, features_df, days=7):
        """
        Predict future prices
        
        Args:
            features_df (pd.DataFrame): Features DataFrame
            days (int): Number of days to predict
        
        Returns:
            list: List of predicted prices
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = []
        current_features = features_df.iloc[-1:].copy()
        
        for day in range(days):
            # Ensure we have the same features as training
            if self.feature_columns is not None:
                # Get common columns and add missing ones with default values
                common_cols = current_features.columns.intersection(self.feature_columns)
                missing_cols = set(self.feature_columns) - set(common_cols)
                
                for col in missing_cols:
                    current_features[col] = 0  # Default value for missing features
                
                current_features = current_features[self.feature_columns]
            
            # Scale if using linear regression
            if self.model_type == 'linear_regression':
                features_scaled = self.scaler.transform(current_features)
                pred = self.model.predict(features_scaled)[0]
            else:
                pred = self.model.predict(current_features)[0]
            
            predictions.append(pred)
            
            # Update features for next prediction
            current_features = self._update_features_for_next_prediction(current_features, pred)
        
        return predictions
    
    def _update_features_for_next_prediction(self, current_features, predicted_price):
        """
        Update features for the next prediction step
        
        Args:
            current_features (pd.DataFrame): Current features
            predicted_price (float): Predicted price
        
        Returns:
            pd.DataFrame: Updated features
        """
        updated_features = current_features.copy()
        
        # Update close price
        updated_features['Close'] = predicted_price
        
        # Update lag features
        for lag in range(1, 11):
            if f'Close_Lag_{lag}' in updated_features.columns:
                if lag == 1:
                    updated_features[f'Close_Lag_{lag}'] = predicted_price
                elif f'Close_Lag_{lag-1}' in updated_features.columns:
                    updated_features[f'Close_Lag_{lag}'] = updated_features[f'Close_Lag_{lag-1}']
        
        # Update price change lag features
        if 'Close_Lag_1' in updated_features.columns and updated_features['Close_Lag_1'].iloc[0] != 0:
            price_change = (predicted_price - updated_features['Close_Lag_1'].iloc[0]) / updated_features['Close_Lag_1'].iloc[0]
            updated_features['Price_Change'] = price_change
            
            for lag in range(1, 11):
                if f'Price_Change_Lag_{lag}' in updated_features.columns:
                    if lag == 1:
                        updated_features[f'Price_Change_Lag_{lag}'] = price_change
                    elif f'Price_Change_Lag_{lag-1}' in updated_features.columns:
                        updated_features[f'Price_Change_Lag_{lag}'] = updated_features[f'Price_Change_Lag_{lag-1}']
        
        # Update moving average ratios (simplified)
        for window in [5, 10, 20, 50]:
            if f'Price_SMA_{window}_Ratio' in updated_features.columns:
                # Use the predicted price as approximation
                updated_features[f'Price_SMA_{window}_Ratio'] = 1.0
        
        return updated_features
    
    def get_feature_importance(self):
        """
        Get feature importance (for tree-based models)
        
        Returns:
            pd.DataFrame: Feature importance
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        if self.model_type == 'random_forest':
            importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance
        else:
            return None

def create_ensemble_prediction(predictors, features_df, days=7):
    """
    Create ensemble predictions from multiple models
    
    Args:
        predictors (list): List of trained StockPredictor objects
        features_df (pd.DataFrame): Features DataFrame
        days (int): Number of days to predict
    
    Returns:
        dict: Ensemble predictions
    """
    individual_predictions = {}
    
    # Get predictions from each model
    for i, predictor in enumerate(predictors):
        model_name = f"{predictor.model_type}_{i}"
        try:
            predictions = predictor.predict_future(features_df, days)
            individual_predictions[model_name] = predictions
        except Exception as e:
            print(f"Error with {model_name}: {str(e)}")
            continue
    
    if not individual_predictions:
        return None
    
    # Calculate ensemble predictions
    ensemble_results = {
        'individual_predictions': individual_predictions,
        'mean_prediction': [],
        'median_prediction': [],
        'weighted_prediction': []
    }
    
    # Calculate mean and median for each day
    for day in range(days):
        day_predictions = [pred[day] for pred in individual_predictions.values() if len(pred) > day]
        
        if day_predictions:
            ensemble_results['mean_prediction'].append(np.mean(day_predictions))
            ensemble_results['median_prediction'].append(np.median(day_predictions))
            
            # Weighted average (simple equal weights for now)
            ensemble_results['weighted_prediction'].append(np.mean(day_predictions))
    
    return ensemble_results

def calculate_prediction_confidence(predictions_list):
    """
    Calculate confidence intervals for predictions
    
    Args:
        predictions_list (list): List of prediction arrays from different models
    
    Returns:
        dict: Confidence intervals
    """
    if not predictions_list:
        return None
    
    predictions_array = np.array(predictions_list)
    
    confidence_intervals = {
        'mean': np.mean(predictions_array, axis=0),
        'std': np.std(predictions_array, axis=0),
        'lower_95': np.percentile(predictions_array, 2.5, axis=0),
        'upper_95': np.percentile(predictions_array, 97.5, axis=0),
        'lower_75': np.percentile(predictions_array, 12.5, axis=0),
        'upper_75': np.percentile(predictions_array, 87.5, axis=0)
    }
    
    return confidence_intervals
