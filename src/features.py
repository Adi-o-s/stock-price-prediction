# =============================================================================
# FEATURE ENGINEERING MODULE (LEAKAGE-FREE)
# =============================================================================

import pandas as pd
import numpy as np

def calculate_rsi(data, window=14):
    """Calculates Relative Strength Index."""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def feature_engineering_train(df):
    """
    Feature engineering for TRAINING set.
    All calculations use only past data (no lookahead bias).
    
    Args:
        df: Training DataFrame with OHLC data
    Returns:
        DataFrame with engineered features
    """
    df_feat = df.copy()
    
    # Ensure single-level column index and proper structure
    if isinstance(df_feat.columns, pd.MultiIndex):
        df_feat.columns = df_feat.columns.get_level_values(0)
    
    # Work directly with column values to avoid any Series/DataFrame confusion
    # Price-based features
    df_feat['Log_Return'] = np.log(df_feat['Close'].values / np.roll(df_feat['Close'].values, 1))
    df_feat['Log_Return'].iloc[0] = np.nan  # First value should be NaN
    
    df_feat['High_Low_Ratio'] = df_feat['High'].values / df_feat['Low'].values
    df_feat['Close_Open_Ratio'] = df_feat['Close'].values / df_feat['Open'].values
    
    # Moving Averages (Trend indicators)
    df_feat['MA_5'] = df_feat['Close'].rolling(window=5).mean().values
    df_feat['MA_10'] = df_feat['Close'].rolling(window=10).mean().values
    df_feat['MA_20'] = df_feat['Close'].rolling(window=20).mean().values
    df_feat['MA_50'] = df_feat['Close'].rolling(window=50).mean().values
    
    # Volatility
    df_feat['Volatility_10'] = df_feat['Close'].rolling(window=10).std().values
    df_feat['Volatility_20'] = df_feat['Close'].rolling(window=20).std().values
    
    # Technical Indicators
    df_feat['RSI'] = calculate_rsi(df_feat['Close'], window=14)
    
    # Volume features - work with values directly
    volume_values = df_feat['Volume'].values
    volume_ma_values = pd.Series(volume_values).rolling(window=10).mean().values
    df_feat['Volume_MA_10'] = volume_ma_values
    
    # Safe division for Volume_Ratio
    volume_ratio = np.zeros_like(volume_values, dtype=float)
    mask = volume_ma_values != 0
    volume_ratio[mask] = volume_values[mask] / volume_ma_values[mask]
    df_feat['Volume_Ratio'] = volume_ratio
    
    # Momentum indicators
    df_feat['Momentum_5'] = df_feat['Close'].values - np.roll(df_feat['Close'].values, 5)
    df_feat['Momentum_10'] = df_feat['Close'].values - np.roll(df_feat['Close'].values, 10)
    
    # Lagged features (crucial for time series)
    log_ret_values = df_feat['Log_Return'].values
    df_feat['Lag_1'] = np.roll(log_ret_values, 1)
    df_feat['Lag_2'] = np.roll(log_ret_values, 2)
    df_feat['Lag_3'] = np.roll(log_ret_values, 3)
    df_feat['Lag_5'] = np.roll(log_ret_values, 5)
    
    # Set first N values to NaN for lagged features
    df_feat['Lag_1'].iloc[0] = np.nan
    df_feat['Lag_2'].iloc[:2] = np.nan
    df_feat['Lag_3'].iloc[:3] = np.nan
    df_feat['Lag_5'].iloc[:5] = np.nan
    
    # Rolling statistics
    df_feat['Return_MA_5'] = pd.Series(log_ret_values).rolling(window=5).mean().values
    df_feat['Return_Std_5'] = pd.Series(log_ret_values).rolling(window=5).std().values
    
    df_feat.dropna(inplace=True)
    return df_feat

def feature_engineering_test(test_df, train_df):
    """
    Feature engineering for TEST set.
    Uses statistics from TRAINING set to avoid data leakage.
    
    Args:
        test_df: Test DataFrame
        train_df: Training DataFrame (for reference statistics)
    Returns:
        DataFrame with engineered features
    """
    # Combine train and test for rolling calculations
    combined = pd.concat([train_df, test_df])
    
    df_feat = combined.copy()
    
    # Ensure single-level column index
    if isinstance(df_feat.columns, pd.MultiIndex):
        df_feat.columns = df_feat.columns.get_level_values(0)
    
    # Work directly with column values to avoid any Series/DataFrame confusion
    # Price-based features
    df_feat['Log_Return'] = np.log(df_feat['Close'].values / np.roll(df_feat['Close'].values, 1))
    df_feat['Log_Return'].iloc[0] = np.nan
    
    df_feat['High_Low_Ratio'] = df_feat['High'].values / df_feat['Low'].values
    df_feat['Close_Open_Ratio'] = df_feat['Close'].values / df_feat['Open'].values
    
    # Moving Averages
    df_feat['MA_5'] = df_feat['Close'].rolling(window=5).mean().values
    df_feat['MA_10'] = df_feat['Close'].rolling(window=10).mean().values
    df_feat['MA_20'] = df_feat['Close'].rolling(window=20).mean().values
    df_feat['MA_50'] = df_feat['Close'].rolling(window=50).mean().values
    
    # Volatility
    df_feat['Volatility_10'] = df_feat['Close'].rolling(window=10).std().values
    df_feat['Volatility_20'] = df_feat['Close'].rolling(window=20).std().values
    
    # Technical Indicators
    df_feat['RSI'] = calculate_rsi(df_feat['Close'], window=14)
    
    # Volume features - work with values directly
    volume_values = df_feat['Volume'].values
    volume_ma_values = pd.Series(volume_values).rolling(window=10).mean().values
    df_feat['Volume_MA_10'] = volume_ma_values
    
    # Safe division for Volume_Ratio
    volume_ratio = np.zeros_like(volume_values, dtype=float)
    mask = volume_ma_values != 0
    volume_ratio[mask] = volume_values[mask] / volume_ma_values[mask]
    df_feat['Volume_Ratio'] = volume_ratio
    
    # Momentum
    df_feat['Momentum_5'] = df_feat['Close'].values - np.roll(df_feat['Close'].values, 5)
    df_feat['Momentum_10'] = df_feat['Close'].values - np.roll(df_feat['Close'].values, 10)
    
    # Lagged features
    log_ret_values = df_feat['Log_Return'].values
    df_feat['Lag_1'] = np.roll(log_ret_values, 1)
    df_feat['Lag_2'] = np.roll(log_ret_values, 2)
    df_feat['Lag_3'] = np.roll(log_ret_values, 3)
    df_feat['Lag_5'] = np.roll(log_ret_values, 5)
    
    # Set first N values to NaN for lagged features
    df_feat['Lag_1'].iloc[0] = np.nan
    df_feat['Lag_2'].iloc[:2] = np.nan
    df_feat['Lag_3'].iloc[:3] = np.nan
    df_feat['Lag_5'].iloc[:5] = np.nan
    
    # Rolling statistics
    df_feat['Return_MA_5'] = pd.Series(log_ret_values).rolling(window=5).mean().values
    df_feat['Return_Std_5'] = pd.Series(log_ret_values).rolling(window=5).std().values
    
    # Return only test portion
    test_feat = df_feat.iloc[len(train_df):].copy()
    test_feat.dropna(inplace=True)
    
    return test_feat