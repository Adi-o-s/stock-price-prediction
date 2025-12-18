# =============================================================================
# DATA LOADING MODULE
# =============================================================================

import yfinance as yf
import pandas as pd

def load_data(tickers, start_date, end_date):
    """
    Fetches historical OHLC data for multiple stock tickers.
    Handles data cleaning, missing values, and anomalies.
    
    Assignment requirement: Clean dataset, handle anomalies and missing values.
    
    Args:
        tickers (list): List of ticker symbols (e.g., ['AAPL', 'MSFT'])
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
    
    Returns:
        dict: Dictionary mapping ticker -> cleaned DataFrame
    """
    stock_data = {}
    
    print(f"Fetching data for {len(tickers)} stocks from {start_date} to {end_date}...")
    
    for ticker in tickers:
        try:
            # Download data using yfinance
            df = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=True  # Adjust for splits and dividends
            )
            
            # Data cleaning steps
            if df.empty:
                print(f"  ✗ {ticker}: No data available")
                continue
            
            # Keep only OHLC and Volume
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

            # Flatten MultiIndex columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # Handle missing values
            # Forward fill first (carry last valid observation forward)
            df.ffill(inplace=True)
            
            # Then backward fill for any remaining NaNs at the start
            df.bfill(inplace=True)
            
            # Remove any remaining rows with NaN
            initial_len = len(df)
            df.dropna(inplace=True)
            
            if len(df) < initial_len:
                print(f"  ⚠ {ticker}: Removed {initial_len - len(df)} rows with missing values")
            
            # Data validation: Remove anomalies
            # 1. Remove days where High < Low (data error)
            invalid_rows = df[df['High'] < df['Low']]
            if len(invalid_rows) > 0:
                print(f"  ⚠ {ticker}: Removing {len(invalid_rows)} rows where High < Low")
                df = df[df['High'] >= df['Low']]
            
            # 2. Remove extreme outliers (price changes > 50% in one day)
            # These are usually data errors or stock splits not handled properly
            df['Price_Change'] = df['Close'].pct_change().abs()
            extreme_changes = df[df['Price_Change'] > 0.5]
            
            if len(extreme_changes) > 0:
                print(f"  ⚠ {ticker}: Found {len(extreme_changes)} days with >50% price change")
                # Keep only reasonable changes
                df = df[df['Price_Change'] <= 0.5]
            
            df.drop('Price_Change', axis=1, inplace=True)
            
            # 3. Ensure Volume is non-negative
            df = df[df['Volume'] >= 0]
            
            # Final check
            if len(df) < 100:
                print(f"  ✗ {ticker}: Insufficient data after cleaning ({len(df)} days)")
                continue
            
            stock_data[ticker] = df
            print(f"  ✓ {ticker}: {len(df)} days of clean data")
            
        except Exception as e:
            print(f"  ✗ {ticker}: Error during download - {str(e)}")
            continue
    
    print(f"\nSuccessfully loaded {len(stock_data)}/{len(tickers)} stocks")
    return stock_data