# =============================================================================
# VISUALIZATION MODULE
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose

def plot_seasonal_analysis(df, ticker):
    """
    Performs seasonal decomposition analysis (EDA requirement).
    
    Args:
        df: DataFrame with Close prices
        ticker: Stock ticker symbol
    """
    try:
        # Use 252 trading days = 1 year for seasonality
        series = df['Close'].dropna()
        
        if len(series) < 252 * 2:
            print(f"  Warning: Not enough data for seasonal decomposition of {ticker}")
            return
        
        result = seasonal_decompose(series, model='multiplicative', period=252)
        
        # Create comprehensive plot
        fig, axes = plt.subplots(4, 1, figsize=(12, 10))
        
        # Original
        axes[0].plot(series.index, series.values, color='blue', linewidth=1)
        axes[0].set_ylabel('Price ($)')
        axes[0].set_title(f'{ticker} - Original Price Series')
        axes[0].grid(True, alpha=0.3)
        
        # Trend
        axes[1].plot(result.trend.index, result.trend.values, color='green', linewidth=1.5)
        axes[1].set_ylabel('Trend')
        axes[1].set_title('Long-term Trend Component')
        axes[1].grid(True, alpha=0.3)
        
        # Seasonality
        axes[2].plot(result.seasonal.index, result.seasonal.values, color='orange', linewidth=1)
        axes[2].set_ylabel('Seasonal')
        axes[2].set_title('Seasonal Component (Annual Pattern)')
        axes[2].grid(True, alpha=0.3)
        
        # Residual
        axes[3].plot(result.resid.index, result.resid.values, color='red', linewidth=0.5, alpha=0.7)
        axes[3].set_ylabel('Residual')
        axes[3].set_title('Residual (Irregular) Component')
        axes[3].set_xlabel('Date')
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Calculate insights
        trend_direction = "Upward" if result.trend.iloc[-1] > result.trend.iloc[0] else "Downward"
        seasonal_range = result.seasonal.max() - result.seasonal.min()
        
        print(f"  Trend: {trend_direction}")
        print(f"  Seasonal variation: {seasonal_range:.4f} (multiplicative factor)")
        
    except Exception as e:
        print(f"  Error in seasonal decomposition: {e}")

def plot_predictions(actual_test, arima_preds, xgb_actual, xgb_preds, ticker):
    """
    Plots predictions vs actual for both models.
    
    Args:
        actual_test: Actual test prices for ARIMA
        arima_preds: ARIMA predictions
        xgb_actual: Actual prices for XGBoost (with lag alignment)
        xgb_preds: XGBoost predictions
        ticker: Stock ticker
    """

    if hasattr(arima_preds, 'values'):
        arima_preds = arima_preds.values
    arima_preds = np.array(arima_preds).flatten()
    
    if hasattr(xgb_preds, 'values'):
        xgb_preds = xgb_preds.values
    xgb_preds = np.array(xgb_preds).flatten()
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # ARIMA Plot
    axes[0].plot(actual_test.index, actual_test.values, label='Actual', color='blue', linewidth=2)
    axes[0].plot(actual_test.index, arima_preds, label='ARIMA Forecast', color='red', linewidth=2, alpha=0.7)
    axes[0].set_title(f'{ticker} - ARIMA Model Predictions vs Actual')
    axes[0].set_ylabel('Price ($)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # XGBoost Plot
    axes[1].plot(xgb_actual.index, xgb_actual.values, label='Actual', color='blue', linewidth=2)
    axes[1].plot(xgb_actual.index, xgb_preds, label='XGBoost Forecast', color='green', linewidth=2, alpha=0.7)
    axes[1].set_title(f'{ticker} - XGBoost Model Predictions vs Actual')
    axes[1].set_ylabel('Price ($)')
    axes[1].set_xlabel('Date')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_portfolio_performance(model_results):
    """
    Visualizes portfolio-level performance comparison.
    
    Args:
        model_results: Dictionary with results for each ticker
    """
    tickers = list(model_results.keys())
    
    arima_rois = [model_results[t]['arima']['roi'] for t in tickers]
    xgb_rois = [model_results[t]['xgboost']['roi'] for t in tickers]
    buy_hold_rois = [model_results[t]['buy_hold'] for t in tickers]
    
    x = np.arange(len(tickers))
    width = 0.25
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # ROI Comparison
    axes[0].bar(x - width, arima_rois, width, label='ARIMA', color='#FF6B6B', alpha=0.8)
    axes[0].bar(x, xgb_rois, width, label='XGBoost', color='#4ECDC4', alpha=0.8)
    axes[0].bar(x + width, buy_hold_rois, width, label='Buy & Hold', color='#95E1D3', alpha=0.8)
    
    axes[0].set_ylabel('ROI (%)', fontsize=12)
    axes[0].set_title('Strategy Performance Comparison by Stock', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(tickers)
    axes[0].legend()
    axes[0].axhline(0, color='black', linewidth=0.8, linestyle='--')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Accuracy Comparison (MAPE)
    arima_mapes = [model_results[t]['arima']['mape'] for t in tickers]
    xgb_mapes = [model_results[t]['xgboost']['mape'] for t in tickers]
    
    axes[1].bar(x - width/2, arima_mapes, width, label='ARIMA', color='#FF6B6B', alpha=0.8)
    axes[1].bar(x + width/2, xgb_mapes, width, label='XGBoost', color='#4ECDC4', alpha=0.8)
    
    axes[1].set_ylabel('MAPE (%)', fontsize=12)
    axes[1].set_title('Prediction Accuracy by Stock (Lower is Better)', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(tickers)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()