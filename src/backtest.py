# =============================================================================
# BACKTESTING MODULE (REALISTIC EXECUTION)
# =============================================================================

import pandas as pd
import numpy as np

def run_backtest(predictions, test_df, initial_capital=10000, transaction_cost=0.0):
    """
    Runs realistic backtest for a single stock.
    
    Key improvements:
    - Uses NEXT day's OPEN price for execution (not close)
    - Compares prediction to current close, executes at next open
    - Includes optional transaction costs
    
    Args:
        predictions: Array/Series of predicted next-day prices
        test_df: DataFrame with actual test data (must have Open, Close)
        initial_capital: Starting cash
        transaction_cost: Percentage fee per trade (e.g., 0.001 = 0.1%)
    
    Returns:
        backtest_df: DataFrame with portfolio values
        roi: Return on investment (%)
    """
    # Align predictions with test data
    if hasattr(predictions, 'values'):
        predictions = predictions.values
    predictions = np.array(predictions).flatten()
    
    # Align predictions with test data
    pred_series = pd.Series(predictions, index=test_df.index[:len(predictions)])
    
    # Prepare backtest dataframe
    df_bt = pd.DataFrame(index=pred_series.index)
    df_bt['Predicted_Price'] = pred_series.values
    df_bt['Current_Close'] = test_df.loc[pred_series.index, 'Close'].values
    
    # Get next day's open price for execution
    next_opens = []
    for i, idx in enumerate(pred_series.index):
        if i < len(pred_series) - 1:
            next_idx = pred_series.index[i + 1]
            next_opens.append(test_df.loc[next_idx, 'Open'])
        else:
            # Last day: use close as approximation
            next_opens.append(test_df.loc[idx, 'Close'])
    
    df_bt['Next_Open'] = next_opens
    
    # Trading logic
    cash = initial_capital
    shares = 0
    portfolio_values = []
    positions = []
    
    for i in range(len(df_bt)):
        pred = df_bt['Predicted_Price'].iloc[i]
        curr_close = df_bt['Current_Close'].iloc[i]
        next_open = df_bt['Next_Open'].iloc[i]
        
        # Decision made at close, executed at next open
        if pred > curr_close * 1.01:  # Buy signal: predict >1% upside
            if cash > 0:
                # BUY at next open
                shares = cash / next_open
                shares = shares * (1 - transaction_cost)  # Apply cost
                cash = 0
                positions.append('LONG')
            else:
                positions.append('HOLD')
        
        elif pred < curr_close * 0.99:  # Sell signal: predict >1% downside
            if shares > 0:
                # SELL at next open
                cash = shares * next_open
                cash = cash * (1 - transaction_cost)  # Apply cost
                shares = 0
                positions.append('CASH')
            else:
                positions.append('HOLD')
        else:
            positions.append('HOLD')
        
        # Portfolio value at current close
        portfolio_value = cash + (shares * curr_close)
        portfolio_values.append(portfolio_value)
    
    df_bt['Portfolio_Value'] = portfolio_values
    df_bt['Position'] = positions
    
    # Calculate ROI
    final_value = portfolio_values[-1]
    roi = ((final_value - initial_capital) / initial_capital) * 100
    
    return df_bt, roi

def calculate_portfolio_metrics(model_results):
    """
    Calculates aggregate portfolio metrics across all stocks.
    
    Args:
        model_results: Dict with results for each ticker
    
    Returns:
        Dictionary with portfolio-level metrics
    """
    arima_rois = []
    xgb_rois = []
    buy_hold_rois = []
    
    arima_rmses = []
    arima_mapes = []
    xgb_rmses = []
    xgb_mapes = []
    
    for ticker, results in model_results.items():
        arima_rois.append(results['arima']['roi'])
        xgb_rois.append(results['xgboost']['roi'])
        buy_hold_rois.append(results['buy_hold'])
        
        arima_rmses.append(results['arima']['rmse'])
        arima_mapes.append(results['arima']['mape'])
        xgb_rmses.append(results['xgboost']['rmse'])
        xgb_mapes.append(results['xgboost']['mape'])
    
    # Portfolio metrics
    metrics = {
        'arima': {
            'avg_rmse': np.mean(arima_rmses),
            'avg_mape': np.mean(arima_mapes),
            'portfolio_roi': np.mean(arima_rois),  # Equal-weighted
            'win_rate': (np.array(arima_rois) > 0).sum() / len(arima_rois) * 100,
            'best_stock': max(model_results.items(), key=lambda x: x[1]['arima']['roi'])[0],
            'worst_stock': min(model_results.items(), key=lambda x: x[1]['arima']['roi'])[0]
        },
        'xgboost': {
            'avg_rmse': np.mean(xgb_rmses),
            'avg_mape': np.mean(xgb_mapes),
            'portfolio_roi': np.mean(xgb_rois),
            'win_rate': (np.array(xgb_rois) > 0).sum() / len(xgb_rois) * 100,
            'best_stock': max(model_results.items(), key=lambda x: x[1]['xgboost']['roi'])[0],
            'worst_stock': min(model_results.items(), key=lambda x: x[1]['xgboost']['roi'])[0]
        },
        'buy_hold': {
            'portfolio_roi': np.mean(buy_hold_rois),
            'win_rate': (np.array(buy_hold_rois) > 0).sum() / len(buy_hold_rois) * 100
        }
    }
    
    return metrics