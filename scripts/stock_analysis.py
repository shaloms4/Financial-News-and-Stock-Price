import pandas as pd
import numpy as np
import talib
import matplotlib.pyplot as plt
import pynance as pn
from uuid import uuid4

# Step 1: Load and Prepare the Data
def load_stock_data(file_path):
    """
    Load stock price data from a CSV file into a pandas DataFrame.
    Ensure Date is parsed and set as index.
    """
    df = pd.read_csv(file_path, parse_dates=['Date'])
    df.set_index('Date', inplace=True)
    # Ensure required columns are present
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"CSV must contain columns: {required_columns}")
    return df

# Step 2: Apply TA-Lib Indicators
def calculate_technical_indicators(df):
    """
    Calculate technical indicators using TA-Lib:
    - SMA (Simple Moving Average, 20-day)
    - RSI (Relative Strength Index, 14-day)
    - MACD (Moving Average Convergence Divergence)
    """
    # Simple Moving Average (20-day)
    df['SMA20'] = talib.SMA(df['Close'], timeperiod=20)
    
    # Relative Strength Index (14-day)
    df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
    
    # MACD (12, 26, 9)
    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = talib.MACD(
        df['Close'], fastperiod=12, slowperiod=26, signalperiod=9
    )
    
    return df

def calculate_financial_metrics(df):
    """
    Calculate financial metrics using pandas:
    - Daily Returns
    - Cumulative Returns
    """
    df['Daily_Returns'] = df['Close'].pct_change()
    df['Cumulative_Returns'] = (1 + df['Daily_Returns']).cumprod() - 1
    return df


# Step 4: Visualize the Data
def plot_stock_analysis(df, stock_symbol):
    """
    Create visualizations for stock price, indicators, and metrics.
    Displays the plot inline in Jupyter notebook.
    """
    # Create a figure with multiple subplots
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    
    # Plot 1: Close Price and SMA
    ax1.plot(df.index, df['Close'], label='Close Price', color='blue')
    ax1.plot(df.index, df['SMA20'], label='20-Day SMA', color='orange')
    ax1.set_title(f'{stock_symbol} Stock Price and 20-Day SMA')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: RSI
    ax2.plot(df.index, df['RSI'], label='RSI (14)', color='purple')
    ax2.axhline(70, color='red', linestyle='--', alpha=0.5, label='Overbought (70)')
    ax2.axhline(30, color='green', linestyle='--', alpha=0.5, label='Oversold (30)')
    ax2.set_title('Relative Strength Index (RSI)')
    ax2.set_ylabel('RSI')
    ax2.legend()
    ax2.grid(True)
    
    # Plot 3: MACD
    ax3.plot(df.index, df['MACD'], label='MACD', color='blue')
    ax3.plot(df.index, df['MACD_Signal'], label='Signal Line', color='orange')
    ax3.bar(df.index, df['MACD_Hist'], label='MACD Histogram', color='gray', alpha=0.3)
    ax3.set_title('MACD')
    ax3.set_ylabel('MACD')
    ax3.legend()
    ax3.grid(True)
    
    # Plot 4: Daily Returns
    ax4.plot(df.index, df['Daily_Returns'], label='Daily Returns', color='green')
    ax4.set_title('Daily Returns')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Returns')
    ax4.legend()
    ax4.grid(True)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Display the plot inline
    plt.show()
    
    return fig  
