import pandas as pd
import numpy as np
from textblob import TextBlob
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from uuid import uuid4

# Step 1: Load and Prepare Data
def load_stock_data(file_path):
    """
    Load stock price data and calculate daily returns.
    Ensures dates are timezone-naive.
    """
    # Load CSV and parse dates
    df = pd.read_csv(file_path)
    
    # Convert to datetime (timezone-naive)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Date'] = df['Date'].dt.normalize()  # Set time to midnight
    df.set_index('Date', inplace=True)
    
    # Calculate daily returns
    df['Daily_Returns'] = df['Close'].pct_change()
    
    print(f"Stock data date range: {df.index.min()} to {df.index.max()}")
    return df[['Daily_Returns']].dropna()

def load_news_data(file_path, save_normalized=False, output_path='normalized_news_data.csv'):
    """
    Load news data, normalize dates, and optionally save to a new CSV.
    Handles multiple date formats and filters out invalid dates.
    Ensures all dates are timezone-naive.
    """
    try:
        # Load CSV
        df = pd.read_csv(file_path)
        print("Sample raw dates from CSV:")
        print(df['date'].head(10))
        
        # Convert 'date' column to datetime and handle timezone information
        df['Date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Convert timezone-aware dates to UTC and then remove timezone info
        df['Date'] = df['Date'].dt.tz_localize(None)
        
        # Filter out rows with invalid dates
        invalid_dates = df['Date'].isna()
        if invalid_dates.any():
            print(f"\nWarning: {invalid_dates.sum()} rows have invalid dates and will be removed")
            print("Sample of rows with invalid dates:")
            print(df[invalid_dates][['date', 'headline']].head())
            # Remove rows with invalid dates
            df = df[~invalid_dates].copy()
        
        # Normalize to date (remove time component)
        df['Date'] = df['Date'].dt.normalize()
        
        print("\nSample normalized dates:")
        print(df['Date'].head(10))
        print(f"\nDate range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"Total valid rows: {len(df)}")
        
        # Optionally save normalized data
        if save_normalized:
            normalized_df = df[['Date', 'headline', 'stock']]
            normalized_df.to_csv(output_path, index=False)
            print(f"\nNormalized news data saved to {output_path}")
        
        return df[['Date', 'headline', 'stock']]
        
    except Exception as e:
        print(f"Error loading news data: {e}")
        raise

# Step 2: Sentiment Analysis
def calculate_sentiment(headline):
    """
    Calculate sentiment score for a headline using TextBlob.
    Returns polarity score (-1 to 1: negative to positive).
    """
    analysis = TextBlob(str(headline))  # Ensure headline is string
    return analysis.sentiment.polarity

# Step 3: Aggregate Sentiments and Align with Stock Data
def prepare_combined_data(stock_df, news_df, stock_symbol):
    """
    Aggregate daily sentiment scores and align with stock returns.
    """
    # Filter news for the specified stock
    news_df = news_df[news_df['stock'] == stock_symbol].copy()  # Create a copy to avoid SettingWithCopyWarning
    print(f"\nFound {len(news_df)} news articles for {stock_symbol}")
    
    # Calculate sentiment for each headline
    news_df['Sentiment'] = news_df['headline'].apply(calculate_sentiment)
    
    # Aggregate sentiment scores by date
    daily_sentiment = news_df.groupby('Date')['Sentiment'].agg(['mean', 'count']).reset_index()
    daily_sentiment.set_index('Date', inplace=True)
    
    print(f"\nDate ranges:")
    print(f"Stock data: {stock_df.index.min()} to {stock_df.index.max()}")
    print(f"News data: {daily_sentiment.index.min()} to {daily_sentiment.index.max()}")
    
    # Merge stock returns and sentiment data
    combined_df = stock_df.join(daily_sentiment, how='inner')
    
    print(f"\nAlignment statistics:")
    print(f"Total stock data points: {len(stock_df)}")
    print(f"Total news data points: {len(daily_sentiment)}")
    print(f"Aligned data points: {len(combined_df)}")
    
    return combined_df.dropna()

# Step 4: Correlation Analysis
def calculate_correlation(df):
    """
    Calculate Pearson correlation between sentiment scores and daily returns.
    Uses the 'mean' column which contains the average daily sentiment scores.
    """
    correlation, p_value = pearsonr(df['Daily_Returns'], df['mean'])
    return correlation, p_value

# Step 5: Visualize Correlation
def plot_correlation(df, stock_symbol, correlation, p_value):
    """
    Create a scatter plot of sentiment vs. daily returns.
    """
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot
    plt.scatter(df['mean'], df['Daily_Returns'], alpha=0.5, label='Data points')
    
    # Add trend line
    z = np.polyfit(df['mean'], df['Daily_Returns'], 1)
    p = np.poly1d(z)
    plt.plot(df['mean'], p(df['mean']), "r--", label='Trend line')
    
    # Add labels and title
    plt.title(f'{stock_symbol} Sentiment vs. Daily Returns\nCorrelation: {correlation:.3f} (p-value: {p_value:.3f})')
    plt.xlabel('Average Daily Sentiment Score')
    plt.ylabel('Daily Stock Returns')
    plt.grid(True)
    plt.legend()
    
    # Add text box with statistics
    stats_text = f'Number of data points: {len(df)}\n'
    stats_text += f'Average articles per day: {df["count"].mean():.1f}\n'
    stats_text += f'Date range: {df.index.min().date()} to {df.index.max().date()}'
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    plot_filename = f'{stock_symbol}_sentiment_correlation.png'
    plt.savefig(plot_filename)
    plt.close()
    return plot_filename

# Main function
def main():
    # File paths
    stock_file = './yfinance_data/AAPL_historical_data.csv'  # Replace with your stock CSV path
    news_file = './raw_analyst_ratings.csv'  # Replace with your news CSV path
    stock_symbol = 'A'  # Stock symbol in news data (e.g., 'A' for Agilent)
    
    # Load data
    stock_df = load_stock_data(stock_file)
    news_df = load_news_data(news_file, save_normalized=True, output_path='normalized_news_data.csv')
    
    # Prepare combined data
    combined_df = prepare_combined_data(stock_df, news_df, stock_symbol)
    
    # Calculate correlation
    correlation, p_value = calculate_correlation(combined_df)
    print(f"Correlation between sentiment and daily returns for {stock_symbol}: {correlation:.3f}")
    print(f"P-value: {p_value:.3f}")
    
    # Visualize results
    plot_file = plot_correlation(combined_df, stock_symbol, correlation, p_value)
    print(f"Correlation plot saved to {plot_file}")
    
    # Save combined data
    output_file = f'{stock_symbol}_sentiment_returns.csv'
    combined_df.to_csv(output_file)
    print(f"Combined data saved to {output_file}")

if __name__ == "__main__":
    main()