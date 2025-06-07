# Financial News Sentiment Analysis

This repository contains the code and analysis for the Nova Financial Solutions challenge, focusing on sentiment analysis of financial news and its correlation with stock price movements.

## Project Structure
- `src/`: Source code for data processing and analysis.
- `notebooks/`: Jupyter notebooks for exploratory data analysis and correlation studies.
- `tests/`: Unit tests for the code.
- `scripts/`: Utility scripts for automation and analysis.
- `yfinance_data/`: Historical stock price data.
- `.github/workflows/`: CI/CD pipelines.

## Setup
1. Clone the repository.
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Unix: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`

## Features

### News Sentiment Analysis
- Load and process financial news data
- Calculate sentiment scores using TextBlob
- Handle multiple date formats and timezone normalization
- Aggregate daily sentiment scores

### Stock Price Analysis
- Load historical stock price data
- Calculate daily returns
- Timezone-naive date handling for consistent analysis

### Correlation Analysis
- Analyze correlation between news sentiment and stock price movements
- Calculate Pearson correlation coefficients and p-values
- Generate visualizations of sentiment vs. stock returns
- Support for multiple stocks and time periods

## Usage

### Using the Script
```python
from scripts.correlation_and_stock_movement import (
    load_stock_data,
    load_news_data,
    prepare_combined_data,
    calculate_correlation,
    plot_correlation
)

# Load data
stock_df = load_stock_data('path/to/stock_data.csv')
news_df = load_news_data('path/to/news_data.csv')

# Analyze correlation
combined_df = prepare_combined_data(stock_df, news_df, 'STOCK_SYMBOL')
correlation, p_value = calculate_correlation(combined_df)

# Visualize results
plot_correlation(combined_df, 'STOCK_SYMBOL', correlation, p_value)
```

### Using the Notebook
1. Navigate to the `notebooks` directory
2. Open `correlation_and_stock_movement.ipynb`
3. Follow the step-by-step analysis in the notebook

## Data Requirements
- Stock data should be in CSV format with columns: Date, Close
- News data should be in CSV format with columns: date, headline, stock

## Output
- Correlation analysis results
- Visualization plots
- Normalized news data (optional)
- Combined sentiment and returns data

## Dependencies
- pandas
- numpy
- textblob
- scipy
- matplotlib
- yfinance (for stock data)



