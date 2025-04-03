import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.stattools import coint
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import time
import os

class PairsTrading:
    def __init__(self, stock1, stock2, start_date, end_date):
        """
        Initialize PairsTrading strategy
        
        Parameters:
        stock1, stock2 (str): Stock ticker symbols
        start_date, end_date (str): Date range for analysis in 'YYYY-MM-DD' format
        """
        self.stock1 = stock1
        self.stock2 = stock2
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.zscore = None
        self.hedge_ratio = None
        
    def fetch_data(self):
        """Fetch historical data for both stocks"""
        try:
            # For testing, use sample data
            data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'sample_data.csv')
            if os.path.exists(data_path):
                print(f"Using sample data from {data_path}")
                self.data = pd.read_csv(data_path, index_col='Date', parse_dates=True)
                return self.data
            
            # If sample data doesn't exist, try yfinance
            max_retries = 3
            retry_delay = 5  # seconds
            
            for attempt in range(max_retries):
                try:
                    s1 = yf.download(self.stock1, start=self.start_date, end=self.end_date, progress=False)
                    s2 = yf.download(self.stock2, start=self.start_date, end=self.end_date, progress=False)
                    
                    if s1.empty or s2.empty:
                        raise ValueError(f"No data received for {self.stock1 if s1.empty else self.stock2}")
                    
                    self.data = pd.DataFrame({
                        self.stock1: s1['Adj Close'],
                        self.stock2: s2['Adj Close']
                    })
                    self.data = self.data.dropna()
                    
                    if self.data.empty:
                        raise ValueError("No valid data after processing")
                    
                    return self.data
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"Attempt {attempt + 1} failed. Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        raise Exception(f"Failed to fetch data after {max_retries} attempts: {str(e)}")
                        
        except Exception as e:
            raise Exception(f"Error fetching data: {str(e)}")
    
    def test_cointegration(self):
        """Test for cointegration between the pairs"""
        score, pvalue, _ = coint(self.data[self.stock1], self.data[self.stock2])
        return score, pvalue
    
    def calculate_hedge_ratio(self):
        """Calculate the hedge ratio using OLS regression"""
        self.hedge_ratio = np.polyfit(self.data[self.stock2], self.data[self.stock1], 1)[0]
        return self.hedge_ratio
    
    def calculate_spread(self):
        """Calculate the spread between pairs"""
        if self.hedge_ratio is None:
            self.calculate_hedge_ratio()
        spread = self.data[self.stock1] - self.hedge_ratio * self.data[self.stock2]
        self.zscore = (spread - spread.mean()) / spread.std()
        return self.zscore
    
    def generate_signals(self, zscore_threshold=2.0):
        """
        Generate trading signals based on z-score threshold
        
        Parameters:
        zscore_threshold (float): Z-score threshold for trading signals
        
        Returns:
        DataFrame with positions for both stocks
        """
        if self.zscore is None:
            self.calculate_spread()
            
        signals = pd.DataFrame(index=self.data.index)
        signals['s1_position'] = np.where(self.zscore > zscore_threshold, -1,
                                        np.where(self.zscore < -zscore_threshold, 1, 0))
        signals['s2_position'] = -signals['s1_position'] * self.hedge_ratio
        return signals
    
    def backtest_strategy(self, zscore_threshold=2.0):
        """
        Backtest the pairs trading strategy
        
        Parameters:
        zscore_threshold (float): Z-score threshold for trading signals
        
        Returns:
        DataFrame with strategy returns
        """
        signals = self.generate_signals(zscore_threshold)
        
        # Calculate returns
        s1_returns = self.data[self.stock1].pct_change()
        s2_returns = self.data[self.stock2].pct_change()
        
        # Calculate strategy returns
        strategy_returns = (signals['s1_position'].shift(1) * s1_returns +
                          signals['s2_position'].shift(1) * s2_returns)
        
        # Calculate cumulative returns
        cumulative_returns = (1 + strategy_returns).cumprod()
        
        return pd.DataFrame({
            'Strategy Returns': strategy_returns,
            'Cumulative Returns': cumulative_returns
        })
    
    def plot_strategy(self, zscore_threshold=2.0):
        """Plot the strategy results"""
        results = self.backtest_strategy(zscore_threshold)
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
        
        # Plot stock prices
        ax1.plot(self.data[self.stock1], label=self.stock1)
        ax1.plot(self.data[self.stock2], label=self.stock2)
        ax1.set_title('Stock Prices')
        ax1.legend()
        
        # Plot z-score
        ax2.plot(self.zscore)
        ax2.axhline(zscore_threshold, color='r', linestyle='--')
        ax2.axhline(-zscore_threshold, color='r', linestyle='--')
        ax2.axhline(0, color='k', linestyle='-')
        ax2.set_title('Z-Score of Spread')
        
        # Plot cumulative returns
        ax3.plot(results['Cumulative Returns'])
        ax3.set_title('Cumulative Strategy Returns')
        
        plt.tight_layout()
        plt.show()

def main():
    # Example usage with longer time period and different stock pair
    pairs = PairsTrading('KO', 'PEP',  # Coca-Cola and PepsiCo (common pairs trading example)
                        start_date='2018-01-01',
                        end_date='2023-12-31')
    
    try:
        # Fetch data and perform analysis
        print("Fetching data...")
        pairs.fetch_data()
        
        print("\nTesting cointegration...")
        score, pvalue = pairs.test_cointegration()
        print(f"Cointegration p-value: {pvalue:.4f}")
        
        # Calculate and display hedge ratio
        print("\nCalculating hedge ratio...")
        hedge_ratio = pairs.calculate_hedge_ratio()
        print(f"Hedge ratio: {hedge_ratio:.4f}")
        
        # Generate signals and backtest
        print("\nBacktesting strategy...")
        results = pairs.backtest_strategy(zscore_threshold=2.0)
        
        # Plot results
        print("\nPlotting results...")
        pairs.plot_strategy()
        
        # Display final performance
        final_return = results['Cumulative Returns'].iloc[-1] - 1
        print(f"\nStrategy total return: {final_return:.2%}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    
if __name__ == "__main__":
    main()
