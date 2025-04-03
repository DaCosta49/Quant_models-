import numpy as np
import pandas as pd
import yfinance as yf
from arch import arch_model
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import time

class GARCHModel:
    def __init__(self, ticker, start_date, end_date):
        """
        Initialize GARCH model for volatility forecasting
        
        Parameters:
        ticker (str): Stock ticker symbol
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.returns = None
        self.model = None
        self.results = None
        
    def fetch_data(self):
        """Fetch historical price data and calculate returns"""
        try:
            # First try to use sample data
            data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'stock_data.csv')
            if os.path.exists(data_path):
                print(f"Using sample data from {data_path}")
                data = pd.read_csv(data_path, index_col='Date', parse_dates=True)
                self.data = data['Adj Close']
            else:
                # Fallback to yfinance with retry logic
                max_retries = 3
                retry_delay = 5
                
                for attempt in range(max_retries):
                    try:
                        self.data = yf.download(self.ticker, 
                                              start=self.start_date, 
                                              end=self.end_date, 
                                              progress=False)['Adj Close']
                        break
                    except Exception as e:
                        if attempt < max_retries - 1:
                            print(f"Attempt {attempt + 1} failed. Retrying in {retry_delay} seconds...")
                            time.sleep(retry_delay)
                            retry_delay *= 2
                        else:
                            raise Exception(f"Failed to fetch data after {max_retries} attempts: {str(e)}")
            
            if self.data.empty:
                raise ValueError("No data available for the specified period")
                
            self.returns = 100 * self.data.pct_change().dropna()
            return self.returns
            
        except Exception as e:
            raise Exception(f"Error fetching data: {str(e)}")
        
    def fit_model(self, p=1, q=1):
        """
        Fit GARCH(p,q) model to returns data
        
        Parameters:
        p (int): Order of ARCH term
        q (int): Order of GARCH term
        """
        self.model = arch_model(self.returns, 
                              vol='Garch', 
                              p=p, 
                              q=q,
                              mean='Zero',
                              dist='normal')
        self.results = self.model.fit(disp='off')
        return self.results
    
    def forecast_volatility(self, horizon=10):
        """
        Forecast volatility for specified horizon
        
        Parameters:
        horizon (int): Number of days to forecast
        
        Returns:
        DataFrame with forecasted volatility
        """
        forecast = self.results.forecast(horizon=horizon)
        return np.sqrt(forecast.variance.values[-1])
    
    def plot_volatility(self):
        """Plot historical volatility and GARCH volatility"""
        if self.results is None:
            raise ValueError("Model must be fit before plotting")
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot returns
        ax1.plot(self.returns)
        ax1.set_title(f'{self.ticker} Returns')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Returns (%)')
        
        # Plot conditional volatility
        conditional_vol = pd.Series(
            np.sqrt(self.results.conditional_volatility),
            index=self.returns.index
        )
        
        ax2.plot(conditional_vol)
        ax2.set_title('GARCH Conditional Volatility')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Volatility (%)')
        
        plt.tight_layout()
        plt.show()
        
    def summary(self):
        """Print model summary"""
        print(self.results.summary())
        
    def calculate_var(self, confidence_level=0.95):
        """
        Calculate Value at Risk (VaR) using GARCH volatility
        
        Parameters:
        confidence_level (float): Confidence level for VaR calculation
        
        Returns:
        float: VaR estimate
        """
        from scipy.stats import norm
        
        # Get latest volatility estimate
        latest_vol = np.sqrt(self.results.conditional_volatility[-1])
        
        # Calculate VaR
        var = norm.ppf(1 - confidence_level) * latest_vol
        return var
    
    def plot_qq(self):
        """Plot Q-Q plot of standardized residuals"""
        from scipy import stats
        
        standardized_residuals = self.results.resid / np.sqrt(self.results.conditional_volatility)
        
        plt.figure(figsize=(8, 6))
        stats.probplot(standardized_residuals, dist="norm", plot=plt)
        plt.title("Q-Q plot of Standardized Residuals")
        plt.show()

def main():
    # Example usage with shorter time period
    garch = GARCHModel('AAPL', '2023-01-01', '2023-12-31')
    
    try:
        # Fetch and prepare data
        print("Fetching data...")
        returns = garch.fetch_data()
        print(f"Number of observations: {len(returns)}")
        
        # Fit GARCH(1,1) model
        print("\nFitting GARCH(1,1) model...")
        results = garch.fit_model(p=1, q=1)
        print("\nModel Parameters:")
        print(results.params)
        
        # Plot volatility
        print("\nPlotting volatility...")
        garch.plot_volatility()
        
        # Calculate and print VaR
        var_95 = garch.calculate_var(0.95)
        print(f"\n95% Value at Risk: {var_95:.2f}%")
        
        # Forecast volatility
        print("\nForecasting volatility...")
        forecast = garch.forecast_volatility(horizon=5)  # Reduced horizon for sample data
        print(f"\nForecasted volatility for next 5 days:")
        print(forecast)
        
        # Plot Q-Q plot
        print("\nGenerating Q-Q plot...")
        garch.plot_qq()
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
