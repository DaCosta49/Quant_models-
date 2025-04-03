import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import time

class StockPredictor:
    def __init__(self, ticker, start_date, end_date):
        """
        Initialize Stock Price Predictor
        
        Parameters:
        ticker (str): Stock ticker symbol
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.scaler = MinMaxScaler()
        self.rf_model = None
        
    def fetch_data(self):
        """Fetch and prepare stock data"""
        try:
            # First try to use sample data
            data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'stock_data.csv')
            if os.path.exists(data_path):
                print(f"Using sample data from {data_path}")
                self.data = pd.read_csv(data_path, index_col='Date', parse_dates=True)
            else:
                # Fallback to yfinance with retry logic
                max_retries = 3
                retry_delay = 5
                
                for attempt in range(max_retries):
                    try:
                        self.data = yf.download(self.ticker, 
                                              start=self.start_date, 
                                              end=self.end_date, 
                                              progress=False)
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
            
            # Calculate technical indicators
            self.data['Returns'] = self.data['Adj Close'].pct_change()
            self.data['SMA_5'] = self.data['Adj Close'].rolling(window=5).mean()
            self.data['RSI'] = self._calculate_rsi(self.data['Adj Close'])
            
            # Drop NaN values
            self.data = self.data.dropna()
            
            if self.data.empty:
                raise ValueError("No valid data after calculating indicators")
                
            return self.data
            
        except Exception as e:
            raise Exception(f"Error fetching data: {str(e)}")
    
    @staticmethod
    def _calculate_rsi(prices, period=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def prepare_data(self, target_col='Adj Close', test_size=0.2):
        """Prepare data for ML models"""
        # Features for RF
        feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 
                         'Returns', 'SMA_5', 'RSI']
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(self.data[feature_columns + [target_col]])
        scaled_df = pd.DataFrame(scaled_data, 
                               columns=feature_columns + [target_col],
                               index=self.data.index)
        
        # Prepare features and target
        X = scaled_df[feature_columns]
        y = scaled_df[target_col]
        
        # Split into train and test sets
        train_size = int(len(X) * (1 - test_size))
        X_train = X[:train_size]
        X_test = X[train_size:]
        y_train = y[:train_size]
        y_test = y[train_size:]
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train, y_train):
        """Train Random Forest model"""
        self.rf_model = RandomForestRegressor(n_estimators=100, 
                                            random_state=42)
        self.rf_model.fit(X_train, y_train)
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate the model"""
        predictions = self.rf_model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)
        return predictions, rmse, r2
    
    def plot_predictions(self, y_test, predictions):
        """Plot actual vs predicted values"""
        plt.figure(figsize=(12, 6))
        plt.plot(y_test.index, y_test, label='Actual', alpha=0.7)
        plt.plot(y_test.index, predictions, label='Predicted', alpha=0.7)
        plt.title('Random Forest Stock Price Predictions')
        plt.xlabel('Date')
        plt.ylabel('Scaled Price')
        plt.legend()
        plt.show()

def main():
    try:
        # Example usage
        predictor = StockPredictor('AAPL', 
                                 start_date='2023-01-01',
                                 end_date='2023-12-31')
        
        # Fetch and prepare data
        print("Fetching and preparing data...")
        predictor.fetch_data()
        X_train, X_test, y_train, y_test = predictor.prepare_data(test_size=0.3)
        
        # Train Random Forest model
        print("\nTraining Random Forest model...")
        predictor.train_model(X_train, y_train)
        
        # Evaluate and display results
        print("\nEvaluating model...")
        predictions, rmse, r2 = predictor.evaluate_model(X_test, y_test)
        
        print("\nModel Performance:")
        print(f"RMSE: {rmse:.4f}")
        print(f"R²: {r2:.4f}")
        
        # Plot predictions
        print("\nPlotting predictions...")
        predictor.plot_predictions(y_test, predictions)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
