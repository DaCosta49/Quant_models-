import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import time

class StockPredictor:
    def __init__(self, ticker, start_date, end_date, sequence_length=60):
        """
        Initialize Stock Price Predictor
        
        Parameters:
        ticker (str): Stock ticker symbol
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        sequence_length (int): Number of time steps for LSTM
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.sequence_length = sequence_length
        self.data = None
        self.scaler = MinMaxScaler()
        self.rf_model = None
        self.lstm_model = None
        
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
            self.data['SMA_20'] = self.data['Adj Close'].rolling(window=20).mean()
            self.data['SMA_50'] = self.data['Adj Close'].rolling(window=50).mean()
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
                         'Returns', 'SMA_20', 'SMA_50', 'RSI']
        
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
        
        # Prepare sequences for LSTM
        X_train_lstm = []
        y_train_lstm = []
        X_test_lstm = []
        y_test_lstm = []
        
        for i in range(self.sequence_length, len(X_train)):
            X_train_lstm.append(X_train[i-self.sequence_length:i].values)
            y_train_lstm.append(y_train[i])
            
        for i in range(self.sequence_length, len(X_test)):
            X_test_lstm.append(X_test[i-self.sequence_length:i].values)
            y_test_lstm.append(y_test[i])
        
        X_train_lstm = np.array(X_train_lstm)
        y_train_lstm = np.array(y_train_lstm)
        X_test_lstm = np.array(X_test_lstm)
        y_test_lstm = np.array(y_test_lstm)
        
        return {
            'rf': (X_train, X_test, y_train, y_test),
            'lstm': (X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm)
        }
    
    def train_random_forest(self, data):
        """Train Random Forest model"""
        X_train, _, y_train, _ = data['rf']
        
        self.rf_model = RandomForestRegressor(n_estimators=100, 
                                            random_state=42)
        self.rf_model.fit(X_train, y_train)
    
    def train_lstm(self, data):
        """Train LSTM model"""
        X_train, _, y_train, _ = data['lstm']
        
        self.lstm_model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(self.sequence_length, X_train.shape[2])),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(1)
        ])
        
        self.lstm_model.compile(optimizer='adam', loss='mse')
        self.lstm_model.fit(X_train, y_train, 
                          epochs=50, 
                          batch_size=32,
                          verbose=0)
    
    def evaluate_models(self, data):
        """Evaluate both models"""
        _, X_test_rf, _, y_test_rf = data['rf']
        _, X_test_lstm, _, y_test_lstm = data['lstm']
        
        # RF predictions
        rf_pred = self.rf_model.predict(X_test_rf)
        rf_rmse = np.sqrt(mean_squared_error(y_test_rf, rf_pred))
        rf_r2 = r2_score(y_test_rf, rf_pred)
        
        # LSTM predictions
        lstm_pred = self.lstm_model.predict(X_test_lstm)
        lstm_rmse = np.sqrt(mean_squared_error(y_test_lstm, lstm_pred))
        lstm_r2 = r2_score(y_test_lstm, lstm_pred)
        
        return {
            'rf': {'rmse': rf_rmse, 'r2': rf_r2, 'pred': rf_pred},
            'lstm': {'rmse': lstm_rmse, 'r2': lstm_r2, 'pred': lstm_pred}
        }
    
    def plot_predictions(self, data, results):
        """Plot actual vs predicted values"""
        _, X_test_rf, _, y_test_rf = data['rf']
        _, X_test_lstm, _, y_test_lstm = data['lstm']
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Random Forest predictions
        ax1.plot(y_test_rf.index, y_test_rf, label='Actual', alpha=0.7)
        ax1.plot(y_test_rf.index, results['rf']['pred'], label='Predicted (RF)', alpha=0.7)
        ax1.set_title('Random Forest Predictions')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Scaled Price')
        ax1.legend()
        
        # LSTM predictions
        ax2.plot(y_test_lstm.index[-len(y_test_lstm):], y_test_lstm, 
                label='Actual', alpha=0.7)
        ax2.plot(y_test_lstm.index[-len(y_test_lstm):], results['lstm']['pred'], 
                label='Predicted (LSTM)', alpha=0.7)
        ax2.set_title('LSTM Predictions')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Scaled Price')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()

def main():
    try:
        # Example usage with shorter time period and reduced parameters for sample data
        predictor = StockPredictor('AAPL', 
                                 start_date='2023-01-01',
                                 end_date='2023-12-31',
                                 sequence_length=5)  # Reduced for sample data
        
        # Fetch and prepare data
        print("Fetching and preparing data...")
        predictor.fetch_data()
        data = predictor.prepare_data(test_size=0.3)  # Increased test size for small sample
        
        # Train Random Forest model
        print("\nTraining Random Forest model...")
        predictor.train_random_forest(data)
        
        # Train LSTM model with reduced epochs
        print("\nTraining LSTM model...")
        predictor.lstm_model = Sequential([
            LSTM(25, return_sequences=True, input_shape=(predictor.sequence_length, data['lstm'][0].shape[2])),
            Dropout(0.1),
            LSTM(25, return_sequences=False),
            Dropout(0.1),
            Dense(1)
        ])
        predictor.lstm_model.compile(optimizer='adam', loss='mse')
        predictor.lstm_model.fit(data['lstm'][0], data['lstm'][2], 
                               epochs=10,  # Reduced epochs for testing
                               batch_size=8,
                               verbose=1)
        
        # Evaluate and display results
        print("\nEvaluating models...")
        results = predictor.evaluate_models(data)
        
        print("\nModel Performance:")
        print("Random Forest:")
        print(f"RMSE: {results['rf']['rmse']:.4f}")
        print(f"R²: {results['rf']['r2']:.4f}")
        
        print("\nLSTM:")
        print(f"RMSE: {results['lstm']['rmse']:.4f}")
        print(f"R²: {results['lstm']['r2']:.4f}")
        
        # Plot predictions
        print("\nPlotting predictions...")
        predictor.plot_predictions(data, results)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        
if __name__ == "__main__":
    main()
