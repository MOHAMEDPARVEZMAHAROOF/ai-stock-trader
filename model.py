"""
AI Stock Trading Model with LSTM Deep Learning
Comprehensive model for stock price prediction and trading signals
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, GRU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

class StockTradingModel:
    """Advanced LSTM-based Stock Trading Model"""
    
    def __init__(self, ticker, lookback=60, epochs=100, batch_size=32):
        self.ticker = ticker
        self.lookback = lookback
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.history = None
        
    def fetch_data(self, start_date='2015-01-01', end_date=None):
        """Fetch stock data from Yahoo Finance"""
        print(f"Fetching data for {self.ticker}...")
        data = yf.download(self.ticker, start=start_date, end=end_date, progress=False)
        
        # Add technical indicators
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['SMA_200'] = data['Close'].rolling(window=200).mean()
        data['EMA_12'] = data['Close'].ewm(span=12).mean()
        data['EMA_26'] = data['Close'].ewm(span=26).mean()
        data['MACD'] = data['EMA_12'] - data['EMA_26']
        data['RSI'] = self.calculate_rsi(data['Close'], 14)
        data['Volatility'] = data['Close'].rolling(window=10).std()
        data['Volume_Change'] = data['Volume'].pct_change()
        
        # Drop NaN values
        data = data.dropna()
        
        print(f"Data shape: {data.shape}")
        return data
    
    def calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def prepare_data(self, data):
        """Prepare data for LSTM model"""
        # Select features
        features = ['Close', 'Volume', 'SMA_50', 'SMA_200', 'MACD', 'RSI', 'Volatility']
        df = data[features].copy()
        
        # Normalize data
        scaled_data = self.scaler.fit_transform(df)
        
        X, y = [], []
        for i in range(self.lookback, len(scaled_data)):
            X.append(scaled_data[i-self.lookback:i])
            y.append(scaled_data[i, 0])  # Predict Close price
        
        X, y = np.array(X), np.array(y)
        
        # Split data
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
        return X_train, X_test, y_train, y_test
    
    def build_model(self, X_train):
        """Build advanced LSTM model"""
        model = Sequential([
            # First Bidirectional LSTM layer
            Bidirectional(LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]))),
            Dropout(0.3),
            
            # Second LSTM layer
            LSTM(64, return_sequences=True),
            Dropout(0.3),
            
            # Third LSTM layer
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            
            # Dense layers
            Dense(25, activation='relu'),
            Dropout(0.2),
            Dense(1)
        ])
        
        # Compile with Adam optimizer
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        print(model.summary())
        return model
    
    def train(self, X_train, y_train, X_test, y_test):
        """Train the LSTM model with callbacks"""
        # Build model
        self.model = self.build_model(X_train)
        
        # Callbacks for better training
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1)
        ]
        
        # Train model
        print("Training model...")
        self.history = self.model.fit(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1
        )
        
        print("Training completed!")
        return self.history
    
    def predict(self, X_test):
        """Make predictions"""
        predictions = self.model.predict(X_test)
        return predictions
    
    def generate_signals(self, data, predictions):
        """Generate buy/sell signals based on predictions"""
        signals = pd.DataFrame(index=data.index[-len(predictions):])
        signals['Predicted_Price'] = predictions.flatten()
        signals['Actual_Price'] = data['Close'].values[-len(predictions):]
        
        # Calculate signal (1 = BUY, -1 = SELL, 0 = HOLD)
        signals['Signal'] = 0
        
        # Buy when predicted price is higher than current price by 2%
        signals.loc[signals['Predicted_Price'] > signals['Actual_Price'] * 1.02, 'Signal'] = 1
        
        # Sell when predicted price is lower than current price by 2%
        signals.loc[signals['Predicted_Price'] < signals['Actual_Price'] * 0.98, 'Signal'] = -1
        
        # Add confidence score
        signals['Confidence'] = abs(signals['Predicted_Price'] - signals['Actual_Price']) / signals['Actual_Price'] * 100
        
        return signals
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        loss, mae = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"\\nModel Evaluation:")
        print(f"Test Loss (MSE): {loss:.6f}")
        print(f"Test MAE: {mae:.6f}")
        
        # Calculate accuracy
        predictions = self.predict(X_test)
        actual = y_test
        
        # Direction accuracy (up/down prediction)
        pred_direction = np.diff(predictions.flatten()) > 0
        actual_direction = np.diff(actual) > 0
        direction_accuracy = np.mean(pred_direction == actual_direction) * 100
        
        print(f"Direction Accuracy: {direction_accuracy:.2f}%")
        return {'loss': loss, 'mae': mae, 'direction_accuracy': direction_accuracy}
    
    def save_model(self, filepath='stock_model.h5'):
        """Save trained model"""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_trained_model(self, filepath='stock_model.h5'):
        """Load pre-trained model"""
        self.model = load_model(filepath)
        print(f"Model loaded from {filepath}")


def main():
    """Main execution function"""
    # Example usage
    ticker = "RELIANCE.NS"  # Reliance Industries (NSE)
    
    # Initialize model
    trading_model = StockTradingModel(ticker=ticker, lookback=60, epochs=100, batch_size=32)
    
    # Fetch data
    data = trading_model.fetch_data(start_date='2015-01-01')
    
    # Prepare data
    X_train, X_test, y_train, y_test = trading_model.prepare_data(data)
    
    # Train model
    history = trading_model.train(X_train, y_train, X_test, y_test)
    
    # Evaluate
    metrics = trading_model.evaluate(X_test, y_test)
    
    # Make predictions
    predictions = trading_model.predict(X_test)
    
    # Denormalize predictions
    predictions_denorm = trading_model.scaler.inverse_transform(
        np.concatenate([predictions, np.zeros((len(predictions), 6))], axis=1)
    )[:, 0]
    
    # Generate trading signals
    signals = trading_model.generate_signals(data, predictions_denorm)
    
    # Display recent signals
    print("\\n=== Recent Trading Signals ===")
    print(signals.tail(10))
    
    # Save model
    trading_model.save_model()
    
    print("\\n=== Model Training Complete ===")
    print(f"Final Direction Accuracy: {metrics['direction_accuracy']:.2f}%")


if __name__ == "__main__":
    main()
