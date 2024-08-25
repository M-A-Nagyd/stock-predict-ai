import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Load datasets from CSV files
def load_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['Date'])
    df.set_index('Date', inplace=True)
    return df

# Preprocess data
def preprocess_data(df):
    df = df[['Close']] 
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)
    
    x_train, y_train = [], []
    for i in range(60, len(scaled_data)):
        x_train.append(scaled_data[i-60:i, 0])
        y_train.append(scaled_data[i, 0])
    
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    return x_train, y_train, scaler

# Create and train model
def create_and_train_model(x_train, y_train):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=1, epochs=3)
    return model

# Predict future values
def predict_future_values(model, scaler, df, days_ahead):
    test_data = df[['Close']].tail(60).values if 'Close' in df.columns else df[['price']].tail(60).values
    x_test = [test_data]
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    
    future_values = predictions[0, 0]
    return future_values

# Calculate returns based on user input
def calculate_returns(savings, current_price, future_price):
    return (future_price / current_price) * savings

# Generate investment plan
def generate_investment_plan():
    # Load data
    google_df = load_data('GOOG.csv')
    apple_df = load_data('AAPL.csv')
    bitcoin_df = load_data('BTC-USD.csv')
    gold_df = load_data('FINAL_USO.csv')

    # Preprocess data
    x_train_google, y_train_google, scaler_google = preprocess_data(google_df)
    x_train_apple, y_train_apple, scaler_apple = preprocess_data(apple_df)
    x_train_bitcoin, y_train_bitcoin, scaler_bitcoin = preprocess_data(bitcoin_df)
    x_train_gold, y_train_gold, scaler_gold = preprocess_data(gold_df)

    # Train models
    model_google = create_and_train_model(x_train_google, y_train_google)
    model_apple = create_and_train_model(x_train_apple, y_train_apple)
    model_bitcoin = create_and_train_model(x_train_bitcoin, y_train_bitcoin)
    model_gold = create_and_train_model(x_train_gold, y_train_gold)
    # Get user input
    name = input("Enter your name: ")
    savings = float(input("Enter your savings: "))
    years = int(input("Enter the number of years for investment: "))

    # Predict future values
    future_google = predict_future_values(model_google, scaler_google, google_df, years)
    future_apple = predict_future_values(model_apple, scaler_apple, apple_df, years)
    future_bitcoin = predict_future_values(model_bitcoin, scaler_bitcoin, bitcoin_df, years)
    future_gold = predict_future_values(model_gold, scaler_gold, gold_df, years)

    # Calculate returns
    returns_google = calculate_returns(savings, google_df['Close'].iloc[-1], future_google)
    returns_apple = calculate_returns(savings, apple_df['Close'].iloc[-1], future_apple)
    returns_bitcoin = calculate_returns(savings, bitcoin_df['Close'].iloc[-1], future_bitcoin)
    returns_gold = calculate_returns(savings, gold_df['Close'].iloc[-1], future_gold)

    # Determine best investment
    returns = {
        'Google Stock': returns_google,
        'Apple Stock': returns_apple,
        'Bitcoin': returns_bitcoin,
        'Gold': returns_gold
    }

    best_investment = max(returns, key=returns.get)
    best_return = returns[best_investment]

    # Display results
    print(f"\n{name}, based on your investment, the best option is:")
    print(f"{best_investment}, which could yield approximately ₹{best_return:,.2f} after {years} years.\n")

    # Display datasets for reference
    print("\nDatasets used for predictions:")
    print("Google Stock Data:")
    print(google_df.tail(5))
    print("Apple Stock Data:")
    print(apple_df.tail(5))
    print("Bitcoin Data:")
    print(bitcoin_df.tail(5))
    print("Gold Data:")
    print(gold_df.tail(5))
    
if __name__=='__main__':
    generate_investment_plan()
