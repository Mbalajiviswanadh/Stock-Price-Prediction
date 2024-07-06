# TCS Stock Price Prediction 📊

This project is a comprehensive tool for predicting the stock prices of Tata Consultancy Services (TCS) using various machine learning models. The application is built using Streamlit and includes several models like LSTM, Linear Regression, Random Forest Regressor, Gradient Boosting Regressor, Support Vector Regression, K-Nearest Neighbors, and Logistic Regression.

## Features

- **Stock Data Retrieval**: Automatically fetches TCS stock data from Yahoo Finance for the period from 2013 to 2023.
- **Visualization**: Interactive charts for visualizing stock prices, including closing prices and moving averages.
- **Model Training & Prediction**: Multiple models are used to predict stock prices, providing a comparison of performance metrics.
- **Performance Metrics**: Displays R² and RMSE for regression models, and accuracy and confusion matrices for the logistic regression model.

## Models Implemented

1. **LSTM (Long Short-Term Memory)**
2. **Linear Regression**
3. **Random Forest Regressor**
4. **Gradient Boosting Regressor**
5. **Support Vector Regression**
6. **K-Nearest Neighbors Regressor**
7. **Logistic Regression**

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/tcs-stock-prediction.git
   cd tcs-stock-prediction
   ```

2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit application:

   ```bash
   streamlit run app.py
   ```

2. Open the application in your browser and select "TCS" from the dropdown to start the analysis.

## Data Visualization

- **Closing Price vs Time**: Displays the closing price over time.
- **Moving Averages**: Includes 100-day and 200-day moving averages for better trend analysis.

## Prediction Models

### LSTM

- Trained on 70% of the data.
- Visualizes predicted vs actual prices.

### Linear Regression

- Provides R² and RMSE metrics.
- Visualizes true vs predicted values.

### Random Forest Regressor

- Provides R² and RMSE metrics.
- Visualizes true vs predicted values.

### Gradient Boosting Regressor

- Provides R² and RMSE metrics.
- Visualizes true vs predicted values.

### Support Vector Regression

- Provides R² and RMSE metrics.
- Visualizes true vs predicted values.

### K-Nearest Neighbors Regressor

- Provides R² and RMSE metrics.
- Visualizes true vs predicted values.

### Logistic Regression

- Converts stock prices to binary labels.
- Provides accuracy metrics and confusion matrices.
- Visualizes confusion matrices for both training and testing data.

## Conclusion

This project provides an interactive and comprehensive tool for analyzing and predicting TCS stock prices using various machine learning techniques. It offers both visualization and performance metrics to assess the accuracy and reliability of different models.
