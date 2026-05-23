import warnings
import os
import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

# ==========================================
# 1. LOAD AND RESAMPLE DATA
# ==========================================
def load_data():
    """Loads household power consumption data."""
    # Try to use a reliable raw link
    url = "https://raw.githubusercontent.com/LearnHit-Data-Science/Time-Series/master/data/household_power.csv"
    file_name = "household_power_consumption.csv"
    
    if not os.path.exists(file_name):
        print("Downloading Dataset...")
        try:
            urllib.request.urlretrieve(url, file_name)
        except:
            print("Download failed, creating simulated data...")
            # Fallback: Simulate data if download fails
            date_rng = pd.date_range(start='1/1/2020', end='1/01/2022', freq='H')
            values = np.random.randn(len(date_rng)) * 20 + 50 + np.sin(np.linspace(0, 50, len(date_rng)))
            df = pd.DataFrame({'datetime': date_rng, 'Global_active_power': values})
            df.to_csv(file_name, index=False)
            return df
    else:
        print("Dataset found locally.")

    # Load Data
    df = pd.read_csv(file_name)
    
    # Ensure datetime column exists
    if 'datetime' not in df.columns:
        # Adjust column names based on typical UCI format if necessary
        # Assuming standard format: Date, Time, Global_active_power...
        # For this generic script, we assume the file has a datetime column or we generate one.
        pass
        
    # If using the raw github link above, columns are usually standard.
    # Let's parse datetime
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    
    # Handle missing values by forward fill
    df.fillna(method='ffill', inplace=True)
    
    # Resample to HOURLY to speed up training (Minute data is too heavy for simple scripts)
    print("Resampling data to Hourly frequency...")
    df_resampled = df.resample('H').mean() # Average usage per hour
    df_resampled = df_resampled.dropna()
    
    return df_resampled

print("="*50)
print("STEP 1: Loading Data...")
print("="*50)
df = load_data()
print(f"Dataset Shape after Resampling: {df.shape}")
print(df.tail())

# ==========================================
# 2. FEATURE ENGINEERING
# ==========================================
print("\n" + "="*50)
print("STEP 2: Feature Engineering...")
print("="*50)

def create_features(df):
    """Create time-based features."""
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['dayofyear'] = df.index.dayofyear
    
    # Weekend flag
    df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
    
    # Lag Features (Crucial for XGBoost)
    df['lag_1h'] = df['Global_active_power'].shift(1)
    df['lag_24h'] = df['Global_active_power'].shift(24) # Previous day
    df['rolling_mean_24h'] = df['Global_active_power'].rolling(window=24).mean()
    
    return df

df = create_features(df)
df.dropna(inplace=True)

print("Features created:")
print(df.head())

# ==========================================
# 3. PREPARE DATA FOR MODELING
# ==========================================
print("\n" + "="*50)
print("STEP 3: Splitting Data...")
print("="*50)

# Use the last 60 hours (approx 2-3 days) as test set for visualization
test_size = 60
train = df[:-test_size]
test = df[-test_size:]

# Features for XGBoost (Supervised)
feature_cols = ['hour', 'dayofweek', 'quarter', 'month', 'is_weekend', 'lag_1h', 'lag_24h', 'rolling_mean_24h']
X_train = train[feature_cols]
y_train = train['Global_active_power']
X_test = test[feature_cols]
y_test = test['Global_active_power']

print(f"Training Size: {len(train)}, Test Size: {len(test)}")

# ==========================================
# 4. MODEL 1: XGBOOST (Supervised)
# ==========================================
print("\n" + "="*50)
print("STEP 4: Training XGBoost...")
print("="*50)

xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
print(f"XGBoost Results -> MAE: {mae_xgb:.4f}, RMSE: {rmse_xgb:.4f}")

# ==========================================
# 5. MODEL 2: ARIMA (Classical)
# ==========================================
print("\n" + "="*50)
print("STEP 5: Training ARIMA...")
print("="*50)

# We need the raw series for ARIMA
train_series = train['Global_active_power']

# Fit ARIMA
# Using a simple order (p,d,q) approximation. 
# For hourly data, ARIMA is heavy. We fit on a subset or simple model.
# Using p=5, d=1, q=2 as a quick baseline.
from statsmodels.tsa.arima.model import ARIMA

model_arima = ARIMA(train_series, order=(5, 1, 2))
model_arima_fit = model_arima.fit()

# Forecast
forecast_arima = model_arima_fit.forecast(steps=test_size)
y_pred_arima = forecast_arima.values # Access values

mae_arima = mean_absolute_error(y_test, y_pred_arima)
rmse_arima = np.sqrt(mean_squared_error(y_test, y_pred_arima))
print(f"ARIMA Results -> MAE: {mae_arima:.4f}, RMSE: {rmse_arima:.4f}")

# ==========================================
# 6. MODEL 3: PROPHET (Facebook)
# ==========================================
print("\n" + "="*50)
print("STEP 6: Training Prophet...")
print("="*50)

try:
    from prophet import Prophet
    
    # Prophet requires specific column names 'ds' and 'y'
    prophet_train = train.reset_index()[['datetime', 'Global_active_power']]
    prophet_train.columns = ['ds', 'y']
    
    m_prophet = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
    m_prophet.fit(prophet_train)
    
    # Create future dataframe
    future = test.reset_index()[['datetime']]
    future.columns = ['ds']
    forecast_prophet = m_prophet.predict(future)
    y_pred_prophet = forecast_prophet['yhat'].values
    
    mae_prophet = mean_absolute_error(y_test, y_pred_prophet)
    rmse_prophet = np.sqrt(mean_squared_error(y_test, y_pred_prophet))
    print(f"Prophet Results -> MAE: {mae_prophet:.4f}, RMSE: {rmse_prophet:.4f}")
    has_prophet = True

except ImportError:
    print("Prophet not installed. Skipping...")
    has_prophet = False
    y_pred_prophet = np.full(test_size, np.nan)

# ==========================================
# 7. VISUALIZATION & MODEL COMPARISON
# ==========================================
print("\n" + "="*50)
print("STEP 7: Visualization...")
print("="*50)

# Plot 1: Model Comparison Bar Chart
plt.figure(figsize=(10, 6))
models = ['XGBoost', 'ARIMA', 'Prophet' if has_prophet else 'Prophet(NA)']
maes = [mae_xgb, mae_arima, mae_prophet if has_prophet else 0]
rmses = [rmse_xgb, rmse_arima, rmse_prophet if has_prophet else 0]

x = np.arange(len(models))
width = 0.35

plt.bar(x - width/2, maes, width, label='MAE')
plt.bar(x + width/2, rmses, width, label='RMSE')
plt.xticks(x, models)
plt.ylabel('Error')
plt.title('Model Performance Comparison')
plt.legend()
plt.tight_layout()
plt.savefig("Model_Comparison.png")
plt.show()

# Plot 2: Actual vs Forecasted (Time Series)
plt.figure(figsize=(14, 7))
plt.plot(test.index, y_test.values, label='Actual', color='black', linewidth=2)
plt.plot(test.index, y_pred_xgb, label=f'XGBoost (MAE={mae_xgb:.2f})', linestyle='--')
plt.plot(test.index, y_pred_arima, label=f'ARIMA (MAE={mae_arima:.2f})', linestyle=':')
if has_prophet:
    plt.plot(test.index, y_pred_prophet, label=f'Prophet (MAE={mae_prophet:.2f})', linestyle='-.')

plt.title('Actual vs Forecasted Energy Consumption')
plt.xlabel('Time')
plt.ylabel('Global Active Power (kW)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("Actual_vs_Forecast.png")
plt.show()

# Plot 3: XGBoost Feature Importance
plt.figure(figsize=(8, 5))
plt.barh(feature_cols, xgb_model.feature_importances_)
plt.xlabel('Feature Importance')
plt.title('XGBoost Feature Importance')
plt.tight_layout()
plt.savefig("XGB_Feature_Importance.png")
plt.show()

print("\n" + "="*50)
print("Task Complete!")
print("="*50)