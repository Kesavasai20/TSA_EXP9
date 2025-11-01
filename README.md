# EX.NO.09        A project on Time series analysis on weather forecasting using ARIMA model 
### Date:1/11/2025
## Register Number: 212223230105
## Nmae : K KESAVA SAI

### AIM:
To Create a project on Time series analysis on weather forecasting using ARIMA model in  Python and compare with other models.
### ALGORITHM:
1. Explore the dataset of weather 
2. Check for stationarity of time series time series plot
   ACF plot and PACF plot
   ADF test
   Transform to stationary: differencing
3. Determine ARIMA models parameters p, q
4. Fit the ARIMA model
5. Make time series predictions
6. Auto-fit the ARIMA model
7. Evaluate model predictions
### PROGRAM:
```py
# === Import Libraries ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore")

# === 1. Load Dataset ===
data = pd.read_csv('/content/Clean_Dataset.csv')

# === 2. Prepare Dataset ===
print("Columns:", data.columns.tolist())

# Ensure required columns exist
if 'days_left' not in data.columns or 'price' not in data.columns:
    raise ValueError("Dataset must contain 'days_left' and 'price' columns.")

# Group by days_left and calculate average price
avg_price_by_day = data.groupby('days_left')['price'].mean().sort_index()

# Create DataFrame for time series modeling
ts_data = pd.DataFrame({'price': avg_price_by_day})
ts_data.index.name = 'days_left'

print("\nFirst 10 rows of time series data:")
print(ts_data.head(10))

# === 3. Split Train/Test Data ===
train_size = int(len(ts_data) * 0.8)
train_data, test_data = ts_data.iloc[:train_size], ts_data.iloc[train_size:]

# === 4. ARIMA Model ===
# You can adjust the (p,d,q) order later for best performance
order = (2, 1, 2)
model = ARIMA(train_data['price'], order=order)
fitted_model = model.fit()

# === 5. Forecast ===
forecast = fitted_model.forecast(steps=len(test_data))

# === 6. Evaluate ===
rmse = np.sqrt(mean_squared_error(test_data['price'], forecast))
print(f"\nRoot Mean Squared Error (RMSE): {rmse:.4f}")

# === 7. Plot Results ===
plt.figure(figsize=(12, 6))
plt.plot(train_data.index, train_data['price'], label='Training Data', color='blue')
plt.plot(test_data.index, test_data['price'], label='Testing Data', color='green')
plt.plot(test_data.index, forecast, label='Forecast', color='red', linestyle='--')
plt.title('ARIMA Forecasting of Flight Prices vs Days Left')
plt.xlabel('Days Left')
plt.ylabel('Average Price')
plt.legend()
plt.grid()
plt.show()

# === 8. Forecast Future Prices ===
future_steps = 15  # forecast next 15 days_left values
future_forecast = fitted_model.forecast(steps=future_steps)

plt.figure(figsize=(12, 6))
plt.plot(ts_data.index, ts_data['price'], label='Historical Data', color='blue')
plt.plot(
    np.arange(ts_data.index[-1] + 1, ts_data.index[-1] + future_steps + 1),
    future_forecast,
    label='Future Forecast',
    color='red',
    linestyle='--'
)
plt.title('Future Price Forecast (Next 15 Days)')
plt.xlabel('Days Left (pseudo-time)')
plt.ylabel('Average Price')
plt.legend()
plt.grid()
plt.show()


```

### OUTPUT:
<img width="1020" height="285" alt="image" src="https://github.com/user-attachments/assets/a3dcfc92-5bcf-4276-a92c-73043214184b" />
<img width="1129" height="558" alt="image" src="https://github.com/user-attachments/assets/4639ce07-9a83-43bd-9f27-8cdecb76733c" />
<img width="1220" height="556" alt="image" src="https://github.com/user-attachments/assets/2eef0b34-cd8f-49df-bda6-0be1341024c0" />


### RESULT:
Thus the program run successfully based on the ARIMA model using python.
