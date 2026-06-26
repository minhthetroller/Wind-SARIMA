import numpy as np, pandas as pd, glob, os
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

base = '../dataset/'
files = sorted(glob.glob(base + 'wind-data*.csv'))
all_hourly = []
for f in files:
    df = pd.read_csv(f)
    ts_col = [c for c in df.columns if 'date' in c.lower() and 'time' in c.lower()][0]
    ws_col = [c for c in df.columns if 'wind' in c.lower() and 'speed' in c.lower()][0]
    df = df.rename(columns={ts_col: 'timestamp', ws_col: 'wind_speed'})
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%m/%d/%Y %H:%M')
    df = df.set_index('timestamp').sort_index()
    hourly = df['wind_speed'].resample('h').mean()
    if hourly.isna().sum() > 0: hourly = hourly.interpolate(method='linear')
    all_hourly.append(hourly)
series = pd.concat(all_hourly)
series = series[~series.index.duplicated(keep='first')].sort_index()
train_series = series[series.index < '2021-01-01']
test_series = series[series.index >= '2021-01-01']

data = np.load('../dataset/models/best_order.npz', allow_pickle=True)
order = tuple(data['order'])
seasonal_order = tuple(data['seasonal'])
print(f'Best order: {order} x {seasonal_order}')

saved_params = np.load('../dataset/models/sarima_params_train.npy')
print(f'Loaded cached train params ({len(saved_params)} coeffs)')

model = SARIMAX(train_series, order=order, seasonal_order=seasonal_order,
                enforce_stationarity=False, enforce_invertibility=False)
fitted = model.fit(start_params=saved_params, maxiter=0, disp=False, low_memory=True, cov_type='approx')
print('Training model loaded.')

print('Evaluating on test set...')
forecast_result = fitted.get_forecast(steps=len(test_series))
predicted_mean = forecast_result.predicted_mean.values

rmse_val = np.sqrt(mean_squared_error(test_series.values, predicted_mean))
mae_val = mean_absolute_error(test_series.values, predicted_mean)
r2_val = r2_score(test_series.values, predicted_mean)

print(f'RMSE: {rmse_val:.4f} m/s, MAE: {mae_val:.4f} m/s, R^2: {r2_val:.4f}')
naive_pred = np.full(len(test_series), train_series.mean())
naive_rmse = np.sqrt(mean_squared_error(test_series.values, naive_pred))
print(f'Naive RMSE: {naive_rmse:.4f} m/s, Improvement: {(1 - rmse_val/naive_rmse)*100:.1f}%')

print('Fitting on full data...')
full_model = SARIMAX(series, order=order, seasonal_order=seasonal_order,
                     enforce_stationarity=False, enforce_invertibility=False)
full_fitted = full_model.fit(disp=False, maxiter=200, cov_type='approx', low_memory=True)
np.save('../dataset/models/sarima_params_full.npy', full_fitted.params.values)

n_future = 168
n_paths = 50
future_dates = pd.date_range(start=series.index[-1] + pd.Timedelta(hours=1), periods=n_future, freq='h')
future_forecast = full_fitted.get_forecast(steps=n_future)
point_forecast = future_forecast.predicted_mean.values

print(f'Generating {n_paths} simulated paths ({len(future_dates)} hours)...')
sim_paths = full_fitted.simulate(nsimulations=n_future, repetitions=n_paths, anchor='end')
sim_median = np.median(sim_paths.values, axis=1)
sim_lower_95 = np.percentile(sim_paths.values, 2.5, axis=1)
sim_upper_95 = np.percentile(sim_paths.values, 97.5, axis=1)
sim_lower_80 = np.percentile(sim_paths.values, 10, axis=1)
sim_upper_80 = np.percentile(sim_paths.values, 90, axis=1)

print(f'Point forecast range: {point_forecast.min():.2f} to {point_forecast.max():.2f} m/s')
print(f'Ensemble median range: {sim_median.min():.2f} to {sim_median.max():.2f} m/s')
print(f'80% band avg width: {(sim_upper_80 - sim_lower_80).mean():.2f} m/s')

print()
print('Horizon    Point   Median  Lower80  Upper80')
print('-' * 45)
horizons = [0, 6, 12, 24, 72, 120, 167]
labels = ['Hour 1', 'Hour 6', 'Hour 12', 'Day 1', 'Day 3', 'Day 5', 'Day 7']
for h, label in zip(horizons, labels):
    print(f'{label:<10} {point_forecast[h]:>6.2f}  {sim_median[h]:>6.2f}  {sim_lower_80[h]:>6.2f}  {sim_upper_80[h]:>6.2f}')
