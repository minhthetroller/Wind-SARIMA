# AGENTS.md — Wind-SARIMA Project

## Architecture Overview

**What this project does:** Hourly wind speed forecasting using SARIMA time-series models.
Trained on 6 years of 10-minute anemometer data (2016–2021), resampled to hourly.

**Model pipeline:**

```
10-min raw CSV → Hourly resample → Fourier annual detrend → SARIMA on residuals → Ensemble forecast
```

### Data Flow

```
dataset/
├── wind-data.csv          # 2016 (partial, Jan 23–Dec 31)
├── wind-data-2017.csv     # 2017 full year
├── wind-data-2018.csv     # 2018 full year
├── wind-data-2019.csv     # 2019 full year
├── wind-data-2020.csv     # 2020 full year
├── wind-data-2021.csv     # 2021 (partial, Jan–Jun)
└── models/                # Cached model artifacts (auto-generated, .gitignored)
    ├── fourier_coeffs_*.npy
    ├── sarima_params_*.npy
    └── best_order.npz
```

**Column name handling:** 3 different formats across years are auto-detected by checking for `'date'` and `'time'` in column names.

### Model Architecture

**Two-stage approach:**

1. **Stage 1 — Fourier annual trend** (`K=4` harmonics, period=8760 hours)
   - Captures deterministic yearly cycle (Jan~7 m/s, Jun~5 m/s)
   - Explains 8.3% of wind speed variance
   - Coefficients saved to `dataset/models/fourier_coeffs_*.npy`

2. **Stage 2 — SARIMA on residuals** (`(1,0,2)(0,1,1,24)`)
   - Models deviations from the annual trend
   - Daily cycle via seasonal differencing at m=24
   - Parameters saved to `dataset/models/sarima_params_*.npy`

**Train/test split:** 2016–2020 (43,308 hours) for training, 2021 (4,344 hours) held out.

## Key Metrics

| Metric | Value |
|---|---|
| Best model | SARIMA(1,0,2)(0,1,1,24) on Fourier-detrended residuals |
| Training data | 47,652 hours (6 years) |
| Blind multi-step RMSE (2021) | 2.78 m/s |
| Improvement over naive climatology | +2.8% |
| Annual Fourier variance explained | 8.3% |
| Model cache size | ~24 KB total |

## Development Guide

### Setup

```bash
pip install -r requirements.txt
```

### Running the Notebook

Open `notebooks/01_wind_forecasting.ipynb` in Jupyter/VSCode.

**Run cells in order.** Section 3 trains models (~3-5 min first time). Section 5 produces
the 7-day ensemble forecast. Cache files are auto-created in `dataset/models/` and
auto-loaded on subsequent runs.

### Caching Strategy

All trained artifacts (Fourier coefficients, grid search results, SARIMA params) are
saved as `.npy`/`.npz` files under `dataset/models/` (total < 1 KB).

On reload:
- Grid search result → loaded from `best_order.npz`
- SARIMA params → loaded via `fit(start_params=saved, maxiter=0)` (instant)
- Fourier coeffs → loaded from `.npy`

**To force retraining:** Delete files in `dataset/models/`.

### Adding New Yearly Data

1. Place new CSV in `dataset/` (naming convention: `wind-data-<YEAR>.csv`)
2. The loader auto-discovers via `glob.glob('wind-data*.csv')`
3. Column name variants are auto-detected (see "Column name handling" above)
4. Update the train/test split date in Section 3 if needed
5. Delete `dataset/models/*` to force retraining on expanded dataset

### Modifying the Model

**Changing SARIMA order:**
Edit the grid search bounds in Section 3, Stage 2 cell:
```python
p_range = [1, 2]      # non-seasonal AR
q_range = [1, 2]      # non-seasonal MA
P_range = [0, 1]      # seasonal AR
Q_range = [0, 1]      # seasonal MA
d, D, m = 0, 1, 24    # non-seasonal diff, seasonal diff, seasonal period
```
Then delete `dataset/models/best_order.npz` and `dataset/models/sarima_params_*.npy` to force re-search.

**Changing Fourier harmonics:**
Change `K` in Section 3, Stage 1 cell (default: 4).

**Changing forecast horizon:**
Change `n_future` in Section 5 (default: 168 for 7 days).

### Known Limitations

1. **Only 1 year (2021) held out for testing** — more test years would improve evaluation robustness.
2. **No exogenous meteorological variables** — temperature, pressure, or NWP model output would improve forecasts.
3. **Annual seasonality is purely Fourier** — doesn't account for year-to-year climate variability.
4. **Ljung-Box test fails** on SARIMA residuals for some lags — expected for wind data with heteroskedasticity.
5. **Blind multi-step forecasts degrade rapidly** — useful skill horizon is ~1–3 days for point forecasts,
   1–7 days for ensemble simulation. Beyond that, the forecast converges to annual climatology.
6. **CI bands can dip slightly negative** at extreme quantiles (95%) for long horizons — a known artifact
   of the Gaussian assumption on non-negative bounded data.

### Future Improvements

- **Exogenous regressors:** Add month dummies or sine/cosine terms directly in SARIMAX `exog`
  (may need to difference exog to match model differencing).
- **GARCH for volatility:** Wind speed variance changes over time (heteroskedasticity).
- **Multi-year evaluation:** With more future data, use rolling-window cross-validation across years.
- **Alternative models:** Prophet, TBATS, or LSTM for comparison benchmarks.
- **Probabilistic metrics:** CRPS (Continuous Ranked Probability Score) instead of just RMSE/R²
  for ensemble evaluation.
