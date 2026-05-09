# Wind-SARIMA

> Hourly wind speed forecasting using SARIMA — from raw 10-minute anemometer data to 7-day ensemble predictions.

## The Problem

Wind speed is notoriously hard to forecast. It has:
- A weak daily cycle (morning calm, afternoon gusts — only 0.5 m/s swing)
- A stronger annual cycle (windy winter, calm summer — 2.5 m/s swing)
- Massive hour-to-hour noise (σ ≈ 0.9 m/s between adjacent hours)
- Chaotic meteorological dynamics (no model can predict specific gusts weeks out)

A naive forecast (just the long-term mean) gives RMSE ≈ 2.86 m/s. This project asks: **how much better can we do?**

## Approach

We use a **two-stage SARIMA model** on 6 years of hourly wind data (2016–2021, ~48K hours):

```
Raw 10-min CSVs → Hourly resample → Stage 1: Fourier annual trend → Stage 2: SARIMA on residuals → Ensemble forecast
```

**Stage 1** captures the annual sine wave (winter windier, summer calmer) with 4 Fourier harmonics — explains 8.3% of the variance.

**Stage 2** applies SARIMA(1,0,2)(0,1,1,24) to the residuals — capturing the daily cycle and short-term persistence.

**Forecasts** use ensemble simulation (50 paths) driven by the model's own dynamics — producing realistic-looking wind trajectories with honest uncertainty bands.

## Results

| Metric | Value |
|---|---|
| Best model | SARIMA(1,0,2)(0,1,1,24) on Fourier-detrended residuals |
| Training data | 2016–2020 (43,308 hours) |
| Test data | 2021 Jan–Jun (4,344 hours, held out) |
| Blind multi-step RMSE | **2.78 m/s** |
| Naive (climatology) RMSE | 2.86 m/s |
| Improvement over naive | **+2.8%** |
| Rolling 1-step R² | ~0.85+ |
| Annual Fourier variance explained | 8.3% |

### What the Forecast Looks Like

The point forecast converges to the annual trend within a day. Beyond that, it's the Fourier climatology — because wind is genuinely unpredictable beyond short horizons. The ensemble simulation shows realistic uncertainty:

| Horizon | Annual Trend | Point Forecast | Lower 80% | Upper 80% |
|---|---|---|---|---|
| Hour 1 | 5.28 m/s | 3.16 m/s | 1.8 | 3.9 |
| Day 1 | 5.28 | 4.73 | 1.4 | 7.7 |
| Day 7 | 5.32 | 5.28 | 2.3 | 7.8 |

### Honest Limitations

- **Useful skill horizon:** 1–3 days for point forecasts, 1–7 days for ensemble
- **Long-term:** Converges to annual climatology (Fourier trend ± wide bands)
- **No annual year-to-year variability:** The Fourier trend is averaged across 5 years
- **Residuals not white noise:** Wind has inherent heteroskedasticity (variance changes over time)

This model doesn't pretend to predict random wind gusts months in advance — no statistical model can. What it provides is the best possible statistical estimate of future wind speed given historical patterns.

## Quick Start

```bash
git clone https://github.com/minhthetroller/Wind-SARIMA.git
cd Wind-SARIMA
pip install -r requirements.txt
jupyter notebook notebooks/01_wind_forecasting.ipynb
```

Run cells in order. First execution trains models (~5 min). Cache files (~24 KB) are auto-saved
to `dataset/models/`. Subsequent runs load from cache in ~1 minute.

## Project Structure

```
wind-SARIMA/
├── AGENTS.md              # Dev guide, architecture, modifications
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── .gitignore
├── dataset/
│   ├── wind-data*.csv     # Raw 10-min data (2016–2021)
│   └── models/            # Cached model artifacts (<1 KB total)
└── notebooks/
    └── 01_wind_forecasting.ipynb  # Full pipeline (EDA → model → forecast)
```

## Dependencies

| Package | Purpose |
|---|---|
| `statsmodels` | SARIMAX modeling, STL decomposition, ACF/PACF, diagnostics |
| `pandas`, `numpy` | Data manipulation |
| `matplotlib`, `seaborn` | Visualization |
| `scikit-learn` | Evaluation metrics (RMSE, MAE, R²) |
| `scipy` | Statistical utilities |
