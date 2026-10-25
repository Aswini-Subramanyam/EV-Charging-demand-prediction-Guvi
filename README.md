# ⚡ EV Charging Demand Prediction & Usage Pattern Analysis

Machine-learning project on EV charging session records: predict how much energy a session will use, classify sessions as **High** or **Normal** demand, forecast daily demand, and explore everything in an interactive **Streamlit** dashboard.

<!-- Add a screenshot: ![Dashboard](docs/dashboard.png) -->

## Highlights

| Task | Target | Best model | Key result |
|---|---|---|---|
| Regression | `Demand` (kWh) | LightGBM | R² 0.81, RMSE 5.87, MAE 3.31 |
| Classification | `HighDemand` (median split) | XGBoost | Accuracy 0.90, F1 0.90, ROC-AUC 0.96 |
| Time-series forecast | Daily total demand | Moving Average (7) | Baseline chosen on walk-forward MAE/RMSE |

Per-prediction explanations (SHAP values) are shown in the app for both the regression and classification models.

## Project structure

```
.
├── Data/
│   ├── ChargingRecords.csv                # raw charging sessions
│   └── EVChargingRecords_Features.csv     # cleaned data + engineered features
├── notebooks/
│   ├── ML_Regression.ipynb
│   ├── ML_Classification.ipynb
│   ├── ML_Time_Series.ipynb
│   ├── regression_model.pkl               # (LightGBM regressor, preprocessor)
│   └── classification_model.pkl           # (XGBoost classifier, preprocessor)
├── Model.py                               # Streamlit app
├── requirements.txt
└── README.md
```

## Approach

**Data preparation**
- Session duration is calculated as end minus start time for every row, rather than using the provided `Duration` column, which can be miscalculated.
- Rows with negative duration (end before start) are handled before modelling. <!-- confirm: dropped? -->
- Engineered features include time of day, day of week, weekend flag, start/end month and day, year, an anomaly flag and the number of previous sessions per user.

**Regression** – Linear, Ridge, Lasso, Random Forest, XGBoost and LightGBM compared on a held-out 20% test set.

| Model | RMSE | MAE | R² |
|---|---:|---:|---:|
| Linear Regression | 11.98 | 8.32 | 0.2062 |
| Ridge | 9.23 | 6.13 | 0.5290 |
| Lasso | 8.70 | 5.53 | 0.5818 |
| Random Forest | 6.02 | 3.37 | 0.7995 |
| XGBoost | 6.07 | 3.45 | 0.7962 |
| **LightGBM** | **5.87** | **3.31** | **0.8097** |

**Classification** – sessions are labelled High/Normal demand by a median split of `Demand`. Training data is balanced with SMOTE; models are compared on a stratified hold-out set, and XGBoost / Random Forest are additionally checked with 5-fold cross-validation (XGBoost CV ROC-AUC ≈ 0.957).

| Model | Accuracy | F1 | ROC-AUC |
|---|---:|---:|---:|
| Logistic Regression | 0.8203 | 0.8219 | 0.8966 |
| KNN | 0.8018 | 0.8059 | 0.8727 |
| Decision Tree | 0.8902 | 0.8917 | 0.8902 |
| Random Forest | 0.8933 | 0.8976 | 0.9506 |
| **XGBoost** | **0.9007** | **0.9038** | **0.9607** |
| LightGBM | 0.9005 | 0.9036 | 0.9606 |

**Time series** – daily demand (overall and per time of day) is non-stationary (ADF test, p > 0.05) with a weekly cycle. Moving Average (7), Simple Exponential Smoothing and Holt-Winters were compared. Moving Average (7) is the recommended model: it was preferred on walk-forward MAE/RMSE, and a gradient-boosting model with extra features did not beat it. Holt-Winters scores slightly lower on a single 7-day hold-out, but not consistently across periods.

## The dashboard

Run from the project root:

```bash
streamlit run Model.py
```

| Tab | What it does |
|---|---|
| **Predict a session** | Enter charger, location, start time, duration → predicted demand (kWh), High/Normal probability, and top SHAP drivers |
| **Forecast demand** | Daily history plus Moving Average, Exponential Smoothing and Holt-Winters forecasts, overall or by time of day |
| **Usage patterns** | Average demand by hour, weekday and location |
| **Model performance** | Comparison tables with the selected model highlighted |

