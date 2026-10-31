"""EV Charging Demand Prediction & Usage Pattern Analysis - Streamlit dashboard.

Run:  streamlit run app.py

Expected files next to this script:
    regression_model.pkl       -> (lgb_reg, preprocessor)   from ML_Regression.ipynb
    classification_model.pkl   -> (xgb, preprocessor)       from ML_Classification.ipynb (see fix notes)
    data/EVChargingRecords_Features.csv                     (for dropdown options + forecasting)
"""
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="EV Charging Demand", page_icon="⚡", layout="wide")

REG_PATH = Path(r"D:\Aswini\EDA\EV_charging\evcharging\notebooks\regression_model.pkl")
CLF_PATH = Path(r"D:\Aswini\EDA\EV_charging\evcharging\notebooks\classification_model.pkl")
# Must be the FEATURES file (has time_of_day, day_of_week, ...), not the raw ChargingRecords.csv
DATA_PATH = Path(r"D:\Aswini\EDA\EV_charging\evcharging\Data\EVChargingRecords_Features.csv")

# ---------------------------------------------------------------------------
# CONFIG - adjust these to match how you built the features in the notebooks
# ---------------------------------------------------------------------------
# Unit stored in the 'Duration' column that the regression model was trained on.
DURATION_UNIT = "hours"          # "seconds" | "minutes" | "hours"
# time_of_day buckets (hour ranges). Labels are replaced by the ones in your CSV if it is loaded.
TIME_OF_DAY = [(5, 12, "Morning"), (12, 17, "Afternoon"), (17, 21, "Evening")]  # else "Night"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
@st.cache_resource
def load_pickle(path: str):
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_data(path: str):
    p = Path(path)
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
    df["StartDatetime"] = pd.to_datetime(df["StartDatetime"])
    # Rebuild derived columns if a non-features file is used
    if "time_of_day" not in df.columns:
        df["time_of_day"] = df["StartDatetime"].dt.hour.map(time_of_day)
    if "day_of_week" not in df.columns:
        df["day_of_week"] = df["StartDatetime"].dt.dayofweek
    return df


def time_of_day(hour: int) -> str:
    for lo, hi, label in TIME_OF_DAY:
        if lo <= hour < hi:
            return label
    return "Night"


# ---------------------------------------------------------------------------
# Feature building: one row, same columns the notebooks trained on
# ---------------------------------------------------------------------------
def build_features(inp: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (regression_row, classification_row) from the form inputs."""
    start = pd.Timestamp(inp["start"])
    end = start + pd.Timedelta(minutes=inp["duration_min"])
    secs = inp["duration_min"] * 60
    dur = {"seconds": secs, "minutes": secs / 60, "hours": secs / 3600}[DURATION_UNIT]
    tod = time_of_day(start.hour)

    # Columns used by ML_Regression.ipynb (categorical + numerical + ordinal).
    # StartDay/StartTime/EndDay/EndTime formats are guesses: compare with
    # df_features.iloc[0] using the "show model input" expander and fix here if needed.
    reg = pd.DataFrame([{
        "ChargerCompany": inp["company"], "Location": inp["location"], "ChargerType": inp["ctype"],
        "StartDay": start.day_name(), "StartTime": start.strftime("%H:%M:%S"),
        "EndDay": end.day_name(), "EndTime": end.strftime("%H:%M:%S"),
        "time_of_day": tod,
        "Duration": dur, "Duration_seconds": secs, "Duration_minutes": secs / 60,
        "Duration_hours": secs / 3600, "prev_sessions": inp["prev_sessions"],
        "start day": start.day, "start month": start.month, "end day": end.day,
        "end month": end.month, "year": start.year, "day_of_week": start.dayofweek,
        "is_weekend": int(start.dayofweek >= 5), "is_anomaly": 0,
    }])

    # Columns used by the (second) preprocessor in ML_Classification.ipynb.
    clf = pd.DataFrame([{
        "Location": inp["location"], "time_of_day": tod,
        "ChargerCompany": inp["company"], "ChargerType": inp["ctype"],
        "Duration_minutes": secs / 60, "hour": start.hour, "start_month": start.month,
        "start_day": start.day, "day_of_week": start.dayofweek,
        "is_weekend": int(start.dayofweek >= 5),
    }])
    return reg, clf


def top_contributions(model, X, feature_names, kind: str, n: int = 10) -> pd.DataFrame:
    """Exact TreeSHAP values for one row via the boosters' built-in pred_contrib."""
    if kind == "lgbm":
        contrib = model.predict(X, pred_contrib=True)[0]
    else:  # xgboost
        import xgboost as xgb
        contrib = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[0]
    contrib = np.asarray(contrib)[:-1]            # last value = bias term
    s = pd.Series(contrib, index=feature_names)
    s = s.reindex(s.abs().sort_values(ascending=False).index)[:n]
    return s.rename("SHAP value").to_frame()


def contrib_chart(df: pd.DataFrame, title: str):
    d = df.iloc[::-1]
    colors = ["#d1495b" if v > 0 else "#2e86ab" for v in d["SHAP value"]]
    fig = go.Figure(go.Bar(x=d["SHAP value"], y=d.index, orientation="h", marker_color=colors))
    fig.update_layout(title=title, height=360, margin=dict(l=10, r=10, t=40, b=10))
    return fig



# ---------------------------------------------------------------------------
# Time-series helpers (mirror ML_Time_Series.ipynb: chronological 80/20 split)
# ---------------------------------------------------------------------------
MA_NAME, ES_NAME, HW_NAME = "Moving Average (7)", "Exponential Smoothing", "Holt-Winters"
BEST_STYLE = "background-color: #d9f2e0; color: #0b3d1e; font-weight: 600"


def make_series(data: pd.DataFrame, which: str) -> pd.Series:
    """Daily demand: overall, or for one time_of_day bucket (zero-filled, like the notebook)."""
    if which == "Overall":
        s = data.set_index("StartDatetime")["Demand"].resample("D").sum()
    else:
        s = (data.assign(date=data["StartDatetime"].dt.normalize())
                 .groupby(["date", "time_of_day"])["Demand"].sum().unstack(fill_value=0)[which])
        s.index = pd.to_datetime(s.index)
    s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D"), fill_value=0)
    s.index.freq = "D"
    return s


def fit_es(series: pd.Series, fixed_alpha: bool):
    """Overall series: alpha fixed at 0.2 (notebook). Time-of-day series: alpha optimised."""
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing
    m = SimpleExpSmoothing(series, initialization_method="heuristic")
    return m.fit(smoothing_level=0.2, optimized=False) if fixed_alpha else m.fit(optimized=True)


@st.cache_data(show_spinner=False)
def backtest(series: pd.Series, fixed_alpha: bool) -> pd.DataFrame:
    """Hold-out (last 20%) RMSE/MAE for MA(7) walk-forward, SES and Holt-Winters."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    n = int(len(series) * 0.8)
    train, test = series.iloc[:n], series.iloc[n:]
    preds = {}

    hist, p = list(train), []
    for v in test:                       # walk-forward: add each actual after predicting it
        p.append(np.mean(hist[-7:]))
        hist.append(v)
    preds[MA_NAME] = p
    preds[ES_NAME] = fit_es(train, fixed_alpha).forecast(len(test))
    try:
        preds[HW_NAME] = ExponentialSmoothing(train, trend="add", seasonal="add",
                                              seasonal_periods=7).fit().forecast(len(test))
    except Exception:
        preds[HW_NAME] = None

    rows = {}
    for name, pr in preds.items():
        if pr is None:
            rows[name] = {"RMSE": np.nan, "MAE": np.nan}
        else:
            rows[name] = {"RMSE": float(np.sqrt(mean_squared_error(test, pr))),
                          "MAE": float(mean_absolute_error(test, pr))}
    return pd.DataFrame(rows).T


def highlight_best_rmse(table: pd.DataFrame):
    """Styler: highlight the row with the lowest RMSE."""
    best = table["RMSE"].idxmin()
    return table.style.apply(lambda r: [BEST_STYLE if r.name == best else "" for _ in r], axis=1) \
                      .format("{:,.2f}")

# ---------------------------------------------------------------------------
# Sidebar / data
# ---------------------------------------------------------------------------
df = load_data(str(DATA_PATH))
reg_bundle = load_pickle(str(REG_PATH))
clf_bundle = load_pickle(str(CLF_PATH))

st.title("⚡ EV Charging Demand Prediction & Usage Patterns")
tab_pred, tab_fc, tab_eda, tab_models = st.tabs(
    ["Predict a session", "Forecast demand", "Usage patterns", "Model performance"]
)

# ---------------------------------------------------------------------------
# Tab 1 - predict
# ---------------------------------------------------------------------------
with tab_pred:
    if reg_bundle is None and clf_bundle is None:
        st.error("No model files found. Check REG_PATH / CLF_PATH at the top of the file.")
    else:
        if df is not None:
            companies = sorted(df["ChargerCompany"].unique())
            locations = sorted(df["Location"].unique())
            ctypes = sorted(df["ChargerType"].unique())
            median_demand = float(df["Demand"].median())
        else:
            companies, locations, ctypes, median_demand = None, None, None, None

        with st.form("session"):
            c1, c2, c3 = st.columns(3)
            with c1:
                company = st.selectbox("Charger company", companies) if companies else st.text_input("Charger company")
                location = st.selectbox("Location", locations) if locations else st.text_input("Location")
                ctype = st.selectbox("Charger type", ctypes) if ctypes else st.text_input("Charger type")
            with c2:
                d = st.date_input("Start date", pd.Timestamp.today().date())
                t = st.time_input("Start time", pd.Timestamp("08:00").time())
                duration_min = st.number_input("Duration (minutes)", 1, 24 * 60, 60, step=5)
            with c3:
                prev = st.number_input("Previous sessions by this user", 0, 10_000, 0)
                explain = st.checkbox("Explain the prediction (SHAP)", value=True)
                debug = st.checkbox("Show model input rows")
            go_btn = st.form_submit_button("Predict", type="primary")

        if go_btn:
            inp = dict(company=company, location=location, ctype=ctype,
                       start=pd.Timestamp.combine(d, t), duration_min=duration_min, prev_sessions=prev)
            reg_row, clf_row = build_features(inp)
            left, right = st.columns(2)

            if reg_bundle is not None:
                lgb_reg, reg_pre = reg_bundle
                try:
                    Xr = reg_pre.transform(reg_row)
                    pred = float(lgb_reg.predict(Xr)[0])
                    left.metric("Predicted demand (kWh)", f"{pred:,.2f}",
                                None if median_demand is None else f"{pred - median_demand:+.2f} vs median")
                    if explain:
                        try:
                            contrib = top_contributions(lgb_reg, Xr, reg_pre.get_feature_names_out(), "lgbm")
                            left.plotly_chart(contrib_chart(contrib, "Top drivers - demand"), use_container_width=True)
                        except Exception as e:
                            left.caption(f"SHAP unavailable: {e}")
                except Exception as e:
                    left.error(f"Regression failed: {e}")

            if clf_bundle is not None:
                xgb_clf, clf_pre = clf_bundle
                try:
                    Xc = clf_pre.transform(clf_row)
                    p = float(xgb_clf.predict_proba(Xc)[0, 1])
                    right.metric("Demand class", "High demand" if p >= 0.5 else "Normal demand",
                                 f"P(high) = {p:.1%}")
                    right.progress(min(max(p, 0.0), 1.0))
                    if explain:
                        try:
                            contrib = top_contributions(xgb_clf, Xc, clf_pre.get_feature_names_out(), "xgb")
                            right.plotly_chart(contrib_chart(contrib, "Top drivers - High vs Normal"),
                                               use_container_width=True)
                        except Exception as e:
                            right.caption(f"SHAP unavailable: {e}")
                except Exception as e:
                    right.error(f"Classification failed: {e}")

            if debug:
                with st.expander("Model input rows", expanded=True):
                    st.write("Regression"); st.dataframe(reg_row.T)
                    st.write("Classification"); st.dataframe(clf_row.T)
                    if df is not None:
                        st.write("A real row from your data, for comparison")
                        st.dataframe(df.iloc[[0]].T)

# ---------------------------------------------------------------------------
# Tab 2 - forecast
# ---------------------------------------------------------------------------
with tab_fc:
    if df is None:
        st.warning(f"Features CSV not found at {DATA_PATH}")
    else:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        c1, c2 = st.columns(2)
        periods = ["Overall"] + sorted(df["time_of_day"].dropna().unique())
        which = c1.selectbox("Series", periods)
        horizon = c2.slider("Forecast horizon (days)", 7, 30, 7)

        series = make_series(df, which)
        fixed_alpha = which == "Overall"
        metrics = backtest(series, fixed_alpha)
        best_method = metrics["RMSE"].idxmin()
        methods = st.multiselect("Methods", [MA_NAME, ES_NAME, HW_NAME], default=[best_method],
                                 key=f"methods_{which}")

        future = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
        fig = go.Figure(go.Scatter(x=series.index, y=series.values, name="Actual", line=dict(color="#888")))
        if MA_NAME in methods:
            fig.add_scatter(x=future, y=[series.tail(7).mean()] * horizon, name=MA_NAME)
        if ES_NAME in methods:
            fig.add_scatter(x=future, y=fit_es(series, fixed_alpha).forecast(horizon).values, name=ES_NAME)
        if HW_NAME in methods:
            try:
                m = ExponentialSmoothing(series, trend="add", seasonal="add", seasonal_periods=7).fit()
                fig.add_scatter(x=future, y=m.forecast(horizon).values, name=HW_NAME)
            except Exception as e:
                st.caption(f"Holt-Winters failed: {e}")
        fig.update_layout(height=450, yaxis_title="Demand (kWh)", hovermode="x unified",
                          margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"**Hold-out accuracy - {which}** (chronological 80/20 split)")
        st.dataframe(highlight_best_rmse(metrics), use_container_width=True)
        st.caption(f"Lowest RMSE for this series: **{best_method}**.")

# ---------------------------------------------------------------------------
# Tab 3 - usage patterns
# ---------------------------------------------------------------------------
with tab_eda:
    if df is None:
        st.warning("Features CSV not found.")
    else:
        a, b = st.columns(2)
        by_hour = df.groupby(df["StartDatetime"].dt.hour)["Demand"].mean()
        a.plotly_chart(go.Figure(go.Bar(x=by_hour.index, y=by_hour.values)).update_layout(
            title="Average demand by start hour", xaxis_title="Hour", yaxis_title="kWh", height=350),
            use_container_width=True)
        dow = df.groupby("day_of_week")["Demand"].mean().reindex(range(7))
        b.plotly_chart(go.Figure(go.Bar(x=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], y=dow.values)).update_layout(
            title="Average demand by weekday", yaxis_title="kWh", height=350), use_container_width=True)
        loc = df.groupby("Location")["Demand"].agg(["mean", "count"]).sort_values("mean", ascending=False)
        st.subheader("Demand by location")
        st.dataframe(loc.rename(columns={"mean": "Avg demand (kWh)", "count": "Sessions"}))

# ---------------------------------------------------------------------------
# Tab 4 - model performance (numbers copied from your notebooks)
# ---------------------------------------------------------------------------
with tab_models:
    CHOSEN = {"reg": "LightGBM", "clf": "XGBoost", "ts": "Moving Average"}

    def show(df_, chosen, why):
        """Table with the chosen model's row highlighted, plus a note."""
        hl = lambda row: ["background-color: #d9f2e0; color: #0b3d1e; font-weight: 600" if row.name == chosen else "" for _ in row]
        st.dataframe(df_.style.apply(hl, axis=1), use_container_width=True)
        st.success(f"✅ Selected: **{chosen}** - {why}")

    st.subheader("Regression (target: Demand)")
    show(pd.DataFrame({
        "Model": ["Linear", "Ridge", "Lasso", "Random Forest", "XGBoost", "LightGBM"],
        "RMSE": [11.98, 9.23, 8.70, 6.02, 6.07, 5.87],
        "MAE": [8.32, 6.13, 5.53, 3.37, 3.45, 3.31],
        "R²": [0.2062, 0.5290, 0.5818, 0.7995, 0.7962, 0.8097],
    }).set_index("Model"), CHOSEN["reg"], "lowest RMSE/MAE and highest R² (0.81).")

    st.subheader("Classification (High vs Normal demand)")
    show(pd.DataFrame({
        "Model": ["Logistic Regression", "KNN", "Decision Tree", "Random Forest", "XGBoost", "LightGBM"],
        "Accuracy": [0.8203, 0.8018, 0.8902, 0.8933, 0.9007, 0.9005],
        "F1": [0.8219, 0.8059, 0.8917, 0.8976, 0.9038, 0.9036],
        "ROC-AUC": [0.8966, 0.8727, 0.8902, 0.9506, 0.9607, 0.9606],
    }).set_index("Model"), CHOSEN["clf"], "best accuracy, F1 and ROC-AUC (5-fold CV ROC-AUC 0.957).")

    st.subheader("Time-series - overall daily demand (chronological 80/20 hold-out)")
    show(pd.DataFrame({
        "Method": ["Moving Average", "Exponential Smoothing", "Holt-Winters"],
        "RMSE": [686.95, 789.10, 1023.45], "MAE": [502.51, 569.07, 786.30],
    }).set_index("Method"), CHOSEN["ts"],
        "lowest RMSE and MAE on the hold-out (walk-forward 7-day average); Holt-Winters is the weakest.")

    st.subheader("Time-series - by time of day (RMSE / MAE)")
    per = pd.DataFrame({
        "Period": ["Afternoon", "Evening", "Morning", "Night"],
        "MA RMSE": [243.94, 386.82, 224.05, 73.78], "MA MAE": [178.46, 299.61, 179.03, 59.62],
        "ES RMSE": [253.89, 447.45, 219.55, 73.44], "ES MAE": [188.58, 336.87, 168.97, 58.04],
        "HW RMSE": [326.24, 509.32, 247.02, 83.01], "HW MAE": [249.85, 394.75, 197.51, 66.76],
    }).set_index("Period")

    def best_cell(row):
        best = row[["MA RMSE", "ES RMSE", "HW RMSE"]].idxmin()
        return [BEST_STYLE if c == best else "" for c in row.index]

    st.dataframe(per.style.apply(best_cell, axis=1).format("{:,.2f}"), use_container_width=True)
    st.success("✅ Best by RMSE: **Moving Average** for Afternoon and Evening; "
               "**Exponential Smoothing** for Morning and Night (only marginally ahead of Moving Average). "
               "Holt-Winters is never best.")
