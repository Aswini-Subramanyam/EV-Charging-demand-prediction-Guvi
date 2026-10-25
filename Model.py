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
        from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing

        c1, c2, c3 = st.columns(3)
        periods = ["Overall"] + sorted(df["time_of_day"].dropna().unique())
        which = c1.selectbox("Series", periods)
        horizon = c2.slider("Forecast horizon (days)", 7, 30, 7)
        methods = c3.multiselect("Methods", ["Moving Average (7)", "Exponential Smoothing", "Holt-Winters"],
                                 default=["Moving Average (7)"])

        src = df if which == "Overall" else df[df["time_of_day"] == which]
        series = src.set_index("StartDatetime")["Demand"].resample("D").sum()
        series.index.freq = "D"
        future = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")

        fig = go.Figure(go.Scatter(x=series.index, y=series.values, name="Actual", line=dict(color="#444")))
        if "Moving Average (7)" in methods:
            fig.add_scatter(x=future, y=[series.tail(7).mean()] * horizon, name="Moving Average (7)")
        if "Exponential Smoothing" in methods:
            m = SimpleExpSmoothing(series, initialization_method="heuristic").fit(smoothing_level=0.2, optimized=False)
            fig.add_scatter(x=future, y=m.forecast(horizon).values, name="Exponential Smoothing")
        if "Holt-Winters" in methods:
            try:
                m = ExponentialSmoothing(series, trend="add", seasonal="add", seasonal_periods=7).fit()
                fig.add_scatter(x=future, y=m.forecast(horizon).values, name="Holt-Winters")
            except Exception as e:
                st.caption(f"Holt-Winters failed: {e}")
        fig.update_layout(height=450, yaxis_title="Demand (kWh)", hovermode="x unified",
                          margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Moving Average (7) is the recommended baseline: Holt-Winters edges it on a single "
                   "7-day hold-out but not consistently across periods.")

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

    st.subheader("Time-series (last 7 days hold-out)")
    show(pd.DataFrame({
        "Method": ["Moving Average", "Exponential Smoothing", "Holt-Winters"],
        "MAE": [327.76, 333.43, 305.99], "RMSE": [357.52, 359.72, 348.71],
    }).set_index("Method"), CHOSEN["ts"],
        "chosen as the baseline: walk-forward MAE/RMSE favoured it, and richer models did not beat it. "
        "Holt-Winters is lower on this single hold-out only.")