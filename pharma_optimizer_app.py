import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pharma AI Optimizer",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Styling ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .metric-card {
        background: white;
        border-radius: 8px;
        padding: 16px 20px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        text-align: center;
    }
    .metric-label { font-size: 13px; color: #666; margin-bottom: 4px; }
    .metric-value { font-size: 28px; font-weight: 700; color: #1F4E79; }
    .metric-delta { font-size: 12px; margin-top: 2px; }
    .section-header {
        font-size: 16px; font-weight: 600;
        color: #1F4E79; border-bottom: 2px solid #1F4E79;
        padding-bottom: 6px; margin-bottom: 16px;
    }
    .insight-box {
        background: #EBF3FB; border-left: 4px solid #1F4E79;
        border-radius: 4px; padding: 12px 16px;
        font-size: 14px; color: #1a1a1a; margin-bottom: 8px;
    }
    .risk-high { color: #C0392B; font-weight: 600; }
    .risk-med  { color: #E67E22; font-weight: 600; }
    .risk-low  { color: #27AE60; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Synthetic data generation ──────────────────────────────────────────────────
@st.cache_data
def generate_data(n=2000, seed=42):
    rng = np.random.default_rng(seed)

    maintenance_freq   = rng.integers(1, 31, n)        # days between maintenance
    qa_staff           = rng.integers(2, 12, n)         # QA headcount
    sensor_interval    = rng.integers(1, 15, n)         # hours between calibration
    changeover_time    = rng.integers(30, 300, n)       # minutes
    line_speed         = rng.uniform(50, 150, n)        # units/min
    ambient_temp_var   = rng.uniform(0.1, 5.0, n)      # °C variance
    humidity_var       = rng.uniform(0.5, 10.0, n)     # % RH variance
    batch_size         = rng.integers(5000, 50000, n)

    # Deviation risk score (0-1) — higher maintenance gap, lower QA, poor calibration = higher risk
    dev_risk_raw = (
        0.30 * (maintenance_freq / 30) +
        0.25 * (1 - qa_staff / 12) +
        0.20 * (sensor_interval / 15) +
        0.10 * (ambient_temp_var / 5) +
        0.10 * (humidity_var / 10) +
        0.05 * rng.uniform(0, 1, n)
    )
    deviation_event = (dev_risk_raw > 0.50).astype(int)

    # Yield (%) — penalized by changeover, maintenance gap, deviations
    yield_pct = (
        97.5
        - 0.08  * (maintenance_freq - 1)
        - 0.04  * (changeover_time - 30) / 10
        - 0.15  * deviation_event * rng.uniform(1, 3, n)
        - 0.02  * (sensor_interval - 1)
        + 0.03  * (qa_staff - 2)
        + rng.normal(0, 0.5, n)
    ).clip(85, 99.5)

    # Throughput (units/hour)
    throughput = (
        line_speed * 60
        - 250 * (changeover_time / 60)
        - 120 * deviation_event
        - 80  * (maintenance_freq > 20).astype(int)
        + rng.normal(0, 100, n)
    ).clip(1500, 9000)

    # Cost per batch ($)
    labor_base   = 4500
    qa_cost      = qa_staff * 350
    maint_cost   = np.where(maintenance_freq <= 7, 800, np.where(maintenance_freq <= 14, 400, 150))
    downtime_cost= deviation_event * rng.uniform(2000, 8000, n)
    scrap_cost   = (100 - yield_pct) / 100 * batch_size * 0.85
    cost_per_batch = labor_base + qa_cost + maint_cost + downtime_cost + scrap_cost + rng.normal(0, 200, n)

    df = pd.DataFrame({
        "maintenance_freq": maintenance_freq,
        "qa_staff": qa_staff,
        "sensor_interval": sensor_interval,
        "changeover_time": changeover_time,
        "line_speed": line_speed,
        "ambient_temp_var": ambient_temp_var,
        "humidity_var": humidity_var,
        "batch_size": batch_size,
        "deviation_event": deviation_event,
        "yield_pct": yield_pct,
        "throughput": throughput,
        "cost_per_batch": cost_per_batch,
    })
    return df

# ── Model training ─────────────────────────────────────────────────────────────
@st.cache_resource
def train_models(df):
    features = ["maintenance_freq", "qa_staff", "sensor_interval",
                "changeover_time", "line_speed", "ambient_temp_var", "humidity_var"]
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Deviation classifier
    clf = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42)
    clf.fit(X_scaled, df["deviation_event"])

    # Yield regressor
    reg_yield = RandomForestRegressor(n_estimators=150, max_depth=8, random_state=42)
    reg_yield.fit(X_scaled, df["yield_pct"])

    # Throughput regressor
    reg_tput = RandomForestRegressor(n_estimators=150, max_depth=8, random_state=42)
    reg_tput.fit(X_scaled, df["throughput"])

    # Cost regressor
    reg_cost = RandomForestRegressor(n_estimators=150, max_depth=8, random_state=42)
    reg_cost.fit(X_scaled, df["cost_per_batch"])

    importances = pd.Series(clf.feature_importances_, index=features).sort_values(ascending=True)

    return clf, reg_yield, reg_tput, reg_cost, scaler, features, importances

def predict_scenario(params, clf, reg_yield, reg_tput, reg_cost, scaler, features):
    X = np.array([[params[f] for f in features]])
    X_scaled = scaler.transform(X)
    dev_prob   = clf.predict_proba(X_scaled)[0][1]
    yield_pred = reg_yield.predict(X_scaled)[0]
    tput_pred  = reg_tput.predict(X_scaled)[0]
    cost_pred  = reg_cost.predict(X_scaled)[0]
    return dev_prob, yield_pred, tput_pred, cost_pred

# ── Load data & models ─────────────────────────────────────────────────────────
df = generate_data()
clf, reg_yield, reg_tput, reg_cost, scaler, features, importances = train_models(df)

# Baseline (median values)
BASELINE = {
    "maintenance_freq": 15,
    "qa_staff": 6,
    "sensor_interval": 8,
    "changeover_time": 165,
    "line_speed": 100.0,
    "ambient_temp_var": 2.5,
    "humidity_var": 5.0,
}
b_dev, b_yield, b_tput, b_cost = predict_scenario(BASELINE, clf, reg_yield, reg_tput, reg_cost, scaler, features)

# ── Sidebar — scenario controls ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Scenario Controls")
    st.markdown("Adjust operational parameters to model outcomes for **Drug XYZ** manufacturing line.")
    st.markdown("---")

    st.markdown("**🔧 Maintenance**")
    maintenance_freq = st.slider("Maintenance Frequency (days)", 1, 30, 15,
        help="Days between scheduled preventive maintenance events")

    st.markdown("**👥 QA Staffing**")
    qa_staff = st.slider("QA Staff on Line", 2, 11, 6,
        help="Number of QA personnel assigned to this manufacturing line")

    st.markdown("**📡 Sensor Calibration**")
    sensor_interval = st.slider("Sensor Calibration Interval (hours)", 1, 14, 8,
        help="Hours between sensor recalibration cycles")

    st.markdown("**⏱️ Changeover**")
    changeover_time = st.slider("Changeover Time (minutes)", 30, 299, 165,
        help="Time required for line changeover between batches")

    st.markdown("**🌡️ Environment**")
    ambient_temp_var = st.slider("Ambient Temp Variance (°C)", 0.1, 5.0, 2.5, step=0.1)
    humidity_var     = st.slider("Humidity Variance (% RH)", 0.5, 10.0, 5.0, step=0.5)

    line_speed = 100.0  # fixed for simplicity

    st.markdown("---")
    if st.button("🔄 Reset to Baseline", use_container_width=True):
        st.rerun()

# ── Current scenario prediction ────────────────────────────────────────────────
params = {
    "maintenance_freq": maintenance_freq,
    "qa_staff": qa_staff,
    "sensor_interval": sensor_interval,
    "changeover_time": changeover_time,
    "line_speed": line_speed,
    "ambient_temp_var": ambient_temp_var,
    "humidity_var": humidity_var,
}
s_dev, s_yield, s_tput, s_cost = predict_scenario(params, clf, reg_yield, reg_tput, reg_cost, scaler, features)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("# ⚗️ Pharma AI Optimizer")
st.markdown("**Drug XYZ Manufacturing Line  |  Scenario-Based Operational & Investment Analysis**")
st.markdown("---")

# ── KPI Cards ──────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

def delta_color(val, baseline, lower_is_better=False):
    diff = val - baseline
    if lower_is_better:
        return "🟢" if diff < 0 else ("🔴" if diff > 0 else "⚪")
    return "🟢" if diff > 0 else ("🔴" if diff < 0 else "⚪")

with col1:
    risk_label = "HIGH" if s_dev > 0.6 else ("MEDIUM" if s_dev > 0.35 else "LOW")
    risk_cls   = "risk-high" if s_dev > 0.6 else ("risk-med" if s_dev > 0.35 else "risk-low")
    delta_dev  = (s_dev - b_dev) * 100
    icon = "🔴" if delta_dev > 0 else ("🟢" if delta_dev < 0 else "⚪")
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Deviation Risk</div>
        <div class="metric-value"><span class="{risk_cls}">{s_dev*100:.1f}%</span></div>
        <div class="metric-delta">{icon} {delta_dev:+.1f}pp vs baseline</div>
    </div>""", unsafe_allow_html=True)

with col2:
    delta_yield = s_yield - b_yield
    icon = delta_color(s_yield, b_yield)
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Predicted Yield</div>
        <div class="metric-value">{s_yield:.1f}%</div>
        <div class="metric-delta">{icon} {delta_yield:+.2f}pp vs baseline</div>
    </div>""", unsafe_allow_html=True)

with col3:
    delta_tput = s_tput - b_tput
    icon = delta_color(s_tput, b_tput)
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Throughput (units/hr)</div>
        <div class="metric-value">{s_tput:,.0f}</div>
        <div class="metric-delta">{icon} {delta_tput:+,.0f} vs baseline</div>
    </div>""", unsafe_allow_html=True)

with col4:
    delta_cost = s_cost - b_cost
    icon = delta_color(s_cost, b_cost, lower_is_better=True)
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Cost per Batch ($)</div>
        <div class="metric-value">${s_cost:,.0f}</div>
        <div class="metric-delta">{icon} ${delta_cost:+,.0f} vs baseline</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Scenario Analysis",
    "🔍 Deviation Risk Drivers",
    "💰 Cost & Capacity Trade-offs",
    "📈 Historical Data Explorer"
])

# ── Tab 1: Scenario Analysis ───────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-header">Scenario vs. Baseline Comparison</div>', unsafe_allow_html=True)

    strategies = {
        "Current Scenario":      params,
        "Baseline (Median Ops)": BASELINE,
        "Aggressive Maintenance": {**BASELINE, "maintenance_freq": 5,  "sensor_interval": 3},
        "Max QA Staffing":        {**BASELINE, "qa_staff": 11},
        "Fast Changeover":        {**BASELINE, "changeover_time": 60},
        "Optimized (All Levers)": {**BASELINE, "maintenance_freq": 5, "qa_staff": 10,
                                    "sensor_interval": 3, "changeover_time": 60,
                                    "ambient_temp_var": 0.5, "humidity_var": 1.0},
    }

    rows = []
    for name, p in strategies.items():
        d, y, t, c = predict_scenario(p, clf, reg_yield, reg_tput, reg_cost, scaler, features)
        rows.append({"Strategy": name,
                     "Deviation Risk (%)": round(d*100, 1),
                     "Yield (%)": round(y, 2),
                     "Throughput (u/hr)": round(t),
                     "Cost/Batch ($)": round(c)})

    results_df = pd.DataFrame(rows)

    col_a, col_b = st.columns(2)
    with col_a:
        fig = px.bar(results_df, x="Strategy", y="Deviation Risk (%)",
                     color="Deviation Risk (%)",
                     color_continuous_scale=["#27AE60", "#F39C12", "#C0392B"],
                     title="Deviation Risk by Strategy")
        fig.update_layout(showlegend=False, height=350,
                          plot_bgcolor="white", paper_bgcolor="white",
                          xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        fig2 = px.bar(results_df, x="Strategy", y="Cost/Batch ($)",
                      color="Cost/Batch ($)",
                      color_continuous_scale=["#27AE60", "#F39C12", "#C0392B"],
                      title="Cost per Batch by Strategy")
        fig2.update_layout(showlegend=False, height=350,
                           plot_bgcolor="white", paper_bgcolor="white",
                           xaxis_tickangle=-30)
        st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(results_df.style.format({
        "Deviation Risk (%)": "{:.1f}%",
        "Yield (%)": "{:.2f}%",
        "Throughput (u/hr)": "{:,.0f}",
        "Cost/Batch ($)": "${:,.0f}"
    }), use_container_width=True, hide_index=True)

    # Insights
    st.markdown('<div class="section-header">AI-Generated Insights</div>', unsafe_allow_html=True)
    best_cost = results_df.loc[results_df["Cost/Batch ($)"].idxmin(), "Strategy"]
    best_risk = results_df.loc[results_df["Deviation Risk (%)"].idxmin(), "Strategy"]
    best_yield = results_df.loc[results_df["Yield (%)"].idxmax(), "Strategy"]

    st.markdown(f'<div class="insight-box">💡 <b>Lowest cost per batch</b> is achieved under the <b>{best_cost}</b> strategy at <b>${results_df.loc[results_df["Strategy"]==best_cost, "Cost/Batch ($)"].values[0]:,.0f}</b>.</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="insight-box">🛡️ <b>Lowest deviation risk</b> is achieved under the <b>{best_risk}</b> strategy at <b>{results_df.loc[results_df["Strategy"]==best_risk, "Deviation Risk (%)"].values[0]:.1f}%</b>.</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="insight-box">📈 <b>Highest yield</b> is achieved under the <b>{best_yield}</b> strategy at <b>{results_df.loc[results_df["Strategy"]==best_yield, "Yield (%)"].values[0]:.2f}%</b>.</div>', unsafe_allow_html=True)

# ── Tab 2: Deviation Risk Drivers ─────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-header">Feature Importance — Deviation Risk Model</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns([1, 1])
    with col_a:
        fig = px.bar(
            x=importances.values, y=importances.index,
            orientation='h',
            labels={"x": "Importance Score", "y": "Feature"},
            color=importances.values,
            color_continuous_scale=["#BDD7EE", "#1F4E79"],
            title="Top Drivers of Deviation Events"
        )
        fig.update_layout(showlegend=False, height=380,
                          plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-header">Risk Probability by Maintenance Frequency</div>', unsafe_allow_html=True)
        maint_range = range(1, 31)
        risk_by_maint = []
        for m in maint_range:
            p = {**params, "maintenance_freq": m}
            d, _, _, _ = predict_scenario(p, clf, reg_yield, reg_tput, reg_cost, scaler, features)
            risk_by_maint.append({"Maintenance Frequency (days)": m, "Deviation Risk (%)": d * 100})
        maint_df = pd.DataFrame(risk_by_maint)
        fig2 = px.line(maint_df, x="Maintenance Frequency (days)", y="Deviation Risk (%)",
                       title="How Maintenance Frequency Drives Risk",
                       color_discrete_sequence=["#1F4E79"])
        fig2.add_hline(y=50, line_dash="dash", line_color="#C0392B",
                       annotation_text="High Risk Threshold")
        fig2.update_layout(height=350, plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-header">Deviation Risk vs. QA Staffing</div>', unsafe_allow_html=True)
    qa_range = range(2, 12)
    risk_by_qa = []
    for q in qa_range:
        p = {**params, "qa_staff": q}
        d, _, _, _ = predict_scenario(p, clf, reg_yield, reg_tput, reg_cost, scaler, features)
        risk_by_qa.append({"QA Staff": q, "Deviation Risk (%)": d * 100})
    qa_df = pd.DataFrame(risk_by_qa)
    fig3 = px.bar(qa_df, x="QA Staff", y="Deviation Risk (%)",
                  color="Deviation Risk (%)",
                  color_continuous_scale=["#27AE60", "#F39C12", "#C0392B"],
                  title="Deviation Risk by QA Staffing Level")
    fig3.update_layout(showlegend=False, height=320,
                       plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig3, use_container_width=True)

# ── Tab 3: Cost & Capacity Trade-offs ─────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">Cost vs. Compliance Risk Trade-off</div>', unsafe_allow_html=True)

    sweep = []
    for mf in [5, 10, 15, 20, 25, 30]:
        for qa in [3, 5, 7, 9, 11]:
            p = {**params, "maintenance_freq": mf, "qa_staff": qa}
            d, y, t, c = predict_scenario(p, clf, reg_yield, reg_tput, reg_cost, scaler, features)
            sweep.append({
                "Maintenance Freq": mf,
                "QA Staff": qa,
                "Deviation Risk (%)": round(d * 100, 1),
                "Yield (%)": round(y, 2),
                "Throughput (u/hr)": round(t),
                "Cost/Batch ($)": round(c),
            })
    sweep_df = pd.DataFrame(sweep)

    fig = px.scatter(sweep_df,
                     x="Cost/Batch ($)", y="Deviation Risk (%)",
                     size="Throughput (u/hr)", color="Yield (%)",
                     hover_data=["Maintenance Freq", "QA Staff"],
                     color_continuous_scale="RdYlGn",
                     title="Cost vs. Risk vs. Throughput (bubble size = throughput, color = yield)",
                     labels={"Cost/Batch ($)": "Cost per Batch ($)",
                             "Deviation Risk (%)": "Deviation Risk (%)"})
    fig.update_layout(height=450, plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Changeover Time Impact on Throughput & Cost</div>', unsafe_allow_html=True)
    co_range = range(30, 300, 15)
    co_data = []
    for co in co_range:
        p = {**params, "changeover_time": co}
        d, y, t, c = predict_scenario(p, clf, reg_yield, reg_tput, reg_cost, scaler, features)
        co_data.append({"Changeover Time (min)": co, "Throughput (u/hr)": t, "Cost/Batch ($)": c})
    co_df = pd.DataFrame(co_data)

    col_a, col_b = st.columns(2)
    with col_a:
        fig2 = px.line(co_df, x="Changeover Time (min)", y="Throughput (u/hr)",
                       title="Throughput vs Changeover Time",
                       color_discrete_sequence=["#1F4E79"])
        fig2.update_layout(height=300, plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig2, use_container_width=True)
    with col_b:
        fig3 = px.line(co_df, x="Changeover Time (min)", y="Cost/Batch ($)",
                       title="Cost vs Changeover Time",
                       color_discrete_sequence=["#C0392B"])
        fig3.update_layout(height=300, plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig3, use_container_width=True)

    # Annual savings calculator
    st.markdown('<div class="section-header">💰 Annual Savings Calculator</div>', unsafe_allow_html=True)
    col_x, col_y = st.columns(2)
    with col_x:
        batches_per_year = st.number_input("Batches per year", min_value=10, max_value=5000, value=500, step=10)
    with col_y:
        opt_p = {**BASELINE, "maintenance_freq": 5, "qa_staff": 10,
                 "sensor_interval": 3, "changeover_time": 60,
                 "ambient_temp_var": 0.5, "humidity_var": 1.0}
        _, _, _, opt_cost = predict_scenario(opt_p, clf, reg_yield, reg_tput, reg_cost, scaler, features)
        annual_saving = (b_cost - opt_cost) * batches_per_year
        st.markdown(f"""
        <div class="metric-card" style="margin-top:28px">
            <div class="metric-label">Estimated Annual Savings (Optimized vs Baseline)</div>
            <div class="metric-value" style="color:{'#27AE60' if annual_saving>0 else '#C0392B'}">
                ${annual_saving:,.0f}
            </div>
            <div class="metric-delta">${b_cost:,.0f} → ${opt_cost:,.0f} per batch</div>
        </div>""", unsafe_allow_html=True)

# ── Tab 4: Historical Data Explorer ───────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-header">Synthetic Historical Dataset — Drug XYZ Line</div>', unsafe_allow_html=True)
    st.caption(f"Showing {len(df):,} simulated production batches calibrated to pharmaceutical industry benchmarks.")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        fig = px.histogram(df, x="yield_pct", nbins=40,
                           title="Yield Distribution",
                           color_discrete_sequence=["#1F4E79"])
        fig.update_layout(height=300, plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        fig2 = px.histogram(df, x="cost_per_batch", nbins=40,
                            title="Cost per Batch Distribution",
                            color_discrete_sequence=["#2E86AB"])
        fig2.update_layout(height=300, plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig2, use_container_width=True)
    with col_c:
        dev_counts = df["deviation_event"].value_counts().reset_index()
        dev_counts.columns = ["Deviation", "Count"]
        dev_counts["Deviation"] = dev_counts["Deviation"].map({0: "No Deviation", 1: "Deviation"})
        fig3 = px.pie(dev_counts, names="Deviation", values="Count",
                      title="Deviation Event Rate",
                      color_discrete_sequence=["#27AE60", "#C0392B"])
        fig3.update_layout(height=300)
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown('<div class="section-header">Yield vs. Maintenance Frequency (colored by deviation)</div>', unsafe_allow_html=True)
    sample = df.sample(500, random_state=42)
    fig4 = px.scatter(sample, x="maintenance_freq", y="yield_pct",
                      color=sample["deviation_event"].map({0: "No Deviation", 1: "Deviation"}),
                      color_discrete_map={"No Deviation": "#27AE60", "Deviation": "#C0392B"},
                      title="Yield vs Maintenance Frequency",
                      labels={"maintenance_freq": "Days Between Maintenance",
                              "yield_pct": "Yield (%)"})
    fig4.update_layout(height=380, plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig4, use_container_width=True)

    with st.expander("View raw data sample (50 rows)"):
        st.dataframe(df.head(50), use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Pharma AI Optimizer — Built by Anuj Raje | Synthetic data calibrated to public pharmaceutical industry benchmarks | Models: Random Forest Classifier & Regressor (scikit-learn)")
