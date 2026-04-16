import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Pharma AI Optimizer",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    .alert-box {
        background: #FDEDEC; border-left: 4px solid #C0392B;
        border-radius: 4px; padding: 12px 16px;
        font-size: 14px; color: #1a1a1a; margin-bottom: 8px;
    }
    .warn-box {
        background: #FEF9E7; border-left: 4px solid #E67E22;
        border-radius: 4px; padding: 12px 16px;
        font-size: 14px; color: #1a1a1a; margin-bottom: 8px;
    }
    .risk-high { color: #C0392B; font-weight: 600; }
    .risk-med  { color: #E67E22; font-weight: 600; }
    .risk-low  { color: #27AE60; font-weight: 600; }
    .facility-badge {
        display: inline-block;
        background: #1F4E79; color: white;
        font-size: 12px; font-weight: 500;
        padding: 4px 12px; border-radius: 2px;
        margin-right: 8px; margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def generate_data(n=2000, seed=42):
    rng = np.random.default_rng(seed)
    maint_interval       = rng.integers(1, 31, n)
    staffing_level       = rng.integers(2, 12, n)
    calibration_interval = rng.integers(1, 15, n)
    changeover_time      = rng.integers(30, 300, n)
    line_speed           = rng.uniform(50, 150, n)
    ambient_temp_var     = rng.uniform(0.1, 5.0, n)
    humidity_var         = rng.uniform(0.5, 10.0, n)
    batch_size           = rng.integers(5000, 50000, n)
    batches_per_shift    = rng.integers(1, 6, n)
    shifts_per_day       = rng.integers(1, 3, n)

    dev_risk_raw = (
        0.30 * (maint_interval / 30) +
        0.25 * (1 - staffing_level / 12) +
        0.20 * (calibration_interval / 15) +
        0.10 * (ambient_temp_var / 5) +
        0.10 * (humidity_var / 10) +
        0.05 * rng.uniform(0, 1, n)
    )
    deviation_event = (dev_risk_raw > 0.50).astype(int)

    yield_pct = (
        97.5
        - 0.08 * (maint_interval - 1)
        - 0.04 * (changeover_time - 30) / 10
        - 0.15 * deviation_event * rng.uniform(1, 3, n)
        - 0.02 * (calibration_interval - 1)
        + 0.03 * (staffing_level - 2)
        + rng.normal(0, 0.5, n)
    ).clip(85, 99.5)

    throughput = (
        line_speed * 60
        - 250 * (changeover_time / 60)
        - 120 * deviation_event
        - 80  * (maint_interval > 20).astype(int)
        + rng.normal(0, 100, n)
    ).clip(1500, 9000)

    labor_base    = staffing_level * 400
    maint_cost    = np.where(maint_interval <= 7, 800, np.where(maint_interval <= 14, 400, 150))
    downtime_cost = deviation_event * rng.uniform(2000, 8000, n)
    scrap_cost    = (100 - yield_pct) / 100 * batch_size * 0.85
    cost_per_batch = labor_base + maint_cost + downtime_cost + scrap_cost + rng.normal(0, 200, n)
    deviations_per_quarter = (dev_risk_raw * 12 + rng.uniform(0, 2, n)).clip(0, 15).round(1)
    annual_downtime_hrs = (deviation_event * rng.uniform(2, 12, n) + (maint_interval > 21).astype(int) * rng.uniform(4, 16, n))

    return pd.DataFrame({
        "maint_interval": maint_interval,
        "staffing_level": staffing_level,
        "calibration_interval": calibration_interval,
        "changeover_time": changeover_time,
        "line_speed": line_speed,
        "ambient_temp_var": ambient_temp_var,
        "humidity_var": humidity_var,
        "batch_size": batch_size,
        "deviation_event": deviation_event,
        "yield_pct": yield_pct,
        "throughput": throughput,
        "cost_per_batch": cost_per_batch,
        "deviations_per_quarter": deviations_per_quarter,
        "annual_downtime_hrs": annual_downtime_hrs,
    })

@st.cache_resource
def train_models(df):
    features = ["maint_interval", "staffing_level", "calibration_interval",
                "changeover_time", "line_speed", "ambient_temp_var", "humidity_var"]
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42)
    clf.fit(X_scaled, df["deviation_event"])
    reg_yield = RandomForestRegressor(n_estimators=150, max_depth=8, random_state=42)
    reg_yield.fit(X_scaled, df["yield_pct"])
    reg_tput = RandomForestRegressor(n_estimators=150, max_depth=8, random_state=42)
    reg_tput.fit(X_scaled, df["throughput"])
    reg_cost = RandomForestRegressor(n_estimators=150, max_depth=8, random_state=42)
    reg_cost.fit(X_scaled, df["cost_per_batch"])
    reg_dev_q = RandomForestRegressor(n_estimators=150, max_depth=8, random_state=42)
    reg_dev_q.fit(X_scaled, df["deviations_per_quarter"])
    importances = pd.Series(clf.feature_importances_, index=features).sort_values(ascending=True)
    return clf, reg_yield, reg_tput, reg_cost, reg_dev_q, scaler, features, importances

def predict_scenario(params, clf, reg_yield, reg_tput, reg_cost, reg_dev_q, scaler, features):
    X = np.array([[params[f] for f in features]])
    X_scaled = scaler.transform(X)
    return (
        clf.predict_proba(X_scaled)[0][1],
        reg_yield.predict(X_scaled)[0],
        reg_tput.predict(X_scaled)[0],
        reg_cost.predict(X_scaled)[0],
        reg_dev_q.predict(X_scaled)[0],
    )

df = generate_data()
clf, reg_yield, reg_tput, reg_cost, reg_dev_q, scaler, features, importances = train_models(df)

BASELINE = {
    "maint_interval": 14, "staffing_level": 6, "calibration_interval": 8,
    "changeover_time": 120, "line_speed": 100.0, "ambient_temp_var": 2.0, "humidity_var": 4.0,
}
b_dev, b_yield, b_tput, b_cost, b_devq = predict_scenario(BASELINE, clf, reg_yield, reg_tput, reg_cost, reg_dev_q, scaler, features)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Line Configuration")
    st.markdown("Adjust parameters to model outcomes for **Drug XYZ — Packaging Line 3**.")
    st.markdown("---")
    st.markdown("**🔧 Preventive Maintenance**")
    maint_interval = st.slider("PM Interval (days)", 1, 30, 14,
        help="Days between scheduled PM events. Industry standard: 7–14 days for high-speed packaging lines.")
    st.markdown("**👷 Line Staffing**")
    staffing_level = st.slider("Operators on Line", 2, 11, 6,
        help="Operators per shift on this line. Typical range: 4–8 for high-speed packaging.")
    st.markdown("**📡 Instrument Calibration**")
    calibration_interval = st.slider("Calibration Interval (hours)", 1, 14, 8,
        help="Hours between in-process instrument calibration. GMP best practice: every 4–8 hours.")
    st.markdown("**⏱️ Changeover**")
    changeover_time = st.slider("Changeover Time (min)", 30, 299, 120,
        help="Time to complete a full line changeover. Industry benchmark: 60–120 min.")
    st.markdown("**🌡️ Environmental Controls**")
    ambient_temp_var = st.slider("Temperature Variance (°C)", 0.1, 5.0, 2.0, step=0.1,
        help="Ambient temperature variance. Acceptable: ±2°C.")
    humidity_var = st.slider("Humidity Variance (% RH)", 0.5, 10.0, 4.0, step=0.5,
        help="Relative humidity variance. Acceptable: ±5% RH.")
    st.markdown("**📦 Production Volume**")
    batches_per_day = st.slider("Batches per Day", 1, 10, 4)
    working_days = st.slider("Working Days per Year", 200, 300, 250)
    line_speed = 100.0
    st.markdown("---")
    if st.button("🔄 Reset to Baseline", use_container_width=True):
        st.rerun()

params = {
    "maint_interval": maint_interval, "staffing_level": staffing_level,
    "calibration_interval": calibration_interval, "changeover_time": changeover_time,
    "line_speed": line_speed, "ambient_temp_var": ambient_temp_var, "humidity_var": humidity_var,
}
s_dev, s_yield, s_tput, s_cost, s_devq = predict_scenario(params, clf, reg_yield, reg_tput, reg_cost, reg_dev_q, scaler, features)
annual_batches = batches_per_day * working_days
annual_cost = s_cost * annual_batches
baseline_annual_cost = b_cost * annual_batches
annual_saving = baseline_annual_cost - annual_cost
annual_devs = s_devq * 4

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("# ⚗️ Pharma AI Optimizer")
st.markdown("Model the operational and financial impact of maintenance, staffing, and changeover decisions on manufacturing cost, capacity, and compliance risk.")
st.markdown("""
<span class="facility-badge">Drug XYZ</span>
<span class="facility-badge">Packaging Line 3</span>
<span class="facility-badge">Solid Dose — Tablet</span>
<span class="facility-badge">FDA-Regulated Site</span>
""", unsafe_allow_html=True)
st.markdown("---")

# ── KPIs ───────────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    risk_cls = "risk-high" if s_dev > 0.6 else ("risk-med" if s_dev > 0.35 else "risk-low")
    delta_dev = (s_dev - b_dev) * 100
    icon = "🔴" if delta_dev > 0 else "🟢"
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Deviation Risk</div>
        <div class="metric-value"><span class="{risk_cls}">{s_dev*100:.1f}%</span></div>
        <div class="metric-delta">{icon} {delta_dev:+.1f}pp vs baseline</div>
    </div>""", unsafe_allow_html=True)
with col2:
    delta_yield = s_yield - b_yield
    icon = "🟢" if delta_yield > 0 else "🔴"
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Batch Yield</div>
        <div class="metric-value">{s_yield:.1f}%</div>
        <div class="metric-delta">{icon} {delta_yield:+.2f}pp vs baseline</div>
    </div>""", unsafe_allow_html=True)
with col3:
    delta_tput = s_tput - b_tput
    icon = "🟢" if delta_tput > 0 else "🔴"
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Line Throughput (units/hr)</div>
        <div class="metric-value">{s_tput:,.0f}</div>
        <div class="metric-delta">{icon} {delta_tput:+,.0f} vs baseline</div>
    </div>""", unsafe_allow_html=True)
with col4:
    delta_cost = s_cost - b_cost
    icon = "🟢" if delta_cost < 0 else "🔴"
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Cost per Batch</div>
        <div class="metric-value">${s_cost:,.0f}</div>
        <div class="metric-delta">{icon} ${delta_cost:+,.0f} vs baseline</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Annual Projections ─────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Annual Line Projections — Drug XYZ Packaging Line 3</div>', unsafe_allow_html=True)
col_a, col_b, col_c, col_d = st.columns(4)
with col_a:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Annual Batches</div>
        <div class="metric-value">{annual_batches:,}</div>
        <div class="metric-delta">{batches_per_day} batches/day × {working_days} days</div>
    </div>""", unsafe_allow_html=True)
with col_b:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Projected Annual Cost</div>
        <div class="metric-value">${annual_cost/1e6:.2f}M</div>
        <div class="metric-delta">${s_cost:,.0f} × {annual_batches:,} batches</div>
    </div>""", unsafe_allow_html=True)
with col_c:
    icon = "🟢" if annual_saving > 0 else "🔴"
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">vs Baseline Annual Cost</div>
        <div class="metric-value" style="color:{'#27AE60' if annual_saving>0 else '#C0392B'}">${annual_saving/1e6:+.2f}M</div>
        <div class="metric-delta">{icon} savings vs standard operations</div>
    </div>""", unsafe_allow_html=True)
with col_d:
    dev_cls = "risk-high" if annual_devs > 8 else ("risk-med" if annual_devs > 4 else "risk-low")
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Projected Annual Deviations</div>
        <div class="metric-value"><span class="{dev_cls}">{annual_devs:.0f}</span></div>
        <div class="metric-delta">estimated written deviations/year</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Strategy Comparison",
    "🔍 Operational Risk Drivers",
    "💰 Cost & Capacity Trade-offs",
    "📈 Historical Line Data"
])

with tab1:
    st.markdown('<div class="section-header">Operating Strategy Comparison</div>', unsafe_allow_html=True)
    strategies = {
        "Current Configuration":   params,
        "Standard Operations":     BASELINE,
        "Aggressive PM (Weekly)":  {**BASELINE, "maint_interval": 7, "calibration_interval": 4},
        "Reduced Staffing":        {**BASELINE, "staffing_level": 3},
        "Full Staffing":           {**BASELINE, "staffing_level": 10},
        "Fast Changeover":         {**BASELINE, "changeover_time": 45},
        "Optimized Configuration": {**BASELINE, "maint_interval": 7, "staffing_level": 8,
                                    "calibration_interval": 4, "changeover_time": 60,
                                    "ambient_temp_var": 0.8, "humidity_var": 1.5},
    }
    rows = []
    for name, p in strategies.items():
        d, y, t, c, dq = predict_scenario(p, clf, reg_yield, reg_tput, reg_cost, reg_dev_q, scaler, features)
        rows.append({"Strategy": name, "Deviation Risk (%)": round(d*100,1),
                     "Batch Yield (%)": round(y,2), "Throughput (units/hr)": round(t),
                     "Cost/Batch ($)": round(c), "Est. Annual Deviations": round(dq*4,1)})
    results_df = pd.DataFrame(rows)

    col_a, col_b = st.columns(2)
    with col_a:
        fig = px.bar(results_df, x="Strategy", y="Deviation Risk (%)",
                     color="Deviation Risk (%)", color_continuous_scale=["#27AE60","#F39C12","#C0392B"],
                     title="Deviation Risk by Operating Strategy")
        fig.update_layout(showlegend=False, height=350, plot_bgcolor="white", paper_bgcolor="white", xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        fig2 = px.bar(results_df, x="Strategy", y="Cost/Batch ($)",
                      color="Cost/Batch ($)", color_continuous_scale=["#27AE60","#F39C12","#C0392B"],
                      title="Cost per Batch by Operating Strategy")
        fig2.update_layout(showlegend=False, height=350, plot_bgcolor="white", paper_bgcolor="white", xaxis_tickangle=-30)
        st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(results_df.style.format({
        "Deviation Risk (%)": "{:.1f}%", "Batch Yield (%)": "{:.2f}%",
        "Throughput (units/hr)": "{:,.0f}", "Cost/Batch ($)": "${:,.0f}",
        "Est. Annual Deviations": "{:.1f}"
    }), use_container_width=True, hide_index=True)

    st.markdown('<div class="section-header">Operational Recommendations</div>', unsafe_allow_html=True)
    if s_dev > 0.6:
        st.markdown(f'<div class="alert-box">🚨 <b>High deviation risk ({s_dev*100:.1f}%)</b> — projected <b>{annual_devs:.0f} written deviations/year</b>. Each deviation averages $3,000–$8,000 in investigation and downtime cost. Recommend reducing PM interval and increasing calibration frequency.</div>', unsafe_allow_html=True)
    elif s_dev > 0.35:
        st.markdown(f'<div class="warn-box">⚠️ <b>Moderate deviation risk ({s_dev*100:.1f}%)</b> — projected <b>{annual_devs:.0f} deviations/year</b>. Consider tightening PM schedule or environmental controls before next inspection cycle.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="insight-box">✅ <b>Deviation risk within acceptable range ({s_dev*100:.1f}%)</b> — projected <b>{annual_devs:.0f} deviations/year</b>.</div>', unsafe_allow_html=True)

    opt_cost_val = results_df.loc[results_df["Strategy"]=="Optimized Configuration","Cost/Batch ($)"].values[0]
    opt_annual_saving = (b_cost - opt_cost_val) * annual_batches
    best_yield = results_df.loc[results_df["Batch Yield (%)"].idxmax(), "Strategy"]
    st.markdown(f'<div class="insight-box">💡 <b>Optimized Configuration</b> is projected to save <b>${opt_annual_saving/1e6:.2f}M annually</b> at {annual_batches:,} batches/year while reducing deviation risk.</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="insight-box">📈 <b>Highest batch yield</b> achieved under <b>{best_yield}</b> — maximizing saleable output per run.</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="section-header">Operational Risk Drivers — What Is Causing Deviations?</div>', unsafe_allow_html=True)
    feature_labels = {
        "maint_interval": "PM Interval (days)", "staffing_level": "Staffing Level",
        "calibration_interval": "Calibration Interval (hrs)", "changeover_time": "Changeover Time (min)",
        "line_speed": "Line Speed (units/min)", "ambient_temp_var": "Temp Variance (°C)", "humidity_var": "Humidity Variance (% RH)",
    }
    imp_labeled = importances.rename(index=feature_labels)

    col_a, col_b = st.columns(2)
    with col_a:
        fig = px.bar(x=imp_labeled.values, y=imp_labeled.index, orientation='h',
                     labels={"x": "Relative Importance", "y": ""},
                     color=imp_labeled.values, color_continuous_scale=["#BDD7EE","#1F4E79"],
                     title="Top Drivers of Deviation Events — Line 3")
        fig.update_layout(showlegend=False, height=380, plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        risk_by_maint = []
        for m in range(1, 31):
            p = {**params, "maint_interval": m}
            d, _, _, _, dq = predict_scenario(p, clf, reg_yield, reg_tput, reg_cost, reg_dev_q, scaler, features)
            risk_by_maint.append({"PM Interval (days)": m, "Deviation Risk (%)": d*100, "Est. Annual Deviations": dq*4})
        maint_df = pd.DataFrame(risk_by_maint)
        fig2 = px.line(maint_df, x="PM Interval (days)", y="Deviation Risk (%)",
                       title="Deviation Risk vs PM Interval", color_discrete_sequence=["#1F4E79"])
        fig2.add_hline(y=60, line_dash="dash", line_color="#C0392B", annotation_text="High Risk (60%)")
        fig2.add_hline(y=35, line_dash="dash", line_color="#E67E22", annotation_text="Moderate Risk (35%)")
        fig2.update_layout(height=380, plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-header">Staffing Level Impact on Risk & Cost</div>', unsafe_allow_html=True)
    staff_data = []
    for s in range(2, 12):
        p = {**params, "staffing_level": s}
        d, y, _, c, dq = predict_scenario(p, clf, reg_yield, reg_tput, reg_cost, reg_dev_q, scaler, features)
        staff_data.append({"Operators on Line": s, "Deviation Risk (%)": round(d*100,1),
                           "Batch Yield (%)": round(y,2), "Cost/Batch ($)": round(c), "Est. Annual Deviations": round(dq*4,1)})
    staff_df = pd.DataFrame(staff_data)
    col_a, col_b = st.columns(2)
    with col_a:
        fig3 = px.line(staff_df, x="Operators on Line", y="Deviation Risk (%)",
                       title="Deviation Risk vs Staffing Level", color_discrete_sequence=["#C0392B"])
        fig3.update_layout(height=300, plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig3, use_container_width=True)
    with col_b:
        fig4 = px.line(staff_df, x="Operators on Line", y="Cost/Batch ($)",
                       title="Cost per Batch vs Staffing Level", color_discrete_sequence=["#1F4E79"])
        fig4.update_layout(height=300, plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig4, use_container_width=True)

with tab3:
    st.markdown('<div class="section-header">Cost vs. Compliance Risk Trade-off</div>', unsafe_allow_html=True)
    sweep = []
    for mf in [7, 10, 14, 21, 28]:
        for sl in [3, 5, 7, 9, 11]:
            p = {**params, "maint_interval": mf, "staffing_level": sl}
            d, y, t, c, dq = predict_scenario(p, clf, reg_yield, reg_tput, reg_cost, reg_dev_q, scaler, features)
            sweep.append({"PM Interval (days)": mf, "Staffing Level": sl,
                          "Deviation Risk (%)": round(d*100,1), "Batch Yield (%)": round(y,2),
                          "Throughput (units/hr)": round(t), "Cost/Batch ($)": round(c),
                          "Est. Annual Deviations": round(dq*4,1)})
    sweep_df = pd.DataFrame(sweep)
    fig = px.scatter(sweep_df, x="Cost/Batch ($)", y="Deviation Risk (%)",
                     size="Throughput (units/hr)", color="Batch Yield (%)",
                     hover_data=["PM Interval (days)", "Staffing Level", "Est. Annual Deviations"],
                     color_continuous_scale="RdYlGn",
                     title="Cost vs Compliance Risk vs Throughput — bubble = throughput, color = yield")
    fig.update_layout(height=450, plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Changeover Time Impact on Throughput & Annual Cost</div>', unsafe_allow_html=True)
    co_data = []
    for co in range(30, 300, 15):
        p = {**params, "changeover_time": co}
        d, y, t, c, dq = predict_scenario(p, clf, reg_yield, reg_tput, reg_cost, reg_dev_q, scaler, features)
        co_data.append({"Changeover Time (min)": co, "Throughput (units/hr)": t,
                        "Annual Cost ($M)": c * annual_batches / 1e6})
    co_df = pd.DataFrame(co_data)
    col_a, col_b = st.columns(2)
    with col_a:
        fig2 = px.line(co_df, x="Changeover Time (min)", y="Throughput (units/hr)",
                       title="Line Throughput vs Changeover Time", color_discrete_sequence=["#1F4E79"])
        fig2.update_layout(height=300, plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig2, use_container_width=True)
    with col_b:
        fig3 = px.line(co_df, x="Changeover Time (min)", y="Annual Cost ($M)",
                       title="Annual Cost vs Changeover Time", color_discrete_sequence=["#C0392B"])
        fig3.update_layout(height=300, plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown('<div class="section-header">💰 Investment Decision Calculator</div>', unsafe_allow_html=True)
    col_x, col_y = st.columns(2)
    with col_x:
        calc_batches = st.number_input("Annual Batches on This Line", min_value=50, max_value=5000, value=annual_batches, step=50)
        avg_batch_revenue = st.number_input("Average Revenue per Batch ($)", min_value=1000, max_value=500000, value=25000, step=1000)
    with col_y:
        opt_p = {**BASELINE, "maint_interval": 7, "staffing_level": 8, "calibration_interval": 4,
                 "changeover_time": 60, "ambient_temp_var": 0.8, "humidity_var": 1.5}
        _, opt_yield, _, opt_cost, opt_devq = predict_scenario(opt_p, clf, reg_yield, reg_tput, reg_cost, reg_dev_q, scaler, features)
        cost_saving = (b_cost - opt_cost) * calc_batches
        yield_gain_revenue = (opt_yield - b_yield) / 100 * avg_batch_revenue * calc_batches
        dev_cost_saving = (b_devq - opt_devq) * 4 * 5000
        total_impact = cost_saving + yield_gain_revenue + dev_cost_saving
        st.markdown(f"""<div class="metric-card" style="margin-top:8px">
            <div class="metric-label">Total Annual Financial Impact — Optimized vs Standard Operations</div>
            <div class="metric-value" style="color:{'#27AE60' if total_impact>0 else '#C0392B'}">${total_impact/1e6:.2f}M</div>
            <div class="metric-delta">Cost savings: ${cost_saving/1e6:.2f}M &nbsp;|&nbsp; Yield gain: ${yield_gain_revenue/1e6:.2f}M &nbsp;|&nbsp; Deviation reduction: ${dev_cost_saving/1e3:.0f}K</div>
        </div>""", unsafe_allow_html=True)

with tab4:
    st.markdown('<div class="section-header">Historical Line Data — Drug XYZ Packaging Line 3</div>', unsafe_allow_html=True)
    st.caption(f"{len(df):,} simulated production batches calibrated to pharmaceutical solid dose manufacturing benchmarks.")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        fig = px.histogram(df, x="yield_pct", nbins=40, title="Batch Yield Distribution (%)",
                           color_discrete_sequence=["#1F4E79"])
        fig.update_layout(height=280, plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        fig2 = px.histogram(df, x="cost_per_batch", nbins=40, title="Cost per Batch Distribution ($)",
                            color_discrete_sequence=["#2E86AB"])
        fig2.update_layout(height=280, plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig2, use_container_width=True)
    with col_c:
        dev_counts = df["deviation_event"].value_counts().reset_index()
        dev_counts.columns = ["Event","Count"]
        dev_counts["Event"] = dev_counts["Event"].map({0:"No Deviation",1:"Deviation Event"})
        fig3 = px.pie(dev_counts, names="Event", values="Count", title="Deviation Event Rate",
                      color_discrete_sequence=["#27AE60","#C0392B"])
        fig3.update_layout(height=280)
        st.plotly_chart(fig3, use_container_width=True)

    sample = df.sample(600, random_state=42)
    fig4 = px.scatter(sample, x="maint_interval", y="yield_pct",
                      color=sample["deviation_event"].map({0:"No Deviation",1:"Deviation Event"}),
                      color_discrete_map={"No Deviation":"#27AE60","Deviation Event":"#C0392B"},
                      labels={"maint_interval":"PM Interval (days)","yield_pct":"Batch Yield (%)"},
                      title="Batch Yield vs PM Interval — Line 3 Historical Data")
    fig4.update_layout(height=380, plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig4, use_container_width=True)

    with st.expander("View raw batch data (50 records)"):
        display_df = df.head(50).copy()
        display_df.columns = ["PM Interval","Staffing","Cal. Interval","Changeover",
                               "Line Speed","Temp Var","Humidity Var","Batch Size",
                               "Deviation Event","Yield %","Throughput","Cost/Batch",
                               "Devs/Quarter","Downtime Hrs"]
        st.dataframe(display_df, use_container_width=True)

st.markdown("---")
st.caption("Pharma AI Optimizer — Built by Anuj Raje | Synthetic data calibrated to pharmaceutical solid dose manufacturing benchmarks | Random Forest Classifier & Regressors (scikit-learn) | Operational decision support and investment analysis")
