import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import json
import time

st.set_page_config(
    page_title="TrafficGuard AI",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Share+Tech+Mono&family=Exo+2:wght@300;400;600;800&display=swap');

html, body, [class*="css"] { font-family: 'Exo 2', sans-serif; }

.stApp { background: #030712; color: #e2e8f0; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0f1e 0%, #050a17 100%);
    border-right: 1px solid #1e3a5f;
}

.metric-card {
    background: linear-gradient(135deg, #0d1b2a 0%, #0a1628 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 22px 24px;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 2px;
    background: linear-gradient(90deg, #f97316, #ef4444);
}
.metric-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.15em;
    color: #64748b;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.metric-value {
    font-family: 'Rajdhani', sans-serif;
    font-size: 36px;
    font-weight: 700;
    color: #f97316;
    line-height: 1;
}
.metric-sub {
    font-size: 11px;
    color: #475569;
    margin-top: 6px;
    font-family: 'Share Tech Mono', monospace;
}
.section-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 20px;
    font-weight: 700;
    color: #f1f5f9;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    border-left: 3px solid #f97316;
    padding-left: 14px;
    margin: 24px 0 14px 0;
}
.pred-badge {
    border-radius: 16px;
    padding: 28px 32px;
    text-align: center;
    margin: 16px 0;
}
.pred-badge-1 { background: linear-gradient(135deg,#14532d,#166534); border: 1px solid #22c55e; color: #86efac; }
.pred-badge-2 { background: linear-gradient(135deg,#713f12,#78350f); border: 1px solid #f59e0b; color: #fde68a; }
.pred-badge-3 { background: linear-gradient(135deg,#7c2d12,#9a3412); border: 1px solid #f97316; color: #fdba74; }
.pred-badge-4 { background: linear-gradient(135deg,#450a0a,#7f1d1d); border: 1px solid #ef4444; color: #fca5a5; }
.sev-label { font-family:'Share Tech Mono',monospace; font-size:11px; letter-spacing:0.2em; opacity:0.8; margin-bottom:6px; }
.sev-num   { font-family:'Rajdhani',sans-serif; font-size:64px; font-weight:700; line-height:1; }
.sev-desc  { font-family:'Exo 2',sans-serif; font-size:16px; font-weight:600; margin-top:8px; }

.stButton > button {
    background: linear-gradient(135deg,#ea580c,#dc2626) !important;
    color: white !important;
    border: none !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em !important;
    border-radius: 8px !important;
    padding: 12px 28px !important;
    width: 100% !important;
    text-transform: uppercase !important;
}
h1,h2,h3 { font-family:'Rajdhani',sans-serif !important; }
.stTabs [data-baseweb="tab"] {
    font-family:'Rajdhani',sans-serif !important;
    font-size:15px !important;
    font-weight:600 !important;
    color:#64748b !important;
}
.stTabs [aria-selected="true"] { color:#f97316 !important; }
</style>
""", unsafe_allow_html=True)


# ── Load model & data ──────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    rf         = joblib.load('models/rf_model.pkl')
    le_weather = joblib.load('models/le_weather.pkl')
    le_road    = joblib.load('models/le_road.pkl')
    le_day     = joblib.load('models/le_day.pkl')
    le_state   = joblib.load('models/le_state.pkl')
    with open('models/model_meta.json') as f:
        meta = json.load(f)
    return rf, le_weather, le_road, le_day, le_state, meta

@st.cache_data
def load_data():
    return pd.read_csv('data/accidents_cleaned.csv')

rf, le_weather, le_road, le_day, le_state, meta = load_model()
df = load_data()


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:20px 0 30px 0;'>
        <div style='font-family:Rajdhani,sans-serif;font-size:28px;font-weight:700;color:#f97316;letter-spacing:0.1em;'>
            🚨 TrafficGuard
        </div>
        <div style='font-family:Share Tech Mono,monospace;font-size:10px;color:#475569;letter-spacing:0.2em;margin-top:4px;'>
            AI PREDICTION SYSTEM
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("Navigation", [
        "🏠  Dashboard",
        "🔮  Predict Severity",
        "📊  Model Analytics",
        "🗂️  Data Explorer",
    ])

    st.markdown("---")
    st.markdown(f"""
    <div style='font-family:Share Tech Mono,monospace;font-size:10px;color:#334155;line-height:2;padding:0 8px;'>
    ALGORITHM &nbsp;&nbsp; Random Forest<br>
    ACCURACY &nbsp;&nbsp;&nbsp; {meta['accuracy']*100:.1f}%<br>
    TREES &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 200<br>
    DATASET &nbsp;&nbsp;&nbsp;&nbsp; {len(df):,} records<br>
    FEATURES &nbsp;&nbsp;&nbsp; {len(meta['features'])}
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  PAGE 1 — DASHBOARD
# ═══════════════════════════════════════════════════════════════
if "Dashboard" in page:
    st.markdown("""
    <div style='padding:10px 0 28px 0;'>
        <div style='font-family:Rajdhani,sans-serif;font-size:42px;font-weight:800;color:#f1f5f9;line-height:1;'>
            Traffic Accident <span style='color:#f97316;'>Prediction</span>
        </div>
        <div style='font-family:Share Tech Mono,monospace;font-size:11px;color:#475569;letter-spacing:0.1em;margin-top:8px;'>
            RANDOM FOREST CLASSIFIER  ·  FINAL YEAR ENGINEERING PROJECT
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>Model Accuracy</div>
            <div class='metric-value'>{meta['accuracy']*100:.1f}%</div>
            <div class='metric-sub'>5-FOLD CV: {meta['cv_mean']*100:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>Total Records</div>
            <div class='metric-value'>{len(df):,}</div>
            <div class='metric-sub'>CLEANED DATASET</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        pct = round(len(df[df['Severity'] >= 3]) / len(df) * 100, 1)
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>High Severity</div>
            <div class='metric-value'>{pct}%</div>
            <div class='metric-sub'>SEVERITY 3 OR 4</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>Decision Trees</div>
            <div class='metric-value'>200</div>
            <div class='metric-sub'>ENSEMBLE MODEL</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='section-title'>Severity Distribution</div>", unsafe_allow_html=True)
        sev = df['Severity'].value_counts().sort_index()
        labels = {1:'Low', 2:'Moderate', 3:'High', 4:'Critical'}
        colors = ['#22c55e','#f59e0b','#f97316','#ef4444']
        fig = go.Figure(go.Bar(
            x=[f"Sev {i} — {labels[i]}" for i in sev.index],
            y=sev.values,
            marker=dict(color=colors),
            text=sev.values, textposition='outside',
            textfont=dict(color='#94a3b8', family='Share Tech Mono', size=11)
        ))
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#64748b', family='Exo 2'),
            xaxis=dict(showgrid=False, color='#334155'),
            yaxis=dict(showgrid=True, gridcolor='#0f1f2e', color='#334155'),
            margin=dict(l=0,r=0,t=10,b=0), height=280
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("<div class='section-title'>Accidents by Hour of Day</div>", unsafe_allow_html=True)
        hourly = df.groupby('Hour').size().reset_index(name='Count')
        fig2 = go.Figure(go.Scatter(
            x=hourly['Hour'], y=hourly['Count'],
            mode='lines+markers',
            line=dict(color='#f97316', width=2.5),
            marker=dict(size=6, color='#ef4444'),
            fill='tozeroy', fillcolor='rgba(249,115,22,0.08)'
        ))
        fig2.add_vline(x=8,  line=dict(color='#facc15',width=1,dash='dash'), annotation_text="Rush AM")
        fig2.add_vline(x=17, line=dict(color='#facc15',width=1,dash='dash'), annotation_text="Rush PM")
        fig2.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#64748b', family='Exo 2'),
            xaxis=dict(showgrid=False, color='#334155', tickvals=list(range(0,24,2))),
            yaxis=dict(showgrid=True, gridcolor='#0f1f2e', color='#334155'),
            margin=dict(l=0,r=0,t=10,b=0), height=280
        )
        st.plotly_chart(fig2, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='section-title'>Weather vs Severity</div>", unsafe_allow_html=True)
        wdf = df.groupby(['Weather_Condition','Severity']).size().reset_index(name='Count')
        fig3 = px.bar(wdf, x='Weather_Condition', y='Count', color='Severity',
                      color_continuous_scale=['#22c55e','#f59e0b','#f97316','#ef4444'])
        fig3.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#64748b', family='Exo 2'),
            xaxis=dict(showgrid=False, color='#334155'),
            yaxis=dict(showgrid=True, gridcolor='#0f1f2e', color='#334155'),
            margin=dict(l=0,r=0,t=10,b=0), height=280
        )
        st.plotly_chart(fig3, use_container_width=True)

    with c2:
        st.markdown("<div class='section-title'>Speed Limit Heatmap</div>", unsafe_allow_html=True)
        heat = df.groupby(['Speed_Limit','Severity']).size().unstack(fill_value=0)
        fig4 = go.Figure(go.Heatmap(
            z=heat.values,
            x=[f"Sev {c}" for c in heat.columns],
            y=heat.index.astype(str),
            colorscale=[[0,'#0a0f1e'],[0.5,'#7c2d12'],[1,'#ef4444']]
        ))
        fig4.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#64748b', family='Exo 2'),
            margin=dict(l=0,r=0,t=10,b=0), height=280
        )
        st.plotly_chart(fig4, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
#  PAGE 2 — PREDICT
# ═══════════════════════════════════════════════════════════════
elif "Predict" in page:
    st.markdown("""
    <div style='padding:10px 0 24px 0;'>
        <div style='font-family:Rajdhani,sans-serif;font-size:38px;font-weight:800;color:#f1f5f9;'>
            🔮 Accident <span style='color:#f97316;'>Severity Predictor</span>
        </div>
        <div style='font-family:Share Tech Mono,monospace;font-size:11px;color:#475569;letter-spacing:0.1em;'>
            FILL IN CONDITIONS  →  GET REAL-TIME PREDICTION
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_form, col_result = st.columns([3, 2], gap="large")

    with col_form:
        st.markdown("#### 🌦 Weather Conditions")
        c1, c2, c3 = st.columns(3)
        with c1: temperature   = st.slider("Temperature (°F)", 5, 105, 65)
        with c2: wind_speed    = st.slider("Wind Speed (mph)", 0, 60, 10)
        with c3: visibility    = st.slider("Visibility (mi)", 0.1, 10.0, 8.0)

        c1, c2, c3 = st.columns(3)
        with c1: precipitation = st.slider("Precipitation (in)", 0.0, 3.0, 0.0)
        with c2: humidity      = st.slider("Humidity (%)", 10, 100, 60)
        with c3: pressure      = st.slider("Pressure (in)", 27.0, 32.0, 29.9)

        weather_cond = st.selectbox("Weather Condition", sorted(meta['weather_classes']))

        st.markdown("#### 🛣 Road Information")
        c1, c2 = st.columns(2)
        with c1:
            road_type   = st.selectbox("Road Type", sorted(meta['road_classes']))
            speed_limit = st.selectbox("Speed Limit (mph)", [25,35,45,55,65,70,75])
        with c2:
            state       = st.selectbox("State", sorted(meta['state_classes']))
            hour        = st.slider("Hour of Day (0=midnight)", 0, 23, 8)

        c1, c2, c3 = st.columns(3)
        with c1:
            junction       = st.checkbox("Junction")
            traffic_signal = st.checkbox("Traffic Signal")
        with c2:
            crossing = st.checkbox("Crossing")
            stop     = st.checkbox("Stop Sign")
        with c3:
            amenity     = st.checkbox("Amenity Nearby")
        day_of_week = st.selectbox("Day of Week", meta['day_classes'])
        month       = st.slider("Month (1=Jan, 12=Dec)", 1, 12, 6)

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("⚡  ANALYZE & PREDICT SEVERITY")

    with col_result:
        st.markdown("<div class='section-title'>Prediction Result</div>", unsafe_allow_html=True)
        SEV_INFO = {
            1: ("LOW",      "Minor incident. Traffic minimally affected.",    "pred-badge-1"),
            2: ("MODERATE", "Significant impact. Expect delays.",             "pred-badge-2"),
            3: ("HIGH",     "Major accident. Road likely blocked.",           "pred-badge-3"),
            4: ("CRITICAL", "Severe accident. Emergency response required.",  "pred-badge-4"),
        }

        if predict_btn:
            is_rush   = 1 if (7 <= hour <= 9) or (16 <= hour <= 18) else 0
            is_night  = 1 if hour < 6 or hour >= 22 else 0
            is_weekend = 1 if day_of_week in ['Saturday','Sunday'] else 0
            bad_w     = 1 if weather_cond in ['Heavy Rain','Snow','Fog','Thunderstorm','Hail'] else 0

            X_pred = np.array([[
                temperature, wind_speed, visibility, precipitation,
                humidity, pressure, speed_limit,
                le_weather.transform([weather_cond])[0],
                le_road.transform([road_type])[0],
                hour,
                le_day.transform([day_of_week])[0],
                month,
                le_state.transform([state])[0],
                int(junction), int(traffic_signal), int(crossing),
                int(stop), int(amenity),
                is_rush, is_night, is_weekend, bad_w
            ]])

            with st.spinner("Running model..."):
                time.sleep(0.5)
                pred  = rf.predict(X_pred)[0]
                proba = rf.predict_proba(X_pred)[0]

            label, desc, badge_cls = SEV_INFO[pred]
            st.markdown(f"""
            <div class='pred-badge {badge_cls}'>
                <div class='sev-label'>PREDICTED SEVERITY</div>
                <div class='sev-num'>{pred}</div>
                <div class='sev-desc'>{label}</div>
                <div style='font-size:13px;opacity:0.75;margin-top:10px;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<div class='section-title'>Class Probabilities</div>", unsafe_allow_html=True)
            prob_colors = ['#22c55e','#f59e0b','#f97316','#ef4444']
            for i, (p, clr) in enumerate(zip(proba, prob_colors), start=1):
                st.markdown(f"""
                <div style='display:flex;align-items:center;gap:10px;margin-bottom:10px;'>
                    <div style='font-family:Share Tech Mono,monospace;font-size:11px;color:#64748b;width:55px;'>SEV {i}</div>
                    <div style='flex:1;background:#0d1b2a;border-radius:4px;height:14px;overflow:hidden;'>
                        <div style='width:{p*100:.1f}%;height:100%;background:{clr};border-radius:4px;'></div>
                    </div>
                    <div style='font-family:Share Tech Mono,monospace;font-size:12px;color:{clr};width:45px;text-align:right;'>{p*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<div class='section-title'>Risk Radar</div>", unsafe_allow_html=True)
            risk_vals = [
                min(wind_speed/60, 1),
                1 - min(visibility/10, 1),
                min(precipitation/3, 1),
                float(bad_w),
                speed_limit/75,
                float(is_night)
            ]
            cats = ['Wind','Low Visibility','Precipitation','Bad Weather','Speed','Night']
            fig_r = go.Figure(go.Scatterpolar(
                r=risk_vals + [risk_vals[0]],
                theta=cats + [cats[0]],
                fill='toself',
                fillcolor='rgba(249,115,22,0.15)',
                line=dict(color='#f97316', width=2),
                marker=dict(color='#ef4444', size=6)
            ))
            fig_r.update_layout(
                polar=dict(
                    bgcolor='rgba(0,0,0,0)',
                    radialaxis=dict(visible=True, range=[0,1], color='#334155', gridcolor='#1e3a5f', tickfont=dict(size=8)),
                    angularaxis=dict(color='#64748b', gridcolor='#1e3a5f')
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#64748b', family='Exo 2'),
                margin=dict(l=20,r=20,t=20,b=20), height=280
            )
            st.plotly_chart(fig_r, use_container_width=True)

        else:
            st.markdown("""
            <div style='background:#0d1b2a;border:1px dashed #1e3a5f;border-radius:12px;
                        padding:60px 30px;text-align:center;margin-top:20px;'>
                <div style='font-size:48px;margin-bottom:16px;'>🎯</div>
                <div style='font-family:Rajdhani,sans-serif;font-size:20px;color:#334155;font-weight:600;'>
                    Set conditions on the left<br>and click ANALYZE
                </div>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  PAGE 3 — MODEL ANALYTICS
# ═══════════════════════════════════════════════════════════════
elif "Analytics" in page:
    st.markdown("""
    <div style='padding:10px 0 24px 0;'>
        <div style='font-family:Rajdhani,sans-serif;font-size:38px;font-weight:800;color:#f1f5f9;'>
            📊 Model <span style='color:#f97316;'>Analytics</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Feature Importance", "Confusion Matrix", "Model Report"])

    with tab1:
        st.markdown("<div class='section-title'>Feature Importances</div>", unsafe_allow_html=True)
        fi_df = pd.DataFrame(list(meta['feature_importances'].items()),
                             columns=['Feature','Importance']).sort_values('Importance')
        fi_df['Feature'] = fi_df['Feature'].str.replace('_',' ').str.title()
        fig_fi = go.Figure(go.Bar(
            x=fi_df['Importance'], y=fi_df['Feature'],
            orientation='h',
            marker=dict(
                color=fi_df['Importance'],
                colorscale=[[0,'#0d1b2a'],[0.4,'#7c2d12'],[1,'#f97316']]
            ),
            text=[f"{v:.4f}" for v in fi_df['Importance']],
            textposition='outside',
            textfont=dict(color='#94a3b8', family='Share Tech Mono', size=10)
        ))
        fig_fi.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#64748b', family='Exo 2'),
            xaxis=dict(showgrid=True, gridcolor='#0f1f2e', color='#334155'),
            yaxis=dict(showgrid=False, color='#94a3b8'),
            margin=dict(l=10,r=80,t=10,b=10), height=520
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    with tab2:
        st.markdown("<div class='section-title'>Confusion Matrix</div>", unsafe_allow_html=True)
        cm = np.array(meta['confusion_matrix'])
        labels = ['Sev 1 Low','Sev 2 Moderate','Sev 3 High','Sev 4 Critical']
        fig_cm = go.Figure(go.Heatmap(
            z=cm, x=labels, y=labels,
            colorscale=[[0,'#030712'],[0.3,'#1e3a5f'],[0.7,'#9a3412'],[1,'#ef4444']],
            text=cm, texttemplate='<b>%{text}</b>',
            textfont=dict(color='white', family='Share Tech Mono', size=14)
        ))
        fig_cm.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#64748b', family='Exo 2'),
            xaxis=dict(title='Predicted', color='#94a3b8'),
            yaxis=dict(title='Actual', color='#94a3b8', autorange='reversed'),
            margin=dict(l=20,r=20,t=30,b=20), height=400
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with tab3:
        st.markdown("<div class='section-title'>Model Summary</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-label'>Test Accuracy</div>
                <div class='metric-value'>{meta['accuracy']*100:.2f}%</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-label'>CV Mean</div>
                <div class='metric-value'>{meta['cv_mean']*100:.2f}%</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-label'>CV Std Dev</div>
                <div class='metric-value'>±{meta['cv_std']*100:.2f}%</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            'Parameter': ['n_estimators','max_depth','min_samples_split','min_samples_leaf',
                          'max_features','class_weight','Train Size','Test Size'],
            'Value': ['200','15','5','2','sqrt','balanced',
                      f"{meta['train_size']:,}", f"{meta['test_size']:,}"]
        }), hide_index=True, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
#  PAGE 4 — DATA EXPLORER
# ═══════════════════════════════════════════════════════════════
elif "Explorer" in page:
    st.markdown("""
    <div style='padding:10px 0 24px 0;'>
        <div style='font-family:Rajdhani,sans-serif;font-size:38px;font-weight:800;color:#f1f5f9;'>
            🗂️ Data <span style='color:#f97316;'>Explorer</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Charts", "Raw Data"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("<div class='section-title'>Day of Week</div>", unsafe_allow_html=True)
            day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
            day_df = df['Day_of_Week'].value_counts().reindex(day_order)
            fig_d = px.bar(x=day_df.index, y=day_df.values,
                           color=day_df.values,
                           color_continuous_scale=['#0d1b2a','#f97316'],
                           labels={'x':'Day','y':'Count'})
            fig_d.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#64748b', family='Exo 2'), showlegend=False,
                coloraxis_showscale=False,
                xaxis=dict(showgrid=False, color='#334155'),
                yaxis=dict(showgrid=True, gridcolor='#0f1f2e', color='#334155'),
                margin=dict(l=0,r=0,t=10,b=0), height=280
            )
            st.plotly_chart(fig_d, use_container_width=True)

        with c2:
            st.markdown("<div class='section-title'>Temperature vs Severity</div>", unsafe_allow_html=True)
            fig_b = go.Figure()
            for sev, clr in zip([1,2,3,4],['#22c55e','#f59e0b','#f97316','#ef4444']):
                sub = df[df['Severity']==sev]['Temperature_F']
                fig_b.add_trace(go.Box(y=sub, name=f'Sev {sev}',
                                       marker_color=clr, line_color=clr))
            fig_b.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#64748b', family='Exo 2'), showlegend=False,
                yaxis=dict(showgrid=True, gridcolor='#0f1f2e', color='#334155'),
                xaxis=dict(color='#334155'),
                margin=dict(l=0,r=0,t=10,b=0), height=280
            )
            st.plotly_chart(fig_b, use_container_width=True)

        st.markdown("<div class='section-title'>Monthly Trend by Severity</div>", unsafe_allow_html=True)
        mdf = df.groupby(['Month','Severity']).size().reset_index(name='Count')
        mdf['Month_Name'] = mdf['Month'].map({
            1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
            7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'
        })
        fig_m = px.line(mdf, x='Month_Name', y='Count', color='Severity',
                        color_discrete_map={1:'#22c55e',2:'#f59e0b',3:'#f97316',4:'#ef4444'})
        fig_m.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#64748b', family='Exo 2'),
            xaxis=dict(showgrid=False, color='#334155'),
            yaxis=dict(showgrid=True, gridcolor='#0f1f2e', color='#334155'),
            margin=dict(l=0,r=0,t=10,b=20), height=300,
            legend=dict(font=dict(color='#94a3b8'))
        )
        st.plotly_chart(fig_m, use_container_width=True)

    with tab2:
        st.markdown("<div class='section-title'>Browse Dataset</div>", unsafe_allow_html=True)
        sev_filter = st.multiselect("Filter by Severity", [1,2,3,4], default=[1,2,3,4])
        display_df = df[df['Severity'].isin(sev_filter)]
        st.dataframe(display_df.head(500), use_container_width=True, height=420)
        st.caption(f"Showing {min(500, len(display_df)):,} of {len(display_df):,} records")