import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import json
import time
import seaborn as sns
import matplotlib.pyplot as plt
import google.generativeai as genai
from sklearn.preprocessing import LabelEncoder

def clean_data(df, fill_method="Forward Fill", handle_outliers=False, drop_cols=[]):
    cleaned = df.copy()
    if drop_cols:
        cleaned = cleaned.drop(columns=drop_cols, errors='ignore')
    original_len = len(cleaned)
    cleaned = cleaned.drop_duplicates()
    dups_removed = original_len - len(cleaned)
    
    if fill_method == "Forward Fill":
        cleaned = cleaned.ffill().bfill()
    elif fill_method == "Mean/Mode":
        for col in cleaned.columns:
            if cleaned[col].dtype == 'object':
                if not cleaned[col].mode().empty:
                    cleaned[col] = cleaned[col].fillna(cleaned[col].mode()[0])
            else:
                cleaned[col] = cleaned[col].fillna(cleaned[col].mean())
    elif fill_method == "Drop Missing Rows":
        cleaned = cleaned.dropna()
                
    outliers_removed = 0
    if handle_outliers:
        num_cols = cleaned.select_dtypes(include=np.number).columns
        for col in num_cols:
            Q1 = cleaned[col].quantile(0.25)
            Q3 = cleaned[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            old_len = len(cleaned)
            cleaned = cleaned[(cleaned[col] >= lower_bound) & (cleaned[col] <= upper_bound)]
            outliers_removed += (old_len - len(cleaned))
            
    return cleaned, dups_removed, outliers_removed

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
def preprocess_data(df, scaler_type="None"):
    processed = df.copy()
    le = LabelEncoder()
    cat_cols = processed.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        processed[col] = le.fit_transform(processed[col].astype(str))
        
    if scaler_type != "None":
        num_cols = processed.select_dtypes(include=np.number).columns
        target_cols = [c for c in processed.columns if 'sev' in c.lower()]
        cols_to_scale = [c for c in num_cols if c not in target_cols]
        
        if scaler_type == "Standard Scaler":
            scaler = StandardScaler()
        elif scaler_type == "MinMax Scaler":
            scaler = MinMaxScaler()
            
        if cols_to_scale:
            processed[cols_to_scale] = scaler.fit_transform(processed[cols_to_scale])
            
    return processed

def generate_insights(df_summary, api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"Analyze the following dataset summary and provide insights on patterns, anomalies, and data cleaning suggestions. Keep it concise, professional, and well-structured.\n\nDataset Summary:\n{df_summary}"
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating insights: {str(e)}\nPlease check your API key."

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
    transform-style: preserve-3d;
    perspective: 1000px;
    transition: transform 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275), box-shadow 0.4s ease;
}
.metric-card:hover {
    transform: translateY(-8px) rotateX(8deg) rotateY(-5deg);
    box-shadow: -10px 15px 30px rgba(249, 115, 22, 0.25), 10px 15px 30px rgba(239, 68, 68, 0.25);
}

@keyframes glow {
    0% { box-shadow: 0 0 5px #f97316; }
    50% { box-shadow: 0 0 20px #ef4444, 0 0 30px #f97316; }
    100% { box-shadow: 0 0 5px #f97316; }
}
.glow-effect {
    animation: glow 3s infinite alternate;
    border-radius: 12px;
}

@keyframes float {
    0% { transform: translateY(0px); }
    50% { transform: translateY(-10px); }
    100% { transform: translateY(0px); }
}
.float-effect {
    animation: float 4s ease-in-out infinite;
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
        "🧪  Data Cleaning Lab",
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
#  PAGE 1.5 — DATA CLEANING LAB
# ═══════════════════════════════════════════════════════════════
elif "Cleaning Lab" in page:
    st.markdown("""
    <div style='padding:10px 0 24px 0;'>
        <div style='font-family:Rajdhani,sans-serif;font-size:38px;font-weight:800;color:#f1f5f9;'>
            🧪 Data <span style='color:#f97316;'>Cleaning Lab</span>
        </div>
        <div style='font-family:Share Tech Mono,monospace;font-size:11px;color:#475569;letter-spacing:0.1em;'>
            FULL PIPELINE: RAW DATA → CLEANING → PROCESSING → VISUALIZATION
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("📤 Upload Raw Dataset (CSV)", type="csv")
    
    if uploaded_file:
        raw_df = pd.read_csv(uploaded_file, na_values=["?", "NA", "N/A", "null", "Null", "", " ", "-"])
        
        st.markdown("<div class='section-title'>1. Raw Data Preview</div>", unsafe_allow_html=True)
        st.dataframe(raw_df.head(10), use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Rows", raw_df.shape[0])
        col2.metric("Total Columns", raw_df.shape[1])
        missing_count = raw_df.isnull().sum().sum()
        col3.metric("Missing Values", missing_count)
        
        st.markdown("#### Missing Values Heatmap")
        if missing_count == 0:
            st.info("💡 **No missing values found in the dataset!** The heatmap below is uniformly colored because there is no missing data to highlight.")
            
        fig_miss, ax_miss = plt.subplots(figsize=(10, 3))
        sns.heatmap(raw_df.isnull(), cbar=False, cmap='viridis', ax=ax_miss)
        fig_miss.patch.set_alpha(0)
        ax_miss.tick_params(colors='white')
        st.pyplot(fig_miss)
        
        st.markdown("<div class='section-title'>2. Data Cleaning Module & Processing</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            fill_method = st.selectbox("Handle Missing Values", ["Forward Fill", "Mean/Mode", "Drop Missing Rows"])
        with c2:
            scaler_type = st.selectbox("Feature Scaling", ["None", "Standard Scaler", "MinMax Scaler"])
        with c3:
            st.markdown("<br>", unsafe_allow_html=True)
            handle_outliers = st.checkbox("Remove Outliers (IQR)")
            
        drop_cols = st.multiselect("Drop Columns (Optional)", raw_df.columns.tolist())
            
        if st.button("⚡ Execute Cleaning Pipeline"):
            with st.spinner("Cleaning and Processing Data..."):
                cleaned_df, dups_rm, outs_rm = clean_data(raw_df, fill_method, handle_outliers, drop_cols)
                processed_df = preprocess_data(cleaned_df, scaler_type)
                st.session_state['processed_df'] = processed_df
                st.session_state['cleaned_df'] = cleaned_df
                
                st.success(f"Pipeline executed! Removed {dups_rm} duplicates and {outs_rm} outliers.")
        
        if 'processed_df' in st.session_state:
            cleaned_df = st.session_state['cleaned_df']
            processed_df = st.session_state['processed_df']
            
            st.markdown("#### BEFORE vs AFTER Cleaning")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Before (Missing Values)**")
                missing_before = raw_df.isnull().sum()[raw_df.isnull().sum() > 0]
                if not missing_before.empty:
                    st.write(missing_before)
                else:
                    st.write("None")
            with c2:
                st.markdown("**After (Missing Values)**")
                missing_after = cleaned_df.isnull().sum()[cleaned_df.isnull().sum() > 0]
                if not missing_after.empty:
                    st.write(missing_after)
                else:
                    st.write("None")
                    
            st.markdown("<div class='section-title'>3. Processed Data (Ready for ML)</div>", unsafe_allow_html=True)
            st.dataframe(processed_df.head(10), use_container_width=True)
            
            csv_data = processed_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Processed Dataset",
                data=csv_data,
                file_name='processed_dataset.csv',
                mime='text/csv',
            )
            
            st.markdown("<div class='section-title'>4. Data Analysis & Visualization</div>", unsafe_allow_html=True)
            
            viz_c1, viz_c2 = st.columns(2)
            with viz_c1:
                st.markdown("#### Correlation Heatmap")
                fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
                corr = processed_df.select_dtypes(include=np.number).corr()
                sns.heatmap(corr, cmap='coolwarm', ax=ax_corr, annot=False)
                fig_corr.patch.set_alpha(0)
                ax_corr.tick_params(colors='white')
                st.pyplot(fig_corr)
                
            with viz_c2:
                st.markdown("#### Severity Distribution")
                target_col = None
                for col in processed_df.columns:
                    if 'sev' in col.lower():
                        target_col = col
                        break
                if target_col:
                    fig_sev = px.histogram(processed_df, x=target_col, color=target_col, color_discrete_sequence=px.colors.qualitative.Set1)
                    fig_sev.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
                    st.plotly_chart(fig_sev, use_container_width=True)
                else:
                    st.info("No Severity column found for distribution chart.")

            st.markdown("#### 3D Feature Scatter Plot")
            num_cols = processed_df.select_dtypes(include=np.number).columns.tolist()
            if len(num_cols) >= 3:
                sc_c1, sc_c2, sc_c3 = st.columns(3)
                x_col = sc_c1.selectbox("X-axis (3D)", num_cols, index=0)
                y_col = sc_c2.selectbox("Y-axis (3D)", num_cols, index=1)
                z_col = sc_c3.selectbox("Z-axis (3D)", num_cols, index=2)
                
                fig_3d = px.scatter_3d(processed_df.sample(min(1000, len(processed_df))), 
                                      x=x_col, y=y_col, z=z_col, color=target_col if target_col else None,
                                      color_continuous_scale=px.colors.sequential.Plasma)
                fig_3d.update_layout(scene=dict(bgcolor='rgba(0,0,0,0)'),
                                     paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'),
                                     margin=dict(l=0, r=0, b=0, t=0))
                st.plotly_chart(fig_3d, use_container_width=True)

            st.markdown("#### Trend Analysis (Time vs Accidents)")
            time_col = None
            for col in processed_df.columns:
                if 'time' in col.lower() or 'hour' in col.lower() or 'date' in col.lower() or 'month' in col.lower():
                    time_col = col
                    break
            if time_col:
                trend = processed_df[time_col].value_counts().sort_index().reset_index()
                trend.columns = [time_col, 'Count']
                fig_trend = px.line(trend, x=time_col, y='Count', markers=True)
                fig_trend.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
                st.plotly_chart(fig_trend, use_container_width=True)
            else:
                st.info("No time-related column found for trend analysis.")
            
            st.markdown("<div class='section-title'>5. AI Insights (Gemini)</div>", unsafe_allow_html=True)
            gemini_key = st.text_input("🔑 Enter Gemini API Key to Generate Insights", type="password")
            if st.button("🧠 Generate AI Insights"):
                if gemini_key:
                    with st.spinner("Analyzing dataset with Gemini AI..."):
                        summary = processed_df.describe().to_string()
                        insights = generate_insights(summary, gemini_key)
                        st.markdown(f'''
                        <div style="background:#0d1b2a;border:1px solid #1e3a5f;border-radius:8px;padding:20px;margin-top:10px;">
                            <div style="color:#f1f5f9;font-family:Exo 2;white-space:pre-wrap;">{insights}</div>
                        </div>
                        ''', unsafe_allow_html=True)
                else:
                    st.warning("Please enter your Gemini API key.")

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
        with c1: temperature   = st.slider("Temperature (°C)", 15, 50, 28)
        with c2: wind_speed    = st.slider("Wind Speed (km/h)", 0, 100, 15)
        with c3: visibility    = st.slider("Visibility (km)", 0.1, 20.0, 10.0)

        c1, c2, c3 = st.columns(3)
        with c1: precipitation = st.slider("Precipitation (mm)", 0.0, 150.0, 0.0)
        with c2: humidity      = st.slider("Humidity (%)", 10, 100, 65)
        with c3: pressure      = st.slider("Pressure (hPa)", 900.0, 1100.0, 1010.0)

        weather_cond = st.selectbox("Weather Condition", sorted(meta['weather_classes']))

        st.markdown("#### 🛣 Road Information")
        c1, c2 = st.columns(2)
        with c1:
            road_type   = st.selectbox("Road Type", sorted(meta['road_classes']))
            speed_limit = st.selectbox("Speed Limit (km/h)", [30,40,50,60,80,100,120])
        with c2:
            state       = st.selectbox("Location (Karnataka)", sorted(meta['state_classes']))
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
                min(wind_speed/100, 1),
                1 - min(visibility/20, 1),
                min(precipitation/100, 1),
                float(bad_w),
                speed_limit/120,
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
                sub = df[df['Severity']==sev]['Temperature_C']
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