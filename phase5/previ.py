# =============================================================================
# odisha_dashboard.py — Odisha Renewable Energy Decision Support System
# Run: streamlit run odisha_dashboard.py
# Install: pip install streamlit plotly pandas numpy scikit-learn
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Odisha Energy Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

* { font-family: 'Space Grotesk', sans-serif; }

/* Dark theme base */
.stApp { background: #080c14; color: #e8eaf0; }
.main .block-container { padding: 1.5rem 2rem; max-width: 100%; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0d1321 !important;
    border-right: 1px solid #1e2d45;
}
section[data-testid="stSidebar"] * { color: #c8d0e0 !important; }

/* KPI cards */
.kpi-card {
    background: linear-gradient(135deg, #0d1828 0%, #111f35 100%);
    border: 1px solid #1e3050;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s, box-shadow 0.2s;
}
.kpi-card:hover { transform: translateY(-2px); box-shadow: 0 8px 24px rgba(0,180,255,0.12); }
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
}
.kpi-solar::before   { background: linear-gradient(90deg, #f59e0b, #fcd34d); }
.kpi-wind::before    { background: linear-gradient(90deg, #3b82f6, #93c5fd); }
.kpi-hybrid::before  { background: linear-gradient(90deg, #06b6d4, #67e8f9); }
.kpi-accuracy::before{ background: linear-gradient(90deg, #10b981, #6ee7b7); }

.kpi-value {
    font-size: 2.4rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1;
    margin-bottom: 0.3rem;
}
.kpi-label { font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em; opacity: 0.7; }
.kpi-sub   { font-size: 0.85rem; opacity: 0.55; margin-top: 0.2rem; }

/* Section headers */
.section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 1.8rem 0 0.8rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1e2d45;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background: #0d1321;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
    border: 1px solid #1e2d45;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 20px;
    font-size: 0.85rem;
    font-weight: 500;
    color: #64748b !important;
}
.stTabs [aria-selected="true"] {
    background: #162035 !important;
    color: #e2e8f0 !important;
}

/* Plotly chart containers */
.js-plotly-plot { border-radius: 12px; }

/* Insight box */
.insight-box {
    background: linear-gradient(135deg, #0a1f15, #0d2918);
    border: 1px solid #1a4d2e;
    border-left: 4px solid #10b981;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
    font-size: 0.9rem;
    color: #a7f3d0;
    line-height: 1.6;
}
.warn-box {
    background: linear-gradient(135deg, #1a1000, #1f1500);
    border: 1px solid #4d3800;
    border-left: 4px solid #f59e0b;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
    font-size: 0.9rem;
    color: #fde68a;
    line-height: 1.6;
}
.info-box {
    background: linear-gradient(135deg, #0a1525, #0d1e35);
    border: 1px solid #1e3a5f;
    border-left: 4px solid #3b82f6;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
    font-size: 0.9rem;
    color: #bfdbfe;
    line-height: 1.6;
}

/* Slider labels */
.stSlider { padding: 0.3rem 0; }
div[data-testid="stSlider"] > label { color: #94a3b8 !important; font-size: 0.85rem !important; }

/* Hide default streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
.viewerBadge_container__1QSob { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ── COLOUR PALETTES ───────────────────────────────────────────────────────────
ENERGY_COLORS = {
    'SOLAR':   '#f59e0b',
    'WIND':    '#3b82f6',
    'BIOMASS': '#10b981',
    'HYBRID':  '#06b6d4',
    'Unknown': '#475569'
}
PLOTLY_TEMPLATE = "plotly_dark"

# ── DATA LOADING ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    import os
    # Try same directory first, then common paths
    for path in ['dashboard_data.csv', 'D:/phase5/dashboard_data.csv',
                 'C:/Users/KIIT0001/OneDrive/Desktop/stuff/miniproject/phase5/dashboard_data.csv']:
        if os.path.exists(path):
            df = pd.read_csv(path)
            break
    else:
        st.error("❌ Could not find dashboard_data.csv — please place it in the same folder as this script.")
        st.stop()

    for path in ['district_summary.csv', 'D:/phase5/district_summary.csv',
                 'C:/Users/KIIT0001/OneDrive/Desktop/stuff/miniproject/phase5/district_summary.csv']:
        if os.path.exists(path):
            dist = pd.read_csv(path)
            break
    else:
        # Auto-generate from block data
        dist = df.groupby('district').agg(
            solar_mean=('solar_mean','mean'), wind_mean=('wind_mean','mean'),
            biomass_mean=('pop_mean','mean'), solar_norm=('solar_norm','mean'),
            wind_norm=('wind_norm','mean'), biomass_norm=('biomass_norm','mean'),
            confidence_mean=('confidence','mean'), block_count=('block_name','count'),
            dominant_class=('final_pred', lambda x: x.value_counts().index[0])
        ).reset_index()

    return df, dist

df, dist_df = load_data()

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ Odisha Energy Intelligence")
    st.markdown("*Block-Level Renewable Energy Decision Support System*")
    st.markdown("---")

    st.markdown("### 🎛️ What-If Scenario Weights")
    st.markdown('<div style="font-size:0.75rem;color:#64748b;margin-bottom:0.8rem">Drag sliders to recompute energy suitability scores in real-time</div>', unsafe_allow_html=True)

    w_solar   = st.slider("☀️ Solar Weight",   0.0, 1.0, 0.32, 0.01, format="%.2f")
    w_wind    = st.slider("💨 Wind Weight",    0.0, 1.0, 0.28, 0.01, format="%.2f")
    w_biomass = st.slider("🌿 Biomass Weight", 0.0, 1.0, 0.15, 0.01, format="%.2f")

    total_w = w_solar + w_wind + w_biomass
    if total_w > 0:
        w_solar_n   = w_solar   / total_w
        w_wind_n    = w_wind    / total_w
        w_biomass_n = w_biomass / total_w
    else:
        w_solar_n = w_wind_n = w_biomass_n = 1/3

    st.markdown(f"""
    <div style="background:#0d1828;border:1px solid #1e3050;border-radius:8px;padding:0.8rem;margin-top:0.5rem;font-family:'JetBrains Mono',monospace;font-size:0.78rem;color:#94a3b8">
    Normalized weights:<br>
    ☀️ {w_solar_n:.1%} &nbsp; 💨 {w_wind_n:.1%} &nbsp; 🌿 {w_biomass_n:.1%}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🔍 Filter Blocks")
    selected_district = st.selectbox("District", ["All Districts"] + sorted(df['district'].dropna().unique().tolist()))
    selected_class    = st.multiselect("Energy Class", ['SOLAR','WIND','BIOMASS','HYBRID'],
                                       default=['SOLAR','WIND','BIOMASS','HYBRID'])
    conf_threshold = st.slider("Min Confidence", 0.0, 1.0, 0.0, 0.05, format="%.0%")

    st.markdown("---")
    st.markdown('<div style="font-size:0.7rem;color:#475569;text-align:center">Phase 5 ML Pipeline<br>314 Blocks · 30 Districts · Odisha<br>RF Accuracy: 97.8%</div>', unsafe_allow_html=True)

# ── COMPUTE WHAT-IF SCORES ────────────────────────────────────────────────────
df['whatif_score'] = (
    df['solar_norm']   * w_solar_n +
    df['wind_norm']    * w_wind_n  +
    df['biomass_norm'] * w_biomass_n
)

def whatif_label(row):
    scores = {
        'SOLAR':   row['solar_norm']   * w_solar_n,
        'WIND':    row['wind_norm']    * w_wind_n,
        'BIOMASS': row['biomass_norm'] * w_biomass_n,
    }
    best = max(scores, key=scores.get)
    s = sorted(scores.values(), reverse=True)
    return 'HYBRID' if (s[0] - s[1]) < 0.08 else best

df['whatif_label'] = df.apply(whatif_label, axis=1)

# ── FILTER DATA ───────────────────────────────────────────────────────────────
dff = df.copy()
if selected_district != "All Districts":
    dff = dff[dff['district'] == selected_district]
dff = dff[dff['final_pred'].isin(selected_class)]
dff = dff[dff['confidence'] >= conf_threshold]

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex;align-items:center;gap:1rem;margin-bottom:1.5rem">
  <div>
    <h1 style="margin:0;font-size:1.8rem;font-weight:700;background:linear-gradient(90deg,#f59e0b,#06b6d4,#10b981);-webkit-background-clip:text;-webkit-text-fill-color:transparent">
      Odisha Renewable Energy Intelligence System
    </h1>
    <p style="margin:0;color:#64748b;font-size:0.9rem">Block-Level ML Classification · Phase 5 + Enhancements · Odisha State, India</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ── KPI ROW ───────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)

solar_count  = (df['final_pred']=='SOLAR').sum()
hybrid_count = (df['final_pred']=='HYBRID').sum()
wind_count   = (df['final_pred']=='WIND').sum()
avg_conf     = df['confidence'].mean()
whatif_dominant = df['whatif_label'].value_counts().index[0]

with c1:
    st.markdown(f"""<div class="kpi-card kpi-solar">
    <div class="kpi-value" style="color:#f59e0b">{solar_count}</div>
    <div class="kpi-label">☀️ Solar Blocks</div>
    <div class="kpi-sub">{solar_count/314*100:.1f}% of Odisha</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="kpi-card kpi-wind">
    <div class="kpi-value" style="color:#3b82f6">{wind_count}</div>
    <div class="kpi-label">💨 Wind Blocks</div>
    <div class="kpi-sub">High wind viable</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="kpi-card kpi-hybrid">
    <div class="kpi-value" style="color:#06b6d4">{hybrid_count}</div>
    <div class="kpi-label">🔀 Hybrid Zones</div>
    <div class="kpi-sub">Multi-source needed</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class="kpi-card kpi-accuracy">
    <div class="kpi-value" style="color:#10b981">{avg_conf:.1%}</div>
    <div class="kpi-label">🎯 Avg Confidence</div>
    <div class="kpi-sub">RF Model certainty</div>
    </div>""", unsafe_allow_html=True)
with c5:
    color = ENERGY_COLORS.get(whatif_dominant, '#94a3b8')
    st.markdown(f"""<div class="kpi-card" style="border-top:3px solid {color}">
    <div class="kpi-value" style="color:{color};font-size:1.4rem">{whatif_dominant}</div>
    <div class="kpi-label">🎛️ What-If Dominant</div>
    <div class="kpi-sub">With your slider weights</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── MAIN TABS ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗺️ Resource Maps",
    "📊 Block Explorer",
    "🏙️ District Analysis",
    "🔀 Scale Comparison",
    "📋 Report & Findings"
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — RESOURCE MAPS
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Resource Intensity Maps — All 314 Odisha Blocks</div>', unsafe_allow_html=True)

    map_mode = st.radio("Select View", ["☀️ Solar Intensity","💨 Wind Intensity","🌿 Biomass Potential","🔀 Energy Classification","🎛️ What-If Scenario","🎯 Confidence Map"],
                        horizontal=True)

    # Sorting selector
    sort_by = st.selectbox("Sort blocks by", ["Solar (High→Low)","Wind (High→Low)","Biomass (High→Low)","Confidence (High→Low)","Alphabetical"], index=0)
    sort_map = {"Solar (High→Low)":('solar_norm',False),"Wind (High→Low)":('wind_norm',False),
                "Biomass (High→Low)":('biomass_norm',False),"Confidence (High→Low)":('confidence',False),
                "Alphabetical":('block_name',True)}
    sc, sa = sort_map[sort_by]
    df_plot = dff.sort_values(sc, ascending=sa).reset_index(drop=True)

    if "Solar" in map_mode:
        fig = px.bar(df_plot, x='block_name', y='solar_mean',
                     color='solar_mean', color_continuous_scale=[[0,'#1a1a2e'],[0.3,'#92400e'],[0.6,'#d97706'],[1.0,'#fcd34d']],
                     hover_data={'block_name':True,'solar_mean':':.3f','district':True,'final_pred':True},
                     labels={'solar_mean':'GHI (kWh/m²/day)','block_name':'Block'},
                     template=PLOTLY_TEMPLATE)
        fig.update_layout(title="☀️ Solar GHI Intensity — All Odisha Blocks",
                          coloraxis_colorbar=dict(title="GHI",tickfont=dict(color='#94a3b8')))
        fig.update_traces(marker_line_width=0)
        fig.add_hline(y=df['solar_mean'].mean(), line_dash="dash", line_color="#f59e0b", opacity=0.5,
                      annotation_text=f"State mean: {df['solar_mean'].mean():.3f}", annotation_position="top right")
        st.markdown(f"""<div class="insight-box">
        ☀️ <b>Solar spans {df['solar_mean'].min():.3f} – {df['solar_mean'].max():.3f} kWh/m²/day</b> across Odisha.
        The narrow range explains why 250/314 blocks (79.6%) are classified SOLAR — the resource is consistently high state-wide.
        Blocks with GHI below {df['solar_mean'].quantile(0.25):.3f} are candidates for multi-resource planning.
        </div>""", unsafe_allow_html=True)

    elif "Wind" in map_mode:
        fig = px.bar(df_plot, x='block_name', y='wind_mean',
                     color='wind_mean', color_continuous_scale=[[0,'#0a1628'],[0.4,'#1d4ed8'],[0.7,'#60a5fa'],[1.0,'#e0f2fe']],
                     hover_data={'block_name':True,'wind_mean':':.3f','district':True,'final_pred':True},
                     labels={'wind_mean':'Wind Speed (m/s)','block_name':'Block'},
                     template=PLOTLY_TEMPLATE)
        fig.update_layout(title="💨 Wind Speed Intensity — All Odisha Blocks",
                          coloraxis_colorbar=dict(title="m/s"))
        fig.update_traces(marker_line_width=0)
        # Mark wind-dominant blocks
        wind_blocks = df_plot[df_plot['final_pred']=='WIND']
        for _, wb in wind_blocks.iterrows():
            fig.add_vline(x=wb['block_name'], line_color='#00ff88', line_width=2,
                         annotation_text=f"⬆️ {wb['block_name']}", annotation_position="top")
        st.markdown(f"""<div class="info-box">
        💨 <b>Wind varies dramatically: {df['wind_mean'].min():.2f} – {df['wind_mean'].max():.2f} m/s</b> — much wider than solar.
        The 2 WIND-dominant blocks (highlighted in green) achieve >5.0 m/s average — coastal/exposed terrain.
        62 HYBRID blocks show elevated wind competing with solar simultaneously.
        </div>""", unsafe_allow_html=True)

    elif "Biomass" in map_mode:
        fig = px.bar(df_plot, x='block_name', y='pop_mean',
                     color='pop_mean', color_continuous_scale=[[0,'#0d1b0d'],[0.3,'#166534'],[0.6,'#22c55e'],[1.0,'#dcfce7']],
                     hover_data={'block_name':True,'pop_mean':':.1f','district':True,'final_pred':True},
                     labels={'pop_mean':'Population Density (proxy)','block_name':'Block'},
                     template=PLOTLY_TEMPLATE)
        fig.update_layout(title="🌿 Biomass Proxy (Population Density) — All Odisha Blocks",
                          coloraxis_colorbar=dict(title="Pop/km²"))
        fig.update_traces(marker_line_width=0)
        st.markdown(f"""<div class="warn-box">
        🌿 <b>Biomass proxy (population density) spans {df['pop_mean'].min():.0f}–{df['pop_mean'].max():.0f} persons/km²</b>.
        High population implies agricultural activity and crop residue availability.
        Despite wide variation, biomass never dominates because its normalized score still loses to Odisha's consistently high solar GHI.
        No block achieves BIOMASS classification — a valid, policy-relevant finding.
        </div>""", unsafe_allow_html=True)

    elif "Classification" in map_mode:
        color_seq = [ENERGY_COLORS.get(l,'#64748b') for l in df_plot['final_pred'].unique()]
        fig = px.bar(df_plot, x='block_name', y='whatif_score',
                     color='final_pred', color_discrete_map=ENERGY_COLORS,
                     hover_data={'block_name':True,'final_pred':True,'confidence':':.1%','district':True,
                                 'solar_mean':':.3f','wind_mean':':.3f','pop_mean':':.1f'},
                     labels={'final_pred':'Energy Class','block_name':'Block','whatif_score':'Composite Score'},
                     template=PLOTLY_TEMPLATE)
        fig.update_layout(title="🔀 Final Energy Classification — All 314 Blocks",
                          legend=dict(orientation='h', yanchor='bottom', y=1.02))
        fig.update_traces(marker_line_width=0)
        st.markdown(f"""<div class="insight-box">
        🔬 <b>Key finding for your research:</b> While 79.6% of blocks are SOLAR, the 62 HYBRID blocks (cyan) reveal
        genuine zones where multi-resource planning is required. The 2 WIND blocks are only detectable at this
        block-level resolution — both are masked at district scale (MAUP finding).
        </div>""", unsafe_allow_html=True)

    elif "What-If" in map_mode:
        fig = px.bar(df_plot, x='block_name', y='whatif_score',
                     color='whatif_label', color_discrete_map=ENERGY_COLORS,
                     hover_data={'block_name':True,'whatif_label':True,'whatif_score':':.3f','district':True},
                     labels={'whatif_label':'What-If Class','block_name':'Block','whatif_score':'Weighted Score'},
                     template=PLOTLY_TEMPLATE)
        fig.update_layout(title=f"🎛️ What-If Scenario — Solar {w_solar_n:.0%} / Wind {w_wind_n:.0%} / Biomass {w_biomass_n:.0%}",
                          legend=dict(orientation='h', yanchor='bottom', y=1.02))
        fig.update_traces(marker_line_width=0)
        new_counts = df['whatif_label'].value_counts()
        st.markdown(f"""<div class="info-box">
        🎛️ <b>With your custom weights</b> (Solar {w_solar_n:.0%} / Wind {w_wind_n:.0%} / Biomass {w_biomass_n:.0%}),
        the classification changes to: {' · '.join([f"{k}: {v}" for k,v in new_counts.items()])}.
        Try increasing Wind weight to 60%+ to see the coastal blocks shift classification!
        </div>""", unsafe_allow_html=True)

    elif "Confidence" in map_mode:
        fig = px.bar(df_plot.sort_values('confidence'), x='block_name', y='confidence',
                     color='confidence', color_continuous_scale=[[0,'#7f1d1d'],[0.4,'#dc2626'],[0.6,'#f59e0b'],[0.8,'#22c55e'],[1.0,'#064e3b']],
                     hover_data={'block_name':True,'confidence':':.1%','final_pred':True,'district':True},
                     labels={'confidence':'Model Confidence','block_name':'Block'},
                     template=PLOTLY_TEMPLATE)
        fig.update_layout(title="🎯 Model Confidence Map — Where is the AI most/least certain?",
                          coloraxis_colorbar=dict(title="Confidence",tickformat=".0%"))
        fig.update_traces(marker_line_width=0)
        fig.add_hline(y=0.80, line_dash="dash", line_color="#22c55e", opacity=0.6,
                      annotation_text="80% high confidence", annotation_position="top right")
        fig.add_hline(y=0.60, line_dash="dash", line_color="#f59e0b", opacity=0.6,
                      annotation_text="60% HYBRID threshold", annotation_position="top right")
        hi = (df['confidence']>=0.80).sum()
        lo = (df['confidence']<0.60).sum()
        st.markdown(f"""<div class="insight-box">
        🎯 <b>Confidence breakdown:</b> {hi} blocks (>{hi/314:.0%}) predicted with high confidence (>80%) — the model is very certain.
        Only {lo} blocks fell below 60% confidence and were overridden to HYBRID.
        Dark red bars = most ambiguous blocks = prime candidates for field validation.
        </div>""", unsafe_allow_html=True)

    fig.update_layout(
        plot_bgcolor='#080c14', paper_bgcolor='#0d1321',
        font=dict(color='#94a3b8', family='Space Grotesk'),
        xaxis=dict(showticklabels=False, title='', showgrid=False, zeroline=False),
        yaxis=dict(gridcolor='#1e2d45', zeroline=False),
        margin=dict(l=50, r=20, t=60, b=30),
        height=420
    )
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — BLOCK EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Block-Level Deep Dive</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns([1.5, 1])

    with col_a:
        # Solar vs Wind scatter
        fig_scatter = px.scatter(
            dff, x='solar_norm', y='wind_norm',
            color='final_pred', color_discrete_map=ENERGY_COLORS,
            size='biomass_norm', size_max=25,
            hover_data={'block_name':True,'district':True,'solar_mean':':.3f',
                        'wind_mean':':.3f','pop_mean':':.0f','confidence':':.1%','final_pred':True},
            labels={'solar_norm':'Solar Intensity (normalized)','wind_norm':'Wind Intensity (normalized)',
                    'final_pred':'Energy Class'},
            template=PLOTLY_TEMPLATE,
            title="Solar vs Wind — Bubble size = Biomass proxy"
        )
        # Annotate wind blocks
        for _, row in dff[dff['final_pred']=='WIND'].iterrows():
            fig_scatter.add_annotation(x=row['solar_norm'], y=row['wind_norm'],
                text=f"⬆️ {row['block_name']}", showarrow=True,
                arrowhead=2, arrowcolor='#00ff88', font=dict(color='#00ff88', size=10),
                ax=-60, ay=-30)

        fig_scatter.update_layout(
            plot_bgcolor='#080c14', paper_bgcolor='#0d1321',
            font=dict(color='#94a3b8'), height=420,
            margin=dict(l=50,r=20,t=50,b=40)
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_b:
        # Resource distribution by class
        fig_box = go.Figure()
        for resource, col_name, color in [('Solar','solar_norm','#f59e0b'),('Wind','wind_norm','#3b82f6'),('Biomass','biomass_norm','#10b981')]:
            for cls in ['SOLAR','HYBRID','WIND']:
                data = dff[dff['final_pred']==cls][col_name].values
                if len(data) > 0:
                    fig_box.add_trace(go.Box(
                        y=data, name=f"{cls}", legendgroup=resource,
                        legendgrouptitle_text=resource if cls=='SOLAR' else None,
                        marker_color=ENERGY_COLORS[cls], showlegend=(resource=='Solar'),
                        boxpoints='outliers', jitter=0.3, pointpos=-1.8
                    ))
        fig_box.update_layout(
            title="Resource Distribution by Energy Class",
            plot_bgcolor='#080c14', paper_bgcolor='#0d1321',
            font=dict(color='#94a3b8'), height=420,
            boxmode='group', margin=dict(l=40,r=20,t=50,b=40),
            legend=dict(orientation='h')
        )
        st.plotly_chart(fig_box, use_container_width=True)

    # Data table
    st.markdown('<div class="section-header">Block Data Table</div>', unsafe_allow_html=True)
    display_cols = ['block_name','district','final_pred','confidence','solar_mean','wind_mean','pop_mean','constraint_pct']
    st.dataframe(
        dff[display_cols].sort_values('confidence', ascending=False)
        .rename(columns={'block_name':'Block','district':'District','final_pred':'Class',
                         'confidence':'Confidence','solar_mean':'Solar GHI','wind_mean':'Wind m/s',
                         'pop_mean':'Biomass Proxy','constraint_pct':'Constraint %'})
        .style.format({'Confidence':'{:.1%}','Solar GHI':'{:.3f}','Wind m/s':'{:.3f}',
                       'Biomass Proxy':'{:.0f}','Constraint %':'{:.1f}'})
        .background_gradient(subset=['Confidence'], cmap='RdYlGn'),
        use_container_width=True, height=350
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DISTRICT ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">District-Level Summary (30 Districts)</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        # District resource heatmap
        fig_heat = go.Figure(data=go.Heatmap(
            z=dist_df[['solar_norm','wind_norm','biomass_norm']].values,
            x=['☀️ Solar','💨 Wind','🌿 Biomass'],
            y=dist_df['district'],
            colorscale=[[0,'#0d1321'],[0.3,'#1e3a5f'],[0.6,'#2563eb'],[1.0,'#93c5fd']],
            text=dist_df[['solar_norm','wind_norm','biomass_norm']].round(3).values,
            texttemplate="%{text:.2f}",
            textfont={"size":9,"color":"white"},
            hoverongaps=False
        ))
        fig_heat.update_layout(
            title="District Resource Intensity Heatmap",
            plot_bgcolor='#080c14', paper_bgcolor='#0d1321',
            font=dict(color='#94a3b8', size=10),
            height=620, margin=dict(l=120,r=30,t=50,b=40)
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    with col2:
        # Block count per district stacked
        block_summary = df.groupby(['district','final_pred']).size().reset_index(name='count')
        fig_stack = px.bar(block_summary, x='count', y='district', color='final_pred',
                           color_discrete_map=ENERGY_COLORS, orientation='h',
                           labels={'count':'Number of Blocks','district':'District','final_pred':'Class'},
                           template=PLOTLY_TEMPLATE,
                           title="Block Classification Breakdown per District")
        fig_stack.update_layout(
            plot_bgcolor='#080c14', paper_bgcolor='#0d1321',
            font=dict(color='#94a3b8', size=10), height=620,
            margin=dict(l=120,r=30,t=50,b=40),
            legend=dict(orientation='h', yanchor='bottom', y=1.01)
        )
        st.plotly_chart(fig_stack, use_container_width=True)

    # District KPIs
    st.markdown('<div class="section-header">District Metrics</div>', unsafe_allow_html=True)
    top_solar  = dist_df.nlargest(1, 'solar_norm').iloc[0]
    top_wind   = dist_df.nlargest(1, 'wind_norm').iloc[0]
    top_bio    = dist_df.nlargest(1, 'biomass_norm').iloc[0]
    most_hybrid= df.groupby('district').apply(lambda x: (x['final_pred']=='HYBRID').sum()).idxmax()
    hybrid_count_d = df[df['district']==most_hybrid]['final_pred'].eq('HYBRID').sum()

    k1,k2,k3,k4 = st.columns(4)
    for col, icon, title, val, sub, color in [
        (k1,'☀️','Best Solar District', top_solar['district'], f"GHI {top_solar['solar_mean']:.3f}", '#f59e0b'),
        (k2,'💨','Best Wind District',  top_wind['district'],  f"Wind {top_wind['wind_mean']:.2f} m/s", '#3b82f6'),
        (k3,'🌿','Best Biomass District',top_bio['district'],  f"Pop {top_bio['biomass_mean']:.0f}/km²", '#10b981'),
        (k4,'🔀','Most Hybrid Blocks',  most_hybrid,           f"{hybrid_count_d} hybrid blocks", '#06b6d4'),
    ]:
        with col:
            st.markdown(f"""<div class="kpi-card" style="border-top:3px solid {color}">
            <div style="font-size:1.6rem">{icon}</div>
            <div class="kpi-value" style="color:{color};font-size:1.1rem;margin:0.3rem 0">{val}</div>
            <div class="kpi-label">{title}</div>
            <div class="kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — SCALE COMPARISON (MAUP)
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">MAUP Analysis — Block vs District Scale</div>', unsafe_allow_html=True)

    st.markdown("""<div class="info-box">
    📐 <b>Modifiable Areal Unit Problem (MAUP):</b> When spatial data is aggregated to coarser units (e.g. districts),
    local patterns are averaged out. This analysis reveals where block-level detail exposes information
    that is invisible at district scale — a core methodological contribution of this research.
    </div>""", unsafe_allow_html=True)

    # Compute block-dominant per district
    block_dom = df.groupby('district')['final_pred'].agg(
        lambda x: x.value_counts().index[0]).reset_index()
    block_dom.columns = ['district','block_dominant']

    # District-level labels (recompute)
    ms2,mw2,mp2 = dist_df['solar_mean'].max(), dist_df['wind_mean'].max(), dist_df.get('biomass_mean', dist_df.get('pop_mean', dist_df['solar_mean'])).max()

    # Merge
    maup_df = block_dom.merge(dist_df[['district','dominant_class']], on='district', how='inner')
    maup_df.columns = ['District','Block Level','District Level']
    maup_df['Agreement'] = maup_df.apply(
        lambda r: '✅ AGREE' if r['Block Level']==r['District Level'] else '❌ DISAGREE', axis=1)

    disagree = maup_df[maup_df['Agreement']=='❌ DISAGREE']

    col1, col2 = st.columns([1.6,1])
    with col1:
        # Agreement visualization
        maup_df['color'] = maup_df['Agreement'].map({'✅ AGREE':'#10b981','❌ DISAGREE':'#ef4444'})
        maup_df['y'] = 1
        fig_maup = px.bar(maup_df, x='District', y='y', color='Agreement',
                          color_discrete_map={'✅ AGREE':'#10b981','❌ DISAGREE':'#ef4444'},
                          text='District Level',
                          template=PLOTLY_TEMPLATE,
                          title=f"Block vs District Agreement — {len(disagree)} Disagreements / 30 Districts")
        fig_maup.update_traces(textposition='inside', textfont=dict(size=9, color='white'))
        fig_maup.update_layout(
            plot_bgcolor='#080c14', paper_bgcolor='#0d1321',
            font=dict(color='#94a3b8'), height=380,
            xaxis=dict(tickangle=45, tickfont=dict(size=9)),
            yaxis=dict(showticklabels=False, title=''),
            showlegend=True, legend=dict(orientation='h'),
            margin=dict(l=20,r=20,t=50,b=100)
        )
        st.plotly_chart(fig_maup, use_container_width=True)

    with col2:
        st.markdown("#### Disagreement Detail")
        if len(disagree) > 0:
            for _, row in disagree.iterrows():
                st.markdown(f"""<div class="warn-box" style="margin:0.4rem 0">
                <b>📍 {row['District']}</b><br>
                Block level → <span style="color:{ENERGY_COLORS.get(row['Block Level'],'#fff')};font-weight:600">{row['Block Level']}</span><br>
                District level → <span style="color:{ENERGY_COLORS.get(row['District Level'],'#fff')};font-weight:600">{row['District Level']}</span><br>
                <span style="font-size:0.8rem;opacity:0.7">District averaging masks block-level dominance</span>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown('<div class="insight-box">✅ Perfect agreement across all 30 districts!</div>', unsafe_allow_html=True)

        st.markdown(f"""<div class="insight-box" style="margin-top:1rem">
        📊 <b>MAUP Summary</b><br>
        Agreement rate: <b>{(len(maup_df)-len(disagree))/len(maup_df):.0%}</b> ({len(maup_df)-len(disagree)}/30 districts)<br>
        Disagreements: <b>{len(disagree)}</b> districts where scale matters<br>
        Direction: District scale inflates HYBRID; block scale reveals SOLAR dominance
        </div>""", unsafe_allow_html=True)

    # Full comparison table
    st.markdown('<div class="section-header">Full Scale Comparison Table</div>', unsafe_allow_html=True)
    display_maup = maup_df[['District','Block Level','District Level','Agreement']].copy()
    st.dataframe(display_maup.style.applymap(
        lambda v: 'color: #10b981' if '✅' in str(v) else ('color: #ef4444' if '❌' in str(v) else ''),
        subset=['Agreement']
    ), use_container_width=True, height=500)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — REPORT & FINDINGS
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">Research Findings & Report Statements</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        # Model accuracy radar
        categories = ['RF Accuracy','XGBoost Acc.','Silhouette','High Confidence','MAUP Agreement']
        values     = [0.978, 0.975, 0.234, 0.965, (len(maup_df)-len(disagree))/len(maup_df)]
        fig_radar = go.Figure(data=go.Scatterpolar(
            r=values, theta=categories, fill='toself',
            fillcolor='rgba(6,182,212,0.2)', line=dict(color='#06b6d4', width=2),
            marker=dict(size=8, color='#06b6d4')
        ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor='#080c14',
                radialaxis=dict(visible=True, range=[0,1], gridcolor='#1e2d45',
                                tickformat='.0%', tickfont=dict(color='#64748b', size=8)),
                angularaxis=dict(gridcolor='#1e2d45', tickfont=dict(color='#94a3b8', size=10))
            ),
            paper_bgcolor='#0d1321', font=dict(color='#94a3b8'),
            title="Model Performance Overview", height=380,
            margin=dict(l=60,r=60,t=50,b=40)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with col2:
        # Feature importance (from Phase 5 results)
        feat_names = ['wind_mean','solar_mean','pop_mean','dist_trans_mean','dist_sub_mean','constraint_pct','dist_roads_mean']
        feat_vals  = [0.3568, 0.2680, 0.1174, 0.0932, 0.0615, 0.0567, 0.0464]
        ahp_vals   = [0.28, 0.32, 0.15, 0.08, 0.04, 0.03, 0.10]
        fig_feat = go.Figure()
        fig_feat.add_trace(go.Bar(name='RF Importance', x=feat_names, y=feat_vals,
                                   marker_color='#06b6d4', opacity=0.85))
        fig_feat.add_trace(go.Bar(name='AHP Expert Weight', x=feat_names, y=ahp_vals,
                                   marker_color='#f59e0b', opacity=0.85))
        fig_feat.update_layout(
            barmode='group', title="RF Feature Importance vs AHP Expert Weights",
            plot_bgcolor='#080c14', paper_bgcolor='#0d1321',
            font=dict(color='#94a3b8'), height=380,
            legend=dict(orientation='h', yanchor='bottom', y=1.01),
            margin=dict(l=50,r=20,t=50,b=60)
        )
        st.plotly_chart(fig_feat, use_container_width=True)

    # Key findings
    st.markdown('<div class="section-header">Key Research Findings (Copy-Ready for Report)</div>', unsafe_allow_html=True)

    findings = [
        ("🎯 Model Performance",
         f"The Random Forest model achieved a 5-fold cross-validation accuracy of 97.8% on 314 Odisha blocks. "
         f"GridSearchCV identified optimal hyperparameters (n_estimators=200, max_depth=5), and XGBoost "
         f"independently validated this result at 97.5% — confirming algorithmic robustness.",
         "insight-box"),
        ("🌬️ Critical AHP-RF Discrepancy",
         f"Feature importance analysis revealed wind_mean as the dominant predictor (RF: 35.7%) despite receiving "
         f"only 28% AHP expert weight — a 7.7 percentage point underestimation. Solar_mean was slightly "
         f"overweighted by experts (AHP: 32% vs RF: 26.8%). This discrepancy suggests the data-driven approach "
         f"captures relationships invisible to domain expertise alone.",
         "warn-box"),
        ("🔀 Hybrid Zone Identification",
         f"62 blocks (19.7%) of Odisha require multi-source energy planning. These HYBRID zones arise where "
         f"wind and solar scores are within 15% of each other — concentrated in coastal/elevated terrain. "
         f"This finding is actionable: these 62 blocks should not be committed to single-source infrastructure.",
         "insight-box"),
        ("📐 MAUP Scale Effects",
         f"4 out of 30 districts (13.3%) showed disagreement between block-level and district-level classification. "
         f"In all 4 cases (Bhadrak, Cuttack, Jajapur, Nabarangapur), district aggregation falsely indicated HYBRID "
         f"while block-level analysis revealed SOLAR dominance. This demonstrates that district-scale planning "
         f"introduces systematic bias toward over-reporting multi-resource complexity.",
         "info-box"),
        ("💨 Wind Viability",
         f"Only 2 blocks achieve viable WIND classification — detectable exclusively at block resolution. "
         f"Both are in coastal/exposed terrain with mean wind speeds exceeding 5.0 m/s. These blocks represent "
         f"Odisha's highest-priority wind energy investment sites and are invisible at district-level analysis.",
         "insight-box"),
    ]

    for title, text, box_class in findings:
        st.markdown(f'<div class="{box_class}"><b>{title}</b><br>{text}</div>', unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown("""<div style="text-align:center;color:#475569;font-size:0.8rem;padding:1rem">
    Solar-Wind-Biomass Decision Support System · Odisha · Phase 5 ML Pipeline · RF + XGBoost + K-Means + AHP
    </div>""", unsafe_allow_html=True)