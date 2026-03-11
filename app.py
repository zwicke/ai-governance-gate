import streamlit as st
import pandas as pd
import plotly.express as px
import json
from engine import InferenceConfig, get_pareto_frontier

# --- UI & THEMING ---
st.set_page_config(page_title="AI Model-Decision-Tradeoff Tool", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .intro-box { 
        background-color: #ffffff; padding: 30px; border-radius: 12px; 
        border: 2px solid #dee2e6; margin-bottom: 35px; box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .executive-report { 
        background-color: #ffffff; padding: 30px; border-radius: 12px; 
        border: 2px solid #e9ecef; color: #1a1c24 !important; margin-bottom: 25px;
    }
    .profile-badge {
        background-color: #00E5FF; color: #000; padding: 6px 14px; 
        border-radius: 20px; font-weight: bold; font-size: 0.9rem;
    }
    .cfo-banner { border-left: 10px solid #dc3545; }
    .tradeoff-banner { border-left: 10px solid #ffc107; }
    .scale-key { font-size: 0.85rem; color: #6c757d; line-height: 1.4; padding: 12px; background: #f1f3f5; border-radius: 8px; }
    .priority-container { background-color: #ffffff; padding: 20px; border-radius: 12px; border: 1px solid #e9ecef; margin-bottom: 30px; }
    </style>
    """, unsafe_allow_html=True)

# --- TOP OF PAGE: STRATEGIC OVERVIEW ---
st.title("🚀 AI Model-Decision-Tradeoff Tool")

with st.container():
    st.markdown("""
    <div class="intro-box">
        <h3>Strategic Purpose</h3>
        <p>This tool is a governance gate designed to move organizations from monolithic "one-size-fits-all" model deployments toward a <b>Diversified AI Strategy</b>. It reveals the quantifiable tradeoffs of current technical decisions against the market's efficiency frontier.</p>
        <hr>
        <h4>Operational Guide:</h4>
        <p><b>Step 1: Define Strategic Priorities</b><br>
        Adjust the sliders below to set your project's internal requirements. This ranks market leaders by how well they match your stance.</p>
        <p><b>Step 2: Map Usage & Identify Efficiency Gaps</b><br>
        Upload your 'Actuals' CSV. This reveals the physical distance between your current baseline (Red) and technical leaders (Blue).</p>
        <p><b>Step 3: Analyze Tradeoffs & Optimization</b><br>
        Review the generated reports to identify the <b>"Inflexibility Tax"</b>—the cost of choosing simplicity over efficiency.</p>
        <p><b>Step 4: Execute Deployment Sign-off</b><br>
        Select a target configuration and document the rationale to create a defensible audit trail.</p>
    </div>
    """, unsafe_allow_html=True)

# --- STEP 1: DEFINE PRIORITIES (NOW IN MAIN COLUMN) ---
st.header("🛡️ 1️⃣ Define Strategic Priorities")
with st.container():
    st.markdown('<div class="priority-container">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        w_q = st.slider("Intelligence Needs", 1, 10, 5, help="1: Formatting | 10: PhD Logic")
    with c2:
        w_c = st.slider("Cost Sensitivity", 1, 10, 5, help="1: Flexible | 10: Strict ROI")
    with c3:
        w_s = st.slider("Sustainability Priority", 1, 10, 5, help="1: Agnostic | 10: Net-Zero Focus")
    
    st.markdown("""<div class="scale-key">
    <b>Logic Check:</b> Intelligence (1-10) vs. Cost/Sustain Sensitivity (1-10). These weights drive the visual size of the market leaders below.
    </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- REACTIVE ENGINE ---
def get_processed_data(w_q, w_c, w_s):
    try:
        with open('prices.json', 'r') as f:
            raw_data = json.load(f)
        configs = [InferenceConfig(**c) for c in raw_data['configurations']]
    except:
        return pd.DataFrame()

    leaders = [c for c in configs if c.model_name in [l.model_name for l in get_pareto_frontier(configs)]]
    data = []
    for l in leaders:
        data.append({"Model": l.model_name, "Intelligence": l.quality, "Cost": l.calculate_normalized_cost(), "Carbon": l.get_carbon_footprint(), "Status": "Market Leader"})
    df_raw = pd.DataFrame(data)

    if df_raw.empty: return df_raw

    def norm(col, reverse=False):
        c_min, c_max = col.min(), col.max()
        if c_max == c_min: return col * 0 + 0.5
        res = (col - c_min) / (c_max - c_min)
        return 1.0 - res if reverse else res

    df_raw['iq_n'] = norm(df_raw['Intelligence'])
    df_raw['cost_n'] = norm(df_raw['Cost'], reverse=True)
    df_raw['sustain_n'] = norm(df_raw['Carbon'], reverse=True)

    # High-Sensitivity Multiplier
    df_raw['Score'] = (df_raw['iq_n'] * (w_q**3.5)) + (df_raw['cost_n'] * (w_c**3.5)) + (df_raw['sustain_n'] * (w_s**3.5))

    s_min, s_max = df_raw['Score'].min(), df_raw['Score'].max()
    s_range = s_max - s_min if s_max != s_min else 1
    df_raw['Match Score'] = ((df_raw['Score'] - s_min) / s_range) * (95 - 15) + 15
    return df_raw

df = get_processed_data(w_q, w_c, w_s)

# --- STEP 2: MAPPING & BENCHMARKING ---
st.markdown('<div style="margin-top: 50px;"></div>', unsafe_allow_html=True)
st.header("📍 2️⃣ Map Usage & Identify Efficiency Gaps")

# Governance Profile Badge (Reactive to Step 1)
priorities = {"Intelligence": w_q, "Cost": w_c, "Sustainability": w_s}
top_p = max(priorities, key=priorities.get)
st.markdown(f"Current Strategic Stance: <span class='profile-badge'>Priority: {top_p}</span>", unsafe_allow_html=True)

st.write("Upload your current model usage (CSV) to identify your position relative to the Frontier.")
uploaded_actuals = st.file_uploader("Upload 'Actuals' CSV", type="csv")

actual_model = None
if uploaded_actuals and uploaded_actuals.size > 0:
    actual_df = pd.read_csv(uploaded_actuals)
    actual_df['Status'] = 'Your Actual Usage'
    actual_df['Match Score'] = 45 
    actual_model = actual_df.iloc[0]
    df = pd.concat([df, actual_df], ignore_index=True)

chart_key = f"{w_q}_{w_c}_{w_s}"
col_map1, col_map2 = st.columns(2)
with col_map1:
    fig_cost = px.scatter(df, x="Cost", y="Intelligence", size="Match Score", color="Status", 
                          hover_name="Model", title="Financial Efficiency Gap",
                          labels={"Cost": "Cost ($ per 1M Tokens)"}, size_max=95,
                          color_discrete_map={'Market Leader': '#00E5FF', 'Your Actual Usage': '#ff4b4b'})
    st.plotly_chart(fig_cost, use_container_width=True, key=f"c_{chart_key}")

with col_map2:
    fig_carb = px.scatter(df, x="Carbon", y="Intelligence", size="Match Score", color="Status", 
                          hover_name="Model", title="Environmental Debt Gap",
                          labels={"Carbon": "Carbon (gCO2 per 1M Tokens)"}, size_max=95,
                          color_discrete_map={'Market Leader': '#00E5FF', 'Your Actual Usage': '#ff4b4b'})
    st.plotly_chart(fig_carb, use_container_width=True, key=f"s_{chart_key}")

# --- STEP 3: AUDIT ---
st.markdown('<div style="margin-top: 50px;"></div>', unsafe_allow_html=True)
st.header("💡 3️⃣ Analyze Tradeoffs & Optimization")

if actual_model is not None:
    best = df[df['Status'] == 'Market Leader'].sort_values('Score', ascending=False).iloc[0]
    savings = ((actual_model['Cost'] - best['Cost']) / actual_model['Cost']) * 100
    
    st.markdown(f"""
    <div class="executive-report cfo-banner">
        <h3>🚨 Audit Discovery: Resource Allocation</h3>
        <p><b>Observation:</b> Your current deployment sits significantly outside the Efficiency Frontier. 
        Relative to the optimized leader for your <b>{top_p}</b> priority—<b>{best['Model']}</b>—the organization 
        is choosing an implicit <b>{savings:.0f}% capital premium.</b></p>
    </div>
    """, unsafe_allow_html=True)

st.markdown(f"""
<div class="executive-report tradeoff-banner">
    <h3>🚲 Tradeoff Analysis: The Cost of Uniformity</h3>
    <p><b>Observation:</b> Choosing a monolithic model for all tasks is a decision to trade efficiency for simplicity.</p>
    <p><b>The Tradeoff:</b> Notice the <b>Llama 3.1 8B</b> dot on the far left. Its distance from the heavyweight models represents 
    the "Inflexibility Tax" of your current architecture. By choosing not to diversify, the organization accepts a higher carbon and 
    capital baseline for tasks that do not require Tier 5 reasoning.</p>
</div>
""", unsafe_allow_html=True)

# --- STEP 4: LEDGER ---
st.markdown('<div style="margin-top: 50px;"></div>', unsafe_allow_html=True)
st.header("📝 4️⃣ Execute Deployment Sign-off")
l1, l2 = st.columns(2)
with l1:
    selected = st.selectbox("Assign Target Configuration", df[df['Status'] == 'Market Leader']['Model'].unique())
    owner = st.text_input("Accountable Decision Maker")
    final_rationale = st.text_area("Audit Rationale", placeholder="Why is this tradeoff right for the organization?")
with l2:
    st.write("### 🧾 Deployment Receipt")
    final_row = df[df['Model'] == selected].iloc[0]
    st.table(pd.DataFrame({
        "Metric": ["Intelligence Tier", "Market Cost ($/1M)", "Carbon Impact (gCO2)"],
        "Target": [f"Tier {final_row['Intelligence']}", f"${final_row['Cost']:.2f}", f"{final_row['Carbon']:.1f}g"]
    }))
    st.download_button("💾 Export Audit Trail", data=f"OWNER: {owner}\nMODEL: {selected}\nRATIONALE: {final_rationale}", file_name="audit.txt")