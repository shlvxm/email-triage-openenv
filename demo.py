import random
import time
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import streamlit as st

# ── Import OpenEnv Modules ───────────────────────────────────────────────────
from models import EmailAction, EmailObservation
from env import EmailTriageEnv, ACTION_NAMES, ACTION_EMOJIS, _REWARD_MATRIX, _LEVEL_CONFIG
from inference import random_agent, rule_based_agent, llm_agent_sim

# ── Streamlit Page Config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Email Triage RL · OpenEnv",
    page_icon="📬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS (same as monolithic version) ───────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Space+Grotesk:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    background-color: #0f1117;
    color: #e8eaf0;
    font-family: 'Space Grotesk', sans-serif;
}
.hero-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #3ecf8e 0%, #56cfe1 50%, #7c83fd 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.2rem;
}
.hero-sub { color: #6b7280; font-size: 0.95rem; margin-bottom: 1.5rem; font-family: 'JetBrains Mono', monospace; }
.metric-card { background: #1a1d27; border: 1px solid #2a2d3d; border-radius: 10px; padding: 1.1rem 1.3rem; text-align: center; }
.metric-val { font-size: 2rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; color: #3ecf8e; }
.metric-lbl { font-size: 0.75rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 0.2rem; }
.tag { display: inline-block; padding: 0.18rem 0.55rem; border-radius: 5px; font-size: 0.75rem; font-weight: 600; font-family: 'JetBrains Mono', monospace; margin-right: 0.3rem; }
.tag-easy   { background:#0d3d2e; color:#3ecf8e; border:1px solid #1a6b4e; }
.tag-medium { background:#3d2d0d; color:#f0a500; border:1px solid #6b4a1a; }
.tag-hard   { background:#3d1515; color:#e05252; border:1px solid #6b2a2a; }
.section-hdr { font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; letter-spacing: 0.15em; text-transform: uppercase; color: #6b7280; border-bottom: 1px solid #2a2d3d; padding-bottom: 0.4rem; margin-bottom: 1rem; }
.obs-bar-wrap { margin-bottom: 0.35rem; }
.obs-bar-label { font-size: 0.72rem; color: #9ca3af; font-family: 'JetBrains Mono', monospace; display: flex; justify-content: space-between; margin-bottom: 0.1rem; }
.obs-bar-track { background: #2a2d3d; border-radius: 3px; height: 7px; overflow: hidden; }
.obs-bar-fill { height: 100%; border-radius: 3px; transition: width 0.4s ease; }
.decision-box { background: #1a1d27; border: 1px solid #2a2d3d; border-radius: 10px; padding: 1rem; margin-top: 0.8rem; }
.decision-correct { border-color: #3ecf8e; }
.decision-wrong   { border-color: #e05252; }
.stButton > button { background: linear-gradient(135deg, #3ecf8e, #56cfe1); color: #0f1117; font-weight: 700; border: none; border-radius: 8px; padding: 0.55rem 1.4rem; font-family: 'JetBrains Mono', monospace; }
</style>
""", unsafe_allow_html=True)

# ── Palette & Styles ──────────────────────────────────────────────────────────
PALETTE = {
    "Random":     "#e05252",
    "Rule-Based": "#f0a500",
    "LLM-Agent":  "#3ecf8e",
    "bg":         "#0f1117",
    "card":       "#1a1d27",
    "border":     "#2a2d3d",
    "text":       "#e8eaf0",
}

def _fig_style(fig, ax):
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["card"])
    ax.tick_params(colors=PALETTE["text"], labelsize=9)
    for spine in ax.spines.values(): spine.set_edgecolor(PALETTE["border"])
    ax.xaxis.label.set_color(PALETTE["text"])
    ax.yaxis.label.set_color(PALETTE["text"])
    ax.title.set_color(PALETTE["text"])
    return fig, ax

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Controls")
    task_level = st.selectbox("Task Level", ["easy", "medium", "hard"], index=1)
    agent_type = st.selectbox("Agent", ["Random", "Rule-Based", "LLM-Agent"], index=2)
    sim_speed = st.slider("Simulation speed (ms/step)", 30, 300, 80, 10)
    st.markdown("---")
    st.markdown("### 📐 OpenEnv Spec")
    st.markdown("""
- **Interface**: `reset()`, `step()`, `state()`
- **Models**: Pydantic-based
- **Features**: 10-dim Box
- **Reward**: Dense shaped floor
    """)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">📬 Email Triage OpenEnv</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Standardised RL benchmarking for Indian inbox triage. Built on OpenEnv spec.</div>', unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🏃 Live Simulator", "📊 Baselines", "🔬 Reward Matrix"])

# ── TAB 1: SIMULATOR ──────────────────────────────────────────────────────────
with tab1:
    col_obs, col_play = st.columns([1, 1.6], gap="medium")
    with col_obs:
        st.markdown('<div class="section-hdr">Current Observation</div>', unsafe_allow_html=True)
        obs_placeholder = st.empty()
    with col_play:
        st.markdown('<div class="section-hdr">Episode Playback</div>', unsafe_allow_html=True)
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1: run_btn = st.button("▶  Run Episode", use_container_width=True)
        metrics_row = st.empty()
        curve_ph = st.empty()

    st.markdown("---")
    conf_col, dist_col = st.columns(2, gap="medium")
    with conf_col:
        st.markdown('<div class="section-hdr">Confusion Matrix</div>', unsafe_allow_html=True)
        conf_ph = st.empty()
    with dist_col:
        st.markdown('<div class="section-hdr">Action Distribution</div>', unsafe_allow_html=True)
        dist_ph = st.empty()

    _FEAT_COLORS = ["#e05252","#3ecf8e","#56cfe1","#f0a500","#7c83fd","#ff6b9d","#f9e24a","#b9ff72","#ff9d6e","#c084fc"]
    _FEAT_NAMES = ["urgency","keyword","sender","spam","promo","social","reply","attach","time","thread"]

    def render_obs(observation: EmailObservation, true_label: int, action: int, reward: float):
        bars_html = ""
        for i, (name, val) in enumerate(zip(_FEAT_NAMES, observation.features)):
            pct = int(val * 100)
            color = _FEAT_COLORS[i]
            bars_html += f"""
            <div class="obs-bar-wrap">
              <div class="obs-bar-label"><span>{name}</span><span style="color:{color}">{val:.3f}</span></div>
              <div class="obs-bar-track"><div class="obs-bar-fill" style="width:{pct}%;background:{color}"></div></div>
            </div>"""
        
        correct = action == true_label
        box_class = "decision-correct" if correct else "decision-wrong"
        icon = "✅" if correct else "❌"
        true_name = f"{ACTION_EMOJIS[EmailAction(true_label)]} {ACTION_NAMES[EmailAction(true_label)]}"
        pred_name = f"{ACTION_EMOJIS[EmailAction(action)]} {ACTION_NAMES[EmailAction(action)]}"
        rwd_color = "#3ecf8e" if reward > 0.6 else "#f0a500" if reward > 0.3 else "#e05252"
        
        html = f"""
        {bars_html}
        <div class="decision-box {box_class}">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.4rem">
            <span style="font-size:0.72rem;color:#6b7280;font-family:monospace">TRIAGE DECISION {icon}</span>
            <span style="font-family:monospace;font-size:0.9rem;font-weight:700;color:{rwd_color}">R={reward:.2f}</span>
          </div>
          <div style="display:flex;gap:1rem">
            <div><div style="font-size:0.65rem;color:#6b7280;font-family:monospace">TRUE</div><div style="font-size:0.88rem;font-weight:600">{true_name}</div></div>
            <div><div style="font-size:0.65rem;color:#6b7280;font-family:monospace">PREDICTED</div><div style="font-size:0.88rem;font-weight:600">{pred_name}</div></div>
          </div>
          <div style="margin-top:0.6rem;font-size:0.75rem;color:#9ca3af;font-style:italic">"{observation.summary}"</div>
        </div>"""
        obs_placeholder.markdown(html, unsafe_allow_html=True)

    if run_btn:
        env = EmailTriageEnv(task_level=task_level, max_steps=50)
        obs = env.reset(seed=random.randint(0, 9999))
        ep_rewards, ep_labels, ep_actions = [], [], []
        conf = np.zeros((6, 6))
        
        for step in range(50):
            if agent_type == "Random": action = random_agent(obs)
            elif agent_type == "Rule-Based": action = rule_based_agent(obs)
            else: action = llm_agent_sim(obs)
            
            res = env.step(action)
            ep_rewards.append(res.reward)
            ep_labels.append(res.info["true_label"])
            ep_actions.append(res.info["predicted_action"])
            conf[res.info["true_label"], res.info["predicted_action"]] += 1
            
            # Update UI
            render_obs(obs, res.info["true_label"], res.info["predicted_action"], res.reward)
            
            avg_r = np.mean(ep_rewards)
            acc = np.mean([1 if ep_labels[i] == ep_actions[i] else 0 for i in range(len(ep_labels))])
            metrics_row.markdown(f"""
            <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:0.6rem;margin-bottom:0.8rem">
              <div class="metric-card"><div class="metric-val">{len(ep_rewards)}</div><div class="metric-lbl">Steps</div></div>
              <div class="metric-card"><div class="metric-val" style="color:#f0a500">{avg_r:.2f}</div><div class="metric-lbl">Avg Reward</div></div>
              <div class="metric-card"><div class="metric-val" style="color:#56cfe1">{acc:.0%}</div><div class="metric-lbl">Accuracy</div></div>
            </div>""", unsafe_allow_html=True)
            
            obs = res.observation
            if res.done: break
            time.sleep(sim_speed/1000)

# ── TAB 2: BASELINES (Simplified for demo) ────────────────────────────────────
with tab2:
    st.markdown('<div class="section-hdr">Precomputed Baseline Scores</div>', unsafe_allow_html=True)
    st.markdown("""
    | Task Level | Random | Rule-Based | LLM-Agent |
    |:---|:---:|:---:|:---:|
    | 🟢 Easy | 0.25 | 0.65| 0.82 |
    | 🟡 Medium | 0.22 | 0.58 | 0.78 |
    | 🔴 Hard | 0.20 | 0.52 | 0.75 |
    """)

# ── TAB 3: REWARD MATRIX ──────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-hdr">Shaped Reward Matrix</div>', unsafe_allow_html=True)
    labels = [f"{ACTION_EMOJIS[EmailAction(i)]} {ACTION_NAMES[EmailAction(i)]}" for i in range(6)]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    _fig_style(fig, ax)
    sns.heatmap(_REWARD_MATRIX, annot=True, fmt=".2f", cmap="YlGnBu", xticklabels=labels, yticklabels=labels, ax=ax)
    st.pyplot(fig)
