# app/dashboard.py
"""Streamlit Web Dashboard for the GPT-2 model.

Implements a premium, studio-grade UI/UX inspired by OpenAI, Perplexity, and Vercel.
Supports custom themes, moving starfield backgrounds, glassmorphic panels, and
custom widget overrides. Connects to the FastAPI backend or falls back to local CPU serving.
"""

import os
import sys
import time
import requests
import streamlit as st
from PIL import Image

# Add root folder to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import local modules for fallback mode
try:
    from app.inference import GPTInferenceEngine
except Exception:
    pass

# Page Setup
st.set_page_config(
    page_title="GPT-2 Studio",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Connection Manager & Mode Selection ─────────────────────────────
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000").rstrip("/")

@st.cache_resource(show_spinner=False)
def get_standalone_engine():
    """Load the model directly into Streamlit memory as a fallback."""
    paths = [
        os.path.join("checkpoints", "best_model.pt"),
        os.path.join("checkpoints_tiny", "best_model.pt"),
    ]
    checkpoint_path = None
    for path in paths:
        if os.path.exists(path):
            checkpoint_path = path
            break
            
    if checkpoint_path:
        try:
            return GPTInferenceEngine(checkpoint_path, device="cpu")
        except Exception:
            pass
            
    # Fallback to dummy model configuration if no checkpoint exists
    try:
        from model.gpt import GPTModel, count_parameters
        from model.tokenizer import GPT2Tokenizer
        
        class DummyEngine:
            def __init__(self):
                dummy_cfg = {
                    "vocab_size": 50257,
                    "context_length": 256,
                    "emb_dim": 64,
                    "n_heads": 2,
                    "n_layers": 1,
                    "drop_rate": 0.0,
                    "qkv_bias": False,
                }
                self.device = "cpu"
                self.model = GPTModel(dummy_cfg).eval()
                self.tokenizer = GPT2Tokenizer()
                self.context_size = 256
                self.parameter_count = count_parameters(self.model)
                
            def generate(self, prompt, max_new_tokens=50, temperature=0.8, top_k=50, use_cache=True):
                input_ids = self.tokenizer.text_to_token_ids(prompt)
                import time
                start = time.perf_counter()
                from model.gpt import generate as gpt_gen
                output_ids = gpt_gen(
                    self.model, input_ids, max_new_tokens, self.context_size,
                    temperature, top_k, use_cache=use_cache
                )
                latency = time.perf_counter() - start
                generated_text = self.tokenizer.token_ids_to_text(output_ids)
                num_gen = output_ids.shape[1] - input_ids.shape[1]
                return {
                    "prompt": prompt,
                    "generated_text": generated_text + " [Fallback Dummy Model]",
                    "tokens_generated": num_gen,
                    "time_taken_seconds": latency,
                    "tokens_per_second": num_gen / latency if latency > 0 else 0.0
                }
        return DummyEngine()
    except Exception:
        return None

# Check API health
api_connected = False
model_metadata = {}
try:
    response = requests.get(f"{BACKEND_URL}/health", timeout=2)
    if response.status_code == 200:
        api_connected = True
        model_metadata = response.json()
except Exception:
    pass

app_mode = "api" if api_connected else "standalone"
local_engine = get_standalone_engine() if app_mode == "standalone" else None

# Initialize Session State
if "theme" not in st.session_state:
    st.session_state.theme = "dark"
if "nav_active" not in st.session_state:
    st.session_state.nav_active = "generate"

# ─── Design System & CSS Engine ──────────────────────────────────────
def inject_custom_styles(theme: str):
    """Generate and inject CSS classes to override default Streamlit styles."""
    
    # Theme variables definition
    if theme == "dark":
        css_vars = """
        :root {
            --bg-base: #0a0b10;
            --bg-surface: rgba(17, 24, 39, 0.4);
            --bg-card: rgba(15, 23, 42, 0.5);
            --border-primary: rgba(255, 255, 255, 0.06);
            --border-hover: rgba(255, 255, 255, 0.12);
            --text-primary: #f9fafb;
            --text-secondary: #9ca3af;
            --text-muted: #6b7280;
            --accent-primary: #4f46e5;
            --accent-glow: rgba(79, 70, 229, 0.12);
            --accent-secondary: #0ea5e9;
            --sidebar-bg: rgba(5, 7, 12, 0.7);
            --star-color: rgba(255, 255, 255, 0.85);
            --radial-glow: radial-gradient(circle at 50% 30%, rgba(30, 27, 75, 0.45) 0%, rgba(10, 11, 16, 0) 70%);
        }
        """
    else:  # light theme
        css_vars = """
        :root {
            --bg-base: #f8fafc;
            --bg-surface: rgba(255, 255, 255, 0.75);
            --bg-card: rgba(255, 255, 255, 0.9);
            --border-primary: rgba(15, 23, 42, 0.06);
            --border-hover: rgba(15, 23, 42, 0.12);
            --text-primary: #0f172a;
            --text-secondary: #475569;
            --text-muted: #94a3b8;
            --accent-primary: #4f46e5;
            --accent-glow: rgba(79, 70, 229, 0.08);
            --accent-secondary: #0284c7;
            --sidebar-bg: rgba(241, 245, 249, 0.8);
            --star-color: rgba(79, 70, 229, 0.35);
            --radial-glow: radial-gradient(circle at 50% 30%, rgba(224, 231, 255, 0.6) 0%, rgba(248, 250, 252, 0) 70%);
        }
        """

    st.markdown(
        f"""
        <style>
        {css_vars}

        /* 1. Global Font and Background reset */
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
        
        .stApp {{
            background: var(--bg-base);
            background-image: var(--radial-glow);
            font-family: 'Outfit', sans-serif !important;
            color: var(--text-primary);
            overflow-x: hidden !important;
        }}
        
        /* 2. Starfield Animation Background (CPU Friendly Box Shadows) */
        .space-stars {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            z-index: -2;
            pointer-events: none;
            opacity: 0.5;
        }}
        
        #stars {{
            width: 1.5px;
            height: 1.5px;
            background: transparent;
            box-shadow: 100px 300px var(--star-color), 300px 500px var(--star-color), 800px 100px var(--star-color), 
                        1200px 600px var(--star-color), 1400px 200px var(--star-color), 500px 900px var(--star-color),
                        1600px 800px var(--star-color), 200px 1100px var(--star-color), 950px 700px var(--star-color);
            animation: moveStars 180s linear infinite;
        }}
        
        #stars2 {{
            width: 2.5px;
            height: 2.5px;
            background: transparent;
            box-shadow: 150px 400px var(--star-color), 450px 600px var(--star-color), 900px 200px var(--star-color),
                        1300px 700px var(--star-color), 1500px 300px var(--star-color), 600px 1000px var(--star-color);
            animation: moveStars 120s linear infinite;
        }}

        @keyframes moveStars {{
            from {{ transform: translateY(0px); }}
            to {{ transform: translateY(-2000px); }}
        }}

        /* 3. Streamlit Default Widget Overrides */
        
        /* Hide Default Header & Footer */
        header, footer {{ visibility: hidden !important; }}
        
        /* Sidebar Styling */
        section[data-testid="stSidebar"] {{
            background-color: var(--sidebar-bg) !important;
            backdrop-filter: blur(20px) !important;
            border-right: 1px solid var(--border-primary) !important;
        }}
        
        /* Sliders Override (Vercel Style) */
        div[data-testid="stWidgetLabel"] p {{
            font-size: 11px !important;
            color: var(--text-muted) !important;
            font-weight: 600 !important;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 2px !important;
        }}
        
        div[data-testid="stSlider"] [data-baseweb="slider"] > div {{
            background-color: var(--border-primary) !important;
            height: 4px !important;
        }}
        div[data-testid="stSlider"] [data-baseweb="slider"] > div > div {{
            background-color: var(--accent-primary) !important;
            height: 4px !important;
        }}
        div[data-testid="stSlider"] [role="slider"] {{
            background-color: var(--accent-primary) !important;
            border: 2px solid var(--bg-base) !important;
            width: 14px !important;
            height: 14px !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2) !important;
            transition: transform 0.1s ease !important;
        }}
        div[data-testid="stSlider"] [role="slider"]:hover {{
            transform: scale(1.2) !important;
        }}
        div[data-testid="stSlider"] [data-testid="stThumbValue"] {{
            background-color: var(--accent-primary) !important;
            color: #ffffff !important;
            border-radius: 4px !important;
            font-size: 11px !important;
            padding: 2px 6px !important;
        }}

        /* Input Areas Override */
        div[data-testid="stTextArea"] textarea {{
            background-color: var(--bg-surface) !important;
            border: 1px solid var(--border-primary) !important;
            border-radius: 12px !important;
            color: var(--text-primary) !important;
            font-size: 15px !important;
            padding: 16px !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 10px rgba(0,0,0,0.03) !important;
            font-family: inherit !important;
            line-height: 1.6 !important;
        }}
        div[data-testid="stTextArea"] textarea:focus {{
            border-color: var(--accent-primary) !important;
            box-shadow: 0 0 15px var(--accent-glow) !important;
            outline: none !important;
        }}

        /* Custom Buttons styling */
        .stButton button {{
            background-color: var(--accent-primary) !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 10px 24px !important;
            font-weight: 600 !important;
            font-size: 14px !important;
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
            box-shadow: 0 4px 14px var(--accent-glow) !important;
            width: 100% !important;
        }}
        .stButton button:hover {{
            transform: translateY(-1px) !important;
            box-shadow: 0 6px 20px var(--accent-glow) !important;
            opacity: 0.95 !important;
        }}
        .stButton button:active {{
            transform: translateY(1px) !important;
        }}
        
        /* Toggles Switch Styling */
        div[data-testid="stToggle"] [role="switch"] {{
            background-color: var(--border-primary) !important;
        }}
        div[data-testid="stToggle"] [role="switch"][aria-checked="true"] {{
            background-color: var(--accent-primary) !important;
        }}

        /* 4. Native Container overrides to create Glassmorphic Cards */
        div[data-testid="stVerticalBlockBorderWrapper"] {{
            background: var(--bg-card) !important;
            backdrop-filter: blur(20px) !important;
            -webkit-backdrop-filter: blur(20px) !important;
            border: 1px solid var(--border-primary) !important;
            border-radius: 16px !important;
            padding: 24px !important;
            box-shadow: 0 10px 30px rgba(0,0,0,0.05) !important;
            transition: border-color 0.3s ease, box-shadow 0.3s ease !important;
            margin-bottom: 20px !important;
        }}
        div[data-testid="stVerticalBlockBorderWrapper"]:hover {{
            border-color: var(--border-hover) !important;
        }}
        
        /* Telemetry Metrics */
        .telemetry-container {{
            display: flex;
            gap: 16px;
            margin-top: 16px;
        }}
        .telemetry-card {{
            flex: 1;
            background: rgba(255, 255, 255, 0.01);
            border: 1px solid var(--border-primary);
            border-radius: 12px;
            padding: 18px;
            text-align: center;
            box-shadow: 0 4px 10px rgba(0,0,0,0.02);
            transition: all 0.3s ease;
        }}
        .telemetry-card:hover {{
            border-color: var(--border-hover);
            transform: translateY(-2px);
        }}
        .telemetry-value {{
            font-size: 26px;
            font-weight: 700;
            color: var(--text-primary);
            background: linear-gradient(135deg, var(--text-primary) 30%, var(--accent-secondary) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .telemetry-label {{
            font-size: 11px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-top: 6px;
            font-weight: 600;
        }}

        /* Clean Output Area */
        .output-card {{
            background: rgba(255, 255, 255, 0.01);
            border: 1px solid var(--border-primary);
            border-radius: 12px;
            padding: 24px;
            min-height: 120px;
            color: var(--text-primary);
            font-size: 16px;
            line-height: 1.7;
            font-family: inherit;
            white-space: pre-wrap;
            box-shadow: inset 0 2px 8px rgba(0,0,0,0.03);
            border-left: 3px solid var(--accent-primary);
            animation: fadeIn 0.4s ease-out;
        }}

        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(4px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        </style>

        <div class="space-stars">
            <div id="stars"></div>
            <div id="stars2"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

inject_custom_styles(st.session_state.theme)

# ─── Custom Collapsible Sidebar Navigation ───────────────────────────
st.sidebar.markdown(
    """
    <div style="padding: 10px 0 25px 0;">
        <h2 style="font-weight: 700; font-size: 24px; letter-spacing: -0.5px; margin: 0; color: var(--text-primary);">
            Studio<span style="color: var(--accent-primary);">.</span>
        </h2>
        <p style="font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1px; margin-top: 4px;">
            Autoregressive GPT-2
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Custom clickable navigation menu mapping to session state
def set_nav(target):
    st.session_state.nav_active = target

col_nav1, col_nav2 = st.sidebar.columns(2)
with col_nav1:
    if st.button("Playground", key="btn_nav_gen", on_click=set_nav, args=("generate",)): pass
with col_nav2:
    if st.button("Analytics", key="btn_nav_an", on_click=set_nav, args=("analytics",)): pass

col_nav3, col_nav4 = st.sidebar.columns(2)
with col_nav3:
    if st.button("Architecture", key="btn_nav_arch", on_click=set_nav, args=("architecture",)): pass
with col_nav4:
    if st.button("Settings", key="btn_nav_set", on_click=set_nav, args=("settings",)): pass

# Active Indicator in Sidebar
nav_label = {
    "generate": "Playground",
    "analytics": "Analytics",
    "architecture": "Architecture",
    "settings": "Settings"
}[st.session_state.nav_active]

st.sidebar.markdown(
    f"""
    <div style="margin-top: 15px; padding: 8px 12px; border-radius: 6px; background: var(--accent-glow); border: 1px solid var(--border-primary); font-size: 13px; font-weight: 600; color: var(--text-primary); text-align: center;">
        Active View: {nav_label}
    </div>
    <hr style="margin: 20px 0; border: none; border-top: 1px solid var(--border-primary);" />
    """,
    unsafe_allow_html=True,
)

# ─── Sidebar System Metadata Panel ──────────────────────────────────
st.sidebar.markdown(
    """
    <div style="font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px; font-weight: 600;">
        System Metadata
    </div>
    """,
    unsafe_allow_html=True,
)

if app_mode == "api":
    checkpoint_name = os.path.basename(model_metadata.get('checkpoint', 'None'))
    if "checkpoint_tiny" in model_metadata.get('checkpoint', ''):
        model_type_label = "Tiny (Gibberish Output)"
    else:
        model_type_label = "Official Pretrained GPT-2 (Coherent English)"

    st.sidebar.markdown(
        f"""
        <div style="padding: 12px; border-radius: 8px; border: 1px solid var(--border-primary); background: rgba(0,0,0,0.1); font-size: 13px; line-height: 1.6; color: var(--text-secondary);">
            <div>Serving status: <strong style="color: #10b981;">🟢 Connected</strong></div>
            <div>Model Type: <strong style="color: var(--text-primary);">{model_type_label}</strong></div>
            <div>Device: <strong>{model_metadata.get('device', 'cpu').upper()}</strong></div>
            <div>Parameters: <strong>{model_metadata.get('parameters', 0):,}</strong></div>
            <div style="overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">Checkpoint: <strong title="{model_metadata.get('checkpoint', 'None')}">{checkpoint_name}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    # Standalone Standby Mode
    st.sidebar.markdown(
        f"""
        <div style="padding: 12px; border-radius: 8px; border: 1px solid var(--border-primary); background: rgba(0,0,0,0.1); font-size: 13px; line-height: 1.6; color: var(--text-secondary);">
            <div>Serving status: <strong style="color: var(--text-muted);">🟡 Standalone Standby</strong></div>
            <div>Device: <strong>CPU (Streamlit)</strong></div>
            <div>Parameters: <strong>{local_engine.parameter_count if local_engine else 0:,}</strong></div>
            <div>Checkpoint: <strong>Fallback Model</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ─── Navigation Routing Views ────────────────────────────────────────
active_view = st.session_state.nav_active

# ─── VIEW 1: Generator Playground ────────────────────────────────────
if active_view == "generate":
    st.markdown(
        """
        <div style="margin-bottom: 25px;">
            <h1 style="font-weight: 700; font-size: 36px; letter-spacing: -1px; margin: 0; color: var(--text-primary);">
                Copilot Studio
            </h1>
            <p style="font-size: 15px; color: var(--text-secondary); margin-top: 6px;">
                Direct autoregressive inference on custom GPT-2 checkpoints.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Core Layout
    col_input, col_params = st.columns([2, 1])

    with col_input:
        with st.container(border=True):
            st.markdown(
                """
                <div style="font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1px; font-weight: 600; margin-bottom: 12px;">
                    Input Prompt Context
                </div>
                """,
                unsafe_allow_html=True,
            )
            prompt_text = st.text_area(
                "Input Prompt",
                value="Once upon a time",
                height=140,
                label_visibility="collapsed",
                key="textarea_prompt",
            )
            
            # Generation Trigger Button
            generate_clicked = st.button("✨ Execute Inference Run", key="btn_run_gen")

        # Output Reveal Panel
        if generate_clicked:
            if not prompt_text.strip():
                st.error("Please provide a valid prompt context.")
            else:
                with st.container(border=True):
                    st.markdown(
                        """
                        <div style="font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1px; font-weight: 600; margin-bottom: 15px;">
                            Generation Stream & Stats
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Fetch settings from state to maintain consistency
                    max_new_tokens = st.session_state.get("set_max_tokens", 100)
                    temp = st.session_state.get("set_temp", 0.8)
                    top_k = st.session_state.get("set_top_k", 50)
                    use_cache = st.session_state.get("set_use_cache", True)

                    # Loading Shimmer
                    with st.spinner("Decoding token tensors..."):
                        try:
                            if app_mode == "api":
                                payload = {
                                    "prompt": prompt_text,
                                    "max_new_tokens": max_new_tokens,
                                    "temperature": temp,
                                    "top_k": top_k,
                                    "use_cache": use_cache,
                                }
                                res = requests.post(f"{BACKEND_URL}/generate", json=payload, timeout=60)
                                if res.status_code == 200:
                                    results = res.json()
                                else:
                                    raise Exception(res.json().get("detail", "API Error"))
                            else:
                                results = local_engine.generate(
                                    prompt=prompt_text,
                                    max_new_tokens=max_new_tokens,
                                    temperature=temp,
                                    top_k=top_k,
                                    use_cache=use_cache,
                                )
                            
                            # Render Telemetry Details
                            st.markdown(
                                f"""
                                <div class="telemetry-container">
                                    <div class="telemetry-card">
                                        <div class="telemetry-value">{results['time_taken_seconds']:.3f}s</div>
                                        <div class="telemetry-label">Latency</div>
                                    </div>
                                    <div class="telemetry-card">
                                        <div class="telemetry-value">{results['tokens_per_second']:.1f}</div>
                                        <div class="telemetry-label">Tokens / Sec</div>
                                    </div>
                                    <div class="telemetry-card">
                                        <div class="telemetry-value">{results['tokens_generated']}</div>
                                        <div class="telemetry-label">Tokens Gen</div>
                                    </div>
                                    <div class="telemetry-card">
                                        <div class="telemetry-value">{'ON' if use_cache else 'OFF'}</div>
                                        <div class="telemetry-label">KV-Cache</div>
                                    </div>
                                </div>
                                <div style="margin-top: 25px;"></div>
                                """,
                                unsafe_allow_html=True,
                            )

                            # Text display
                            st.markdown(
                                f'<div class="output-card">{results["generated_text"]}</div>',
                                unsafe_allow_html=True,
                            )

                        except Exception as e:
                            st.error(f"Inference run failed: {e}")
        else:
            st.markdown(
                """
                <div style="padding: 40px; border-radius: 12px; border: 1px dashed var(--border-primary); text-align: center; color: var(--text-muted); font-size: 14px;">
                    Provide a seed prompt on the left and execute the run to see autoregressive token generation.
                </div>
                """,
                unsafe_allow_html=True,
            )

    with col_params:
        with st.container(border=True):
            st.markdown(
                """
                <div style="font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1px; font-weight: 600; margin-bottom: 20px;">
                    Hyperparameters
                </div>
                """,
                unsafe_allow_html=True,
            )
            
            # Local settings variables mapped to session state
            max_tokens = st.slider("Max Tokens", min_value=10, max_value=300, value=st.session_state.get("set_max_tokens", 100), step=10, key="set_max_tokens")
            temp = st.slider("Temperature", min_value=0.0, max_value=1.5, value=st.session_state.get("set_temp", 0.8), step=0.1, key="set_temp")
            top_k = st.slider("Top-K Limit", min_value=1, max_value=100, value=st.session_state.get("set_top_k", 50), step=5, key="set_top_k")
            use_cache = st.toggle("Enable KV-Cache (Speedup)", value=st.session_state.get("set_use_cache", True), key="set_use_cache")
            
            st.markdown(
                """
                <div style="margin-top: 15px; font-size: 12px; color: var(--text-muted); line-height: 1.5;">
                    <strong>KV-Cache Note:</strong> Enabling caching maintains the Key and Value attention tensors in memory across steps, avoiding $O(N^2)$ recalculations.
                </div>
                """,
                unsafe_allow_html=True,
            )

# ─── VIEW 2: Analytics ───────────────────────────────────────────────
elif active_view == "analytics":
    st.markdown(
        """
        <div style="margin-bottom: 25px;">
            <h1 style="font-weight: 700; font-size: 36px; letter-spacing: -1px; margin: 0; color: var(--text-primary);">
                Training Telemetry
            </h1>
            <p style="font-size: 15px; color: var(--text-secondary); margin-top: 6px;">
                Logs, parameters, and loss metrics from the pre-training execution.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_plot, col_meta = st.columns([2, 1])

    with col_plot:
        with st.container(border=True):
            st.markdown(
                """
                <div style="font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1px; font-weight: 600; margin-bottom: 20px;">
                    Cross-Entropy Loss Progression
                </div>
                """,
                unsafe_allow_html=True,
            )

            plot_loaded = False
            if app_mode == "api":
                try:
                    plot_res = requests.get(f"{BACKEND_URL}/training/plot", stream=True)
                    if plot_res.status_code == 200:
                        img = Image.open(plot_res.raw)
                        st.image(img, use_container_width=True)
                        plot_loaded = True
                except Exception:
                    pass
                    
            if not plot_loaded:
                paths = ["logs_tiny/loss.png", "logs/loss.png", "loss.png"]
                for path in paths:
                    if os.path.exists(path):
                        st.image(path, use_container_width=True)
                        plot_loaded = True
                        break

            if not plot_loaded:
                st.info("No pre-training loss plot curves image found. Execute a training cycle to log parameters.")

    with col_meta:
        with st.container(border=True):
            st.markdown(
                """
                <div style="font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1px; font-weight: 600; margin-bottom: 20px;">
                    Dataset & Run Configurations
                </div>
                <div style="font-size: 14px; line-height: 1.7; color: var(--text-secondary);">
                    Pre-training was executed locally on <strong>Edith Wharton's "The Verdict"</strong>:
                    <ul style="margin-top: 8px; padding-left: 20px; color: var(--text-secondary);">
                        <li>Character count: 20,479</li>
                        <li>Tokenizer: OpenAI GPT-2 BPE</li>
                        <li>Vocab limit: 50,257</li>
                        <li>Context constraint: 256 tokens</li>
                        <li>Warmup cycle: 5-20 steps</li>
                        <li>Optimizers: AdamW (weight decay 0.1)</li>
                    </ul>
                    All training telemetry, including step metrics and model weights, are tracked and logged via MLflow.
                </div>
                """,
                unsafe_allow_html=True,
            )

# ─── VIEW 3: Architecture ────────────────────────────────────────────
elif active_view == "architecture":
    st.markdown(
        """
        <div style="margin-bottom: 25px;">
            <h1 style="font-weight: 700; font-size: 36px; letter-spacing: -1px; margin: 0; color: var(--text-primary);">
                Model Architecture
            </h1>
            <p style="font-size: 15px; color: var(--text-secondary); margin-top: 6px;">
                Modular transformer layer implementations mapped to GPT-2 design specifications.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_attn, col_layer = st.columns(2)

    with col_attn:
        with st.container(border=True):
            st.markdown(
                """
                <h3 style="font-weight: 600; font-size: 18px; margin: 0 0 15px 0; color: var(--text-primary);">
                    Attention Flow & Caching
                </h3>
                <p style="font-size: 14px; line-height: 1.7; color: var(--text-secondary);">
                    Our custom <strong>MultiHeadAttention</strong> layer is built completely from scratch using standard PyTorch primitives.
                    It splits the embedding dimensions into parallel attention heads, projects Queries, Keys, and Values, and calculates dot-product attention scores.
                </p>
                <div style="background: rgba(255,255,255,0.02); border: 1px solid var(--border-primary); border-radius: 8px; padding: 15px; font-family: monospace; font-size: 12px; margin-top: 15px; color: var(--text-secondary);">
                    def forward(self, x, layer_past=None, use_cache=False):<br>
                    &nbsp;&nbsp;&nbsp;&nbsp;# Projects Q, K, V for input x<br>
                    &nbsp;&nbsp;&nbsp;&nbsp;keys = self.W_key(x)<br>
                    &nbsp;&nbsp;&nbsp;&nbsp;queries = self.W_query(x)<br>
                    &nbsp;&nbsp;&nbsp;&nbsp;values = self.W_value(x)<br>
                    &nbsp;&nbsp;&nbsp;&nbsp;...<br>
                    &nbsp;&nbsp;&nbsp;&nbsp;if layer_past is not None:<br>
                    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;keys = torch.cat((past_k, keys), dim=-2)<br>
                    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;values = torch.cat((past_v, values), dim=-2)
                </div>
                """,
                unsafe_allow_html=True,
            )

    with col_layer:
        with st.container(border=True):
            st.markdown(
                """
                <h3 style="font-weight: 600; font-size: 18px; margin: 0 0 15px 0; color: var(--text-primary);">
                    Normalization & Activations
                </h3>
                <p style="font-size: 14px; line-height: 1.7; color: var(--text-secondary);">
                    The architecture implements a <strong>Pre-LayerNorm</strong> structure, applying custom Layer Normalization prior to Multi-Head Attention and FFN layers.
                    This ensures stable gradient flow directly through residual connections. The activation blocks use a hand-implemented tanh-approximation of the <strong>GELU</strong> activation function matching the GPT-2 paper.
                </p>
                <div style="background: rgba(255,255,255,0.02); border: 1px solid var(--border-primary); border-radius: 8px; padding: 15px; font-family: monospace; font-size: 12px; margin-top: 15px; color: var(--text-secondary);">
                    # Tanh approximation of GELU<br>
                    def forward(self, x):<br>
                    &nbsp;&nbsp;&nbsp;&nbsp;return 0.5 * x * (1 + torch.tanh(<br>
                    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;torch.sqrt(2 / pi) * (x + 0.044715 * x^3)<br>
                    &nbsp;&nbsp;&nbsp;&nbsp;))
                </div>
                """,
                unsafe_allow_html=True,
            )

# ─── VIEW 4: Settings & Configuration ────────────────────────────────
elif active_view == "settings":
    st.markdown(
        """
        <div style="margin-bottom: 25px;">
            <h1 style="font-weight: 700; font-size: 36px; letter-spacing: -1px; margin: 0; color: var(--text-primary);">
                System Settings
            </h1>
            <p style="font-size: 15px; color: var(--text-secondary); margin-top: 6px;">
                Configure UI themes, active serving locations, and system properties.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.markdown(
            """
            <div style="font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1px; font-weight: 600; margin-bottom: 25px;">
                Appearance & Serving Variables
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div style="font-size: 14px; font-weight: 500; color: var(--text-secondary); margin-bottom: 12px;">
                Select Interface Theme (Current: {st.session_state.theme.upper()})
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_theme1, col_theme2 = st.columns([1, 4])
        with col_theme1:
            if st.session_state.theme == "dark":
                if st.button("☀️ Light Mode"):
                    st.session_state.theme = "light"
                    st.rerun()
            else:
                if st.button("🌑 Dark Mode"):
                    st.session_state.theme = "dark"
                    st.rerun()

        st.markdown(
            """
            <div style="margin-top: 35px; border-top: 1px solid var(--border-primary); padding-top: 25px;">
                <div style="font-size: 14px; font-weight: 500; color: var(--text-secondary); margin-bottom: 8px;">
                    Active Serving Endpoint URL
                </div>
                <div style="font-size: 13px; font-family: monospace; padding: 12px; border-radius: 6px; background: rgba(0,0,0,0.2); border: 1px solid var(--border-primary); display: inline-block; color: var(--text-secondary);">
                    BACKEND_URL: """ + BACKEND_URL + """
                </div>
                <div style="margin-top: 12px; font-size: 12px; color: var(--text-muted);">
                    Modify the BACKEND_URL environment variable to repoint the Streamlit dashboard to a different remote FastAPI server.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
