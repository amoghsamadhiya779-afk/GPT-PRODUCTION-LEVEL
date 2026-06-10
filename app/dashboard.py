# app/dashboard.py
"""Streamlit Web Dashboard for the GPT-2 model.

Acts as an interactive playground. Detects the status of the FastAPI backend:
1. API Mode: Queries FastAPI at http://localhost:8000 for generation & stats.
2. Standalone Mode: Falls back to loading the model directly on local CPU,
   guaranteeing a functional deployment even on free hosts (like Hugging Face Spaces).
"""

import os
import sys
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
    page_title="GPT-2 From Scratch Playground",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Premium Design Aesthetics
st.markdown(
    """
    <style>
    .reportview-container {
        background: #0b0f19;
    }
    /* Stat Card styling */
    .metric-card {
        background-color: #16213e;
        border: 1px solid #1f305e;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #00d2ff;
    }
    .metric-label {
        font-size: 12px;
        color: #8a99ad;
        text-transform: uppercase;
        margin-top: 5px;
    }
    /* Prompt box styling */
    .prompt-box {
        background-color: #1e1e2f;
        border-radius: 5px;
        padding: 10px;
        border-left: 5px solid #00d2ff;
        color: #e0e0e0;
        font-family: monospace;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Connection Manager & Mode Selection ─────────────────────────────
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000").rstrip("/")

@st.cache_resource(show_spinner=False)
def get_standalone_engine():
    """Load the model directly into Streamlit memory as a fallback."""
    # Find default checkpoint locations
    paths = [
        os.path.join("checkpoints_tiny", "best_model.pt"),
        os.path.join("checkpoints", "best_model.pt"),
    ]
    checkpoint_path = None
    for path in paths:
        if os.path.exists(path):
            checkpoint_path = path
            break
            
    if checkpoint_path:
        try:
            return GPTInferenceEngine(checkpoint_path, device="cpu")
        except Exception as e:
            st.sidebar.error(f"Failed to load local checkpoint: {e}")
            
    # If no checkpoint exists, create a dummy engine
    try:
        # Mock class for streamlit if no model exists
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
                    "generated_text": generated_text + " [Standalone Untrained Fallback Model Output]",
                    "tokens_generated": num_gen,
                    "time_taken_seconds": latency,
                    "tokens_per_second": num_gen / latency if latency > 0 else 0.0
                }
        return DummyEngine()
    except Exception as e:
        st.sidebar.critical(f"Failed to initialize dummy model: {e}")
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

# ─── Sidebar Status Panel ────────────────────────────────────────────
st.sidebar.title("🧠 System Status")

if api_connected:
    st.sidebar.success("🟢 API Connected")
    app_mode = "api"
    # Show metadata from FastAPI
    st.sidebar.markdown(f"**Loaded Checkpoint:** `{os.path.basename(model_metadata.get('checkpoint', 'None'))}`")
    st.sidebar.markdown(f"**Backend Device:** `{model_metadata.get('device', 'cpu')}`")
    st.sidebar.markdown(f"**Trainable Parameters:** `{model_metadata.get('parameters', 0):,}`")
else:
    st.sidebar.warning("🟡 Standalone Mode (Local CPU)")
    app_mode = "standalone"
    local_engine = get_standalone_engine()
    if local_engine:
        st.sidebar.markdown("**Loaded Checkpoint:** `Standalone Fallback Model`")
        st.sidebar.markdown("**Backend Device:** `cpu (Streamlit Thread)`")
        st.sidebar.markdown(f"**Trainable Parameters:** `{local_engine.parameter_count:,}`")
    else:
        st.sidebar.error("🔴 Offline - No Engine Available")

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    ### FAANG Scaling Design
    - **Microservice Decoupling**: API runs model serving, frontend runs UI.
    - **KV-Caching**: Generation runs in $O(N)$ instead of $O(N^2)$ space/time complexity.
    - **Lifespan Initialization**: Model is preloaded once at start.
    """
)

# ─── Main Content Title ──────────────────────────────────────────────
st.title("🧠 GPT-2 From Scratch: Serving Playground")
st.markdown("An interactive, production-ready demonstration of a GPT-2 model trained entirely from scratch.")

# Create tabs
tab_playground, tab_training, tab_architecture = st.tabs(
    ["🎮 Model Playground", "📊 Training Analytics", "🧱 Transformer Architecture"]
)

# ─── Tab 1: Model Playground ─────────────────────────────────────────
with tab_playground:
    st.header("Model Playground")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Configuration")
        
        prompt = st.text_area(
            "Prompt",
            value="Every effort moves you",
            placeholder="Type a starting prompt here...",
            height=100
        )
        
        max_tokens = st.slider("Max New Tokens", min_value=10, max_value=300, value=100, step=10)
        
        col_temp, col_k = st.columns(2)
        with col_temp:
            temperature = st.slider("Temperature (Creativity)", min_value=0.0, max_value=1.5, value=0.8, step=0.1)
        with col_k:
            top_k = st.slider("Top-k Sampling Limit", min_value=1, max_value=100, value=50, step=5)
            
        use_cache = st.toggle("Enable KV-Cache (FAANG Optimization)", value=True)
        
        generate_btn = st.button("🚀 Generate Text", use_container_width=True)
        
    with col2:
        st.subheader("Output & Telemetry")
        
        if generate_btn:
            if not prompt.strip():
                st.error("Please enter a valid prompt.")
            else:
                with st.spinner("Model is generating text..."):
                    try:
                        # 1. Execution Mode: API or Standalone local
                        if app_mode == "api":
                            payload = {
                                "prompt": prompt,
                                "max_new_tokens": max_tokens,
                                "temperature": temperature,
                                "top_k": top_k,
                                "use_cache": use_cache
                            }
                            res = requests.post(f"{BACKEND_URL}/generate", json=payload, timeout=60)
                            if res.status_code == 200:
                                results = res.json()
                            else:
                                raise Exception(res.json().get("detail", "API Error"))
                        else:
                            # Standalone execution
                            results = local_engine.generate(
                                prompt=prompt,
                                max_new_tokens=max_tokens,
                                temperature=temperature,
                                top_k=top_k,
                                use_cache=use_cache
                            )
                            
                        # 2. Render Telemetry Cards
                        st.markdown(
                            f"""
                            <div style="display: flex; gap: 10px; margin-bottom: 20px;">
                                <div class="metric-card" style="flex: 1;">
                                    <div class="metric-value">{results['time_taken_seconds']:.3f}s</div>
                                    <div class="metric-label">Latency</div>
                                </div>
                                <div class="metric-card" style="flex: 1;">
                                    <div class="metric-value">{results['tokens_per_second']:.1f}</div>
                                    <div class="metric-label">Tokens / Sec</div>
                                </div>
                                <div class="metric-card" style="flex: 1;">
                                    <div class="metric-value">{results['tokens_generated']}</div>
                                    <div class="metric-label">Generated</div>
                                </div>
                                <div class="metric-card" style="flex: 1;">
                                    <div class="metric-value">{'ON' if use_cache else 'OFF'}</div>
                                    <div class="metric-label">KV-Cache</div>
                                </div>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                        
                        # 3. Render Output Text
                        st.markdown("##### Generated Text Output:")
                        st.markdown(f'<div class="prompt-box">{results["generated_text"]}</div>', unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"Text generation failed: {e}")
        else:
            st.info("Adjust the configurations on the left and click 'Generate Text' to interact with the model.")

# ─── Tab 2: Training Analytics ───────────────────────────────────────
with tab_training:
    st.header("Training Progress & Metrics")
    
    col_plot, col_stats = st.columns([3, 2])
    
    with col_plot:
        st.subheader("Training Loss Curve")
        
        # Load and display training plot
        plot_shown = False
        if app_mode == "api":
            try:
                # Retrieve from API Response
                plot_res = requests.get(f"{BACKEND_URL}/training/plot", stream=True)
                if plot_res.status_code == 200:
                    img = Image.open(plot_res.raw)
                    st.image(img, caption="Pre-training loss curve plotted dynamically.", use_container_width=True)
                    plot_shown = True
            except Exception:
                pass
                
        if not plot_shown:
            # Check local file paths
            paths = ["logs_tiny/loss.png", "logs/loss.png", "loss.png"]
            for path in paths:
                if os.path.exists(path):
                    st.image(path, caption="Pre-training loss curve.", use_container_width=True)
                    plot_shown = True
                    break
                    
        if not plot_shown:
            st.info("No training loss curve image found. Run a training cycle to generate metrics.")
            
    with col_stats:
        st.subheader("Pre-training Dataset details")
        st.markdown(
            """
            This model was trained on **Edith Wharton's short story 'The Verdict'** downloaded directly 
            from the [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) repository.
            
            - **Text Length:** 20,479 characters
            - **Vocabulary:** tiktoken OpenAI GPT-2 BPE (50,257 tokens)
            - **Sequence length (context window):** 256 tokens
            - **Data Splitting:** 90% Training / 10% Validation
            
            #### Pre-training Hyperparameters
            - **Optimizer:** AdamW with weight decay ($0.1$)
            - **LR Schedule:** Cosine decay with linear warmup (peak learning rate: $1.0 \\times 10^{-3}$)
            - **Gradient Clipping:** Max norm of $1.0$
            - **Autoregressive Target:** Next-token prediction loss ($Y_{target}$ shifted by 1 token)
            """
        )

# ─── Tab 3: Transformer Architecture ─────────────────────────────────
with tab_architecture:
    st.header("GPT-2 Architectural Overview")
    st.markdown("This model follows the standard GPT-2 decoder-only transformer architecture built completely from scratch.")
    
    col_diagram, col_details = st.columns([1, 1])
    
    with col_diagram:
        st.markdown(
            """
            #### Attention Flow & KV-Cache
            Standard attention calculates matrix multiplications for all tokens:
            
            $$\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{Q K^T}{\\sqrt{d_k}}\\right) V$$
            
            During auto-regressive generation, tokens are predicted step-by-step. Without a **KV-Cache**, computing this formula at step $t$ requires calculating $Q$, $K$, and $V$ for all tokens from $0$ to $t$. This results in $O(t^2)$ latency.
            
            By implementing a **KV-Cache**:
            1. We store the computed Keys ($K$) and Values ($V$) from previous steps in memory.
            2. At step $t$, we project $Q, K, V$ for **only the newest token** (shape `1 x d_in`).
            3. We retrieve the cached Keys and Values, concatenate them with the new $K, V$, and compute attention scores.
            4. This reduces the complexity to $O(t)$ operations, optimizing inference speed.
            """
        )
        
    with col_details:
        st.markdown(
            """
            #### Modular Layers
            * **Token & Position Embeddings**: Maps token IDs to continuous spaces of dimension $E$, combined with a learnable absolute position embedding layer.
            * **Pre-LayerNorm Transformer Blocks**: Pre-LayerNorm applies normalization *before* feeding into attention and feed-forward layers. This allows gradients to flow directly through the residual skip connections, stabilizing training.
            * **GELU Activation**: Custom implementation of the Gaussian Error Linear Unit (tanh approximation) to provide non-linear transitions:
              $$GELU(x) = 0.5 \\cdot x \\cdot \\left(1 + \\tanh\\left(\\sqrt{\\frac{2}{\\pi}} \\cdot (x + 0.044715 \\cdot x^3)\\right)\\right)$$
            * **Output Head**: Linear layer mapping embedding states back to the vocabulary dimension ($50,257$) to output prediction logits.
            """
        )
