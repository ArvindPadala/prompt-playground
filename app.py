import streamlit as st
import concurrent.futures
import time

from models import call_mistral, call_gemma, call_llama

# ===== Constants =====
MODEL_MAP = {
    "Mistral": ("mistral", call_mistral),
    "Gemma":   ("gemma",   call_gemma),
    "Llama2":  ("llama2",  call_llama),
}

MODEL_COLORS = {
    "Mistral": "#6C63FF",
    "Gemma":   "#00C9A7",
    "Llama2":  "#FF6B6B",
}

DEFAULT_OLLAMA_URL = "http://localhost:11434"
MAX_HISTORY = 10

# ===== Page Config =====
st.set_page_config(
    page_title="Prompt Engineering Playground",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===== Custom CSS =====
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Hero Header ── */
.hero {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 28px;
    color: white;
}
.hero h1 {
    font-size: 2.2rem;
    font-weight: 700;
    margin: 0 0 6px 0;
    letter-spacing: -0.5px;
}
.hero p {
    font-size: 1rem;
    margin: 0;
    opacity: 0.88;
}

/* ── Output Cards ── */
.model-card {
    border-radius: 12px;
    padding: 20px 24px;
    margin-top: 8px;
    border-left: 4px solid;
    background: #fafafa;
    min-height: 160px;
    position: relative;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    line-height: 1.7;
    font-size: 0.95rem;
    color: #1a1a2e;
    white-space: pre-wrap;
    word-break: break-word;
}
.model-header {
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    margin-bottom: 10px;
}

/* ── Metric Cards ── */
.stat-block {
    background: #f0f4ff;
    border-radius: 10px;
    padding: 14px 20px;
    text-align: center;
    margin-top: 4px;
}
.stat-value {
    font-size: 1.4rem;
    font-weight: 700;
    color: #4f46e5;
}
.stat-label {
    font-size: 0.75rem;
    color: #6b7280;
    margin-top: 2px;
    font-weight: 500;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #f8f9fc;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stTextInput label {
    font-weight: 600;
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #374151;
}

/* ── Section labels ── */
.section-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 1.3px;
    text-transform: uppercase;
    color: #9ca3af;
    margin-bottom: 10px;
    margin-top: 16px;
}

/* ── History item ── */
.history-item {
    background: #ffffff;
    border-radius: 8px;
    padding: 8px 12px;
    margin-bottom: 6px;
    font-size: 0.82rem;
    color: #374151;
    border: 1px solid #e5e7eb;
    cursor: pointer;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

/* ── Prompt area ── */
textarea {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
    line-height: 1.6 !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    font-size: 0.95rem;
    padding: 10px 28px;
    transition: opacity 0.2s ease, transform 0.1s ease;
}
.stButton > button:hover {
    opacity: 0.9;
    transform: translateY(-1px);
}
.stButton > button:active {
    transform: translateY(0px);
}

/* ── Copy code blocks ── */
.copy-label {
    font-size: 0.73rem;
    color: #9ca3af;
    text-align: right;
    margin-top: 4px;
}

div[data-testid="stExpander"] {
    border: 1px solid #e5e7eb;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ===== Session State Init =====
if "prompt_history" not in st.session_state:
    st.session_state.prompt_history = []
if "restore_prompt" not in st.session_state:
    st.session_state.restore_prompt = None


# ===== Sidebar =====
with st.sidebar:
    st.markdown("## 🧠 Playground")
    st.markdown("---")

    # ── Model Selection ──
    st.markdown('<div class="section-label">Models</div>', unsafe_allow_html=True)
    selected_models = []
    for model_name in MODEL_MAP:
        color = MODEL_COLORS[model_name]
        checked = st.checkbox(
            f"**{model_name}**",
            value=(model_name == "Mistral"),
            key=f"model_{model_name}",
        )
        if checked:
            selected_models.append(model_name)

    st.markdown("---")

    # ── Parameters ──
    st.markdown('<div class="section-label">Parameters</div>', unsafe_allow_html=True)
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.05,
                            help="Higher = more creative; lower = more deterministic")
    top_p = st.slider("Top-p (nucleus)", 0.0, 1.0, 1.0, 0.05,
                       help="Probability mass to sample from. 1.0 = no restriction")
    top_k = st.slider("Top-k", 0, 100, 50, 5,
                       help="Sample from top-k tokens. 0 = disabled")

    st.markdown("---")

    # ── Ollama URL ──
    st.markdown('<div class="section-label">Connection</div>', unsafe_allow_html=True)
    ollama_url = st.text_input("Ollama URL", value=DEFAULT_OLLAMA_URL,
                                help="URL of your running Ollama instance")

    st.markdown("---")

    # ── Prompt History ──
    st.markdown('<div class="section-label">📜 Prompt History</div>', unsafe_allow_html=True)
    if st.session_state.prompt_history:
        for i, entry in enumerate(reversed(st.session_state.prompt_history[-MAX_HISTORY:])):
            label = entry["prompt"][:55] + "…" if len(entry["prompt"]) > 55 else entry["prompt"]
            if st.button(label, key=f"history_{i}", use_container_width=True):
                st.session_state.restore_prompt = entry["prompt"]
                st.rerun()
    else:
        st.caption("No history yet. Submit a prompt to start.")

    st.markdown("---")
    st.markdown(
        '<div style="text-align:center; color:#9ca3af; font-size:0.78rem;">Made by Arvind Padala</div>',
        unsafe_allow_html=True,
    )


# ===== Hero Header =====
st.markdown("""
<div class="hero">
  <h1>🧠 Prompt Engineering Playground</h1>
  <p>Experiment with prompts, tune decoding parameters, and compare how different LLMs respond — in real time.</p>
</div>
""", unsafe_allow_html=True)


# ===== Prompt Input Area =====
default_prompt = st.session_state.restore_prompt or "Tell me a joke about data scientists."
if st.session_state.restore_prompt:
    st.session_state.restore_prompt = None

system_prompt = ""
with st.expander("🛠️ System Prompt (optional)", expanded=False):
    system_prompt = st.text_area(
        "System / Instruction Prompt",
        placeholder="e.g. You are a helpful assistant that always responds in bullet points.",
        height=100,
        label_visibility="collapsed",
    )

user_prompt = st.text_area(
    "Enter your prompt:",
    value=default_prompt,
    height=120,
    max_chars=4000,
    placeholder="Write your prompt here…",
)

col_btn, col_warn = st.columns([1, 3])
with col_btn:
    submit = st.button("🚀 Generate", use_container_width=True)
with col_warn:
    if not selected_models:
        st.warning("⚠️ Select at least one model in the sidebar.")


# ===== Helpers =====
def run_model_blocking(model_name, prompt, system_prompt, temperature, top_p, top_k, ollama_url):
    """Runs a model to completion (non-streaming). Returns (model_name, text, stats)."""
    _, fn = MODEL_MAP[model_name]
    full_text = ""
    final_stats = {}
    for token, stats in fn(
        prompt=prompt,
        system_prompt=system_prompt,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        ollama_url=ollama_url,
    ):
        full_text += token
        if stats:
            final_stats = stats
    return model_name, full_text, final_stats


def make_token_generator(model_name, prompt, system_prompt, temperature, top_p, top_k, ollama_url):
    """Returns a plain str generator (for st.write_stream)."""
    _, fn = MODEL_MAP[model_name]
    for token, _ in fn(
        prompt=prompt,
        system_prompt=system_prompt,
        temperature=temperature,
        top_p=top_k,
        top_k=top_k,
        ollama_url=ollama_url,
    ):
        yield token


def render_stats(stats: dict):
    """Renders token usage stats as three metric columns."""
    duration_s = stats.get("duration_ns", 0) / 1e9
    output_tokens = stats.get("output_tokens", 0)
    tps = round(output_tokens / duration_s, 1) if duration_s > 0 else 0

    c1, c2, c3, _ = st.columns([1, 1, 1, 3])
    with c1:
        st.markdown(f"""
        <div class="stat-block">
            <div class="stat-value">{stats.get('prompt_tokens', 0)}</div>
            <div class="stat-label">Prompt Tokens</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="stat-block">
            <div class="stat-value">{output_tokens}</div>
            <div class="stat-label">Output Tokens</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="stat-block">
            <div class="stat-value">{tps}</div>
            <div class="stat-label">Tokens / sec</div>
        </div>""", unsafe_allow_html=True)


# ===== Main Execution =====
if submit and user_prompt and selected_models:

    # Save to history
    st.session_state.prompt_history.append({
        "prompt": user_prompt,
        "models": selected_models,
    })
    if len(st.session_state.prompt_history) > MAX_HISTORY:
        st.session_state.prompt_history.pop(0)

    st.markdown("---")

    # ── Single model: stream live ──
    if len(selected_models) == 1:
        model_name = selected_models[0]
        color = MODEL_COLORS[model_name]
        st.markdown(
            f'<div class="model-header" style="color:{color};">{model_name}</div>',
            unsafe_allow_html=True,
        )
        _, fn = MODEL_MAP[model_name]
        collected = []
        final_stats = {}

        # Wrap generator so we capture stats from it
        def stream_and_capture():
            for token, s in fn(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                ollama_url=ollama_url,
            ):
                collected.append(token)
                if s:
                    final_stats.update(s)
                yield token

        st.write_stream(stream_and_capture())

        if final_stats:
            st.markdown("##### 📊 Token Stats")
            render_stats(final_stats)

        full_text = "".join(collected)
        with st.expander("📋 Copy Output"):
            st.code(full_text, language=None)

    # ── Multiple models: run concurrently, display in columns ──
    else:
        with st.spinner(f"Running {len(selected_models)} models in parallel…"):
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(selected_models)) as executor:
                futures = {
                    executor.submit(
                        run_model_blocking,
                        name, user_prompt, system_prompt,
                        temperature, top_p, top_k, ollama_url,
                    ): name
                    for name in selected_models
                }
                results = {}
                for future in concurrent.futures.as_completed(futures):
                    name, text, stats = future.result()
                    results[name] = (text, stats)

        for model_name in selected_models:
            color = MODEL_COLORS[model_name]
            text, stats = results[model_name]

            st.markdown(
                f'<div class="model-header" style="color:{color}; margin-top:20px;">'
                f'● {model_name}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="model-card" style="border-left-color:{color};">{text}</div>',
                unsafe_allow_html=True,
            )
            if stats:
                st.markdown("")
                render_stats(stats)
            with st.expander("📋 Copy output"):
                st.code(text, language=None)

            st.markdown("---")

elif submit and not user_prompt:
    st.warning("Please enter a prompt before generating.")
