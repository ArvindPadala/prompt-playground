import streamlit as st
import requests
import json

# ===== Streamlit Page Config =====
st.set_page_config(page_title="Prompt Engineering Playground", layout="wide")

# ===== Sidebar - Model & Parameters =====
st.sidebar.title("🔧 Playground Settings")

model_choice = st.sidebar.selectbox("Select Model", ["Mistral (Ollama)", "Gemma (Ollama)", "Llama2 (Ollama)"])
temperature = st.sidebar.slider("Temperature", 0.1, 1.5, 0.7, 0.1)
top_p = st.sidebar.slider("Top-p", 0.1, 1.0, 1.0, 0.05)
top_k = st.sidebar.slider("Top-k", 0, 100, 50, 5)

# ===== Prompt Input =====
st.title("🧠 Prompt Engineering Playground")
user_prompt = st.text_area("Enter your prompt:", "Tell me a joke about data scientists")

submit = st.button("🚀 Generate Outputs")

# ===== Ollama API Call =====
def call_ollama(model, prompt, temperature, top_p, top_k):
    try:
        url = "http://localhost:11434/api/generate"
        response = requests.post(
            url,
            json={
                "model": model.lower(),
                "prompt": prompt,
                "options": {
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k
                }
            },
            stream=True
        )

        # Stream and concatenate response
        result_text = ""
        for line in response.iter_lines():
            if line:
                chunk = line.decode('utf-8')
                data = json.loads(chunk)  # ✅ Correct way
                result_text += data.get("response", "")

        return result_text

    except Exception as e:
        return f"Error: {e}"

# ===== Main Execution =====
if submit and user_prompt:
    st.markdown(f"### {model_choice} Output")
    model_map = {
        "Mistral (Ollama)": "mistral",
        "Gemma (Ollama)": "gemma",
        "Llama2 (Ollama)": "llama2"
    }
    output = call_ollama(model_map[model_choice], user_prompt, temperature, top_p, top_k)
    st.write(output)

st.sidebar.markdown("---")
st.sidebar.write("Made by Arvind Padala")
