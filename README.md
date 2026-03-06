# prompt-playground

## 🚀 Project Title
Prompt Engineering Playground: Experiment, Compare & Learn

## 🎯 Project Description
The Prompt Engineering Playground is an interactive web app that lets you experiment with prompt design, decoding strategies, and model behaviour of modern Large Language Models (LLMs) — Mistral, Gemma, and Llama2 — running locally via [Ollama](https://ollama.ai/).

Input a prompt, optionally set a system instruction, tweak temperature/top-k/top-p, and watch all selected models respond in real time. Run a single model with live token streaming, or select multiple models for instant side-by-side comparison.

## 🌟 Features

| Feature | Status |
|---|---|
| Prompt Input | ✅ |
| System Prompt (instruction/persona) | ✅ |
| Model Selection — Mistral, Gemma, Llama2 | ✅ |
| Decoding Controls — temperature, top-k, top-p | ✅ |
| Real-Time Token Streaming | ✅ |
| Multi-Model Side-by-Side Comparison | ✅ |
| Prompt History (last 10) | ✅ |
| Token Usage Stats (prompt tokens, output tokens, tokens/sec) | ✅ |
| Configurable Ollama Endpoint | ✅ |
| Copy-to-clipboard for Outputs | ✅ |

## 🔧 Tech Stack

- Python 3.10+
- [Streamlit](https://streamlit.io/) `>=1.32`
- [Ollama](https://ollama.ai/) (local LLM runtime)
- `requests` (HTTP client for Ollama API)

## 🚦 Prerequisites

1. Install [Ollama](https://ollama.ai/) and pull your desired models:
   ```bash
   ollama pull mistral
   ollama pull gemma
   ollama pull llama2
   ```

2. Make sure Ollama is running:
   ```bash
   ollama serve
   ```

## ⚡ Quick Start

```bash
git clone https://github.com/your-username/prompt-playground.git
cd prompt-playground
pip install -r requirements.txt
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

## 📁 Project Structure

```
prompt-playground/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md
└── models/
    ├── __init__.py         # Exports all model functions
    ├── gemma_model.py      # Gemma streaming generator
    ├── llama_model.py      # Llama2 streaming generator
    └── mistral_model.py    # Mistral streaming generator
```

## 📈 Planned Enhancements

- Save/export prompt history as JSON or CSV
- User feedback: "Which output was better?"
- Prompt template library
- Token probability / logprob visualisation

## 🙌 Learning Outcomes

By building this, you'll master:

- Prompt design principles
- LLM decoding parameters (temperature, top-k, top-p)
- Streaming API responses with NDJSON
- Concurrency in Python (`ThreadPoolExecutor`)
- Streamlit session state and layout APIs
