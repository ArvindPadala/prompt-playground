import requests
import json


def call_mistral(prompt, system_prompt="", temperature=0.7, top_p=1.0, top_k=50, ollama_url="http://localhost:11434"):
    """
    Generator that streams tokens from the Mistral model via Ollama.
    Yields (token: str, stats: dict | None) tuples.
    stats is populated only on the final chunk when done=True.
    """
    url = f"{ollama_url.rstrip('/')}/api/generate"
    payload = {
        "model": "mistral",
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": int(top_k),
        },
    }
    if system_prompt.strip():
        payload["system"] = system_prompt

    try:
        response = requests.post(url, json=payload, stream=True, timeout=120)
        response.raise_for_status()
        for line in response.iter_lines():
            if line:
                data = json.loads(line.decode("utf-8"))
                token = data.get("response", "")
                if data.get("done", False):
                    stats = {
                        "prompt_tokens": data.get("prompt_eval_count", 0),
                        "output_tokens": data.get("eval_count", 0),
                        "duration_ns": data.get("eval_duration", 0),
                    }
                    yield token, stats
                else:
                    yield token, None
    except requests.exceptions.Timeout:
        yield "⚠️ Error: Request timed out after 120 seconds.", None
    except requests.exceptions.ConnectionError:
        yield f"⚠️ Error: Could not connect to Ollama at {ollama_url}. Is it running?", None
    except requests.exceptions.HTTPError as e:
        yield f"⚠️ HTTP Error: {e}", None
    except json.JSONDecodeError as e:
        yield f"⚠️ JSON parse error: {e}", None
    except Exception as e:
        yield f"⚠️ Unexpected error: {e}", None
