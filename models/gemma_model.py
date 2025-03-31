import requests

def call_gemma(prompt, temperature=0.7, top_p=1.0, top_k=50):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "gemma",
                "prompt": prompt,
                "options": {
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k
                }
            }
        )
        result = response.json()
        return result.get("response", "No response")
    except Exception as e:
        return f"Error: {e}"
