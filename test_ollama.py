import ollama

MODEL = "qwen3:1.7b"

try:
    models = [m.model for m in ollama.list().models]
    print(f"Ollama reachable. Models available: {models}")
    if MODEL not in models:
        raise RuntimeError(f"{MODEL!r} not pulled -- run: ollama pull {MODEL}")

    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": "Reply with exactly: Ollama is working"}],
        think=False,
    )
    print(response.message.content)
except Exception as exc:
    print(type(exc).__name__)
    print(exc)
