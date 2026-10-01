"""Chapter 2: stream tokens as they arrive (needs ANTHROPIC_API_KEY)."""
from agentic.llm import stream_text

prompt = "Explain an agent loop in 3 lines."
for chunk in stream_text([{"role": "user", "content": prompt}]):
    print(chunk, end="", flush=True)
print()
