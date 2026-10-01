# Agentic Python

Production AI agents in plain Python, no framework.

This is the companion code for my book *Agentic Python: Build Production
AI Agents from Scratch*. Every code listing in the book comes from this
repository, and every module is covered by the test suite.

## Quick start

```bash
git clone https://github.com/ashishjsharda/agentic-python
cd agentic-python
pip install -r requirements-dev.txt
pytest                                   # runs offline, no API key
python -m examples.first_agent
```

Set `DATABASE_URL` to a Postgres database with pgvector to include the
Postgres tests; without it they're skipped.

## Running against a live model

Set `ANTHROPIC_API_KEY`, or `AGENT_PROVIDER=openai` with
`OPENAI_API_KEY` and `AGENT_MODEL`, then:

```bash
python -m examples.assistant_v1 "How tall is Mount Everest?"
python -m research_assistant.mcp_server   # its tools over MCP (stdio)
python -m evals.run                       # the eval suite
```

## The research assistant as a service

```bash
export ANTHROPIC_API_KEY=...
export AGENT_PRICE_IN_PER_MTOK=...  AGENT_PRICE_OUT_PER_MTOK=...
docker compose up --build -d
docker compose exec api python -m research_assistant.admin create-key you
curl -X POST localhost:8000/research \
  -H "authorization: Bearer ra_..." -H "content-type: application/json" \
  -d '{"question": "When was the Eiffel Tower completed?"}'
```

## What's inside

`agentic/` is the library you build chapter by chapter.
`research_assistant/` is the application that grows through the book.

| Chapter | Topic | Code |
| --- | --- | --- |
| 2  | Model APIs and provider portability | `agentic/llm.py` |
| 3  | Structured outputs | `agentic/structured.py`, `research_assistant/models.py` |
| 4  | Tools and dispatch | `agentic/tools.py`, `research_assistant/sources.py` |
| 5  | The core loop; assistant v1 | `agentic/agent.py`, `research_assistant/assistant.py` |
| 6  | Memory, Postgres + pgvector | `agentic/memory.py`, `agentic/pgmemory.py`, `research_assistant/memory.py` |
| 7  | Retrieval over your documents | `agentic/rag.py`, `research_assistant/documents.py` |
| 8  | Planning and review | `research_assistant/planner.py` |
| 9  | Multi-agent teams | `agentic/multi.py` |
| 10 | Human-in-the-loop approval | `agentic/hitl.py` |
| 11 | MCP client and server | `agentic/mcp.py`, `research_assistant/mcp_server.py` |
| 12 | Retries, timeouts, guardrails | `agentic/reliability.py` |
| 13 | Tracing and replay | `agentic/observability.py` |
| 14 | Tests, evals, regression gate | `agentic/testing.py`, `agentic/evals.py`, `evals/`, `.github/workflows/` |
| 15 | Serving and scaling | `agentic/service.py`, `agentic/scaling.py` |
| 16 | Security | `agentic/security.py` |
| 17 | The same agent in LangGraph | `research_assistant/langgraph_version.py` |
| 18 | The authenticated service | `research_assistant/{db,logs,service,app,admin}.py`, `Dockerfile`, `docker-compose.yml` |

## License

MIT. See [LICENSE](LICENSE).
