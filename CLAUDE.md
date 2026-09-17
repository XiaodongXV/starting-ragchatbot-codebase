# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment notes (Windows / Sanofi corporate network)

- **Port 8000 is blocked** by corporate security software — always use port 8080.
- **Zscaler SSL interception** breaks Python's default certificate verification. `truststore` is already listed as a dependency and activated at the top of `backend/app.py` (`truststore.inject_into_ssl()`). Any new script that makes HTTPS calls must also call this before importing `requests`, `httpx`, or `huggingface_hub`.
  - `import litellm` itself makes an HTTPS call (tiktoken fetches the `cl100k_base` encoding at import time), so `backend/ai_generator.py` activates `truststore` before importing it. A one-off `uv run python -c` that imports any backend module transitively imports `litellm` — start such scripts with `import truststore; truststore.inject_into_ssl()` or they fail on certificate verification.
- **Do not use `--reload`** with uvicorn. It spawns a multiprocessing child that can leave orphaned processes holding the port after the parent dies.
- Python 3.13 is in `.venv/` (managed by uv). The system Python is 3.10.11 — do not use it for this project.
- **Frontend cache busting**: `index.html` references `style.css` and `script.js` with a `?v=N` query string. Increment N whenever frontend files are modified, otherwise browsers serve stale cached versions (304 Not Modified).

## Package and script execution

Always use `uv` — never `pip install` or `python` directly:

```bash
uv add <package>          # add a dependency (updates pyproject.toml and uv.lock)
uv run python script.py   # run any Python file inside the project venv
uv sync                   # install/sync all dependencies after pulling changes
```

## Running the app

```bash
# Install dependencies (first time or after pyproject.toml changes)
uv sync

# Start the server
cd backend
uv run uvicorn app:app --port 8080
```

App is available at `http://localhost:8080`. API docs at `http://localhost:8080/docs`.

## Environment variables

Create `backend/.env` (or `.env` in the root) with:
```
LLM_PROVIDER=anthropic   # anthropic | gemini | openai

ANTHROPIC_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here

# Optional — overrides the provider's default model
# LLM_MODEL=gemini/gemini-2.0-flash
```

Only the active provider's key needs to be set. `LLM_PROVIDER` selects the model and key at startup; an unrecognised value raises at import rather than falling back to a default, which would otherwise surface as a confusing auth error against the wrong vendor. Provider defaults live in `PROVIDERS` in `backend/config.py`.

Without a key, the app starts and serves static content and `/api/courses`, but `/api/query` will fail.

## Architecture

The entry point is `backend/app.py` (the root `main.py` is an unused stub). On startup, FastAPI calls `rag_system.add_course_folder("../docs")`, which scans for `.txt/.pdf/.docx` files and indexes any course not already in ChromaDB. Re-indexing only happens when the course title is new — modifying an existing file's content requires deleting `backend/chroma_db/` and restarting.

**Query flow:** `POST /api/query` → `RAGSystem.query()` → first LLM call with `search_course_content` and `get_course_outline` defined → if the model invokes a tool, it runs locally against ChromaDB → second LLM call with the tool results (no tools attached) → answer returned.

**LLM access goes through LiteLLM**, so all provider traffic uses OpenAI-format messages and tool definitions. `backend/ai_generator.py` is the only file that knows the wire format: the tool classes in `search_tools.py` still declare themselves in Anthropic shape (`input_schema`), and `AIGenerator._to_openai_tools()` translates on the way out. When editing the tool-use loop, note that OpenAI format delivers tool arguments as a JSON string (needs `json.loads`), keys results by `tool_call_id` on a `role: "tool"` message, and signals tool use via `finish_reason`/`message.tool_calls` rather than Anthropic's `stop_reason`.

**Two ChromaDB collections** (stored in `backend/chroma_db/`, git-ignored):

- `course_catalog` — one entry per course, used only to resolve fuzzy course names to canonical titles via semantic search before filtering `course_content`.
  - metadata: `title`, `instructor`, `course_link`, `lesson_count`, `lessons_json` (JSON-serialised list of `{lesson_number, lesson_title, lesson_link}` — stored as a string because ChromaDB metadata does not support nested objects)
- `course_content` — one entry per text chunk (~800 chars, 100-char overlap), the actual retrieval target.
  - metadata: `course_title`, `lesson_number`, `chunk_index`

**Chunk IDs** are `"{course_title_underscored}_{chunk_index}"`. Duplicate IDs cause a ChromaDB error, which is why the startup loader checks for existing titles before adding.

**Session history** is stored in memory only (lost on restart). `MAX_HISTORY = 2` means only the last 2 exchanges are passed to Claude as plain text in the system prompt — not as `messages` entries.
