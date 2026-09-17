# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment notes (Windows / Sanofi corporate network)

- **Port 8000 is blocked** by corporate security software — always use port 8080.
- **Zscaler SSL interception** breaks Python's default certificate verification. `truststore` is already listed as a dependency and activated at the top of `backend/app.py` (`truststore.inject_into_ssl()`). Any new script that makes HTTPS calls must also call this before importing `requests`, `httpx`, or `huggingface_hub`.
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

## Environment variable

Create `backend/.env` (or `.env` in the root) with:
```
ANTHROPIC_API_KEY=your_key_here
```

Without a key, the app starts and serves static content and `/api/courses`, but `/api/query` will fail.

## Architecture

The entry point is `backend/app.py` (the root `main.py` is an unused stub). On startup, FastAPI calls `rag_system.add_course_folder("../docs")`, which scans for `.txt/.pdf/.docx` files and indexes any course not already in ChromaDB. Re-indexing only happens when the course title is new — modifying an existing file's content requires deleting `backend/chroma_db/` and restarting.

**Query flow:** `POST /api/query` → `RAGSystem.query()` → first Claude API call with `search_course_content` tool defined → if Claude invokes the tool, `VectorStore.search()` runs locally against ChromaDB → second Claude API call with search results → answer returned.

**Two ChromaDB collections** (stored in `backend/chroma_db/`, git-ignored):

- `course_catalog` — one entry per course, used only to resolve fuzzy course names to canonical titles via semantic search before filtering `course_content`.
  - metadata: `title`, `instructor`, `course_link`, `lesson_count`, `lessons_json` (JSON-serialised list of `{lesson_number, lesson_title, lesson_link}` — stored as a string because ChromaDB metadata does not support nested objects)
- `course_content` — one entry per text chunk (~800 chars, 100-char overlap), the actual retrieval target.
  - metadata: `course_title`, `lesson_number`, `chunk_index`

**Chunk IDs** are `"{course_title_underscored}_{chunk_index}"`. Duplicate IDs cause a ChromaDB error, which is why the startup loader checks for existing titles before adding.

**Session history** is stored in memory only (lost on restart). `MAX_HISTORY = 2` means only the last 2 exchanges are passed to Claude as plain text in the system prompt — not as `messages` entries.
