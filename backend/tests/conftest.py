"""Shared fixtures for the backend test-suite.

Layout of the system under test (see CLAUDE.md):
    RAGSystem -> AIGenerator (litellm) -> ToolManager -> CourseSearchTool -> VectorStore (Chroma)
"""
# Zscaler re-signs TLS; litellm/huggingface fail at import without this (CLAUDE.md).
import truststore
truststore.inject_into_ssl()

import shutil
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from models import Course, CourseChunk, Lesson  # noqa: E402
from vector_store import SearchResults, VectorStore  # noqa: E402

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

COMPUTER_USE = "Building Towards Computer Use with Anthropic"
MCP = "MCP: Build Rich-Context AI Apps with Anthropic"

# ---- seed data: two tiny courses; COMPUTER_USE has a "Lesson 0" like the real docs ----
SEED_COURSES = [
    Course(
        title=COMPUTER_USE,
        course_link="https://example.com/computer-use",
        instructor="Colt Steele",
        lessons=[
            Lesson(lesson_number=0, title="Introduction", lesson_link="https://example.com/cu/0"),
            Lesson(lesson_number=1, title="Overview of model capabilities", lesson_link="https://example.com/cu/1"),
        ],
    ),
    Course(
        title=MCP,
        course_link="https://example.com/mcp",
        instructor="Elie Schoppik",
        lessons=[
            Lesson(lesson_number=1, title="Why MCP", lesson_link="https://example.com/mcp/1"),
        ],
    ),
]

SEED_CHUNKS = [
    CourseChunk(
        content=f"Course {COMPUTER_USE} Lesson 0 content: Computer use lets the model look at a screenshot "
                "and generate mouse clicks and keystrokes to operate a computer.",
        course_title=COMPUTER_USE, lesson_number=0, chunk_index=0),
    CourseChunk(
        content=f"Course {COMPUTER_USE} Lesson 1 content: Claude models support multimodal requests, "
                "so you can send images and ask questions about them.",
        course_title=COMPUTER_USE, lesson_number=1, chunk_index=1),
    CourseChunk(
        content=f"Course {MCP} Lesson 1 content: The Model Context Protocol standardises how "
                "applications expose tools and resources to language models.",
        course_title=MCP, lesson_number=1, chunk_index=0),
]


def _seed(store: VectorStore) -> VectorStore:
    for course in SEED_COURSES:
        store.add_course_metadata(course)
    store.add_course_content(SEED_CHUNKS)
    return store


@pytest.fixture(scope="session")
def tmp_store(tmp_path_factory) -> VectorStore:
    """Real VectorStore (real embeddings, real Chroma) on a throw-away dir, seeded with 2 tiny courses.
    Session-scoped (embedding model load is slow) — treat as read-only in tests."""
    path = tmp_path_factory.mktemp("chroma_seed")
    return _seed(VectorStore(str(path), EMBEDDING_MODEL, max_results=5))


@pytest.fixture
def real_db_copy(tmp_path) -> str:
    """Copy of the real backend/chroma_db so live tests never mutate the real index."""
    src = BACKEND_DIR / "chroma_db"
    if not src.exists():
        pytest.skip("backend/chroma_db does not exist")
    dst = tmp_path / "chroma_copy"
    shutil.copytree(src, dst)
    return str(dst)


@pytest.fixture
def sample_results() -> SearchResults:
    return SearchResults(
        documents=["chunk A about computer use", "chunk B about computer use", "chunk C about MCP"],
        metadata=[
            {"course_title": COMPUTER_USE, "lesson_number": 1, "chunk_index": 0},
            {"course_title": COMPUTER_USE, "lesson_number": 1, "chunk_index": 1},
            {"course_title": MCP, "lesson_number": 2, "chunk_index": 0},
        ],
        distances=[0.1, 0.2, 0.3],
    )


@pytest.fixture
def mock_store(sample_results) -> MagicMock:
    """VectorStore stand-in: search() returns sample_results, lesson links resolve to a URL."""
    store = MagicMock(spec=VectorStore)
    store.search.return_value = sample_results
    store.get_lesson_link.side_effect = lambda title, n: f"https://example.com/{n}"
    return store
