import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# provider -> (default model, env var holding that provider's API key)
PROVIDERS = {
    "anthropic": ("anthropic/claude-sonnet-4-20250514", "ANTHROPIC_API_KEY"),
    "gemini": ("gemini/gemini-3.5-flash", "GOOGLE_API_KEY"),
    "openai": ("openai/gpt-4o-mini", "OPENAI_API_KEY"),
}


def _provider() -> str:
    name = os.getenv("LLM_PROVIDER", "anthropic").strip().lower()
    if name not in PROVIDERS:
        raise ValueError(
            f"Unknown LLM_PROVIDER {name!r}. Supported: {', '.join(PROVIDERS)}"
        )
    return name


def _model() -> str:
    return os.getenv("LLM_MODEL", "").strip() or PROVIDERS[_provider()][0]


def _fallback_models() -> list:
    # Comma-separated LiteLLM model names tried in order when the primary
    # model fails (e.g. a free-tier 429 quota error or a 503 overload)
    raw = os.getenv("LLM_FALLBACK_MODELS", "")
    return [m.strip() for m in raw.split(",") if m.strip()]


def _api_key() -> str:
    return os.getenv(PROVIDERS[_provider()][1], "")


@dataclass
class Config:
    """Configuration settings for the RAG system"""
    # LLM settings — provider is selected at startup via LLM_PROVIDER
    LLM_PROVIDER: str = field(default_factory=_provider)
    LLM_MODEL: str = field(default_factory=_model)
    LLM_FALLBACK_MODELS: list = field(default_factory=_fallback_models)
    LLM_API_KEY: str = field(default_factory=_api_key)

    # Embedding model settings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # Document processing settings
    CHUNK_SIZE: int = 800       # Size of text chunks for vector storage
    CHUNK_OVERLAP: int = 100     # Characters to overlap between chunks
    MAX_RESULTS: int = 5         # Maximum search results to return
    # Open-ended searches drop chunks farther than this (L2, all-MiniLM-L6-v2) from the query.
    # Measured on the real index: on-topic chunks sit at 0.5-1.1, loosely related ones (e.g. "tool
    # use" for an MCP question) at ~0.9-1.15, off-topic ones at 1.24+ ("Italian pasta" = 1.62).
    MAX_RESULT_DISTANCE: float = 1.2
    MAX_HISTORY: int = 2         # Number of conversation messages to remember

    # Database paths
    CHROMA_PATH: str = "./chroma_db"  # ChromaDB storage location

config = Config()
