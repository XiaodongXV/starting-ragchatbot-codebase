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


def _api_key() -> str:
    return os.getenv(PROVIDERS[_provider()][1], "")


@dataclass
class Config:
    """Configuration settings for the RAG system"""
    # LLM settings — provider is selected at startup via LLM_PROVIDER
    LLM_PROVIDER: str = field(default_factory=_provider)
    LLM_MODEL: str = field(default_factory=_model)
    LLM_API_KEY: str = field(default_factory=_api_key)

    # Embedding model settings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # Document processing settings
    CHUNK_SIZE: int = 800       # Size of text chunks for vector storage
    CHUNK_OVERLAP: int = 100     # Characters to overlap between chunks
    MAX_RESULTS: int = 5         # Maximum search results to return
    MAX_HISTORY: int = 2         # Number of conversation messages to remember

    # Database paths
    CHROMA_PATH: str = "./chroma_db"  # ChromaDB storage location

config = Config()
