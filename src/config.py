"""Central config. Reads from .env so secrets never live in code."""
import os
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "clinical_policies")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Retrieval settings
TOP_K = 7            # how many chunks to retrieve per query
CHUNK_SIZE = 800     # characters per chunk
CHUNK_OVERLAP = 150  # overlap between chunks (helps preserve context at boundaries)
