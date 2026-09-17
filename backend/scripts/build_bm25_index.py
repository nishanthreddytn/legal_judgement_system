import sys
from pathlib import Path

# Add the project root to sys.path so we can import app modules
BASE = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE))

from app.services.bm25_search import _load_or_build_index

if __name__ == "__main__":
    print("Building BM25 Index...")
    _load_or_build_index()
    print("Done building BM25 Index!")
