from pathlib import Path
import json
import numpy as np
from sentence_transformers import SentenceTransformer

CASES = Path("data/processed/cases.json")
OUT = Path("data/embeddings")
OUT.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

cases = json.loads(CASES.read_text(encoding="utf-8"))

texts = []

for c in cases:
    text = " ".join([
        c.get("title", ""),
        c.get("summary", ""),
        c.get("facts", ""),
        c.get("judgment", ""),
    ])

    texts.append(text[:12000])

print(f"Loading model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)

print(f"Creating embeddings for {len(texts)} cases...")

embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True,
)

embeddings = np.asarray(embeddings, dtype=np.float32)

np.save(OUT / "embeddings.npy", embeddings)

print("Saved:", OUT / "embeddings.npy")
print("Shape:", embeddings.shape)