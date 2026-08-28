from __future__ import annotations

import json
import os
import re
import threading
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
import requests


class OllamaError(RuntimeError):
    pass


# Exact out-of-scope message requested.
# If you want to change wording, change only this template.
OUT_OF_SCOPE_TEMPLATE = (
    "I can only respond to question on the specific of the documents {names}"
)


def out_of_scope_message(document_names: List[str]) -> str:
    names = ", ".join(document_names) if document_names else "no documents"
    return OUT_OF_SCOPE_TEMPLATE.format(names=names)


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, max_chars: int = 900, overlap: int = 120) -> List[str]:
    text = _normalize_text(text)
    if not text:
        return []

    max_chars = max(100, int(max_chars))
    overlap = max(0, min(int(overlap), max_chars // 2))
    chunks: List[str] = []

    # Prefer splitting on blank lines first.
    segments = re.split(r"\n\s*\n", text)
    buffer = ""

    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue

        # If buffer + segment fits, keep appending.
        if len(buffer) + len(segment) + 2 <= max_chars:
            buffer = f"{buffer}\n\n{segment}".strip()
            continue

        # Otherwise flush buffer.
        if buffer:
            chunks.append(buffer)

        # If a single segment is still too long, hard split it.
        if len(segment) > max_chars:
            start = 0
            while start < len(segment):
                end = min(start + max_chars, len(segment))
                chunk = segment[start:end].strip()
                if chunk:
                    chunks.append(chunk)
                if end == len(segment):
                    break
                start = max(start + overlap, end - overlap)
            buffer = ""
        else:
            buffer = segment

    if buffer:
        chunks.append(buffer)

    # Final safety pass.
    final_chunks: List[str] = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            final_chunks.append(chunk)
            continue

        start = 0
        while start < len(chunk):
            end = min(start + max_chars, len(chunk))
            part = chunk[start:end].strip()
            if part:
                final_chunks.append(part)
            if end == len(chunk):
                break
            start = max(start + overlap, end - overlap)

    return [chunk for chunk in final_chunks if chunk]


class LocalRag:
    def __init__(
        self,
        data_dir: Path,
        base_url: str,
        llm_model: str,
        embed_model: str,
        chunk_max_chars: int = 900,
        chunk_overlap: int = 120,
        top_k: int = 5,
        min_score: float = 0.25,
        embed_batch_size: int = 32,
    ):
        self._data_dir = Path(data_dir)
        self._base_url = base_url.rstrip("/")
        self._llm_model = llm_model
        self._embed_model = embed_model
        self._chunk_max_chars = int(chunk_max_chars)
        self._chunk_overlap = int(chunk_overlap)
        self._top_k = int(top_k)
        self._min_score = float(min_score)
        self._embed_batch_size = int(embed_batch_size)

        self._lock = threading.RLock()
        self._chunks: List[Dict[str, Any]] = []
        self._vectors: np.ndarray = np.empty((0, 0), dtype="float32")
        self._index: Optional[faiss.Index] = None

        self._load_state()

    # -----------------------------
    # persistence helpers
    # -----------------------------

    def _meta_path(self) -> Path:
        return self._data_dir / "meta.json"

    def _chunks_path(self) -> Path:
        return self._data_dir / "chunks.json"

    def _vectors_path(self) -> Path:
        return self._data_dir / "vectors.npy"

    def _index_path(self) -> Path:
        return self._data_dir / "index.faiss"

    def _clear_state(self) -> None:
        self._chunks = []
        self._vectors = np.empty((0, 0), dtype="float32")
        self._index = None

        self._data_dir.mkdir(parents=True, exist_ok=True)

        for path in (
            self._meta_path(),
            self._chunks_path(),
            self._vectors_path(),
            self._index_path(),
        ):
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    def _save_state(self) -> None:
        if not self._chunks:
            self._clear_state()
            return

        self._data_dir.mkdir(parents=True, exist_ok=True)

        if self._index is None:
            self._clear_state()
            return

        faiss.write_index(self._index, str(self._index_path()))
        self._chunks_path().write_text(
            json.dumps(self._chunks, ensure_ascii=False),
            encoding="utf-8",
        )
        np.save(str(self._vectors_path()), self._vectors)
        self._meta_path().write_text(
            json.dumps(
                {
                    "embedding_model": self._embed_model,
                    "dim": int(self._vectors.shape[1]),
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def _load_state(self) -> None:
        self._chunks = []
        self._vectors = np.empty((0, 0), dtype="float32")
        self._index = None

        paths = (
            self._meta_path(),
            self._chunks_path(),
            self._vectors_path(),
            self._index_path(),
        )

        if not all(path.exists() for path in paths):
            return

        try:
            meta = json.loads(self._meta_path().read_text(encoding="utf-8"))
            if meta.get("embedding_model") != self._embed_model:
                raise ValueError("Embedding model changed")

            self._chunks = json.loads(self._chunks_path().read_text(encoding="utf-8"))
            self._vectors = np.load(str(self._vectors_path()))
            self._index = faiss.read_index(str(self._index_path()))

            if self._vectors.ndim != 2:
                raise ValueError("Vector file is invalid")
            if self._index.d != self._vectors.shape[1]:
                raise ValueError("Vector dimension mismatch")
            if self._index.ntotal != len(self._chunks):
                raise ValueError("Chunk/vector count mismatch")
        except Exception:
            self._clear_state()

    def _rebuild_index(self) -> None:
        if not self._chunks:
            self._index = None
            return

        self._vectors = np.ascontiguousarray(self._vectors, dtype="float32")

        if self._vectors.size == 0:
            self._index = None
            return

        faiss.normalize_L2(self._vectors)
        self._index = faiss.IndexFlatIP(self._vectors.shape[1])
        self._index.add(self._vectors)

    # -----------------------------
    # Ollama helpers
    # -----------------------------

    def _embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype="float32")

        embeddings_out: List[List[float]] = []

        for i in range(0, len(texts), self._embed_batch_size):
            batch = texts[i : i + self._embed_batch_size]

            try:
                resp = requests.post(
                    f"{self._base_url}/api/embed",
                    json={
                        "model": self._embed_model,
                        "input": batch,
                    },
                    timeout=300,
                )
            except requests.RequestException as exc:
                raise OllamaError(f"Ollama embedding request failed: {exc}")

            if resp.status_code != 200:
                raise OllamaError(f"Ollama embedding failed: {resp.text}")

            data = resp.json()
            embeddings = data.get("embeddings")

            if not isinstance(embeddings, list):
                raise OllamaError("Unexpected Ollama embedding response")

            # Some responses may return a single vector for a single input.
            if len(batch) == 1 and embeddings and not isinstance(embeddings[0], list):
                embeddings = [embeddings]

            if len(embeddings) != len(batch):
                raise OllamaError("Unexpected Ollama embedding response length")

            embeddings_out.extend(embeddings)

        arr = np.asarray(embeddings_out, dtype="float32")
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        return np.ascontiguousarray(arr, dtype="float32")

    # -----------------------------
    # public API
    # -----------------------------

    def document_names(self) -> List[str]:
        with self._lock:
            names: List[str] = []
            seen = set()
            for chunk in self._chunks:
                name = chunk["doc"]
                if name not in seen:
                    seen.add(name)
                    names.append(name)
            return names

    def clear(self) -> None:
        with self._lock:
            self._clear_state()

    def add_document(self, name: str, text: str) -> List[str]:
        with self._lock:
            name = os.path.basename(name) or "document.txt"

            # Replace existing document if the same name is uploaded again.
            self._remove_document(name)

            chunks = chunk_text(text, self._chunk_max_chars, self._chunk_overlap)
            if not chunks:
                return []

            vectors = self._embed(chunks)

            if self._vectors.size == 0:
                self._vectors = vectors
            else:
                self._vectors = np.vstack((self._vectors, vectors))

            start_id = len(self._chunks)
            for i, chunk in enumerate(chunks):
                self._chunks.append({
                    "doc": name,
                    "id": start_id + i,
                    "text": chunk,
                })

            self._rebuild_index()
            self._save_state()
            return chunks

    def search(self, question: str):
        with self._lock:
            if not self._chunks or self._index is None:
                return []

            qvec = self._embed([question])
            qvec = np.ascontiguousarray(qvec, dtype="float32")
            faiss.normalize_L2(qvec)

            k = min(self._top_k, self._index.ntotal)
            scores, indices = self._index.search(
                np.array([qvec[0]], dtype="float32"),
                k,
            )

            results = []
            for score, idx in zip(scores[0], indices[0]):
                idx = int(idx)
                if idx < 0 or idx >= len(self._chunks):
                    continue

                score = float(score)
                if score < self._min_score:
                    continue

                results.append((score, self._chunks[idx]))

            return results

    def answer(self, question: str) -> str:
        with self._lock:
            doc_names = self.document_names()

            if not doc_names:
                return out_of_scope_message(doc_names)

            results = self.search(question)
            if not results:
                return out_of_scope_message(doc_names)

            context_blocks = []
            for _score, chunk in results:
                context_blocks.append(
                    f"[Document: {chunk['doc']}]\n{chunk['text']}"
                )
            context = "\n\n".join(context_blocks)

            out_of_scope = out_of_scope_message(doc_names)

            system = f"""You are a strict document-only RAG assistant.
You must answer ONLY using the CONTEXT below.
If the CONTEXT does not contain enough information to answer the QUESTION, reply with exactly:
{out_of_scope}
When replying with the out-of-scope message, do not add any extra words, punctuation, or explanation.
Never use outside knowledge.
"""

            user = f"""CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

            try:
                resp = requests.post(
                    f"{self._base_url}/api/chat",
                    json={
                        "model": self._llm_model,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        "stream": False,
                        "options": {
                            "temperature": 0.0,
                            "num_ctx": 8192,
                        },
                    },
                    timeout=600,
                )
            except requests.RequestException as exc:
                raise OllamaError(f"Ollama chat request failed: {exc}")

            if resp.status_code != 200:
                raise OllamaError(f"Ollama chat failed: {resp.text}")

            data = resp.json()
            answer = (data.get("message", {}).get("content") or "").strip()

            if not answer:
                return out_of_scope

            # Normalize common out-of-scope variants to the exact required message.
            if answer.lower().startswith("i can only respond"):
                return out_of_scope

            return answer

    # -----------------------------
    # internal helpers
    # -----------------------------

    def _remove_document(self, name: str) -> None:
        if not self._chunks:
            return

        keep_mask = np.array(
            [chunk["doc"] != name for chunk in self._chunks],
            dtype=bool,
        )

        if bool(keep_mask.all()):
            return

        self._chunks = [
            chunk
            for chunk, keep in zip(self._chunks, keep_mask)
            if keep
        ]

        if keep_mask.size:
            self._vectors = self._vectors[keep_mask]
        else:
            dim = self._vectors.shape[1] if self._vectors.ndim == 2 else 0
            self._vectors = np.empty((0, dim), dtype="float32")

        if not self._chunks:
            self._clear_state()
        else:
            self._rebuild_index()
            self._save_state()
