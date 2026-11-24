"""Client for an OpenAI-compatible /embeddings endpoint."""

from __future__ import annotations

import asyncio
import math

import aiohttp

from .const import EMBED_BATCH


class EmbeddingError(Exception):
    """The embedding service could not be reached or returned something unusable."""


class EmbeddingAuthError(EmbeddingError):
    """The embedding service rejected the API key."""


class EmbeddingClient:
    """Embeds texts through `POST <base_url>/embeddings`."""

    def __init__(self, session: aiohttp.ClientSession, base_url: str, api_key: str | None) -> None:
        """Initialize with the API base (including /v1) and an optional key."""
        self._session = session
        self._url = base_url.rstrip("/") + "/embeddings"
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    async def embed(self, texts: list[str], seconds: float) -> list[list[float]]:
        """Return one unit-length vector per text, in order."""
        vectors: list[list[float]] = []
        async with asyncio.timeout(seconds):
            for start in range(0, len(texts), EMBED_BATCH):
                vectors.extend(await self._embed_batch(texts[start : start + EMBED_BATCH]))
        return vectors

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        try:
            async with self._session.post(
                self._url,
                json={"input": texts, "model": "default"},
                headers=self._headers,
            ) as resp:
                if resp.status in (401, 403):
                    raise EmbeddingAuthError(f"HTTP {resp.status}")
                if resp.status != 200:
                    raise EmbeddingError(f"HTTP {resp.status}: {(await resp.text())[:200]}")
                body = await resp.json()
        except aiohttp.ClientError as err:
            raise EmbeddingError(str(err)) from err

        try:
            items = sorted(body["data"], key=lambda item: item["index"])
            vectors = [_normalize(item["embedding"]) for item in items]
        except (KeyError, TypeError) as err:
            raise EmbeddingError(f"unexpected response: {err}") from err
        if len(vectors) != len(texts):
            raise EmbeddingError(f"asked for {len(texts)} vectors, got {len(vectors)}")
        return vectors


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0:
        raise EmbeddingError("zero vector")
    return [x / norm for x in vector]


def dot(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two unit vectors."""
    return sum(x * y for x, y in zip(a, b, strict=True))
