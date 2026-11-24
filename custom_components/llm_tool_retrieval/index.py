"""Embedding cache for tool descriptions, persisted across restarts.

A tool is embedded from a text built out of its name and description (and,
when one has been written, a summary in the user's language). Vectors are
keyed by a hash of that text, so a tool is re-embedded exactly when what is
embedded changes, and a new tool costs one embedding the first time it is seen.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.helpers.storage import Store

from .const import DOMAIN, INDEX_TIMEOUT, STORAGE_VERSION
from .embedding import EmbeddingClient

_LOGGER = logging.getLogger(__name__)

SAVE_DELAY = 10


def text_key(text: str) -> str:
    """Return the cache key for a text."""
    return hashlib.sha256(text.encode()).hexdigest()


class ToolIndex:
    """Vectors for embedded texts and summaries for tool descriptions."""

    def __init__(self, hass: HomeAssistant, entry_id: str, client: EmbeddingClient) -> None:
        """Initialize an empty index; call async_load before use."""
        self._client = client
        self._store: Store[dict[str, Any]] = Store(
            hass, STORAGE_VERSION, f"{DOMAIN}.{entry_id}", private=True
        )
        self._vectors: dict[str, list[float]] = {}
        self._summaries: dict[str, str] = {}
        self._lock = asyncio.Lock()

    async def async_load(self, embedding_url: str) -> None:
        """Load the cache. Vectors from another embedding service are dropped."""
        data = await self._store.async_load() or {}
        if data.get("embedding_url") == embedding_url:
            self._vectors = data.get("vectors", {})
        self._summaries = data.get("summaries", {})
        self._embedding_url = embedding_url

    def vector(self, text: str) -> list[float] | None:
        """Return the cached vector for a text, if any."""
        return self._vectors.get(text_key(text))

    def summary(self, source: str) -> str | None:
        """Return the summary written for a tool description, if any."""
        return self._summaries.get(text_key(source))

    def add_summaries(self, summaries: dict[str, str]) -> None:
        """Remember summaries, keyed by the source text (name and description) they summarise."""
        for source, summary in summaries.items():
            self._summaries[text_key(source)] = summary
        self._store.async_delay_save(self._data, SAVE_DELAY)

    async def async_index(self, texts: list[str]) -> None:
        """Embed every text that has no vector yet."""
        async with self._lock:
            missing = list(dict.fromkeys(t for t in texts if text_key(t) not in self._vectors))
            if not missing:
                return
            vectors = await self._client.embed(missing, INDEX_TIMEOUT)
            for text, vector in zip(missing, vectors, strict=True):
                self._vectors[text_key(text)] = [round(x, 6) for x in vector]
            _LOGGER.debug("Embedded %d tool descriptions", len(missing))
            self._store.async_delay_save(self._data, SAVE_DELAY)

    def _data(self) -> dict[str, Any]:
        return {
            "embedding_url": self._embedding_url,
            "vectors": self._vectors,
            "summaries": self._summaries,
        }
