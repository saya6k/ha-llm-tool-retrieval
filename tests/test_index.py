"""Tests for the embedding cache and the AI Task summaries."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from homeassistant.core import HomeAssistant

from custom_components.llm_tool_retrieval.index import ToolIndex
from custom_components.llm_tool_retrieval.summaries import _parse, async_write_summaries
from tests.common import FakeEmbeddingClient

URL = "http://embedding:8080/v1"


async def test_vectors_survive_a_restart(hass: HomeAssistant) -> None:
    """A reloaded index has the vectors and summaries the last one saved."""
    client = FakeEmbeddingClient()
    index = ToolIndex(hass, "entry", client)  # type: ignore[arg-type]
    await index.async_load(URL)
    await index.async_index(["title: a | text: coffee"])
    index.add_summaries({"a\ncoffee": "커피"})
    await index._store.async_save(index._data())

    reloaded = ToolIndex(hass, "entry", FakeEmbeddingClient())  # type: ignore[arg-type]
    await reloaded.async_load(URL)

    assert reloaded.vector("title: a | text: coffee") is not None
    assert reloaded.summary("a\ncoffee") == "커피"


async def test_another_embedding_service_starts_over(hass: HomeAssistant) -> None:
    """Vectors from a different service are not comparable, so they are dropped."""
    index = ToolIndex(hass, "entry", FakeEmbeddingClient())  # type: ignore[arg-type]
    await index.async_load(URL)
    await index.async_index(["title: a | text: coffee"])
    index.add_summaries({"a\ncoffee": "커피"})
    await index._store.async_save(index._data())

    moved = ToolIndex(hass, "entry", FakeEmbeddingClient())  # type: ignore[arg-type]
    await moved.async_load("http://elsewhere:8080/v1")

    assert moved.vector("title: a | text: coffee") is None
    assert moved.summary("a\ncoffee") == "커피"


@pytest.mark.parametrize(
    "answer",
    [
        {"a": "하나"},
        '{"a": "하나"}',
        '```json\n{"a": "하나"}\n```',
        'Here you go:\n{"a": "하나"}',
    ],
)
def test_parse_accepts_what_models_return(answer: object) -> None:
    """A dict, bare JSON, or JSON in a fence or after a preamble all parse."""
    assert _parse(answer) == {"a": "하나"}


async def test_summaries_skip_a_failed_batch(hass: HomeAssistant) -> None:
    """A batch whose answer cannot be used is skipped; the others still count."""
    tools = {f"tool{i}": f"Description {i}" for i in range(30)}
    answers = [
        SimpleNamespace(data={f"tool{i}": f"요약 {i}" for i in range(25)}),
        SimpleNamespace(data="not json"),
    ]
    with patch(
        "homeassistant.components.ai_task.async_generate_data",
        AsyncMock(side_effect=answers),
    ) as generate:
        written = await async_write_summaries(hass, "ai_task.test", tools)

    assert generate.await_count == 2
    assert len(written) == 25
    assert written["tool0"] == "요약 0"
