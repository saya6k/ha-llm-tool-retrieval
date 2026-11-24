"""Tests for choosing tools out of the source APIs."""

from __future__ import annotations

from typing import Any

import pytest
from homeassistant.core import HomeAssistant

from custom_components.llm_tool_retrieval.const import (
    CONF_ALWAYS_INCLUDE,
    CONF_PIN_DEVICE_CONTROL,
    CONF_SOURCE_APIS,
    CONF_TOP_K,
)
from custom_components.llm_tool_retrieval.index import ToolIndex
from custom_components.llm_tool_retrieval.retrieval import Retrieval
from tests.common import (
    FakeAPI,
    FakeEmbeddingClient,
    FakeTool,
    assist_tools,
    llm_context,
    other_tools,
    register,
)

URL = "http://embedding:8080/v1"


async def _retrieval(
    hass: HomeAssistant, client: FakeEmbeddingClient | None = None, **options: Any
) -> tuple[Retrieval, FakeEmbeddingClient]:
    client = client or FakeEmbeddingClient()
    index = ToolIndex(hass, "test", client)  # type: ignore[arg-type]
    await index.async_load(URL)
    options = {CONF_SOURCE_APIS: ["assist", "other"], CONF_TOP_K: 2, **options}
    return Retrieval(hass, options, index, client), client  # type: ignore[arg-type]


@pytest.fixture
def apis(hass: HomeAssistant) -> None:
    register(
        hass,
        FakeAPI(hass, "assist", assist_tools(), "Control devices."),
        FakeAPI(hass, "other", other_tools(), "Use the other tools."),
    )


async def _select(retrieval: Retrieval, query: str | None) -> tuple[list[str], str]:
    candidates, _ = await retrieval.async_gather(llm_context())
    await retrieval.async_refresh(candidates)
    selection = await retrieval.async_select(candidates, query)
    return [tool.name for tool in selection.tools], selection.reason


@pytest.mark.usefixtures("apis")
async def test_keeps_device_control_builtins_and_the_closest_tools(hass: HomeAssistant) -> None:
    """Built-in Device control tools stay; scripts and other tools are ranked."""
    retrieval, _ = await _retrieval(hass)

    names, reason = await _select(retrieval, "how much money did I spend on coffee")

    assert reason == "ranked"
    assert names == [
        "intent__HassTurnOn",
        "llm__GetDateTime",
        "script__morning_coffee",
        "get_transactions",
    ]


@pytest.mark.usefixtures("apis")
async def test_pinning_can_be_turned_off_and_tools_named(hass: HomeAssistant) -> None:
    """Without pinning only ranked tools and the named ones are offered."""
    retrieval, _ = await _retrieval(
        hass, **{CONF_PIN_DEVICE_CONTROL: False, CONF_ALWAYS_INCLUDE: ["save"], CONF_TOP_K: 1}
    )

    names, _ = await _select(retrieval, "turn on the light")

    assert names == ["intent__HassTurnOn", "save"]


@pytest.mark.usefixtures("apis")
async def test_offers_everything_without_a_request(hass: HomeAssistant) -> None:
    """Outside a conversation (no user text) nothing is filtered."""
    retrieval, _ = await _retrieval(hass)

    names, reason = await _select(retrieval, None)

    assert reason == "no_query"
    assert len(names) == 8


@pytest.mark.usefixtures("apis")
async def test_offers_everything_when_the_embedding_service_is_down(
    hass: HomeAssistant,
) -> None:
    """A failed query embedding fails open instead of leaving the agent toolless."""
    retrieval, client = await _retrieval(hass)
    candidates, _ = await retrieval.async_gather(llm_context())
    client.fail = True

    selection = await retrieval.async_select(candidates, "turn on the light")

    assert selection.reason == "embedding_unavailable"
    assert len(selection.tools) == 8


@pytest.mark.usefixtures("apis")
async def test_offers_tools_not_embedded_yet(hass: HomeAssistant) -> None:
    """Before the index has a tool, the tool is offered rather than hidden."""
    retrieval, _ = await _retrieval(hass)
    candidates, _ = await retrieval.async_gather(llm_context())

    selection = await retrieval.async_select(candidates, "turn on the light")

    assert selection.reason == "ranked"
    assert len(selection.tools) == 8


@pytest.mark.usefixtures("apis")
async def test_embeds_each_tool_once(hass: HomeAssistant) -> None:
    """A second index pass embeds nothing new; pinned tools are never embedded."""
    retrieval, client = await _retrieval(hass)
    candidates, _ = await retrieval.async_gather(llm_context())

    await retrieval.async_refresh(candidates)
    first = len(client.embedded)
    await retrieval.async_refresh(candidates)

    assert first == 6
    assert len(client.embedded) == first
    assert not any("Turns on a light" in text for text in client.embedded)


async def test_namespaces_a_name_two_apis_share(hass: HomeAssistant) -> None:
    """Like Home Assistant's own merge, a shared name is prefixed with each API id."""
    register(
        hass,
        FakeAPI(hass, "assist", [FakeTool("search", "Search devices.")]),
        FakeAPI(hass, "other", [FakeTool("search", "Search memory.")]),
    )
    retrieval, _ = await _retrieval(hass)

    candidates, _ = await retrieval.async_gather(llm_context())

    assert sorted(c.tool.name for c in candidates) == ["assist__search", "other__search"]


async def test_skips_missing_and_admin_only_sources(hass: HomeAssistant) -> None:
    """A source that is gone or not allowed for this user drops out on its own."""
    register(
        hass,
        FakeAPI(hass, "assist", assist_tools()),
        FakeAPI(hass, "admin", [FakeTool("reload", "Reload.")], requires_admin=True),
    )
    retrieval, _ = await _retrieval(hass, **{CONF_SOURCE_APIS: ["assist", "admin", "gone"]})

    candidates, instances = await retrieval.async_gather(llm_context())

    assert list(instances) == ["assist"]
    assert len(candidates) == 3


@pytest.mark.usefixtures("apis")
async def test_prompt_covers_only_apis_with_selected_tools(hass: HomeAssistant) -> None:
    """Instructions for an API none of whose tools were picked are left out."""
    retrieval, _ = await _retrieval(hass, **{CONF_TOP_K: 1})
    candidates, instances = await retrieval.async_gather(llm_context())
    await retrieval.async_refresh(candidates)
    selection = await retrieval.async_select(candidates, "brew coffee")
    api = FakeAPI(hass, "retrieval", [])

    instance = retrieval.build_instance(api, llm_context(), instances, candidates, selection)

    assert "Control devices." in instance.api_prompt
    assert "Use the other tools." not in instance.api_prompt


@pytest.mark.usefixtures("apis")
async def test_summaries_are_embedded_with_the_description(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A summary in the user's language is put in front of the description."""

    async def write(hass: HomeAssistant, entity_id: str, tools: dict[str, str]) -> dict[str, str]:
        return {"forecast": "날씨 weather 예보"}

    monkeypatch.setattr(
        "custom_components.llm_tool_retrieval.retrieval.async_write_summaries", write
    )
    retrieval, client = await _retrieval(hass, ai_task_entity="ai_task.test")
    candidates, _ = await retrieval.async_gather(llm_context())

    await retrieval.async_refresh(candidates)

    assert "title: forecast | text: 날씨 weather 예보 Weather forecast." in client.embedded
