"""End-to-end tests: the entry's API inside a real chat log."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from homeassistant.components import conversation
from homeassistant.core import Context, HomeAssistant
from homeassistant.helpers import chat_session, llm
from homeassistant.setup import async_setup_component

from custom_components.llm_tool_retrieval.const import (
    CONF_EMBEDDING_URL,
    CONF_SOURCE_APIS,
    CONF_TOP_K,
    DOMAIN,
)
from tests.common import (
    FakeAPI,
    FakeEmbeddingClient,
    assist_tools,
    llm_context,
    other_tools,
    register,
)
from tests.vendor.ha_common import MockConfigEntry

OPTIONS = {
    CONF_SOURCE_APIS: ["assist", "other"],
    CONF_EMBEDDING_URL: "http://embedding:8080/v1",
    CONF_TOP_K: 1,
}


@pytest.fixture
async def client(hass: HomeAssistant) -> FakeEmbeddingClient:
    # Not the conversation component itself: it would register the real
    # Device control API under the id the fake below uses.
    assert await async_setup_component(hass, "homeassistant", {})
    register(
        hass,
        FakeAPI(hass, "assist", assist_tools(), "Control devices."),
        FakeAPI(hass, "other", other_tools(), "Use the other tools."),
    )
    return FakeEmbeddingClient()


async def _setup(hass: HomeAssistant, client: FakeEmbeddingClient) -> MockConfigEntry:
    entry = MockConfigEntry(domain=DOMAIN, title="Voice tools", options=OPTIONS)
    entry.add_to_hass(hass)
    with patch("custom_components.llm_tool_retrieval.EmbeddingClient", return_value=client):
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()
    return entry


async def test_setup_registers_the_api_and_indexes_its_tools(
    hass: HomeAssistant, client: FakeEmbeddingClient
) -> None:
    """The entry's API is listed under its title, and its tools are embedded at startup."""
    entry = await _setup(hass, client)

    apis = {api.id: api.name for api in llm.async_get_apis(hass)}
    assert apis[f"{DOMAIN}__{entry.entry_id}"] == "Voice tools"
    assert len(client.embedded) == 6


async def test_unload_unregisters_the_api(
    hass: HomeAssistant, client: FakeEmbeddingClient
) -> None:
    """Once unloaded, agents can no longer select the API."""
    entry = await _setup(hass, client)

    assert await hass.config_entries.async_unload(entry.entry_id)

    assert f"{DOMAIN}__{entry.entry_id}" not in {api.id for api in llm.async_get_apis(hass)}


async def test_a_conversation_gets_the_tools_its_request_needs(
    hass: HomeAssistant, client: FakeEmbeddingClient
) -> None:
    """The user's words in the chat log pick the tools, and a tool result passes through
    untouched, so cards a tool returns still reach the satellite."""
    entry = await _setup(hass, client)
    user_input = conversation.ConversationInput(
        text="play some music",
        context=Context(),
        conversation_id=None,
        device_id=None,
        satellite_id=None,
        language="en",
        agent_id="conversation.test",
    )

    with (
        chat_session.async_get_chat_session(hass, context=None) as session,
        conversation.async_get_chat_log(hass, session, user_input) as chat_log,
    ):
        await chat_log.async_provide_llm_data(llm_context(), f"{DOMAIN}__{entry.entry_id}")
        assert chat_log.llm_api is not None
        names = [tool.name for tool in chat_log.llm_api.tools]
        result = await chat_log.llm_api.async_call_tool(
            llm.ToolInput(tool_name="play_music", tool_args={})
        )

    assert names == ["intent__HassTurnOn", "llm__GetDateTime", "play_music"]
    assert result.data == {"tool": "play_music", "card": {"type": "markdown"}}


async def test_preview_service_shows_the_ranking(
    hass: HomeAssistant, client: FakeEmbeddingClient
) -> None:
    """The preview service lists pinned tools and every ranked tool with its score."""
    entry = await _setup(hass, client)

    response = await hass.services.async_call(
        DOMAIN,
        "preview",
        {"config_entry_id": entry.entry_id, "text": "show my money"},
        blocking=True,
        return_response=True,
    )

    assert response is not None
    assert response["reason"] == "ranked"
    assert response["pinned"] == ["intent__HassTurnOn", "llm__GetDateTime"]
    top = response["ranked"][0]
    assert top["tool"] == "get_transactions"
    assert top["selected"] is True
    assert len(response["ranked"]) == 6
