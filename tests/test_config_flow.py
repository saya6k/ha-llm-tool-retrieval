"""Tests for the config and options flow."""

from __future__ import annotations

import aiohttp
import pytest
from homeassistant import config_entries
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType

from custom_components.llm_tool_retrieval.const import DOMAIN
from tests.common import FakeAPI, assist_tools, register
from tests.vendor.aiohttp_mock import AiohttpClientMocker
from tests.vendor.ha_common import MockConfigEntry

URL = "http://embedding:8080/v1"
EMBEDDING = {"data": [{"index": 0, "embedding": [0.6, 0.8]}]}
INPUT = {
    "name": "Voice tools",
    "source_apis": ["assist"],
    "embedding_url": URL,
    "top_k": 8.0,
    "pin_device_control": True,
    "always_include": ["save"],
    "query_prefix": "task: search result | query: ",
    "document_template": "title: {title} | text: {text}",
}


@pytest.fixture(autouse=True)
def apis(hass: HomeAssistant) -> None:
    register(hass, FakeAPI(hass, "assist", assist_tools()))


async def _submit(hass: HomeAssistant, data: dict) -> dict:
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    assert result["type"] is FlowResultType.FORM
    return await hass.config_entries.flow.async_configure(result["flow_id"], data)


async def test_user_flow_creates_an_entry(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker
) -> None:
    """A working embedding URL creates the entry; the name becomes its title."""
    aioclient_mock.post(f"{URL}/embeddings", json=EMBEDDING)

    result = await _submit(hass, INPUT)

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["title"] == "Voice tools"
    options = result["options"]
    assert "name" not in options
    assert options["top_k"] == 8
    assert options["always_include"] == ["save"]
    assert aioclient_mock.mock_calls[0][2]["input"] == ["task: search result | query: test"]


async def test_the_form_lists_registered_apis_but_not_its_own(hass: HomeAssistant) -> None:
    """Only other LLM APIs can be sources."""
    register(hass, FakeAPI(hass, f"{DOMAIN}__abc", []))
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    fields = {str(key): value for key, value in result["data_schema"].schema.items()}

    values = [option["value"] for option in fields["source_apis"].config["options"]]
    assert values == ["assist"]
    assert fields["api_key"].config["type"] == "password"


@pytest.mark.parametrize(
    ("response", "field", "error"),
    [
        ({"exc": aiohttp.ClientError("boom")}, "embedding_url", "cannot_connect"),
        ({"status": 500}, "embedding_url", "cannot_connect"),
        ({"status": 401}, "api_key", "invalid_auth"),
        ({"json": {"data": []}}, "embedding_url", "cannot_connect"),
    ],
)
async def test_user_flow_reports_embedding_errors(
    hass: HomeAssistant,
    aioclient_mock: AiohttpClientMocker,
    response: dict,
    field: str,
    error: str,
) -> None:
    """An unreachable service, a rejected key or an empty answer is shown on the form."""
    aioclient_mock.post(f"{URL}/embeddings", **response)

    result = await _submit(hass, INPUT)

    assert result["type"] is FlowResultType.FORM
    assert result["errors"] == {field: error}


@pytest.mark.parametrize("template", ["title: {title}", "{text} {other}"])
async def test_user_flow_rejects_a_bad_template(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker, template: str
) -> None:
    """The template needs {text} and may only use {title} and {text}."""
    aioclient_mock.post(f"{URL}/embeddings", json=EMBEDDING)

    result = await _submit(hass, {**INPUT, "document_template": template})

    assert result["errors"] == {"document_template": "invalid_template"}


async def test_options_flow_updates_the_settings(
    hass: HomeAssistant, aioclient_mock: AiohttpClientMocker
) -> None:
    """The options form starts from the current settings and saves the new ones."""
    aioclient_mock.post(f"{URL}/embeddings", json=EMBEDDING)
    options = {k: v for k, v in INPUT.items() if k != "name"}
    entry = MockConfigEntry(domain=DOMAIN, title="Voice tools", options=options)
    entry.add_to_hass(hass)

    result = await hass.config_entries.options.async_init(entry.entry_id)
    assert result["type"] is FlowResultType.FORM
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {**options, "top_k": 5.0, "always_include": []}
    )

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert entry.options["top_k"] == 5
    assert entry.options["always_include"] == []
