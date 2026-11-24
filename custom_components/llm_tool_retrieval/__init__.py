"""LLM Tool Retrieval: offer a conversation agent only the tools a request needs."""

from __future__ import annotations

from typing import Any

import voluptuous as vol
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import (
    HomeAssistant,
    ServiceCall,
    ServiceResponse,
    SupportsResponse,
    callback,
)
from homeassistant.exceptions import ServiceValidationError
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers import llm
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.start import async_at_started
from homeassistant.helpers.typing import ConfigType

from .api import RetrievalAPI
from .const import (
    ATTR_CONFIG_ENTRY_ID,
    ATTR_TEXT,
    CONF_API_KEY,
    CONF_EMBEDDING_URL,
    DOMAIN,
    SERVICE_PREVIEW,
)
from .embedding import EmbeddingClient
from .index import ToolIndex
from .retrieval import Retrieval

type RetrievalConfigEntry = ConfigEntry[RetrievalAPI]

CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)

PREVIEW_SCHEMA = vol.Schema(
    {vol.Required(ATTR_CONFIG_ENTRY_ID): cv.string, vol.Required(ATTR_TEXT): cv.string}
)


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Register the preview service."""

    async def preview(call: ServiceCall) -> ServiceResponse:
        entry: RetrievalConfigEntry | None = hass.config_entries.async_get_entry(
            call.data[ATTR_CONFIG_ENTRY_ID]
        )
        if entry is None or entry.domain != DOMAIN or not hasattr(entry, "runtime_data"):
            raise ServiceValidationError(
                translation_domain=DOMAIN, translation_key="entry_not_loaded"
            )
        return await _async_preview(hass, entry.runtime_data, call)

    hass.services.async_register(
        DOMAIN,
        SERVICE_PREVIEW,
        preview,
        schema=PREVIEW_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )
    return True


async def _async_preview(
    hass: HomeAssistant, api: RetrievalAPI, call: ServiceCall
) -> dict[str, Any]:
    """Show which tools a request would get, with the scores of the ranked ones."""
    retrieval = api.retrieval
    candidates, _ = await retrieval.async_gather(_context(hass, call))
    await retrieval.async_refresh(candidates)
    selection = await retrieval.async_select(candidates, call.data[ATTR_TEXT])
    chosen = {tool.name for tool in selection.tools}
    return {
        "reason": selection.reason,
        "available": len(candidates),
        "selected": len(selection.tools),
        "pinned": [c.tool.name for c in candidates if c.pinned],
        "ranked": [
            {"tool": name, "score": round(score, 4), "selected": name in chosen}
            for name, score in sorted(
                selection.scores.items(), key=lambda item: item[1], reverse=True
            )
        ],
    }


def _context(hass: HomeAssistant, call: ServiceCall | None = None) -> llm.LLMContext:
    from homeassistant.components.conversation import DOMAIN as CONVERSATION  # noqa: PLC0415
    from homeassistant.core import Context  # noqa: PLC0415

    return llm.LLMContext(
        platform=DOMAIN,
        context=call.context if call else Context(),
        language=hass.config.language,
        assistant=CONVERSATION,
        device_id=None,
    )


async def async_setup_entry(hass: HomeAssistant, entry: RetrievalConfigEntry) -> bool:
    """Register this entry's LLM API and start indexing its tools."""
    options = entry.options
    client = EmbeddingClient(
        async_get_clientsession(hass), options[CONF_EMBEDDING_URL], options.get(CONF_API_KEY)
    )
    index = ToolIndex(hass, entry.entry_id, client)
    await index.async_load(options[CONF_EMBEDDING_URL])
    api = RetrievalAPI(hass, entry, Retrieval(hass, options, index, client))
    entry.runtime_data = api
    entry.async_on_unload(llm.async_register_api(hass, api))

    async def index_tools() -> None:
        # Without a user, admin-only sources are skipped here; their tools are
        # indexed the first time an admin's request reaches them.
        candidates, _ = await api.retrieval.async_gather(_context(hass))
        api.schedule_refresh(candidates)

    @callback
    def start_indexing(_: HomeAssistant) -> None:
        entry.async_create_background_task(hass, index_tools(), f"{DOMAIN} startup index")

    # Other integrations register their APIs while Home Assistant starts, so
    # wait for it to finish starting before looking at the sources.
    entry.async_on_unload(async_at_started(hass, start_indexing))
    return True


async def async_unload_entry(hass: HomeAssistant, entry: RetrievalConfigEntry) -> bool:
    """Unregister the LLM API."""
    return True
