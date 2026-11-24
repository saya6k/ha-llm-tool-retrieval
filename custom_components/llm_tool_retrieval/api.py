"""The LLM API a conversation agent selects instead of the APIs it wraps."""

from __future__ import annotations

import logging
from typing import override

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm

from .const import DOMAIN
from .retrieval import Candidate, Retrieval, current_request_text

_LOGGER = logging.getLogger(__name__)


class RetrievalAPI(llm.API):
    """Offers, per request, the source APIs' tools that the request needs."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry, retrieval: Retrieval) -> None:
        """Initialize with the entry's id and title."""
        super().__init__(hass=hass, id=f"{DOMAIN}__{entry.entry_id}", name=entry.title)
        self._entry = entry
        self.retrieval = retrieval
        self._refreshing = False

    @override
    async def async_get_api_instance(self, llm_context: llm.LLMContext) -> llm.APIInstance:
        """Return an instance holding only the tools this request needs.

        Home Assistant asks for the instance once per conversation, on its first
        turn, and keeps the tools for the turns that follow, so the first thing
        the user says decides the tools for the whole conversation.
        """
        candidates, instances = await self.retrieval.async_gather(llm_context)
        query = current_request_text()
        selection = await self.retrieval.async_select(candidates, query)
        _LOGGER.debug(
            "%s: %d of %d tools (%s) for %r: %s",
            self.name,
            len(selection.tools),
            len(candidates),
            selection.reason,
            query,
            [tool.name for tool in selection.tools],
        )
        self.schedule_refresh(candidates)
        return self.retrieval.build_instance(
            self, llm_context, instances, candidates, selection
        )

    def schedule_refresh(self, candidates: list[Candidate]) -> None:
        """Embed tools not seen before, in the background (one pass at a time)."""
        if self._refreshing:
            return
        self._refreshing = True
        self._entry.async_create_background_task(
            self.hass, self._async_refresh(candidates), f"{DOMAIN} index {self._entry.entry_id}"
        )

    async def _async_refresh(self, candidates: list[Candidate]) -> None:
        try:
            await self.retrieval.async_refresh(candidates)
        except Exception as err:  # retried on the next request
            _LOGGER.warning("Could not index tools for %s: %s", self.name, err)
        finally:
            self._refreshing = False
