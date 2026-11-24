"""Pick the tools relevant to a request out of several LLM APIs."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import slugify as unicode_slug
from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm

from .const import (
    ASSIST_API_ID,
    CONF_AI_TASK_ENTITY,
    CONF_ALWAYS_INCLUDE,
    CONF_DOCUMENT_TEMPLATE,
    CONF_PIN_DEVICE_CONTROL,
    CONF_QUERY_PREFIX,
    CONF_SOURCE_APIS,
    CONF_TOP_K,
    DEFAULT_DOCUMENT_TEMPLATE,
    DEFAULT_QUERY_PREFIX,
    DEFAULT_TOP_K,
    DESCRIPTION_CHARS,
    QUERY_TIMEOUT,
    SCRIPT_TOOL_PREFIX,
)
from .embedding import EmbeddingClient, EmbeddingError, dot
from .index import ToolIndex
from .summaries import async_write_summaries

_LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class Candidate:
    """A tool offered by one of the source APIs."""

    tool: llm.Tool
    api_id: str
    pinned: bool
    source: str
    """The tool's name and description: what a summary is written from."""


@dataclass(slots=True)
class Selection:
    """The outcome of choosing tools for one request."""

    tools: list[llm.Tool]
    reason: str
    """`ranked`, or why every tool was offered instead."""
    scores: dict[str, float] = field(default_factory=dict)


class Retrieval:
    """Gathers the source APIs' tools and keeps the ones a request needs."""

    def __init__(
        self,
        hass: HomeAssistant,
        options: Mapping[str, Any],
        index: ToolIndex,
        client: EmbeddingClient,
    ) -> None:
        """Initialize from the config entry's options."""
        self._hass = hass
        self._index = index
        self._client = client
        self._source_ids: list[str] = list(options.get(CONF_SOURCE_APIS, []))
        self._top_k: int = int(options.get(CONF_TOP_K, DEFAULT_TOP_K))
        self._pin_device_control: bool = options.get(CONF_PIN_DEVICE_CONTROL, True)
        self._always_include: set[str] = set(options.get(CONF_ALWAYS_INCLUDE, []))
        self._ai_task_entity: str | None = options.get(CONF_AI_TASK_ENTITY) or None
        self._query_prefix: str = options.get(CONF_QUERY_PREFIX, DEFAULT_QUERY_PREFIX)
        self._template: str = options.get(CONF_DOCUMENT_TEMPLATE, DEFAULT_DOCUMENT_TEMPLATE)
        # Tools a summary was already asked for since startup. One that came
        # back without a summary is not asked about again on every request.
        self._summary_asked: set[str] = set()

    async def async_gather(
        self, llm_context: llm.LLMContext
    ) -> tuple[list[Candidate], dict[str, llm.APIInstance]]:
        """Return every tool the source APIs offer in this context.

        A source API that is missing (an MCP server that is down, say) or that
        this user may not use (an admin-only API for a non-admin) is skipped,
        rather than failing every other source with it.
        """
        registered = {api.id: api for api in llm.async_get_apis(self._hass)}
        instances: dict[str, llm.APIInstance] = {}
        for api_id in self._source_ids:
            if (api := registered.get(api_id)) is None:
                _LOGGER.debug("Source LLM API %s is not registered; skipping it", api_id)
                continue
            try:
                instances[api_id] = await llm.async_get_authorized_api_instance(api, llm_context)
            except Exception:  # one broken source must not take down the rest
                _LOGGER.debug("Skipping source LLM API %s", api_id, exc_info=True)

        # Same rule as Home Assistant's MergedAPI: a tool name offered by more
        # than one API gets that API's id as a namespace; unique names stay.
        by_name: dict[str, list[tuple[str, llm.Tool]]] = {}
        for api_id, instance in instances.items():
            for tool in instance.tools:
                entries = by_name.setdefault(tool.name, [])
                if all(candidate != tool for _, candidate in entries):
                    entries.append((api_id, tool))

        candidates: list[Candidate] = []
        for name, entries in by_name.items():
            for api_id, tool in entries:
                exposed = (
                    tool
                    if len(entries) == 1
                    else llm.NamespacedTool(unicode_slug.slugify(api_id), tool)
                )
                candidates.append(
                    Candidate(
                        tool=exposed,
                        api_id=api_id,
                        pinned=self._is_pinned(api_id, name, exposed.name),
                        source=f"{name}\n{tool.description or ''}",
                    )
                )
        return candidates, instances

    def _is_pinned(self, api_id: str, name: str, exposed_name: str) -> bool:
        if name in self._always_include or exposed_name in self._always_include:
            return True
        return (
            self._pin_device_control
            and api_id == ASSIST_API_ID
            and not name.startswith(SCRIPT_TOOL_PREFIX)
        )

    def document(self, candidate: Candidate) -> str:
        """Return the text a tool is embedded from."""
        name, _, description = candidate.source.partition("\n")
        text = description.strip()[:DESCRIPTION_CHARS]
        if summary := self._index.summary(candidate.source):
            text = f"{summary} {text}"
        return self._template.format(title=_title(name), text=text)

    async def async_select(self, candidates: list[Candidate], query: str | None) -> Selection:
        """Keep the pinned tools and the `top_k` closest to the query.

        Fails open: without a query, or when the embedding service cannot
        answer, every tool is offered. A tool that has not been embedded yet
        is offered too, until the background index catches up.
        """
        everything = [c.tool for c in candidates]
        if not query:
            return Selection(everything, "no_query")
        ranked = [c for c in candidates if not c.pinned]
        if len(ranked) <= self._top_k:
            return Selection(everything, "few_tools")
        try:
            (query_vector,) = await self._client.embed(
                [self._query_prefix + query], QUERY_TIMEOUT
            )
        except (EmbeddingError, TimeoutError) as err:
            _LOGGER.warning("Embedding service unavailable, offering every tool: %s", err)
            return Selection(everything, "embedding_unavailable")

        scores: dict[str, float] = {}
        unindexed: set[str] = set()
        for candidate in ranked:
            vector = self._index.vector(self.document(candidate))
            if vector is None:
                unindexed.add(candidate.tool.name)
            else:
                scores[candidate.tool.name] = dot(query_vector, vector)
        best = set(sorted(scores, key=scores.__getitem__, reverse=True)[: self._top_k])
        keep = best | unindexed
        tools = [c.tool for c in candidates if c.pinned or c.tool.name in keep]
        return Selection(tools, "ranked", scores)

    async def async_refresh(self, candidates: list[Candidate]) -> None:
        """Write missing summaries, then embed every tool not embedded yet."""
        if self._ai_task_entity:
            unsummarised = {
                c.source.partition("\n")[0]: c.source
                for c in candidates
                if not c.pinned
                and c.source not in self._summary_asked
                and self._index.summary(c.source) is None
            }
            if unsummarised:
                self._summary_asked.update(unsummarised.values())
                written = await async_write_summaries(
                    self._hass,
                    self._ai_task_entity,
                    {name: source.partition("\n")[2] for name, source in unsummarised.items()},
                )
                self._index.add_summaries(
                    {unsummarised[name]: summary for name, summary in written.items()}
                )
        await self._index.async_index([self.document(c) for c in candidates if not c.pinned])

    def build_instance(
        self,
        api: llm.API,
        llm_context: llm.LLMContext,
        instances: dict[str, llm.APIInstance],
        candidates: list[Candidate],
        selection: Selection,
    ) -> llm.APIInstance:
        """Return an API instance holding only the selected tools.

        Only the source APIs that still contribute a tool add their prompt, so
        instructions for tools the model cannot see are left out too.
        """
        kept = {id(tool) for tool in selection.tools}
        prompt_parts: list[str] = []
        for api_id, instance in instances.items():
            names = [
                c.tool.name for c in candidates if c.api_id == api_id and id(c.tool) in kept
            ]
            if not names:
                continue
            if len(instances) == 1:
                prompt_parts.append(instance.api_prompt)
                break
            prompt_parts.append(
                f'Follow these instructions for tools from "{unicode_slug.slugify(api_id)}" '
                f"({', '.join(names)}):\n{instance.api_prompt}\n\n"
            )
        return llm.APIInstance(
            api=api,
            api_prompt="".join(prompt_parts),
            llm_context=llm_context,
            tools=selection.tools,
            custom_serializer=_merged_serializer(instances.values()),
        )


def _title(name: str) -> str:
    """Turn `intent__HassTurnOn` into `Turn On`, `get_air_quality` into `get air quality`."""
    name = name.rpartition("__")[2]
    for prefix in ("Hass", "llm_script_for_"):
        name = name.removeprefix(prefix)
    spaced = "".join(f" {ch}" if ch.isupper() and i else ch for i, ch in enumerate(name))
    return spaced.replace("_", " ").strip()


def _merged_serializer(
    instances: Any,
) -> Callable[[Any], Any] | None:
    serializers = [i.custom_serializer for i in instances if i.custom_serializer is not None]
    if not serializers:
        return None

    def merged(value: Any) -> Any:
        for serializer in serializers:
            if (result := serializer(value)) is not None:
                return result
        return None

    return merged


def current_request_text() -> str | None:
    """Return what the user just said, from the chat log this turn runs in."""
    from homeassistant.components.conversation import UserContent  # noqa: PLC0415
    from homeassistant.components.conversation.chat_log import (  # noqa: PLC0415
        current_chat_log,
    )

    if (chat_log := current_chat_log.get()) is None:
        return None
    for content in reversed(chat_log.content):
        if isinstance(content, UserContent):
            return content.content
    return None
