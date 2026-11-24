"""Config and options flow for LLM Tool Retrieval."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import voluptuous as vol
from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    OptionsFlowWithReload,
)
from homeassistant.const import CONF_NAME
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import llm
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.selector import (
    BooleanSelector,
    EntitySelector,
    EntitySelectorConfig,
    NumberSelector,
    NumberSelectorConfig,
    NumberSelectorMode,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TextSelector,
    TextSelectorConfig,
    TextSelectorType,
)

from .const import (
    CONF_AI_TASK_ENTITY,
    CONF_ALWAYS_INCLUDE,
    CONF_API_KEY,
    CONF_DOCUMENT_TEMPLATE,
    CONF_EMBEDDING_URL,
    CONF_PIN_DEVICE_CONTROL,
    CONF_QUERY_PREFIX,
    CONF_SOURCE_APIS,
    CONF_TOP_K,
    DEFAULT_DOCUMENT_TEMPLATE,
    DEFAULT_QUERY_PREFIX,
    DEFAULT_TOP_K,
    DOMAIN,
)
from .embedding import EmbeddingAuthError, EmbeddingClient, EmbeddingError

DEFAULT_NAME = "Tool retrieval"
# The template's own placeholders, passed in so the frontend's message
# formatter does not take them for its own.
PLACEHOLDERS = {"title_field": "{title}", "text_field": "{text}"}
VALIDATE_TIMEOUT = 15.0


def _schema(hass: HomeAssistant, defaults: Mapping[str, Any], *, with_name: bool) -> vol.Schema:
    apis = [
        SelectOptionDict(value=api.id, label=api.name)
        for api in llm.async_get_apis(hass)
        if not api.id.startswith(f"{DOMAIN}__")
    ]
    fields: dict[Any, Any] = {}
    if with_name:
        fields[vol.Required(CONF_NAME, default=defaults.get(CONF_NAME, DEFAULT_NAME))] = (
            TextSelector()
        )
    fields.update(
        {
            vol.Required(CONF_SOURCE_APIS, default=defaults.get(CONF_SOURCE_APIS, [])): (
                SelectSelector(
                    SelectSelectorConfig(
                        options=apis, multiple=True, mode=SelectSelectorMode.LIST
                    )
                )
            ),
            vol.Required(CONF_EMBEDDING_URL, default=defaults.get(CONF_EMBEDDING_URL, "")): (
                TextSelector(TextSelectorConfig(type=TextSelectorType.URL))
            ),
            vol.Optional(
                CONF_API_KEY, description={"suggested_value": defaults.get(CONF_API_KEY)}
            ): TextSelector(TextSelectorConfig(type=TextSelectorType.PASSWORD)),
            vol.Required(CONF_TOP_K, default=defaults.get(CONF_TOP_K, DEFAULT_TOP_K)): (
                NumberSelector(
                    NumberSelectorConfig(min=1, max=50, step=1, mode=NumberSelectorMode.BOX)
                )
            ),
            vol.Required(
                CONF_PIN_DEVICE_CONTROL, default=defaults.get(CONF_PIN_DEVICE_CONTROL, True)
            ): BooleanSelector(),
            vol.Optional(
                CONF_ALWAYS_INCLUDE, default=defaults.get(CONF_ALWAYS_INCLUDE, [])
            ): SelectSelector(
                SelectSelectorConfig(options=[], multiple=True, custom_value=True)
            ),
            vol.Optional(
                CONF_AI_TASK_ENTITY,
                description={"suggested_value": defaults.get(CONF_AI_TASK_ENTITY)},
            ): EntitySelector(EntitySelectorConfig(domain="ai_task")),
            vol.Required(
                CONF_QUERY_PREFIX, default=defaults.get(CONF_QUERY_PREFIX, DEFAULT_QUERY_PREFIX)
            ): TextSelector(),
            vol.Required(
                CONF_DOCUMENT_TEMPLATE,
                default=defaults.get(CONF_DOCUMENT_TEMPLATE, DEFAULT_DOCUMENT_TEMPLATE),
            ): TextSelector(),
        }
    )
    return vol.Schema(fields)


async def _async_validate(hass: HomeAssistant, data: dict[str, Any]) -> dict[str, str]:
    errors: dict[str, str] = {}
    if not data.get(CONF_SOURCE_APIS):
        errors[CONF_SOURCE_APIS] = "no_sources"
    try:
        data[CONF_DOCUMENT_TEMPLATE].format(title="", text="")
    except (KeyError, IndexError, ValueError):
        errors[CONF_DOCUMENT_TEMPLATE] = "invalid_template"
    if "{text}" not in data[CONF_DOCUMENT_TEMPLATE]:
        errors[CONF_DOCUMENT_TEMPLATE] = "invalid_template"

    client = EmbeddingClient(
        async_get_clientsession(hass), data[CONF_EMBEDDING_URL], data.get(CONF_API_KEY)
    )
    try:
        await client.embed([data[CONF_QUERY_PREFIX] + "test"], VALIDATE_TIMEOUT)
    except EmbeddingAuthError:
        errors[CONF_API_KEY] = "invalid_auth"
    except (EmbeddingError, TimeoutError):
        errors[CONF_EMBEDDING_URL] = "cannot_connect"
    return errors


def _clean(data: dict[str, Any]) -> dict[str, Any]:
    options = {k: v for k, v in data.items() if k != CONF_NAME and v not in (None, "")}
    options[CONF_TOP_K] = int(options[CONF_TOP_K])
    return options


class RetrievalConfigFlow(ConfigFlow, domain=DOMAIN):
    """Create a retrieval API. Each entry is one API an agent can select."""

    VERSION = 1

    async def async_step_user(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """Choose the source APIs and the embedding service."""
        errors: dict[str, str] = {}
        if user_input is not None:
            errors = await _async_validate(self.hass, user_input)
            if not errors:
                return self.async_create_entry(
                    title=user_input[CONF_NAME], data={}, options=_clean(user_input)
                )
        return self.async_show_form(
            step_id="user",
            data_schema=_schema(self.hass, user_input or {}, with_name=True),
            errors=errors,
            description_placeholders=PLACEHOLDERS,
        )

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: ConfigEntry) -> RetrievalOptionsFlow:
        """Return the options flow."""
        return RetrievalOptionsFlow()


class RetrievalOptionsFlow(OptionsFlowWithReload):
    """Change the sources, the embedding service and how tools are picked."""

    async def async_step_init(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """Show the same settings as setup, filled in with the current ones."""
        errors: dict[str, str] = {}
        if user_input is not None:
            errors = await _async_validate(self.hass, user_input)
            if not errors:
                return self.async_create_entry(data=_clean(user_input))
        return self.async_show_form(
            step_id="init",
            data_schema=_schema(
                self.hass, user_input or self.config_entry.options, with_name=False
            ),
            errors=errors,
            description_placeholders=PLACEHOLDERS,
        )
