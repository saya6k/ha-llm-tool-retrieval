"""Constants for LLM Tool Retrieval."""

from __future__ import annotations

from typing import Final

DOMAIN: Final = "llm_tool_retrieval"

CONF_SOURCE_APIS: Final = "source_apis"
CONF_EMBEDDING_URL: Final = "embedding_url"
CONF_API_KEY: Final = "api_key"
CONF_TOP_K: Final = "top_k"
CONF_PIN_DEVICE_CONTROL: Final = "pin_device_control"
CONF_ALWAYS_INCLUDE: Final = "always_include"
CONF_AI_TASK_ENTITY: Final = "ai_task_entity"
CONF_QUERY_PREFIX: Final = "query_prefix"
CONF_DOCUMENT_TEMPLATE: Final = "document_template"

DEFAULT_TOP_K: Final = 8
# EmbeddingGemma's task prompts: a query and a titled passage are embedded
# differently, and both prefixes are part of what the model was trained on.
DEFAULT_QUERY_PREFIX: Final = "task: search result | query: "
DEFAULT_DOCUMENT_TEMPLATE: Final = "title: {title} | text: {text}"

# The Device control (Assist) API. Its built-in tools are short and cover the
# everyday requests, so they are kept on every turn; its scripts are ranked
# like any other tool, since there can be many and they can be long.
ASSIST_API_ID: Final = "assist"
SCRIPT_TOOL_PREFIX: Final = "script__"

# How much of a tool's description is embedded. The opening states what a
# tool is for; the rest is mostly usage detail that dilutes the match.
DESCRIPTION_CHARS: Final = 300

# A request waits this long for its own query embedding before every tool is
# offered instead. Indexing tool descriptions happens in the background.
QUERY_TIMEOUT: Final = 5.0
INDEX_TIMEOUT: Final = 300.0
EMBED_BATCH: Final = 32

# Tools summarised per AI Task call when writing descriptions in the user's
# language.
SUMMARY_BATCH: Final = 25

STORAGE_VERSION: Final = 1

SERVICE_PREVIEW: Final = "preview"
ATTR_CONFIG_ENTRY_ID: Final = "config_entry_id"
ATTR_TEXT: Final = "text"
