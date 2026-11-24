"""Fakes shared by the tests: tools, LLM APIs and an embedding service."""

from __future__ import annotations

import math
from typing import override

from homeassistant.core import Context, HomeAssistant
from homeassistant.helpers import llm

from custom_components.llm_tool_retrieval.embedding import EmbeddingError

# Each axis of a fake vector is one topic; a text points along the topics it
# mentions. Plenty to tell "lights" from "coffee" without a real model.
TOPICS = ("light", "timer", "coffee", "money", "map", "music", "weather", "memory")


def fake_vector(text: str) -> list[float]:
    """Return a unit vector along the topics a text mentions."""
    lowered = text.lower()
    vector = [1.0 if topic in lowered else 0.0 for topic in TOPICS] + [0.05]
    norm = math.sqrt(sum(x * x for x in vector))
    return [x / norm for x in vector]


class FakeEmbeddingClient:
    """Stands in for EmbeddingClient; records every text it embeds."""

    def __init__(self) -> None:
        self.embedded: list[str] = []
        self.fail = False

    async def embed(self, texts: list[str], seconds: float) -> list[list[float]]:
        if self.fail:
            raise EmbeddingError("down")
        self.embedded.extend(texts)
        return [fake_vector(t) for t in texts]


class FakeTool(llm.Tool):
    """A tool that returns a card, like the voice satellite's tools do."""

    def __init__(self, name: str, description: str) -> None:
        self.name = name
        self.description = description

    @override
    async def async_call(
        self, hass: HomeAssistant, tool_input: llm.ToolInput, llm_context: llm.LLMContext
    ) -> llm.ToolResult:
        return llm.ToolResult(data={"tool": self.name, "card": {"type": "markdown"}})


class FakeAPI(llm.API):
    """An LLM API with a fixed set of tools and prompt."""

    def __init__(
        self,
        hass: HomeAssistant,
        api_id: str,
        tools: list[llm.Tool],
        prompt: str = "",
        requires_admin: bool = False,
    ) -> None:
        super().__init__(hass=hass, id=api_id, name=api_id, requires_admin=requires_admin)
        self.tools = tools
        self.prompt = prompt or f"Prompt of {api_id}."

    @override
    async def async_get_api_instance(self, llm_context: llm.LLMContext) -> llm.APIInstance:
        return llm.APIInstance(
            api=self, api_prompt=self.prompt, llm_context=llm_context, tools=self.tools
        )


def assist_tools() -> list[llm.Tool]:
    """Device control's built-in tools plus one exposed script."""
    return [
        FakeTool("intent__HassTurnOn", "Turns on a light or switch."),
        FakeTool("llm__GetDateTime", "Returns the date and time."),
        FakeTool("script__morning_coffee", "Brews the morning coffee."),
    ]


def other_tools() -> list[llm.Tool]:
    """Tools of a second, unrelated API."""
    return [
        FakeTool("get_transactions", "Lists money transactions."),
        FakeTool("get_directions", "Finds a route on the map."),
        FakeTool("play_music", "Plays music."),
        FakeTool("forecast", "Weather forecast."),
        FakeTool("save", "Saves a memory."),
    ]


def register(hass: HomeAssistant, *apis: llm.API) -> None:
    """Register LLM APIs for a test."""
    for api in apis:
        llm.async_register_api(hass, api)


def llm_context(user_id: str | None = None) -> llm.LLMContext:
    """Return an LLM context, optionally for a given user."""
    return llm.LLMContext(
        platform="test",
        context=Context(user_id=user_id),
        language="en",
        assistant="conversation",
        device_id=None,
    )
