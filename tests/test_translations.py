"""Guard translation strings against ICU message-format pitfalls.

The HA frontend renders translation strings through an ICU message formatter,
so any `{...}` is treated as a placeholder. Every placeholder used must be a
bare identifier the flow passes in.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

from custom_components.llm_tool_retrieval.config_flow import PLACEHOLDERS

PLACEHOLDER = re.compile(r"\{([^{}]*)\}")
COMPONENT = Path(__file__).parent.parent / "custom_components" / "llm_tool_retrieval"
TRANSLATIONS = COMPONENT / "translations"


def _load(name: str) -> dict:
    return json.loads((TRANSLATIONS / f"{name}.json").read_text(encoding="utf-8"))


def _iter_strings(obj: object) -> list[str]:
    if isinstance(obj, dict):
        return [s for value in obj.values() for s in _iter_strings(value)]
    if isinstance(obj, list):
        return [s for value in obj for s in _iter_strings(value)]
    if isinstance(obj, str):
        return [obj]
    return []


@pytest.mark.parametrize("name", ["en", "ko"])
def test_placeholders_are_ones_the_flow_passes(name: str) -> None:
    """Every {placeholder} is one the config flow supplies."""
    for text in _iter_strings(_load(name)):
        for placeholder in PLACEHOLDER.findall(text):
            assert placeholder in PLACEHOLDERS, f"{name}.json: unknown {{{placeholder}}}"


def test_korean_has_every_english_key() -> None:
    """The Korean file translates every string the English one has."""

    def keys(obj: object, prefix: str = "") -> set[str]:
        if not isinstance(obj, dict):
            return {prefix}
        return {k for key, value in obj.items() for k in keys(value, f"{prefix}.{key}")}

    assert keys(_load("ko")) == keys(_load("en"))


@pytest.mark.parametrize("name", ["en", "ko"])
def test_every_service_field_is_translated(name: str) -> None:
    """Each services.yaml field has a name and description in the translation file."""
    services = yaml.safe_load((COMPONENT / "services.yaml").read_text(encoding="utf-8"))
    data = _load(name)
    for service, spec in services.items():
        fields = data["services"][service]["fields"]
        for field in spec.get("fields", {}):
            assert fields[field]["name"], f"{name}.json: {service}.{field} has no name"
            assert fields[field]["description"], f"{name}.json: {service}.{field}"
