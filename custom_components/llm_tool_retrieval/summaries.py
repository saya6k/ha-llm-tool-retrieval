"""Short tool summaries in the user's language, written by an AI Task entity.

Tool descriptions are mostly English, while requests come in the user's
language. A one-line summary in that language, embedded alongside the
description, brings the two closer: measured on Korean requests it moved the
right tool into the top 8 for 98% of them, against 93% from English alone.
The summaries only feed the embedding; the model still sees the original
descriptions.
"""

from __future__ import annotations

import json
import logging
import re

from homeassistant.core import HomeAssistant

from .const import DOMAIN, SUMMARY_BATCH

_LOGGER = logging.getLogger(__name__)

INSTRUCTIONS = """\
Below are tools an AI assistant for a smart home can call, as JSON objects
with a name and a description. For each tool, write one short sentence in the
language with code "{language}" saying what it does, followed by a few words a
user might say when they want it. Write only in that language. Answer with a
single JSON object that maps each tool name to its sentence, and nothing else.

{tools}
"""


async def async_write_summaries(
    hass: HomeAssistant, entity_id: str, tools: dict[str, str]
) -> dict[str, str]:
    """Return summaries for `tools` (name -> description), keyed by tool name.

    Batches that fail are skipped and logged; those tools keep being embedded
    from their description alone and are tried again on the next index pass.
    """
    from homeassistant.components.ai_task import async_generate_data  # noqa: PLC0415

    language = hass.config.language
    written: dict[str, str] = {}
    names = list(tools)
    for start in range(0, len(names), SUMMARY_BATCH):
        batch = {name: tools[name] for name in names[start : start + SUMMARY_BATCH]}
        payload = json.dumps(
            [{"name": n, "description": d} for n, d in batch.items()], ensure_ascii=False
        )
        try:
            result = await async_generate_data(
                hass,
                task_name=f"{DOMAIN} tool summaries",
                entity_id=entity_id,
                instructions=INSTRUCTIONS.format(language=language, tools=payload),
            )
            answer = _parse(result.data)
        except Exception:  # one bad batch must not stop the rest
            _LOGGER.warning("Could not write tool summaries with %s", entity_id, exc_info=True)
            continue
        for name in batch:
            summary = answer.get(name)
            if isinstance(summary, str) and summary.strip():
                written[name] = summary.strip()
    return written


def _parse(data: object) -> dict[str, object]:
    if isinstance(data, dict):
        return data
    text = str(data).strip()
    # Models like to wrap JSON in a code fence even when told not to.
    fenced = re.search(r"\{.*\}", text, re.DOTALL)
    parsed = json.loads(fenced.group(0) if fenced else text)
    if not isinstance(parsed, dict):
        raise TypeError("expected a JSON object")
    return parsed
