# LLM Tool Retrieval — Home Assistant Integration

> Requires Home Assistant **2026.12.0.dev202611220217** or later.

[![Built with Claude Code](https://img.shields.io/badge/Built%20with%20Claude%20Code-D97757?style=for-the-badge&logo=claude&logoColor=white)](https://claude.ai/code)
[![Home Assistant](https://img.shields.io/badge/Home%20Assistant-41BDF5?style=for-the-badge&logo=homeassistant&logoColor=white)](https://www.home-assistant.io/)
[![HACS](https://img.shields.io/badge/HACS-Custom-41BDF5?style=for-the-badge&logo=homeassistantcommunitystore&logoColor=white)](https://hacs.xyz/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

A conversation agent with many LLM APIs selected sends every tool to the model
on every request — well over a hundred with a few MCP servers. That costs
tokens and time, and the more tools a model sees, the more often it picks the
wrong one.

This integration registers an LLM API that the agent selects **instead of**
the APIs it wraps. For each request it offers only:

- **Device control's built-in tools** (turning things on and off, timers,
  media, the date and live context), which are short and cover everyday
  requests;
- the tools you name to **always include**;
- the **closest matches by meaning** to what the user said, from every other
  tool — `top_k` of them, 8 by default.

Matching uses an OpenAI-compatible embeddings API, such as the
[Embedding](https://github.com/saya6k/ha-app-embedding) app. Tools are passed
through untouched, so their results — including cards the voice satellite
shows — arrive exactly as before.

Measured on 41 Korean requests against 177 tools, the right tool was among the
ones offered for 93% of them with English descriptions alone, and 98% with a
one-line summary in Korean added (see [AI Task summaries](#ai-task-summaries)).
The tools sent per request went from about 167,000 characters to about 17,000.

## Installation (HACS)

1. HACS → ⋮ → **Custom repositories** → add
   `https://github.com/saya6k/ha-llm-tool-retrieval` as an **Integration**.
2. Install **LLM Tool Retrieval** and restart Home Assistant.
3. Settings → Devices & Services → **Add integration** → "LLM Tool Retrieval".

## Configuration

| Setting | Default | |
|---|---|---|
| Name | Tool retrieval | Shown in the agent's list of LLM APIs |
| Source APIs | — | The APIs whose tools this one chooses from |
| Embedding service URL | — | Including `/v1`, e.g. `http://<hostname>:8080/v1` for the Embedding app |
| API key | — | Optional Bearer token |
| Tools picked per request | 8 | On top of the ones always included |
| Always include Device control's built-in tools | on | Device control's scripts are ranked like any other tool |
| Always include these tools | — | Tool names as the agent sees them, e.g. `search`, `save` |
| AI Task for summaries in your language | — | Optional, see below |
| Query prefix | `task: search result \| query: ` | EmbeddingGemma's; clear it for other models |
| Tool text template | `title: {title} \| text: {text}` | EmbeddingGemma's; use `{text}` for other models |

Then, in the conversation agent's settings, **clear the source APIs and select
this one instead**. Selecting both offers the same tools twice. Entity exposure
still applies: Device control only sees the entities exposed to Assist.

Add one entry per set of sources you need — for example one for the voice
assistant and another, with more sources, for a text agent.

### AI Task summaries

Tool descriptions are mostly English. When requests come in another language,
pick an AI Task entity and the integration has it write a one-line summary of
each tool in Home Assistant's language, which is embedded along with the
description. The model never sees these summaries. Each tool is summarised
once, and again only when its description changes.

## Limits

- **The first request decides the tools for the conversation.** Home Assistant
  asks for an API's tools once per conversation and keeps them for the turns
  that follow, so a follow-up on a different topic in the same conversation
  may not find its tool. Voice requests are usually one turn, so this rarely
  matters there.
- **This is not a security boundary.** A tool that matches a request is
  offered, whatever it does. Leave APIs whose tools must never run on a
  request's say-so — a browser, a code interpreter — out of the sources.
- **It fails open.** Without a request (an AI Task, say), when the embedding
  service does not answer within 5 seconds, or for a tool not embedded yet,
  every such tool is offered rather than hidden.
- Admin-only APIs (such as Home Assistant) are only offered to admins; for
  anyone else they are skipped, and the other sources still work.

## Services

### `llm_tool_retrieval.preview`

Shows which tools a request would be offered, with the similarity score of
every ranked tool — useful for choosing `top_k` and what to always include.

```yaml
action: llm_tool_retrieval.preview
data:
  config_entry_id: 01K...
  text: 거실 불 꺼줘
# reason: ranked | no_query | few_tools | embedding_unavailable
# available / selected: tool counts
# pinned: tools always included
# ranked: [{tool, score, selected}] best first
```

## How it works

At startup, and whenever a request reaches a tool it has not seen, each tool is
embedded from its name and the first 300 characters of its description (plus
its summary, if any). The vectors are stored and reused until that text
changes. Per request, the user's words are embedded once (about 40 ms with
EmbeddingGemma on a CPU) and compared with every stored vector.
