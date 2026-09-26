# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

SoulEngine generates one side of a text-message correspondence between two fictional characters — it is explicitly **not** framed as an assistant answering a user. Read "Design philosophy" below before touching any prompt content (`config/main_prompt.md`, any `bible.md`/`relationship.md`) — the framing choices there are deliberate and easy to accidentally undo by "improving" the wording.

Alongside it runs a second, fully isolated circuit that renders images from the same world for the *reader* of the correspondence — see "Photo mode" below. Its prompts (`config/photo_prompt.md`, `config/compression_prompt.md`) play by different rules on purpose.

## Commands

Environment is managed by uv — `uv.lock` pins every version, `.python-version` pins the interpreter, the venv lives in `.venv/`:
```bash
uv sync                  # make .venv/ match uv.lock exactly
uv add <pkg>             # add a dependency (updates pyproject.toml + uv.lock)
uv lock --upgrade        # deliberately move to newer versions
```
`uv run <cmd>` runs inside `.venv/` and re-syncs it first if the lock changed — no activation needed. Commit `uv.lock` with any dependency change; CI runs `uv sync --locked`, which fails if it drifted from `pyproject.toml`.

Run the bot (loads `.env`, polls Telegram, `Ctrl+C` to stop):
```bash
./run.sh
```

Switch the active character pairing:
```bash
./switch-setting.sh              # no args: lists available settings
./switch-setting.sh <setting>    # the only argument — see below on why self/other aren't passed in
```
The same switch is also available live, from inside the running chat, as the `/list` and `/switch <setting>` control commands (`engine/command_executor.py`) — useful when you're already talking to the bot and don't want to shell out. `/help` in the chat prints every command from both dispatchers (`HELP_TEXT` in `engine/orchestrator.py`; a test keeps it in sync with `KNOWN_COMMANDS`). Unlike `/reset`/`/back`, `/switch` can't just mutate files under a running `Orchestrator`: the setting is baked into a whole tree of cached objects (PromptBuilder, Compressor, MemoryStore, the LLM provider...) built once at startup, so `/switch` writes the new setting into `soul/config.yaml` and then re-execs the whole process (`Orchestrator._restart_process`) so everything gets rebuilt from scratch — a few seconds of reconnect, not a true hot-swap, chosen deliberately over hot-swapping each cached object in place.

Author a character's face for photo mode (one-off, output reviewed by eye —
see "Photo mode" below):
```bash
set -a; source .env; set +a
uv run python scripts/make_face.py            # candidates into debug/faces/<slug>/
uv run python scripts/make_face.py --accept 2 # adopt one next to bible.md
```

Tests:
```bash
uv run pytest -v
uv run pytest tests/test_compressor.py -v
uv run pytest tests/test_compressor.py::test_should_compress_below_threshold
```

Docker (`docker-compose up --build`) exists and volume-mounts host `soul/`/`config/`/`debug/` over the container, so it runs the same files as `./run.sh` — it hasn't been exercised recently in this project's development; verify before relying on it for anything.

## Architecture

### Settings: the isolation boundary

A **setting** is a fully self-contained character pairing — nothing is shared between settings. Everything about one pairing lives under **one** directory:
```
soul/settings/<name>/
  characters/<slug>/bible.md   # static, authored — one character's identity
  characters.yaml               # static, authored — which slug is `self` (generated) and which is `other`
  relationship.md               # static, authored — shared backstory
  characters/<slug>/face.png     # optional, static, authored — likeness anchor for photo mode (NOT gitignored)
  main_prompt.md                 # optional, static — overrides config/main_prompt.md for this setting only
  compression_prompt.md          # optional, static — overrides config/compression_prompt.md for this setting only
  photo_prompt.md                # optional, static — overrides config/photo_prompt.md for this setting only
  llm.yaml                       # optional, static — overrides individual keys of soul/config.yaml's `llm:` block for this setting only
  images.yaml                    # optional, static — opens photo modes here + overrides individual `images:` keys
  runtime/                       # everything generated — gitignored as a whole directory
    chat_log.jsonl               #   raw manuscript history
    summaries.jsonl              #   archival per-chunk digests
    memory.jsonl, memory_vectors.npy  # retrieval store
    continuity.md                #   written by Compressor
    state.md, user_model.md      #   session counters, Telegram metadata
    photos/, photos.jsonl        #   generated frames + a record of every request
```
The authored/`runtime` split is the only boundary inside a setting, and it lines up exactly with what git tracks: **nothing under `soul/` is ignored except `soul/settings/*/runtime/`**, which `git ls-files -ci --exclude-standard soul/` will tell you (it must print nothing). That is also why the ignore rule names a directory rather than a list of file names — a name-by-name list only covers the runtime files that existed when it was written.

Runtime lives *inside* the setting rather than in a parallel `data/settings/<name>/` tree (where it was until 2026-09-19). The parallel tree let the two halves drift: deleting a setting left its history orphaned, and creating a setting under that name again silently inherited the old history — messages, id counters and retrieval vectors included, with nothing failing. Same family as the `self`/`other` flip below: no error, just wrong output. One directory per world means `rm -rf` leaves nothing behind. Paths are built by `setting_dir`/`runtime_dir` in `engine/config_utils.py` — use them rather than re-gluing `"soul" / "settings" / ...`, so `RUNTIME_DIR_NAME` stays the single answer to where runtime lives.

`soul/config.yaml`'s `setting:` key picks the active one; `characters.self`/`characters.other` says which character slugs within it are being generated (`self`) vs. real incoming messages (`other`). Character bibles are not shared resources across settings — duplicating a bible file into a second setting is expected, not a smell.

That `characters:` block in `soul/config.yaml` is **derived, not authored**: the pairing belongs to the setting and is declared in `soul/settings/<name>/characters.yaml`, which both `/switch` and `switch-setting.sh` read and copy into `soul/config.yaml` alongside the `setting:` line. Neither takes self/other as an argument, for two reasons found the hard way:
- Carrying the previous setting's slugs into a new setting doesn't fail loudly — `PromptBuilder._load_markdown_file` returns `""` with a warning for a missing bible, so the bot comes up with two empty bibles and keeps going.
- Flipping `self`/`other` on a setting that already has history silently relabels that history: `chat_log.jsonl` stores only `role` (`user`/`assistant`), and the character name is applied at prompt-build time via `PromptBuilder.role_names` — so every past `assistant` line, written in the old `self`'s voice, would be re-attributed to the new one. Flipping a pairing is an authoring-time edit to `characters.yaml` plus a `/reset`, not a runtime switch.

`main_prompt.md` and `compression_prompt.md` default to the shared files named in `soul/config.yaml` (`prompt.main_prompt_file`, `compression.prompt_file`) — but if `soul/settings/<name>/main_prompt.md` or `.../compression_prompt.md` exists, `Orchestrator` uses that instead (see `_resolve_prompt_file` in `engine/orchestrator.py`). This exists so experimenting with prompt wording for one setting can't silently change every other setting's output — fork the file into the setting directory to iterate in isolation, delete it to fall back to the shared version.

`soul/config.yaml`'s `llm:` block (provider, model, temperature, max_tokens, reasoning_effort, provider_routing) follows the same isolation-boundary idea but merges per-key rather than swapping a whole file: if `soul/settings/<name>/llm.yaml` exists, `_resolve_llm_config` in `engine/main.py` overlays only the keys it defines on top of the shared `llm:` block before the provider is constructed — model choice is voice-dependent per character pairing (see the model-selection comments in `soul/config.yaml`'s `llm:` block), not a global constant. A setting can override just `model` without redeclaring `temperature`/`max_tokens`/etc.; delete the file to fall back to the shared config entirely.

### Prompt assembly (`engine/prompt_builder.py`)

Every turn produces exactly 2 messages: one `system` (the frame, from `config/main_prompt.md`) and one `user` (self bible + other bible + relationship + continuity + the entire chat history collapsed into a single "Name: text" manuscript block + a generation trigger line). There's no multi-turn `user`/`assistant` alternation in the API call — the whole correspondence-so-far is one text block, because alternating roles in the request itself pulls the model back toward assistant-mode regardless of what the system prompt says.

The model's response is the message text itself — no markers, no wrapper. It goes straight to Telegram and to `chat_log.jsonl`.

Two mechanisms were removed to get here, in this order. First a separate "Reflector" LLM call that ran before/after every message to update character state — folded into the same completion as a private `===THOUGHT===` block, halving API calls per turn and keeping everything inside one fictional frame. Then the `===THOUGHT===` block itself (2026-09-08), after six paired trigger comparisons showed it made the character consistently *more* guarded, not less: forcing an explicit private articulation step moves the model into a cognitive, self-monitoring register and gives it something to narratively "resolve" in the reply — usually by deflection. Removing it let the character confront, unprompted, the exact fears her bible names. Rewording the block (`===FEELING===`, raw sensation rather than plan) did not help; the presence of any private pre-step was what mattered, not its wording. **This is the same phenomenon as the message-role finding above, at a different grain — don't reintroduce a private block in any form.**

The response format is markerless rather than a lone `===LINE===` for the same reason: a format spec is itself assistant-register, and it would have been the last thing the model read. The cost was two safety mechanisms that used the marker as an anchor — see `_generate_and_send` in `engine/orchestrator.py` for how each is handled now (empty `content` with non-empty `reasoning` is logged and failed, never published; truncation is caught via `finish_reason == "length"`). `parse_markers` remains, used only by `Compressor`.

`orchestrator.py._handle_message` builds the prompt from chat history **before** logging the incoming message — `build_prompt()` both reads history from `chat_logger` and appends `current_message` explicitly, so logging the message first would duplicate it in the manuscript. Log only after `build_prompt()` returns.

### Memory: continuity, not reflection

`engine/compressor.py` triggers once raw chat history exceeds `compression.trigger_chars` (`soul/config.yaml`), digesting the oldest chunk into `summaries.jsonl` (archival only, never re-read into the prompt) and rewriting `continuity.md`. `continuity.md` carries no markdown header; it's loaded by the same generic loader as `bible.md`/`relationship.md`, so whatever is written to the file goes into the prompt verbatim.

Alongside it there's a second, independent memory path: **retrieval** (`engine/retriever.py` + `engine/memory_store.py` + `engine/embeddings/`, config under `memory:`). Every message is embedded and appended to `soul/settings/<name>/runtime/memory.jsonl` with its vector in `memory_vectors.npy`; each turn the incoming message is embedded and the top-K closest fragments from beyond the manuscript window are injected as their own block. Deliberately not a vector DB — a flat jsonl plus an in-process numpy cosine search, so the store stays readable and hand-editable. Several knobs (`query_window_messages`, `recall_cooldown_turns`, `min_score`) carry their measured justification inline in `soul/config.yaml` and were tuned against real history; `scripts/probe_memory.py` scores a query against the active setting's store, printing results regardless of `min_score`, which is how to check a retuning rather than reading `/memory` output by hand.

Note the invariant in `prompt.budget`'s comment: the manuscript's share must stay larger than `compression.trigger_chars`, or messages that no longer fit the manuscript but haven't been compressed yet fall out of both paths. `Orchestrator` logs a warning when that happens (only when retrieval is enabled — that's the path the gap breaks).

### Photo mode: a second engine, not a feature of the first

`/shot`, `/scene`, `/selfie`, `/again`, `/photos` render frames from the
characters' world (`engine/photo_director.py`, `engine/photo_store.py`,
`engine/images/`, `config/photo_prompt.md`, config under `images:`). What
follows is the part that is easy to break by accident; the reasoning behind
each rule is in the comment at the site it governs.

**The reader asks for the frame, not the character.** This single decision
generates the rest. The characters can see each other inside their world and
have no reason to send each other photos; we are outside, reading, and want
to look. Consequences:

- **Isolation is an invariant, not an optimisation.** The photo circuit never
  writes to `chat_log.jsonl`, `memory.jsonl`, `continuity.md`, `state.md` or
  `summaries.jsonl`, never touches compression counters or retrieval, and
  reads bible/relationship/continuity/manuscript read-only. There is
  deliberately no config option to turn this off — a disabled switch invites
  being enabled.
- **Inside this circuit the Design-philosophy prohibitions do not apply.**
  English, third person, markers, JSON, format specs are all legal here, and
  the describer prompt uses them. Those rules exist because such things pull
  the model into assistant register *inside the fiction*; there is no fiction
  here to contaminate. Same status as `Compressor`.
- **A refusal is a fact about the world, not the character declining.** No
  frame is available → a third-person remark with no addressee ("no frame:
  she's in the dark and hasn't turned on the light"). A character who knows
  she's being watched and says no has broken the fourth wall and become an
  assistant.
- **Don't add in-fiction photo-request detection** ("she notices he asked for
  a picture"). That is an explicit "what is being asked of me" deliberation
  before every reply — structurally the same trap as the removed
  `===THOUGHT===` block and the rejected group chat. The command is the
  design, not a shortcut.

**`PhotoDirector` must not call `build_prompt()`** — that glues the epistolary
frame and the generation trigger, i.e. the whole circuit this engine stays
outside of. It assembles its own context from `load_markdown_file` + a
manuscript tail.

**The face artifact.** `soul/settings/<setting>/characters/<slug>/face.png`
(`.jpg`/`.jpeg` also accepted — image models return JPEG and there's no
converter here; Pillow is deliberately not a dependency). It is authored like
`bible.md`: generated once with `scripts/make_face.py`, chosen by eye,
committed, **never regenerated automatically**. `FACE_POLICY` in
`photo_director.py` says which camera uses it — `selfie` requires it (missing
→ loud failure at startup, not a silent skip), `scene` attaches it when
present, `shot` never does. `scene` is deliberately "optional" rather than
"required": it shipped and ran live before faces existed, and requiring one
would break a working mode in every setting that hasn't authored a face.

The anchor is unstyled on purpose — plain light, no makeup, natural hair.
Measured (2026-09-16): prompt text reliably overrides the reference's
*mutable* attributes (hair colour, makeup, lenses) while identity carries
over, and the reference does **not** leak its pose, background or framing.
So an anchor carrying a look of its own would leak that look everywhere. Two
side effects worth knowing: a reference cuts generation from ~85s to ~15s,
and it pulls the model into a real-photograph register far better than the
prompt's genre paragraph does — but it also drags lighting toward its own
flat light, which is why `config/photo_prompt.md` tells the describer to
push harder on light when a reference is attached.

**Reference images go through OpenRouter's `/api/v1/images` endpoint** (not
chat completions) as `input_references`. How many a model accepts is a
property of the model, published at `/api/v1/images/models` — several accept
none. Check there, don't guess, when changing `images.model`; the limit lives
in config as `images.max_references`.

**Model choice here is per-setting and hard-won.** Both the image model and
the describer are voice-/content-dependent, exactly like the main `llm:`
block, and a setting's `images.yaml` is where that history belongs — written
inline, next to the value it justifies. Three failures cost real time and are
worth knowing before re-litigating a model choice: a Gemini moderation block
first misdiagnosed as a context-size problem (cutting the manuscript hid the
symptom and quietly degraded every frame), a model that silently continued
the story in character instead of writing JSON, and a reasoning model that
needed triple `max_tokens` to leave room for its own output.

### Provider/connector abstraction

`LLMProvider` (`engine/llm/base.py`), `Connector` (`engine/connectors/base.py`), `EmbeddingProvider` (`engine/embeddings/base.py`) and `ImageProvider` (`engine/images/base.py`) are `Protocol`s. Current implementations: `MistralProvider` and `OpenRouterProvider` (picked by `llm.provider` in `soul/config.yaml`, registered in `LLM_PROVIDERS` in `engine/main.py`), `TelegramConnector`, `OpenRouterEmbeddingProvider` (registered in `EMBEDDING_PROVIDERS` in `engine/orchestrator.py`), and `OpenRouterImageProvider` (`IMAGE_PROVIDERS`, same file). Adding another means one new file implementing the Protocol plus a line in the relevant registry — no changes to `Orchestrator` itself.

### Design philosophy — read before touching prompt content

- Never name the messaging platform (Telegram) anywhere the model can see it — the name alone carries "customer-service bot" associations in training data.
- No narrator, no third-person prose. The genre is a plain text-message exchange, not a novel or a play — that's a genre choice made specifically to avoid the model fighting its own completion tendency toward narration.
- Informational symmetry between `self` and `other`: the model is never told which side is "real" vs. generated. Both characters' bible files should read in the same register — as a character in a correspondence, not "the user."
- Prefer short, factual task statements over long meta-explanations of why the model shouldn't act like an assistant — arguing against assistant-ness in the prompt is itself still assistant-flavored performance.
- **These rules govern the fiction circuit only.** `config/compression_prompt.md` and `config/photo_prompt.md` are machinery outside the fiction and legitimately use English, third person, markers and JSON — see "Photo mode" above for why that isn't an inconsistency.

## Testing conventions

- Fixtures in `tests/conftest.py`: `state_manager`/`chat_logger` on `tmp_path`, `fake_llm` (a `FakeLLMProvider` duck-typing the `LLMProvider` Protocol — no mocking library, no network).
- Any test that exercises `Compressor.compress()` must `monkeypatch.chdir(tmp_path)` first — it calls `save_prompt_to_file`/`append_debug_log` on paths hardcoded relative to CWD (`Path("debug/...")`), so without the chdir a test run silently overwrites real files under the project's own `debug/`. (`Orchestrator` passes absolute paths built from `base_path` and doesn't have this problem.)
- The photo circuit builds every path from `base_path`, never from CWD, so it does **not** inherit the `Compressor` chdir trap above — keep it that way when adding to it.
- `prompt_builder.py`'s manuscript assembly, `orchestrator.py`, and the thin external-SDK wrappers (`engine/llm/mistral.py`, `engine/connectors/telegram.py`, the HTTP round trip in `engine/images/openrouter.py` — its payload assembly *is* tested) are intentionally not deeply unit-tested — the wrappers are near-passthroughs, and prompt/orchestration logic currently changes with prompt-design iteration faster than tests on it would earn back their cost.

## Known gotcha

The installed `mistralai` is v2.x, which restructured its package layout — `Mistral` is not importable from the package root. Use `from mistralai.client import Mistral` (see `engine/llm/mistral.py`), not `from mistralai import Mistral`, which fails with a confusing "unknown location" ImportError.
