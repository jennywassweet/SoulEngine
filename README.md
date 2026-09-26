# SoulEngine

An engine that generates one side of a text-message correspondence between two
fictional characters, over Telegram, with memory and with images rendered from
the same world.

The goal of the project is to prevent the model from identifying itself as a
virtual assistant. The approach involves stripping the prompt of the standard
"user-assistant" dialogue structure: no alternating roles, no narrator, no
hidden pre-reasoning stages, and no cues that would allow the model to deduce
which of the two characters is the human.

**This is a research project with a working implementation, not a fully
supported product.** No technical support is provided, and issue reports may
go unanswered. The project is distributed under the MIT License; you are free
to use any parts of it that you find useful.

## What it looks like

The first exchange of a fresh run — empty history, no warm-up, the generated
side (Alexis) reproduced unedited:

```
Sasha    Hi, how are you?

Alexis   table four spilled an entire side of ranch into the vent under the
         booth. so that's going to be a fun smell by tomorrow.

         how's the cardboard box empire

Sasha    A table with a spicy smell? You're at work — I'm not interrupting,
         am I?

Alexis   it's 2:30. the only people here are two guys arguing about hay
         prices and a woman doing a crossword in pen. you're fine.

         also ranch doesn't smell spicy it smells like warm mayonnaise and
         regret. you city people don't know anything about shelf-stable
         buttermilk
```

She does not answer the question she was asked. She teases him about a job
she was never instructed to bring up, and she corrects him. None of that is
asked for anywhere in the prompt — it is what is left once the shape of an
assistant conversation is taken out of the request.

The frame she describes in that first message appears under
[Photo mode](#photo-mode) below.

## Philosophy

SoulEngine generates one side of a text-message correspondence between two
characters. It never frames itself as an assistant answering a user — the
model is asked to continue an exchange between two symmetric characters,
with no narrator and no mention of the underlying messaging platform. A
pairing (its characters, relationship, and accumulated continuity) is a
**setting** — a fully self-contained unit under `soul/settings/<name>/`
(authored files at the top, everything generated in its `runtime/`
subdirectory), so a different pairing can be tried without any crossover.
The engine provides: brain (LLM), memory (rolling continuity
notes + semantic recall), and world connection (connectors). The characters
and their relationship are defined by configuration and plain-text files,
not code.

### Motivation

The project began as an attempt to get the model out of the assistant role —
to make its behaviour toward the user autonomous, and to simulate live human
conversation.

A model builds an assistant persona oriented toward serving the user's
interests. Even given roleplay instructions in which that assistant plays an
autonomous character, the assistant patterns stay the model's priority — the
user remains the trigger for every act of communication. The conclusion was
to move the assistant outside the frame being created: inside the dialogue,
erase the difference between assistant and user.

So inside SoulEngine the model does not answer a user's request — it writes
the next line of a character in a novel. It does not identify with that
character either; it works from the context of the novel. The model is never
told which participant in the dialogue is the user and which is itself.

A check (a comparison against a roleplay prompt) confirmed that this extra
layer of abstraction gives convincing results. The character is not bound by
the assistant's stylistic and — more importantly — semantic patterns, which
also loosens the constraints the model places on itself. This is not a
jailbreak and not a way around safety filters; it makes it possible to build
a convincing character without projecting the assistant's motivation onto
them. Decisions follow from the content of the novel rather than from the
instructions, and the patterns of conversation become more natural.

### How it is built

The prompt is assembled in four layers. All of it reaches the API as exactly
two messages: layer 1 is the `system` message, and layers 2–4 are
concatenated into a single `user` message — there is no `user`/`assistant`
alternation, because the alternation itself signals a service exchange.

**Layer 1 — the frame.** A system prompt: you are helping write a novel in
the epistolary genre; the novel is a correspondence.

**Layer 2 — static, authored.**

- character 1
- character 2
- the relationship between them: shared backstory and the starting
  conditions for the exchange

**Layer 3 — dynamic**, regenerated as the dialogue goes on, which is what
keeps the context window under control.

- the main events of the story so far
- fragments from vector memory relevant to the current moment

**Layer 4 — the correspondence itself.**

```
Character 1: ...
Character 2: ...
(...)
Character 1: ...
→ the model writes Character 2's next line
```

Both characters are symmetric as far as the model is concerned: the same kind
of file, the same level of detail, with nothing marking either one as real.

The approach produced convincing results — the dialogue is alive and
dynamic. It does require a model clever enough to hold instructions of this
complexity together with a long context. The scheme above was arrived at
empirically.

### Approaches that worked less well

- **An assistant playing a role.** The assistant pattern stays present and
  does not simulate autonomy.
- **A hidden reasoning loop** — either an extra call or a private block
  inside the same response, kept out of the final output. The expectation was
  that it would make the model more inertial: holding internal state,
  concealing intent. The effect was the opposite. The loop reinforced
  existing patterns and pinned the character inside the setting instead of
  letting them develop. Worth knowing if you are building something similar:
  a private deliberation step breaks the effect at any level and in any
  wording — rewriting the block to ask for raw feeling instead of intent
  changed nothing.

### What it is for

Research into natural conversation with fictional characters. It produces
convincing roleplay without roleplay instructions, which makes it a way to
explore a character. Photo mode was added to deepen the immersion; its
parameters are described below.

## What works now

- **Epistolary prompt assembly** — per-character bibles, shared relationship
  note, the whole correspondence collapsed into one manuscript block,
  markerless message-only responses.
- **Rolling continuity** — old history is compressed into `continuity.md`
  (injected into every prompt) and archived per-chunk digests.
- **Retrieval memory** — every message is embedded; semantically close
  fragments from beyond the manuscript window are recalled alongside
  continuity. Backed by a plain `memory.jsonl` + a numpy vector file, with
  recency weighting and a per-fragment cooldown.
- **Photo mode** — `/shot`, `/scene` and `/selfie` render a frame from the
  current moment of the correspondence: a describer model reads the bible,
  continuity and manuscript and writes an image prompt, an image model
  renders it. A per-character face artifact keeps the likeness consistent
  across frames. Fully separate from the text circuit — see below.
- **Two LLM providers** — Mistral and OpenRouter, behind one Protocol.
- **Per-setting overrides** — a setting can fork the main/compression prompt
  and override individual LLM parameters without touching any other setting.
- **Live setting switching** — `/switch` from inside the chat, no shell needed.

## Quick Start

### Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/) — it fetches
  Python 3.12 itself if you don't have it
- An LLM API key — [OpenRouter](https://openrouter.ai/) (current default) or
  [Mistral](https://console.mistral.ai/). Retrieval memory and photo mode go
  through OpenRouter specifically, so those two need its key regardless of
  which provider writes the messages.
- Telegram bot token ([create bot with @BotFather](https://t.me/botfather))
- Your Telegram user ID ([get it from @userinfobot](https://t.me/userinfobot))

### Setup

1. Clone the repository and navigate to it

2. Install the exact dependency versions from `uv.lock` into a local `.venv/`:
```bash
uv sync
```
   Without uv, plain pip works too (Python 3.12+, pip 25.1+), but it resolves
   fresh versions instead of the locked ones:
```bash
python3 -m venv .venv
.venv/bin/pip install -e . --group dev
```
   and then use `.venv/bin/python` wherever this README says `uv run python`.

3. Create `.env` file from example:
```bash
cp .env.example .env
```

4. Edit `.env` and add your credentials (only the key for the provider you
   actually use in `soul/config.yaml` is required):
```bash
OPENROUTER_API_KEY=your_openrouter_api_key_here
MISTRAL_API_KEY=your_mistral_api_key_here
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
TELEGRAM_ALLOWED_USERS=your_telegram_user_id
```

### Running

```bash
./run.sh          # loads .env, polls Telegram, Ctrl+C to stop
```

You should see logs showing engine initialization, the active setting and its
resolved LLM config, connector setup, and a bot-ready message. Then open
Telegram and send a message to your bot.

Tests:
```bash
uv run pytest -v
```

### Docker

```bash
docker-compose up --build
```
Volume-mounts host `soul/`/`config/`/`debug/` over the container, so it
runs the same files as `./run.sh`. This path hasn't been exercised recently in
the project's development — verify before relying on it.

## Project Structure

```
soulengine/
├── engine/                  # Engine core
│   ├── llm/                 # LLM providers: base Protocol + mistral, openrouter
│   ├── connectors/          # Communication connectors: base Protocol + telegram (aiogram)
│   ├── embeddings/          # Embedding providers for retrieval memory
│   ├── images/              # Image providers: base Protocol + openrouter (/api/v1/images)
│   ├── photo_director.py    # Photo mode: context, describer call, camera/face policy
│   ├── photo_store.py       # photos/<id>.png + photos.jsonl (every request, refusals included)
│   ├── orchestrator.py      # Links LLM with connectors, routes messages, debug commands
│   ├── prompt_builder.py    # Assembles the system frame + one collapsed user turn
│   ├── compressor.py        # Compresses old chat history into rolling continuity notes
│   ├── memory_store.py      # Append-only message store + vector matrix for recall
│   ├── retriever.py         # Semantic recall: scoring, recency weighting, cooldown
│   ├── chat_logger.py       # chat_log.jsonl reader/writer (atomic rewrites)
│   ├── state_manager.py     # Frontmatter-backed counters in state.md / user_model.md
│   ├── config_utils.py      # Reads config.yaml + per-setting overrides; writes active setting
│   ├── command_parser.py    # Recognizes control commands (/reset, /back, /list, /switch)
│   ├── command_executor.py  # Runs control commands against a setting's runtime files
│   ├── response_parser.py   # Marker parsing (used by Compressor only)
│   ├── debug_logger.py      # Last-prompt dumps + append-only prompt log
│   └── main.py              # Entry point
│
├── soul/
│   ├── settings/
│   │   └── <name>/                   # One self-contained pairing
│   │       ├── characters/
│   │       │   └── <slug>/
│   │       │       ├── bible.md      # Per-character identity (static, authored)
│   │       │       └── face.jpeg     # Optional: likeness anchor for photo mode (authored)
│   │       ├── characters.yaml       # Which slug is `self` (generated) / `other` (static)
│   │       ├── relationship.md       # Shared backstory (static, authored)
│   │       ├── opening_line.md       # Optional: authored first message, used on reset
│   │       ├── main_prompt.md        # Optional: forks config/main_prompt.md for this setting
│   │       ├── compression_prompt.md # Optional: forks config/compression_prompt.md
│   │       ├── llm.yaml              # Optional: overrides individual `llm:` keys
│   │       ├── images.yaml           # Optional: opens photo modes here; overrides `images:` keys
│   │       ├── photo_prompt.md       # Optional: forks config/photo_prompt.md
│   │       └── runtime/              # Everything generated (gitignored as a directory)
│   │           ├── chat_log.jsonl    #   Raw manuscript history
│   │           ├── summaries.jsonl   #   Archival per-chunk digests
│   │           ├── memory.jsonl      #   Retrieval store + memory_vectors.npy
│   │           ├── continuity.md     #   Rolling notes, written by Compressor
│   │           ├── state.md          #   Session counters
│   │           ├── user_model.md     #   Telegram user metadata
│   │           └── photos/           #   Generated frames + photos.jsonl
│   └── config.yaml                   # Parameters, incl. active `setting`
│
├── config/                      # Prompt templates (shared defaults across settings)
│   ├── main_prompt.md           # System frame: genre + response constraints
│   ├── compression_prompt.md
│   └── photo_prompt.md          # Frame describer: genre, camera positions, reference rules
│
├── scripts/
│   ├── probe_memory.py          # Probe retrieval memory against real history
│   ├── make_face.py             # Generate a character's face artifact from their bible
│   ├── probe_image_models.py    # Compare image models on genre + moderation prompts
│   └── probe_face_reference.py  # Measure how a face reference behaves on a given model
├── tests/                       # pytest suite (no mocking library, no network)
├── examples/                    # The frames shown in this README, from one live run
├── debug/                       # Last-prompt dumps + append-only prompt log (gitignored)
├── run.sh, switch-setting.sh, reset-setting.sh
├── docker-compose.yml
└── Dockerfile
```

## Configuration

Edit `soul/config.yaml` to customize:
- Which setting is active (`setting`). The `characters:` block below it is
  **derived** — written from the setting's own `characters.yaml` by the switch
  scripts, not authored by hand.
- LLM provider, model and parameters (`llm:`)
- Prompt char budget (`prompt.budget`) and compression thresholds (`compression:`)
- Retrieval memory (`memory:`) — embedding model, `top_k`, score floor, recency
  half-life, recall cooldown. Most of these carry comments recording what was
  actually measured on real history; read them before retuning.
- Photo mode (`images:`) — global on/off switch, image model, describer model,
  manuscript budget, reference limit. These are **defaults**: photo mode is off
  in any setting that doesn't declare its own `images.yaml`.

### Per-setting overrides

A setting can override shared configuration without affecting any other setting.
Drop the file into `soul/settings/<name>/` and it takes precedence; delete it to
fall back to the shared default.

| File | Overrides |
|---|---|
| `main_prompt.md` | `config/main_prompt.md` (whole file) |
| `compression_prompt.md` | `config/compression_prompt.md` (whole file) |
| `llm.yaml` | individual keys of `soul/config.yaml`'s `llm:` block |
| `images.yaml` | individual keys of the `images:` block — **and** declares which photo modes are open here |
| `photo_prompt.md` | `config/photo_prompt.md` (whole file) |

Prompt forks are whole-file swaps; `llm.yaml` and `images.yaml` merge **per
key**, so a setting can override just `model` without redeclaring
`temperature`, `max_tokens` and the rest. This exists because model choice
turns out to be voice-dependent per character pairing, not a global constant —
and the same is true of how a pairing should *look*.

### Switching and resetting settings

```bash
./switch-setting.sh            # no args: list available settings
./switch-setting.sh <setting>  # change the active pairing in soul/config.yaml
./reset-setting.sh <setting>   # wipe a setting's runtime state and start over
```

`switch-setting.sh` takes only the setting name — the `self`/`other` pairing is
read from that setting's `characters.yaml`, never passed in. Carrying the
previous setting's slugs over would silently load empty bibles (a missing
`bible.md` logs a warning and reads as `""`), and flipping `self`/`other` on a
setting that already has history would re-attribute every past message, since
`chat_log.jsonl` stores only `role` and the character name is applied at
prompt-build time.

`reset-setting.sh` clears `chat_log.jsonl`, `summaries.jsonl`, the retrieval
memory files, `state.md`, `user_model.md`, `continuity.md`, and the debug prompt
logs. If the setting has an `opening_line.md`, `chat_log.jsonl` is reseeded with
it as the first message from `self` — useful when the correspondence is meant to
start with a specific, authored first line rather than an empty history. It
delegates to `engine/command_executor.py`, so the shell path and the in-chat
`/reset` run exactly the same code.

### In-chat commands

Sent as a normal chat message, intercepted before the LLM ever sees them.
`/help` prints this list from inside the chat.

**Control commands** (`command_parser.py` / `command_executor.py`) — change runtime state:

| Command | Effect |
|---|---|
| `/list` | List available settings, marking the active one |
| `/switch <setting>` | Switch the active setting, then restart the process |
| `/reset` | Wipe the active setting's runtime state (see above) and start over |
| `/back N` | Roll back the last N exchanges |

**Where were we:**

| Command | Effect |
|---|---|
| `/recap` | The character's name and where the story stands — the same continuity notes the model is working from, falling back to the tail of the manuscript before the first compression |

The chat window is a view, not storage: the manuscript lives in
`runtime/chat_log.jsonl`, so clearing the chat (or switching settings and
coming back) costs nothing but the visible scrollback. `/recap` is how you
pick the thread back up.

**Debug commands** (`orchestrator.py`'s own dispatcher):

| Command | Effect |
|---|---|
| `/help` | List every command from both dispatchers |
| `/debug` | Last prompt's char budget breakdown |
| `/state` | Current session counters |
| `/user` | User model metadata |
| `/config` | Active compression config + counters |
| `/memory` | Retrieval memory state and what was recalled last turn |
| `/compress` | Force a compression pass (writes continuity + summaries) |
| `/regen` | Regenerate the last reply |
| `/reset_counters` | Zero the compression bookkeeping counters in `state.md` — does **not** touch chat history; for that use `/reset` |

**Photo commands** (same dispatcher, see Photo mode below):

| Command | Effect |
|---|---|
| `/shot [hint]` | Her phone, her point of view — her face is not in frame |
| `/scene [hint]` | An unseen observer in the room — her and the world around her |
| `/selfie [hint]` | Her phone pointed at herself — her face in frame |
| `/again [tweak]` | Re-render the last frame, optionally revising its prompt |
| `/photos [n]` | The last n photo requests: decision, prompt and note |

The optional hint points at a moment (`/scene that night on the roof`) — without one,
the frame is the present moment of the correspondence.

`/switch` can't simply mutate files under a running orchestrator the way
`/reset` does: the active setting is baked into a tree of objects built once at
startup (prompt builder, compressor, memory store, the LLM provider). So it
writes the new setting into `soul/config.yaml` and re-execs the whole process,
rebuilding everything from scratch — a few seconds of reconnect, chosen
deliberately over hot-swapping each cached object in place.

## Photo mode

Photo mode renders single frames from the world of the correspondence. Its
design note is the whole feature: **the reader asks for the frame, not the
character.** The two characters live inside their world and can see each
other; we are outside it, reading, and sometimes want to look. Everything
else follows from that.

- Nothing a photo request does is written back into the story —
  `chat_log.jsonl`, `memory.jsonl` and `continuity.md` are read-only here.
  The characters never learn a frame was taken.
- Because there is no fiction inside this circuit, the project's usual
  prompt rules don't apply to it: the describer works in English, in the
  third person, with JSON and format specs. It has the same status as the
  compressor — machinery, not story.
- When no frame is available, the answer is a third-person remark about the
  world ("no frame: she's in the dark and hasn't turned on the light"),
  never the character refusing the reader. A character who knows she's being
  watched and declines is exactly the assistant-shaped thing this project
  avoids.

### Three commands are three camera positions

| Command | Camera | Her face | Face artifact |
|---|---|---|---|
| `/shot` | her own phone, her point of view | not in frame | never sent |
| `/scene` | an unseen observer in the room | in frame, with the world | sent if present |
| `/selfie` | her phone, pointed at herself | in frame | required |

All four below are from the same run as the exchange above. The anchor is
authored once with `scripts/make_face.py` and committed like a bible file;
the three frames were rendered from the correspondence and are unedited.

| identity anchor | `/scene` | `/shot` | `/selfie` |
|---|---|---|---|
| <img src="soul/settings/sasha_alexis/characters/alexis/face.jpeg" width="200"> | <img src="examples/scene.png" width="200"> | <img src="examples/shot.png" width="200"> | <img src="examples/selfie.png" width="200"> |

The anchor carries identity and deliberately nothing else — flat light, no
makeup, grey background — and none of that reaches the frames: her own
clothes, her own light, her own framing. `/shot` has no face in it at all,
by policy, and what it shows is the spill she mentions in her first message.

### Enabling it for a setting

Photo mode is off in any setting without `soul/settings/<name>/images.yaml`,
regardless of the global `images.enabled` switch — the two mean different
things: `enabled: false` is "don't spend money", a missing `images.yaml` is
"not this pairing".

```yaml
# soul/settings/<name>/images.yaml
modes: [shot, scene, selfie]   # which camera positions are open here
model: x-ai/grok-imagine-image-2.0   # optional: this pairing's image model
llm:                                  # optional: this pairing's describer
  model: deepseek/deepseek-v3.2
```

Changing the image model is an expected move, not an edge case — one pairing
wants photorealism, another something cheaper or stranger. If you change it,
check what that model actually supports at
`https://openrouter.ai/api/v1/images/models`: reference-image support and the
number allowed vary per model (several accept none at all), and the catalog
served there does not overlap with `/api/v1/models`.

### The face artifact

`/selfie` and `/scene` send a character's face as a likeness reference so the
same person appears in every frame. It lives next to their bible as
`characters/<slug>/face.png` (or `.jpg`/`.jpeg` — image models return JPEG
and nothing here converts), and it is **authored**: generated once, chosen by
eye, committed to git, never regenerated automatically. A setting that opens
`selfie` without one fails at startup with a message saying so, rather than
rejecting every command later.

The anchor is deliberately unstyled — plain light, plain background, no
makeup, natural hair. Measured behaviour: the prompt reliably overrides the
reference's *mutable* attributes (hair colour, makeup, lenses) while identity
carries over, so an anchor carrying a look of its own would leak that look
into every frame. Makeup, hair and mood come from the story, per frame.

```bash
set -a; source .env; set +a
uv run python scripts/make_face.py                      # candidates for the active setting's `self`
uv run python scripts/make_face.py --slug alexis --n 4
uv run python scripts/make_face.py --extra "softer jawline, early 30s"
uv run python scripts/make_face.py --accept 2           # adopt candidate 2
```

It reads the character's bible, has the setting's describer turn it into an
English identity-anchor prompt, and renders candidates into
`debug/faces/<slug>/`. It never writes the artifact itself except on
`--accept`, and refuses to overwrite a face that already exists.

Each candidate costs whatever `images.model` charges per frame, at the slow,
no-reference path — there's no face yet to attach, so the default
(`x-ai/grok-imagine-image-2.0`) runs at ~85s and ~$0.06 per candidate, not
the ~15s a reference-conditioned frame gets once one exists. Three
candidates (the default `--n`) is a couple of minutes and about $0.18.

### Where output goes

Frames land in `soul/settings/<name>/runtime/photos/`, and every request — including
refusals — is appended to `photos.jsonl` with its decision, prompt, note, and
which reference was used. That file, not the images, is the feature's real
artifact: it's where you can read how the model understands the character
over time, the same role `summaries.jsonl` plays for compression. `/photos`
prints the tail of it from inside the chat.

Frames are never deleted by `/reset`: the story's state is the characters',
but the photographs are the reader's.

## Architecture

**Everything is an abstraction with implementation.** LLM provider, connector,
embedding provider and image provider are each defined by an interface
(`Protocol`). The current implementations are concrete (Mistral/OpenRouter,
Telegram via aiogram), but replacing one means adding a single new file plus a
line in a registry, not rewriting the core.

Memory is deliberately not a vector database: the retrieval store is an
append-only `memory.jsonl` plus a numpy matrix of embeddings, searched by cosine
similarity in process. Continuity is a plain markdown file the compressor
rewrites. Both are readable and editable by hand, which matters for a project
where tuning the character means reading exactly what the model was given.

Photo mode is a second engine running alongside the first rather than a feature
inside it. It has its own store, its own model calls and its own prompt, and it
touches the text circuit only by reading it. The isolation is an invariant, not
an optimisation: it's what lets the photo circuit use markers, JSON and English
freely without any of that leaking into the fiction.

Design decisions carry their reasoning at the site they govern: the comment
next to a config value says what was measured to land on it, and the comment
above a rule says what breaks without it. That includes the ideas that were
tried and rejected — those are the expensive part to rediscover.

## License

MIT — see [LICENSE](LICENSE).
