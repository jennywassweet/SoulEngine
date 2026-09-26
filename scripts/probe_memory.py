"""
Probe the active setting's retrieval memory from the command line.

The point is to judge the *search* on its own, separately from how the model
then uses what it finds — if a fragment never surfaces here, no prompt wording
will fix it, and if it surfaces with a weak score, that's a min_score question,
not a model problem.

Results are printed regardless of memory.min_score, with a PASS/CUT marker,
so the threshold can be calibrated against real scores rather than guessed.
Cosine and recency are shown separately: cosine is "is this about the same
thing", recency is only how much age discounts it.

Usage:
    set -a; source .env; set +a
    uv run python scripts/probe_memory.py "what is my favourite colour"
    uv run python scripts/probe_memory.py --turn     # query the bot would actually send
    uv run python scripts/probe_memory.py            # store stats only

--turn is the honest test: a hand-typed question is a different shape from
what the bot sends (the last few messages of the exchange, in the same
register as what's stored), and question-to-statement retrieval scores
noticeably lower than statement-to-statement.
"""

import asyncio
import sys
from datetime import datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.config_utils import runtime_dir
from engine.embeddings.openrouter import OpenRouterEmbeddingProvider
from engine.memory_store import MemoryStore
from engine.retriever import Retriever

TOP_N_SHOWN = 10


def load() -> tuple[dict, MemoryStore, Retriever]:
    base = Path(__file__).resolve().parent.parent
    config = yaml.safe_load((base / "soul" / "config.yaml").read_text(encoding="utf-8"))
    setting = config.get("setting", "default")
    memory_config = config.get("memory", {})

    provider = OpenRouterEmbeddingProvider(model=memory_config.get("model", "openai/text-embedding-3-small"))
    runtime = runtime_dir(base, setting)
    store = MemoryStore(
        jsonl_path=runtime / "memory.jsonl",
        npy_path=runtime / "memory_vectors.npy",
        model_name=provider.model_name,
    )

    characters = config.get("characters", {})
    role_names = {
        "assistant": characters.get("self", "self").capitalize(),
        "user": characters.get("other", "other").capitalize(),
    }
    retriever = Retriever(store, provider, memory_config, role_names=role_names)
    return {"setting": setting, "memory": memory_config}, store, retriever


def print_stats(meta: dict, store: MemoryStore, retriever: Retriever) -> None:
    print(f"setting:  {meta['setting']}")
    print(f"model:    {store.model_name}")
    print(f"archived: {len(store)} messages" + ("  [DEGRADED — run reindex]" if store.degraded else ""))
    print(
        f"knobs:    top_k={retriever.top_k}  min_score={retriever.min_score}  "
        f"half_life={retriever.recency_half_life_days}d  max_chars={retriever.max_chars}"
    )


async def main() -> None:
    meta, store, retriever = load()
    print_stats(meta, store, retriever)

    if len(store) == 0:
        print("\nStore is empty — nothing to probe yet.")
        return

    if len(sys.argv) < 2:
        print("\nPass a query to probe, e.g.:")
        print('  uv run python scripts/probe_memory.py "what is my favourite colour"')
        print("  uv run python scripts/probe_memory.py --turn")
        return

    if sys.argv[1] == "--turn":
        from engine.chat_logger import ChatLogger

        base = Path(__file__).resolve().parent.parent
        chat_logger = ChatLogger(runtime_dir(base, meta["setting"]) / "chat_log.jsonl")
        history = chat_logger.read_all()
        if not history:
            print("\nChat log is empty — nothing to build a turn query from.")
            return
        # Same shape _generate_and_send builds: the window before the last
        # message, with the last message standing in for the incoming one.
        query = retriever.build_query_text(history[:-1], history[-1].get("content", ""))
        excluded = {e["message_id"] for e in history if e.get("message_id") is not None}
        print(f"\n(--turn: {len(excluded)} ids currently in chat_log are excluded from retrieval)")
    else:
        query = sys.argv[1]
        excluded = set()

    now = datetime.now()

    [vector] = await retriever.embedding_provider.embed([query])
    raw = store.search(vector, top_k=len(store), exclude_ids=excluded)  # cosine only, unfiltered by min_score

    shown_query = query if len(query) < 200 else query[:200] + " …"
    print(f'\nquery: "{shown_query}"\n')
    print(f"{'':4s} {'id':>5s} {'cosine':>7s} {'recency':>8s} {'score':>7s}  {'when':16s}  who: text")
    print("-" * 100)

    for entry, cosine in raw[:TOP_N_SHOWN]:
        recency = retriever._recency_weight(entry.get("timestamp", ""), now)
        gate = "PASS" if cosine >= retriever.min_score else "cut "
        when = entry.get("timestamp", "")[:16].replace("T", " ")
        who = retriever.role_names.get(entry.get("role", "user"), entry.get("role", "?"))
        text = " ".join(entry.get("content", "").split())[:60]
        print(
            f"{gate:4s} {entry.get('message_id', '?'):>5} {cosine:7.3f} {recency:8.3f} "
            f"{cosine * recency:7.3f}  {when:16s}  {who}: {text}"
        )

    passing = sum(1 for _, c in raw if c >= retriever.min_score)
    print(
        f"\n{passing} of {len(raw)} archived messages clear min_score={retriever.min_score}; "
        f"the prompt would show the top {retriever.top_k} of those, oldest first."
    )


if __name__ == "__main__":
    asyncio.run(main())
