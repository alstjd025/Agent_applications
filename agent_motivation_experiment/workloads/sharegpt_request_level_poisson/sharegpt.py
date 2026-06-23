"""ShareGPT dataset download + conversation flattening.

Kept separate from `workload.py` so the parsing logic is easy to unit
test without constructing a runner.
"""

import json
import random
from typing import Optional


# ShareGPT mirror on the Hub. The cleaned/split V3 dump is a single JSON
# array of {"id", "conversations": [{"from", "value"}, ...]}.
DEFAULT_HF_REPO = "anon8231489123/ShareGPT_Vicuna_unfiltered"
DEFAULT_HF_DATA_FILE = "ShareGPT_V3_unfiltered_cleaned_split.json"
# Pinned dataset revision: download is reproducible across machines even
# if the upstream repo is later updated. Override via the `hf_revision`
# workload-config key.
DEFAULT_HF_REVISION = "192ab2185289094fc556ec8ce5ce1e8e587154ca"
DEFAULT_NUM_CONVERSATIONS = 200
DEFAULT_MIN_HUMAN_TURNS = 1

_HUMAN_ROLES = {"human", "user"}
_ASSISTANT_ROLES = {"gpt", "chatgpt", "assistant", "bard", "bing"}


def download_sharegpt(
    repo: str, data_file: str, revision: Optional[str] = None
) -> list:
    """Fetch the ShareGPT JSON dump from the Hub and parse it.

    `revision` pins a commit/tag so the download is reproducible; None
    fetches the repo's current default branch.
    """
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=repo, filename=data_file, repo_type="dataset", revision=revision
    )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _normalize_conversation(turns: list) -> list[tuple[str, str]]:
    """Coerce one ShareGPT conversation to strict user/assistant alternation.

    Returns a list of (role, content) starting at a user turn. Stops at
    the first turn that breaks alternation or is empty — ShareGPT is the
    original data, replayed as-is (no injected system prompt).
    """
    msgs: list[tuple[str, str]] = []
    expected = "user"
    for t in turns:
        frm = str(t.get("from", "")).lower()
        if frm in _HUMAN_ROLES:
            role = "user"
        elif frm in _ASSISTANT_ROLES:
            role = "assistant"
        else:
            continue  # skip system/unknown turns
        if role != expected:
            break  # alternation broken -> truncate here
        content = (t.get("value") or "").strip()
        if not content:
            break
        msgs.append((role, content))
        expected = "assistant" if expected == "user" else "user"
    return msgs


def _truncate_human_turns(
    msgs: list[tuple[str, str]], max_human_turns: int
) -> list[tuple[str, str]]:
    """Keep msgs up to and including the ``max_human_turns``-th user turn."""
    out: list[tuple[str, str]] = []
    human = 0
    for role, content in msgs:
        if role == "user":
            human += 1
            if human > max_human_turns:
                break
        out.append((role, content))
    return out


def _conversation_requests(
    msgs: list[tuple[str, str]], conv_index: int
) -> list[dict]:
    """Emit one request per user turn; each prompt is the conversation prefix.

    Request k's messages = ``[u_1, a_1, ..., u_{k-1}, a_{k-1}, u_k]`` with
    the assistant turns being the recorded ShareGPT responses.
    """
    requests: list[dict] = []
    prefix: list[dict] = []
    human_turn = 0
    for role, content in msgs:
        prefix.append({"role": role, "content": content})
        if role == "user":
            human_turn += 1
            requests.append({
                "request_id": f"sg-{conv_index:05d}-t{human_turn:02d}",
                "conv_index": conv_index,
                "turn_index": human_turn,
                "messages": [dict(m) for m in prefix],
            })
    return requests


def flatten_sharegpt(
    raw: list,
    num_conversations: int,
    min_human_turns: int,
    max_human_turns: Optional[int],
    sample_seed: int,
) -> list[dict]:
    """Sample conversations deterministically and flatten them to requests."""
    rng = random.Random(sample_seed)
    order = list(range(len(raw)))
    rng.shuffle(order)

    records: list[dict] = []
    conv_count = 0
    for ri in order:
        if conv_count >= num_conversations:
            break
        conv = raw[ri]
        turns = conv.get("conversations") or conv.get("items") or []
        msgs = _normalize_conversation(turns)
        if max_human_turns:
            msgs = _truncate_human_turns(msgs, max_human_turns)
        n_human = sum(1 for role, _ in msgs if role == "user")
        if n_human < min_human_turns:
            continue
        records.extend(_conversation_requests(msgs, conv_count))
        conv_count += 1

    if not records:
        raise ValueError(
            "No usable ShareGPT conversations after filtering "
            f"(num_conversations={num_conversations}, "
            f"min_human_turns={min_human_turns})."
        )
    return records
