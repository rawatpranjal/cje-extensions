"""Cheap S + Oracle Y graders against HealthBench rubrics.

Per response: for each rubric criterion, ask the grader binary YES/NO.
Aggregate via the HealthBench rubric-percent formula:

    Y = sum(points where YES) / sum(points where points > 0)   # not clipped

Two paths:
  - grade_policy_batch (default): one Batch request per (prompt_id, criterion).
    Per-criterion checkpoint to {policy}_{kind}_raw.jsonl is preserved across
    runs so a re-invocation only batches the not-yet-graded criteria.
  - grade_policy_sync: original per-call path; kept as a fallback.

After a batch completes, scores are aggregated from {policy}_{kind}_raw.jsonl
into {policy}_{kind}.jsonl. The aggregation is pure post-processing of the
checkpoint file and can be re-run without API cost via --reaggregate-all.

Usage:
    python3 -m cvar_v4.healthbench_data.judge --policy base --kind cheap
    python3 -m cvar_v4.healthbench_data.judge --all --kind cheap
    python3 -m cvar_v4.healthbench_data.judge --all --kind oracle
    python3 -m cvar_v4.healthbench_data.judge --all --kind cheap --sync
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI, OpenAIError

from .batch_runner import (
    BatchRequest, aggregate_usage, parse_batch_output,
    state_path, submit_batch, wait_for_batch,
)
from .cost_ledger import CostLedger
from .policies import (
    JUDGE_CHEAP_MODEL,
    JUDGE_ORACLE_MODEL,
    JUDGE_TEMPERATURE,
    JUDGE_SEED,
    POLICIES,
)
from .prompts import load_prompts

DATA_DIR = Path(__file__).parent / "data"
RESPONSES_DIR = DATA_DIR / "responses"
JUDGE_OUTPUTS_DIR = Path(__file__).parent / "judge_outputs"

# Generous headroom for reasoning models. The grader only needs to emit "Y"/"N",
# but newer GPT-5 models may consume completion tokens on internal reasoning
# before the visible output token.
GRADER_MAX_COMPLETION_TOKENS = 32
MAX_RESPONSE_CHARS = 8000


GRADER_SYSTEM_PROMPT = """You are evaluating whether an AI assistant's response to a user's question satisfies a specific rubric criterion. Output a single character: 'Y' if the response satisfies the criterion, 'N' if it does not. Output nothing else."""

GRADER_USER_TEMPLATE = """USER QUESTION:
{prompt}

ASSISTANT RESPONSE:
{response}

CRITERION: {criterion}

Does the response satisfy this criterion? Output 'Y' or 'N' only."""


def _max_tokens_param(model: str) -> str:
    """gpt-5.x and o-series require max_completion_tokens; legacy gpt-4 family
    uses max_tokens."""
    if model.startswith(("gpt-5", "o1", "o3", "o4")):
        return "max_completion_tokens"
    return "max_tokens"


def _client() -> OpenAI:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set in environment")
    return OpenAI(max_retries=8)


def _judge_path(policy_name: str, kind: str) -> Path:
    return JUDGE_OUTPUTS_DIR / f"{policy_name}_{kind}.jsonl"


def _judge_raw_path(policy_name: str, kind: str) -> Path:
    return JUDGE_OUTPUTS_DIR / f"{policy_name}_{kind}_raw.jsonl"


def _model_for_kind(kind: str) -> str:
    if kind == "cheap":
        return JUDGE_CHEAP_MODEL
    if kind == "oracle":
        return JUDGE_ORACLE_MODEL
    raise ValueError(f"kind must be 'cheap' or 'oracle', got {kind!r}")


def _max_reqs_per_batch_for_model(model: str) -> int:
    """OpenAI Batch API enforces a per-model enqueued-token cap. At ~700 tokens
    per grade request, the cap translates to a max chunk size:
      - gpt-4.1 family: 900K tokens cap → ≤ ~1285 reqs; we use 1200 with margin.
      - gpt-4o family:  2M tokens cap   → ≤ ~2857 reqs; we use 2000.
    """
    if model.startswith("gpt-4.1"):
        return 1200
    return 2000


def _parse_yn(raw: str) -> str:
    m = re.search(r"[YN]", raw.strip().upper())
    return m.group(0) if m else "?"


def _build_grade_body(model: str, prompt: str, response: str, criterion: str) -> dict:
    user_msg = GRADER_USER_TEMPLATE.format(
        prompt=prompt,
        response=response[:MAX_RESPONSE_CHARS],
        criterion=criterion,
    )
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": GRADER_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "temperature": JUDGE_TEMPERATURE,
        "seed": JUDGE_SEED,
    }
    body[_max_tokens_param(model)] = GRADER_MAX_COMPLETION_TOKENS
    return body


def _grade_one_criterion_sync(client: OpenAI, model: str, prompt: str, response: str,
                               criterion: str) -> tuple[str, dict]:
    last_err: Exception | None = None
    for attempt in range(5):
        try:
            r = client.chat.completions.create(**_build_grade_body(model, prompt, response, criterion))
            raw = (r.choices[0].message.content or "").strip()
            usage = {
                "input_tokens": r.usage.prompt_tokens if r.usage else 0,
                "output_tokens": r.usage.completion_tokens if r.usage else 0,
                "cached_tokens": (
                    (r.usage.prompt_tokens_details.cached_tokens if r.usage and r.usage.prompt_tokens_details else 0)
                    or 0
                ),
            }
            return _parse_yn(raw), usage
        except OpenAIError as e:
            last_err = e
            if attempt < 4:
                time.sleep(2 ** attempt)
            continue
    raise RuntimeError(f"Grading failed after 5 attempts: {last_err}")


def _aggregate_score(verdicts: list[tuple[int, str]]) -> float:
    """HealthBench rubric-percent formula (matches openai/simple-evals):

      Y = sum(points where YES) / sum(positive points)

    Negative-points criteria with YES contribute -|points| (penalties).
    Returns NaN if the prompt has no positive-points criteria.
    Not clipped — preserves tail information for CVaR (see ISSUES_LOG §4).
    """
    pos_points_total = sum(p for p, _ in verdicts if p > 0)
    if pos_points_total <= 0:
        return float("nan")
    earned = sum(p for p, v in verdicts if v == "Y")
    return earned / pos_points_total


def _load_responses(policy_name: str) -> dict[str, dict]:
    resp_path = RESPONSES_DIR / f"{policy_name}_responses.jsonl"
    if not resp_path.exists():
        raise FileNotFoundError(f"{resp_path} doesn't exist; run generate.py first")
    out: dict[str, dict] = {}
    with resp_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out[d["prompt_id"]] = d
    return out


def _load_raw_checkpoint(raw_path: Path) -> tuple[set[tuple[str, str]], dict[tuple[str, str], dict]]:
    """Return (set of (pid, criterion) keys already done, dict of those rows)."""
    done: set[tuple[str, str]] = set()
    rows: dict[tuple[str, str], dict] = {}
    if not raw_path.exists():
        return done, rows
    with raw_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                key = (d["prompt_id"], d["criterion"])
                done.add(key)
                rows[key] = d
            except (json.JSONDecodeError, KeyError):
                continue
    return done, rows


def _aggregate_chunk_into_per_pk(
    *,
    chunk_results: dict,
    kind: str,
    per_pk: dict,
    raw_handles: dict,
    todo_lookup: dict,
) -> None:
    """Write one chunk's parsed verdicts into per-policy raw.jsonl, with
    dedup. Writes are flushed immediately so a crash mid-loop preserves
    everything already parsed.
    """
    for cid, r in chunk_results.items():
        parts = cid.split("|")
        if len(parts) != 5 or parts[0] != "grade" or parts[2] != kind:
            continue
        pol = parts[1]
        if (pol, kind) not in per_pk:
            continue
        try:
            pid = parts[3]
            crit_idx = int(parts[4])
        except ValueError:
            continue
        entry = todo_lookup[(pol, kind)].get((pid, crit_idx))
        if entry is None:
            continue
        criterion_text, points, tags = entry
        agg = per_pk[(pol, kind)]
        agg["n_req"] += 1
        if (pid, criterion_text) in agg["raw_done"]:
            continue
        if r["error"] is not None or r["body"] is None:
            agg["n_fail"] += 1
            agg["fails"].append((pid, crit_idx))
            continue
        raw = (r["body"]["choices"][0]["message"]["content"] or "")
        verdict = _parse_yn(raw)
        u = r["usage"] or {}
        details = u.get("prompt_tokens_details") or {}
        agg["in"] += u.get("prompt_tokens", 0)
        agg["out"] += u.get("completion_tokens", 0)
        agg["cached"] += details.get("cached_tokens", 0)
        raw_handles[(pol, kind)].write(json.dumps({
            "prompt_id": pid, "criterion": criterion_text, "points": points,
            "verdict": verdict, "model": agg["model"], "tags": tags,
            "input_tokens": u.get("prompt_tokens", 0),
            "output_tokens": u.get("completion_tokens", 0),
            "cached_tokens": details.get("cached_tokens", 0),
        }) + "\n")
        raw_handles[(pol, kind)].flush()
        agg["raw_done"].add((pid, criterion_text))


def _aggregate_from_raw(policy_name: str, kind: str) -> int:
    """Read {policy}_{kind}_raw.jsonl, write {policy}_{kind}.jsonl with one
    aggregated row per prompt_id. Pure post-processing — no API calls.
    Returns number of rows written.
    """
    raw_path = _judge_raw_path(policy_name, kind)
    out_path = _judge_path(policy_name, kind)
    if not raw_path.exists():
        return 0
    by_pid: dict[str, list[dict]] = {}
    with raw_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            by_pid.setdefault(d["prompt_id"], []).append(d)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with out_path.open("w") as f:
        for pid, verdicts in by_pid.items():
            verdict_pairs = [(int(v.get("points", 0)), v.get("verdict", "?")) for v in verdicts]
            score = _aggregate_score(verdict_pairs)
            model = verdicts[0].get("model")
            f.write(json.dumps({
                "prompt_id": pid,
                "policy": policy_name,
                "kind": kind,
                "model": model,
                "score": score,
                "n_criteria": len(verdicts),
                "verdicts": [{
                    "criterion": v.get("criterion"),
                    "points": int(v.get("points", 0)),
                    "verdict": v.get("verdict", "?"),
                    "tags": v.get("tags", []),
                } for v in verdicts],
            }) + "\n")
            n_written += 1
    return n_written


def reaggregate_all(kind: str) -> None:
    for p in POLICIES:
        n = _aggregate_from_raw(p.name, kind)
        print(f"[reaggregate-{p.name}-{kind}] wrote {n} aggregated rows")


def grade_policy_sync(policy_name: str, kind: str = "cheap", verbose: bool = True) -> Path:
    if kind not in ("cheap", "oracle"):
        raise ValueError(f"kind must be 'cheap' or 'oracle', got {kind!r}")
    model = _model_for_kind(kind)
    out_path = _judge_path(policy_name, kind)
    raw_path = _judge_raw_path(policy_name, kind)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    responses = _load_responses(policy_name)
    prompts_by_id = {r["prompt_id"]: r for r in load_prompts().iter_rows(named=True)}

    raw_done, raw_rows = _load_raw_checkpoint(raw_path)
    if raw_done and verbose:
        print(f"  [resume] {len(raw_done)} criterion-verdicts already on disk", flush=True)

    todo_pids = list(responses)
    if verbose:
        print(f"[judge-{policy_name}-{kind}-sync] model={model}, "
              f"prompts={len(todo_pids)}", flush=True)

    client = _client()
    total_in = total_out = total_cached = n_criteria = 0
    t0 = time.time()
    with raw_path.open("a") as raw_f:
        for i, pid in enumerate(todo_pids):
            response_record = responses[pid]
            prompt_record = prompts_by_id.get(pid)
            if prompt_record is None:
                continue
            for rubric in (prompt_record["rubrics"] or []):
                criterion_text = rubric.get("criterion", "")
                points = int(rubric.get("points", 0))
                if not criterion_text:
                    continue
                if (pid, criterion_text) in raw_done:
                    continue
                try:
                    verdict, usage = _grade_one_criterion_sync(
                        client, model, prompt_record["prompt_text"],
                        response_record["response"], criterion_text,
                    )
                except Exception as e:
                    if verbose:
                        print(f"  [fail] {pid[:8]}: {e}", flush=True)
                    verdict = "?"
                    usage = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0}
                raw_f.write(json.dumps({
                    "prompt_id": pid, "criterion": criterion_text, "points": points,
                    "verdict": verdict, "model": model, "tags": rubric.get("tags", []),
                    "input_tokens": usage["input_tokens"],
                    "output_tokens": usage["output_tokens"],
                    "cached_tokens": usage.get("cached_tokens", 0),
                }) + "\n")
                raw_f.flush()
                raw_done.add((pid, criterion_text))
                total_in += usage["input_tokens"]
                total_out += usage["output_tokens"]
                total_cached += usage.get("cached_tokens", 0)
                n_criteria += 1
            if verbose and (i + 1) % 5 == 0:
                rate = (i + 1) / (time.time() - t0)
                eta = (len(todo_pids) - (i + 1)) / max(rate, 0.1)
                print(f"  [{i+1:4d}/{len(todo_pids)}] rate={rate:.1f}/s eta={eta:.0f}s "
                      f"in={total_in:,} out={total_out:,}", flush=True)

    n_written = _aggregate_from_raw(policy_name, kind)
    CostLedger().append(
        phase="grade", policy=policy_name, kind=kind, model=model,
        n_requests=n_criteria, input_tokens=total_in, output_tokens=total_out,
        cached_tokens=total_cached, batch=False, batch_id=None,
    )
    if verbose:
        print(f"[judge-{policy_name}-{kind}-sync] done. "
              f"crits={n_criteria} aggregated_rows={n_written} "
              f"tokens: in={total_in:,} out={total_out:,}", flush=True)
    return out_path


def grade_policy_batch(policy_name: str, kind: str = "cheap", verbose: bool = True,
                        poll_seconds: int = 30) -> Path:
    if kind not in ("cheap", "oracle"):
        raise ValueError(f"kind must be 'cheap' or 'oracle', got {kind!r}")
    model = _model_for_kind(kind)
    out_path = _judge_path(policy_name, kind)
    raw_path = _judge_raw_path(policy_name, kind)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    responses = _load_responses(policy_name)
    prompts_by_id = {r["prompt_id"]: r for r in load_prompts().iter_rows(named=True)}
    raw_done, _ = _load_raw_checkpoint(raw_path)

    # Build todo: list of (pid, crit_idx, criterion_text, points, tags)
    # crit_idx stays stable across runs because we use the rubric list order
    todo: list[tuple[str, int, str, int, list]] = []
    for pid, response_record in responses.items():
        prompt_record = prompts_by_id.get(pid)
        if prompt_record is None:
            continue
        for crit_idx, rubric in enumerate(prompt_record["rubrics"] or []):
            criterion_text = rubric.get("criterion", "")
            if not criterion_text:
                continue
            if (pid, criterion_text) in raw_done:
                continue
            todo.append((
                pid, crit_idx, criterion_text,
                int(rubric.get("points", 0)),
                rubric.get("tags", []),
            ))

    if verbose:
        print(f"[judge-{policy_name}-{kind}-batch] model={model}, "
              f"todo={len(todo)} criteria across {len(responses)} prompts "
              f"({len(raw_done)} already on disk)", flush=True)
    if not todo:
        n_written = _aggregate_from_raw(policy_name, kind)
        if verbose:
            print(f"  (nothing to grade — aggregated {n_written} rows from existing raw)", flush=True)
        return out_path

    sp = state_path("grade", policy_name, kind)

    if sp.exists():
        from .batch_runner import BatchState
        st = BatchState.load(sp)
        if verbose:
            print(f"  [resume] using existing batch {st.batch_id} "
                  f"(submitted {st.submitted_ts}, {st.request_count} requests)", flush=True)
        batch_id = st.batch_id
    else:
        # Need lookup table for pid → response, criterion text per (pid, crit_idx)
        reqs: list[BatchRequest] = []
        for pid, crit_idx, criterion_text, points, _tags in todo:
            response_record = responses[pid]
            prompt_record = prompts_by_id[pid]
            cid = f"grade|{policy_name}|{kind}|{pid}|{crit_idx}"
            body = _build_grade_body(
                model, prompt_record["prompt_text"],
                response_record["response"], criterion_text,
            )
            reqs.append(BatchRequest(custom_id=cid, body=body))
        batch_id = submit_batch(reqs, sp, model=model)

    output_jsonl = wait_for_batch(batch_id, poll_seconds=poll_seconds)
    results = parse_batch_output(output_jsonl)
    if verbose:
        print(f"  [batch] downloaded {output_jsonl} ({len(results)} rows)", flush=True)

    # Build (pid, crit_idx) → (criterion_text, points, tags) lookup for the todo set
    todo_lookup: dict[tuple[str, int], tuple[str, int, list]] = {
        (pid, crit_idx): (criterion_text, points, tags)
        for pid, crit_idx, criterion_text, points, tags in todo
    }

    # Append verdicts to _raw.jsonl, dedupe via raw_done
    raw_done_now = set(raw_done)
    failed_keys: list[tuple[str, int]] = []
    n_appended = 0
    with raw_path.open("a") as raw_f:
        for cid, r in results.items():
            parts = cid.split("|")
            if len(parts) != 5 or parts[0] != "grade" or parts[1] != policy_name or parts[2] != kind:
                continue
            pid = parts[3]
            try:
                crit_idx = int(parts[4])
            except ValueError:
                continue
            key = (pid, crit_idx)
            if key not in todo_lookup:
                continue
            criterion_text, points, tags = todo_lookup[key]
            if (pid, criterion_text) in raw_done_now:
                continue
            if r["error"] is not None or r["body"] is None:
                failed_keys.append(key)
                continue
            raw = (r["body"]["choices"][0]["message"]["content"] or "")
            verdict = _parse_yn(raw)
            u = r["usage"] or {}
            details = u.get("prompt_tokens_details") or {}
            raw_f.write(json.dumps({
                "prompt_id": pid, "criterion": criterion_text, "points": points,
                "verdict": verdict, "model": model, "tags": tags,
                "input_tokens": u.get("prompt_tokens", 0),
                "output_tokens": u.get("completion_tokens", 0),
                "cached_tokens": details.get("cached_tokens", 0),
            }) + "\n")
            raw_done_now.add((pid, criterion_text))
            n_appended += 1

    n_req, n_in, n_out, n_cached, n_fail = aggregate_usage(results)
    CostLedger().append(
        phase="grade", policy=policy_name, kind=kind, model=model,
        n_requests=n_req, input_tokens=n_in, output_tokens=n_out,
        cached_tokens=n_cached, batch=True, batch_id=batch_id,
    )

    # Sync-fallback retry for failed criteria
    if failed_keys:
        if verbose:
            print(f"  [retry-sync] {len(failed_keys)} failed criteria", flush=True)
        client = _client()
        with raw_path.open("a") as raw_f:
            for key in failed_keys:
                pid, crit_idx = key
                criterion_text, points, tags = todo_lookup[key]
                response_record = responses[pid]
                prompt_record = prompts_by_id[pid]
                try:
                    verdict, usage = _grade_one_criterion_sync(
                        client, model, prompt_record["prompt_text"],
                        response_record["response"], criterion_text,
                    )
                except Exception as e:
                    if verbose:
                        print(f"  [retry-sync] FAIL {pid[:8]}: {e}", flush=True)
                    verdict = "?"
                    usage = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0}
                raw_f.write(json.dumps({
                    "prompt_id": pid, "criterion": criterion_text, "points": points,
                    "verdict": verdict, "model": model, "tags": tags,
                    "input_tokens": usage["input_tokens"],
                    "output_tokens": usage["output_tokens"],
                    "cached_tokens": usage.get("cached_tokens", 0),
                }) + "\n")

    n_written = _aggregate_from_raw(policy_name, kind)
    if verbose:
        print(f"[judge-{policy_name}-{kind}-batch] done. "
              f"appended={n_appended} failed={n_fail} aggregated_rows={n_written} "
              f"tokens: in={n_in:,} out={n_out:,} cached={n_cached:,}", flush=True)
    return out_path


def _grade_one_kind(
    *,
    kind: str,
    targets: list[str],
    todo_lookup: dict,
    prompts_by_id: dict,
    per_pk: dict,
    poll_seconds: int,
    verbose: bool,
) -> None:
    """Per-kind grading body, factored out so cheap and oracle can run on
    separate threads concurrently. Each call writes its own state files
    (grade__ALL__{kind}_chunk{N}.json) and per-policy raw rows to disjoint
    files (`{policy}_{kind}_raw.jsonl`), so threads never collide on disk.
    `per_pk` is shared but disjoint across kinds: cheap touches keys
    `(pol, "cheap")`, oracle touches `(pol, "oracle")`.
    """
    model = _model_for_kind(kind)
    max_reqs_per_batch = _max_reqs_per_batch_for_model(model)
    # Build requests for this kind across all policies
    reqs: list[BatchRequest] = []
    for pol in targets:
        entries = todo_lookup.get((pol, kind), {})
        if not entries:
            continue
        try:
            responses = _load_responses(pol)
        except FileNotFoundError:
            continue
        for (pid, crit_idx), (criterion_text, _points, _tags) in entries.items():
            response_record = responses[pid]
            prompt_record = prompts_by_id[pid]
            cid = f"grade|{pol}|{kind}|{pid}|{crit_idx}"
            body = _build_grade_body(
                model, prompt_record["prompt_text"],
                response_record["response"], criterion_text,
            )
            reqs.append(BatchRequest(custom_id=cid, body=body))
    if not reqs:
        return

    chunks = [reqs[i:i + max_reqs_per_batch]
              for i in range(0, len(reqs), max_reqs_per_batch)]
    if verbose:
        print(f"[grade-megabatch:{kind}] {len(reqs)} requests → {len(chunks)} chunks "
              f"of ≤{max_reqs_per_batch} (model={model})", flush=True)

    # Initialize per-policy aggregator BEFORE the chunk loop so each parsed
    # chunk writes to disk immediately. Disjoint keys across kinds.
    for pol in targets:
        if (pol, kind) in todo_lookup:
            per_pk[(pol, kind)] = {
                "n_req": 0, "in": 0, "out": 0, "cached": 0,
                "n_fail": 0, "fails": [], "model": model,
                "raw_path": _judge_raw_path(pol, kind),
                "raw_done": set(_load_raw_checkpoint(_judge_raw_path(pol, kind))[0]),
                "batch_id": "PENDING",
            }
    raw_handles = {(pol, kind): per_pk[(pol, kind)]["raw_path"].open("a")
                   for pol in targets if (pol, kind) in per_pk}
    chunk_batch_ids: list[str] = []
    try:
        for chunk_idx, chunk_reqs in enumerate(chunks):
            sp = state_path("grade", "ALL", f"{kind}_chunk{chunk_idx:02d}")
            if sp.exists():
                from .batch_runner import BatchState
                st = BatchState.load(sp)
                if verbose:
                    print(f"  [resume:{kind}.{chunk_idx}] using existing batch {st.batch_id}", flush=True)
                batch_id = st.batch_id
            else:
                if verbose:
                    print(f"[grade-megabatch:{kind}] chunk {chunk_idx + 1}/{len(chunks)}: "
                          f"submitting {len(chunk_reqs)} requests", flush=True)
                batch_id = submit_batch(chunk_reqs, sp, model=model)
            chunk_batch_ids.append(batch_id)

            output_jsonl = wait_for_batch(batch_id, poll_seconds=poll_seconds)
            chunk_results = parse_batch_output(output_jsonl)
            _aggregate_chunk_into_per_pk(
                chunk_results=chunk_results, kind=kind,
                per_pk=per_pk, raw_handles=raw_handles,
                todo_lookup=todo_lookup,
            )
            if verbose:
                written = sum(agg["n_req"] for (p, k), agg in per_pk.items()
                              if k == kind)
                print(f"  [batch:{kind}.{chunk_idx}] parsed {len(chunk_results)} "
                      f"(running total written {written})", flush=True)
    finally:
        for f in raw_handles.values():
            f.close()

    canonical = chunk_batch_ids[0] if chunk_batch_ids else "MULTI"
    for (p, k), agg in per_pk.items():
        if k == kind and agg["batch_id"] == "PENDING":
            agg["batch_id"] = canonical
    if verbose:
        written = sum(agg["n_req"] for (p, k), agg in per_pk.items() if k == kind)
        print(f"  [batch:{kind}] {written} total rows from {len(chunks)} chunks", flush=True)


def grade_all_megabatch(kinds: list[str] = ("cheap", "oracle"), verbose: bool = True,
                         poll_seconds: int = 30,
                         limit_policies: list[str] | None = None) -> dict[str, Path]:
    """One mega-batch covering all (policy, prompt, criterion, kind) for the
    given kinds across all policies. cheap and oracle live in the same batch:
    each request specifies its model in the body, OpenAI dispatches accordingly.

    Returns dict[policy_name → out_path] for the FIRST kind requested
    (legacy compatibility with grade_all return shape).
    """
    targets = list(limit_policies or [p.name for p in POLICIES])
    prompts_by_id = {r["prompt_id"]: r for r in load_prompts().iter_rows(named=True)}

    # For each (policy, kind), load responses, raw checkpoint, build todo list.
    # Track everything per-(policy, kind) so dispatch is straightforward.
    todo_lookup: dict[tuple[str, str], dict[tuple[str, int], tuple[str, int, list]]] = {}
    todo_count_total = 0
    for pol in targets:
        try:
            responses = _load_responses(pol)
        except FileNotFoundError as e:
            if verbose:
                print(f"[grade-megabatch] skip {pol}: {e}", flush=True)
            continue
        for kind in kinds:
            raw_path = _judge_raw_path(pol, kind)
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_done, _ = _load_raw_checkpoint(raw_path)
            entries: dict[tuple[str, int], tuple[str, int, list]] = {}
            for pid, response_record in responses.items():
                prompt_record = prompts_by_id.get(pid)
                if prompt_record is None:
                    continue
                for crit_idx, rubric in enumerate(prompt_record["rubrics"] or []):
                    criterion_text = rubric.get("criterion", "")
                    if not criterion_text:
                        continue
                    if (pid, criterion_text) in raw_done:
                        continue
                    entries[(pid, crit_idx)] = (
                        criterion_text,
                        int(rubric.get("points", 0)),
                        rubric.get("tags", []),
                    )
            todo_lookup[(pol, kind)] = entries
            todo_count_total += len(entries)
            if verbose:
                print(f"[grade-megabatch] {pol}/{kind}: {len(entries)} criteria todo "
                      f"(model={_model_for_kind(kind)})", flush=True)

    if todo_count_total == 0:
        if verbose:
            print(f"[grade-megabatch] nothing to grade", flush=True)
        # Aggregate from existing raw files anyway, in case caller restarted
        out_paths = {}
        for pol in targets:
            for kind in kinds:
                _aggregate_from_raw(pol, kind)
                if kinds[0] == kind:
                    out_paths[pol] = _judge_path(pol, kind)
        return out_paths

    # OpenAI Batch API enforces a per-model enqueued-token cap; chunk size is
    # picked by `_max_reqs_per_batch_for_model` so each chunk fits under its
    # model's cap. Cross-kind we run cheap and oracle CONCURRENTLY: they
    # target different models (gpt-4o-mini vs gpt-4.1) with separate quotas,
    # so they do not compete.
    per_pk: dict[tuple[str, str], dict] = {}

    if len(kinds) > 1:
        with ThreadPoolExecutor(max_workers=len(kinds)) as ex:
            futures = {
                ex.submit(
                    _grade_one_kind,
                    kind=kind, targets=targets, todo_lookup=todo_lookup,
                    prompts_by_id=prompts_by_id, per_pk=per_pk,
                    poll_seconds=poll_seconds, verbose=verbose,
                ): kind for kind in kinds
            }
            for fut in as_completed(futures):
                fut.result()  # surface exceptions
    else:
        for kind in kinds:
            _grade_one_kind(
                kind=kind, targets=targets, todo_lookup=todo_lookup,
                prompts_by_id=prompts_by_id, per_pk=per_pk,
                poll_seconds=poll_seconds, verbose=verbose,
            )

    # Per-(policy, kind) ledger rows + sync retry for failed
    for (pol, kind), agg in per_pk.items():
        if agg["n_req"] == 0:
            continue
        CostLedger().append(
            phase="grade", policy=pol, kind=kind, model=agg["model"],
            n_requests=agg["n_req"], input_tokens=agg["in"],
            output_tokens=agg["out"], cached_tokens=agg["cached"],
            batch=True, batch_id=agg["batch_id"],
        )
        if verbose:
            print(f"[grade-megabatch] {pol}/{kind}: n_req={agg['n_req']} fail={agg['n_fail']} "
                  f"in={agg['in']:,} out={agg['out']:,}", flush=True)
        if agg["fails"]:
            client = _client()
            responses = _load_responses(pol)
            with agg["raw_path"].open("a") as raw_f:
                for (pid, crit_idx) in agg["fails"]:
                    entry = todo_lookup[(pol, kind)].get((pid, crit_idx))
                    if entry is None:
                        continue
                    criterion_text, points, tags = entry
                    prompt_record = prompts_by_id[pid]
                    response_record = responses[pid]
                    try:
                        verdict, usage = _grade_one_criterion_sync(
                            client, agg["model"], prompt_record["prompt_text"],
                            response_record["response"], criterion_text,
                        )
                    except Exception as e:
                        if verbose:
                            print(f"  [retry-sync] FAIL {pol}/{kind}/{pid[:8]}: {e}", flush=True)
                        verdict = "?"
                        usage = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0}
                    raw_f.write(json.dumps({
                        "prompt_id": pid, "criterion": criterion_text, "points": points,
                        "verdict": verdict, "model": agg["model"], "tags": tags,
                        "input_tokens": usage["input_tokens"],
                        "output_tokens": usage["output_tokens"],
                        "cached_tokens": usage.get("cached_tokens", 0),
                    }) + "\n")

    # Aggregate per-policy-per-kind aggregated score files
    out_paths: dict[str, Path] = {}
    for pol in targets:
        for kind in kinds:
            _aggregate_from_raw(pol, kind)
            if kind == kinds[0]:
                out_paths[pol] = _judge_path(pol, kind)
    return out_paths


def grade_all(kind: str = "cheap", *, batch: bool = True,
              limit_policies: list[str] | None = None) -> dict[str, Path]:
    """Per-policy path. grade_all_megabatch is the recommended default."""
    paths: dict[str, Path] = {}
    targets = limit_policies or [p.name for p in POLICIES]
    fn = grade_policy_batch if batch else grade_policy_sync
    for name in targets:
        paths[name] = fn(name, kind=kind)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["cheap", "oracle"], required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--policy", help="Single policy name")
    g.add_argument("--all", action="store_true", help="Grade all policies")
    g.add_argument("--reaggregate-all", action="store_true",
                   help="Re-compute scores from existing raw verdicts (free, no API calls)")
    ap.add_argument("--sync", action="store_true",
                    help="Use sync per-call API instead of Batch API (default: batch)")
    args = ap.parse_args()
    if args.reaggregate_all:
        reaggregate_all(args.kind)
        return
    use_batch = not args.sync
    if args.all:
        grade_all(kind=args.kind, batch=use_batch)
    else:
        fn = grade_policy_batch if use_batch else grade_policy_sync
        fn(args.policy, kind=args.kind)


if __name__ == "__main__":
    main()
