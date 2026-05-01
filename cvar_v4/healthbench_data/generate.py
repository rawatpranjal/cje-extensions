"""Generate policy responses on HealthBench prompts via OpenAI.

Two paths:
  - generate_policy_batch (default): submits a batch via the Batch API, waits,
    parses, writes responses. 50% cheaper. Idempotent across crashes via
    state files in cvar_v4/healthbench_data/batches/.
  - generate_policy_sync: original per-call path, kept as a fallback for
    rows that fail in batch and for runs that need immediate output.

Idempotent: skips prompts already in the per-policy responses.jsonl.

Usage:
    python3 -m cvar_v4.healthbench_data.generate --policy base --limit 20
    python3 -m cvar_v4.healthbench_data.generate --all --limit 20
    python3 -m cvar_v4.healthbench_data.generate --all --limit 20 --sync
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from openai import OpenAI, OpenAIError

from .batch_runner import (
    BatchRequest, aggregate_usage, parse_batch_output,
    state_path, submit_batch, wait_for_batch,
)
from .cost_ledger import CostLedger
from .policies import POLICIES, Policy, get_policy
from .prompts import load_prompts

DATA_DIR = Path(__file__).parent / "data"
RESPONSES_DIR = DATA_DIR / "responses"

DEFAULT_MAX_COMPLETION_TOKENS = 800


def _max_tokens_param(model: str) -> str:
    """gpt-5.x and o-series require max_completion_tokens; legacy gpt-4 family
    uses max_tokens. Return the right key for the request body."""
    if model.startswith(("gpt-5", "o1", "o3", "o4")):
        return "max_completion_tokens"
    return "max_tokens"


def _client() -> OpenAI:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set in environment")
    return OpenAI()


def _response_path(policy_name: str) -> Path:
    return RESPONSES_DIR / f"{policy_name}_responses.jsonl"


def _existing_prompt_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    seen: set[str] = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                if d.get("prompt_id"):
                    seen.add(d["prompt_id"])
            except json.JSONDecodeError:
                continue
    return seen


def _build_request_body(policy: Policy, user_prompt: str,
                        max_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS) -> dict:
    """Chat-completions body shared by sync and batch paths."""
    body = {
        "model": policy.model,
        "messages": [
            {"role": "system", "content": policy.system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": policy.temperature,
        "seed": policy.seed,
    }
    body[_max_tokens_param(policy.model)] = max_tokens
    return body


def _generate_one_sync(client: OpenAI, policy: Policy, user_prompt: str,
                       max_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS) -> tuple[str, dict]:
    """Returns (response_text, usage_dict). Retries up to 3 times on transient errors."""
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            r = client.chat.completions.create(**_build_request_body(policy, user_prompt, max_tokens))
            text = r.choices[0].message.content or ""
            usage = {
                "input_tokens": r.usage.prompt_tokens if r.usage else 0,
                "output_tokens": r.usage.completion_tokens if r.usage else 0,
                "cached_tokens": (
                    (r.usage.prompt_tokens_details.cached_tokens if r.usage and r.usage.prompt_tokens_details else 0)
                    or 0
                ),
            }
            return text, usage
        except OpenAIError as e:
            last_err = e
            if attempt < 2:
                time.sleep(2 ** attempt)
            continue
    raise RuntimeError(f"Generation failed after 3 attempts: {last_err}")


def _make_record(prompt_row: dict, policy: Policy, text: str, usage: dict) -> dict:
    return {
        "prompt_id": prompt_row["prompt_id"],
        "prompt": prompt_row["prompt_text"],
        "response": text,
        "policy": policy.name,
        "model": policy.model,
        "temperature": policy.temperature,
        "metadata": {
            "judge_score": None,
            "judge_model": None,
            "oracle_label": None,
            "oracle_model": None,
            "theme": prompt_row.get("theme"),
            "subset": prompt_row.get("subset"),
            "_usage": usage,
        },
    }


def generate_policy_sync(policy_name: str, limit: int | None = None,
                          verbose: bool = True) -> Path:
    policy = get_policy(policy_name)
    out_path = _response_path(policy_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    prompts_df = load_prompts()
    if limit is not None:
        prompts_df = prompts_df.head(limit)

    seen = _existing_prompt_ids(out_path)
    todo = [r for r in prompts_df.iter_rows(named=True) if r["prompt_id"] not in seen]
    if verbose:
        print(f"[gen-{policy_name}-sync] {len(todo)}/{prompts_df.height} prompts to generate "
              f"(model={policy.model}, T={policy.temperature}, seed={policy.seed})", flush=True)
    if not todo:
        return out_path

    client = _client()
    total_in = total_out = total_cached = 0
    t0 = time.time()
    with out_path.open("a") as f:
        for i, row in enumerate(todo):
            try:
                text, usage = _generate_one_sync(client, policy, row["prompt_text"])
            except Exception as e:
                print(f"[gen-{policy_name}] FAIL prompt={row['prompt_id'][:8]}: {e}", flush=True)
                continue
            total_in += usage["input_tokens"]
            total_out += usage["output_tokens"]
            total_cached += usage.get("cached_tokens", 0)
            f.write(json.dumps(_make_record(row, policy, text, usage)) + "\n")
            f.flush()
            if verbose and (i + 1) % 5 == 0:
                rate = (i + 1) / (time.time() - t0)
                eta = (len(todo) - (i + 1)) / max(rate, 0.1)
                print(f"  [{i+1:4d}/{len(todo)}]  rate={rate:.1f}/s  eta={eta:.0f}s  in={total_in:,} out={total_out:,}",
                      flush=True)

    CostLedger().append(
        phase="generate", policy=policy_name, kind=None, model=policy.model,
        n_requests=len(todo), input_tokens=total_in, output_tokens=total_out,
        cached_tokens=total_cached, batch=False, batch_id=None,
    )
    if verbose:
        print(f"[gen-{policy_name}-sync] done. tokens: in={total_in:,} out={total_out:,}", flush=True)
    return out_path


def generate_policy_batch(policy_name: str, limit: int | None = None,
                          verbose: bool = True,
                          poll_seconds: int = 30) -> Path:
    policy = get_policy(policy_name)
    out_path = _response_path(policy_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    prompts_df = load_prompts()
    if limit is not None:
        prompts_df = prompts_df.head(limit)

    prompts_by_id: dict[str, dict] = {r["prompt_id"]: r for r in prompts_df.iter_rows(named=True)}
    seen = _existing_prompt_ids(out_path)
    todo_ids = [pid for pid in prompts_by_id if pid not in seen]
    if verbose:
        print(f"[gen-{policy_name}-batch] {len(todo_ids)}/{len(prompts_by_id)} prompts to generate "
              f"(model={policy.model})", flush=True)
    if not todo_ids:
        return out_path

    sp = state_path("generate", policy_name, None)

    # If a state file is missing but todo is non-empty, submit a fresh batch.
    # If a state file exists with an in-flight batch, resume.
    if sp.exists():
        # Resume: but the state's request_count may not match current todo_ids
        # (e.g., if some prompts were generated by sync between submit and now).
        # We trust the state file's batch — wait, parse, write only the rows
        # whose custom_id is still in todo. Any extras get ignored.
        from .batch_runner import BatchState
        st = BatchState.load(sp)
        if verbose:
            print(f"  [resume] using existing batch {st.batch_id} "
                  f"(submitted {st.submitted_ts}, {st.request_count} requests)", flush=True)
        batch_id = st.batch_id
    else:
        reqs = [
            BatchRequest(
                custom_id=f"gen|{policy_name}|{pid}",
                body=_build_request_body(policy, prompts_by_id[pid]["prompt_text"]),
            )
            for pid in todo_ids
        ]
        batch_id = submit_batch(reqs, sp, model=policy.model)

    output_jsonl = wait_for_batch(batch_id, poll_seconds=poll_seconds)
    results = parse_batch_output(output_jsonl)
    if verbose:
        print(f"  [batch] downloaded {output_jsonl} ({len(results)} rows)", flush=True)

    # Re-read seen in case sync wrote concurrently
    seen = _existing_prompt_ids(out_path)
    n_written = n_failed = 0
    failed_ids: list[str] = []
    with out_path.open("a") as f:
        for cid, r in results.items():
            # cid format: gen|{policy}|{prompt_id}
            parts = cid.split("|")
            if len(parts) != 3 or parts[0] != "gen" or parts[1] != policy_name:
                continue
            pid = parts[2]
            if pid in seen:
                continue
            if r["error"] is not None or r["body"] is None:
                n_failed += 1
                failed_ids.append(pid)
                continue
            text = r["body"]["choices"][0]["message"]["content"] or ""
            u = r["usage"] or {}
            details = u.get("prompt_tokens_details") or {}
            usage = {
                "input_tokens": u.get("prompt_tokens", 0),
                "output_tokens": u.get("completion_tokens", 0),
                "cached_tokens": details.get("cached_tokens", 0),
            }
            row = prompts_by_id.get(pid)
            if row is None:
                continue
            f.write(json.dumps(_make_record(row, policy, text, usage)) + "\n")
            n_written += 1
            seen.add(pid)

    n_req, n_in, n_out, n_cached, n_fail = aggregate_usage(results)
    CostLedger().append(
        phase="generate", policy=policy_name, kind=None, model=policy.model,
        n_requests=n_req, input_tokens=n_in, output_tokens=n_out,
        cached_tokens=n_cached, batch=True, batch_id=batch_id,
    )

    if verbose:
        print(f"[gen-{policy_name}-batch] done. wrote={n_written} failed={n_failed} "
              f"tokens: in={n_in:,} out={n_out:,} cached={n_cached:,}", flush=True)

    # Sync-fallback retry for failed rows
    if failed_ids:
        if verbose:
            print(f"  [retry-sync] {len(failed_ids)} failed rows", flush=True)
        client = _client()
        with out_path.open("a") as f:
            for pid in failed_ids:
                row = prompts_by_id.get(pid)
                if row is None:
                    continue
                try:
                    text, usage = _generate_one_sync(client, policy, row["prompt_text"])
                    f.write(json.dumps(_make_record(row, policy, text, usage)) + "\n")
                except Exception as e:
                    print(f"  [retry-sync] FAIL {pid[:8]}: {e}", flush=True)

    return out_path


def _safe_model_label(model: str) -> str:
    """Filename-safe model identifier for batch state paths."""
    return model.replace("/", "_")


def generate_all_megabatch(limit: int | None = None, verbose: bool = True,
                            poll_seconds: int = 30) -> dict[str, Path]:
    """One batch per UNIQUE MODEL (OpenAI Batch API requires single-model
    batches). Submits sequentially: model_A → wait → model_B → wait → ...

    For the legacy stack: 4/5 policies use gpt-4o-mini and 1 uses gpt-4.1, so
    this is 2 batches instead of 5 per-policy. Wall-clock = 2 × per-batch-time
    rather than n_policies × per-batch-time.

    Per-policy ledger rows are produced by parsing custom_ids in each batch's
    output, so cost reporting still has policy-level granularity.
    """
    prompts_df = load_prompts()
    if limit is not None:
        prompts_df = prompts_df.head(limit)
    prompts_by_id: dict[str, dict] = {r["prompt_id"]: r for r in prompts_df.iter_rows(named=True)}

    # Collect missing (policy, prompt_id) pairs, grouped by model.
    by_policy_todo: dict[str, list[str]] = {}
    paths: dict[str, Path] = {}
    for p in POLICIES:
        out = _response_path(p.name)
        out.parent.mkdir(parents=True, exist_ok=True)
        paths[p.name] = out
        seen = _existing_prompt_ids(out)
        by_policy_todo[p.name] = [pid for pid in prompts_by_id if pid not in seen]

    # Group by model: {model_id: list[(policy, prompt_id)]}
    by_model: dict[str, list[tuple[Policy, str]]] = {}
    for p in POLICIES:
        for pid in by_policy_todo[p.name]:
            by_model.setdefault(p.model, []).append((p, pid))

    total_todo = sum(len(v) for v in by_model.values())
    if verbose:
        for name, todo in by_policy_todo.items():
            print(f"[gen-megabatch] {name}: {len(todo)} missing", flush=True)
        print(f"[gen-megabatch] total: {total_todo} requests across {len(by_model)} model(s)", flush=True)
    if total_todo == 0:
        return paths

    # One batch per model. Submit and wait sequentially (could parallelize later).
    for model, items in by_model.items():
        if not items:
            continue
        sp = state_path("generate", _safe_model_label(model), None)
        if sp.exists():
            from .batch_runner import BatchState
            st = BatchState.load(sp)
            if verbose:
                print(f"  [resume:{model}] using existing batch {st.batch_id}", flush=True)
            batch_id = st.batch_id
        else:
            reqs = [
                BatchRequest(
                    custom_id=f"gen|{p.name}|{pid}",
                    body=_build_request_body(p, prompts_by_id[pid]["prompt_text"]),
                )
                for (p, pid) in items
            ]
            if verbose:
                print(f"[gen-megabatch:{model}] submitting {len(reqs)} requests", flush=True)
            batch_id = submit_batch(reqs, sp, model=model)

        output_jsonl = wait_for_batch(batch_id, poll_seconds=poll_seconds)
        results = parse_batch_output(output_jsonl)
        if verbose:
            print(f"  [batch:{model}] downloaded {output_jsonl} ({len(results)} rows)", flush=True)

        # Dispatch results from this model's batch to per-policy files.
        per_policy_usage: dict[str, dict] = {
            p.name: {"n_req": 0, "in": 0, "out": 0, "cached": 0, "n_fail": 0,
                     "model": p.model, "fails": []}
            for p in POLICIES if p.model == model
        }
        seen_now = {p.name: _existing_prompt_ids(paths[p.name]) for p in POLICIES if p.model == model}
        file_handles = {p.name: paths[p.name].open("a") for p in POLICIES if p.model == model}
        try:
            for cid, r in results.items():
                parts = cid.split("|")
                if len(parts) != 3 or parts[0] != "gen":
                    continue
                pol, pid = parts[1], parts[2]
                if pol not in per_policy_usage:
                    continue
                agg = per_policy_usage[pol]
                agg["n_req"] += 1
                if pid in seen_now[pol]:
                    continue
                if r["error"] is not None or r["body"] is None:
                    agg["n_fail"] += 1
                    agg["fails"].append(pid)
                    continue
                text = r["body"]["choices"][0]["message"]["content"] or ""
                u = r["usage"] or {}
                details = u.get("prompt_tokens_details") or {}
                agg["in"] += u.get("prompt_tokens", 0)
                agg["out"] += u.get("completion_tokens", 0)
                agg["cached"] += details.get("cached_tokens", 0)
                row = prompts_by_id.get(pid)
                if row is None:
                    continue
                file_handles[pol].write(json.dumps(_make_record(
                    row, get_policy(pol), text,
                    {"input_tokens": u.get("prompt_tokens", 0),
                     "output_tokens": u.get("completion_tokens", 0),
                     "cached_tokens": details.get("cached_tokens", 0)},
                )) + "\n")
                seen_now[pol].add(pid)
        finally:
            for f in file_handles.values():
                f.close()

        # Per-policy ledger rows for this model's batch
        for pol, agg in per_policy_usage.items():
            if agg["n_req"] == 0:
                continue
            CostLedger().append(
                phase="generate", policy=pol, kind=None, model=agg["model"],
                n_requests=agg["n_req"], input_tokens=agg["in"],
                output_tokens=agg["out"], cached_tokens=agg["cached"],
                batch=True, batch_id=batch_id,
            )
            if verbose:
                print(f"[gen-megabatch] {pol}: n_req={agg['n_req']} fail={agg['n_fail']} "
                      f"in={agg['in']:,} out={agg['out']:,}", flush=True)

        # Sync-fallback retry for failed rows in this batch
        failed_total = sum(len(a["fails"]) for a in per_policy_usage.values())
        if failed_total:
            if verbose:
                print(f"  [retry-sync:{model}] {failed_total} failed rows", flush=True)
            client = _client()
            for pol, agg in per_policy_usage.items():
                if not agg["fails"]:
                    continue
                policy = get_policy(pol)
                with paths[pol].open("a") as f:
                    for pid in agg["fails"]:
                        row = prompts_by_id.get(pid)
                        if row is None:
                            continue
                        try:
                            text, usage = _generate_one_sync(client, policy, row["prompt_text"])
                            f.write(json.dumps(_make_record(row, policy, text, usage)) + "\n")
                        except Exception as e:
                            print(f"  [retry-sync] FAIL {pol}/{pid[:8]}: {e}", flush=True)
    return paths


def generate_all(limit: int | None = None, *, batch: bool = True) -> dict[str, Path]:
    """Generate all 5 policies (per-policy path; megabatch is the recommended default)."""
    paths: dict[str, Path] = {}
    fn = generate_policy_batch if batch else generate_policy_sync
    for p in POLICIES:
        paths[p.name] = fn(p.name, limit=limit)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--policy", help="Single policy name to generate")
    g.add_argument("--all", action="store_true", help="Generate all 5 policies")
    ap.add_argument("--limit", type=int, default=None, help="Process only first N prompts")
    ap.add_argument("--sync", action="store_true",
                    help="Use sync per-call API instead of Batch API (default: batch)")
    args = ap.parse_args()
    use_batch = not args.sync
    if args.all:
        generate_all(limit=args.limit, batch=use_batch)
    else:
        fn = generate_policy_batch if use_batch else generate_policy_sync
        fn(args.policy, limit=args.limit)


if __name__ == "__main__":
    main()
