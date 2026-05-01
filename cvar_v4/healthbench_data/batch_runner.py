"""OpenAI Batch API runner with idempotent state-file resume.

Flow:
    1. submit_batch(reqs, state_path)  -> batch_id
       Uploads a JSONL of requests, creates a batch, persists state to
       state_path so a re-invocation does not double-submit.
    2. wait_for_batch(batch_id, ...)   -> local output JSONL path
       Polls every poll_seconds until status is terminal. On 'completed',
       downloads the output file. Raises on failed/expired/cancelled.
    3. parse_batch_output(output_path) -> {custom_id: {body, usage, error}}
       body is the chat-completion body if status_code==200; error is the
       OpenAI error object otherwise.

Resume rules:
    - If state_path exists and stored batch_id is in {validating, in_progress,
      finalizing}, submit_batch returns that batch_id without re-uploading.
    - If stored batch_id is 'completed', submit_batch returns it and
      wait_for_batch fetches the existing output (no re-download).
    - If stored batch_id is failed/expired/cancelled, submit_batch raises
      RuntimeError so the caller can decide whether to delete the state
      file and retry from scratch.

State file schema (JSON):
    {
      "batch_id": "batch_...",
      "input_file_id": "file_...",
      "submitted_ts": "2026-04-29T...",
      "request_count": 50,
      "model": "gpt-5.4-nano-2026-03-17",
      "endpoint": "/v1/chat/completions"
    }
"""
from __future__ import annotations

import io
import json
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import openai
from openai import OpenAI


BATCH_DIR = Path(__file__).parent / "batches"
TERMINAL_OK = {"completed"}
TERMINAL_BAD = {"failed", "expired", "cancelled"}
IN_FLIGHT = {"validating", "in_progress", "finalizing", "cancelling"}


def _client() -> OpenAI:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set in environment")
    return OpenAI()


def _retry_call(fn, *, label: str, max_retries: int = 8, base_delay: float = 5.0):
    """Retry a network call on transient connection / timeout errors.

    A long-running poll loop must not die on a single DNS hiccup, so we wrap
    the actual SDK call. Re-raises after max_retries so a genuinely broken
    state (auth error, etc.) still surfaces.
    """
    delay = base_delay
    for attempt in range(max_retries):
        try:
            return fn()
        except (openai.APIConnectionError, openai.APITimeoutError) as e:
            if attempt == max_retries - 1:
                raise
            print(f"[batch:retry] {label}: {type(e).__name__}: {e}; "
                  f"attempt {attempt + 1}/{max_retries}, sleep {delay:.0f}s",
                  flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 300.0)


@dataclass
class BatchRequest:
    custom_id: str
    body: dict


@dataclass
class BatchState:
    batch_id: str
    input_file_id: str
    submitted_ts: str
    request_count: int
    model: str
    endpoint: str

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> "BatchState":
        return cls(**json.loads(path.read_text()))


def state_path(phase: str, policy: str, kind: str | None) -> Path:
    """Stable on-disk location for the (phase, policy, kind) state file."""
    kind_part = kind if kind is not None else "NA"
    return BATCH_DIR / f"{phase}__{policy}__{kind_part}.json"


def submit_batch(
    reqs: list[BatchRequest],
    state_path: Path,
    *,
    model: str,
    endpoint: str = "/v1/chat/completions",
    completion_window: str = "24h",
) -> str:
    """Idempotent submit. Returns batch_id."""
    if not reqs:
        raise ValueError("submit_batch called with empty request list")

    if state_path.exists():
        state = BatchState.load(state_path)
        client = _client()
        b = _retry_call(
            lambda: client.batches.retrieve(state.batch_id),
            label=f"retrieve {state.batch_id[:20]}",
        )
        if b.status in IN_FLIGHT or b.status in TERMINAL_OK:
            print(f"[batch] resume {state.batch_id} (status={b.status}, "
                  f"requests={state.request_count})", flush=True)
            return state.batch_id
        if b.status in TERMINAL_BAD:
            raise RuntimeError(
                f"State file {state_path} points at batch {state.batch_id} "
                f"with terminal-bad status {b.status!r}. Delete the state "
                f"file to resubmit from scratch, or investigate the failure."
            )
        # Unknown status — fall through and resubmit, but warn.
        print(f"[batch] WARN: stale state file with unknown status {b.status!r}; "
              f"resubmitting from scratch", flush=True)

    client = _client()
    # Upload input JSONL (in-memory; the SDK accepts a bytes-like file object).
    buf = io.BytesIO()
    for r in reqs:
        line = json.dumps({
            "custom_id": r.custom_id,
            "method": "POST",
            "url": endpoint,
            "body": r.body,
        }) + "\n"
        buf.write(line.encode("utf-8"))
    buf.seek(0)
    upl = client.files.create(file=("batch_input.jsonl", buf, "application/jsonl"),
                              purpose="batch")
    batch = client.batches.create(
        input_file_id=upl.id,
        endpoint=endpoint,
        completion_window=completion_window,
    )
    state = BatchState(
        batch_id=batch.id,
        input_file_id=upl.id,
        submitted_ts=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        request_count=len(reqs),
        model=model,
        endpoint=endpoint,
    )
    state.save(state_path)
    print(f"[batch] submitted {batch.id} ({len(reqs)} requests, model={model})", flush=True)
    return batch.id


def wait_for_batch(
    batch_id: str,
    *,
    poll_seconds: int = 30,
    timeout_hours: int = 24,
) -> Path:
    """Poll until terminal. On completion, download output to a local file
    and return its path. Raises on failed/expired/cancelled."""
    client = _client()
    deadline = time.time() + timeout_hours * 3600
    last_status = None
    while True:
        b = _retry_call(
            lambda: client.batches.retrieve(batch_id),
            label=f"retrieve {batch_id[:20]}",
        )
        if b.status != last_status:
            counts = b.request_counts
            print(f"[batch] {batch_id} status={b.status} "
                  f"completed={counts.completed if counts else 0}"
                  f"/{counts.total if counts else '?'} "
                  f"failed={counts.failed if counts else 0}", flush=True)
            last_status = b.status
        if b.status in TERMINAL_OK:
            break
        if b.status in TERMINAL_BAD:
            raise RuntimeError(f"Batch {batch_id} ended with status {b.status!r}; "
                               f"errors={b.errors}")
        if time.time() > deadline:
            raise TimeoutError(f"Batch {batch_id} did not complete within "
                               f"{timeout_hours}h")
        time.sleep(poll_seconds)

    if b.output_file_id is None:
        raise RuntimeError(f"Batch {batch_id} completed but has no output_file_id")
    out_dir = BATCH_DIR / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{batch_id}.jsonl"
    if not out_path.exists():
        raw = _retry_call(
            lambda: client.files.content(b.output_file_id).read(),
            label=f"download {b.output_file_id}",
        )
        if isinstance(raw, str):
            raw = raw.encode("utf-8")
        out_path.write_bytes(raw)
    return out_path


def parse_batch_output(output_path: Path) -> dict[str, dict]:
    """{custom_id: {body, usage, error}}.

    body is the chat-completion body (dict) if status_code==200, else None.
    usage is body['usage'] if present, else None.
    error is the OpenAI error object (dict) if non-200, else None.
    """
    out: dict[str, dict] = {}
    with output_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            cid = d["custom_id"]
            resp = d.get("response")
            err = d.get("error")
            if resp and resp.get("status_code") == 200:
                body = resp.get("body")
                usage = body.get("usage") if body else None
                out[cid] = {"body": body, "usage": usage, "error": None}
            else:
                # Either a non-200 response or error is set
                out[cid] = {
                    "body": None,
                    "usage": None,
                    "error": err or (resp or {}).get("body"),
                }
    return out


def aggregate_usage(results: dict[str, dict]) -> tuple[int, int, int, int, int]:
    """Sum (n_requests, input_tokens, output_tokens, cached_tokens, n_failed)
    across all rows in the parsed batch output."""
    n_req = n_in = n_out = n_cached = n_fail = 0
    for cid, r in results.items():
        n_req += 1
        if r["error"] is not None or r["usage"] is None:
            n_fail += 1
            continue
        u = r["usage"]
        n_in += u.get("prompt_tokens", 0)
        n_out += u.get("completion_tokens", 0)
        details = u.get("prompt_tokens_details") or {}
        n_cached += details.get("cached_tokens", 0)
    return n_req, n_in, n_out, n_cached, n_fail
