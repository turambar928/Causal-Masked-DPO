#!/usr/bin/env python
"""Bounded, resumable API localization audit; never logs credentials or raw errors."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cmdpo.api_client import DEFAULT_QWEN_MODEL, load_api_config, make_client, stream_chat_completion
from cmdpo.data import read_jsonl
from scripts.localize_errors_api_judge import build_judge_prompt, _parse_json_object


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model", default=DEFAULT_QWEN_MODEL)
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--interval", type=float, default=6.0, help="Minimum seconds between requests")
    a = p.parse_args()
    config = load_api_config(preferred_model=a.model)
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    old = read_jsonl(out) if out.exists() else []
    if any(r.get("model") != a.model for r in old):
        raise ValueError("Use a separate output file for each judge model; historical labels must not be mixed")
    done = {r["sample_id"] for r in old if r.get("valid") and r.get("model") == a.model}
    rows = read_jsonl(a.input)[:a.limit]
    request_lock = threading.Lock()
    last_request = [0.0]
    stop_requests = threading.Event()

    def call(row):
        sample_id = hashlib.sha256((row["prompt"] + row["rejected"]).encode()).hexdigest()
        result = {"sample_id": sample_id, "model": a.model, "oracle": row["first_error_step"],
                  "template": row.get("metadata", {}).get("template"), "valid": False}
        if sample_id in done or stop_requests.is_set():
            return None
        try:
            with make_client(config, direct=True).with_options(max_retries=0) as client:
                for attempt in range(3):
                    with request_lock:
                        time.sleep(max(0, a.interval - (time.monotonic() - last_request[0])))
                        if stop_requests.is_set():
                            return None
                        last_request[0] = time.monotonic()
                    try:
                        response = stream_chat_completion(client, a.model,
                            build_judge_prompt(row["prompt"], row["answer"], row["rejected_steps"]),
                            temperature=0, max_tokens=512)
                        break
                    except Exception as exc:
                        if getattr(exc, "status_code", None) == 429 and attempt < 2:
                            retry_after = getattr(exc, "response", None)
                            seconds = float(retry_after.headers.get("retry-after", 60)) if retry_after else 60
                            with request_lock:
                                last_request[0] = time.monotonic() + min(max(seconds, 30), 120)
                            continue
                        raise
            raw = response.text
            parsed = _parse_json_object(raw)
            m = parsed.get("first_error_step")
            confidence = parsed.get("confidence")
            valid = (response.finish_reason == "stop" and type(m) is int and 0 <= m < len(row["rejected_steps"])
                     and isinstance(confidence, (int, float)) and 0 <= confidence <= 1)
            result.update(valid=valid, predicted=m, confidence=confidence, raw=raw,
                          exact=valid and m == row["first_error_step"],
                          usage=response.usage, served_model=response.served_model,
                          finish_reason=response.finish_reason, stream=True)
        except Exception as e:
            result.update(error_type=type(e).__name__, status=getattr(e, "status_code", None))
            if getattr(e, "status_code", None) in (401, 403):
                stop_requests.set()
        return result

    with out.open("a") as f, ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures = [pool.submit(call, row) for row in rows]
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
            if i % 10 == 0:
                print(f"Audited {i}/{len(rows)}", flush=True)
    latest = {r["sample_id"]: r for r in read_jsonl(out)}
    records = list(latest.values())
    summary = {"model": a.model, "requested": len(rows), "records": len(records),
               "valid": sum(r["valid"] for r in records), "exact": sum(r.get("exact", False) for r in records),
               "stopped_on_auth_error": stop_requests.is_set(), "stream": True,
               "warning": "API-versus-template-oracle audit; not human validation and not a real-data localization accuracy estimate"}
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
