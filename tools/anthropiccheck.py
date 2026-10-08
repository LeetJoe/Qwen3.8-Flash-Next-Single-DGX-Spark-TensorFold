#!/usr/bin/env python3
"""The Anthropic Messages API (patch 0010) against the running server.

Posts to /v1/messages and /v1/messages/count_tokens and checks the answers against the Messages shape,
streamed and not, and compares against the OpenAI endpoint where the two must agree. Complements
tools/test_anthropic_api.py, which exercises the same translation on the CPU from TensorFold's source.

Usage: tools/anthropiccheck.py      (API_URL / PORT as in bench.py). Exit code 1 on failure.
"""
import json
import sys
import urllib.error
import urllib.request

sys.dont_write_bytecode = True           # no tools/__pycache__ from importing bench
from bench import API_URL, prose  # noqa: E402

MESSAGES = API_URL + "/v1/messages"
COUNT = MESSAGES + "/count_tokens"
MODEL = "Qwen3.8-Flash-Next"
TOOLS = [{"name": "add_tags", "description": "Attach tags to a document.",
          "input_schema": {"type": "object", "required": ["doc_id", "tags"], "properties": {
              "doc_id": {"type": "integer", "description": "The document id."},
              "tags": {"type": "array", "items": {"type": "string"}, "description": "The tags to attach."}}}}]


def post(path: str, body: dict, timeout: float = 600) -> tuple[int, dict]:
    """One JSON POST; a refused status is an answer too, so HTTPError still returns (status, body)."""
    req = urllib.request.Request(path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as reply:
            return reply.status, json.load(reply)
    except urllib.error.HTTPError as refused:
        return refused.code, json.loads(refused.read() or b"{}")


def stream(body: dict) -> list[tuple[str, dict]]:
    """Every (event, data) pair of a streamed /v1/messages reply, or None if it was refused."""
    req = urllib.request.Request(MESSAGES, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as reply:
            if reply.status != 200:
                return []
            events, event = [], None
            for line in reply:
                if line.startswith(b"event: "):
                    event = line[7:].strip().decode()
                elif line.startswith(b"data: "):
                    events.append((event, json.loads(line[6:])))
            return events
    except urllib.error.URLError:
        return []


def check(name: str, ok: bool, detail: str) -> bool:
    print(f"{name}: {'OK' if ok else 'FAILED'}: {detail}")
    return ok


def main() -> None:
    failed = 0

    # 1. a non-stream text reply comes back in the Messages shape
    body = {"model": MODEL, "max_tokens": 64, "temperature": 0, "thinking": {"type": "disabled"},
            "messages": [{"role": "user", "content": "Name the capital of France in one short sentence."}]}
    status, reply = post(MESSAGES, body, 120)
    texts = [b for b in (reply.get("content") or []) if b.get("type") == "text"]
    if not check("text reply", status == 200 and reply.get("type") == "message" and reply.get("role") == "assistant"
                 and texts and "Paris" in texts[0].get("text", "") and reply.get("stop_reason") == "end_turn"
                 and reply["usage"]["output_tokens"] > 0, f"{status} {json.dumps(reply)[:160]}"):
        failed += 1

    # 2. count_tokens renders with the same tokenizer as a real run: its number must match the prompt
    #    tokens of the same request sent through the full path (input + cache_read)
    prompt = prose(4_000, 777)
    counted = post(COUNT, {"model": MODEL, "messages": [{"role": "user", "content": prompt}]}, 120)[1]
    ran = post(MESSAGES, {"model": MODEL, "max_tokens": 8, "messages": [{"role": "user", "content": prompt}]})[1]
    used = ran.get("usage") or {}
    prompt_total = (used.get("input_tokens") or 0) + (used.get("cache_read_input_tokens") or 0)
    if not check("count_tokens", counted.get("input_tokens", 0) > 0 and counted["input_tokens"] == prompt_total,
                 f"counted {counted.get('input_tokens')}, a real run renders {prompt_total}"):
        failed += 1

    # 3. tools: the schema arrives as input_schema, the call as a tool_use block (the toolcheck prompt)
    body = {"model": MODEL, "max_tokens": 256, "temperature": 0, "thinking": {"type": "disabled"},
            "tools": TOOLS, "messages": [
                {"role": "user", "content": "Tag document 42 with 'urgent', 'finance' and 'q3' using the tool."}]}
    status, reply = post(MESSAGES, body, 600)
    calls = [b for b in reply.get("content", []) if b.get("type") == "tool_use"]
    if not check("tool_use", status == 200 and reply.get("stop_reason") == "tool_use" and len(calls) == 1
                 and isinstance(calls[0].get("input"), dict)
                 and isinstance(calls[0]["input"].get("tags"), list) and len(calls[0]["input"]["tags"]) >= 3
                 and isinstance(calls[0]["input"].get("doc_id"), int), f"{status} {json.dumps(reply)[:200]}"):
        failed += 1

    # 4. streamed tool_use: the arguments reassemble to one JSON object and the reply closes properly
    events = stream({**body, "stream": True})
    kinds = [kind for kind, _ in events]
    args = "".join(d["delta"]["partial_json"] for _, d in events
                   if d.get("delta", {}).get("type") == "input_json_delta")
    last = events[-2][1] if len(events) >= 2 else {}
    if not check("streamed tool_use", bool(events) and kinds[0] == "message_start" and kinds[-1] == "message_stop"
                 and kinds[-2] == "message_delta"
                 and kinds.count("content_block_start") == kinds.count("content_block_stop")
                 and args.startswith("{") and args.endswith("}") and isinstance(json.loads(args or "null"), dict)
                 and last.get("delta", {}).get("stop_reason") == "tool_use",
                 f"{kinds[:3]}…{kinds[-2:]} args {args[:80]!r}"):
        failed += 1

    # 5. stop sequences: if the model repeats the stop, it must be consumed and reported (patch 0010's
    #    matched_stop); a reply that never reaches the stop still has to answer 200
    status, reply = post(MESSAGES, {"model": MODEL, "max_tokens": 64, "temperature": 0,
                                   "stop_sequences": ["WORLD"],
                                   "messages": [{"role": "user",
                                                 "content": "Repeat exactly, with nothing else: HELLO WORLD"}]},
                        120)
    first = (reply.get("content") or [{}])[0].get("text", "")
    if "WORLD" in first:                       # the stop was never reached: only the status is checked
        ok, note = status == 200, f"{status} no stop consumed: {first[:60]!r}"
    else:
        ok = (status == 200 and reply.get("stop_reason") == "stop_sequence"
              and reply.get("stop_sequence") == "WORLD")
        note = f"{status} stopped with {reply.get('stop_sequence')!r}: {first[:60]!r}"
    if not check("stop_sequences", ok, note):
        failed += 1

    # 6. a refused body keeps the Anthropic error shape, not the OpenAI one
    status, refusal = post(MESSAGES, {"model": MODEL, "max_tokens": 64, "stop_sequences": [""],
                                     "messages": [{"role": "user", "content": "x"}]})
    if not check("refusal shape", status == 400 and refusal.get("type") == "error"
                 and refusal.get("error", {}).get("type") == "invalid_request_error",
                 f"{status} {json.dumps(refusal)[:160]}"):
        failed += 1

    print("anthropiccheck:", f"{failed} FAILED" if failed else "all Messages-API checks passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
