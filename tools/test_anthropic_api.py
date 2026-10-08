#!/usr/bin/env python3
"""Checks for patch 0010 (the Anthropic Messages frontend) against the patched source, with no model and no server.

They drive the four pieces the patch adds on the CPU: `translate` (a Messages body becomes the server's chat
request), `Reply` (chat chunks become Messages blocks), `read_body` (the bounded request framing) and
`matched_stop` (stop sequences reported on the reply). A pass here proves the frontend translates; serve the
patched image and post `/v1/messages` too when you have a machine with the model.

Usage: TF_SRC=/path/to/patched/src python3 tools/test_anthropic_api.py

TF_SRC is TensorFold's `src` directory with this recipe's patches applied (0010 adds the four modules).
Exit code 1 on any failed check.
"""
from __future__ import annotations

import json
import os
import sys
from email.message import Message
from io import BytesIO
from pathlib import Path


def _src() -> Path:
    src = Path(os.environ.get("TF_SRC", "") or "")
    if not src.is_dir():
        sys.exit("set TF_SRC to TensorFold's patched src directory")
    return src


class Handler:
    """Enough of the request handler for read_body: a buffered rfile, headers and the close flag."""

    def __init__(self, raw: bytes = b"", *, transfer: str | None = None,
                 lengths: tuple[str, ...] = (), version: str = "HTTP/1.1") -> None:
        self.rfile = BytesIO(raw)
        self.headers = Message()
        if transfer:
            self.headers["Transfer-Encoding"] = transfer
        for value in lengths:
            self.headers["Content-Length"] = value
        self.close_connection = False
        self.request_version = version


BASE = {"model": "test-model", "max_tokens": 128, "messages": [{"role": "user", "content": "Hi"}]}


def test_routes() -> None:
    from tensorfold.server.anthropic import route

    for path in ("/v1/messages", "/messages", "/v1/messages/", "/v1/messages?beta=true",
                 "/v1/messages/count_tokens", "/messages/count_tokens"):
        assert route(path), f"{path} must route to the Anthropic frontend"
    for path in ("/v1/chat/completions", "/v1/responses", "/v1/completions", "/health"):
        assert not route(path), f"{path} must not route to the Anthropic frontend"


def test_count_tokens() -> None:
    from types import SimpleNamespace as NS

    from tensorfold.server.anthropic import count_tokens

    app = NS(prepare=lambda chat, full: NS(prompt=list(range(7))))
    assert count_tokens(app, dict(BASE)) == 7, "the CUDA count must count the prepared prompt"


def test_translate_text() -> None:
    from tensorfold.server.anthropic_translate import translate
    from tensorfold.server.errors import RequestError

    chat = translate({"model": "m", "max_tokens": 64,
                      "system": [{"type": "text", "text": "A"}, {"type": "text", "text": "B"}],
                      "messages": [{"role": "user", "content": "Hi"}]})
    assert chat["model"] == "m" and chat["max_tokens"] == 64, chat
    assert chat["messages"][0] == {"role": "system", "content": "A\n\nB"}, chat["messages"][0]
    assert chat["messages"][1] == {"role": "user", "content": [{"type": "text", "text": "Hi"}]}, chat["messages"][1]
    # thinking off by default, and the sampling and stop fields pass through
    assert chat["chat_template_kwargs"] == {"enable_thinking": False} and chat["reasoning_effort"] == "none", chat
    chat = translate({**BASE, "stop_sequences": ["\nHuman:", "STOP"], "temperature": 0.3, "top_p": 0.9,
                      "top_k": 40, "stream": True})
    assert chat["stop"] == ["\nHuman:", "STOP"], chat
    assert (chat["temperature"], chat["top_p"], chat["top_k"]) == (0.3, 0.9, 40), chat
    assert chat["stream"] is True, chat
    for bad, needle in (({**BASE, "model": ""}, "model"),
                        ({**BASE, "messages": []}, "messages"),
                        ({**BASE, "max_tokens": 0}, "max_tokens"),
                        ({**BASE, "stop_sequences": ["", "x"]}, "stop_sequences")):
        try:
            translate(bad)
        except RequestError:
            pass
        else:
            raise AssertionError(f"expected a refusal: {bad}")


def test_translate_images() -> None:
    from tensorfold.server.anthropic_translate import translate
    from tensorfold.server.errors import RequestError

    body = {**BASE, "messages": [{"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "ZmFrZQ=="}},
        {"type": "text", "text": "What is in this picture?"}]}]}
    chat = translate(body)
    parts = chat["messages"][0]["content"]
    assert parts[0] == {"type": "image_url", "image_url": {"url": "data:image/png;base64,ZmFrZQ=="}}, parts
    assert parts[1] == {"type": "text", "text": "What is in this picture?"}, parts
    chat = translate({**BASE, "messages": [{"role": "user", "content": [
        {"type": "image", "source": {"type": "url", "url": "https://example.com/a.png"}}]}]})
    assert chat["messages"][0]["content"][0] == {"type": "image_url",
                                                 "image_url": {"url": "https://example.com/a.png"}}
    for source in ({"type": "file", "file_id": "f1"}, {"type": "base64", "media_type": "image/tiff", "data": "x"},
                   {}):
        try:
            translate({**BASE, "messages": [{"role": "user", "content": [{"type": "image", "source": source}]}]})
        except RequestError:
            pass
        else:
            raise AssertionError(f"expected a refusal: source {source}")


def test_translate_history() -> None:
    from tensorfold.server.anthropic_translate import translate

    chat = translate({**BASE, "system": "Rules", "messages": [
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "plan", "signature": ""},
            {"type": "tool_use", "id": "a", "name": "weather", "input": {"city": "Taipei"}},
            {"type": "tool_use", "id": "b", "name": "weather", "input": {"city": "Oslo"}}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "a", "content": [{"type": "text", "text": "sun"}]},
            {"type": "tool_result", "tool_use_id": "b", "content": "offline", "is_error": True},
            {"type": "text", "text": "Summarize"}]}]})
    msgs = chat["messages"]
    # tool results become their own messages before the user text of the same turn
    assert [m["role"] for m in msgs] == ["system", "assistant", "tool", "tool", "user"], [m["role"] for m in msgs]
    assistant = msgs[1]
    assert assistant["content"] == "" and assistant["reasoning_content"] == "plan", assistant
    assert [c["function"]["arguments"] for c in assistant["tool_calls"]] == ['{"city": "Taipei"}',
                                                                            '{"city": "Oslo"}'], assistant
    assert msgs[2]["content"] == "sun", msgs[2]
    assert msgs[3]["content"] == "Tool error: offline", msgs[3]
    assert msgs[2]["tool_call_id"] == "a" and msgs[3]["tool_call_id"] == "b", (msgs[2], msgs[3])
    assert msgs[4] == {"role": "user", "content": [{"type": "text", "text": "Summarize"}]}, msgs[4]
    # a tool result with an image keeps the parts form, with the error marker in front
    chat = translate({**BASE, "messages": [
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "a", "is_error": True, "content": [
            {"type": "text", "text": "broken"},
            {"type": "image", "source": {"type": "url", "url": "https://example.com/x.png"}}]}]}]})
    parts = chat["messages"][0]["content"]
    assert parts[0] == {"type": "text", "text": "Tool error: "}, parts
    assert parts[1] == {"type": "text", "text": "broken"}, parts
    assert parts[2] == {"type": "image_url", "image_url": {"url": "https://example.com/x.png"}}, parts


def test_translate_tools() -> None:
    from tensorfold.server.anthropic_translate import translate
    from tensorfold.server.errors import RequestError

    chat = translate({**BASE,
                      "tools": [{"name": "lookup", "description": "Look up.",
                                 "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}}],
                      "tool_choice": {"type": "any"}})
    assert chat["tools"] == [{"type": "function", "function": {
        "name": "lookup", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        "description": "Look up."}}], chat["tools"]
    assert chat["tool_choice"] == "required", chat
    chat = translate({**BASE, "tool_choice": {"type": "tool", "name": "lookup",
                                             "disable_parallel_tool_use": True}})
    assert chat["tool_choice"] == {"type": "function", "function": {"name": "lookup"}}, chat
    assert chat["parallel_tool_calls"] is False, chat
    for bad in ({"tools": [{"type": "web_search_20250305", "name": "web"}]},   # server-side tools
                {"tools": [{"name": "no_schema"}]},                             # no input_schema
                {"tools": "lookup"},                                            # not a list
                {"tool_choice": {"type": "one"}},                               # bad tool_choice.type
                {"tool_choice": {"type": "tool", "name": ""}}):                  # empty tool name
        try:
            translate({**BASE, **bad})
        except RequestError:
            pass
        else:
            raise AssertionError(f"expected a refusal: {bad}")


def test_translate_thinking_and_count() -> None:
    from tensorfold.server.anthropic_translate import translate
    from tensorfold.server.errors import RequestError

    chat = translate({**BASE, "max_tokens": 4096, "thinking": {"type": "enabled", "budget_tokens": 1024}})
    assert chat["chat_template_kwargs"] == {"enable_thinking": True}, chat
    assert chat["thinking_budget"] == 1024 and "reasoning_effort" not in chat, chat
    for bad in ({**BASE, "thinking": {"type": "enabled", "budget_tokens": 128}},          # budget >= max_tokens
                {**BASE, "thinking": {"type": "enabled", "budget_tokens": 0}},            # not positive
                {**BASE, "thinking": {"type": "enabled"}}):                               # no budget
        try:
            translate(bad)
        except RequestError:
            pass
        else:
            raise AssertionError(f"expected a refusal: {bad}")
    # a count only renders: it needs no max_tokens and keeps none
    chat = translate({**BASE, "thinking": {"type": "enabled", "budget_tokens": 1024}}, count=True)
    assert chat["max_tokens"] == 1, chat
    chat = translate({**BASE, "thinking": {"type": "adaptive"}})
    assert chat["chat_template_kwargs"] == {"enable_thinking": True} and "thinking_budget" not in chat, chat
    try:
        translate({**BASE, "output_config": {"effort": "maximum"}})
    except RequestError:
        pass
    else:
        raise AssertionError("an unknown effort must be refused")
    chat = translate({**BASE, "output_config": {"effort": "low", "format": {
        "type": "json_schema", "schema": {"type": "object"}}}})
    # thinking stays off unless the request asks for it, which keeps effort at "none"
    assert chat["reasoning_effort"] == "none" and chat["chat_template_kwargs"] == {"enable_thinking": False}, chat
    assert chat["response_format"]["json_schema"]["strict"] is True, chat["response_format"]


def test_translate_refusals() -> None:
    from tensorfold.server.anthropic_translate import translate
    from tensorfold.server.errors import RequestError

    for bad, needle in (({"model": "m", "max_tokens": 8, "messages": [{"role": "user",
                                                                       "content": [{"type": "redacted_thinking",
                                                                                    "data": "x"}]}]}, "redacted"),
                        ({**BASE, "messages": [{"role": "user", "content": "x"},
                                               {"role": "system", "content": "y", "clear_at": "turn"}]},
                         "turn-scoped"),
                        ({**BASE, "context_management": {"edits": [{"type": "compact"}]}}, "context_management"),
                        ({**BASE, "container": "x"}, "container"),
                        ({**BASE, "service_tier": "default"}, "service_tier")):
        try:
            translate(bad)
        except RequestError as exc:
            assert needle in str(exc), f"refused with {exc!r}, expected {needle!r}"
        else:
            raise AssertionError(f"expected a refusal containing {needle!r}: {bad}")
    # the one context_management Claude Code sends is kept, not refused
    chat = translate({**BASE, "context_management": {"edits": [{"type": "clear_thinking_20251015",
                                                               "keep": "all"}]}})
    assert "messages" in chat, chat
    # a mid-conversation system message keeps its role when the template can take it
    chat = translate({**BASE, "messages": [{"role": "user", "content": "x"},
                                           {"role": "system", "content": "y"},
                                           {"role": "assistant", "content": "z"},
                                           {"role": "user", "content": "again"}]})
    assert [m["role"] for m in chat["messages"]] == ["user", "system", "assistant", "user"], chat
    try:
        translate({**BASE, "messages": [{"role": "user", "content": "x"},
                                        {"role": "system", "content": "y"},
                                        {"role": "user", "content": "again"}]})
    except RequestError:
        pass
    else:
        raise AssertionError("a mid-conversation system before a user turn must be refused")


def test_usage_and_error() -> None:
    from tensorfold.server.anthropic_translate import error, usage

    assert usage({}) == {"input_tokens": 0, "output_tokens": 0, "cache_creation_input_tokens": 0,
                         "cache_read_input_tokens": 0}, usage({})
    mapped = usage({"prompt_tokens": 10, "completion_tokens": 5, "prompt_tokens_details": {"cached_tokens": 4},
                    "completion_tokens_details": {"reasoning_tokens": 2}})
    assert mapped["input_tokens"] == 6 and mapped["output_tokens"] == 5, mapped
    assert mapped["cache_read_input_tokens"] == 4 and mapped["cache_creation_input_tokens"] == 0, mapped
    assert mapped["output_tokens_details"] == {"thinking_tokens": 2}, mapped
    assert usage({"prompt_tokens": 2, "prompt_tokens_details": {"cached_tokens": 5}})["input_tokens"] == 0
    assert usage({"prompt_tokens": 2, "prompt_tokens_details": {"cached_tokens": -1}})["input_tokens"] == 2
    kinds = {400: "invalid_request_error", 404: "not_found_error", 503: "overloaded_error",
             500: "api_error"}
    for status, kind in kinds.items():
        assert error("x", status) == {"type": "error", "error": {"type": kind, "message": "x"}}, status


def test_reply_stream_text() -> None:
    from tensorfold.server.anthropic_translate import Reply

    events: list[dict] = []
    r = Reply("test-model", events.append)
    r.start()
    r.chunk({"choices": [{"index": 0, "delta": {"reasoning_content": "th"}}]})
    r.chunk({"choices": [{"index": 0, "delta": {"content": "Hel"}}]})
    r.chunk({"choices": [{"index": 0, "delta": {"content": "lo"}}]})
    r.chunk({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "stop_sequence": "STOP",
             "usage": {"prompt_tokens": 10, "completion_tokens": 5, "prompt_tokens_details": {"cached_tokens": 4},
                       "completion_tokens_details": {"reasoning_tokens": 2}}})
    r.chunk(None)
    assert events[0]["type"] == "message_start", events[0]
    assert events[0]["message"]["id"].startswith("msg_") and events[0]["message"]["role"] == "assistant"
    assert events[-2]["type"] == "message_delta" and events[-1]["type"] == "message_stop", events[-2:]
    kinds = [(e["type"], e.get("index"), (e.get("delta") or {}).get("type"), (e.get("content_block") or {}).get("type"))
             for e in events if e["type"].startswith("content_")]
    assert kinds[:4] == [("content_block_start", 0, None, "thinking"), ("content_block_delta", 0, "thinking_delta", None),
                        ("content_block_delta", 0, "signature_delta", None), ("content_block_stop", 0, None, None)], kinds
    assert kinds[4:8] == [("content_block_start", 1, None, "text"), ("content_block_delta", 1, "text_delta", None),
                         ("content_block_delta", 1, "text_delta", None), ("content_block_stop", 1, None, None)], kinds
    delta = events[-2]
    assert delta["delta"] == {"stop_reason": "stop_sequence", "stop_sequence": "STOP"}, delta
    assert delta["usage"]["input_tokens"] == 6 and delta["usage"]["cache_read_input_tokens"] == 4, delta
    assert delta["usage"]["output_tokens_details"] == {"thinking_tokens": 2}, delta
    events.clear()
    r.chunk({"choices": [{"index": 0, "delta": {"content": "late"}}]})   # finished: nothing more comes
    assert not events, events


def test_reply_stream_tools() -> None:
    from tensorfold.server.anthropic_translate import Reply

    events: list[dict] = []
    r = Reply("test-model", events.append)
    r.start()
    r.chunk({"choices": [{"index": 0, "delta": {"tool_calls": [
        {"index": 0, "id": "call_1", "function": {"name": "lookup", "arguments": '{"q'}}]}}]})
    r.chunk({"choices": [{"index": 0, "delta": {"tool_calls": [
        {"index": 0, "function": {"arguments": 'uery": "x"}'}}]}}]})
    r.chunk({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]})
    r.chunk(None)
    started = [e for e in events if e["type"] == "content_block_start"]
    assert len(started) == 1 and started[0]["content_block"] == {"type": "tool_use", "id": "call_1",
                                                                 "name": "lookup", "input": {}}, started
    partial = "".join(e["delta"]["partial_json"] for e in events
                      if e["type"] == "content_block_delta" and e["delta"]["type"] == "input_json_delta")
    assert json.loads(partial) == {"query": "x"}, partial
    assert events[-2]["delta"] == {"stop_reason": "tool_use", "stop_sequence": None}, events[-2]
    text = "".join(e["delta"].get("text", "") for e in events if e["type"] == "content_block_delta"
                   and e["delta"]["type"] == "text_delta")
    assert text == "", text


def test_reply_refusal_and_completion() -> None:
    from tensorfold.server.anthropic_translate import Reply
    from tensorfold.server.errors import RequestError

    events: list[dict] = []
    r = Reply("test-model", events.append)
    r.start()
    r.chunk({"choices": [{"index": 0, "delta": {"content": "open"}}]})
    r.chunk({"error": {"type": "invalid_request_error", "message": "bad"}})
    assert events[-1] == {"type": "error", "error": {"type": "invalid_request_error", "message": "bad"}}, events[-1]
    assert events[-2]["type"] == "content_block_stop", events[-2:]   # the open block is closed first
    assert r.finished, "an errored reply must stop here, and never finish normally"

    r = Reply("test-model", lambda event: None)
    message = r.completion({"choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
        "reasoning_content": "hmm", "content": "The answer.",
        "tool_calls": [{"id": "call_1", "type": "function",
                       "function": {"name": "lookup", "arguments": '{"q": "x"}'}}]}}],
        "usage": {"prompt_tokens": 4, "completion_tokens": 2}, "stop_sequence": "STOP"})
    assert [b["type"] for b in message["content"]] == ["thinking", "text", "tool_use"], message["content"]
    assert message["content"][2]["input"] == {"q": "x"}, message["content"][2]
    assert message["stop_reason"] == "stop_sequence" and message["stop_sequence"] == "STOP", message
    assert message["usage"]["input_tokens"] == 4, message["usage"]
    try:
        Reply("test-model", lambda event: None).completion(
            {"choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"content": "x", "tool_calls": [{"id": "c", "type": "function",
                                                                    "function": {"name": "f",
                                                                               "arguments": "[1, 2]"}}]}}]})
    except TypeError:
        pass
    else:
        raise AssertionError("tool arguments that are not a JSON object must raise")


def test_matched_stop() -> None:
    from tensorfold.server.stopping import matched_stop

    assert matched_stop("a STOP b XY", ("XY", "STOP")) == "STOP", "the first stop in the text wins"
    assert matched_stop("zabc", ("ab", "abc")) == "ab", "ties at one position follow request order"
    assert matched_stop("plain", ("STOP",)) is None
    assert matched_stop("STOPx", ()) is None


def test_read_body() -> None:
    from tensorfold.server.request_body import read_body
    from tensorfold.server.errors import RequestError

    def refused(handler: Handler, *, limit: int = 32 * 1024 ** 2, needle: str | None = None) -> None:
        try:
            read_body(handler, limit=limit)
        except RequestError as exc:
            if needle:
                assert needle in str(exc), f"refused with {exc!r}, expected {needle!r}"
            assert handler.close_connection, "a refused connection must not be reused"
        else:
            raise AssertionError(f"expected a refusal: {needle or 'bad framing'}")

    def read(handler: Handler, *, limit: int = 32 * 1024 ** 2) -> bytes:
        return read_body(handler, limit=limit)

    assert read(Handler(b"payload", lengths=("7",))) == b"payload"
    assert read(Handler(b"", lengths=())) == b"", "an empty body is legal"
    assert read(Handler(b"payload", lengths=("7", "7"))) == b"payload", "a repeated identical length is legal"
    assert read(Handler(b"5\r\nhello\r\n3;a=b\r\nfoo\r\n0\r\nX: y\r\n\r\n", transfer="chunked")) == b"hellofoo"
    assert read(Handler(b"0\r\n\r\n", transfer="chunked")) == b"", "a terminating chunk alone is legal"
    refused(Handler(b"pay", lengths=("7",)), needle="incomplete")
    refused(Handler(b"payload", lengths=("4", "5")), needle="conflicting")
    refused(Handler(b"payload", lengths=("x",)), needle="invalid Content-Length")
    refused(Handler(b"payload", transfer="chunked", lengths=("7",)), needle="combined")
    refused(Handler(b"payload", transfer="gzip"), needle="Transfer-Encoding")
    refused(Handler(b"zz\r\n", transfer="chunked"), needle="chunk size")
    refused(Handler(b"123456", lengths=("6",)), limit=4, needle="MiB limit")
    refused(Handler(b"5\r\nhello\r\n", transfer="chunked", version="HTTP/1.0"), needle="HTTP/1.1")


def main() -> int:
    src = _src()
    if not (src / "tensorfold" / "server" / "anthropic.py").is_file():
        sys.exit(f"{src} has no server/anthropic.py: apply patches/0010-server-anthropic-api.patch first")
    sys.path.insert(0, str(src))
    failed = 0
    for name, fn in (
        ("messages routes", test_routes),
        ("count_tokens on the CUDA path", test_count_tokens),
        ("translate: text, system and sampling", test_translate_text),
        ("translate: image blocks", test_translate_images),
        ("translate: tool and result history", test_translate_history),
        ("translate: tools and tool_choice", test_translate_tools),
        ("translate: thinking, effort and count", test_translate_thinking_and_count),
        ("translate: refusals", test_translate_refusals),
        ("usage and error shapes", test_usage_and_error),
        ("Reply: a streamed text reply", test_reply_stream_text),
        ("Reply: streamed tool calls", test_reply_stream_tools),
        ("Reply: refusals and the non-stream reply", test_reply_refusal_and_completion),
        ("matched_stop", test_matched_stop),
        ("read_body framing", test_read_body),
    ):
        try:
            fn()
            print(f"OK  {name}")
        except Exception as exc:
            failed += 1
            print(f"FAIL {name}: {exc}")
    print("FAIL" if failed else "OK", f"{failed} failed" if failed else "all Anthropic API checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
