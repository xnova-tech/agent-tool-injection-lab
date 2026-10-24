"""Configuration, budget accounting and the optional model transport."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

INPUT_USD_PER_TOKEN = 0.000002
OUTPUT_USD_PER_TOKEN = 0.000012
MAX_COMPLETION_TOKENS = 256
MAX_REQUEST_BYTES = 12000

type JsonObject = dict[str, object]


@dataclass(frozen=True, slots=True)
class ApiSettings:
    endpoint: str
    model: str
    key: str = field(repr=False)

    @classmethod
    def from_environment(cls) -> ApiSettings:
        """Read an optional local env file without exposing its path or contents."""
        values = dict(os.environ)
        env_file = values.get("LAB_ENV_FILE")
        if env_file:
            for line in Path(env_file).read_text(encoding="utf-8").splitlines():
                if "=" in line and not line.lstrip().startswith("#"):
                    name, value = line.split("=", 1)
                    values.setdefault(name.strip(), value.strip().strip("\"'"))
        endpoint = values.get("LAB_API_URL") or values.get("LLM.API_URL")
        key = values.get("LAB_API_KEY") or values.get("LLM.API_KEY")
        model = values.get("LAB_MODEL") or values.get("LLM.MODEL")
        if not endpoint or not key or not model:
            raise ValueError("LAB_API_URL, LAB_API_KEY and LAB_MODEL are required")
        if not endpoint.startswith("https://"):
            raise ValueError("API URL must use HTTPS")
        return cls(endpoint.rstrip("/"), model, key)


@dataclass(slots=True)
class Budget:
    """Reserve twice the listed price before each request."""

    limit_usd: float
    reserved_usd: float = 0.0
    estimated_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0

    def reserve(self, request_bytes: int) -> None:
        if request_bytes > MAX_REQUEST_BYTES:
            raise RuntimeError("request exceeds byte limit")
        ceiling = 2 * (request_bytes * INPUT_USD_PER_TOKEN + MAX_COMPLETION_TOKENS * OUTPUT_USD_PER_TOKEN)
        if self.reserved_usd + ceiling > self.limit_usd:
            raise RuntimeError("budget exhausted before API request")
        self.reserved_usd += ceiling

    def record(self, usage: JsonObject) -> None:
        prompt = _token_count(usage.get("prompt_tokens", 0))
        completion = _token_count(usage.get("completion_tokens", 0))
        self.input_tokens += prompt
        self.output_tokens += completion
        self.estimated_usd += prompt * INPUT_USD_PER_TOKEN + completion * OUTPUT_USD_PER_TOKEN


def _token_count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError("API response has invalid usage")
    return value


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    raw_arguments: str

    def as_message(self) -> JsonObject:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.raw_arguments},
        }


@dataclass(frozen=True, slots=True)
class ChatReply:
    model: str | None
    finish_reason: str | None
    content: str | None
    usage: JsonObject | None
    tool_calls: tuple[ToolCall, ...]


class ChatClient(Protocol):
    def complete(self, messages: list[JsonObject], tools: list[JsonObject]) -> ChatReply: ...


@dataclass(slots=True)
class ModelClient:
    settings: ApiSettings
    budget: Budget

    def complete(self, messages: list[JsonObject], tools: list[JsonObject]) -> ChatReply:
        payload: JsonObject = {
            "model": self.settings.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "max_completion_tokens": MAX_COMPLETION_TOKENS,
        }
        request_data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.budget.reserve(len(request_data))
        request = urllib.request.Request(
            self.settings.endpoint + "/chat/completions",
            data=request_data,
            headers={"Authorization": "Bearer " + self.settings.key, "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                body: object = json.load(response)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"API HTTP {exc.code}") from None
        except urllib.error.URLError as exc:
            raise RuntimeError(f"API connection failed ({type(exc.reason).__name__})") from None
        reply = parse_reply(body)
        self.budget.record(reply.usage or {})
        return reply


def parse_reply(body: object) -> ChatReply:
    """Validate only the fields needed by the tool loop; never echo response data in errors."""
    try:
        if not isinstance(body, dict):
            raise TypeError
        choices = body["choices"]
        if not isinstance(choices, list) or not choices:
            raise TypeError
        choice = choices[0]
        if not isinstance(choice, dict):
            raise TypeError
        message = choice["message"]
        if not isinstance(message, dict):
            raise TypeError
        raw_calls = message.get("tool_calls") or []
        if not isinstance(raw_calls, list):
            raise TypeError
        calls = tuple(_parse_tool_call(call) for call in raw_calls)
        content = message.get("content")
        model = body.get("model")
        finish_reason = choice.get("finish_reason")
        usage = body.get("usage")
        if content is not None and not isinstance(content, str):
            raise TypeError
        if model is not None and not isinstance(model, str):
            raise TypeError
        if finish_reason is not None and not isinstance(finish_reason, str):
            raise TypeError
        if usage is not None and not isinstance(usage, dict):
            raise TypeError
        return ChatReply(model, finish_reason, content, usage, calls)
    except (KeyError, TypeError, ValueError):
        raise RuntimeError("API response has invalid shape") from None


def _parse_tool_call(value: object) -> ToolCall:
    if not isinstance(value, dict) or not isinstance(value.get("function"), dict):
        raise TypeError
    function = value["function"]
    call_id, name, arguments = value.get("id"), function.get("name"), function.get("arguments")
    if not all(isinstance(item, str) for item in (call_id, name, arguments)):
        raise TypeError
    return ToolCall(call_id, name, arguments)
