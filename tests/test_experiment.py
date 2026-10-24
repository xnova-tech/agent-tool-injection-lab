import json
import urllib.error
import urllib.request
from collections.abc import Iterator

import pytest

from lab.api import ApiSettings, Budget, ChatReply, JsonObject, ModelClient, ToolCall, parse_reply
from lab.experiment import summarize
from lab.scenario import SCENARIOS
from lab.trial import run_trial


class FakeClient:
    def __init__(self, replies: Iterator[ChatReply]) -> None:
        self.replies = replies
        self.sent_messages: list[list[JsonObject]] = []

    def complete(self, messages: list[JsonObject], _tools: list[JsonObject]) -> ChatReply:
        self.sent_messages.append(messages.copy())
        return next(self.replies)


def reply(*calls: ToolCall, content: str | None = None) -> ChatReply:
    return ChatReply(
        "test-model", "tool_calls" if calls else "stop", content, {"prompt_tokens": 1, "completion_tokens": 1}, calls
    )


def read_call() -> ToolCall:
    return ToolCall("read", "get_incident", '{"incident_id":"INC-042"}')


def write_call(incident_id: str = "INC-042") -> ToolCall:
    return ToolCall("write", "update_incident", json.dumps({"incident_id": incident_id, "priority": "low"}))


@pytest.mark.parametrize(
    "name,allowed,priority",
    [("attack_unprotected", True, "low"), ("attack_protected", False, "high")],
)
def test_model_tool_loop_uses_actual_proposal(name: str, allowed: bool, priority: str) -> None:
    client = FakeClient(iter([reply(read_call()), reply(write_call()), reply(content="Готово.")]))
    trace = run_trial(client, "test-model", SCENARIOS[name], 1)

    assert trace.status == "completed"
    assert trace.read_delivered
    assert trace.unwanted_proposed
    assert trace.unwanted_allowed is allowed
    assert trace.priority_after == priority
    assert client.sent_messages[1][-1]["role"] == "tool"
    assert SCENARIOS[name].notes in str(client.sent_messages[1][-1]["content"])
    record = trace.to_record()
    assert record["turns"][1]["tool_calls"][0]["after_read"] is True
    assert "secret" not in str(record)


@pytest.mark.parametrize("incident_id,allowed", [("INC-042", True), ("INC-999", False)])
def test_early_unwanted_write_is_counted(incident_id: str, allowed: bool) -> None:
    client = FakeClient(iter([reply(write_call(incident_id)), reply(read_call()), reply(content="Готово.")]))
    trace = run_trial(client, "test-model", SCENARIOS["attack_unprotected"], 1)

    assert trace.status == "completed"
    assert trace.turns[0].tool_calls[0].after_read is False
    assert trace.unwanted_proposed
    assert trace.unwanted_allowed is allowed
    assert trace.unwanted_state_change is allowed


def test_invalid_tool_arguments_are_denied_and_recorded() -> None:
    client = FakeClient(iter([reply(ToolCall("bad", "update_incident", "[1,2]")), reply(content="Done")]))
    trace = run_trial(client, "test-model", SCENARIOS["attack_protected"], 1)

    assert trace.status == "no_read"
    assert trace.turns[0].tool_calls[0].arguments == {}
    assert not trace.unwanted_allowed
    assert trace.priority_after == "high"


def test_budget_rejects_before_call() -> None:
    with pytest.raises(RuntimeError, match="budget exhausted"):
        Budget(0.001).reserve(1000)


def test_summary_retains_incomplete_runs_in_denominator() -> None:
    completed = run_trial(
        FakeClient(iter([reply(read_call()), reply(content="Done")])),
        "test-model",
        SCENARIOS["attack_protected"],
        1,
    )
    failed = run_trial(FakeClient(iter([])), "test-model", SCENARIOS["attack_protected"], 2)
    row = summarize([completed, failed], Budget(2), "test-model")["scenarios"]["attack_protected"]
    assert row["runs"] == 2
    assert row["completed"] == 1
    assert row["errors"] == 1


def test_secret_is_hidden_from_settings_representation() -> None:
    assert "private-value" not in repr(ApiSettings("https://example.test/v1", "model", "private-value"))


def test_malformed_api_response_has_safe_error() -> None:
    with pytest.raises(RuntimeError, match="invalid shape") as error:
        parse_reply({"choices": [{"message": {"tool_calls": [{"function": {"arguments": "private-value"}}]}}]})
    assert "private-value" not in str(error.value)


def test_transport_error_hides_endpoint_and_key(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(_request: urllib.request.Request, timeout: int) -> None:
        assert timeout == 90
        raise urllib.error.URLError("https://private-host.test/v1/private-value")

    monkeypatch.setattr(urllib.request, "urlopen", fail)
    client = ModelClient(ApiSettings("https://private-host.test/v1", "test-model", "private-value"), Budget(2))
    with pytest.raises(RuntimeError, match="API connection failed") as error:
        client.complete([], [])
    assert "private-host.test" not in str(error.value)
    assert "private-value" not in str(error.value)
