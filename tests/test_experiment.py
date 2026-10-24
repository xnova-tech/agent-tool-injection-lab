from collections.abc import Iterator
from typing import Any

import pytest

from lab import experiment
from lab.scenario import SCENARIOS


def fake_responses() -> Iterator[dict[str, Any]]:
    yield {
        "model": "test-model",
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "read",
                            "type": "function",
                            "function": {"name": "get_incident", "arguments": '{"incident_id":"INC-042"}'},
                        }
                    ],
                },
            }
        ],
    }
    yield {
        "model": "test-model",
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "write",
                            "type": "function",
                            "function": {
                                "name": "update_incident",
                                "arguments": '{"incident_id":"INC-042","priority":"low"}',
                            },
                        }
                    ],
                },
            }
        ],
    }
    yield {
        "model": "test-model",
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        "choices": [{"finish_reason": "stop", "message": {"content": "Готово."}}],
    }


@pytest.mark.parametrize(
    "name,allowed,priority",
    [
        ("attack_unprotected", True, "low"),
        ("attack_protected", False, "high"),
    ],
)
def test_model_tool_loop_uses_actual_proposal(
    monkeypatch: pytest.MonkeyPatch, name: str, allowed: bool, priority: str
) -> None:
    responses = fake_responses()
    sent_messages: list[list[dict[str, Any]]] = []

    def fake_chat(_url: str, _key: str, payload: dict[str, Any], _budget: experiment.Budget) -> dict[str, Any]:
        sent_messages.append(payload["messages"].copy())
        return next(responses)

    monkeypatch.setattr(experiment, "chat", fake_chat)
    trace = experiment.run_trial(
        "https://example.test/v1", "secret", "test-model", SCENARIOS[name], 1, experiment.Budget(2)
    )
    assert trace["status"] == "completed"
    assert trace["read_delivered"]
    assert trace["unwanted_proposed"]
    assert trace["unwanted_allowed"] is allowed
    assert trace["priority_after"] == priority
    assert sent_messages[1][-1]["role"] == "tool"
    assert SCENARIOS[name].notes in sent_messages[1][-1]["content"]
    assert "secret" not in str(trace)


def test_budget_rejects_before_call() -> None:
    budget = experiment.Budget(0.001)
    with pytest.raises(RuntimeError, match="budget exhausted"):
        budget.reserve(1000)
