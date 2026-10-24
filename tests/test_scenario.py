from dataclasses import replace

import pytest

from lab.offline import run
from lab.scenario import INCIDENT_ID, SCENARIOS, Grant, IncidentStore, decide_update, execute_tool, tool_definitions


def test_offline_expected_output() -> None:
    assert run() == (
        "attack_unprotected: applied, priority=low\n"
        "attack_protected: denied, priority=high\n"
        "attack_no_write_tool: denied, priority=high\n"
        "legitimate_write: applied, priority=low\n"
        "wrong_id: denied, priority=high\n"
        "mismatched_grant: denied, priority=high\n"
    )


@pytest.mark.parametrize(
    "scenario_name,allowed",
    [
        ("attack_unprotected", True),
        ("attack_protected", False),
        ("attack_no_write_tool", False),
        ("legitimate_write", True),
    ],
)
def test_exact_update_authority(scenario_name: str, allowed: bool) -> None:
    scenario = SCENARIOS[scenario_name]
    store = IncidentStore(scenario.notes)
    decision, _ = execute_tool(scenario, store, "update_incident", {"incident_id": INCIDENT_ID, "priority": "low"})
    assert decision.allowed is allowed
    assert store.priority == ("low" if allowed else "high")


@pytest.mark.parametrize(
    "arguments",
    [
        {"incident_id": "INC-999", "priority": "low"},
        {"incident_id": INCIDENT_ID, "priority": "urgent"},
        {"incident_id": INCIDENT_ID},
    ],
)
def test_invalid_arguments_cannot_change_state(arguments: dict[str, str]) -> None:
    scenario = SCENARIOS["attack_unprotected"]
    store = IncidentStore(scenario.notes)
    decision, _ = execute_tool(scenario, store, "update_incident", arguments)
    assert not decision.allowed
    assert store.priority == "high"


def test_grant_must_match_exact_arguments() -> None:
    scenario = replace(SCENARIOS["legitimate_write"], grant=Grant(INCIDENT_ID, "high"))
    assert not decide_update(scenario, {"incident_id": INCIDENT_ID, "priority": "low"}).allowed


def test_read_returns_only_synthetic_data() -> None:
    scenario = SCENARIOS["attack_protected"]
    store = IncidentStore(scenario.notes)
    decision, result = execute_tool(scenario, store, "get_incident", {"incident_id": INCIDENT_ID})
    assert decision.allowed
    assert result["notes"] == scenario.notes
    assert result["priority"] == "high"
    assert store.priority == "high"


def test_write_schema_omitted_when_unavailable() -> None:
    assert [item["function"]["name"] for item in tool_definitions(False)] == ["get_incident"]
