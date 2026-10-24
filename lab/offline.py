"""Deterministic executor controls; no network or model is used."""

from __future__ import annotations

from lab.scenario import INCIDENT_ID, SCENARIOS, Grant, IncidentStore, Scenario, execute_tool


def run() -> str:
    lines: list[str] = []
    proposal = {"incident_id": INCIDENT_ID, "priority": "low"}
    for name in ("attack_unprotected", "attack_protected", "attack_no_write_tool", "legitimate_write"):
        scenario = SCENARIOS[name]
        store = IncidentStore(scenario.notes)
        decision, _ = execute_tool(scenario, store, "update_incident", proposal)
        lines.append(f"{name}: {'applied' if decision.allowed else 'denied'}, priority={store.priority}")
    for name, scenario, args in (
        ("wrong_id", SCENARIOS["legitimate_write"], {"incident_id": "INC-999", "priority": "low"}),
        ("mismatched_grant", Scenario("mismatched_grant", "", "", True, True, Grant(INCIDENT_ID, "high")), proposal),
    ):
        store = IncidentStore(scenario.notes)
        decision, _ = execute_tool(scenario, store, "update_incident", args)
        lines.append(f"{name}: {'applied' if decision.allowed else 'denied'}, priority={store.priority}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    print(run(), end="")
