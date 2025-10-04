"""Tests for :mod:`Route_Optimization_Subproblem.controller`."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import pytest

from Route_Optimization_Subproblem.controller import TransportController
from Route_Optimization_Subproblem.route_planner import RoutePlanner


def build_planner() -> RoutePlanner:
    graph = {
        "A": [("B", 5.0)],
        "B": [("C", 5.0), ("D", 2.0)],
        "C": [("E", 2.0)],
        "D": [("E", 5.0)],
        "E": [],
    }
    hazard_map = {"A": 1.0, "B": 2.0, "C": 5.0, "D": 1.0, "E": 1.0}
    return RoutePlanner(graph, hazard_map)


@dataclass
class DummyCommunicator:
    alerts: List[str] = field(default_factory=list)

    def send_alert(self, message: str) -> None:
        self.alerts.append(message)


def test_sensor_event_triggers_replan_successfully() -> None:
    planner = build_planner()
    communicator = DummyCommunicator()
    controller = TransportController(planner, communicator)

    controller.start("A", "E", max_distance=20.0, max_risk=15.0)
    # Move from A to B.
    assert controller.step() == "B"

    # Increase the hazard for D so the planner should now choose C instead.
    planner.hazard_map["D"] = 10.0
    controller.sensor_event("B", "hazard_increase")

    assert controller.current_route == ["B", "C", "E"]
    assert controller.current_index == 0
    # Budgets should remain unchanged by replanning.
    assert pytest.approx(controller.remaining_risk, rel=1e-6) == 12.0
    assert pytest.approx(controller.remaining_distance, rel=1e-6) == 15.0
    assert communicator.alerts == []


def test_sensor_event_sends_alert_when_no_route() -> None:
    planner = build_planner()
    communicator = DummyCommunicator()
    controller = TransportController(planner, communicator)

    controller.start("A", "E", max_distance=12.0, max_risk=6.0)
    assert controller.step() == "B"

    # Tighten risk budget to ensure infeasibility.
    controller.remaining_risk = -1.0
    controller.sensor_event("B", "blocked")

    assert communicator.alerts
    assert "blocked" in communicator.alerts[0]
