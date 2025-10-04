"""Unit tests for :mod:`Route_Optimization_Subproblem.route_planner`."""
from __future__ import annotations

import pytest

from Route_Optimization_Subproblem.route_planner import RoutePlanner


@pytest.fixture
def sample_graph() -> RoutePlanner:
    graph = {
        "A": [("B", 2.0), ("C", 2.0)],
        "B": [("D", 3.0)],
        "C": [("D", 3.0)],
        "D": [],
    }
    hazard_map = {"A": 1.0, "B": 2.0, "C": 1.0, "D": 1.0}
    return RoutePlanner(graph=graph, hazard_map=hazard_map)


def test_compute_route_prefers_lower_risk(sample_graph: RoutePlanner) -> None:
    route = sample_graph.compute_route("A", "D", max_distance=10.0, max_risk=10.0)
    assert route == ["A", "C", "D"]


def test_compute_route_breaks_risk_ties_with_distance() -> None:
    graph = {
        "A": [("B", 2.0), ("C", 4.0)],
        "B": [("D", 2.0)],
        "C": [("D", 2.0)],
        "D": [],
    }
    hazard_map = {"A": 1.0, "B": 2.0, "C": 2.0, "D": 1.0}
    planner = RoutePlanner(graph, hazard_map)
    route = planner.compute_route("A", "D", max_distance=10.0, max_risk=10.0)
    # Both paths have identical risk (1 + 2 + 1 == 1 + 2 + 1) but the
    # A->B->D option is shorter.
    assert route == ["A", "B", "D"]


def test_start_equals_end(sample_graph: RoutePlanner) -> None:
    assert sample_graph.compute_route("A", "A", 5.0, 5.0) == ["A"]


def test_no_feasible_route_due_to_risk(sample_graph: RoutePlanner) -> None:
    with pytest.raises(ValueError, match="No feasible route"):
        sample_graph.compute_route("A", "D", max_distance=10.0, max_risk=2.0)


def test_negative_budget_raises(sample_graph: RoutePlanner) -> None:
    with pytest.raises(ValueError):
        sample_graph.compute_route("A", "D", max_distance=-1.0, max_risk=5.0)
