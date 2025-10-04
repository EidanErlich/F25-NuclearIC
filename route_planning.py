"""Route planning module for nuclear-waste transport monitoring system.

This module provides a constrained multi-criteria route planner and a
transport controller that can replan routes in reaction to sensor events.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
import heapq
import unittest

Node = Any


@dataclass(frozen=True)
class _State:
    """Internal representation of a search state.

    Attributes:
        node: Current node of the state.
        risk: Cumulative hazard risk accumulated so far.
        distance: Cumulative distance traveled so far.
        parent_id: Identifier of the predecessor state for path reconstruction.
    """

    node: Node
    risk: float
    distance: float
    parent_id: Optional[int]


class RoutePlanner:
    """Planner capable of finding low-risk routes under distance/risk budgets.

    The planner performs a label-setting search similar to multi-objective
    Dijkstra. Each *label* is a state with cumulative distance and risk values.
    Labels that are dominated by another label (higher risk and higher or equal
    distance) are discarded. The search frontier is a priority queue ordered by
    (risk, distance) to ensure that the first feasible route found has minimal
    risk and, on ties, minimal distance.
    """

    def __init__(self,
                 graph: Dict[Node, List[Tuple[Node, float]]],
                 hazard_map: Dict[Node, float]):
        """Store graph and hazard map; perform basic validation."""
        self.graph = graph
        self.hazard_map = hazard_map

    def compute_route(self,
                      start: Node,
                      end: Node,
                      max_distance: float,
                      max_risk: float) -> List[Node]:
        """Compute a constrained minimum-risk path.

        Args:
            start: Starting node.
            end: Destination node.
            max_distance: Maximum allowable cumulative distance.
            max_risk: Maximum allowable cumulative risk.

        Returns:
            List of nodes representing the route from start to end (inclusive).

        Raises:
            ValueError: If no feasible path exists or invalid inputs are given.
        """
        if max_distance < 0 or max_risk < 0:
            raise ValueError("Budgets must be non-negative")

        if start not in self.graph or end not in self.graph:
            raise ValueError("Start or end node missing from graph")

        start_risk = self._hazard_cost(start)
        if start_risk > max_risk:
            raise ValueError("No feasible path")

        state_info: Dict[int, _State] = {}
        pq: List[Tuple[float, float, int]] = []  # (risk, distance, state_id)
        best_labels: Dict[Node, List[Tuple[float, float, int]]] = {}
        next_state_id = 0

        def add_state(node: Node, risk: float, distance: float, parent_id: Optional[int]) -> None:
            nonlocal next_state_id
            if risk > max_risk + 1e-9 or distance > max_distance + 1e-9:
                return

            # Reject negative values to avoid pathological inputs.
            if risk < -1e-9 or distance < -1e-9:
                raise ValueError("Negative risk or distance encountered")

            labels = best_labels.setdefault(node, [])
            for existing_risk, existing_distance, _ in labels:
                if existing_risk <= risk + 1e-9 and existing_distance <= distance + 1e-9:
                    # Dominated by an existing label.
                    return

            # Remove labels dominated by the new one.
            filtered_labels = []
            for existing_risk, existing_distance, existing_id in labels:
                if not (risk <= existing_risk + 1e-9 and distance <= existing_distance + 1e-9):
                    filtered_labels.append((existing_risk, existing_distance, existing_id))
            labels[:] = filtered_labels

            state_id = next_state_id
            next_state_id += 1
            state_info[state_id] = _State(node=node, risk=risk, distance=distance, parent_id=parent_id)
            labels.append((risk, distance, state_id))
            heapq.heappush(pq, (risk, distance, state_id))

        add_state(start, start_risk, 0.0, None)

        while pq:
            risk, distance, state_id = heapq.heappop(pq)
            state = state_info[state_id]

            if risk > max_risk + 1e-9 or distance > max_distance + 1e-9:
                continue

            if state.node == end:
                return self._reconstruct_path(state_id, state_info)

            for neighbor, edge_distance in self.graph.get(state.node, []):
                if edge_distance < 0:
                    raise ValueError("Negative distance encountered")

                new_distance = distance + edge_distance
                neighbor_risk = self._hazard_cost(neighbor)
                new_risk = risk + neighbor_risk
                add_state(neighbor, new_risk, new_distance, state_id)

        raise ValueError("No feasible path")

    def _hazard_cost(self, node: Node) -> float:
        """Return hazard cost for node, defaulting to zero if absent."""
        cost = float(self.hazard_map.get(node, 0.0))
        if cost < -1e-9:
            raise ValueError("Negative hazard cost encountered")
        return cost

    @staticmethod
    def _reconstruct_path(state_id: int, state_info: Dict[int, _State]) -> List[Node]:
        """Reconstruct route by following parent pointers."""
        route: List[Node] = []
        current_id: Optional[int] = state_id
        while current_id is not None:
            state = state_info[current_id]
            route.append(state.node)
            current_id = state.parent_id
        route.reverse()
        return route


class TransportController:
    """Controller that executes a route and replans on anomalies."""

    def __init__(self,
                 planner: RoutePlanner,
                 communicator: Any):
        self.planner = planner
        self.communicator = communicator
        self.current_route: List[Node] = []
        self.current_index: int = 0
        self.destination: Optional[Node] = None
        self.remaining_distance: float = 0.0
        self.remaining_risk: float = 0.0

    def start(self,
              start: Node,
              end: Node,
              max_distance: float,
              max_risk: float) -> None:
        """Plan an initial route and initialize budgets."""
        route = self.planner.compute_route(start, end, max_distance, max_risk)
        self.destination = end
        self.current_route = route
        self.current_index = 0
        self.remaining_distance = max_distance
        start_risk = self.planner.hazard_map.get(route[0], 0.0)
        self.remaining_risk = max_risk - start_risk

    def step(self) -> Optional[Node]:
        """Advance to the next node, updating budgets; return the new node."""
        if not self.current_route:
            return None

        if self.current_index + 1 >= len(self.current_route):
            return None

        current_node = self.current_route[self.current_index]
        next_node = self.current_route[self.current_index + 1]
        edge_cost = self._edge_cost(current_node, next_node)
        hazard_cost = self.planner.hazard_map.get(next_node, 0.0)

        self.remaining_distance -= edge_cost
        self.remaining_risk -= hazard_cost
        self.current_index += 1
        return next_node

    def sensor_event(self, current_node: Node, event: str) -> None:
        """Handle anomaly by attempting to replan from the current node."""
        if self.destination is None:
            return

        current_hazard = self.planner.hazard_map.get(current_node, 0.0)
        available_risk = self.remaining_risk + current_hazard
        available_distance = self.remaining_distance

        try:
            new_route = self.planner.compute_route(
                current_node,
                self.destination,
                available_distance,
                available_risk,
            )
        except ValueError:
            self.communicator.send_alert(
                f"No feasible route after anomaly at node {current_node}"
            )
            return

        self.current_route = new_route
        self.current_index = 0
        self.remaining_distance = available_distance
        self.remaining_risk = available_risk - current_hazard

    def _edge_cost(self, u: Node, v: Node) -> float:
        """Return the distance cost between two consecutive nodes."""
        for neighbor, cost in self.planner.graph.get(u, []):
            if neighbor == v:
                if cost < 0:
                    raise ValueError("Negative distance encountered")
                return cost
        raise ValueError(f"Edge {u}->{v} not found in graph")


class DummyCommunicator:
    """Simple communicator used for unit tests."""

    def __init__(self):
        self.alerts: List[str] = []

    def send_alert(self, msg: str) -> None:
        self.alerts.append(msg)


class RoutePlannerTests(unittest.TestCase):
    """Unit tests for RoutePlanner."""

    def setUp(self) -> None:
        self.graph = {
            "A": [("B", 5), ("C", 3)],
            "B": [("D", 4)],
            "C": [("D", 6)],
            "D": [],
        }
        self.hazard_map = {"A": 1, "B": 2, "C": 1, "D": 3}
        self.planner = RoutePlanner(self.graph, self.hazard_map)

    def test_selects_lowest_risk_route(self) -> None:
        """Route with lower risk should be chosen even if longer distance."""
        route = self.planner.compute_route("A", "D", max_distance=20, max_risk=10)
        self.assertEqual(route, ["A", "C", "D"])

    def test_no_feasible_route_raises(self) -> None:
        """Planner should raise when constraints forbid all routes."""
        with self.assertRaises(ValueError):
            self.planner.compute_route("A", "D", max_distance=5, max_risk=10)

    def test_tie_breaks_on_distance(self) -> None:
        """When risk is equal, shorter distance path is preferred."""
        graph = {
            1: [(2, 2), (3, 1)],
            2: [(4, 2)],
            3: [(4, 4)],
            4: [],
        }
        hazard_map = {1: 1, 2: 2, 3: 2, 4: 3}
        planner = RoutePlanner(graph, hazard_map)
        route = planner.compute_route(1, 4, max_distance=10, max_risk=10)
        self.assertEqual(route, [1, 2, 4])


class TransportControllerTests(unittest.TestCase):
    """Tests for TransportController re-planning behavior."""

    def test_replans_on_sensor_event(self) -> None:
        graph = {
            "S": [("A", 2), ("B", 4)],
            "A": [("T", 2)],
            "B": [("T", 1)],
            "T": [],
        }
        hazard_map = {"S": 1, "A": 2, "B": 1, "T": 3}
        planner = RoutePlanner(graph, hazard_map)
        communicator = DummyCommunicator()
        controller = TransportController(planner, communicator)

        controller.start("S", "T", max_distance=10, max_risk=10)
        # The initial plan prefers the safer route S -> B -> T.
        self.assertEqual(controller.current_route, ["S", "B", "T"])

        # Move to node B using the planned route.
        self.assertEqual(controller.step(), "B")

        # Inject sensor event at node B and ensure controller attempts to replan.
        controller.sensor_event("B", "shock")

        # After replanning, the route should again begin at the current node and end at T.
        self.assertEqual(controller.current_route[0], "B")
        self.assertEqual(controller.current_route[-1], "T")
        self.assertFalse(communicator.alerts)


if __name__ == "__main__":
    unittest.main()
