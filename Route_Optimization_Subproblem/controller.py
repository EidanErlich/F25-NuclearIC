"""Transport orchestration utilities for the Route Optimization subproblem."""
from __future__ import annotations

from typing import Any, List, Optional

from .route_planner import Node, RoutePlanner


class TransportController:
    """Coordinate transport operations and trigger replanning when needed."""

    def __init__(self, planner: RoutePlanner, communicator: Any) -> None:
        """Store dependencies and initialize internal state."""
        self.planner = planner
        self.communicator = communicator
        self.current_route: List[Node] = []
        self.current_index: int = 0
        self.destination: Optional[Node] = None
        self.remaining_distance: float = 0.0
        self.remaining_risk: float = 0.0

    def start(self, start: Node, end: Node, max_distance: float, max_risk: float) -> None:
        """Plan an initial route and initialize tracking information."""
        self.destination = end
        self.current_route = self.planner.compute_route(start, end, max_distance, max_risk)
        self.current_index = 0
        self.remaining_distance = max_distance
        # Account for the hazard of the starting node immediately.
        self.remaining_risk = max_risk - self.planner.hazard_map.get(start, 0.0)

    def step(self) -> Optional[Node]:
        """Advance to the next node on the current route.

        Returns ``None`` if the destination has already been reached, otherwise
        returns the newly reached node.
        """
        if self.current_index + 1 < len(self.current_route):
            previous_node = self.current_route[self.current_index]
            self.current_index += 1
            new_node = self.current_route[self.current_index]
            self._update_consumed_budgets(previous_node, new_node)
            return new_node
        return None

    def sensor_event(self, current_node: Node, event: str) -> None:
        """Handle a sensor anomaly at ``current_node`` by attempting to replan.

        When replanning we keep the already consumed distance/risk budgets and
        request a fresh path from the planner. If no viable route remains we
        notify via ``communicator.send_alert``.
        """
        if self.destination is None:
            return

        # Remaining budgets exclude the hazard of the current node because it
        # has already been accounted for. Add it back before replanning so the
        # planner can count it once more when starting from ``current_node``.
        hazard_current = self.planner.hazard_map.get(current_node, 0.0)
        adjusted_risk_budget = self.remaining_risk + hazard_current

        try:
            new_route = self.planner.compute_route(
                current_node,
                self.destination,
                self.remaining_distance,
                adjusted_risk_budget,
            )
        except ValueError:
            message = (
                f"Replanning failed from {current_node} due to event '{event}'. "
                "No feasible route remains."
            )
            self.communicator.send_alert(message)
            return

        # Replanning succeeded: reset the route starting from the current node.
        self.current_route = new_route
        self.current_index = 0
        # Restore budgets so that the hazard of the current node remains
        # accounted for exactly once.
        self.remaining_risk = adjusted_risk_budget - hazard_current

    def _update_consumed_budgets(self, previous: Node, current: Node) -> None:
        """Update the remaining budgets after traversing an edge."""
        distance_cost = self._edge_distance(previous, current)
        self.remaining_distance -= distance_cost
        self.remaining_risk -= self.planner.hazard_map.get(current, 0.0)

    def _edge_distance(self, source: Node, target: Node) -> float:
        """Return the distance cost between ``source`` and ``target``."""
        for neighbor, distance in self.planner.graph.get(source, []):
            if neighbor == target:
                return float(distance)
        raise ValueError(f"No edge between {source} and {target}")
