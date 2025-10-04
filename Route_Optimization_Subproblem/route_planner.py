"""Route planning utilities for the Route Optimization subproblem."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
import heapq

Node = Any


@dataclass(frozen=True)
class _State:
    """Container describing a search state for path reconstruction."""
    node: Node
    distance: float
    risk: float
    parent_id: Optional[int]


class RoutePlanner:
    """Plan constrained routes on a weighted graph with hazard costs."""

    def __init__(
        self,
        graph: Dict[Node, List[Tuple[Node, float]]],
        hazard_map: Dict[Node, float],
    ) -> None:
        """Store the road network and associated hazard costs.

        Args:
            graph: Adjacency list mapping each node to a list of tuples of the
                form ``(neighbor, distance_cost)``.
            hazard_map: Mapping of each node to the hazard/risk cost incurred
                when visiting the node.
        """
        self.graph = graph
        self.hazard_map = hazard_map

    def _get_hazard(self, node: Node) -> float:
        """Return the hazard for ``node`` (defaults to zero if unknown)."""
        return float(self.hazard_map.get(node, 0.0))

    def compute_route(
        self,
        start: Node,
        end: Node,
        max_distance: float,
        max_risk: float,
    ) -> List[Node]:
        """Compute a path obeying distance and risk budgets.

        A multi-criteria label-setting search (variant of Dijkstra) is used to
        simultaneously track the total distance and accumulated risk. We keep a
        set of non-dominated states per node. A state ``s1`` dominates ``s2`` if
        its distance and risk are both less-than-or-equal to ``s2``. Dominated
        states can never lead to a better solution and are therefore pruned.
        The priority queue is keyed by ``(risk, distance)`` to ensure that the
        first time we pop the destination node we have found the optimal route
        (minimal risk, then minimal distance).
        """
        if max_distance < 0 or max_risk < 0:
            raise ValueError("Budgets must be non-negative")

        start_hazard = self._get_hazard(start)
        if start == end:
            if start_hazard <= max_risk:
                return [start]
            raise ValueError("No feasible route")

        # Priority queue entries: (risk, distance, state_id)
        priority_queue: List[Tuple[float, float, int]] = []
        state_id_counter = 0
        states: Dict[int, _State] = {}
        per_node_states: Dict[Node, List[int]] = {}

        # Initialize with the starting node.
        initial_state = _State(node=start, distance=0.0, risk=start_hazard, parent_id=None)
        states[state_id_counter] = initial_state
        per_node_states[start] = [state_id_counter]
        heapq.heappush(priority_queue, (initial_state.risk, initial_state.distance, state_id_counter))

        # Track which states are currently valid (non-dominated).
        valid_states = {state_id_counter}

        while priority_queue:
            current_risk, current_distance, state_id = heapq.heappop(priority_queue)
            if state_id not in valid_states:
                # Skip stale entries that were dominated later on.
                continue

            current_state = states[state_id]
            node = current_state.node

            if current_risk > max_risk or current_distance > max_distance:
                continue

            if node == end:
                # Optimal solution reached due to queue ordering.
                return self._reconstruct_path(state_id, states)

            for neighbor, edge_distance in self.graph.get(node, []):
                new_distance = current_state.distance + float(edge_distance)
                if new_distance > max_distance:
                    continue

                neighbor_hazard = self._get_hazard(neighbor)
                new_risk = current_state.risk + neighbor_hazard
                if new_risk > max_risk:
                    continue

                # Dominance check: ensure the new state is not worse than an
                # existing state for the same neighbor.
                neighbor_state_ids = per_node_states.setdefault(neighbor, [])
                dominated = False
                dominated_ids: List[int] = []
                for existing_id in neighbor_state_ids:
                    existing_state = states[existing_id]
                    if existing_state.distance <= new_distance and existing_state.risk <= new_risk:
                        dominated = True
                        break
                    if new_distance <= existing_state.distance and new_risk <= existing_state.risk:
                        dominated_ids.append(existing_id)

                if dominated:
                    continue

                # Remove any states that are dominated by the new state.
                if dominated_ids:
                    for dominated_id in dominated_ids:
                        if dominated_id in valid_states:
                            valid_states.remove(dominated_id)
                    neighbor_state_ids[:] = [sid for sid in neighbor_state_ids if sid not in dominated_ids]

                state_id_counter += 1
                new_state = _State(
                    node=neighbor,
                    distance=new_distance,
                    risk=new_risk,
                    parent_id=state_id,
                )
                states[state_id_counter] = new_state
                neighbor_state_ids.append(state_id_counter)
                valid_states.add(state_id_counter)
                heapq.heappush(priority_queue, (new_state.risk, new_state.distance, state_id_counter))

        raise ValueError("No feasible route")

    def _reconstruct_path(self, state_id: int, states: Dict[int, _State]) -> List[Node]:
        """Reconstruct the path ending at ``state_id``."""
        path: List[Node] = []
        current_id: Optional[int] = state_id
        while current_id is not None:
            state = states[current_id]
            path.append(state.node)
            current_id = state.parent_id
        path.reverse()
        return path
