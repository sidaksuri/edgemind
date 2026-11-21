"""EdgeMind: closed-loop dynamic autonomous driving agent."""
# drive_agent.py
from __future__ import annotations

from dataclasses import dataclass, field

from vlmplan.drive_world import DrivingWorld, astar
from drive_planner import (
    OllamaDrivePlanner,
    DriveDecision,
    is_likely_driving_instruction,
)


ACTIVE_STATES = {"ready", "moving", "replanning", "waiting"}


@dataclass
class DriveStatus:
    state: str = "idle"
    message: str = "Waiting for an instruction."
    path: list[tuple[int, int]] = field(default_factory=list)
    steps: int = 0

    llm_calls: int = 0
    llm_navigate: int = 0
    llm_wait: int = 0
    llm_stop: int = 0
    llm_clarify: int = 0
    llm_errors: int = 0

    replans: int = 0
    replan_detours: int = 0
    replan_waits: int = 0
    safety_stops: int = 0
    safety_overrides: int = 0

    path_length_constrained: int = 0
    path_length_unconstrained: int = 0
    risk_cells_crossed: int = 0
    risk_cells_unconstrained: int = 0

    destination_source: str = "none"


class DriveAgent:

    def __init__(
        self,
        world: DrivingWorld,
        planner: OllamaDrivePlanner | None = None,
    ):
        self.world = world
        self.planner = planner or OllamaDrivePlanner()

        self.status = DriveStatus()
        self.last_decision: DriveDecision | None = None
        self.current_instruction = ""

        self._planned_revision = -1

        self.use_llm = True
        self.image_provider = None
        self.current_avoid_types: set[str] = set()
        self.soft_avoid_cells: set[tuple[int, int]] = set()

    def _call_astar(self):
        # No try/except TypeError. If astar is outdated, this will crash loudly.
        return astar(
            self.world,
            self._position(),
            self._destination(),
            soft_avoid_cells=self.soft_avoid_cells,
        )

    def _position(self) -> tuple[int, int]:
        return self.world.car.position

    def _destination(self) -> tuple[int, int]:
        return self.world.car.destination

    def _refresh_soft_avoid_cells(self):
        if not self.current_avoid_types:
            self.soft_avoid_cells = set()
            return
        cells: set[tuple[int, int]] = set()
        for obs in self.world.active_obstacles():
            if obs.kind not in self.current_avoid_types:
                continue
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nx, ny = obs.x + dx, obs.y + dy
                    if self.world.inside_grid(nx, ny):
                        cells.add((nx, ny))
        self.soft_avoid_cells = cells

    def _calculate_path(self):
        self._refresh_soft_avoid_cells()
        path = self._call_astar()
        self.status.path = list(path) if path else []
        self.status.path_length_constrained = len(path) if path else 0
        self.status.risk_cells_crossed = sum(
            1 for p in (path or []) if p in self.soft_avoid_cells
        )

        unconstrained = astar(
            self.world,
            self._position(),
            self._destination(),
        )

        self.status.path_length_unconstrained = (
            len(unconstrained) if unconstrained else 0
        )
        self.status.risk_cells_unconstrained = sum(
            1 for p in (unconstrained or []) if p in self.soft_avoid_cells
        )

        return path

    def _validate_path(self, path) -> bool:
        if not path:
            return False
        if path[0] != self._position():
            return False
        if self.world.blocked(*path[0]):
            return False
        for current, nxt in zip(path, path[1:]):
            distance = (
                abs(current[0] - nxt[0]) + abs(current[1] - nxt[1])
            )
            if distance != 1:
                return False
            if self.world.blocked(*nxt):
                return False
        return True

    def _set_state(self, state: str, message: str):
        self.status.state = state
        self.status.message = message
        self.world.log.append(message)

    def _get_scene_image(self):
        if self.image_provider is None:
            return None
        try:
            return self.image_provider()
        except Exception as error:
            self.world.log.append(f"Scene render failed: {error}")
            return None

    def _describe_changes(self, changes) -> str:
        parts = []
        for change in changes:
            action = change.get("action")
            if action == "obstacle_added":
                parts.append(
                    f"new {change.get('kind', 'obstacle')} at "
                    f"{change.get('position')}"
                )
            elif action == "obstacle_moved":
                parts.append(
                    f"obstacle moved to {change.get('new_position')}"
                )
            elif action == "obstacle_removed":
                parts.append(
                    f"obstacle removed from {change.get('old_position')}"
                )
            elif action == "destination_changed":
                parts.append(
                    f"destination changed to {change.get('destination')}"
                )
        return "; ".join(parts) if parts else "environment updated"

    def _plan_and_validate(self):
        path = self._calculate_path()

        if path is None:
            self._set_state(
                "blocked",
                "No valid route exists. The car has stopped.",
            )
            return False

        if not self._validate_path(path):
            self._set_state(
                "safety_stop",
                "Route validation failed. The car has stopped.",
            )
            self.status.safety_stops += 1
            return False

        self._planned_revision = self.world.revision
        return True

    def _consult_llm_replan(self, reason: str, detour_cost: int) -> str:
        try:
            self.status.llm_calls += 1
            decision = self.planner.decide_replan(
                reason=reason,
                world=self.world,
                detour_cost=detour_cost,
                image_data_uri=self._get_scene_image(),
            )
            if decision.decision == "navigate":
                self.status.llm_navigate += 1
            elif decision.decision == "wait":
                self.status.llm_wait += 1
            elif decision.decision == "stop":
                self.status.llm_stop += 1
            else:
                self.status.llm_clarify += 1
            return decision.decision
        except Exception as error:
            self.status.llm_errors += 1
            self.world.log.append(
                f"LLM replan failed: {error}. Falling back to A*."
            )
            return "navigate"

    def start(
        self,
        instruction: str,
        image_provider=None,
        use_llm: bool = True,
    ):
        self.current_instruction = instruction.strip()
        self.image_provider = image_provider
        self.use_llm = use_llm

        if not self.current_instruction:
            raise ValueError("Please provide a driving instruction.")

        if use_llm:
            allowed, refusal_reason = is_likely_driving_instruction(
                self.current_instruction
            )
            if not allowed:
                self.status.destination_source = "none"
                self._set_state("clarify", refusal_reason)
                return DriveDecision(
                    decision="clarify",
                    destination=None,
                    reason=refusal_reason,
                    avoid_obstacle_types=[],
                )

        if not use_llm:
            self.status.steps = 0
            self.status.destination_source = "dropdown"
            if not self._plan_and_validate():
                return None
            self._set_state(
                "ready",
                f"A*-only route to {self._destination()}. "
                f"Path length: {max(0, len(self.status.path) - 1)} steps. "
                f"(Instruction ignored - dropdown destination used.)",
            )
            return DriveDecision(
                decision="navigate",
                destination=None,
                reason="A*-only mode (LLM disabled).",
            )

        self.status.llm_calls += 1
        decision = self.planner.decide(
            self.current_instruction,
            self.world,
            route_available=True,
            image_data_uri=self._get_scene_image(),
        )
        self.last_decision = decision

        if decision.decision == "navigate":
            self.status.llm_navigate += 1
        elif decision.decision == "wait":
            self.status.llm_wait += 1
        elif decision.decision == "stop":
            self.status.llm_stop += 1
        else:
            self.status.llm_clarify += 1

        if decision.decision != "navigate":
            self.status.destination_source = "none"
            self._set_state(decision.decision, decision.reason)
            return decision

        if decision.destination not in self.world.parking_zones:
            self.status.destination_source = "none"
            self._set_state(
                "clarify",
                "The LLM selected an unknown destination.",
            )
            return decision

        try:
            if (
                self.world.parking_zones[decision.destination]
                != self.world.car.destination
            ):
                self.world.set_destination(decision.destination)
        except ValueError as error:
            self.status.destination_source = "none"
            self._set_state(
                "clarify",
                f"Cannot set destination: {error}",
            )
            return decision

        self.status.destination_source = "llm"
        self.current_avoid_types = set(decision.avoid_obstacle_types)
        if self.current_avoid_types:
            self.world.log.append(
                f"LLM extracted avoidance: {sorted(self.current_avoid_types)}"
            )

        self.status.steps = 0

        if not self._plan_and_validate():
            return decision

        self._set_state(
            "ready",
            f"Route ready to {decision.destination}. "
            f"Path length: {len(self.status.path) - 1} steps. "
            f"Avoid: {sorted(self.current_avoid_types) or 'none'}.",
        )
        return decision

    def _replan_if_world_changed(self) -> bool:
        if self.world.revision == self._planned_revision:
            return True

        self.status.replans += 1
        reason = self._describe_changes(
            self.world.changes_since(self._planned_revision)
        )

        old_length = len(self.status.path)
        new_path = self._call_astar()
        detour_cost = (
            len(new_path) - old_length if new_path is not None else -1
        )

        if self.use_llm:
            decision = self._consult_llm_replan(reason, detour_cost)

            if decision == "wait":
                self.status.replan_waits += 1
                self._set_state(
                    "waiting",
                    f"LLM chose to wait (detour {detour_cost}): {reason}.",
                )
                self._planned_revision = self.world.revision
                return False

            if decision == "stop":
                self.status.safety_stops += 1
                self._set_state(
                    "safety_stop",
                    f"LLM stopped the car: {reason}.",
                )
                return False

            if decision == "clarify":
                self._set_state(
                    "waiting",
                    f"LLM needs clarification: {reason}.",
                )
                self._planned_revision = self.world.revision
                return False

            self.status.replan_detours += 1
            self._set_state(
                "replanning",
                f"World changed (detour {detour_cost}): {reason}. "
                f"Recalculating.",
            )
            ok = self._plan_and_validate()

            if not ok:
                self.status.safety_overrides += 1
                self._set_state(
                    "waiting",
                    f"Safety override (LLM said navigate, no path): "
                    f"{reason}. Waiting for environment to change.",
                )
                self._planned_revision = self.world.revision
                return False

            return True

        self.status.replan_detours += 1
        self._set_state(
            "replanning",
            f"World changed (detour {detour_cost}): {reason}. "
            f"Recalculating.",
        )
        return self._plan_and_validate()

    def step(self) -> DriveStatus:
        if self.status.state not in ACTIVE_STATES:
            return self.status

        if self.status.state == "waiting":
            if self.world.revision == self._planned_revision:
                return self.status

        if self._position() == self._destination():
            self._set_state(
                "completed", "Destination reached successfully."
            )
            return self.status

        if not self._replan_if_world_changed():
            return self.status

        path = self.status.path

        if not path or path[0] != self._position():
            self._set_state(
                "replanning",
                "Current position differs from planned route.",
            )
            if not self._plan_and_validate():
                return self.status
            path = self.status.path

        if not self._validate_path(path):
            self.status.safety_stops += 1
            self._set_state("safety_stop", "Route validation failed.")
            return self.status

        if len(path) < 2:
            self._set_state(
                "completed", "Destination reached successfully."
            )
            return self.status

        next_position = path[1]

        if self.world.blocked(*next_position):
            self.status.safety_stops += 1
            self._set_state("safety_stop", "The next cell is blocked.")
            return self.status

        old_position = self._position()
        dx = next_position[0] - old_position[0]
        dy = next_position[1] - old_position[1]
        directions = {
            (1, 0): "east", (-1, 0): "west",
            (0, 1): "south", (0, -1): "north",
        }
        direction = directions.get((dx, dy))

        if direction is None:
            self.status.safety_stops += 1
            self._set_state("safety_stop", "Invalid movement direction.")
            return self.status

        self.world.car.direction = direction
        self.world.car.move_to(next_position)
        self.status.steps += 1
        self.status.path = path[1:]
        self._set_state(
            "moving",
            f"Car moved from {old_position} to {next_position}.",
        )

        if self._position() == self._destination():
            self._set_state(
                "completed", "Destination reached successfully."
            )

        return self.status

    def run(self, max_steps: int = 100) -> DriveStatus:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1.")
        for _ in range(max_steps):
            if self.status.state not in ACTIVE_STATES:
                break
            self.step()
            if self.status.state == "completed":
                break
        if self.status.state in ACTIVE_STATES:
            self._set_state("paused", "Maximum step limit reached.")
        return self.status

    def reset(self):
        self.status = DriveStatus()
        self.last_decision = None
        self.current_instruction = ""
        self._planned_revision = -1
        self.image_provider = None
        self.use_llm = True
        self.current_avoid_types = set()
        self.soft_avoid_cells = set()