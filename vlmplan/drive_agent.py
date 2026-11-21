"""EdgeMind: closed-loop dynamic autonomous driving agent."""

from __future__ import annotations

from dataclasses import dataclass, field

from vlmplan.drive_world import DrivingWorld, astar
from drive_planner import OllamaDrivePlanner, DriveDecision


@dataclass
class DriveStatus:
    state: str = "idle"
    message: str = "Waiting for an instruction."
    path: list[tuple[int, int]] = field(default_factory=list)
    steps: int = 0


class DriveAgent:
    """
    Closed-loop driving controller.

    Flow:
    1. Understand instruction using Ollama.
    2. Generate an A* route.
    3. Validate the route.
    4. Execute one movement.
    5. Check the world revision.
    6. Replan if the environment has changed.
    7. Continue until arrival or a safety stop.
    """

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

        # Revision against which the current route was planned.
        self._planned_revision = -1

    def _position(self) -> tuple[int, int]:
        return self.world.car.position

    def _destination(self) -> tuple[int, int]:
        return self.world.car.destination

    def _calculate_path(self):
        path = astar(
            self.world,
            self._position(),
            self._destination(),
        )

        self.status.path = list(path) if path else []
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
                abs(current[0] - nxt[0])
                + abs(current[1] - nxt[1])
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

    def _plan_and_validate(self):
        """Generate and validate a route against the current world."""

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
            return False

        # Remember the exact world version used for this route.
        self._planned_revision = self.world.revision
        return True

    def start(self, instruction: str) -> DriveDecision:
        """Understand an instruction and prepare the initial route."""

        self.current_instruction = instruction.strip()

        if not self.current_instruction:
            raise ValueError("Please provide a driving instruction.")

        decision = self.planner.decide(
            self.current_instruction,
            self.world,
            route_available=True,
        )

        self.last_decision = decision

        if decision.decision != "navigate":
            self._set_state(
                decision.decision,
                decision.reason,
            )
            return decision

        if decision.destination not in self.world.parking_zones:
            self._set_state(
                "clarify",
                "The selected destination is not available.",
            )
            return decision

        self.world.set_destination(decision.destination)

        self.status.steps = 0

        if not self._plan_and_validate():
            return decision

        self._set_state(
            "ready",
            f"Route ready to {decision.destination}. "
            f"Path length: {len(self.status.path) - 1} steps.",
        )

        return decision

    def _replan_if_world_changed(self) -> bool:
        """Replan only when the environment revision has changed."""

        if self.world.revision == self._planned_revision:
            return True

        changes = self.world.changes_since(self._planned_revision)

        descriptions = []

        for change in changes:
            action = change["action"]

            if action == "obstacle_added":
                descriptions.append(
                    f"new {change['kind']} at {change['position']}"
                )

            elif action == "obstacle_moved":
                descriptions.append(
                    f"obstacle moved from {change['old_position']} "
                    f"to {change['new_position']}"
                )

            elif action == "obstacle_removed":
                descriptions.append(
                    f"obstacle removed from {change['old_position']}"
                )

            elif action == "destination_changed":
                descriptions.append("destination changed")

        reason = (
            "; ".join(descriptions)
            if descriptions
            else "environment updated"
        )

        self._set_state(
            "replanning",
            f"World changed: {reason}. Recalculating route.",
        )

        return self._plan_and_validate()

    def step(self) -> DriveStatus:
        """Execute one safe movement, replanning when necessary."""

        if self.status.state not in {
            "ready",
            "moving",
            "replanning",
        }:
            return self.status

        if self._position() == self._destination():
            self._set_state(
                "completed",
                "Destination reached successfully.",
            )
            return self.status

        # Check whether the environment changed since the last plan.
        if not self._replan_if_world_changed():
            return self.status

        path = self.status.path

        # The stored route must start at the current car position.
        if not path or path[0] != self._position():
            self._set_state(
                "replanning",
                "Current position differs from the planned route. "
                "Calculating a new route.",
            )

            if not self._plan_and_validate():
                return self.status

            path = self.status.path

        if not self._validate_path(path):
            self._set_state(
                "safety_stop",
                "Route safety validation failed. The car has stopped.",
            )
            return self.status

        if len(path) < 2:
            self._set_state(
                "completed",
                "Destination reached successfully.",
            )
            return self.status

        next_position = path[1]

        # Final safety check immediately before movement.
        if self.world.blocked(*next_position):
            self._set_state(
                "safety_stop",
                "The next cell is blocked. The car has stopped.",
            )
            return self.status

        # Protect against a world update between planning and execution.
        if self.world.revision != self._planned_revision:
            self._set_state(
                "replanning",
                "World changed before execution. Replanning required.",
            )

            if not self._plan_and_validate():
                return self.status

            path = self.status.path

            if len(path) < 2:
                self._set_state(
                    "completed",
                    "Destination reached successfully.",
                )
                return self.status

            next_position = path[1]

            if self.world.blocked(*next_position):
                self._set_state(
                    "safety_stop",
                    "Next cell is blocked after replanning.",
                )
                return self.status

        old_position = self._position()

        dx = next_position[0] - old_position[0]
        dy = next_position[1] - old_position[1]

        directions = {
            (1, 0): "east",
            (-1, 0): "west",
            (0, 1): "south",
            (0, -1): "north",
        }

        direction = directions.get((dx, dy))

        if direction is None:
            self._set_state(
                "safety_stop",
                "Invalid movement direction. The car has stopped.",
            )
            return self.status

        # Execute exactly one grid movement.
        self.world.car.direction = direction
        self.world.car.move_to(next_position)

        self.status.steps += 1

        # Remove the completed cell from the remaining route.
        self.status.path = path[1:]

        self._set_state(
            "moving",
            f"Car moved from {old_position} to {next_position}.",
        )

        if self._position() == self._destination():
            self._set_state(
                "completed",
                "Destination reached successfully.",
            )

        return self.status

    def run(self, max_steps: int = 100) -> DriveStatus:
        """Run until arrival, a safety stop, or the step limit."""

        if max_steps < 1:
            raise ValueError("max_steps must be at least 1.")

        for _ in range(max_steps):
            if self.status.state not in {
                "ready",
                "moving",
                "replanning",
            }:
                break

            self.step()

            if self.status.state == "completed":
                break

        if self.status.state in {"ready", "moving", "replanning"}:
            self._set_state(
                "paused",
                "Maximum step limit reached. Simulation paused.",
            )

        return self.status

    def reset(self):
        """Reset execution state without changing the environment."""

        self.status = DriveStatus()
        self.last_decision = None
        self.current_instruction = ""
        self._planned_revision = -1
