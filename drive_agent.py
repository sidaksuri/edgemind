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

from drive_safety import validate_path as validate_path_shield


ACTIVE_STATES = {"ready", "moving", "replanning", "waiting", "braking", "turning"}
RECOVERABLE_STATES = {"blocked", "safety_stop"}
WAIT_TIMEOUT_STEPS = 10

BRAKE_TICKS = 1
TURN_TICKS = 1

HAZARD_BRAKE_THRESHOLD = 3.0
ROUTE_NEIGHBOURHOOD_RADIUS = 2

# Manhattan radius within which an LLM-supplied obstacle coordinate is
# snapped to the nearest real obstacle of the same kind. VLM coordinate
# drift of 1-2 cells is normal; beyond this, the spec is treated as a
# hallucination and dropped.
SNAP_RADIUS = 2

DIRECTIONS = {(1, 0): "east", (-1, 0): "west", (0, 1): "south", (0, -1): "north"}

MOVING_KINDS = {"traffic_car", "opposition_car"}


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

    hazard_weight_crossed: float = 0.0
    hazard_weight_unconstrained: float = 0.0
    soft_avoid_cell_count: int = 0
    soft_avoid_specs_snapped: int = 0   # NEW: diagnostics

    risk_cells_crossed: int = 0
    risk_cells_unconstrained: int = 0

    path_length_constrained: int = 0
    path_length_unconstrained: int = 0

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
        self._wait_steps = 0
        self._brake_ticks = 0
        self._turn_ticks = 0
        self._skip_brake_once = False

        self.use_llm = True
        self.image_provider = None

        self.current_avoid_obstacles: list[dict] = []
        self.current_safety_vs_time: float = 0.5
        self.soft_avoid_weights: dict[tuple[int, int], float] = {}

        self._last_planned_route_obstacles: frozenset[str] = frozenset()
        self._last_planned_static: frozenset = frozenset()

    @property
    def soft_avoid_cells(self) -> set[tuple[int, int]]:
        return set(self.soft_avoid_weights.keys())

    @property
    def current_avoid_types(self) -> set[str]:
        return {o["kind"] for o in self.current_avoid_obstacles}

    def _call_astar(self):
        return astar(
            self.world,
            self._position(),
            self._destination(),
            soft_avoid_weights=self.soft_avoid_weights,
        )

    def _position(self) -> tuple[int, int]:
        return self.world.car.position

    def _destination(self) -> tuple[int, int]:
        return self.world.car.destination

    def _diagnose_no_path(self) -> str:
        dest = self._destination()
        dest_occupants = [o for o in self.world.active_obstacles() if o.position == dest]
        if dest_occupants:
            listing = ", ".join(f"{o.id} ({o.kind})" for o in dest_occupants)
            moving = any(o.kind in MOVING_KINDS for o in dest_occupants)
            suffix = " — waiting for it to move" if moving else " — static block, needs editor fix"
            return f"Destination {dest} is occupied by {listing}{suffix}."
        return f"No path from {self._position()} to {dest}."

    def _static_obstacle_snapshot(self) -> frozenset:
        return frozenset(
            (o.id, o.x, o.y, o.kind)
            for o in self.world.active_obstacles()
            if o.kind not in MOVING_KINDS
        )

    def _refresh_soft_avoid_weights(self):
        """Build the per-cell soft-avoid cost field from the LLM's specs.

        VLM coordinate drift: the model frequently returns an obstacle
        position that is 1-2 cells off from the ground-truth obstacle.
        If we required an exact ``(x, y, kind)`` match, a single missed
        cell would silently collapse the entire weight field to ``{}``,
        making the ghost and active A* routes identical.

        This function snaps each spec to the nearest real obstacle of the
        same kind within ``SNAP_RADIUS`` (Manhattan), and drops the spec
        otherwise. Multiple specs may not snap to the same obstacle —
        the nearest one claims it first.
        """
        # Reset the diagnostics every refresh.
        self.status.soft_avoid_specs_snapped = 0

        if not self.current_avoid_obstacles:
            self.soft_avoid_weights = {}
            self.status.soft_avoid_cell_count = 0
            return

        safety_factor = 0.5 + 1.5 * max(0.0, min(1.0, self.current_safety_vs_time))
        live_obstacles = self.world.active_obstacles()
        obstacle_at = {(o.x, o.y): o for o in live_obstacles}

        # Index live obstacles by kind so we can search for the nearest
        # match when the spec coordinate misses.
        by_kind: dict[str, list] = {}
        for o in live_obstacles:
            by_kind.setdefault(o.kind, []).append(o)

        claimed: set[str] = set()
        weights: dict[tuple[int, int], float] = {}

        for spec in self.current_avoid_obstacles:
            pos = spec.get("position")
            kind = spec.get("kind")
            if not isinstance(pos, (list, tuple)) or len(pos) != 2:
                continue
            try:
                orig_px, orig_py = int(pos[0]), int(pos[1])
            except (TypeError, ValueError):
                continue

            actual_obs = obstacle_at.get((orig_px, orig_py))
            if actual_obs is None or actual_obs.kind != kind:
                # Snap to nearest live obstacle of the right kind within
                # SNAP_RADIUS, skipping obstacles already claimed by another
                # spec.
                candidates = [
                    o for o in by_kind.get(kind, [])
                    if o.id not in claimed
                    and abs(o.x - orig_px) + abs(o.y - orig_py) <= SNAP_RADIUS
                ]
                if not candidates:
                    # No real obstacle of this kind nearby — treat as a
                    # hallucination and drop the spec.
                    self.world.log.append(
                        f"Dropped LLM spec ({orig_px}, {orig_py}) {kind}: "
                        f"no live {kind} within {SNAP_RADIUS} cells."
                    )
                    continue
                candidates.sort(
                    key=lambda o: (abs(o.x - orig_px) + abs(o.y - orig_py), o.id)
                )
                actual_obs = candidates[0]
                self.status.soft_avoid_specs_snapped += 1
                self.world.log.append(
                    f"Snapped LLM spec ({orig_px}, {orig_py}) {kind} -> "
                    f"({actual_obs.x}, {actual_obs.y}) id={actual_obs.id}."
                )

            claimed.add(actual_obs.id)
            px, py = actual_obs.x, actual_obs.y

            base = float(spec.get("weight", 5.0)) * safety_factor
            if base <= 0.0:
                continue

            radius = getattr(actual_obs, "fallout", 1)
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    nx, ny = px + dx, py + dy
                    if not self.world.inside_grid(nx, ny):
                        continue
                    if self.world.blocked(nx, ny):
                        continue
                    distance = max(abs(dx), abs(dy))
                    falloff = 1.0 / (distance + 1)
                    effective = base * falloff
                    if effective > weights.get((nx, ny), 0.0):
                        weights[(nx, ny)] = effective

        self.soft_avoid_weights = weights
        self.status.soft_avoid_cell_count = len(weights)

    def _route_neighbourhood_obstacles(self) -> frozenset[str]:
        path = self.status.path
        if not path:
            return frozenset()
        radius = ROUTE_NEIGHBOURHOOD_RADIUS
        result = set()
        for obs in self.world.active_obstacles():
            ox, oy = obs.x, obs.y
            for px, py in path:
                if abs(ox - px) <= radius and abs(oy - py) <= radius:
                    result.add(obs.id)
                    break
        return frozenset(result)

    def _calculate_path(self):
        self._refresh_soft_avoid_weights()
        path = self._call_astar()
        self.status.path = list(path) if path else []
        self.status.path_length_constrained = len(path) if path else 0

        self.status.hazard_weight_crossed = sum(
            self.soft_avoid_weights.get(p, 0.0) for p in (path or [])
        )
        self.status.risk_cells_crossed = sum(
            1 for p in (path or []) if p in self.soft_avoid_weights
        )

        unconstrained = astar(self.world, self._position(), self._destination())
        self.status.path_length_unconstrained = len(unconstrained or [])
        self.status.hazard_weight_unconstrained = sum(
            self.soft_avoid_weights.get(p, 0.0) for p in (unconstrained or [])
        )
        self.status.risk_cells_unconstrained = sum(
            1 for p in (unconstrained or []) if p in self.soft_avoid_weights
        )
        return path

    def _validate_path(self, path) -> bool:
        ok, reason = validate_path_shield(self.world, path)
        if not ok:
            self.world.log.append(f"SafetyShield rejected path: {reason}")
        return ok

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
            kind = change.get("kind", "obstacle")
            if action == "obstacle_added":
                parts.append(f"new {kind} at {change.get('position')}")
            elif action == "obstacle_moved":
                parts.append(f"{kind} moved to {change.get('new_position')}")
            elif action == "obstacle_removed":
                parts.append(f"{kind} removed from {change.get('old_position')}")
            elif action == "destination_changed":
                parts.append(f"destination changed to {change.get('destination')}")
        return "; ".join(parts) if parts else "environment updated"

    def _only_moving_obstacles_changed(self) -> bool:
        return self._static_obstacle_snapshot() == self._last_planned_static

    def _opposition_behind(self) -> bool:
        car_x, car_y = self._position()
        car_dir = self.world.car.direction
        for obs in self.world.active_obstacles():
            if obs.kind not in MOVING_KINDS:
                continue
            odx = getattr(obs, "dx", 0) or 0
            ody = getattr(obs, "dy", 0) or 0
            if car_dir == "east":
                if obs.y == car_y and 0 < car_x - obs.x <= 3 and odx > 0:
                    return True
            elif car_dir == "west":
                if obs.y == car_y and 0 < obs.x - car_x <= 3 and odx < 0:
                    return True
            elif car_dir == "north":
                if obs.x == car_x and 0 < obs.y - car_y <= 3 and ody < 0:
                    return True
            elif car_dir == "south":
                if obs.x == car_x and 0 < car_y - obs.y <= 3 and ody > 0:
                    return True
        return False

    def _apply_navigate_decision(self, decision: DriveDecision) -> None:
        if decision.destination in self.world.parking_zones:
            new_dest = self.world.parking_zones[decision.destination]
            if new_dest != self.world.car.destination:
                try:
                    self.world.set_destination(decision.destination)
                except ValueError as error:
                    self.world.log.append(f"Cannot adopt LLM destination: {error}")
            self.status.destination_source = "llm"

        # None == "field not returned by this LLM call" (e.g. replan).
        # Only overwrite when the LLM actually provided the field.
        if decision.avoid_obstacles is not None:
            self.current_avoid_obstacles = list(decision.avoid_obstacles)
        if decision.safety_vs_time is not None:
            self.current_safety_vs_time = float(decision.safety_vs_time)

    def _plan_and_validate(self):
        path = self._calculate_path()
        if path is None:
            reason = self._diagnose_no_path()
            self._set_state("blocked", f"No valid route right now. {reason} Retrying as world changes.")
            self._planned_revision = self.world.revision
            self._last_planned_route_obstacles = frozenset()
            self._last_planned_static = self._static_obstacle_snapshot()
            return False
        if not self._validate_path(path):
            self._set_state("safety_stop", "Route validation failed. Retrying as world changes.")
            self.status.safety_stops += 1
            self._planned_revision = self.world.revision
            self._last_planned_route_obstacles = frozenset()
            self._last_planned_static = self._static_obstacle_snapshot()
            return False
        self._planned_revision = self.world.revision
        self._last_planned_route_obstacles = self._route_neighbourhood_obstacles()
        self._last_planned_static = self._static_obstacle_snapshot()
        return True

    def needs_llm_replan(self) -> bool:
        if self.world.revision == self._planned_revision:
            return False
        if self._only_moving_obstacles_changed():
            return False

        self._refresh_soft_avoid_weights()
        new_path = self._call_astar()
        if new_path is None:
            return True
        if list(new_path) != list(self.status.path):
            return True
        if self._route_neighbourhood_obstacles() != self._last_planned_route_obstacles:
            return True
        return False

    def start(self, instruction: str, image_provider=None, use_llm: bool = True):
        self.current_instruction = instruction.strip()
        self.image_provider = image_provider
        self.use_llm = use_llm

        if not self.current_instruction:
            raise ValueError("Please provide a driving instruction.")

        if use_llm:
            allowed, refusal_reason = is_likely_driving_instruction(self.current_instruction)
            if not allowed:
                self.status.destination_source = "none"
                self._set_state("clarify", refusal_reason)
                return DriveDecision(decision="clarify", destination=None, reason=refusal_reason)

        if not use_llm:
            self.status.steps = 0
            self.status.destination_source = "dropdown"
            if not self._plan_and_validate():
                return None
            self._set_state("ready", f"A*-only route to {self._destination()}.")
            return DriveDecision(decision="navigate", destination=None, reason="A*-only mode.")

        self.status.llm_calls += 1
        decision = self.planner.decide(
            self.current_instruction, self.world, route_available=True, image_data_uri=self._get_scene_image()
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
            self._set_state("clarify", "The LLM selected an unknown destination.")
            return decision

        try:
            if self.world.parking_zones[decision.destination] != self.world.car.destination:
                self.world.set_destination(decision.destination)
        except ValueError as error:
            self.status.destination_source = "none"
            self._set_state("clarify", f"Cannot set destination: {error}")
            return decision

        self.status.destination_source = "llm"
        self.current_avoid_obstacles = list(decision.avoid_obstacles or [])
        self.current_safety_vs_time = float(
            decision.safety_vs_time if decision.safety_vs_time is not None else 0.5
        )
        self.status.steps = 0

        if not self._plan_and_validate():
            return decision

        self._set_state("ready", f"Route ready to {decision.destination}.")
        return decision

    def _replan_if_world_changed(self, skip_llm: bool = False) -> bool:
        if self.world.revision == self._planned_revision:
            return True

        self._skip_brake_once = False
        self.status.replans += 1
        reason = self._describe_changes(self.world.changes_since(self._planned_revision))
        new_path = self._call_astar()

        if new_path is None:
            reason = self._diagnose_no_path()
            self._set_state("blocked", f"No valid route right now. {reason} Retrying as world changes.")
            self._planned_revision = self.world.revision
            self._last_planned_route_obstacles = frozenset()
            self._last_planned_static = self._static_obstacle_snapshot()
            return False

        cells_changed = list(new_path) != list(self.status.path)
        neighbourhood_changed = (
            self._route_neighbourhood_obstacles() != self._last_planned_route_obstacles
        )
        if not cells_changed and not neighbourhood_changed:
            self._planned_revision = self.world.revision
            self._last_planned_static = self._static_obstacle_snapshot()
            return True

        detour_cost = len(new_path) - len(self.status.path)

        if self.use_llm and not skip_llm:
            try:
                self.status.llm_calls += 1
                decision = self.planner.decide_replan(
                    reason=reason,
                    world=self.world,
                    detour_cost=detour_cost,
                    image_data_uri=self._get_scene_image(),
                )
                if decision.decision == "wait":
                    self.status.replan_waits += 1
                    self._wait_steps = 0
                    self._set_state("waiting", f"LLM chose to wait (detour {detour_cost:+d}): {reason}.")
                    self._planned_revision = self.world.revision
                    self._last_planned_static = self._static_obstacle_snapshot()
                    return False
                if decision.decision == "stop":
                    self.status.safety_stops += 1
                    self._set_state("safety_stop", f"LLM stopped the car: {reason}.")
                    self._planned_revision = self.world.revision
                    self._last_planned_static = self._static_obstacle_snapshot()
                    return False
                if decision.decision == "navigate":
                    self._apply_navigate_decision(decision)
            except Exception as error:
                self.status.llm_errors += 1
                self.world.log.append(f"LLM replan failed: {error}. Falling back to A*.")

        self.status.replan_detours += 1
        self._set_state("replanning", f"World changed (detour {detour_cost:+d}): {reason}. Recalculating.")
        ok = self._plan_and_validate()
        if not ok:
            self.status.safety_overrides += 1
            self._set_state("waiting", f"Safety override (no path): {reason}.")
            self._wait_steps = 0
            self._planned_revision = self.world.revision
            self._last_planned_static = self._static_obstacle_snapshot()
            return False
        return True

    def step(self) -> DriveStatus:
        if self.status.state in RECOVERABLE_STATES:
            if self.world.revision == self._planned_revision:
                return self.status
            self._set_state("ready", f"Recovery from {self.status.state}: environment changed.")

        if self.status.state not in ACTIVE_STATES:
            return self.status

        if self.status.state == "braking":
            self._brake_ticks -= 1
            if self._brake_ticks <= 0:
                self._set_state("ready", "Braking complete. Proceeding.")
                self._skip_brake_once = True
            else:
                self.status.message = f"Braking slowly... ({self._brake_ticks} ticks left)"
            return self.status

        if self.status.state == "turning":
            self._turn_ticks -= 1
            if self._turn_ticks <= 0:
                self._set_state("ready", "Turn complete. Resuming.")
            else:
                self.status.message = f"Turning... ({self._turn_ticks} ticks left)"
            return self.status

        if self.status.state == "waiting":
            if self.world.revision == self._planned_revision:
                self._wait_steps += 1
                if self._wait_steps < WAIT_TIMEOUT_STEPS:
                    return self.status
                self._set_state("replanning", "Wait timeout — forcing fresh plan.")
                if not self._plan_and_validate():
                    return self.status
            else:
                self._wait_steps = 0

        if self._position() == self._destination():
            self._set_state("completed", "Destination reached successfully.")
            return self.status

        skip_llm = (
            self.world.revision != self._planned_revision
            and self._only_moving_obstacles_changed()
        )
        if not self._replan_if_world_changed(skip_llm=skip_llm):
            return self.status

        path = self.status.path
        if not path or path[0] != self._position():
            if not self._plan_and_validate():
                return self.status
            path = self.status.path

        if not self._validate_path(path):
            self.status.safety_stops += 1
            self._set_state("safety_stop", "Route validation failed. Retrying as world changes.")
            self._planned_revision = self.world.revision
            return self.status

        if len(path) < 2:
            self._set_state("completed", "Destination reached successfully.")
            return self.status

        next_position = path[1]

        if self.world.blocked(*next_position):
            self.status.safety_stops += 1
            self._set_state("safety_stop", f"Next cell {next_position} is blocked. Waiting.")
            self._planned_revision = self.world.revision
            return self.status

        old_position = self._position()
        dx = next_position[0] - old_position[0]
        dy = next_position[1] - old_position[1]
        new_direction = DIRECTIONS.get((dx, dy))
        if new_direction is None:
            self.status.safety_stops += 1
            self._set_state("safety_stop", "Invalid movement direction.")
            self._planned_revision = self.world.revision
            return self.status

        hazard_at_next = self.soft_avoid_weights.get(next_position, 0.0)
        is_traffic_ahead = any(
            o.kind in MOVING_KINDS and o.position == next_position
            for o in self.world.active_obstacles()
        )
        rear_threat = self._opposition_behind()
        wants_brake = (
            not self._skip_brake_once
            and (hazard_at_next > HAZARD_BRAKE_THRESHOLD or is_traffic_ahead)
        )
        if wants_brake and not rear_threat:
            self._brake_ticks = BRAKE_TICKS
            self._set_state(
                "braking",
                f"Hazard at {next_position} (weight {hazard_at_next:.1f}). Braking slowly...",
            )
            return self.status
        if wants_brake and rear_threat:
            self.world.log.append(
                "Opposition behind: skipping hazard brake to avoid rear-end."
            )
            self._skip_brake_once = True

        if new_direction != self.world.car.direction:
            self.world.car.direction = new_direction
            self._turn_ticks = TURN_TICKS
            self._set_state("turning", f"Turning to face {new_direction}...")
            return self.status

        self.world.car.direction = new_direction
        self.world.car.move_to(next_position)
        self.status.steps += 1
        self.status.path = path[1:]
        self._skip_brake_once = False
        self._set_state("moving", f"Car moved from {old_position} to {next_position}.")

        if self._position() == self._destination():
            self._set_state("completed", "Destination reached successfully.")

        return self.status