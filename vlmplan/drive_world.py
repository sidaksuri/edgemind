"""Dynamic driving world for the EdgeMind digital twin."""
from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field
from typing import Optional


GRID_SIZE = 20


@dataclass
class Car:
    x: int = 1
    y: int = 10
    destination: tuple[int, int] = (18, 10)
    direction: str = "east"

    @property
    def position(self) -> tuple[int, int]:
        return self.x, self.y

    def move_to(self, position: tuple[int, int]) -> None:
        self.x, self.y = position


@dataclass
class Obstacle:
    id: str
    x: int
    y: int
    kind: str = "barrier"
    active: bool = True

    @property
    def position(self) -> tuple[int, int]:
        return self.x, self.y


@dataclass
class DrivingWorld:
    car: Car
    obstacles: dict[str, Obstacle] = field(default_factory=dict)
    parking_zones: dict[str, tuple[int, int]] = field(
        default_factory=lambda: {
            "parking_A": (18, 3),
            "parking_B": (18, 10),
            "parking_C": (18, 17),
        }
    )
    log: list[str] = field(default_factory=list)
    revision: int = 0
    change_history: list[dict] = field(default_factory=list)
    _next_obstacle_id: int = 1

    def inside_grid(self, x: int, y: int) -> bool:
        return 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE

    def blocked(self, x, y, ignore_obstacle_id=None):
        if not self.inside_grid(x, y):
            return True
        for obstacle in self.obstacles.values():
            if not obstacle.active:
                continue
            if obstacle.id == ignore_obstacle_id:
                continue
            if obstacle.position == (x, y):
                return True
        return False

    def occupied_by_car(self, x, y):
        return self.car.position == (x, y)

    def cell_is_free(self, x, y, ignore_obstacle_id=None):
        return (
            self.inside_grid(x, y)
            and not self.blocked(x, y, ignore_obstacle_id)
            and not self.occupied_by_car(x, y)
        )

    def _record_change(self, action, details):
        self.revision += 1
        record = {"revision": self.revision, "action": action, **details}
        self.change_history.append(record)
        self.log.append(f"World revision {self.revision}: {action}")

    def changes_since(self, revision):
        return [c for c in self.change_history if c["revision"] > revision]

    def add_obstacle(self, x, y, kind="barrier", obstacle_id=None):
        if not self.inside_grid(x, y):
            raise ValueError("Obstacle position is outside the grid.")
        if self.occupied_by_car(x, y):
            raise ValueError("Cannot place an obstacle on the car.")
        if self.blocked(x, y):
            raise ValueError("Another obstacle already occupies this cell.")
        if obstacle_id is None:
            obstacle_id = f"obstacle_{self._next_obstacle_id}"
            self._next_obstacle_id += 1
        if obstacle_id in self.obstacles:
            raise ValueError(f"Obstacle ID already exists: {obstacle_id}")
        obstacle = Obstacle(id=str(obstacle_id), x=x, y=y, kind=kind, active=True)
        self.obstacles[obstacle.id] = obstacle
        self._record_change(
            "obstacle_added",
            {"obstacle_id": obstacle.id, "kind": kind, "position": (x, y)},
        )
        return obstacle

    def move_obstacle(self, obstacle_id, new_x, new_y):
        if obstacle_id not in self.obstacles:
            raise ValueError(f"Unknown obstacle: {obstacle_id}")
        if not self.inside_grid(new_x, new_y):
            raise ValueError("New obstacle position is outside the grid.")
        if self.occupied_by_car(new_x, new_y):
            raise ValueError("Cannot move an obstacle onto the car.")
        obstacle = self.obstacles[obstacle_id]
        if not obstacle.active:
            raise ValueError("Cannot move an inactive obstacle.")
        if self.blocked(new_x, new_y, ignore_obstacle_id=obstacle_id):
            raise ValueError("Another obstacle already occupies this cell.")
        old_position = obstacle.position
        obstacle.x = new_x
        obstacle.y = new_y
        self._record_change(
            "obstacle_moved",
            {
                "obstacle_id": obstacle.id,
                "old_position": old_position,
                "new_position": (new_x, new_y),
            },
        )
        return obstacle

    def remove_obstacle(self, obstacle_id):
        if obstacle_id not in self.obstacles:
            raise ValueError(f"Unknown obstacle: {obstacle_id}")
        obstacle = self.obstacles[obstacle_id]
        if not obstacle.active:
            return
        old_position = obstacle.position
        obstacle.active = False
        self._record_change(
            "obstacle_removed",
            {"obstacle_id": obstacle.id, "old_position": old_position},
        )

    def active_obstacles(self):
        return [o for o in self.obstacles.values() if o.active]

    def set_destination(self, zone):
        if zone not in self.parking_zones:
            raise ValueError(f"Unknown parking zone: {zone}")
        destination = self.parking_zones[zone]
        if self.blocked(*destination):
            raise ValueError("The selected parking zone is blocked.")
        self.car.destination = destination
        self._record_change(
            "destination_changed",
            {"destination": destination, "zone": zone},
        )

    def describe(self):
        lines = [
            "CURRENT DRIVING WORLD",
            f"Grid size: {GRID_SIZE} x {GRID_SIZE}",
            f"Car position: {self.car.position}",
            f"Car destination: {self.car.destination}",
            f"World revision: {self.revision}",
            "",
            "ACTIVE OBSTACLES:",
        ]
        active = self.active_obstacles()
        if not active:
            lines.append("No active obstacles.")
        else:
            for obstacle in active:
                lines.append(
                    f"- {obstacle.id}: {obstacle.kind} at {obstacle.position}"
                )
        lines.append("")
        lines.append("PARKING ZONES:")
        for name, position in self.parking_zones.items():
            state = "BLOCKED" if self.blocked(*position) else "AVAILABLE"
            lines.append(f"- {name}: {position} ({state})")
        return "\n".join(lines)

    def snapshot(self):
        return {
            "revision": self.revision,
            "car_position": self.car.position,
            "destination": self.car.destination,
            "obstacles": [
                {"id": o.id, "kind": o.kind, "x": o.x, "y": o.y}
                for o in self.active_obstacles()
            ],
        }


def heuristic(current, goal):
    return abs(current[0] - goal[0]) + abs(current[1] - goal[1])


def astar(
    world: DrivingWorld,
    start: tuple[int, int],
    goal: tuple[int, int],
    soft_avoid_cells: Optional[set[tuple[int, int]]] = None,
    soft_avoid_penalty: int = 100,
) -> Optional[list[tuple[int, int]]]:
    """Find a path through the latest world state.

    soft_avoid_cells adds a cost penalty to those cells, allowing
    risk-weighted routing without blocking. Hard constraints
    (world.blocked) still prevent traversal entirely.
    """

    if not world.inside_grid(*start):
        return None
    if not world.inside_grid(*goal):
        return None
    if world.blocked(*start) or world.blocked(*goal):
        return None
    if start == goal:
        return [start]

    soft_avoid = soft_avoid_cells or set()
    counter = itertools.count()
    open_set = [(heuristic(start, goal), 0, next(counter), start)]
    came_from = {}
    g_score = {start: 0}
    visited = set()
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    while open_set:
        _, current_cost, _, current = heapq.heappop(open_set)
        if current in visited:
            continue
        visited.add(current)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        for dx, dy in directions:
            neighbor = (current[0] + dx, current[1] + dy)
            if not world.inside_grid(*neighbor):
                continue
            if world.blocked(*neighbor):
                continue
            step_cost = 1 + (
                soft_avoid_penalty if neighbor in soft_avoid else 0
            )
            tentative_cost = current_cost + step_cost
            if tentative_cost < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_cost
                priority = tentative_cost + heuristic(neighbor, goal)
                heapq.heappush(
                    open_set,
                    (priority, tentative_cost, next(counter), neighbor),
                )

    return None


def create_demo_world():
    car = Car(x=1, y=10, destination=(18, 10), direction="east")
    world = DrivingWorld(car=car)
    world.add_obstacle(8, 10, kind="barrier")
    world.add_obstacle(8, 11, kind="barrier")
    world.add_obstacle(12, 7, kind="crate")
    world.add_obstacle(13, 13, kind="cone")
    return world