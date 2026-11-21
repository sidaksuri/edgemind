
"""Dynamic tabletop world for EdgeMind.

The true world state is stored here. The planner should receive observations
through the perception and scene modules, not access this state directly.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

COLORS = {
    "red": (220, 40, 40),
    "green": (40, 170, 70),
    "blue": (40, 90, 220),
    "yellow": (235, 200, 30),
}

SHAPES = ("cube", "ball")
TABLE = (0.0, 1.0)
REACH = 0.62
BASE = (0.05, 0.5)
OBST_RADIUS = 0.065
STACK_LIMIT = 3
STACK_OFFSET = (0.08, 0.08)
BIN_HALF = (0.17, 0.09)

# A unique colour reserved for obstacle rendering and perception.
OBSTACLE_COLOR = (180, 40, 180)


@dataclass
class Obj:
    id: str
    kind: str
    color: str
    xy: tuple[float, float]
    on: str | None = None
    capacity: int = 0

    @property
    def graspable(self) -> bool:
        return self.kind in SHAPES

    @property
    def name(self) -> str:
        return f"{self.color} {self.kind}"


@dataclass
class Obstacle:
    """A fixed obstacle in the tabletop world.

    shape:
        "circle"    - radius is used
        "rectangle" - width and height are used
    """
    id: str
    xy: tuple[float, float]
    shape: str = "circle"
    radius: float = 0.065
    width: float = 0.14
    height: float = 0.14

    def __post_init__(self):
        if self.shape not in ("circle", "rectangle"):
            raise ValueError("Obstacle shape must be 'circle' or 'rectangle'.")

        x, y = self.xy
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ValueError("Obstacle position must be inside the table.")

        if self.radius <= 0 or self.width <= 0 or self.height <= 0:
            raise ValueError("Obstacle dimensions must be positive.")


@dataclass
class World:
    objects: dict[str, Obj]
    holding: str | None = None
    log: list[str] = field(default_factory=list)
    obstacles: dict[str, Obstacle] = field(default_factory=dict)

    def copy(self) -> "World":
        return copy.deepcopy(self)

    # -------------------------------------------------------------
    # Object relationships
    # -------------------------------------------------------------

    def above(self, oid: str) -> list[str]:
        return [
            obj.id for obj in self.objects.values()
            if obj.on == oid
        ]

    def is_clear(self, oid: str) -> bool:
        return not self.above(oid)

    def height(self, oid: str) -> int:
        h = 1
        cur = self.objects[oid]

        while cur.on is not None:
            parent = self.objects[cur.on]
            if parent.kind == "bin":
                break
            h += 1
            cur = parent

        return h

    def in_bin(self, oid: str) -> str | None:
        cur = self.objects[oid]

        while cur.on is not None:
            nxt = self.objects[cur.on]
            if nxt.kind == "bin":
                return nxt.id
            cur = nxt

        return None

    def items(self) -> list[Obj]:
        return [
            obj for obj in self.objects.values()
            if obj.kind != "bin"
        ]

    def bins(self) -> list[Obj]:
        return [
            obj for obj in self.objects.values()
            if obj.kind == "bin"
        ]

    # -------------------------------------------------------------
    # Reachability and obstacle collision
    # -------------------------------------------------------------

    def reachable(self, oid: str) -> bool:
        """Check arm reach only. Preserves the original behaviour."""
        x, y = self.objects[oid].xy
        return float(np.hypot(x - BASE[0], y - BASE[1])) <= REACH

    @staticmethod
    def _point_inside_obstacle(
        point: tuple[float, float],
        obstacle: Obstacle,
        margin: float = 0.0,
    ) -> bool:
        x, y = point
        ox, oy = obstacle.xy

        if obstacle.shape == "circle":
            distance = float(np.hypot(x - ox, y - oy))
            return distance <= obstacle.radius + margin

        half_w = obstacle.width / 2 + margin
        half_h = obstacle.height / 2 + margin

        return (
            abs(x - ox) <= half_w
            and abs(y - oy) <= half_h
        )

    def position_blocked(
        self,
        point: tuple[float, float],
        margin: float = 0.0,
    ) -> bool:
        """Return True if a position overlaps any obstacle."""
        return any(
            self._point_inside_obstacle(point, obstacle, margin)
            for obstacle in self.obstacles.values()
        )

    def path_blocked(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        margin: float = 0.0,
        ignore_obstacle: str | None = None,
    ) -> bool:
        """Check whether a straight-line path intersects an obstacle.

        The segment is sampled at short intervals. This is a simple
        collision test; a later path planner can search for alternate routes.
        """
        sx, sy = start
        ex, ey = end

        distance = float(np.hypot(ex - sx, ey - sy))
        samples = max(2, int(distance / 0.005) + 1)

        for i in range(samples + 1):
            t = i / samples
            point = (
                sx + t * (ex - sx),
                sy + t * (ey - sy),
            )

            for obstacle in self.obstacles.values():
                if obstacle.id == ignore_obstacle:
                    continue

                if self._point_inside_obstacle(
                    point, obstacle, margin
                ):
                    return True

        return False

    def reachable_path(self, oid: str) -> bool:
        """Check both arm reach and a clear straight-line path."""
        if not self.reachable(oid):
            return False

        return not self.path_blocked(BASE, self.objects[oid].xy)

    # -------------------------------------------------------------
    # Dynamic obstacle management
    # -------------------------------------------------------------

    def add_obstacle(self, obstacle: Obstacle) -> None:
        if obstacle.id in self.obstacles:
            raise ValueError(
                f"Obstacle '{obstacle.id}' already exists."
            )

        self.obstacles[obstacle.id] = obstacle
        self.log.append(
            f"Obstacle {obstacle.id} added at {obstacle.xy}."
        )

    def move_obstacle(
        self,
        obstacle_id: str,
        new_xy: tuple[float, float],
    ) -> None:
        if obstacle_id not in self.obstacles:
            raise KeyError(
                f"Unknown obstacle: {obstacle_id}"
            )

        x, y = new_xy
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ValueError(
                "Obstacle position must be inside the table."
            )

        obstacle = self.obstacles[obstacle_id]
        obstacle.xy = (float(x), float(y))

        self.log.append(
            f"Obstacle {obstacle_id} moved to {obstacle.xy}."
        )

    def remove_obstacle(self, obstacle_id: str) -> None:
        if obstacle_id not in self.obstacles:
            raise KeyError(
                f"Unknown obstacle: {obstacle_id}"
            )

        del self.obstacles[obstacle_id]
        self.log.append(
            f"Obstacle {obstacle_id} removed."
        )


# -------------------------------------------------------------
# Initial world generation
# -------------------------------------------------------------

def random_world(
    rng: np.random.Generator,
    n_items: int = 6,
    n_bins: int = 2,
    far_prob: float = 0.1,
) -> World:
    objs: dict[str, Obj] = {}

    bin_colors = rng.choice(
        list(COLORS),
        size=n_bins,
        replace=False,
    )

    for i, color in enumerate(bin_colors):
        objs[f"bin{i}"] = Obj(
            f"bin{i}",
            "bin",
            str(color),
            (0.22 + 0.36 * i, 0.18),
            capacity=4,
        )

    taken = [obj.xy for obj in objs.values()]
    k = 0

    while k < n_items:
        far = rng.random() < far_prob

        if far:
            xy = (
                float(rng.uniform(0.5, 0.95)),
                float(rng.uniform(0.05, 0.95)),
            )
        else:
            xy = (
                float(rng.uniform(0.15, 0.6)),
                float(rng.uniform(0.36, 0.9)),
            )

        if any(
            np.hypot(xy[0] - x, xy[1] - y) < 0.16
            for x, y in taken
        ):
            continue

        taken.append(xy)

        objs[f"o{k}"] = Obj(
            f"o{k}",
            str(rng.choice(SHAPES, p=[0.65, 0.35])),
            str(rng.choice(list(COLORS))),
            xy,
        )

        k += 1

    world = World(objs)

    # Sometimes start with a small stack.
    cubes = [
        obj for obj in world.items()
        if obj.kind == "cube"
    ]

    if len(cubes) >= 2 and rng.random() < 0.5:
        top, base = cubes[0], cubes[1]
        top.on = base.id
        top.xy = (
            base.xy[0] + STACK_OFFSET[0],
            base.xy[1] + STACK_OFFSET[1],
        )

    return world
