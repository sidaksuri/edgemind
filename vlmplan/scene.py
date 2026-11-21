"""The robot's belief: a scene graph built from perception (or, for ablations, from ground truth)."""
# scene.py
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .vision import perceive
from .world import BASE, OBST_RADIUS, REACH, STACK_LIMIT, STACK_OFFSET, World


@dataclass
class Item:
    id: str
    kind: str  # cube | ball | obstacle
    color: str
    xy: tuple[float, float]
    on: str | None = None  # item id below it
    in_bin: str | None = None  # bin id

    @property
    def name(self) -> str:
        return f"{self.color} {self.kind}"

    @property
    def graspable(self) -> bool:
        return self.kind in ("cube", "ball")


@dataclass
class Scene:
    items: dict[str, Item]
    bins: dict[str, str]  # bin id -> colour
    bin_xy: dict[str, tuple[float, float]] = field(default_factory=dict)
    blocked_bins: set[str] = field(default_factory=set)  # bins currently jammed / unavailable
    capacity: int = 4
    holding: str | None = None

    def above(self, iid: str) -> list[str]:
        return [i.id for i in self.items.values() if i.on == iid]

    def height(self, iid: str) -> int:
        h, cur = 1, self.items[iid]
        while cur.on is not None:
            h, cur = h + 1, self.items[cur.on]
        return h

    def reachable(self, iid: str) -> bool:
        x, y = self.items[iid].xy
        return float(np.hypot(x - BASE[0], y - BASE[1])) <= REACH

    def blocked(self, iid: str) -> bool:
        """True if an obstacle sits close enough to make ``iid`` unreachable."""
        x, y = self.items[iid].xy
        return any(o.kind == "obstacle" and float(np.hypot(x - o.xy[0], y - o.xy[1])) < OBST_RADIUS
                   for o in self.items.values())

    def bin_blocked(self, bid: str) -> bool:
        return bid in self.blocked_bins

    def bin_count(self, bid: str) -> int:
        return sum(i.in_bin == bid for i in self.items.values())

    def find(self, color: str | None = None, kind: str | None = None) -> list[str]:
        return [i.id for i in self.items.values() if (color is None or i.color == color) and (kind is None or i.kind == kind)]

    def bin_by_color(self, color: str) -> str | None:
        return next((b for b, c in self.bins.items() if c == color), None)

    def describe(self) -> str:
        lines = [f"{i.id}: {i.name} at ({i.xy[0]:.2f}, {i.xy[1]:.2f})"
                 + (f", on {i.on}" if i.on else "") + (f", in {i.in_bin}" if i.in_bin else "")
                 + ("" if self.reachable(i.id) else ", OUT OF REACH")
                 + (", BLOCKED BY OBSTACLE" if i.kind != "obstacle" and self.blocked(i.id) else "")
                 for i in self.items.values()]
        lines += [f"{b}: {c} bin ({self.bin_count(b)}/{self.capacity} used)"
                 + (" BLOCKED" if self.bin_blocked(b) else "") for b, c in self.bins.items()]
        return "\n".join(lines)


def scene_from_image(img: np.ndarray) -> Scene:
    dets, bins = perceive(img)
    bin_ids = {f"bin_{b.color}": b.color for b in bins}
    bin_xy = {f"bin_{b.color}": b.xy for b in bins}
    blocked_bins = {f"bin_{b.color}" for b in bins if b.blocked}
    objs = sorted((d for d in dets if d.kind != "obstacle"), key=lambda d: (d.xy[1], d.xy[0]))
    items = {}
    for k, d in enumerate(objs):
        items[f"d{k}"] = Item(f"d{k}", d.kind, d.color, d.xy, in_bin=f"bin_{d.in_bin}" if d.in_bin else None)
    levels = {f"d{k}": d.level for k, d in enumerate(objs)}
    for iid, lvl in levels.items():  # a level-L item sits on the level-(L-1) item at the stacking offset
        if lvl > 1 and items[iid].in_bin is None:
            x, y = items[iid].xy
            cand = [j for j, l2 in levels.items() if l2 == lvl - 1 and j != iid]
            if cand:
                j = min(cand, key=lambda j: np.hypot(items[j].xy[0] + STACK_OFFSET[0] - x,
                                                     items[j].xy[1] + STACK_OFFSET[1] - y))
                items[iid].on = j
    obst_dets = [d for d in dets if d.kind == "obstacle"]
    for k, d in enumerate(obst_dets):
        items[f"obs{k}"] = Item(f"obs{k}", "obstacle", d.color, d.xy)
    return Scene(items, bin_ids, bin_xy, blocked_bins=blocked_bins)


def scene_from_world(w: World) -> Scene:
    """Ground-truth belief (perfect perception), for ablations."""
    items = {}
    for o in w.items():
        on = o.on if o.on is not None and w.objects[o.on].kind != "bin" else None
        items[o.id] = Item(o.id, o.kind, o.color, o.xy, on=on, in_bin=w.in_bin(o.id))
    for o in w.obstacles():
        items[o.id] = Item(o.id, "obstacle", o.color, o.xy)
    blocked_bins = {b.id for b in w.bins() if b.blocked}
    return Scene(items, {b.id: b.color for b in w.bins()}, {b.id: b.xy for b in w.bins()}, blocked_bins=blocked_bins)


__all__ = ["Item", "Scene", "scene_from_image", "scene_from_world", "STACK_LIMIT"]