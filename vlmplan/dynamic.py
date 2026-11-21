"""Dynamic worlds: the ground truth can change between observations, independent of the
robot's own actions — an obstacle appears, someone moves an object, a bin jams shut, the
camera misfires for a frame. ``DynamicWorld`` schedules such events by observation-frame
number; ``agent.run_task``/``run_task_steps`` ticks it once per perceive-plan-act cycle,
so it's the robot's *next look* at the scene — not magic — that reveals the change and
drives the replan.

Events do not have to be known in advance. ``DynamicWorld.inject`` lets something external
(a human, in the Streamlit demo) add a brand-new event to a schedule that is already
running, for the very next observation. Because the schedule is just a plain list that
``tick`` re-scans every frame, this is safe to call between any two perception cycles —
the event did not exist when the task started, and the planner had no way to know it was
coming.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .scene import Scene
from .world import COLORS, Obj, World

EventFn = Callable[["DynamicWorld", np.random.Generator], str]


@dataclass
class Event:
    at_frame: int  # fires the first time the observation-frame counter reaches this
    apply: EventFn
    name: str = ""
    fired: bool = False


@dataclass
class DynamicWorld:
    """Wraps a :class:`World` with a schedule of off-script changes. The robot's belief
    (a ``Scene``, its digital twin of the table) can diverge from this; perception and
    replanning are what bring it back in sync."""

    world: World
    events: list[Event] = field(default_factory=list)
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))
    log: list[str] = field(default_factory=list)
    glitch_pending: bool = False  # armed by `misidentify`; consumed by `apply_glitch`
    last_frame: int = 0  # the most recent observation-frame number seen by `tick`

    def tick(self, frame: int) -> None:
        """Fire any events scheduled for this observation frame."""
        self.last_frame = frame
        for e in self.events:
            if not e.fired and frame >= e.at_frame:
                e.fired = True
                self.log.append(f"[frame {frame}] {e.name}: {e.apply(self, self.rng)}")

    def inject(self, apply_fn: EventFn, name: str = "live disruption", in_frames: int = 1) -> Event:
        """Schedule a brand-new event from *outside* a run already in progress (e.g. a
        human clicking a button mid-demo), to fire ``in_frames`` observations from now.
        This is what makes a live demo disruption genuine rather than scripted: the event
        is created at a moment and place the planner could not have anticipated, and is
        only ever revealed to the robot the next time it perceives."""
        ev = Event(self.last_frame + in_frames, apply_fn, name)
        self.events.append(ev)
        return ev

    def apply_glitch(self, scene: Scene) -> None:
        """Corrupt one detection in a freshly perceived belief scene, the way a bad camera
        frame would. The true world is untouched, so the *next* frame sees clearly again."""
        cands = [i for i, it in scene.items.items() if it.kind != "obstacle"]
        if not cands:
            self.glitch_pending = False
            return
        iid = str(self.rng.choice(cands))
        it = scene.items[iid]
        if self.rng.random() < 0.5:
            wrong = [c for c in COLORS if c != it.color] or [it.color]
            was, it.color = it.color, str(self.rng.choice(wrong))
            self.log.append(f"camera glitch: {iid} misread as {it.color} (was {was})")
        else:
            was, it.kind = it.kind, ("ball" if it.kind == "cube" else "cube")
            self.log.append(f"camera glitch: {iid} misread as a {it.kind} (was {was})")
        self.glitch_pending = False


# ------------------------------------------------------------------ event constructors
def spawn_obstacle(oid: str, xy: tuple[float, float]) -> EventFn:
    def _fn(dw: DynamicWorld, rng: np.random.Generator) -> str:
        dw.world.objects[oid] = Obj(oid, "obstacle", "gray", xy)
        return f"obstacle appeared ({oid}) at {xy}"
    return _fn


def remove_obstacle(oid: str) -> EventFn:
    def _fn(dw: DynamicWorld, rng: np.random.Generator) -> str:
        dw.world.objects.pop(oid, None)
        return f"obstacle removed ({oid})"
    return _fn


def nudge(oid: str, xy: tuple[float, float] | None = None) -> EventFn:
    """Someone moves an object on the table, off-script."""
    def _fn(dw: DynamicWorld, rng: np.random.Generator) -> str:
        o = dw.world.objects[oid]
        new_xy = xy if xy is not None else (float(rng.uniform(0.1, 0.9)), float(rng.uniform(0.36, 0.95)))
        o.on, o.xy = None, new_xy
        return f"{o.name} {oid} moved to ({new_xy[0]:.2f}, {new_xy[1]:.2f})"
    return _fn


def set_bin_blocked(bid: str, blocked: bool = True) -> EventFn:
    def _fn(dw: DynamicWorld, rng: np.random.Generator) -> str:
        dw.world.objects[bid].blocked = blocked
        return f"{bid} {'blocked' if blocked else 'cleared'}"
    return _fn


def misidentify() -> EventFn:
    """Schedules a one-frame perception fault (does not touch the true world)."""
    def _fn(dw: DynamicWorld, rng: np.random.Generator) -> str:
        dw.glitch_pending = True
        return "camera glitch armed for next frame"
    return _fn


__all__ = ["Event", "DynamicWorld", "spawn_obstacle", "remove_obstacle", "nudge",
          "set_bin_blocked", "misidentify"]