"""Diagnose the Ollama VLM call outside Streamlit.

    python examples/test_llm.py [--model qwen2.5vl:3b] [--no-image] [--max-dim 300]
"""
# test_llm.py
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import time

from vlmplan.drive_world import Car, DrivingWorld
from drive_planner import OllamaDrivePlanner, OllamaUnavailable
from drive_render import render_scene_snapshot   # no Streamlit import


def build_world():
    w = DrivingWorld(car=Car(x=2, y=10, destination=(18, 10), direction="east"))
    w.set_destination("parking_B")
    w.add_obstacle(8, 10, kind="cone", obstacle_id="c1")
    w.add_obstacle(14, 10, kind="cone", obstacle_id="c2")
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--no-image", action="store_true")
    ap.add_argument("--max-dim", type=int, default=400)
    ap.add_argument("--timeout", type=float, default=300.0)
    args = ap.parse_args()

    kwargs = {"timeout": args.timeout}
    if args.model:
        kwargs["model"] = args.model
    planner = OllamaDrivePlanner(**kwargs)
    print(f"Model: {planner.model} | Timeout: {planner.timeout:.0f}s")

    print("\n[1/3] Health check...")
    ok, status = planner.status()
    print(f"  -> {ok}: {status}")
    if not ok:
        return

    world = build_world()
    image = None
    if args.no_image:
        print("\n[2/3] Skipping image")
    else:
        image = render_scene_snapshot(world, None, max_dim=args.max_dim)
        print(f"\n[2/3] Rendered scene ({len(image)} chars)")

    print("\n[3/3] decide()... (30-300s on CPU)")
    t0 = time.time()
    try:
        d = planner.decide("Drive to parking B, stay far away from cones", world, image_data_uri=image)
        print(f"  -> {time.time() - t0:.1f}s | {d.decision} | dest={d.destination} | safety={d.safety_vs_time}")
        for spec in d.avoid_obstacles:
            print("      ", spec)
        print("  -> reasoning:", d.reasoning)
    except OllamaUnavailable as e:
        print(f"  -> FAILED after {time.time() - t0:.1f}s: {e}")


if __name__ == "__main__":
    main()