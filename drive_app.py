from __future__ import annotations

import base64
import copy
import io
import json
import random
import re
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw

from vlmplan.drive_world import Car, DrivingWorld, astar
from drive_agent import DriveAgent
from drive_planner import OllamaUnavailable


st.set_page_config(page_title="EdgeMind - Driving Digital Twin", page_icon="🚗", layout="wide")

ROOT_DIR = Path(__file__).resolve().parent
EDITOR_DIR = ROOT_DIR / "edgemind_editor"

WIDTH = 900
HEIGHT = 600
GRID = 20
CELL_W = WIDTH / GRID
CELL_H = HEIGHT / GRID
TARGET_OBSTACLE_COUNT = 5

PARKING_ZONES = {
    "A": ("parking_A", (18, 3)),
    "B": ("parking_B", (18, 10)),
    "C": ("parking_C", (18, 17)),
}

MOVING_KINDS = ("traffic_car", "opposition_car")

editor_component = components.declare_component("edgemind_editor", path=str(EDITOR_DIR))


def _add_obstacle_compat(world, x, y, *, kind, obstacle_id,
                        fallout=1, speed=0, dx=0, dy=0, **_unused):
    kwargs = {
        "kind": kind,
        "obstacle_id": obstacle_id,
        "fallout": fallout,
        "speed": speed,
        "dx": dx,
        "dy": dy,
    }

    while True:
        try:
            world.add_obstacle(x, y, **kwargs)
            break
        except TypeError as exc:
            match = re.search(r"unexpected keyword argument '([^']+)'", str(exc))
            if not match:
                raise
            rejected = match.group(1)
            if rejected in ("kind", "obstacle_id"):
                raise
            kwargs.pop(rejected, None)

    obstacles = getattr(world, "obstacles", None)
    stored = None
    if isinstance(obstacles, dict):
        stored = obstacles.get(obstacle_id)
    elif obstacles is not None:
        try:
            stored = obstacles[obstacle_id]
        except (KeyError, IndexError, TypeError):
            stored = None

    if stored is not None:
        modified = False
        for attr, desired in (("fallout", int(fallout)),
                              ("speed", speed),
                              ("dx", dx),
                              ("dy", dy)):
            if getattr(stored, attr, None) == desired:
                continue
            try:
                setattr(stored, attr, desired)
                modified = True
            except (AttributeError, TypeError):
                pass
        # FIX #5: if we bypassed the world's add_obstacle (older API rejected
        # some kwargs), the revision counter was not bumped for the attribute
        # changes we just applied. Record the change so DriveAgent's
        # revision-gated replanning stays consistent.
        if modified and hasattr(world, "_record_change"):
            try:
                world._record_change(
                    "obstacle_attributes_updated",
                    {"obstacle_id": obstacle_id},
                )
            except Exception:
                pass


DEFAULTS = {
    "editor_objects": [],
    "planned_route": [],
    "ghost_route": [],
    "soft_avoid_cells": [],
    "hazard_cells": [],
    "last_plan_status": "No route planned yet.",
    "background_data": None,
    "car_sprite": None,
    "cone_sprite": None,
    "barrier_sprite": None,
    "opposition_car_sprite": None,
    "editor_revision": 0,
    "drive_world": None,
    "drive_agent": None,
    "auto_drive": False,
    "auto_drive_toggle_revision": 0,
    "editor_warning": None,
    "dynamic_events_enabled": False,
    "dynamic_event_tick_counter": 0,
    "dynamic_event_count": 0,
    "dynamic_event_history": [],
    "dynamic_event_id_counter": 0,
    "movement_tick_counter": 0,
    "traffic_speed_counter": 0,
    "opposition_car_tick_counter": 0,
    "last_processed_component_key": None,
    "benchmark_result": None,
    "demo_lock": None,
    "demo_lock_countdown": 0,
    "_pending_plan_request": None,
    "_pending_step": False,
    "_step_consumed_this_run": False,
    "_opp_targets": {},
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value
st.session_state.pop("auto_drive_toggle", None)


def _parking_cells(world):
    try:
        return set(world.parking_zones.values())
    except (AttributeError, TypeError):
        return set()


def image_to_data_uri(uploaded_file, max_size=(900, 600)):
    if uploaded_file is None: return None
    try:
        uploaded_file.seek(0)
        image = Image.open(uploaded_file).convert("RGBA")
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        output = io.BytesIO()
        image.save(output, format="PNG", optimize=True)
        encoded = base64.b64encode(output.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"
    except Exception as error:
        st.error(f"Image processing failed: {error}")
        return None


def canvas_to_grid(x, y):
    return max(0, min(GRID - 1, int(float(x) / CELL_W))), max(0, min(GRID - 1, int(float(y) / CELL_H)))


def grid_to_canvas(x, y):
    return ((x + 0.5) * CELL_W, (y + 0.5) * CELL_H)


def objects_equal_by_grid(a, b) -> bool:
    if not isinstance(a, list) or not isinstance(b, list): return a == b
    if len(a) != len(b): return False
    a_by_id = {o.get("id"): o for o in a if isinstance(o, dict)}
    b_by_id = {o.get("id"): o for o in b if isinstance(o, dict)}
    if set(a_by_id) != set(b_by_id): return False
    for oid in a_by_id:
        ao, bo = a_by_id[oid], b_by_id[oid]
        if ao.get("type") != bo.get("type"): return False
        try:
            if canvas_to_grid(ao["x"], ao["y"]) != canvas_to_grid(bo["x"], bo["y"]): return False
        except (KeyError, TypeError, ValueError): return False
    return True


def render_scene_snapshot(world, background_data_uri=None, max_dim=400):
    base = Image.new("RGB", (WIDTH, HEIGHT), "#e5e7eb")
    if background_data_uri:
        try:
            encoded = background_data_uri.split(",", 1)[-1]
            bg = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB").resize((WIDTH, HEIGHT))
            base = bg
        except Exception: pass
    draw = ImageDraw.Draw(base)
    for i in range(GRID + 1):
        draw.line([(i * CELL_W, 0), (i * CELL_W, HEIGHT)], fill=(120, 130, 145))
        draw.line([(0, i * CELL_H), (WIDTH, i * CELL_H)], fill=(120, 130, 145))
    zone_colors = {"parking_A": (22, 163, 74), "parking_B": (37, 99, 235), "parking_C": (147, 51, 234)}
    for name, (gx, gy) in world.parking_zones.items():
        px, py = grid_to_canvas(gx, gy)
        draw.rectangle([px - CELL_W / 2, py - CELL_H / 2, px + CELL_W / 2, py + CELL_H / 2],
                       outline=zone_colors.get(name, (100, 100, 100)), width=3)
    for obs in world.active_obstacles():
        ox, oy = grid_to_canvas(obs.x, obs.y)
        if obs.kind == "opposition_car": color = (220, 38, 38)
        elif obs.kind == "traffic_car": color = (16, 185, 129)
        elif obs.kind == "cone": color = (249, 115, 22)
        else: color = (153, 27, 27)
        draw.ellipse([ox - 15, oy - 15, ox + 15, oy + 15], fill=color)
    cx, cy = grid_to_canvas(*world.car.position)
    draw.rectangle([cx - 20, cy - 20, cx + 20, cy + 20], fill=(37, 99, 235))
    dx, dy = grid_to_canvas(*world.car.destination)
    draw.rectangle([dx - 22, dy - 22, dx + 22, dy + 22], outline=(22, 163, 74), width=4)
    if max_dim and max(base.size) > max_dim:
        ratio = max_dim / max(base.size)
        base = base.resize((max(1, int(base.width * ratio)), max(1, int(base.height * ratio))), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    base.save(buf, format="PNG", optimize=True)
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"


def clean_editor_objects(value):
    if not isinstance(value, list): return None
    cleaned = []
    for obj in value:
        if isinstance(obj, dict) and obj.get("type") in ("car", "cone", "barrier", "traffic_car", "opposition_car"):
            try:
                cleaned.append({
                    "id": str(obj["id"]), "type": obj["type"],
                    "x": max(0.0, min(WIDTH, float(obj["x"]))),
                    "y": max(0.0, min(HEIGHT, float(obj["y"]))),
                    "size": float(obj.get("size", 55)),
                    "fallout": int(obj.get("fallout", 1)),
                })
            except (KeyError, ValueError, TypeError): continue
    return cleaned


def refresh_editor_component(): st.session_state.editor_revision += 1


def stop_automatic_driving():
    if st.session_state.get("auto_drive", False):
        st.session_state.auto_drive = False
        st.session_state.auto_drive_toggle_revision = st.session_state.get("auto_drive_toggle_revision", 0) + 1


def clear_navigation():
    st.session_state.drive_world = None
    st.session_state.drive_agent = None
    st.session_state.planned_route = []
    st.session_state.ghost_route = []
    st.session_state.soft_avoid_cells = []
    st.session_state.hazard_cells = []
    stop_automatic_driving()
    st.session_state.dynamic_event_tick_counter = 0
    st.session_state.dynamic_event_count = 0
    st.session_state.dynamic_event_history = []
    st.session_state.movement_tick_counter = 0
    st.session_state.traffic_speed_counter = 0
    st.session_state.opposition_car_tick_counter = 0
    st.session_state["_opp_targets"] = {}
    st.session_state["_step_consumed_this_run"] = False


def _refresh_comparison_overlays(world, agent):
    if world is None or agent is None:
        st.session_state.ghost_route = []
        st.session_state.soft_avoid_cells = []
        st.session_state.hazard_cells = []
        return
    try:
        ghost = astar(world, world.car.position, world.car.destination)
        st.session_state.ghost_route = [list(p) for p in (ghost or [])]
    except Exception as e:
        st.session_state.ghost_route = []
        try: world.log.append(f"Ghost route computation failed: {e}")
        except Exception: pass
    st.session_state.soft_avoid_cells = [
        [int(c[0]), int(c[1]), float(agent.soft_avoid_weights[c])]
        for c in agent.soft_avoid_weights
    ]
    actual_path = agent.status.path or []
    st.session_state.hazard_cells = [list(p) for p in actual_path if p in agent.soft_avoid_weights]


def sync_car_to_editor():
    world, agent = st.session_state.drive_world, st.session_state.drive_agent
    if world is None or agent is None: return
    car_x, car_y = grid_to_canvas(*world.car.position)
    updated = []
    for obj in st.session_state.editor_objects:
        copy_obj = dict(obj)
        if copy_obj["type"] == "car":
            copy_obj["x"], copy_obj["y"] = car_x, car_y
        updated.append(copy_obj)

    moving_in_world = {o.id: o for o in world.active_obstacles() if o.kind in MOVING_KINDS}
    moving_in_editor = {o["id"]: o for o in updated if o["type"] in MOVING_KINDS}
    updated = [o for o in updated if o["type"] not in MOVING_KINDS or o["id"] in moving_in_world]
    for tid, obs in moving_in_world.items():
        tx, ty = grid_to_canvas(obs.x, obs.y)
        if tid in moving_in_editor:
            for o in updated:
                if o["id"] == tid: o["x"], o["y"] = tx, ty; break
        else:
            updated.append({"id": tid, "type": obs.kind, "x": tx, "y": ty, "size": 55})

    st.session_state.editor_objects = updated
    st.session_state.planned_route = [list(p) for p in agent.status.path]
    _refresh_comparison_overlays(world, agent)
    st.session_state.last_plan_status = f"{agent.status.state.capitalize()}: {agent.status.message}"


def load_demo_scenario():
    demo_objects = []
    cx, cy = grid_to_canvas(2, 10)
    demo_objects.append({"id": "demo_car", "type": "car", "x": cx, "y": cy, "size": 55})
    ox1, oy1 = grid_to_canvas(8, 10)
    demo_objects.append({"id": "demo_cone_0", "type": "cone", "x": ox1, "y": oy1, "size": 55, "fallout": 4})
    ox2, oy2 = grid_to_canvas(14, 10)
    demo_objects.append({"id": "demo_cone_1", "type": "cone", "x": ox2, "y": oy2, "size": 55, "fallout": 2})
    st.session_state.editor_objects = demo_objects
    st.session_state.demo_lock = [dict(o) for o in demo_objects]
    st.session_state.demo_lock_countdown = 6
    clear_navigation()
    st.session_state.last_processed_component_key = None
    st.session_state.destination_select = "B"
    st.session_state.instruction_text_input = "Drive to parking B, stay far away from cones"
    st.session_state.benchmark_result = None
    refresh_editor_component()


def find_free_grid_cell(world, ignore_obstacle_id=None, prefer_route=True):
    protected = set()
    for zone_xy in world.parking_zones.values():
        protected.add(zone_xy)
        zx, zy = zone_xy
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)): protected.add((zx + dx, zy + dy))
    agent = st.session_state.drive_agent
    if prefer_route and agent is not None:
        candidates = [p for p in agent.status.path[2:] if p not in protected and world.cell_is_free(p[0], p[1], ignore_obstacle_id)]
        if candidates: return random.choice(candidates)
    ignored_pos = None
    if ignore_obstacle_id:
        obstacles = getattr(world, "obstacles", None)
        ignored_obs = None
        if isinstance(obstacles, dict):
            ignored_obs = obstacles.get(ignore_obstacle_id)
        elif obstacles is not None:
            try: ignored_obs = obstacles[ignore_obstacle_id]
            except (KeyError, IndexError, TypeError): ignored_obs = None
        if ignored_obs is not None:
            ignored_pos = getattr(ignored_obs, "position", None)
    free_cells = [
        (x, y) for y in range(GRID) for x in range(GRID)
        if (x, y) not in protected
        and not (ignored_pos is not None and (x, y) == ignored_pos)
        and world.cell_is_free(x, y, ignore_obstacle_id)
    ]
    return random.choice(free_cells) if free_cells else None


def next_obstacle_id(prefix):
    st.session_state.dynamic_event_id_counter += 1
    return f"{prefix}_{st.session_state.dynamic_event_id_counter}"


def append_editor_obstacle(obstacle_id, kind, position, fallout=1):
    canvas_x, canvas_y = grid_to_canvas(*position)
    st.session_state.editor_objects.append({"id": obstacle_id, "type": kind, "x": canvas_x, "y": canvas_y, "size": 55, "fallout": fallout})


def spawn_opposition_car(world):
    parking = _parking_cells(world)
    agent = st.session_state.drive_agent
    path = agent.status.path if agent else []

    # Preferred: spawn ahead on the main car's path to force an interaction
    if path and len(path) > 8:
        candidates = path[5:12]
        random.shuffle(candidates)
        for x, y in candidates:
            if (world.cell_is_free(x, y) and (x, y) not in parking
                    and (x, y) != world.car.position and (x, y) != world.car.destination):
                tid = next_obstacle_id("opp")
                try:
                    _add_obstacle_compat(world, x, y, kind="opposition_car", obstacle_id=tid,
                                         fallout=1, speed=1, dx=0, dy=0)
                    append_editor_obstacle(tid, "opposition_car", (x, y))
                    return True, tid
                except (ValueError, TypeError, KeyError): continue

    # Fallback: random spawn
    for _ in range(30):
        x, y = random.randint(2, 17), random.randint(2, 17)
        if not world.cell_is_free(x, y): continue
        if (x, y) in parking: continue
        if abs(x - world.car.position[0]) < 4 and abs(y - world.car.position[1]) < 4: continue
        tid = next_obstacle_id("opp")
        try:
            _add_obstacle_compat(world, x, y, kind="opposition_car", obstacle_id=tid,
                                 fallout=1, speed=1, dx=0, dy=0)
            append_editor_obstacle(tid, "opposition_car", (x, y))
            return True, tid
        except (ValueError, TypeError, KeyError): continue
    return False, "No free cell available."


def update_opposition_cars(world):
    if not st.session_state.get("auto_drive", False): return
    parking = _parking_cells(world)

    agent = st.session_state.drive_agent
    path = agent.status.path if agent else []

    opp_targets = st.session_state.get("_opp_targets") or {}
    if not isinstance(opp_targets, dict): opp_targets = {}

    for obs in list(world.active_obstacles()):
        if obs.kind != "opposition_car": continue

        # Force move off parking cells
        if (obs.x, obs.y) in parking:
            for d in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = obs.x + d[0], obs.y + d[1]
                if (world.inside_grid(nx, ny) and (nx, ny) not in parking
                        and world.car.position != (nx, ny) and not world.blocked(nx, ny, ignore_obstacle_id=obs.id)):
                    try:
                        world.move_obstacle(obs.id, nx, ny)
                        for obj in st.session_state.editor_objects:
                            if obj["id"] == obs.id:
                                obj["x"], obj["y"] = grid_to_canvas(nx, ny); break
                        break
                    except (ValueError, TypeError): continue
            continue

        target = opp_targets.get(obs.id)

        # Pick new target if none or if we just reached the current one
        if not isinstance(target, tuple) or target == (obs.x, obs.y):
            if path and len(path) > 4 and random.random() < 0.6:
                # 60% chance to aim for a point on the main car's path ahead
                target = path[random.randint(3, min(12, len(path) - 1))]
            else:
                # 40% chance to just wander to a random map coordinate
                target = (random.randint(1, GRID-2), random.randint(1, GRID-2))
            opp_targets[obs.id] = target

        tx, ty = target
        moves = []

        # Greedy move towards target
        if tx > obs.x: moves.append((1, 0))
        elif tx < obs.x: moves.append((-1, 0))
        if ty > obs.y: moves.append((0, 1))
        elif ty < obs.y: moves.append((0, -1))

        # Fallback directions if the direct path is blocked
        fallbacks = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        random.shuffle(fallbacks)
        for f in fallbacks:
            if f not in moves: moves.append(f)

        moved = False
        for dx, dy in moves:
            nx, ny = obs.x + dx, obs.y + dy
            if (not world.inside_grid(nx, ny) or (nx, ny) in parking or world.car.position == (nx, ny)
                    or world.blocked(nx, ny, ignore_obstacle_id=obs.id)):
                continue
            try:
                world.move_obstacle(obs.id, nx, ny)
                try: obs.dx, obs.dy = dx, dy
                except (AttributeError, TypeError): pass
                for obj in st.session_state.editor_objects:
                    if obj["id"] == obs.id:
                        obj["x"], obj["y"] = grid_to_canvas(nx, ny); break
                moved = True
                break
            except (ValueError, TypeError):
                continue

        if not moved:
            # If completely stuck, clear target to pick a new one next tick
            opp_targets[obs.id] = None

    st.session_state["_opp_targets"] = opp_targets


def spawn_traffic(world):
    parking = _parking_cells(world)
    for _ in range(20):
        x, y = random.randint(2, 17), random.randint(2, 17)
        if not world.cell_is_free(x, y): continue
        if (x, y) in parking: continue
        if abs(x - world.car.x) < 5 and abs(y - world.car.y) < 5: continue
        dx, dy = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
        speed = random.choice([1, 2])
        try:
            tid = next_obstacle_id("traffic")
            _add_obstacle_compat(world, x, y, kind="traffic_car", obstacle_id=tid,
                                 fallout=1, speed=speed, dx=dx, dy=dy)
            append_editor_obstacle(tid, "traffic_car", (x, y))
            return True
        except (ValueError, TypeError, KeyError): continue
    return False


def update_traffic(world):
    if not st.session_state.get("auto_drive", False): return
    parking = _parking_cells(world)
    st.session_state.traffic_speed_counter = (st.session_state.get("traffic_speed_counter", 0) + 1) % 10
    for obs in world.active_obstacles():
        if obs.kind != "traffic_car": continue
        # FIX #4: clamp speed to >=1 up front.
        speed = max(1, int(getattr(obs, "speed", 1) or 1))
        if st.session_state.traffic_speed_counter % speed != 0: continue

        if (obs.x, obs.y) in parking:
            for d in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = obs.x + d[0], obs.y + d[1]
                if (world.inside_grid(nx, ny) and (nx, ny) not in parking
                        and world.car.position != (nx, ny) and not world.blocked(nx, ny, ignore_obstacle_id=obs.id)):
                    try: world.move_obstacle(obs.id, nx, ny); break
                    except (ValueError, TypeError): continue
            continue

        cur_dx = getattr(obs, "dx", 0) or 0
        cur_dy = getattr(obs, "dy", 0) or 0
        if cur_dx == 0 and cur_dy == 0: continue

        nx, ny = obs.x + cur_dx, obs.y + cur_dy
        if (not world.inside_grid(nx, ny) or (nx, ny) in parking or world.blocked(nx, ny, ignore_obstacle_id=obs.id)
                or world.car.position == (nx, ny)):
            cur_dx, cur_dy = -cur_dx, -cur_dy
            nx, ny = obs.x + cur_dx, obs.y + cur_dy
            try: obs.dx, obs.dy = cur_dx, cur_dy
            except (AttributeError, TypeError): pass
        if (world.inside_grid(nx, ny) and (nx, ny) not in parking
                and not world.blocked(nx, ny, ignore_obstacle_id=obs.id) and world.car.position != (nx, ny)):
            try: world.move_obstacle(obs.id, nx, ny)
            except (ValueError, TypeError):
                try: obs.dx, obs.dy = -cur_dx, -cur_dy
                except (AttributeError, TypeError): pass


def add_dynamic_obstacle(world, event_label="Dynamic event"):
    for _ in range(5):
        position = find_free_grid_cell(world, prefer_route=True)
        if position is None: return False, "No free cell available."
        kind = random.choice(["cone", "barrier"])
        obstacle_id = next_obstacle_id("dynamic_obstacle")
        fallout_radius = random.randint(1, 4)
        try:
            _add_obstacle_compat(world, position[0], position[1], kind=kind, obstacle_id=obstacle_id, fallout=fallout_radius)
        except (ValueError, TypeError, KeyError): continue
        if astar(world, world.car.position, world.car.destination) is None:
            world.remove_obstacle(obstacle_id); continue
        append_editor_obstacle(obstacle_id, kind, position, fallout_radius)
        return True, f"{event_label}: Added {kind} at {position} with fallout {fallout_radius}"
    return False, "Could not find a safe position."


def move_dynamic_obstacle(world):
    obstacles = [o for o in world.active_obstacles() if o.kind in ("cone", "barrier")]
    if not obstacles: return False, "No static obstacle available to move."
    obstacle = random.choice(obstacles)
    old_position = obstacle.position
    new_position = find_free_grid_cell(world, ignore_obstacle_id=obstacle.id, prefer_route=True)
    if new_position is None: return False, "No free position available."
    try:
        world.move_obstacle(obstacle.id, new_position[0], new_position[1])
        if astar(world, world.car.position, world.car.destination) is None:
            world.move_obstacle(obstacle.id, old_position[0], old_position[1]); return False, "Move would seal destination."
        for obj in st.session_state.editor_objects:
            if obj["id"] == obstacle.id: obj["x"], obj["y"] = grid_to_canvas(*new_position); break
        return True, f"Moved {obstacle.id}: {old_position} → {new_position}"
    except (ValueError, TypeError, KeyError) as error: return False, str(error)


def remove_dynamic_obstacle(world):
    obstacles = world.active_obstacles()
    if not obstacles: return False, "No obstacle available to remove."
    agent = st.session_state.drive_agent
    remaining = set(agent.status.path[1:]) if agent is not None else set()
    route_obs = [o for o in obstacles if o.position in remaining]
    obstacle = random.choice(route_obs or obstacles)
    obstacle_id, old_position = obstacle.id, obstacle.position
    try:
        world.remove_obstacle(obstacle_id)
        st.session_state.editor_objects = [obj for obj in st.session_state.editor_objects if obj["id"] != obstacle_id]
        return True, f"Removed {obstacle_id} from {old_position}"
    except (ValueError, TypeError, KeyError) as error: return False, str(error)


def generate_dynamic_event(world):
    obstacles = list(world.active_obstacles())
    actions = ["add", "opposition"] if not obstacles else ["move", "remove"] if len(obstacles) >= 15 else ["add", "add", "move", "move", "remove", "traffic", "opposition"]
    action = random.choice(actions)
    if action == "add": success, message = add_dynamic_obstacle(world)
    elif action == "move": success, message = move_dynamic_obstacle(world)
    elif action == "traffic": success, message = (True, "New traffic car entered the map") if spawn_traffic(world) else (False, "Failed to spawn traffic")
    elif action == "opposition":
        ok, result = spawn_opposition_car(world)
        success, message = (ok, f"Opposition car {result} entered the map" if ok else f"Failed to spawn opposition car: {result}")
    else: success, message = remove_dynamic_obstacle(world)
    if success:
        st.session_state.dynamic_event_count += 1
        st.session_state.dynamic_event_history.append({"Event": st.session_state.dynamic_event_count, "Action": action.capitalize(), "Details": message, "World revision": world.revision})
        st.session_state.dynamic_event_history = st.session_state.dynamic_event_history[-100:]
    return success, message


def populate_initial_obstacles(count=TARGET_OBSTACLE_COUNT, destination_letter="B"):
    # FIX #6: drop pre-existing static obstacles that overlap the car or the
    # destination before building the temp world.
    car_objects = [obj for obj in st.session_state.editor_objects if obj["type"] == "car"]
    if len(car_objects) != 1:
        st.warning("Add exactly one car first.")
        return 0
    car_x, car_y = canvas_to_grid(car_objects[0]["x"], car_objects[0]["y"])
    zone_name, destination = PARKING_ZONES[destination_letter]

    def _valid_static(obj):
        if obj["type"] not in ("cone", "barrier"):
            return True
        pos = canvas_to_grid(obj["x"], obj["y"])
        return pos not in ((car_x, car_y), destination)

    st.session_state.editor_objects = [
        o for o in st.session_state.editor_objects if _valid_static(o)
    ]

    obstacle_objects = [obj for obj in st.session_state.editor_objects if obj["type"] in ("cone", "barrier")]
    required = max(0, count - len(obstacle_objects))
    if required == 0:
        return 0
    temp_world = DrivingWorld(car=Car(x=car_x, y=car_y, destination=destination, direction="east"))
    existing_ids = {obj["id"] for obj in st.session_state.editor_objects}
    for obj in obstacle_objects:
        position = canvas_to_grid(obj["x"], obj["y"])
        try: _add_obstacle_compat(temp_world, position[0], position[1], kind=obj["type"], obstacle_id=obj["id"])
        except (ValueError, TypeError, KeyError): continue
    added = 0
    for _ in range(required):
        position = find_free_grid_cell(temp_world, prefer_route=False)
        if position is None: break
        kind, obstacle_id = random.choice(["cone", "barrier"]), next_obstacle_id("auto_obstacle")
        while obstacle_id in existing_ids: obstacle_id = next_obstacle_id("auto_obstacle")
        try:
            _add_obstacle_compat(temp_world, position[0], position[1], kind=kind, obstacle_id=obstacle_id)
            append_editor_obstacle(obstacle_id, kind, position); existing_ids.add(obstacle_id); added += 1
        except (ValueError, TypeError, KeyError): continue
    return added


def build_world(objects, destination_letter):
    cars = [obj for obj in objects if obj["type"] == "car"]
    if len(cars) != 1: return None, "The editor must contain exactly one car."
    if destination_letter not in PARKING_ZONES: return None, "Invalid parking destination."
    zone_name, destination = PARKING_ZONES[destination_letter]
    car_x, car_y = canvas_to_grid(cars[0]["x"], cars[0]["y"])
    if (car_x, car_y) == destination: return None, "The car is already at its destination."
    world = DrivingWorld(car=Car(x=car_x, y=car_y, destination=destination, direction="east"))
    try:
        world.set_destination(zone_name)
        occupied = set()
        for obj in objects:
            if obj["type"] not in ("cone", "barrier", "traffic_car", "opposition_car"): continue
            x, y = canvas_to_grid(obj["x"], obj["y"])
            if (x, y) == (car_x, car_y): return None, "An obstacle overlaps the car."
            if (x, y) == destination: return None, "An obstacle blocks the destination."
            if (x, y) in occupied: continue
            occupied.add((x, y))
            extra = {}
            if obj["type"] in MOVING_KINDS:
                dx, dy = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
                extra = {"speed": 1, "dx": dx, "dy": dy}
            _add_obstacle_compat(world, x, y, kind=obj["type"], obstacle_id=obj["id"], fallout=obj.get("fallout", 1), **extra)
    except (ValueError, TypeError, KeyError) as error: return None, str(error)
    return world, None


def apply_editor_changes(incoming):
    world, agent, old_objects = st.session_state.drive_world, st.session_state.drive_agent, st.session_state.editor_objects
    if world is None or agent is None:
        if incoming != old_objects:
            st.session_state.editor_objects = incoming
            clear_navigation()
            st.session_state.last_plan_status = "Editor changed. Plan a new route."
        return
    if not st.session_state.auto_drive:
        for obj in incoming:
            if obj.get("type") == "car" and canvas_to_grid(obj["x"], obj["y"]) != world.car.position:
                st.session_state.editor_objects = incoming
                clear_navigation()
                st.session_state.last_plan_status = "Car repositioned. Plan a new route."
                return
    new_by_id = {obj["id"]: obj for obj in incoming}
    backend_obstacles = {o.id: o for o in world.active_obstacles()}
    backend_car_canvas = grid_to_canvas(*world.car.position)
    requested = {}
    for obj in incoming:
        if obj["type"] not in ("cone", "barrier", "traffic_car", "opposition_car"): continue
        position = canvas_to_grid(obj["x"], obj["y"])
        if position in (world.car.destination, world.car.position) or position in requested:
            st.session_state.editor_warning = "Invalid obstacle placement."
            refresh_editor_component(); return
        requested[position] = obj["id"]
    changes = []
    try:
        for obstacle_id in list(backend_obstacles):
            if obstacle_id not in new_by_id:
                world.remove_obstacle(obstacle_id)
                changes.append(f"Removed {obstacle_id}")
        for obj in incoming:
            if obj["type"] not in ("cone", "barrier", "traffic_car", "opposition_car"): continue
            obstacle_id = obj["id"]
            gx, gy = canvas_to_grid(obj["x"], obj["y"])
            if obstacle_id in backend_obstacles:
                if backend_obstacles[obstacle_id].position != (gx, gy):
                    world.move_obstacle(obstacle_id, gx, gy)
                    changes.append(f"Moved {obstacle_id} to ({gx}, {gy})")
            else:
                extra = {}
                if obj["type"] in MOVING_KINDS:
                    dx, dy = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
                    extra = {"speed": 1, "dx": dx, "dy": dy}
                _add_obstacle_compat(world, gx, gy, kind=obj["type"], obstacle_id=obstacle_id, fallout=obj.get("fallout", 1), **extra)
                changes.append(f"Added {obstacle_id} at ({gx}, {gy})")
    except (ValueError, KeyError, TypeError) as error:
        st.session_state.editor_warning = f"Editor update rejected: {error}."
        refresh_editor_component(); return
    normalized = []
    for obj in incoming:
        copy_obj = dict(obj)
        if copy_obj["type"] == "car": copy_obj["x"], copy_obj["y"] = backend_car_canvas
        normalized.append(copy_obj)
    st.session_state.editor_objects = normalized
    if changes: st.session_state.editor_warning = "Environment updated: " + "; ".join(changes)


def _process_pending_plan_request():
    request = st.session_state.pop("_pending_plan_request", None)
    if not request: return
    use_llm, instruction_text, destination_letter = request["use_llm"], request["instruction"], request["destination_letter"]
    try:
        world, error = build_world(st.session_state.editor_objects, destination_letter)
        if error: clear_navigation(); st.session_state.last_plan_status = error; st.error(error); return
        agent = DriveAgent(world)
        def image_provider(_world=world): return render_scene_snapshot(_world, st.session_state.background_data)
        decision = None
        if use_llm:
            ok, status = agent.planner.status()
            if not ok: clear_navigation(); st.error(f"**LLM is not ready.** {status}"); return
            with st.spinner("Consulting local VLM… (~40s on CPU)"):
                try: decision = agent.start(instruction=instruction_text, image_provider=image_provider, use_llm=True)
                except OllamaUnavailable as llm_error: clear_navigation(); st.error(f"**LLM call failed.** {llm_error}"); return
                except Exception as llm_error: clear_navigation(); st.error(f"**LLM error.** {llm_error}"); return
        else:
            with st.spinner("Planning with A*…"): decision = agent.start(instruction=instruction_text, image_provider=image_provider, use_llm=False)
        started = (decision is not None and decision.decision == "navigate" and agent.status.state == "ready")
        if started:
            st.session_state.drive_world, st.session_state.drive_agent = world, agent
            st.session_state.auto_drive, st.session_state.auto_drive_toggle_revision = True, st.session_state.auto_drive_toggle_revision + 1
            st.session_state.planned_route, st.session_state.dynamic_event_count = [list(p) for p in agent.status.path], 0
            st.session_state.dynamic_event_tick_counter, st.session_state.dynamic_event_history, st.session_state.movement_tick_counter, st.session_state.traffic_speed_counter = 0, [], 0, 0
            st.session_state.opposition_car_tick_counter = 0
            _refresh_comparison_overlays(world, agent)
            shown_destination = decision.destination if decision.destination is not None else world.car.destination
            st.session_state.last_plan_status = f"Started ({agent.status.destination_source} destination: {shown_destination}). Reason: {decision.reason}"
            refresh_editor_component()
            st.success(f"✅ Route ready. Auto-driving started.")
        else:
            clear_navigation()
            if decision and decision.decision == "clarify": st.warning(f"⚠️ **LLM rejected instruction.** Reason: {decision.reason}")
            elif decision and decision.decision == "stop": st.error(f"LLM stopped: {decision.reason}")
            elif decision and decision.decision == "wait": st.info(f"LLM chose to wait: {decision.reason}")
            else: st.error("Planning failed.")
    except Exception as exc: clear_navigation(); st.session_state.last_plan_status = f"Planning failed: {exc}"; st.error(f"Planning failed: {exc}")


def _process_pending_step():
    if not st.session_state.pop("_pending_step", False): return
    # FIX #2: signal to main_panel that a step has already been consumed in
    # this app run, so it must not call agent.step() again in the same cycle.
    st.session_state["_step_consumed_this_run"] = True
    agent = st.session_state.drive_agent
    if agent is None or st.session_state.drive_world is None: return
    try:
        if agent.use_llm:
            with st.spinner("Replanning with VLM… (~40s)"): agent.step()
        else: agent.step()
        sync_car_to_editor()
    except Exception as exc: st.error(f"Step failed: {exc}")


def run_llm_vs_astar_benchmark(editor_objects, destination_letter, instruction_text, background_data):
    cars, cones, barriers = [o for o in editor_objects if o["type"] == "car"], [o for o in editor_objects if o["type"] == "cone"], [o for o in editor_objects if o["type"] == "barrier"]
    car_pos = canvas_to_grid(cars[0]["x"], cars[0]["y"]) if len(cars) == 1 else None
    st.info(f"**Benchmark scene:** car @ {car_pos} · {len(cones)} cone(s) · {len(barriers)} barrier(s) · destination = parking_{destination_letter}")
    probe_world, probe_error = build_world(copy.deepcopy(editor_objects), destination_letter)
    if probe_world is not None:
        unreachable = [f"{z}{xy}" for z, xy in probe_world.parking_zones.items() if xy != probe_world.car.position and astar(probe_world, probe_world.car.position, xy) is None]
        if unreachable: st.warning(f"⚠️ **Unreachable parking zones:** {', '.join(unreachable)}.")
    elif probe_error: return None, f"Cannot build world: {probe_error}"
    results, llm_soft_avoid_weights = {}, {}
    for mode in ("LLM", "A*"):
        world, error = build_world(copy.deepcopy(editor_objects), destination_letter)
        if error: return None, f"{mode}: build_world failed — {error}"
        agent = DriveAgent(world)
        def image_provider(_w=world): return render_scene_snapshot(_w, background_data)
        try:
            if mode == "LLM":
                decision = agent.start(instruction=instruction_text, image_provider=image_provider, use_llm=True)
                if agent.current_avoid_obstacles: st.success(f"🧠 LLM extracted **{len(agent.current_avoid_obstacles)}** weighted obstacle(s).")
                else: st.warning("⚠️ **LLM returned `avoid_obstacles: []`.**")
                if agent.soft_avoid_weights: llm_soft_avoid_weights = dict(agent.soft_avoid_weights)
            else: decision = agent.start(instruction=instruction_text, image_provider=image_provider, use_llm=False)
        except Exception as exc: return None, f"{mode} start failed — {exc}"
        if decision is None or decision.decision != "navigate": return None, f"{mode}: planner returned no decision or refused."
        if agent.status.state != "ready": return None, f"{mode}: LLM chose navigate but A* found no route."
        initial_path = list(agent.status.path)

        captured_hazard_crossed = agent.status.hazard_weight_crossed
        captured_hazard_unconstrained = agent.status.hazard_weight_unconstrained
        captured_risk_crossed = agent.status.risk_cells_crossed
        captured_risk_unconstrained = agent.status.risk_cells_unconstrained

        if mode == "A*":
            astar_unconstrained = astar(world, world.car.position, world.car.destination)
            captured_hazard_crossed = sum(llm_soft_avoid_weights.get(p, 0.0) for p in (initial_path or []))
            captured_hazard_unconstrained = sum(llm_soft_avoid_weights.get(p, 0.0) for p in (astar_unconstrained or []))
            captured_risk_crossed = sum(1 for p in (initial_path or []) if p in llm_soft_avoid_weights)
            captured_risk_unconstrained = sum(1 for p in (astar_unconstrained or []) if p in llm_soft_avoid_weights)

        for _ in range(300):
            if agent.status.state not in {"ready", "moving", "replanning", "waiting", "braking", "turning", "blocked", "safety_stop"}: break
            agent.step()

        results[mode] = {
            "final_state": agent.status.state,
            "arrived": agent.status.state == "completed",
            "steps": agent.status.steps,
            "path_length": len(initial_path) if initial_path else 0,
            "hazard_weight_crossed": captured_hazard_crossed,
            "hazard_weight_unconstrained": captured_hazard_unconstrained,
            "risk_cells_crossed": captured_risk_crossed,
            "risk_cells_unconstrained": captured_risk_unconstrained,
            "replans": agent.status.replans,
            "llm_calls": agent.status.llm_calls,
            "safety_stops": agent.status.safety_stops,
            "avoid_obstacles": list(agent.current_avoid_obstacles),
            "soft_avoid_cells": len(agent.soft_avoid_weights),
        }
    return results, None


def compute_effectiveness(llm_res, astar_res):
    llm_w, astar_w = llm_res.get("hazard_weight_crossed", 0.0), astar_res.get("hazard_weight_crossed", 0.0)
    return {"hazard_avoidance_rate": ((astar_w - llm_w) / astar_w * 100) if astar_w > 0.001 else None, "step_cost": llm_res.get("steps", 0) - astar_res.get("steps", 0), "llm_avoid_obstacles": llm_res.get("avoid_obstacles", []), "soft_avoid_cells": llm_res.get("soft_avoid_cells", 0), "hazard_weight_llm": llm_w, "hazard_weight_astar": astar_w}


st.sidebar.title("EdgeMind Setup")
bg_upload = st.sidebar.file_uploader("Background", type=["png", "jpg", "jpeg"])
cam_upload = st.sidebar.camera_input("Webcam")
if bg_upload: st.session_state.background_data = image_to_data_uri(bg_upload)
elif cam_upload: st.session_state.background_data = image_to_data_uri(cam_upload)
st.sidebar.divider(); st.sidebar.subheader("Sprites")
car_up = st.sidebar.file_uploader("Car", type=["png", "jpg", "jpeg"])
cone_up = st.sidebar.file_uploader("Cone", type=["png", "jpg", "jpeg"])
bar_up = st.sidebar.file_uploader("Barrier", type=["png", "jpg", "jpeg"])
opp_up = st.sidebar.file_uploader(
    "Opposition car (optional)", type=["png", "jpg", "jpeg"],
    help="Sprite used for opposition_car obstacles. Leave empty to reuse the main car sprite.",
)
if car_up: st.session_state.car_sprite = image_to_data_uri(car_up, (180, 180))
if cone_up: st.session_state.cone_sprite = image_to_data_uri(cone_up, (180, 180))
if bar_up: st.session_state.barrier_sprite = image_to_data_uri(bar_up, (180, 180))
if opp_up: st.session_state.opposition_car_sprite = image_to_data_uri(opp_up, (180, 180))

st.title("🚗 EdgeMind — Driving Digital Twin")
st.caption("Hierarchical control: **LLM extracts weighted constraints** → **A* plans risk-aware routes** → **runtime safety tier validates every step**")

_process_pending_plan_request()
_process_pending_step()


@st.fragment(run_every=0.5)
def main_panel():
    for _key, _default in (
        ("dynamic_events_enabled", False),
        ("auto_drive", False),
        ("auto_drive_toggle_revision", 0),
        ("movement_tick_counter", 0),
        ("traffic_speed_counter", 0),
        ("opposition_car_tick_counter", 0),
        ("dynamic_event_tick_counter", 0),
        ("dynamic_event_count", 0),
        ("dynamic_event_history", []),
        ("dynamic_event_id_counter", 0),
        ("editor_objects", []),
        ("planned_route", []),
        ("ghost_route", []),
        ("soft_avoid_cells", []),
        ("hazard_cells", []),
        ("drive_world", None),
        ("drive_agent", None),
        ("last_plan_status", "No route planned yet."),
        ("opposition_car_sprite", None),
        ("_opp_targets", {}),
        ("_step_consumed_this_run", False),
    ):
        if _key not in st.session_state:
            st.session_state[_key] = _default

    agent, world = st.session_state.drive_agent, st.session_state.drive_world
    active_states = {"ready", "moving", "replanning", "waiting", "braking", "turning", "blocked", "safety_stop"}

    if agent is not None and world is not None:
        st.subheader("🔴 Live Navigation")
        ctrl1, ctrl2, ctrl3 = st.columns(3)
        with ctrl1:
            auto_drive = st.toggle("Automatic driving", value=st.session_state.auto_drive, key=f"auto_drive_toggle_{st.session_state.auto_drive_toggle_revision}")
            st.session_state.auto_drive = auto_drive
        with ctrl2:
            _dyn_on = st.session_state.get("dynamic_events_enabled", False)
            st.markdown("**Dynamic Environment**\n" + ("🟢 **Enabled**" if _dyn_on else "⚪ **Disabled**"))
        with ctrl3: movement_delay = st.select_slider("Move interval", options=[0.5, 1.0, 1.5, 2.0], value=0.5, format_func=lambda x: f"{x:.1f}s")

        st.markdown("---")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("State", agent.status.state.capitalize())
        m2.metric("Position", str(world.car.position))
        m3.metric("Steps", agent.status.steps)
        m4.metric("Revision", world.revision)

        if agent.status.state == "braking": st.warning(f"🛑 {agent.status.message}")
        elif agent.status.state == "turning": st.info(f"↪️ {agent.status.message}")
        elif agent.status.state in ("blocked", "safety_stop"): st.warning(f"⏳ {agent.status.message}")
        else: st.caption(agent.status.message)

        # ── Ghost vs Active route diagnostic ─────────────────────────────
        # Ghost = unconstrained A*; active = A* with soft_avoid_weights.
        # These coincide whenever the weight field is empty, or the
        # optimal route already bypasses the weighted cells, or the
        # weights are too weak to justify a detour. Surface *which*
        # of these is the case instead of leaving it as a visual mystery.
        _ghost_cells = st.session_state.get('ghost_route', [])
        _active_cells = st.session_state.get('planned_route', [])
        _routes_differ = _ghost_cells != _active_cells
        st.caption(
            f"**Ghost route:** {len(_ghost_cells)} cells · "
            f"**Actual route:** {len(_active_cells)} cells · "
            f"**Routes differ:** {'✅ yes' if _routes_differ else '⬜ no (identical)'} · "
            f"**Soft-avoid cells:** {len(agent.soft_avoid_weights)} · "
            f"**Snapped specs:** {agent.status.soft_avoid_specs_snapped}"
        )

        if (not _routes_differ
                and agent.current_avoid_obstacles
                and not agent.soft_avoid_weights):
            st.info(
                "ℹ️ The LLM returned obstacle specs, but **none of them matched a "
                "live obstacle of the requested kind within the snap radius**, so "
                "no soft-avoid cells were generated — this is why the ghost and "
                "active routes are identical. Check the *World change history* "
                "table below for `Dropped LLM spec` log entries to see which "
                "coordinates were rejected."
            )
        elif (not _routes_differ
                and agent.soft_avoid_weights
                and agent.status.hazard_weight_crossed < 0.01):
            st.caption(
                "ℹ️ Ghost and active routes coincide because the shortest route "
                "already avoids all weighted cells — no detour is needed to "
                "satisfy the constraint."
            )
        elif (not _routes_differ
                and agent.soft_avoid_weights
                and agent.status.hazard_weight_crossed > 0.01
                and abs(agent.status.hazard_weight_crossed - agent.status.hazard_weight_unconstrained) < 0.01):
            st.caption(
                "ℹ️ Ghost and active routes coincide because the weights are too "
                "low to justify a detour — A* still threads through the halo. "
                "Raise `safety_vs_time` or strengthen the instruction (e.g. "
                "*'strictly avoid'*) to force a divergence."
            )
        # ─────────────────────────────────────────────────────────────────

        source_map = {"llm": "🧠 LLM", "dropdown": "⚙️ Dropdown", "none": "❌ None"}
        st.caption(f"**Destination:** {world.car.destination} — **source:** {source_map.get(agent.status.destination_source, 'unknown')}")

        if agent.status.state not in active_states: stop_automatic_driving()
        if agent.status.state == "completed": st.success("✅ Destination reached!")
        elif agent.status.state in ("blocked", "safety_stop"): st.warning(agent.status.message)
        elif agent.status.state == "paused": st.warning(agent.status.message)
        elif agent.status.state == "waiting": st.info(f"⏸️ {agent.status.message}")
        elif agent.status.state == "clarify": st.warning(f"❓ {agent.status.message}")

        interval_ticks = max(1, round(movement_delay / 0.5))
        st.session_state.movement_tick_counter += 1

        if st.session_state.auto_drive and agent.status.state in active_states:
            update_traffic(world)
            update_opposition_cars(world)

        if st.session_state.get("dynamic_events_enabled", False) and st.session_state.auto_drive and agent.status.state in active_states:
            st.session_state.dynamic_event_tick_counter += 1
            if st.session_state.dynamic_event_tick_counter >= max(1, round(st.session_state.get("dynamic_event_interval", 3.0) / 0.5)):
                st.session_state.dynamic_event_tick_counter = 0
                success, message = generate_dynamic_event(world)
                if success: st.toast(f"🌐 {message}", icon="🚧")

        should_move = st.session_state.auto_drive and agent.status.state in active_states and (st.session_state.movement_tick_counter % interval_ticks == 0)

        if should_move:
            # FIX #2: _process_pending_step may have already stepped the agent
            # during this app run. If so, do not step again.
            if st.session_state.pop("_step_consumed_this_run", False):
                pass
            elif agent.use_llm and agent.status.state not in ("braking", "blocked", "safety_stop") and agent.needs_llm_replan():
                st.session_state._pending_step = True
                st.rerun(scope="app")
            else:
                agent.step()
                sync_car_to_editor()

        st.markdown("### 🎯 LLM vs A* — what the LLM changes, live")
        if agent.current_avoid_obstacles and agent.status.hazard_weight_unconstrained > 0:
            llm_steps, astar_steps = agent.status.path_length_constrained, agent.status.path_length_unconstrained
            llm_haz_w, astar_haz_w = agent.status.hazard_weight_crossed, agent.status.hazard_weight_unconstrained
            weight_avoided = astar_haz_w - llm_haz_w
            st.success(f"**The LLM's weight field removes {weight_avoided:.1f} units of hazard cost** at a cost of {llm_steps - astar_steps:+d} extra step(s).")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("LLM route", f"{llm_steps} steps", delta=f"{llm_steps - astar_steps:+d}", delta_color="inverse")
            c2.metric("A* ghost route", f"{astar_steps} steps")
            c3.metric("Hazard cost — LLM", f"{llm_haz_w:.1f}", delta=f"{-weight_avoided:+.1f}", delta_color="normal" if weight_avoided > 0 else "off")
            c4.metric("Hazard cost — A* alone", f"{astar_haz_w:.1f}")
        elif not agent.current_avoid_obstacles: st.warning("⚠️ **The LLM extracted no avoidance constraints.**")
        else: st.caption("No soft-avoid cells are on the current route.")

        st.markdown("#### 🧠 LLM-Extracted Mission Constraints")
        if agent.current_avoid_obstacles:
            st.success(f"Extracted **{len(agent.current_avoid_obstacles)}** weight(s) · `safety_vs_time = {agent.current_safety_vs_time:.2f}` · **{len(agent.soft_avoid_weights)}** soft-avoid cells")
            st.dataframe([{"Position": f"({spec['position'][0]}, {spec['position'][1]})", "Kind": spec["kind"], "Weight": round(spec["weight"], 2), "Reason": spec["reason"][:60]} for spec in agent.current_avoid_obstacles], use_container_width=True, hide_index=True)
        else: st.warning("⚠️ **No avoidance constraints extracted.**")
        st.divider()

    st.subheader("🗺️ Interactive World Editor")
    component_value = editor_component(
        background=st.session_state.background_data,
        sprites={
            "car": st.session_state.car_sprite,
            "cone": st.session_state.cone_sprite,
            "barrier": st.session_state.barrier_sprite,
            # Traffic and opposition cars use car sprites, not cone sprites.
            "traffic_car": st.session_state.car_sprite,
            "opposition_car": (
                st.session_state.opposition_car_sprite
                or st.session_state.car_sprite
            ),
        },
        objects=st.session_state.editor_objects,
        route=st.session_state.planned_route,
        ghostRoute=st.session_state.ghost_route,
        softAvoidCells=st.session_state.soft_avoid_cells,
        hazardCells=st.session_state.hazard_cells,
        carState=agent.status.state if agent else "idle",
        default=st.session_state.editor_objects,
        key=f"edgemind_editor_{st.session_state.editor_revision}",
        default_height=700,
    )
    incoming = clean_editor_objects(component_value)
    if st.session_state.demo_lock_countdown > 0:
        st.session_state.editor_objects = [dict(o) for o in st.session_state.demo_lock]
        st.session_state.demo_lock_countdown -= 1
        if st.session_state.demo_lock_countdown == 0: st.session_state.demo_lock = None
        st.session_state.last_processed_component_key = None
        refresh_editor_component(); st.rerun()

    if incoming is not None:
        incoming_key = json.dumps(incoming, sort_keys=True)
        if incoming_key != st.session_state.last_processed_component_key:
            st.session_state.last_processed_component_key = incoming_key
            if not objects_equal_by_grid(incoming, st.session_state.editor_objects):
                apply_editor_changes(incoming); st.rerun()

    if st.session_state.editor_warning: st.warning(st.session_state.editor_warning); st.session_state.editor_warning = None
    st.divider()
    st.markdown("#### 🧭 Mission Setup")
    mode_col1, mode_col2 = st.columns([3, 1])
    with mode_col1: instruction_text = st.text_input("Driving instruction (natural language)", value=st.session_state.get("instruction_text_input", "Drive to parking B, stay far away from cones"), key="instruction_text_input")
    with mode_col2: use_llm = st.toggle("Use LLM", value=True, key="use_llm_toggle")
    if use_llm: st.success("🧠 **LLM mode** — the LLM converts your instruction into a continuous per-obstacle cost field.")
    else: st.warning("⚙️ **A* only mode** — the text instruction is ignored.")

    dest_col, dyn_col = st.columns([2, 1])
    with dest_col: destination_letter = st.selectbox("Fallback parking destination", ["A", "B", "C"], format_func=lambda x: f"Parking Zone {x}", key="destination_select")
    with dyn_col: st.toggle("Dynamic obstacles", key="dynamic_events_enabled")

    population_col, plan_col, clear_col = st.columns(3)
    with population_col:
        if st.button("🚧 Auto-populate 5 obstacles", use_container_width=True, disabled=(st.session_state.drive_agent is not None)):
            added = populate_initial_obstacles(TARGET_OBSTACLE_COUNT, destination_letter)
            if added: refresh_editor_component(); st.toast(f"Added {added} obstacles.", icon="🚧")
            st.rerun()
    with plan_col:
        if st.button("🧭 Plan route and start driving", use_container_width=True):
            st.session_state._pending_plan_request = {"use_llm": use_llm, "instruction": instruction_text, "destination_letter": destination_letter}
            st.rerun(scope="app")
    with clear_col:
        if st.button("Clear navigation", use_container_width=True):
            clear_navigation(); st.session_state.last_plan_status = "Navigation cleared."; refresh_editor_component(); st.rerun()

    if agent is not None and world is not None:
        manual_col, pause_col, spawn_col = st.columns(3)
        with manual_col:
            if st.button("Execute one step", use_container_width=True, disabled=(agent.status.state not in active_states or st.session_state.auto_drive)):
                if agent.use_llm and agent.status.state not in ("braking", "blocked", "safety_stop") and agent.needs_llm_replan():
                    st.session_state._pending_step = True; st.rerun(scope="app")
                else: agent.step(); sync_car_to_editor(); st.rerun()
        with pause_col:
            if st.button("Pause / Resume", use_container_width=True, disabled=agent.status.state not in active_states):
                st.session_state.auto_drive = not st.session_state.auto_drive
                st.session_state.auto_drive_toggle_revision += 1
                st.rerun()
        with spawn_col:
            if st.button("🚗 Spawn opposition car", use_container_width=True):
                ok, result = spawn_opposition_car(world)
                if ok:
                    st.toast(f"Opposition car {result} spawned!", icon="🚗")
                    sync_car_to_editor()
                    st.rerun()
                else:
                    st.warning(f"Could not spawn opposition car: {result}")

        st.divider(); st.subheader("Dynamic Environment Event History")
        if st.session_state.dynamic_event_history: st.dataframe(st.session_state.dynamic_event_history, use_container_width=True, hide_index=True)
        else: st.caption("No dynamic events yet.")
        st.subheader("World Change History")
        if world.change_history: st.dataframe(world.change_history, use_container_width=True, hide_index=True)
        else: st.caption("No world changes yet.")

main_panel()

st.divider()
st.subheader("📊 LLM vs A* Benchmark — Accuracy & Effectiveness")
st.caption("Runs the same map twice (LLM ON, then LLM OFF) with dynamic events disabled.")

with st.expander("▶ Run benchmark", expanded=st.session_state.benchmark_result is None):
    st.warning("⚠️ **Important for a clean demo:** The **🎯 Load Demo Scenario** button places the car and **two cones** so that A* alone threads through their halos while the LLM detours around them.")
    bench_col1, bench_col2, bench_col3 = st.columns([1, 1, 1])
    with bench_col1: st.button("🎯 Load Demo Scenario", use_container_width=True, key="load_demo_btn", on_click=load_demo_scenario)
    with bench_col2:
        if st.button("🚀 Run LLM vs A* benchmark", type="primary", use_container_width=True, key="run_benchmark_button"):
            objs = st.session_state.editor_objects
            cars, cones, barriers = [o for o in objs if o["type"] == "car"], [o for o in objs if o["type"] == "cone"], [o for o in objs if o["type"] == "barrier"]
            if len(objs) != 3 or len(cars) != 1 or len(cones) != 2 or barriers: st.warning("This is not the demo scene. Click **🎯 Load Demo Scenario** first.")
            with st.spinner("Running both modes to completion…"):
                results, error = run_llm_vs_astar_benchmark(objs, st.session_state.get("destination_select", "B"), st.session_state.get("instruction_text_input", "Drive to parking B, stay far away from cones."), st.session_state.background_data)
            if error: st.error(f"**Benchmark failed.** {error}"); st.session_state.benchmark_result = None
            else: st.session_state.benchmark_result = results
    with bench_col3:
        if st.session_state.benchmark_result is not None:
            if st.button("🗑️ Clear benchmark result", use_container_width=True, key="clear_benchmark_button"):
                st.session_state.benchmark_result = None; st.rerun()

if st.session_state.benchmark_result is not None:
    llm_res, astar_res = st.session_state.benchmark_result["LLM"], st.session_state.benchmark_result["A*"]
    eff = compute_effectiveness(llm_res, astar_res)
    st.markdown("##### Side-by-side comparison")
    st.dataframe({
        "Metric": ["Arrived at destination", "Final state", "Steps taken", "Initial path length", "Hazard cost crossed (weighted)", "Hazard cost A* alone would cross", "Hazard cell count crossed", "Hazard cell count A* alone", "Replans", "LLM calls", "Safety stops", "Weighted obstacles extracted", "Soft-avoid cells"],
        "LLM ON": ["✅" if llm_res["arrived"] else "❌", llm_res["final_state"], llm_res["steps"], llm_res["path_length"], f"{llm_res['hazard_weight_crossed']:.2f}", f"{llm_res['hazard_weight_unconstrained']:.2f}", llm_res["risk_cells_crossed"], llm_res["risk_cells_unconstrained"], llm_res["replans"], llm_res["llm_calls"], llm_res["safety_stops"], len(llm_res["avoid_obstacles"]), llm_res["soft_avoid_cells"]],
        "A* only": ["✅" if astar_res["arrived"] else "❌", astar_res["final_state"], astar_res["steps"], astar_res["path_length"], f"{astar_res['hazard_weight_crossed']:.2f}", f"{astar_res['hazard_weight_unconstrained']:.2f}", astar_res["risk_cells_crossed"], astar_res["risk_cells_unconstrained"], astar_res["replans"], astar_res["llm_calls"], astar_res["safety_stops"], len(astar_res["avoid_obstacles"]), astar_res["soft_avoid_cells"]]
    }, use_container_width=True, hide_index=True)
    st.markdown("##### LLM effectiveness score")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Hazard Avoidance Rate", "N/A" if eff["hazard_avoidance_rate"] is None else f"{eff['hazard_avoidance_rate']:.0f}%")
    s2.metric("Step cost of safety", f"{eff['step_cost']:+d}")
    s3.metric("Weighted obstacles extracted", len(eff["llm_avoid_obstacles"]))
    s4.metric("Soft-avoid cells", eff["soft_avoid_cells"])
    if eff["soft_avoid_cells"] == 0: st.warning("No obstacles of the requested kind were on the map.")
    elif eff["hazard_avoidance_rate"] is None: st.info("**Both routes are hazard-free on this map.** Load the **🎯 Load Demo Scenario** to measure the LLM's effect.")
    elif eff["hazard_avoidance_rate"] > 0 and llm_res["arrived"]:
        direction = "extra" if eff["step_cost"] >= 0 else "fewer"
        st.success(f"**LLM contribution quantified.** {eff['hazard_avoidance_rate']:.0f}% fewer weighted hazard cost ({eff['hazard_weight_astar']:.1f} → {eff['hazard_weight_llm']:.1f}) at a cost of {abs(eff['step_cost'])} {direction} step(s).")
    else: st.info("Both routes similar on this map.")

st.divider()
st.subheader("World Summary")
objects = st.session_state.editor_objects
counts = {
    "Cars": sum(obj["type"] == "car" for obj in objects),
    "Cones": sum(obj["type"] == "cone" for obj in objects),
    "Barriers": sum(obj["type"] == "barrier" for obj in objects),
    "Traffic": sum(obj["type"] == "traffic_car" for obj in objects),
    "Opposition": sum(obj["type"] == "opposition_car" for obj in objects),
}
a, b, c, d, e = st.columns(5)
a.metric("Cars", counts["Cars"]); b.metric("Cones", counts["Cones"]); c.metric("Barriers", counts["Barriers"]); d.metric("Traffic", counts["Traffic"]); e.metric("Opposition", counts["Opposition"])
if objects:
    st.dataframe([{"ID": obj["id"], "Type": obj["type"], "Canvas X": round(obj["x"], 1), "Canvas Y": round(obj["y"], 1), "Grid X": canvas_to_grid(obj["x"], obj["y"])[0], "Grid Y": canvas_to_grid(obj["x"], obj["y"])[1]} for obj in objects], use_container_width=True, hide_index=True)

st.write("**Status:**", st.session_state.last_plan_status)
if st.button("Reset entire editor"):
    st.session_state.editor_objects = []; st.session_state.planned_route = []; st.session_state.ghost_route = []; st.session_state.soft_avoid_cells = []; st.session_state.hazard_cells = []
    st.session_state.last_plan_status = "No route planned yet."; clear_navigation()
    st.session_state.editor_revision += 1; st.session_state.dynamic_event_id_counter = 0; st.session_state.auto_drive_toggle_revision += 1
    st.session_state.last_processed_component_key = None; st.session_state.benchmark_result = None; st.session_state.demo_lock = None; st.session_state.demo_lock_countdown = 0
    st.session_state._pending_plan_request = None; st.session_state._pending_step = False; st.session_state._step_consumed_this_run = False; st.rerun()

with st.expander("📖 Defensible claims (open before defense)"):
    st.markdown("""
**What this project proves:**
1. **Deterministic instruction gate.** A word-list filter runs **before** any LLM call. Instructions containing manipulation verbs with no driving vocabulary are refused.
2. **No silent fallback.** If you toggle LLM on but Ollama fails, the app refuses to start navigation and shows an error.
3. **The LLM outputs a scene-conditioned, per-obstacle continuous cost field.** A regex produces a **single** cost field; the LLM produces distinct fields across paraphrases.
4. **Safety is guaranteed regardless of LLM judgment.** `drive_safety.validate_path` is the single canonical validator.
5. **Closed loop.** `LLM calls = 1 + Replans`. Terminal states recover automatically.
6. **Graceful degradation** — but explicit, not silent.
7. **Moving obstacles are handled locally.** Opposition cars and traffic cars trigger local A* replans, not LLM calls. The static-obstacle snapshot in `DriveAgent.needs_llm_replan` is what guarantees the LLM is only re-consulted on *static* environment changes, never on opposition movement.
8. **Rear-end protection.** The agent detects opposition tailing behind it and skips hazard brakes to keep moving out of the way.
9. **Transient blocks self-recover.** `"blocked"` and `"safety_stop"` are recoverable states kept in `active_states`, so a moving obstacle that temporarily occludes the destination clears the failure automatically. Only *static* blocks require operator intervention.
10. **Moving obstacles never occupy the destination.** All spawn and move paths skip parking-zone cells, and any obstacle already on a parking zone is forcibly relocated.
11. **Constraint field survives replans.** `REPLAN_PROMPT` deliberately omits `avoid_obstacles` and `safety_vs_time`; `DriveDecision` uses `None` (not `[]`/`0.5`) as the "field absent" sentinel, and `_apply_navigate_decision` leaves the original instruction's weighted constraint field untouched when those fields are `None`.
12. **Coordinate snap recovers VLM drift.** `_refresh_soft_avoid_weights` snaps each LLM obstacle spec to the nearest live obstacle of the same kind within `SNAP_RADIUS` (Manhattan), logging each snap. Without this, a 1-cell hallucination would silently collapse the entire weight field to `{}` and make ghost == active with no explanation.
""")
"""Ollama-based constraint extractor for EdgeMind driving."""