## drive_app.py

from __future__ import annotations

import base64
import copy
import io
import json
import random
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw

from vlmplan.drive_world import Car, DrivingWorld, astar
from drive_agent import DriveAgent
from drive_planner import OllamaUnavailable


# =========================================================
# CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="EdgeMind - Driving Digital Twin",
    page_icon="🚗",
    layout="wide",
)

ROOT_DIR = Path(__file__).resolve().parent
EDITOR_DIR = ROOT_DIR / "edgemind_editor"

WIDTH = 900
HEIGHT = 600
GRID = 20

CELL_W = WIDTH / GRID
CELL_H = HEIGHT / GRID

TARGET_OBSTACLE_COUNT = 5
DYNAMIC_EVENT_INTERVAL = 3.0

PARKING_ZONES = {
    "A": ("parking_A", (18, 3)),
    "B": ("parking_B", (18, 10)),
    "C": ("parking_C", (18, 17)),
}

editor_component = components.declare_component(
    "edgemind_editor",
    path=str(EDITOR_DIR),
)


# =========================================================
# SESSION STATE
# =========================================================

DEFAULTS = {
    "editor_objects": [],
    "planned_route": [],
    "last_plan_status": "No route planned yet.",
    "background_data": None,
    "car_sprite": None,
    "cone_sprite": None,
    "barrier_sprite": None,
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
    "last_processed_component_key": None,
    "benchmark_result": None,
    "demo_lock": None,
    "demo_lock_countdown": 0,
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

st.session_state.pop("auto_drive_toggle", None)


# =========================================================
# HELPERS
# =========================================================

def image_to_data_uri(uploaded_file, max_size=(900, 600)):
    if uploaded_file is None:
        return None
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
    gx = max(0, min(GRID - 1, int(float(x) / CELL_W)))
    gy = max(0, min(GRID - 1, int(float(y) / CELL_H)))
    return gx, gy


def grid_to_canvas(x, y):
    return ((x + 0.5) * CELL_W, (y + 0.5) * CELL_H)


def render_scene_snapshot(world, background_data_uri=None):
    base = Image.new("RGB", (WIDTH, HEIGHT), "#e5e7eb")
    if background_data_uri:
        try:
            encoded = background_data_uri.split(",", 1)[-1]
            bg = Image.open(
                io.BytesIO(base64.b64decode(encoded))
            ).convert("RGB").resize((WIDTH, HEIGHT))
            base = bg
        except Exception:
            pass
    draw = ImageDraw.Draw(base)
    for i in range(GRID + 1):
        x, y = i * CELL_W, i * CELL_H
        draw.line([(x, 0), (x, HEIGHT)], fill=(120, 130, 145))
        draw.line([(0, y), (WIDTH, y)], fill=(120, 130, 145))
    zone_colors = {
        "parking_A": (22, 163, 74),
        "parking_B": (37, 99, 235),
        "parking_C": (147, 51, 234),
    }
    for name, (gx, gy) in world.parking_zones.items():
        px, py = grid_to_canvas(gx, gy)
        color = zone_colors.get(name, (100, 100, 100))
        draw.rectangle(
            [px - CELL_W / 2, py - CELL_H / 2,
             px + CELL_W / 2, py + CELL_H / 2],
            outline=color, width=3,
        )
    for obs in world.active_obstacles():
        ox, oy = grid_to_canvas(obs.x, obs.y)
        color = (249, 115, 22) if obs.kind == "cone" else (220, 38, 38)
        draw.ellipse([ox - 15, oy - 15, ox + 15, oy + 15], fill=color)
    cx, cy = grid_to_canvas(*world.car.position)
    draw.rectangle([cx - 20, cy - 20, cx + 20, cy + 20], fill=(37, 99, 235))
    draw.text((cx - 10, cy - 6), "CAR", fill=(255, 255, 255))
    dx, dy = grid_to_canvas(*world.car.destination)
    draw.rectangle(
        [dx - 22, dy - 22, dx + 22, dy + 22],
        outline=(22, 163, 74), width=4,
    )
    draw.text((dx - 12, dy - 6), "DST", fill=(0, 0, 0))
    buf = io.BytesIO()
    base.save(buf, format="PNG", optimize=True)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def clean_editor_objects(value):
    if not isinstance(value, list):
        return None
    cleaned = []
    for obj in value:
        if not isinstance(obj, dict):
            continue
        if obj.get("type") not in ("car", "cone", "barrier"):
            continue
        try:
            cleaned.append({
                "id": str(obj["id"]),
                "type": obj["type"],
                "x": max(0.0, min(WIDTH, float(obj["x"]))),
                "y": max(0.0, min(HEIGHT, float(obj["y"]))),
                "size": float(obj.get("size", 55)),
            })
        except (KeyError, ValueError, TypeError):
            continue
    return cleaned


def refresh_editor_component():
    st.session_state.editor_revision += 1


def stop_automatic_driving():
    if st.session_state.auto_drive:
        st.session_state.auto_drive = False
        st.session_state.auto_drive_toggle_revision += 1


def clear_navigation():
    st.session_state.drive_world = None
    st.session_state.drive_agent = None
    st.session_state.planned_route = []
    stop_automatic_driving()
    st.session_state.dynamic_event_tick_counter = 0
    st.session_state.dynamic_event_count = 0
    st.session_state.dynamic_event_history = []
    st.session_state.movement_tick_counter = 0


def sync_car_to_editor():
    world = st.session_state.drive_world
    agent = st.session_state.drive_agent
    if world is None or agent is None:
        return
    car_x, car_y = grid_to_canvas(*world.car.position)
    updated = []
    for obj in st.session_state.editor_objects:
        copy_obj = dict(obj)
        if copy_obj["type"] == "car":
            copy_obj["x"] = car_x
            copy_obj["y"] = car_y
        updated.append(copy_obj)
    st.session_state.editor_objects = updated
    st.session_state.planned_route = [
        list(p) for p in agent.status.path
    ]
    st.session_state.last_plan_status = (
        f"{agent.status.state.capitalize()}: {agent.status.message}"
    )


# =========================================================
# DEMO SCENARIO
# =========================================================

def load_demo_scenario():
    """Load the benchmark demo scene: car @ (2,10), cone @ (10,10)."""
    demo_objects = []

    cx, cy = grid_to_canvas(2, 10)
    demo_objects.append({
        "id": "demo_car", "type": "car",
        "x": cx, "y": cy, "size": 55,
    })

    ox, oy = grid_to_canvas(10, 10)
    demo_objects.append({
        "id": "demo_cone_0", "type": "cone",
        "x": ox, "y": oy, "size": 55,
    })

    st.session_state.editor_objects = demo_objects
    st.session_state.demo_lock = [dict(o) for o in demo_objects]
    st.session_state.demo_lock_countdown = 6

    clear_navigation()
    st.session_state.last_processed_component_key = None
    st.session_state.destination_select = "B"
    st.session_state.instruction_text_input = (
        "Drive to parking B, stay away from cones"
    )
    st.session_state.benchmark_result = None
    refresh_editor_component()


# =========================================================
# DYNAMIC OBSTACLES
# =========================================================

def find_free_grid_cell(world, ignore_obstacle_id=None, prefer_route=True):
    """Return a random free cell that is safe to place an obstacle on."""
    protected = set()
    for zone_xy in world.parking_zones.values():
        protected.add(zone_xy)
        zx, zy = zone_xy
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            protected.add((zx + dx, zy + dy))

    agent = st.session_state.drive_agent

    if prefer_route and agent is not None:
        candidates = []
        for position in agent.status.path[2:]:
            x, y = position
            if position in protected:
                continue
            if world.cell_is_free(x, y, ignore_obstacle_id=ignore_obstacle_id):
                candidates.append(position)
        if candidates:
            return random.choice(candidates)

    free_cells = []
    for y in range(GRID):
        for x in range(GRID):
            position = (x, y)
            if position in protected:
                continue
            if (
                ignore_obstacle_id is not None
                and position == world.obstacles[ignore_obstacle_id].position
            ):
                continue
            if world.cell_is_free(x, y, ignore_obstacle_id=ignore_obstacle_id):
                free_cells.append(position)
    return random.choice(free_cells) if free_cells else None


def next_obstacle_id(prefix):
    st.session_state.dynamic_event_id_counter += 1
    return f"{prefix}_{st.session_state.dynamic_event_id_counter}"


def append_editor_obstacle(obstacle_id, kind, position):
    canvas_x, canvas_y = grid_to_canvas(*position)
    st.session_state.editor_objects.append({
        "id": obstacle_id, "type": kind,
        "x": canvas_x, "y": canvas_y, "size": 55,
    })


def add_dynamic_obstacle(world, event_label="Dynamic event"):
    position = find_free_grid_cell(world, prefer_route=True)
    if position is None:
        return False, "No free cell available."
    kind = random.choice(["cone", "barrier"])
    obstacle_id = next_obstacle_id("dynamic_obstacle")
    try:
        world.add_obstacle(
            position[0], position[1],
            kind=kind, obstacle_id=obstacle_id,
        )
        append_editor_obstacle(obstacle_id, kind, position)
        return True, f"{event_label}: Added {kind} at {position}"
    except (ValueError, TypeError, KeyError) as error:
        return False, str(error)


def move_dynamic_obstacle(world):
    obstacles = list(world.active_obstacles())
    if not obstacles:
        return False, "No obstacle available to move."
    obstacle = random.choice(obstacles)
    new_position = find_free_grid_cell(
        world, ignore_obstacle_id=obstacle.id, prefer_route=True,
    )
    if new_position is None:
        return False, "No free position available."
    old_position = obstacle.position
    try:
        world.move_obstacle(obstacle.id, new_position[0], new_position[1])
        for obj in st.session_state.editor_objects:
            if obj["id"] == obstacle.id:
                obj["x"], obj["y"] = grid_to_canvas(*new_position)
                break
        return True, f"Moved {obstacle.id}: {old_position} → {new_position}"
    except (ValueError, TypeError, KeyError) as error:
        return False, str(error)


def remove_dynamic_obstacle(world):
    obstacles = list(world.active_obstacles())
    if not obstacles:
        return False, "No obstacle available to remove."
    agent = st.session_state.drive_agent
    remaining = (
        set(agent.status.path[1:]) if agent is not None else set()
    )
    route_obs = [o for o in obstacles if o.position in remaining]
    obstacle = random.choice(route_obs or obstacles)
    obstacle_id = obstacle.id
    old_position = obstacle.position
    try:
        world.remove_obstacle(obstacle_id)
        st.session_state.editor_objects = [
            obj for obj in st.session_state.editor_objects
            if obj["id"] != obstacle_id
        ]
        return True, f"Removed {obstacle_id} from {old_position}"
    except (ValueError, TypeError, KeyError) as error:
        return False, str(error)


def generate_dynamic_event(world):
    obstacles = list(world.active_obstacles())
    if not obstacles:
        actions = ["add"]
    elif len(obstacles) >= 15:
        actions = ["move", "remove"]
    else:
        actions = ["add", "add", "move", "move", "remove"]
    action = random.choice(actions)
    if action == "add":
        success, message = add_dynamic_obstacle(world)
    elif action == "move":
        success, message = move_dynamic_obstacle(world)
    else:
        success, message = remove_dynamic_obstacle(world)
    if success:
        st.session_state.dynamic_event_count += 1
        st.session_state.dynamic_event_history.append({
            "Event": st.session_state.dynamic_event_count,
            "Action": action.capitalize(),
            "Details": message,
            "World revision": world.revision,
        })
        st.session_state.dynamic_event_history = (
            st.session_state.dynamic_event_history[-100:]
        )
    return success, message


def populate_initial_obstacles(count=TARGET_OBSTACLE_COUNT, destination_letter="B"):
    obstacle_objects = [
        obj for obj in st.session_state.editor_objects
        if obj["type"] in ("cone", "barrier")
    ]
    required = max(0, count - len(obstacle_objects))
    if required == 0:
        return 0
    car_objects = [
        obj for obj in st.session_state.editor_objects
        if obj["type"] == "car"
    ]
    if len(car_objects) != 1:
        st.warning("Add exactly one car first.")
        return 0
    car_x, car_y = canvas_to_grid(car_objects[0]["x"], car_objects[0]["y"])
    zone_name, destination = PARKING_ZONES[destination_letter]
    temp_world = DrivingWorld(
        car=Car(x=car_x, y=car_y, destination=destination, direction="east")
    )
    existing_ids = {obj["id"] for obj in st.session_state.editor_objects}
    for obj in obstacle_objects:
        position = canvas_to_grid(obj["x"], obj["y"])
        if position in ((car_x, car_y), destination):
            continue
        try:
            temp_world.add_obstacle(
                position[0], position[1],
                kind=obj["type"], obstacle_id=obj["id"],
            )
        except (ValueError, TypeError, KeyError):
            continue
    added = 0
    for _ in range(required):
        position = find_free_grid_cell(temp_world, prefer_route=False)
        if position is None:
            break
        kind = random.choice(["cone", "barrier"])
        obstacle_id = next_obstacle_id("auto_obstacle")
        while obstacle_id in existing_ids:
            obstacle_id = next_obstacle_id("auto_obstacle")
        try:
            temp_world.add_obstacle(
                position[0], position[1],
                kind=kind, obstacle_id=obstacle_id,
            )
            append_editor_obstacle(obstacle_id, kind, position)
            existing_ids.add(obstacle_id)
            added += 1
        except (ValueError, TypeError, KeyError):
            continue
    return added


def build_world(objects, destination_letter):
    cars = [obj for obj in objects if obj["type"] == "car"]
    if len(cars) != 1:
        return None, "The editor must contain exactly one car."
    if destination_letter not in PARKING_ZONES:
        return None, "Invalid parking destination."
    zone_name, destination = PARKING_ZONES[destination_letter]
    car_x, car_y = canvas_to_grid(cars[0]["x"], cars[0]["y"])
    if (car_x, car_y) == destination:
        return None, "The car is already at its destination."
    world = DrivingWorld(
        car=Car(x=car_x, y=car_y, destination=destination, direction="east")
    )
    try:
        world.set_destination(zone_name)
        occupied = set()
        for obj in objects:
            if obj["type"] not in ("cone", "barrier"):
                continue
            x, y = canvas_to_grid(obj["x"], obj["y"])
            if (x, y) == (car_x, car_y):
                return None, "An obstacle overlaps the car."
            if (x, y) == destination:
                return None, "An obstacle blocks the destination."
            if (x, y) in occupied:
                continue
            occupied.add((x, y))
            world.add_obstacle(
                x, y, kind=obj["type"], obstacle_id=obj["id"],
            )
    except (ValueError, TypeError, KeyError) as error:
        return None, str(error)
    return world, None


def apply_editor_changes(incoming):
    world = st.session_state.drive_world
    agent = st.session_state.drive_agent
    old_objects = st.session_state.editor_objects

    if world is None or agent is None:
        if incoming != old_objects:
            st.session_state.editor_objects = incoming
            clear_navigation()
            st.session_state.last_plan_status = (
                "Editor changed. Plan a new route."
            )
        return

    if not st.session_state.auto_drive:
        for obj in incoming:
            if obj.get("type") != "car":
                continue
            new_grid = canvas_to_grid(obj["x"], obj["y"])
            if new_grid != world.car.position:
                st.session_state.editor_objects = incoming
                clear_navigation()
                st.session_state.last_plan_status = (
                    "Car repositioned. Plan a new route."
                )
                return

    new_by_id = {obj["id"]: obj for obj in incoming}
    backend_obstacles = {o.id: o for o in world.active_obstacles()}
    backend_car_canvas = grid_to_canvas(*world.car.position)

    requested = {}
    for obj in incoming:
        if obj["type"] not in ("cone", "barrier"):
            continue
        position = canvas_to_grid(obj["x"], obj["y"])
        if position == world.car.destination:
            st.session_state.editor_warning = (
                "An obstacle cannot occupy the destination."
            )
            refresh_editor_component()
            return
        if position == world.car.position:
            st.session_state.editor_warning = (
                "An obstacle cannot occupy the car."
            )
            refresh_editor_component()
            return
        if position in requested:
            st.session_state.editor_warning = (
                "Two obstacles cannot occupy the same grid cell."
            )
            refresh_editor_component()
            return
        requested[position] = obj["id"]

    changes = []
    try:
        for obstacle_id in list(backend_obstacles):
            if obstacle_id not in new_by_id:
                world.remove_obstacle(obstacle_id)
                changes.append(f"Removed {obstacle_id}")
        for obj in incoming:
            if obj["type"] not in ("cone", "barrier"):
                continue
            obstacle_id = obj["id"]
            gx, gy = canvas_to_grid(obj["x"], obj["y"])
            if obstacle_id in backend_obstacles:
                if backend_obstacles[obstacle_id].position != (gx, gy):
                    world.move_obstacle(obstacle_id, gx, gy)
                    changes.append(f"Moved {obstacle_id} to ({gx}, {gy})")
            else:
                world.add_obstacle(
                    gx, gy, kind=obj["type"], obstacle_id=obstacle_id,
                )
                changes.append(f"Added {obstacle_id} at ({gx}, {gy})")
    except (ValueError, KeyError, TypeError) as error:
        st.session_state.editor_warning = (
            f"Editor update rejected: {error}."
        )
        refresh_editor_component()
        return

    normalized = []
    for obj in incoming:
        copy_obj = dict(obj)
        if copy_obj["type"] == "car":
            copy_obj["x"] = backend_car_canvas[0]
            copy_obj["y"] = backend_car_canvas[1]
        normalized.append(copy_obj)
    st.session_state.editor_objects = normalized
    if changes:
        st.session_state.editor_warning = (
            "Environment updated: " + "; ".join(changes)
        )


# =========================================================
# LLM vs A* BENCHMARK HELPERS
# =========================================================

def run_llm_vs_astar_benchmark(
    editor_objects,
    destination_letter,
    instruction_text,
    background_data,
):
    cars = [o for o in editor_objects if o["type"] == "car"]
    cones = [o for o in editor_objects if o["type"] == "cone"]
    barriers = [o for o in editor_objects if o["type"] == "barrier"]
    car_pos = (
        canvas_to_grid(cars[0]["x"], cars[0]["y"])
        if len(cars) == 1 else None
    )
    cone_pos = [canvas_to_grid(o["x"], o["y"]) for o in cones]
    barrier_pos = [canvas_to_grid(o["x"], o["y"]) for o in barriers]
    st.info(
        f"**Benchmark scene:** "
        f"car @ {car_pos} · "
        f"{len(cones)} cone(s) @ {cone_pos} · "
        f"{len(barriers)} barrier(s) @ {barrier_pos} · "
        f"destination = parking_{destination_letter}"
    )

    probe_world, probe_error = build_world(
        copy.deepcopy(editor_objects), destination_letter,
    )
    if probe_world is not None:
        unreachable = []
        for zone_name, zone_xy in probe_world.parking_zones.items():
            if zone_xy == probe_world.car.position:
                continue
            probe_path = astar(
                probe_world,
                probe_world.car.position,
                zone_xy,
            )
            if probe_path is None:
                unreachable.append(f"{zone_name}{zone_xy}")
        if unreachable:
            st.warning(
                f"⚠️ **Unreachable parking zones on this map:** "
                f"{', '.join(unreachable)}. "
                f"Cones or barriers have completely sealed them off. "
                f"If the LLM happens to choose one of these, the "
                f"benchmark will fail with 'A* found no route'. "
                f"Move or remove the blocking obstacles, or click "
                f"**🎯 Load Demo Scenario**."
            )
    elif probe_error:
        return None, f"Cannot build world: {probe_error}"

    results = {}
    llm_soft_avoid = set()

    for mode in ("LLM", "A*"):
        world, error = build_world(
            copy.deepcopy(editor_objects), destination_letter,
        )
        if error:
            return None, f"{mode}: build_world failed — {error}"

        agent = DriveAgent(world)

        def image_provider(_w=world):
            return render_scene_snapshot(_w, background_data)

        try:
            if mode == "LLM":
                decision = agent.start(
                    instruction=instruction_text,
                    image_provider=image_provider,
                    use_llm=True,
                )
                
                extracted = agent.current_avoid_types
                if extracted:
                    st.success(f"🧠 LLM extracted avoidance types: **{list(extracted)}** — {len(agent.soft_avoid_cells)} soft-avoid cells created.")
                else:
                    st.warning("⚠️ **LLM failed to extract any avoidance types** from the instruction. The path will be identical to A* only. Try typing: 'Drive to parking B, stay away from cones'.")

                if agent.soft_avoid_cells:
                    llm_soft_avoid = set(agent.soft_avoid_cells)
            else:
                decision = agent.start(
                    instruction=instruction_text,
                    image_provider=image_provider,
                    use_llm=False,
                )
        except OllamaUnavailable:
            raise
        except Exception as exc:
            return None, f"{mode} start failed — {exc}"

        if decision is None:
            return None, (
                f"{mode}: the planner returned no decision at all."
            )
        if decision.decision != "navigate":
            return None, (
                f"{mode}: the LLM chose **'{decision.decision}'** "
                f"instead of 'navigate'. "
                f"Reason: {decision.reason}"
            )
        if agent.status.state != "ready":
            chosen_dest = (
                decision.destination or f"parking_{destination_letter}"
            )
            return None, (
                f"{mode}: the LLM chose **'navigate to {chosen_dest}'**, "
                f"but the **A\\* planner could not produce a valid route**. "
                f"Agent ended in state **'{agent.status.state}'** "
                f"(message: *{agent.status.message}*). "
                f"**Almost always this means the chosen parking zone is "
                f"enclosed by obstacles** — check the World Summary table, "
                f"move a cone or barrier out of the destination's 4-neighbour "
                f"cells, or click **🎯 Load Demo Scenario**."
            )

        initial_path = list(agent.status.path)

        if mode == "A*":
            risk_crossed = sum(1 for p in initial_path if p in llm_soft_avoid)
            agent.status.risk_cells_crossed = risk_crossed
            agent.status.risk_cells_unconstrained = risk_crossed

        for _ in range(300):
            if agent.status.state not in {
                "ready", "moving", "replanning", "waiting"
            }:
                break
            agent.step()

        results[mode] = {
            "final_state": agent.status.state,
            "arrived": agent.status.state == "completed",
            "steps": agent.status.steps,
            "path_length": len(initial_path) if initial_path else 0,
            "path_length_unconstrained": agent.status.path_length_unconstrained,
            "risk_crossed": agent.status.risk_cells_crossed,
            "risk_unconstrained": agent.status.risk_cells_unconstrained,
            "replans": agent.status.replans,
            "llm_calls": agent.status.llm_calls,
            "safety_stops": agent.status.safety_stops,
            "avoid_types": sorted(agent.current_avoid_types),
            "soft_avoid_cells": len(agent.soft_avoid_cells),
        }

    return results, None


def compute_effectiveness(llm_res, astar_res):
    llm_risk = llm_res.get("risk_crossed", 0)
    astar_risk = astar_res.get("risk_crossed", 0)
    llm_steps = llm_res.get("steps", 0)
    astar_steps = astar_res.get("steps", 0)

    if astar_risk > 0:
        hazard_avoidance_rate = (astar_risk - llm_risk) / astar_risk * 100
    else:
        hazard_avoidance_rate = None

    step_cost = llm_steps - astar_steps

    return {
        "hazard_avoidance_rate": hazard_avoidance_rate,
        "step_cost": step_cost,
        "llm_extracted_types": llm_res.get("avoid_types", []),
        "soft_avoid_cells": llm_res.get("soft_avoid_cells", 0),
    }


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("EdgeMind Setup")

background_upload = st.sidebar.file_uploader(
    "Background image", type=["png", "jpg", "jpeg"],
    key="background_upload",
)
camera_upload = st.sidebar.camera_input(
    "Or webcam", key="camera_upload",
)

if background_upload is not None:
    st.session_state.background_data = image_to_data_uri(background_upload)
elif camera_upload is not None:
    st.session_state.background_data = image_to_data_uri(camera_upload)

st.sidebar.divider()
st.sidebar.subheader("Object sprites")

car_upload = st.sidebar.file_uploader(
    "Car", type=["png", "jpg", "jpeg"], key="car_sprite_upload")
cone_upload = st.sidebar.file_uploader(
    "Cone", type=["png", "jpg", "jpeg"], key="cone_sprite_upload")
barrier_upload = st.sidebar.file_uploader(
    "Barrier", type=["png", "jpg", "jpeg"], key="barrier_sprite_upload")

if car_upload is not None:
    st.session_state.car_sprite = image_to_data_uri(car_upload, (180, 180))
if cone_upload is not None:
    st.session_state.cone_sprite = image_to_data_uri(cone_upload, (180, 180))
if barrier_upload is not None:
    st.session_state.barrier_sprite = image_to_data_uri(
        barrier_upload, (180, 180))


# =========================================================
# MAIN PAGE
# =========================================================

st.title("🚗 EdgeMind — Driving Digital Twin")

st.caption(
    "Hierarchical control: **LLM extracts constraints** → "
    "**A* plans risk-aware routes** → **runtime safety tier validates every step**"
)


# =========================================================
# MAIN FRAGMENT
# =========================================================

@st.fragment(run_every=0.6)
def main_panel():

    agent = st.session_state.drive_agent
    world = st.session_state.drive_world
    active_states = {"ready", "moving", "replanning", "waiting"}

    if agent is not None and world is not None:
        st.subheader("🔴 Live Navigation")

        ctrl1, ctrl2, ctrl3 = st.columns(3)

        with ctrl1:
            toggle_key = (
                f"auto_drive_toggle_"
                f"{st.session_state.auto_drive_toggle_revision}"
            )
            auto_drive = st.toggle(
                "Automatic driving",
                value=st.session_state.auto_drive,
                key=toggle_key,
            )
            st.session_state.auto_drive = auto_drive

        with ctrl2:
            st.markdown("**Dynamic Obstacles**")
            if st.session_state.dynamic_events_enabled:
                st.markdown("🟢 **Enabled**")
            else:
                st.markdown("⚪ **Disabled**")

        with ctrl3:
            movement_delay = st.select_slider(
                "Move interval",
                options=[0.3, 0.45, 0.6, 0.8, 1.0, 1.2],
                value=0.6,
                format_func=lambda x: f"{x:.2f}s",
                key="movement_delay",
            )

        st.markdown("---")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("State", agent.status.state.capitalize())
        m2.metric("Position", str(world.car.position))
        m3.metric("Steps", agent.status.steps)
        m4.metric("Revision", world.revision)
        st.caption(agent.status.message)

        source_map = {
            "llm": "🧠 LLM (from instruction)",
            "dropdown": "⚙️ Dropdown (A* only)",
            "none": "❌ None (LLM refused)",
        }
        st.caption(
            f"**Destination:** {world.car.destination} — "
            f"**source:** {source_map.get(agent.status.destination_source, 'unknown')}"
        )

        if agent.status.state not in active_states:
            stop_automatic_driving()

        if agent.status.state == "completed":
            st.success("✅ Destination reached!")
        elif agent.status.state in ("blocked", "safety_stop"):
            st.error(agent.status.message)
        elif agent.status.state == "paused":
            st.warning(agent.status.message)
        elif agent.status.state == "waiting":
            st.info(f"⏸️ {agent.status.message}")
        elif agent.status.state == "clarify":
            st.warning(f"❓ {agent.status.message}")

        interval_ticks = max(1, round(movement_delay / 0.6))
        st.session_state.movement_tick_counter += 1

        should_move = (
            st.session_state.auto_drive
            and agent.status.state in active_states
            and (st.session_state.movement_tick_counter % interval_ticks == 0)
        )

        if (
            st.session_state.dynamic_events_enabled
            and st.session_state.auto_drive
            and agent.status.state in active_states
        ):
            st.session_state.dynamic_event_tick_counter += 1
            event_ticks = max(
                1,
                round(st.session_state.get(
                    "dynamic_event_interval", 3.0
                ) / 0.6),
            )
            if st.session_state.dynamic_event_tick_counter >= event_ticks:
                st.session_state.dynamic_event_tick_counter = 0
                success, message = generate_dynamic_event(world)
                if success:
                    st.toast(f"🌐 {message}", icon="🚧")

        if should_move:
            agent.step()
            sync_car_to_editor()

        st.markdown("#### 🧠 LLM-Extracted Mission Constraints")
        if agent.current_avoid_types:
            st.success(
                f"Avoidance: **{', '.join(sorted(agent.current_avoid_types))}** "
                f"— {len(agent.soft_avoid_cells)} risk-weighted cells"
            )
        else:
            st.warning(
                "⚠️ **No avoidance constraints extracted.** The LLM failed to find 'cone' or 'barrier' in your instruction. "
                "Try typing: 'Drive to parking B, stay away from cones'."
            )

        st.markdown("#### 🎯 Route Comparison (LLM's contribution, measured)")
        if (
            agent.current_avoid_types
            and agent.status.risk_cells_unconstrained > 0
        ):
            rc1, rc2, rc3, rc4 = st.columns(4)
            rc1.metric(
                "Path w/ LLM constraint",
                agent.status.path_length_constrained,
            )
            rc2.metric(
                "Path w/ A* alone",
                agent.status.path_length_unconstrained,
            )
            rc3.metric(
                "Hazard cells (LLM)",
                agent.status.risk_cells_crossed,
            )
            rc4.metric(
                "Hazard cells (A* alone)",
                agent.status.risk_cells_unconstrained,
            )
        else:
            st.caption(
                "Add cones to the map and use an avoidance instruction "
                "to see the route comparison."
            )

        st.markdown("#### 📊 LLM Decision Metrics")
        l1, l2, l3, l4, l5, l6 = st.columns(6)
        l1.metric("Calls", agent.status.llm_calls)
        l2.metric("Navigate", agent.status.llm_navigate)
        l3.metric("Wait", agent.status.llm_wait)
        l4.metric("Stop", agent.status.llm_stop)
        l5.metric("Safety overrides", agent.status.safety_overrides)
        l6.metric("Errors", agent.status.llm_errors)

        st.markdown("#### 📈 Replanning Metrics")
        r1, r2, r3, r4, r5 = st.columns(5)
        r1.metric("Replans", agent.status.replans)
        r2.metric("Detours", agent.status.replan_detours)
        r3.metric("Waits", agent.status.replan_waits)
        r4.metric("Safety stops", agent.status.safety_stops)
        r5.metric("Events", st.session_state.dynamic_event_count)

        st.divider()

    st.subheader("🗺️ Interactive World Editor")

    sprites = {
        "car": st.session_state.car_sprite,
        "cone": st.session_state.cone_sprite,
        "barrier": st.session_state.barrier_sprite,
    }

    component_value = editor_component(
        background=st.session_state.background_data,
        sprites=sprites,
        objects=st.session_state.editor_objects,
        route=st.session_state.planned_route,
        default=st.session_state.editor_objects,
        key=f"edgemind_editor_{st.session_state.editor_revision}",
        default_height=700,
    )

    incoming = clean_editor_objects(component_value)

    if st.session_state.demo_lock_countdown > 0:
        st.session_state.editor_objects = [
            dict(o) for o in st.session_state.demo_lock
        ]
        st.session_state.demo_lock_countdown -= 1
        if st.session_state.demo_lock_countdown == 0:
            st.session_state.demo_lock = None
        st.session_state.last_processed_component_key = None
        refresh_editor_component()
        st.rerun()

    if incoming is not None:
        incoming_key = json.dumps(incoming, sort_keys=True)
        if incoming_key != st.session_state.last_processed_component_key:
            st.session_state.last_processed_component_key = incoming_key
            if incoming != st.session_state.editor_objects:
                apply_editor_changes(incoming)
                st.rerun()

    if st.session_state.editor_warning:
        st.warning(st.session_state.editor_warning)
        st.session_state.editor_warning = None

    st.divider()

    # =====================================================
    # MISSION SETUP
    # =====================================================

    st.markdown("#### 🧭 Mission Setup")

    mode_col1, mode_col2 = st.columns([3, 1])
    with mode_col1:
        instruction_text = st.text_input(
            "Driving instruction (natural language)",
            value="Drive to parking B, stay away from cones",
            key="instruction_text_input",
        )
    with mode_col2:
        use_llm = st.toggle(
            "Use LLM", value=True, key="use_llm_toggle",
            help=(
                "ON: the LLM interprets your instruction and decides "
                "the destination. Off-topic instructions are refused. "
                "OFF: instruction is ignored; destination comes from the "
                "dropdown."
            ),
        )

    if use_llm:
        st.success(
            "🧠 **LLM mode** — your instruction controls the destination. "
            "The LLM will refuse instructions that are not about driving "
            "to a parking zone (e.g. *'put apples in basket'* is refused "
            "because EdgeMind is a driving planner, not a manipulation "
            "robot)."
        )
    else:
        st.warning(
            "⚙️ **A* only mode** — the text instruction is ignored. "
            "The dropdown below selects the destination directly."
        )

    st.caption(
        "Destination dropdown is a **fallback** — used only when LLM is "
        "off, or when the LLM interprets a *vague but valid* driving "
        "instruction (e.g. *'drive to the selected zone'*)."
    )

    dest_col, dyn_col = st.columns([2, 1])
    with dest_col:
        destination_letter = st.selectbox(
            "Fallback parking destination", ["A", "B", "C"],
            format_func=lambda x: f"Parking Zone {x}",
            key="destination_select",
        )
    with dyn_col:
        st.toggle(
            "Dynamic obstacles",
            key="dynamic_events_enabled",
            help="If enabled, obstacles will randomly appear/move/remove while driving.",
        )

    population_col, plan_col, clear_col = st.columns(3)

    with population_col:
        if st.button(
            "🚧 Auto-populate 5 obstacles",
            use_container_width=True,
            disabled=(st.session_state.drive_agent is not None),
        ):
            added = populate_initial_obstacles(
                TARGET_OBSTACLE_COUNT, destination_letter,
            )
            if added:
                refresh_editor_component()
                st.toast(f"Added {added} obstacles.", icon="🚧")
            st.rerun()

    with plan_col:
        if st.button(
            "🧭 Plan route and start driving",
            use_container_width=True,
        ):
            try:
                world, error = build_world(
                    st.session_state.editor_objects, destination_letter,
                )

                if error:
                    clear_navigation()
                    st.session_state.last_plan_status = error
                    st.error(error)
                    st.rerun()

                agent = DriveAgent(world)

                def image_provider(_world=world):
                    return render_scene_snapshot(
                        _world, st.session_state.background_data,
                    )

                decision = None

                if use_llm:
                    ok, status = agent.planner.status()
                    if not ok:
                        clear_navigation()
                        st.error(
                            f"**LLM is not ready.** {status}\n\n"
                            f"Options:\n"
                            f"1. Start Ollama and pull `qwen2.5vl:7b`; or\n"
                            f"2. Toggle **Use LLM** off to use the dropdown "
                            f"destination directly with A*."
                        )
                        st.session_state.last_plan_status = (
                            f"LLM not ready: {status}"
                        )
                        st.rerun()

                if use_llm:
                    with st.spinner("Consulting local VLM…"):
                        try:
                            decision = agent.start(
                                instruction=instruction_text,
                                image_provider=image_provider,
                                use_llm=True,
                            )
                        except OllamaUnavailable as llm_error:
                            clear_navigation()
                            st.error(
                                f"**LLM call failed.** {llm_error}\n\n"
                                f"Navigation was NOT started. "
                                f"Either fix Ollama, or toggle **Use LLM** "
                                f"off to use the dropdown destination "
                                f"with A*."
                            )
                            st.session_state.last_plan_status = (
                                f"LLM failed: {llm_error}"
                            )
                            st.rerun()
                        except Exception as llm_error:
                            clear_navigation()
                            st.error(
                                f"**LLM error.** {llm_error}\n\n"
                                f"Navigation was NOT started."
                            )
                            st.session_state.last_plan_status = (
                                f"LLM error: {llm_error}"
                            )
                            st.rerun()
                else:
                    with st.spinner("Planning with A*…"):
                        decision = agent.start(
                            instruction=instruction_text,
                            image_provider=image_provider,
                            use_llm=False,
                        )

                started = (
                    decision is not None
                    and decision.decision == "navigate"
                    and agent.status.state == "ready"
                )

                if started:
                    st.session_state.drive_world = world
                    st.session_state.drive_agent = agent
                    st.session_state.auto_drive = True
                    st.session_state.auto_drive_toggle_revision += 1
                    st.session_state.planned_route = [
                        list(p) for p in agent.status.path
                    ]
                    st.session_state.dynamic_event_count = 0
                    st.session_state.dynamic_event_tick_counter = 0
                    st.session_state.dynamic_event_history = []
                    st.session_state.movement_tick_counter = 0

                    source = agent.status.destination_source
                    dest_desc = (
                        decision.destination
                        if decision.destination
                        else f"parking_{destination_letter}"
                    )
                    st.session_state.last_plan_status = (
                        f"Started ({source} destination: {dest_desc}). "
                        f"Reason: {decision.reason}"
                    )
                    refresh_editor_component()

                else:
                    clear_navigation()

                    if (
                        decision is not None
                        and decision.decision == "clarify"
                    ):
                        st.warning(
                            f"⚠️ **The LLM rejected this instruction.**\n\n"
                            f"**Reason:** {decision.reason}\n\n"
                            f"EdgeMind is a *driving* mission planner "
                            f"— it cannot manipulate objects, cook, or "
                            f"perform tasks that are not about sending "
                            f"the car to a parking zone. Try:\n\n"
                            f"- *\"drive to parking A\"*\n"
                            f"- *\"go to parking B, stay away from cones\"*\n"
                            f"- *\"navigate to the nearest parking zone\"*\n\n"
                            f"If you want to use the dropdown destination "
                            f"directly (ignoring the instruction), toggle "
                            f"**Use LLM** off."
                        )
                        st.session_state.last_plan_status = (
                            f"LLM refused: {decision.reason}"
                        )

                    elif decision is not None and decision.decision == "stop":
                        st.error(f"LLM stopped: {decision.reason}")
                        st.session_state.last_plan_status = (
                            f"LLM stopped: {decision.reason}"
                        )

                    elif decision is not None and decision.decision == "wait":
                        st.info(f"LLM chose to wait: {decision.reason}")
                        st.session_state.last_plan_status = (
                            f"LLM wait: {decision.reason}"
                        )

                    else:
                        st.session_state.last_plan_status = (
                            "Planning failed. See logs."
                        )
                        st.error("Planning failed.")

            except Exception as exc:
                clear_navigation()
                st.session_state.last_plan_status = (
                    f"Planning failed: {exc}"
                )
                st.error(f"Planning failed: {exc}")

            st.rerun()

    with clear_col:
        if st.button("Clear navigation", use_container_width=True):
            clear_navigation()
            st.session_state.last_plan_status = "Navigation cleared."
            refresh_editor_component()
            st.rerun()

    if agent is not None and world is not None:
        manual_col, pause_col = st.columns(2)
        with manual_col:
            if st.button(
                "Execute one step",
                use_container_width=True,
                disabled=(
                    agent.status.state not in active_states
                    or st.session_state.auto_drive
                ),
            ):
                agent.step()
                sync_car_to_editor()
                st.rerun()
        with pause_col:
            if st.button(
                "Pause / Resume",
                use_container_width=True,
                disabled=agent.status.state not in active_states,
            ):
                st.session_state.auto_drive = (
                    not st.session_state.auto_drive
                )
                st.session_state.auto_drive_toggle_revision += 1
                st.rerun()

        st.divider()
        st.subheader("Dynamic Environment Event History")
        if st.session_state.dynamic_event_history:
            st.dataframe(
                st.session_state.dynamic_event_history,
                use_container_width=True, hide_index=True,
            )
        else:
            st.caption("No dynamic events yet.")

        st.subheader("World Change History")
        if world.change_history:
            st.dataframe(
                world.change_history,
                use_container_width=True, hide_index=True,
            )
        else:
            st.caption("No world changes yet.")


main_panel()


# =========================================================
# LLM vs A* BENCHMARK
# =========================================================

st.divider()
st.subheader("📊 LLM vs A* Benchmark — Accuracy & Effectiveness")

st.caption(
    "Runs the same map twice (LLM ON, then LLM OFF) with dynamic events "
    "disabled. The only variable is the LLM tier. Produces the Hazard "
    "Avoidance Rate — the fraction of A*'s hazard exposure that the LLM "
    "eliminated by extracting the avoidance constraint from your text."
)

with st.expander(
    "▶ Run benchmark",
    expanded=st.session_state.benchmark_result is None,
):
    st.warning(
        "⚠️ **Important for a clean demo:** The **🎯 Load Demo Scenario** "
        "button below places the car and cone in mathematically optimal "
        "positions. Do not add or move any objects after clicking it, "
        "or the A* baseline might naturally avoid the hazard zone by accident."
    )

    bench_col1, bench_col2, bench_col3 = st.columns([1, 1, 1])

    with bench_col1:
        st.button(
            "🎯 Load Demo Scenario",
            use_container_width=True,
            key="load_demo_btn",
            on_click=load_demo_scenario,
        )

    with bench_col2:
        if st.button(
            "🚀 Run LLM vs A* benchmark",
            type="primary",
            use_container_width=True,
            key="run_benchmark_button",
        ):
            objs = st.session_state.editor_objects
            cars = [o for o in objs if o["type"] == "car"]
            cones = [o for o in objs if o["type"] == "cone"]
            barriers = [o for o in objs if o["type"] == "barrier"]
            if len(objs) != 2 or len(cars) != 1 or len(cones) != 1 or barriers:
                st.warning(
                    "This is not the demo scene (expected exactly 1 car + 1 cone). "
                    "Click **🎯 Load Demo Scenario** first if you want the "
                    "documented 100% / +2 result."
                )

            with st.spinner(
                "Running both modes to completion — "
                "this may take a minute if the VLM is on CPU…"
            ):
                results, error = run_llm_vs_astar_benchmark(
                    objs,
                    st.session_state.get("destination_select", "B"),
                    st.session_state.get(
                        "instruction_text_input",
                        "Drive to parking B, stay away from cones.",
                    ),
                    st.session_state.background_data,
                )

            if error:
                st.error(f"**Benchmark failed.** {error}")
                st.session_state.benchmark_result = None
            else:
                st.session_state.benchmark_result = results

    with bench_col3:
        if st.session_state.benchmark_result is not None:
            if st.button(
                "🗑️ Clear benchmark result",
                use_container_width=True,
                key="clear_benchmark_button",
            ):
                st.session_state.benchmark_result = None
                st.rerun()

if st.session_state.benchmark_result is not None:
    llm_res = st.session_state.benchmark_result["LLM"]
    astar_res = st.session_state.benchmark_result["A*"]
    eff = compute_effectiveness(llm_res, astar_res)

    st.markdown("##### Side-by-side comparison")

    comparison_table = {
        "Metric": [
            "Arrived at destination",
            "Final state",
            "Steps taken",
            "Initial path length",
            "Hazard-adjacent cells on path",
            "Hazard cells A* would have crossed",
            "Replans",
            "LLM calls",
            "Safety stops",
            "Extracted avoidance types",
            "Soft-avoid cells",
        ],
        "LLM ON": [
            "✅" if llm_res["arrived"] else "❌",
            llm_res["final_state"],
            llm_res["steps"],
            llm_res["path_length"],
            llm_res["risk_crossed"],
            llm_res["risk_unconstrained"],
            llm_res["replans"],
            llm_res["llm_calls"],
            llm_res["safety_stops"],
            ", ".join(llm_res["avoid_types"]) or "none",
            llm_res["soft_avoid_cells"],
        ],
        "A* only": [
            "✅" if astar_res["arrived"] else "❌",
            astar_res["final_state"],
            astar_res["steps"],
            astar_res["path_length"],
            astar_res["risk_crossed"],
            astar_res["risk_unconstrained"],
            astar_res["replans"],
            astar_res["llm_calls"],
            astar_res["safety_stops"],
            ", ".join(astar_res["avoid_types"]) or "none",
            astar_res["soft_avoid_cells"],
        ],
    }

    st.dataframe(
        comparison_table,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("##### LLM effectiveness score")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric(
        "Hazard Avoidance Rate",
        "N/A" if eff["hazard_avoidance_rate"] is None
        else f"{eff['hazard_avoidance_rate']:.0f}%",
    )
    s2.metric(
        "Step cost of safety",
        f"{'+' if eff['step_cost'] >= 0 else ''}{eff['step_cost']}",
    )
    s3.metric("LLM constraint types", len(eff["llm_extracted_types"]))
    s4.metric("Soft-avoid cells", eff["soft_avoid_cells"])

    if eff["soft_avoid_cells"] == 0:
        st.warning(
            "No obstacles of the requested type were on the map."
        )
    elif eff["hazard_avoidance_rate"] is None:
        st.info(
            "**Both routes are hazard-free on this map.** A* happened to "
            "avoid every soft-avoid cell even without the LLM's constraint, "
            "so there is no hazard exposure to reduce. This is not a 100% "
            "win — it means the map does not test the LLM's contribution. "
            "**Load the demo scene** (cone directly on the direct path) to "
            "measure the LLM's actual effect."
        )
    elif eff["hazard_avoidance_rate"] > 0 and llm_res["arrived"]:
        st.success(
            f"**LLM contribution quantified.** "
            f"{eff['hazard_avoidance_rate']:.0f}% fewer hazard cells "
            f"crossed at a cost of {eff['step_cost']} extra step(s)."
        )
    else:
        st.info("Both routes similar on this map.")


# =========================================================
# WORLD SUMMARY
# =========================================================

st.divider()
st.subheader("World Summary")

objects = st.session_state.editor_objects
counts = {
    "Cars": sum(obj["type"] == "car" for obj in objects),
    "Cones": sum(obj["type"] == "cone" for obj in objects),
    "Barriers": sum(obj["type"] == "barrier" for obj in objects),
}
a, b, c = st.columns(3)
a.metric("Cars", counts["Cars"])
b.metric("Cones", counts["Cones"])
c.metric("Barriers", counts["Barriers"])

if objects:
    rows = []
    for obj in objects:
        gx, gy = canvas_to_grid(obj["x"], obj["y"])
        rows.append({
            "ID": obj["id"], "Type": obj["type"],
            "Canvas X": round(obj["x"], 1),
            "Canvas Y": round(obj["y"], 1),
            "Grid X": gx, "Grid Y": gy,
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)

st.write("**Status:**", st.session_state.last_plan_status)

if st.button("Reset entire editor"):
    st.session_state.editor_objects = []
    st.session_state.planned_route = []
    st.session_state.last_plan_status = "No route planned yet."
    clear_navigation()
    st.session_state.editor_revision += 1
    st.session_state.dynamic_event_id_counter = 0
    st.session_state.auto_drive_toggle_revision += 1
    st.session_state.last_processed_component_key = None
    st.session_state.benchmark_result = None
    st.session_state.demo_lock = None
    st.session_state.demo_lock_countdown = 0
    st.rerun()


with st.expander("📖 Defensible claims (open before defense)"):
    st.markdown(
        """
**What this project proves:**

1. **Deterministic instruction gate.** A word-list filter runs
   **before** any LLM call. Instructions containing manipulation
   verbs ("put", "place", "pick"...) with no driving vocabulary are
   refused without ever reaching the model. This works even if
   Ollama is down or the model is wrong.

2. **No silent fallback.** If you toggle LLM on but Ollama fails,
   the app refuses to start navigation and shows an error. It never
   quietly switches to A*-only behind your back.

3. **LLM extracts constraint objects.**
   *"Go to parking B, stay away from cones"* → LLM emits
   `{destination: parking_B, avoid_obstacle_types: ['cone']}` →
   A* plans a risk-weighted path. The benchmark quantifies the
   **Hazard Avoidance Rate** — but only on maps where A* actually
   has hazard exposure to reduce. On maps where A* is naturally
   hazard-free, the benchmark reports **N/A** rather than a
   misleading 100%.

4. **Safety is guaranteed regardless of LLM judgment.**

5. **Closed loop.** `LLM calls = 1 + Replans`.

6. **Graceful degradation** — but explicit, not silent.
        """)