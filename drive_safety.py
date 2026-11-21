# drive_safety.py
from vlmplan.drive_world import DrivingWorld


def validate_path(world: DrivingWorld, path: list[tuple[int, int]]):
    if not path:
        return False, "No route provided."

    if path[0] != (world.car.x, world.car.y):
        return False, "Route does not start at the car."

    if path[-1] != world.car.destination:
        return False, "Route does not end at the destination."

    for i, (x, y) in enumerate(path):
        if not world.inside_grid(x, y):
            return False, f"Outside grid: {(x, y)}"

        if world.blocked(x, y):
            return False, f"Blocked cell: {(x, y)}"

        if i > 0:
            px, py = path[i - 1]

            if abs(x - px) + abs(y - py) != 1:
                return False, "Invalid movement detected."

    return True, "Route passed safety validation."
