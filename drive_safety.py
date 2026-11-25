"""Runtime safety shield for EdgeMind (single canonical path validator)."""
# drive_safety.py
from vlmplan.drive_world import DrivingWorld


def validate_path(world: DrivingWorld, path: list[tuple[int, int]]):
    """Validate that ``path`` is a well-formed, currently-unblocked route.

    Checks: non-empty, starts at the car, ends at the destination, every
    cell is inside the grid and unblocked, consecutive cells are 4-connected.

    This is NOT a guarantee against cells becoming blocked between
    validation and motion; ``DriveAgent.step`` re-checks ``world.blocked``
    immediately before each move.
    """
    if not path:
        return False, "No route provided."
    if tuple(path[0]) != (world.car.x, world.car.y):
        return False, "Route does not start at the car."
    if tuple(path[-1]) != tuple(world.car.destination):
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