
"""Rendering and perception for the EdgeMind tabletop digital twin."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from .world import (
    BIN_HALF,
    COLORS,
    OBSTACLE_COLOR,
    World,
)

PX = 400


def _px(value: float) -> int:
    return int(round(value * PX))


def render(
    world: World,
    noise: float = 6.0,
    seed: int = 0,
) -> np.ndarray:
    """Render the current world as a top-down RGB camera image."""

    img = Image.new("RGB", (PX, PX), (235, 235, 230))
    draw = ImageDraw.Draw(img)

    # Draw bins.
    for bin_obj in world.bins():
        x, y = bin_obj.xy

        draw.rectangle(
            [
                _px(x - BIN_HALF[0]),
                _px(y - BIN_HALF[1]),
                _px(x + BIN_HALF[0]),
                _px(y + BIN_HALF[1]),
            ],
            outline=COLORS[bin_obj.color],
            width=6,
        )

    # Draw obstacles before objects.
    # This ensures objects in front of an obstacle remain visible.
    for obstacle in world.obstacles.values():
        x, y = obstacle.xy

        if obstacle.shape == "circle":
            r = obstacle.radius

            draw.ellipse(
                [
                    _px(x - r),
                    _px(y - r),
                    _px(x + r),
                    _px(y + r),
                ],
                fill=OBSTACLE_COLOR,
                outline=(80, 20, 80),
                width=3,
            )

        else:
            half_w = obstacle.width / 2
            half_h = obstacle.height / 2

            draw.rectangle(
                [
                    _px(x - half_w),
                    _px(y - half_h),
                    _px(x + half_w),
                    _px(y + half_h),
                ],
                fill=OBSTACLE_COLOR,
                outline=(80, 20, 80),
                width=3,
            )

    # Draw stacked items bottom-up.
    order = sorted(
        world.items(),
        key=lambda obj: world.height(obj.id),
    )

    for obj in order:
        x, y = obj.xy

        r = 0.035
        level = world.height(obj.id)

        # Stack shadow ring.
        draw.ellipse(
            [
                _px(x - r - 0.006 * level),
                _px(y - r - 0.006 * level),
                _px(x + r + 0.006 * level),
                _px(y + r + 0.006 * level),
            ],
            fill=(60, 60, 60),
        )

        box = [
            _px(x - r),
            _px(y - r),
            _px(x + r),
            _px(y + r),
        ]

        if obj.kind == "cube":
            draw.rectangle(
                box,
                fill=COLORS[obj.color],
            )
        else:
            draw.ellipse(
                box,
                fill=COLORS[obj.color],
            )

    arr = np.asarray(img).astype(float)

    arr += np.random.default_rng(seed).normal(
        0,
        noise,
        arr.shape,
    )

    return np.clip(arr, 0, 255).astype(np.uint8)


@dataclass
class Detection:
    kind: str
    color: str
    xy: tuple[float, float]
    level: int = 1
    in_bin: str | None = None


@dataclass
class ObstacleDetection:
    shape: str
    xy: tuple[float, float]
    width: float
    height: float
    radius: float


def _color_mask(
    image: np.ndarray,
    color: tuple[int, int, int],
    threshold: float = 60,
) -> np.ndarray:
    """Find pixels close to a particular RGB colour."""

    return (
        np.linalg.norm(
            image.astype(float) - np.array(color),
            axis=2,
        ) < threshold
    )


def perceive(
    img: np.ndarray,
) -> tuple[list[Detection], list[Detection]]:
    """Return detected items and bins.

    The two-value return format is preserved for compatibility.
    """

    f = img.astype(float)

    items: list[Detection] = []
    bins: list[Detection] = []

    for color_name, rgb in COLORS.items():
        mask = np.linalg.norm(
            f - np.array(rgb),
            axis=2,
        ) < 60

        mask = ndimage.binary_opening(
            mask,
            iterations=1,
        )

        labels, count = ndimage.label(mask)

        for label_id in range(1, count + 1):
            ys, xs = np.nonzero(
                labels == label_id
            )

            if len(xs) < 40:
                continue

            width = np.ptp(xs) + 1
            height = np.ptp(ys) + 1

            fill = len(xs) / (width * height)

            cx = xs.mean() / PX
            cy = ys.mean() / PX

            if width > 70:
                bins.append(
                    Detection(
                        "bin",
                        color_name,
                        (cx, cy),
                    )
                )
            else:
                kind = (
                    "cube"
                    if fill > 0.88
                    else "ball"
                )

                items.append(
                    Detection(
                        kind,
                        color_name,
                        (cx, cy),
                    )
                )

    # Estimate stack levels from dark shadow rings.
    dark = np.all(f < 100, axis=2)

    for item in items:
        x = _px(item.xy[0])
        y = _px(item.xy[1])

        radius = _px(0.035)
        ring = 0

        if not (
            0 <= y < PX
            and 0 <= x < PX
        ):
            continue

        for k in range(1, 12):
            xx = min(PX - 1, x + radius + k)

            if dark[y, xx]:
                ring = k
            else:
                break

        item.level = max(
            1,
            int(round(ring / (0.006 * PX))),
        )

    # Detect objects inside bins.
    for item in items:
        for bin_obj in bins:
            if (
                abs(item.xy[0] - bin_obj.xy[0])
                < BIN_HALF[0]
                and
                abs(item.xy[1] - bin_obj.xy[1])
                < BIN_HALF[1]
            ):
                item.in_bin = bin_obj.color

    return items, bins


def perceive_obstacles(
    img: np.ndarray,
) -> list[ObstacleDetection]:
    """Detect the dedicated obstacle colour from camera pixels."""

    mask = _color_mask(
        img,
        OBSTACLE_COLOR,
        threshold=65,
    )

    mask = ndimage.binary_opening(
        mask,
        iterations=1,
    )

    labels, count = ndimage.label(mask)

    obstacles: list[ObstacleDetection] = []

    for label_id in range(1, count + 1):
        ys, xs = np.nonzero(
            labels == label_id
        )

        if len(xs) < 30:
            continue

        width_px = np.ptp(xs) + 1
        height_px = np.ptp(ys) + 1

        cx = float(xs.mean() / PX)
        cy = float(ys.mean() / PX)

        width = width_px / PX
        height = height_px / PX

        # Estimate the visible geometry.
        ratio = width_px / max(height_px, 1)

        if 0.8 <= ratio <= 1.25:
            shape = "circle"
            radius = max(width, height) / 2
        else:
            shape = "rectangle"
            radius = 0.0

        obstacles.append(
            ObstacleDetection(
                shape=shape,
                xy=(cx, cy),
                width=width,
                height=height,
                radius=radius,
            )
        )

    return obstacles
