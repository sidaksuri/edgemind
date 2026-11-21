"""Ollama-based high-level decision maker for EdgeMind driving."""
# drive_planner.py
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request

from dataclasses import dataclass, field

from vlmplan.drive_world import DrivingWorld


OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5vl:7b"


# =========================================================
# DETERMINISTIC INSTRUCTION FILTER
# =========================================================

DRIVING_KEYWORDS = {
    "drive", "driving", "navigate", "navigation", "go", "head",
    "park", "parking", "zone", "destination", "route", "road",
    "car", "vehicle", "travel", "reach", "arrive", "steer",
    "turn", "return", "leave", "move", "steer", "wheel",
}

MANIPULATION_VERBS = {
    "put", "place", "pick", "grab", "take", "lift", "stack",
    "sort", "arrange", "assemble", "fill", "empty", "pour",
    "cook", "make", "bake", "prepare", "clean", "wash", "wipe",
    "cut", "chop", "write", "draw", "paint", "type", "email",
    "message", "call", "buy", "sell", "order", "book", "open",
    "close", "flip", "press", "tap", "throw", "catch", "roll",
    "juggle", "spread", "mix", "stir", "blend", "chop", "peel",
    "assemble", "disassemble", "attach", "detach", "wire",
    "screw", "tighten", "loosen", "hammer", "nail", "saw",
}


def is_likely_driving_instruction(text: str) -> tuple[bool, str]:
    words = set(re.findall(r"[a-z]+", text.lower()))
    has_driving = bool(words & DRIVING_KEYWORDS)
    has_manip = bool(words & MANIPULATION_VERBS)

    if has_manip and not has_driving:
        triggers = ", ".join(sorted(words & MANIPULATION_VERBS))
        return False, (
            f"The instruction contains manipulation verb(s) "
            f"({triggers}) but no driving vocabulary. "
            f"EdgeMind is a driving mission planner, not a manipulation "
            f"robot. Try: 'drive to parking A', 'go to parking B', or "
            f"'navigate to the nearest parking zone'."
        )

    return True, ""


# =========================================================
# PROMPTS
# =========================================================

PROMPT = """
You are the mission planner for a simulated autonomous CAR.

Your ONLY job is to interpret DRIVING instructions that send the car
to a parking zone. You are NOT a manipulation robot, NOT a cooking
assistant, NOT a general-purpose chatbot.

Valid destinations (from the scene): parking_A, parking_B, parking_C.

If the user's instruction is about ANYTHING other than driving the car
to a parking zone - manipulation, cooking, general questions, any
unrelated task - return decision "clarify".

If the instruction is a valid driving instruction but does NOT name a
specific zone (e.g. "drive to the selected zone"), use the car's
CURRENT destination shown in the scene.

CRITICAL RULE: If the instruction mentions avoiding, dodging, or staying away from specific objects, you MUST list them in "avoid_obstacle_types". Valid types are "cone" and "barrier". If the instruction does not mention avoiding anything, leave it as an empty list [].

Return valid JSON only, using this exact schema.

{
  "decision": "<navigate|wait|stop|clarify>",
  "destination": "<parking_A|parking_B|parking_C|null>",
  "avoid_obstacle_types": [],
  "reason": "<one short sentence>"
}

Examples of CORRECT behaviour:

  Instruction: "drive to parking A"
  Output: {"decision":"navigate","destination":"parking_A",
           "avoid_obstacle_types":[],"reason":"Clear driving request."}

  Instruction: "go to B, stay away from cones"
  Output: {"decision":"navigate","destination":"parking_B",
           "avoid_obstacle_types":["cone"],
           "reason":"Driving request with cone avoidance."}

  Instruction: "put apples in basket"
  Output: {"decision":"clarify","destination":null,
           "avoid_obstacle_types":[],
           "reason":"Manipulation task - not a driving instruction."}

Now interpret the user's actual instruction below.

USER INSTRUCTION:
<<INSTRUCTION>>

CURRENT ENVIRONMENT:
<<SCENE>>

ROUTE AVAILABLE:
<<ROUTE>>
"""


REPLAN_PROMPT = """
You are the strategic replanning controller of EdgeMind.

The car is en route to its destination. The environment just changed.

You see the current scene (image if provided, plus a text description).
DETOUR COST tells you how many extra steps the new route requires
relative to the route the car was following:

  detour_cost = 0     : route unchanged; continue.
  detour_cost = 1-5   : cheap detour; navigate.
  detour_cost = 6-10  : moderate detour; navigate unless dangerous.
  detour_cost > 10    : expensive; PREFER WAIT - the obstacle may clear.
  detour_cost = -1    : no path exists at all; WAIT.

Possible decisions: navigate, wait, stop, clarify.

Rules:
- WAIT is the correct answer when detour_cost > 10 or detour_cost = -1.
- STOP only when the scene clearly shows danger.

Return valid JSON only:

{
  "decision": "<navigate|wait|stop|clarify>",
  "reason": "<one sentence>"
}

RECENT CHANGES:
<<CHANGES>>

DETOUR COST (extra steps to re-route, or -1 if no path exists):
<<DETOUR_COST>>

CURRENT ENVIRONMENT:
<<SCENE>>
"""


@dataclass
class DriveDecision:
    decision: str
    destination: str | None
    reason: str
    avoid_obstacle_types: list[str] = field(default_factory=list)


class OllamaUnavailable(RuntimeError):
    pass


def _strip_data_uri_prefix(image_data_uri: str) -> str:
    if not image_data_uri:
        return ""
    if "," in image_data_uri:
        return image_data_uri.split(",", 1)[1]
    return image_data_uri


class OllamaDrivePlanner:

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        url: str = OLLAMA_URL,
        timeout: float = 60.0,
    ):
        self.model = model
        self.url = url.rstrip("/")
        self.timeout = timeout

        self.stats = {
            "calls": 0, "navigate": 0, "wait": 0,
            "stop": 0, "clarify": 0, "errors": 0,
        }

    def status(self) -> tuple[bool, str]:
        try:
            request = urllib.request.Request(self.url + "/api/tags")
            with urllib.request.urlopen(request, timeout=5) as response:
                models = json.load(response).get("models", [])
            names = [m.get("name", "") for m in models]
            if self.model not in names:
                return (
                    False,
                    f"Model {self.model} not installed. Available: {names}",
                )
            return True, "Ollama is ready."
        except Exception as error:
            return (False, f"Ollama is unavailable: {error}")

    def _chat(
        self,
        prompt: str,
        image_data_uri: str | None = None,
    ) -> str:
        message: dict = {"role": "user", "content": prompt}

        if image_data_uri:
            image_b64 = _strip_data_uri_prefix(image_data_uri)
            if image_b64:
                message["images"] = [image_b64]

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are EdgeMind's mission planner. You only "
                        "interpret DRIVING instructions. Refuse off-topic "
                        "requests with decision='clarify'. If an image is "
                        "provided, read the scene from it; do not "
                        "hallucinate obstacles."
                    ),
                },
                message,
            ],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0, "num_predict": 250},
        }

        request = urllib.request.Request(
            self.url + "/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout
            ) as response:
                data = json.load(response)
            return data["message"]["content"]
        except (
            urllib.error.URLError,
            TimeoutError,
            ConnectionError,
        ) as error:
            raise OllamaUnavailable(
                f"Could not communicate with Ollama: {error}"
            ) from error

    @staticmethod
    def _parse(raw: str) -> dict:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            start = raw.find("{")
            end = raw.rfind("}")
            if start == -1 or end == -1:
                raise ValueError("Model did not return valid JSON.")
            data = json.loads(raw[start:end + 1])
        if not isinstance(data, dict):
            raise ValueError("Model response must be a JSON object.")
        return data

    def decide(
        self,
        instruction: str,
        world: DrivingWorld,
        route_available: bool,
        image_data_uri: str | None = None,
    ) -> DriveDecision:
        scene = world.describe()

        prompt = (
            PROMPT
            .replace("<<INSTRUCTION>>", instruction)
            .replace("<<SCENE>>", scene)
            .replace("<<ROUTE>>", "YES" if route_available else "NO")
        )

        self.stats["calls"] += 1

        try:
            raw = self._chat(prompt, image_data_uri=image_data_uri)
            result = self._parse(raw)

            decision = str(result.get("decision", "clarify")).lower()
            destination = result.get("destination")
            reason = str(result.get("reason", "No explanation provided."))
            avoid = result.get("avoid_obstacle_types", [])

            if not isinstance(avoid, list):
                avoid = []
            avoid = [
                str(t).lower() for t in avoid
                if str(t).lower() in ("cone", "barrier")
            ]

            # --- DETERMINISTIC SAFETY NET ---
            # If the VLM missed a constraint that is explicitly in the text,
            # we force it here. This guarantees the simulation never breaks.
            text_lower = instruction.lower()
            if "cone" in text_lower and "cone" not in avoid:
                avoid.append("cone")
            if "barrier" in text_lower and "barrier" not in avoid:
                avoid.append("barrier")

            allowed = {"navigate", "wait", "stop", "clarify"}
            if decision not in allowed:
                decision = "clarify"

            if destination is not None:
                destination = str(destination)

            if (
                destination is not None
                and destination not in world.parking_zones
            ):
                decision = "clarify"
                destination = None
                reason = "The proposed destination does not exist."

            if decision == "navigate" and not route_available:
                decision = "stop"
                reason = (
                    "The model proposed navigation, but A* "
                    "could not find a valid route."
                )

            if decision == "navigate" and destination is None:
                decision = "clarify"
                reason = "No valid destination was selected."

            self.stats[decision] += 1

            return DriveDecision(
                decision=decision,
                destination=destination,
                reason=reason,
                avoid_obstacle_types=avoid,
            )

        except Exception:
            self.stats["errors"] += 1
            raise

    def decide_replan(
        self,
        reason: str,
        world: DrivingWorld,
        detour_cost: int = 0,
        image_data_uri: str | None = None,
    ) -> DriveDecision:
        scene = world.describe()

        prompt = (
            REPLAN_PROMPT
            .replace("<<CHANGES>>", reason or "environment updated")
            .replace("<<DETOUR_COST>>", str(detour_cost))
            .replace("<<SCENE>>", scene)
        )

        self.stats["calls"] += 1

        try:
            raw = self._chat(prompt, image_data_uri=image_data_uri)
            result = self._parse(raw)

            decision = str(result.get("decision", "navigate")).lower()
            reason_text = str(result.get("reason", ""))

            if decision not in {"navigate", "wait", "stop", "clarify"}:
                decision = "navigate"

            self.stats[decision] += 1

            return DriveDecision(
                decision=decision,
                destination=None,
                reason=reason_text,
            )

        except Exception:
            self.stats["errors"] += 1
            raise