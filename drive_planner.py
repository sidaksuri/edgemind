"""Ollama-based constraint extractor for EdgeMind driving."""
# drive_planner.py
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request

from dataclasses import dataclass, field

from vlmplan.drive_world import DrivingWorld


OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5vl:7b"   # switch to "qwen2.5vl:3b" if too slow on CPU


# =========================================================
# DETERMINISTIC INSTRUCTION FILTER
# =========================================================

DRIVING_KEYWORDS = {
    "drive", "driving", "navigate", "navigation", "go", "head",
    "park", "parking", "zone", "destination", "route", "road",
    "car", "vehicle", "travel", "reach", "arrive", "steer",
    "turn", "return", "leave", "move", "wheel",
}

MANIPULATION_VERBS = {
    "put", "place", "pick", "grab", "take", "lift", "stack",
    "sort", "arrange", "assemble", "fill", "empty", "pour",
    "cook", "make", "bake", "prepare", "clean", "wash", "wipe",
    "cut", "chop", "write", "draw", "paint", "type", "email",
    "message", "call", "buy", "sell", "order", "book", "open",
    "close", "flip", "press", "tap", "throw", "catch", "roll",
    "juggle", "spread", "mix", "stir", "blend", "peel",
    "disassemble", "attach", "detach", "wire",
    "screw", "tighten", "loosen", "hammer", "nail", "saw",
}

_MULTI_WORD_NEGATIONS = ("do not", "don't", "dont")
_SINGLE_WORD_NEGATIONS = {"not", "never", "avoid", "skip", "without", "no"}


def _has_negation(lowered: str, words: set[str]) -> bool:
    for phrase in _MULTI_WORD_NEGATIONS:
        if phrase in lowered:
            return True
    return bool(words & _SINGLE_WORD_NEGATIONS)


def is_likely_driving_instruction(text: str) -> tuple[bool, str]:
    normalized = text.replace("\u2019", "'").replace("\u2018", "'")
    lowered = normalized.lower()
    words = set(re.findall(r"[a-z]+", lowered))

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

    has_destination_token = bool(words & {"a", "b", "c", "zone", "parking"})
    if _has_negation(lowered, words) and not has_destination_token:
        return False, (
            "The instruction negates driving without naming a "
            "destination. Please tell EdgeMind where to drive "
            "(e.g. 'drive to parking B')."
        )

    return True, ""


# =========================================================
# PROMPTS
# =========================================================

PROMPT = """
You are the constraint-extraction tier for an autonomous car.

You do NOT plan routes and you do NOT control the car. A separate A*
planner consumes your output. Your ONLY job is to translate the user's
natural-language instruction into a structured mission spec.

You are also given an image of the current scene. Use it to locate
obstacles and parking zones visually. If the image and the text summary
disagree, trust the text summary (it comes from the ground-truth world).

OUTPUT SCHEMA — return strict JSON, no prose outside the braces:

{
  "decision": "<navigate|wait|stop|clarify>",
  "destination": "<parking_A|parking_B|parking_C|null>",
  "safety_vs_time": <float 0.0-1.0>,
  "avoid_obstacles": [
    {
      "position": [<int x>, <int y>],
      "kind": "<cone|barrier>",
      "weight": <float 0.0-50.0>,
      "reason": "<one short clause>"
    }
  ],
  "reasoning": "<2-3 sentences explaining the weights you chose>",
  "reason": "<one sentence: overall decision>"
}

WEIGHT INTERPRETATION — this is the whole point, read carefully:

  "avoid cones"                            -> weight ≈ 5.0 per cone
  "stay far away from cones"               -> weight ≈ 12.0 per cone
  "strictly avoid / at all costs"          -> weight ≈ 40.0
  "prefer to avoid"                        -> weight ≈ 2.0
  "avoid if it's free / doesn't cost much" -> weight ≈ 1.5
  "ignore the cones / don't care"          -> weight = 0.0

CRITICAL: two cones on the same map MUST receive DIFFERENT weights if
they are in different strategic positions. Example:

  cone at (8,10) blocks the direct corridor -> weight ≈ 15.0
  cone at (14,10) is tucked near the wall  -> weight ≈ 2.0

Explain WHY in "reasoning". If the instruction does not tell you to
distinguish obstacles, use the SAME weight for all obstacles of that kind.

SAFETY_VS_TIME — a global knob the A* planner multiplies the weights by:

  "drive carefully / be safe"              -> 0.85
  neutral                                   -> 0.50
  "get there fast / as quick as possible"  -> 0.25

REFUSAL: if the instruction is not about driving to a parking zone
(manipulation, cooking, chat), return "clarify" with all other fields
empty (avoid_obstacles: [], safety_vs_time: 0.5).

EXAMPLES:

  Instruction: "drive to parking A"
  Output: {"decision":"navigate","destination":"parking_A",
           "safety_vs_time":0.5,"avoid_obstacles":[],
           "reasoning":"No avoidance requested; neutral safety.",
           "reason":"Clear driving request."}

  Instruction: "go to B, stay far away from the cones"
  Output: {"decision":"navigate","destination":"parking_B",
           "safety_vs_time":0.8,
           "avoid_obstacles":[
             {"position":[8,10],"kind":"cone","weight":15.0,
              "reason":"blocks direct row-10 corridor"},
             {"position":[14,10],"kind":"cone","weight":3.0,
              "reason":"near wall, small detour suffices"}],
           "reasoning":"Car travels east on row 10. The first cone
             forces a large detour so I weight it heavily. The second
             is tucked near the wall so tolerating its halo is cheap.",
           "reason":"Driving request with cone avoidance."}

  Instruction: "drive to B, avoid cones if it doesn't cost extra"
  Output: {"decision":"navigate","destination":"parking_B",
           "safety_vs_time":0.3,
           "avoid_obstacles":[
             {"position":[8,10],"kind":"cone","weight":1.5,
              "reason":"low cost — only avoid if free"},
             {"position":[14,10],"kind":"cone","weight":1.5,
              "reason":"low cost — only avoid if free"}],
           "reasoning":"User allows cone exposure when avoiding is
             costly; low weights let A* choose the direct route.",
           "reason":"Driving request with soft avoidance."}

  Instruction: "put apples in basket"
  Output: {"decision":"clarify","destination":null,
           "safety_vs_time":0.5,"avoid_obstacles":[],
           "reasoning":"Not a driving task.","reason":"Manipulation."}

USER INSTRUCTION:
<<INSTRUCTION>>

CURRENT ENVIRONMENT:
<<SCENE>>
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
- Do NOT return avoid_obstacles or safety_vs_time; the original
  instruction's constraint field is preserved across replans.

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
    # ``None`` means "field was not returned by the LLM" (e.g. replan
    # decisions, which deliberately do not re-extract constraints).
    # Callers MUST distinguish None (leave existing constraints alone)
    # from an explicit empty list (obstacle avoidance explicitly cleared).
    avoid_obstacles: list[dict] | None = None
    safety_vs_time: float | None = None
    reasoning: str = ""

    @property
    def avoid_obstacle_types(self) -> list[str]:
        """Backward-compat: distinct obstacle kinds requested."""
        return sorted({o["kind"] for o in (self.avoid_obstacles or [])})


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
        timeout: float = 300.0,     # was 90 — a 7B VLM on CPU needs headroom
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

    def _scene_summary(self, world: DrivingWorld) -> str:
        car = world.car
        obstacles = [
            f"{o.kind}@({o.x},{o.y})" for o in world.active_obstacles()
        ]
        zones = ", ".join(
            f"{name}=({xy[0]},{xy[1]})"
            for name, xy in world.parking_zones.items()
        )
        return (
            f"Car at ({car.position[0]},{car.position[1]}) "
            f"facing {car.direction}. "
            f"Current destination {car.destination}. "
            f"Parking zones: {zones}. "
            f"Obstacles: {obstacles or 'none'}."
        )

    def _chat(self, prompt: str, image_data_uri: str | None = None) -> str:
        message: dict = {"role": "user", "content": prompt}
        if image_data_uri:
            image_b64 = _strip_data_uri_prefix(image_data_uri)
            if image_b64:
                message["images"] = [image_b64]

        payload = {
            "model": self.model,
            "messages": [message],
            "stream": False,
            # IMPORTANT: intentionally NO "format": "json".
            # On multiple Ollama versions this flag causes vision
            # models to hang or silently ignore the image. We parse
            # the JSON ourselves in _parse_decision().
            "options": {
                "num_predict": 768,     # cap generation — kills runaway hangs
                "temperature": 0.1,     # deterministic JSON
            },
            "keep_alive": "10m",        # keep model warm between calls
        }

        request = urllib.request.Request(
            self.url + "/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout
            ) as response:
                data = json.load(response)
        except TimeoutError as error:
            raise OllamaUnavailable(
                f"Ollama timed out after {self.timeout:.0f}s. "
                f"The model ({self.model}) may be too large for your "
                f"machine. Try `ollama pull qwen2.5vl:3b` and set "
                f"DEFAULT_MODEL accordingly."
            ) from error
        except urllib.error.URLError as error:
            raise OllamaUnavailable(
                f"Ollama request failed: {error}. "
                f"Check that `ollama serve` is running."
            ) from error
        except json.JSONDecodeError as error:
            raise OllamaUnavailable(
                f"Ollama returned a non-JSON HTTP body: {error}"
            ) from error

        content = data.get("message", {}).get("content", "")
        if not content:
            raise OllamaUnavailable(
                "Ollama returned an empty response. This usually means "
                "the model crashed while processing the image. Try a "
                "smaller image (max_dim=400), or a smaller model."
            )
        return content

    def _parse_decision(self, raw: str) -> DriveDecision:
        text = raw.strip()
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            text = m.group(0)
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return DriveDecision(
                decision="clarify", destination=None,
                reason="LLM returned invalid JSON.",
            )

        decision = str(data.get("decision", "clarify")).lower()
        if decision not in {"navigate", "wait", "stop", "clarify"}:
            decision = "clarify"

        dest = data.get("destination")
        if dest in (None, "null", "", "None"):
            dest = None

        # Distinguish "field absent from response" (None) from
        # "explicitly provided" (parsed value, possibly empty).
        # REPLAN_PROMPT deliberately omits these fields so that
        # DriveAgent._apply_navigate_decision preserves the constraints
        # extracted from the original user instruction across replans.
        safety: float | None = None
        if "safety_vs_time" in data and data["safety_vs_time"] is not None:
            try:
                safety = max(0.0, min(1.0, float(data["safety_vs_time"])))
            except (TypeError, ValueError):
                safety = None

        avoid: list[dict] | None = None
        if "avoid_obstacles" in data:
            parsed: list[dict] = []
            for item in (data.get("avoid_obstacles") or []):
                if not isinstance(item, dict):
                    continue
                pos = item.get("position")
                kind = item.get("kind")
                if not (
                    isinstance(pos, list) and len(pos) == 2
                    and kind in ("cone", "barrier")
                ):
                    continue
                try:
                    px, py = int(pos[0]), int(pos[1])
                except (TypeError, ValueError):
                    continue
                try:
                    weight = float(item.get("weight", 5.0))
                except (TypeError, ValueError):
                    weight = 5.0
                weight = max(0.0, min(50.0, weight))
                parsed.append({
                    "position": [px, py],
                    "kind": kind,
                    "weight": weight,
                    "reason": str(item.get("reason", "")),
                })
            avoid = parsed

        return DriveDecision(
            decision=decision,
            destination=dest,
            reason=str(data.get("reason", "")) or "No reason provided.",
            avoid_obstacles=avoid,
            safety_vs_time=safety,
            reasoning=str(data.get("reasoning", "")),
        )

    def decide(
        self,
        instruction: str,
        world: DrivingWorld,
        route_available: bool = True,
        image_data_uri: str | None = None,
    ) -> DriveDecision:
        self.stats["calls"] += 1
        prompt = PROMPT.replace("<<INSTRUCTION>>", instruction).replace(
            "<<SCENE>>", self._scene_summary(world)
        )
        try:
            raw = self._chat(prompt, image_data_uri=image_data_uri)
        except OllamaUnavailable:
            self.stats["errors"] += 1
            raise
        except Exception as error:
            self.stats["errors"] += 1
            raise OllamaUnavailable(str(error)) from error

        decision = self._parse_decision(raw)
        self.stats[decision.decision] = self.stats.get(decision.decision, 0) + 1
        return decision

    def decide_replan(
        self,
        reason: str,
        world: DrivingWorld,
        detour_cost: int,
        image_data_uri: str | None = None,
    ) -> DriveDecision:
        self.stats["calls"] += 1
        prompt = (
            REPLAN_PROMPT
            .replace("<<CHANGES>>", reason)
            .replace("<<DETOUR_COST>>", str(detour_cost))
            .replace("<<SCENE>>", self._scene_summary(world))
        )
        try:
            raw = self._chat(prompt, image_data_uri=image_data_uri)
        except OllamaUnavailable:
            self.stats["errors"] += 1
            raise
        except Exception as error:
            self.stats["errors"] += 1
            raise OllamaUnavailable(str(error)) from error

        decision = self._parse_decision(raw)
        self.stats[decision.decision] = self.stats.get(decision.decision, 0) + 1
        return decision