"""Run with the benchmark's Python, not Buffalo's dependency environment.

Only official data, formatters, scorers and interactive environments are loaded.
stdout is a private protocol; all upstream logs go to stderr. Gold data stays here.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import traceback
from pathlib import Path

WIRE = sys.stdout
CURRENT_REQUEST_ID = None
SOURCES = {
    "arc-agi-3": "https://github.com/arcprize/arc-agi",
}


def emit(value):
    value["request_id"] = CURRENT_REQUEST_ID
    WIRE.write(json.dumps(value, ensure_ascii=False) + "\n")
    WIRE.flush()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def command(*args):
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.PIPE, timeout=90).strip()
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"{' '.join(args)}: {exc.stderr.strip() or exc}") from exc


class FixedArcGame:
    """Count committed actions on one retained official environment."""

    max_actions = 500
    max_batch = 20

    def __init__(self, env, game_id, action_type):
        self.env, self.game_id = env, game_id
        self.action_type = action_type
        self.actions = 0
        self.terminal = "ACTIVE"
        self.last = env.reset()

    def observe(self):
        return {
            "game_id": self.game_id,
            "state": self.last.state.name,
            "levels_completed": self.last.levels_completed,
            "available_actions": list(self.last.available_actions),
            "frame": [f.tolist() if hasattr(f, "tolist") else f for f in self.last.frame],
            "actions_taken": self.actions,
            "actions_remaining": max(0, self.max_actions - self.actions),
            "terminal": self.terminal,
        }

    def act(self, actions):
        if not isinstance(actions, list) or not 1 <= len(actions) <= self.max_batch:
            raise ValueError("act takes 1..20 actions")
        if self.terminal != "ACTIVE":
            return self.observe()
        before = self.last.levels_completed
        for action in actions:
            name, data = action.get("name"), action.get("data") or {}
            if name not in {"RESET", *[f"ACTION{i}" for i in range(1, 8)]}:
                raise ValueError(f"Bad action: {name}")
            if name == "ACTION6":
                if not all(isinstance(data.get(k), int) and 0 <= data[k] <= 63 for k in ("x", "y")):
                    raise ValueError("ACTION6 needs integer x,y in 0..63")
                self.last = self.env.step(
                    self.action_type[name], data={"x": data["x"], "y": data["y"]}
                )
            elif name == "RESET":
                self.last = self.env.reset()
            else:
                self.last = self.env.step(self.action_type[name])
            if self.last is None:
                raise RuntimeError("Official environment returned no observation after action")
            self.actions += 1
            if self.last.state.name == "GAME_OVER" and self.actions < self.max_actions:
                self.last = self.env.reset()
                self.actions += 1
                break
            if self.last.state.name == "WIN":
                self.terminal = "WIN"
                break
            if self.last.levels_completed != before or self.actions >= self.max_actions:
                break
        if self.actions >= self.max_actions and self.terminal == "ACTIVE":
            self.terminal = "ACTION_CAP"
        return self.observe()


class Official:
    def prepare(self, benchmark, setup, output):
        self.benchmark, self.setup = benchmark, setup
        self.output = Path(output)
        self.options = setup["options"]
        if benchmark not in SOURCES:
            raise ValueError(f"Unsupported benchmark: {benchmark}")
        if not setup["source"]:
            raise ValueError(
                f"Missing official checkout: configure benchmarks.{benchmark}.source ({SOURCES[benchmark]})"
            )
        self.source = Path(setup["source"]).resolve()
        if not (self.source / ".git").exists():
            raise ValueError(f"Missing official Git checkout: {self.source}")
        origin = command("git", "-C", str(self.source), "remote", "get-url", "origin")
        if origin.removesuffix(".git").lower() != SOURCES[benchmark].lower():
            raise ValueError(f"Expected official origin {SOURCES[benchmark]}, found {origin}")
        commit = command("git", "-C", str(self.source), "rev-parse", "HEAD")
        if setup["commit"] and setup["commit"] != commit:
            raise ValueError(
                f"Official checkout commit mismatch: expected {setup['commit']}, found {commit}"
            )
        if command("git", "-C", str(self.source), "status", "--porcelain", "--untracked-files=no"):
            raise ValueError(f"Official evaluator has modified tracked files: {self.source}")
        sys.path.insert(0, str(self.source))
        self.provenance = {
            "benchmark_version": commit,
            "official_source": origin,
            "dataset_environment_version": None,
            "python": sys.version,
            "task_ids": [],
            "starting_state": None,
        }
        self.rows = {}
        import arc_agi
        from arcengine import GameAction

        self.arc_agi, self.GameAction = arc_agi, GameAction
        self.arc_options = dict(self.options)
        mode = self.arc_options.pop("operation_mode", "OFFLINE")
        if mode not in {"OFFLINE", "COMPETITION"}:
            raise ValueError(
                "ARC mode must be OFFLINE or COMPETITION; no silent local/remote fallback"
            )
        environments = self.arc_options.get("environments_dir")
        if mode == "OFFLINE" and (not environments or not Path(environments).is_dir()):
            raise ValueError(
                "Missing ARC-AGI-3 environment files: configure options.environments_dir"
            )
        self.arc_options["operation_mode"] = getattr(arc_agi.OperationMode, mode)
        probe = arc_agi.Arcade(**self.arc_options, recordings_dir=str(self.output / "recordings"))
        self.environment_info = [x.model_dump(mode="json") for x in probe.get_environments()]
        self.rows = {x["game_id"]: x for x in self.environment_info}
        if not self.rows:
            raise ValueError("ARC toolkit returned no official environments")
        self.provenance["dataset_environment_version"] = {
            "environments": self.environment_info,
            "mode": mode,
            "arcengine": importlib.metadata.version("arcengine"),
            "files": {
                str(p.relative_to(environments)): sha(p)
                for p in sorted(Path(environments).rglob("*"))
                if p.is_file() and p.suffix in {".py", ".json"}
            }
            if environments
            else {},
        }
        requested = setup["task_ids"]
        if requested:
            missing = set(requested) - self.rows.keys()
            if missing:
                raise ValueError(f"Unknown official task IDs: {sorted(missing)}")
            self.rows = {key: self.rows[key] for key in requested}
        if not self.rows:
            raise ValueError("Official dataset contains no selected tasks")
        self.provenance["task_ids"] = list(self.rows)
        self.provenance["starting_state"] = self.provenance["dataset_environment_version"]
        return self.provenance

    def start_profile(self, profile, seed):
        self.profile, self.seed = profile, seed
        self.arc = self.arc_agi.Arcade(
            **self.arc_options, recordings_dir=str(self.output / profile / "recordings")
        )
        self.card_id = self.arc.open_scorecard(tags=["buffalo-evaluation", profile])
        return {"scorecard_id": self.card_id}

    def start_task(self, task_id, fixed_game=False):
        self.env = self.arc.make(
            task_id,
            seed=self.seed,
            scorecard_id=self.card_id,
            save_recording=True,
            **({"include_frame_data": True} if fixed_game else {}),
        )
        if self.env is None or self.env.observation_space is None:
            raise RuntimeError(f"ARC toolkit could not initialize {task_id}")
        if fixed_game:
            self.fixed_game = FixedArcGame(self.env, task_id, self.GameAction)
            return {"observation": self.fixed_game.observe()}
        return {
            "messages": [
                {
                    "role": "user",
                    "content": "Explore this unfamiliar interactive environment and complete its levels. "
                    "Call benchmark_action with an action name and data (x,y for complex actions). "
                    "Infer the rules from observations.\n" + json.dumps(self.arc_observation()),
                }
            ]
        }

    def arc_observation(self):
        observation = self.env.observation_space.model_dump(mode="json")
        # Preserve every animation pixel without repeatedly spelling out solid backgrounds.
        # This is lossless transport, identical for both harnesses; SDK recordings stay raw.
        frames = []
        for frame in self.env.observation_space.frame:
            rows = frame.tolist()
            runs = []
            for row in rows:
                for pixel in row:
                    if runs and runs[-1][0] == pixel:
                        runs[-1][1] += 1
                    else:
                        runs.append([pixel, 1])
            frames.append({"height": len(rows), "width": len(rows[0]), "runs": runs})
        observation["frame_rle"] = frames
        observation["frame_encoding"] = (
            "All frames, in animation order. Each runs pair is [color, count], "
            "flattened row-major. Expand runs then reshape to (height, width). "
            "Colors are the original ARC palette indices; the last frame is current."
        )
        # Transport session GUIDs differ after a fresh reset; game state and frames do not.
        observation.pop("guid", None)
        return {"observation": observation, "actions": [a.name for a in self.env.action_space]}

    def game_query(self, op, actions=None):
        if op in {"observe", "status"}:
            return self.fixed_game.observe()
        if op == "act":
            return self.fixed_game.act(actions)
        raise ValueError(f"Unknown fixed-game operation: {op}")

    def snapshot_arc(self):
        """Read-only official scoring; never close/reset a live environment."""
        from arc_agi.models import EnvironmentInfo
        from arc_agi.scorecard import EnvironmentScorecard

        card = self.arc.scorecard_manager.get_scorecard(self.card_id, self.arc.arc_api_key)
        if card is None:
            raise RuntimeError("Live official scorecard unavailable")
        frozen = card.model_copy(deep=True)
        score = EnvironmentScorecard.from_scorecard(
            frozen, [EnvironmentInfo.model_validate(info) for info in self.environment_info]
        )
        return {
            "scorecard_id": self.card_id,
            "primary_score": score.score,
            "metric": "RHAE (%)",
            "raw": score.model_dump(mode="json", exclude={"api_key"}),
            "official_card": frozen.model_dump(mode="json", exclude={"api_key"}),
        }

    def action(self, action=None, data=None):
        selected = self.GameAction[action]
        if selected not in self.env.action_space and action != "RESET":
            raise ValueError(f"Action unavailable: {action}")
        obs = self.env.step(selected, data=data or {})
        if obs is None:
            raise RuntimeError("ARC toolkit action returned no observation")
        return self.arc_observation()

    def finish_profile(self):
        raw_card = self.arc.scorecard_manager.get_scorecard(self.card_id, self.arc.arc_api_key)
        card = self.arc.close_scorecard(self.card_id)
        if card is None:
            raise RuntimeError("Official ARC toolkit returned no final scorecard")
        raw = card.model_dump(mode="json")
        return {
            "primary_score": card.score,
            "metric": "RHAE (%)",
            "raw": raw,
            "scorecard_id": self.card_id,
            "official_card": raw_card.model_dump(mode="json", exclude={"api_key"})
            if raw_card
            else None,
        }

    def aggregate_arc(self, cards, task_ids):
        """Combine disjoint official game cards, then let the SDK score the full set."""
        from arc_agi.models import EnvironmentInfo
        from arc_agi.scorecard import EnvironmentScorecard, Scorecard

        merged = {}
        for card in cards:
            for game_id, game in card["cards"].items():
                if game_id in merged:
                    raise ValueError(f"Duplicate ARC game card: {game_id}")
                merged[game_id] = game
        if set(merged) != set(task_ids):
            raise ValueError("ARC scorecard coverage does not match selected official games")
        official = Scorecard.model_validate({"card_id": "parallel-evaluation", "cards": merged})
        scored = EnvironmentScorecard.from_scorecard(
            official, [EnvironmentInfo.model_validate(info) for info in self.environment_info]
        )
        return {
            "primary_score": scored.score,
            "metric": "RHAE (%)",
            "raw": scored.model_dump(mode="json", exclude={"api_key"}),
        }


def main():
    global CURRENT_REQUEST_ID
    worker = Official()
    for line in sys.stdin:
        operation = None
        try:
            request = json.loads(line)
            CURRENT_REQUEST_ID = request.pop("request_id", None)
            operation = request.pop("operation")
            if operation not in {
                "prepare",
                "start_profile",
                "start_task",
                "action",
                "finish_profile",
                "aggregate_arc",
                "arc_observation",
                "game_query",
                "snapshot_arc",
            }:
                raise ValueError(f"Unknown operation: {operation}")
            with contextlib.redirect_stdout(sys.stderr):
                result = getattr(worker, operation)(**request)
            emit({"result": result})
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            if operation == "action" and isinstance(exc, (ValueError, KeyError)):
                emit({"result": {"error": f"{type(exc).__name__}: {exc}"}})
            else:
                emit({"error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    main()
