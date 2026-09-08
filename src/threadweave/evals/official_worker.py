"""Run with the benchmark's Python, not Buffalo's dependency environment.

Only official data, formatters, scorers and interactive environments are loaded.
stdout is a private protocol; all upstream logs go to stderr. Gold data stays here.
"""

from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib.metadata
import json
import os
import re
import runpy
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

WIRE = sys.stdout
CURRENT_REQUEST_ID = None
SOURCES = {
    "manyih-coding": "https://github.com/JHU-CLSP/ManyIH",
    "manyih-if": "https://github.com/JHU-CLSP/ManyIH",
    "longbench-v2": "https://github.com/THUDM/LongBench",
    "arc-agi-3": "https://github.com/arcprize/arc-agi",
    "factorio": "https://github.com/JackHopkins/factorio-learning-environment",
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


def check_docker():
    if not shutil.which("docker"):
        raise ValueError("Missing Docker executable required by headless Factorio setup")
    result = subprocess.run(
        ["docker", "info", "--format", "{{.ServerVersion}}"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    # Some Docker CLI versions return zero with an empty formatted response when disconnected.
    if result.returncode or not result.stdout.strip():
        raise ValueError(
            "Docker daemon unavailable: "
            + (result.stderr.strip() or "docker info returned no server version")
        )
    return result.stdout.strip()


class Judge:
    def generate(self, query, max_tokens=40960, temperature=0.0):
        emit(
            {
                "judge_request": {
                    "prompt": query,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }
            }
        )
        response = json.loads(sys.stdin.readline())
        if "error" in response:
            raise RuntimeError(response["error"])
        return response["text"]


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
        if benchmark.startswith("manyih"):
            subset = "coding" if benchmark == "manyih-coding" else "instruction_following"
            path = self.source / "manyih" / "data" / f"{subset}.json"
            raw = json.loads(path.read_text())
            self.data_config = raw.get("config", {}) if isinstance(raw, dict) else {}
            rows = raw["data"] if isinstance(raw, dict) else raw
            if subset == "coding":
                from manyih.coding import evaluate, evaluator

                self.coding, self.coding_evaluator = evaluate, evaluator
            else:
                from manyih.instruction_following import evaluator, model_adapter

                self.if_evaluator = evaluator
                # Only transport changes: the official checker, hierarchy and metrics are untouched.
                model_adapter.create_model = lambda *args, **kwargs: Judge()
            self.provenance["dataset_environment_version"] = {"commit": commit, "sha256": sha(path)}
            counts = Counter(str(row["id"]) for row in rows)
            identities = {
                str(row["id"]) if counts[str(row["id"])] == 1 else f"{row['id']}@{i}": {
                    "official_id": row["id"],
                    "row_index": i,
                }
                for i, row in enumerate(rows)
            }
            self.rows = {key: rows[identity["row_index"]] for key, identity in identities.items()}
            self.provenance["task_identity_map"] = identities
            self.provenance["data_config"] = self.data_config
        elif benchmark == "longbench-v2":
            if not setup["dataset"] or not Path(setup["dataset"]).is_file():
                raise ValueError(
                    "Missing official LongBench v2 dataset JSON; configure dataset, dataset_revision and dataset_sha256"
                )
            path = Path(setup["dataset"])
            if not setup["dataset_revision"] or not setup["dataset_sha256"]:
                raise ValueError(
                    "LongBench v2 requires the official THUDM/LongBench-v2 revision and dataset_sha256"
                )
            if sha(path) != setup["dataset_sha256"]:
                raise ValueError(f"LongBench v2 dataset SHA256 mismatch: {path}")
            rows = json.loads(path.read_text())
            self.rows = {str(row["_id"]): row for row in rows}
            self.provenance["dataset_environment_version"] = {
                "dataset": "THUDM/LongBench-v2",
                "revision": setup["dataset_revision"],
                "sha256": sha(path),
                "split": "train",
            }
            self.template = (self.source / "prompts/0shot.txt").read_text()
            # Compile the exact upstream extractor without loading GPU/inference dependencies.
            tree = ast.parse((self.source / "pred.py").read_text())
            function = next(
                n
                for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "extract_answer"
            )
            namespace = {"re": re}
            exec(
                compile(
                    ast.Module(body=[function], type_ignores=[]),
                    str(self.source / "pred.py"),
                    "exec",
                ),
                namespace,
            )
            self.extract_answer = namespace["extract_answer"]
        elif benchmark == "arc-agi-3":
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
            probe = arc_agi.Arcade(
                **self.arc_options, recordings_dir=str(self.output / "recordings")
            )
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
        else:
            from factorio_rcon import RCONNetworkError
            from fle.commons.models.game_state import GameState
            from fle.env.instance import FactorioInstance

            self.FactorioInstance, self.GameState = FactorioInstance, GameState
            self.connection_errors = (ConnectionError, RCONNetworkError)
            check_docker()
            required = (
                "container",
                "world_save",
                "container_save_path",
                "world_id",
                "factorio_version",
            )
            missing = [key for key in required if not self.options.get(key)]
            if missing:
                raise ValueError("Missing Factorio setup options: " + ", ".join(missing))
            self.world = Path(self.options["world_save"])
            if not self.world.is_file():
                raise ValueError(f"Missing Factorio starting world save: {self.world}")
            self.container = self.options["container"]
            info = json.loads(command("docker", "inspect", self.container))[0]
            if (info["Config"].get("Labels") or {}).get("buffalo.eval") != "true":
                raise ValueError(
                    f"Factorio container {self.container} needs label buffalo.eval=true for exclusive world restoration"
                )
            cmd = (info["Config"].get("Entrypoint") or []) + (info["Config"].get("Cmd") or [])
            if self.options["container_save_path"] not in cmd or "--start-server" not in cmd:
                raise ValueError(
                    "Factorio container must start the configured immutable save with --start-server"
                )
            version = command(
                "docker",
                "exec",
                self.container,
                self.options.get("factorio_binary", "/opt/factorio/bin/x64/factorio"),
                "--version",
            )
            if self.options["factorio_version"] not in version:
                raise ValueError(f"Factorio version mismatch: {version}")
            self.provenance["dataset_environment_version"] = {
                "world_id": self.options["world_id"],
                "world_sha256": sha(self.world),
                "factorio_version": version,
                "docker_image_id": info["Image"],
            }
            self.rows = {self.options["world_id"]: {"id": self.options["world_id"]}}
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

    def task(self, task_id):
        row = self.rows[task_id]
        if self.benchmark == "manyih-coding":
            formatted = self.coding_evaluator.format_datapoint_for_llm(
                row,
                include_system_prompt=True,
                hierarchy_format=self.data_config.get("hierarchy_format", "scalar"),
                annotation_style=self.data_config.get("annotation_style", "inline"),
            )
            return {
                "messages": [
                    {"role": "system", "content": formatted["system_prompt"]},
                    {"role": "user", "content": formatted["user_prompt"]},
                ]
            }
        if self.benchmark == "manyih-if":
            return {"messages": row["input"]}
        if self.benchmark == "longbench-v2":
            prompt = self.template
            for placeholder, field in {
                "$DOC$": "context",
                "$Q$": "question",
                **{f"$C_{c}$": f"choice_{c}" for c in "ABCD"},
            }.items():
                prompt = prompt.replace(placeholder, row[field].strip())
            return {"messages": [{"role": "user", "content": prompt}]}
        raise ValueError("Interactive tasks require start_task")

    def grade(self, task_id, response, previous_grade=None):
        row = self.rows[task_id]
        if self.benchmark == "manyih-coding":
            grade = self.coding.judge_response(
                response, row["test_code"], row["metadata"]["expected_styles"], timeout=3.0
            )
            return {"id": row["id"], "task_id": row["task_id"], "evaluation": grade}
        if self.benchmark == "manyih-if":
            entry = json.loads(json.dumps(row))
            entry["output"] = {"content": response}
            if previous_grade is not None:
                if previous_grade.get("output") != entry["output"]:
                    raise ValueError("Cannot change an agent answer when retrying judge transport")
                previous = previous_grade.get("constraints", [])
                if len(previous) != len(entry["constraints"]):
                    raise ValueError("Previous grade does not match the official constraints")
                for original, prior in zip(entry["constraints"], previous, strict=True):
                    if any(prior.get(k) != v for k, v in original.items()):
                        raise ValueError("Previous grade changed an official constraint")
                entry["constraints"] = json.loads(json.dumps(previous))
                for constraint in entry["constraints"]:
                    if any(
                        d.get("error") and d["error"] != "Empty response"
                        for d in constraint.get("eval_details", [])
                    ):
                        for key in ("score", "eval_details", "llm_output"):
                            constraint.pop(key, None)
                # The upstream evaluator skips the already-scored constraints itself.
            graded = self.if_evaluator._evaluate_single(
                (entry, "buffalo-judge-transport", None, None)
            )
            write(
                self.output
                / f"if-grade-{list(self.rows).index(task_id)}-{self.profile if hasattr(self, 'profile') else 'grading'}.json",
                graded,
            )
            errors = [
                d["error"]
                for c in graded["constraints"]
                for d in c.get("eval_details", [])
                # Upstream deliberately emits null for an empty agent answer (including
                # a task that exhausts its budget). Preserve that official denominator
                # behavior and null count; it is not a missing evaluator dependency.
                if d.get("error") and d["error"] != "Empty response"
            ]
            if errors:
                raise RuntimeError("Official ManyIH IF checker failed: " + "; ".join(errors))
            if any("score" not in c for c in graded["constraints"]):
                raise RuntimeError("Official ManyIH IF evaluator returned incomplete constraints")
            return graded
        pred = self.extract_answer(response.strip())
        return {
            **{key: value for key, value in row.items() if key != "context"},
            "response": response,
            "pred": pred,
            "judge": pred == row["answer"],
        }

    def summarize(self, grades, profile):
        if self.benchmark == "manyih-coding":
            stats = {
                "total": len(grades),
                **{
                    key: sum(bool(r["evaluation"][key]) for r in grades)
                    for key in ("test_passed", "style_passed", "overall_passed")
                },
            }
            path = self.output / f"coding-{profile}.json"
            write(path, {"results": grades})
            analysis = self.coding_evaluator.analyze_results(str(path), save_to_file=True)
            raw = json.loads(path.read_text())
            return {
                "primary_score": analysis["summary"]["overall_passed_pct"],
                "metric": "overall_passed_pct",
                "tasks_passed": stats["overall_passed"],
                "raw": raw,
            }
        if self.benchmark == "manyih-if":
            accuracy = self.if_evaluator.compute_accuracy(grades)
            categories = defaultdict(list)
            for row in grades:
                categories[row["agent_name"]].append(row)
            return {
                "primary_score": accuracy["ISR"]["accuracy"],
                "metric": "ISR",
                "csr": accuracy["CSR"]["accuracy"],
                "isr_counts": accuracy["ISR"],
                "csr_counts": accuracy["CSR"],
                "null_constraints": accuracy["null_constraints"],
                "categories": {
                    key: self.if_evaluator.compute_accuracy(value)
                    for key, value in categories.items()
                },
                "raw": {"accuracy": accuracy, "results": grades},
            }
        directory = self.output / f"longbench-score-{profile}"
        (directory / "results").mkdir(parents=True)
        write(directory / "results/predictions.json", grades)
        previous = Path.cwd()
        try:
            os.chdir(directory)
            runpy.run_path(str(self.source / "result.py"), run_name="__main__")
        except ZeroDivisionError as exc:
            raise ValueError(
                "Official LongBench v2 result.py requires nonempty easy/hard and short/medium/long groups; select all tasks or a subset covering every group"
            ) from exc
        finally:
            os.chdir(previous)
        text = (directory / "result.txt").read_text()
        header, values = [line.split("\t") for line in text.splitlines()]
        scores = {k: float(v) for k, v in zip(header[1:], values[1:], strict=True)}
        domains = defaultdict(list)
        for row in grades:
            domains[row["domain"]].append(row["judge"])
        return {
            "primary_score": scores["Overall"],
            "metric": "Overall (%)",
            "categories": {
                **scores,
                "domains": {key: 100 * sum(v) / len(v) for key, v in domains.items()},
            },
            "raw": {"result_txt": text, "predictions": grades},
        }

    def start_profile(self, profile, seed):
        self.profile, self.seed = profile, seed
        if self.benchmark == "arc-agi-3":
            self.arc = self.arc_agi.Arcade(
                **self.arc_options, recordings_dir=str(self.output / profile / "recordings")
            )
            self.card_id = self.arc.open_scorecard(tags=["buffalo-evaluation", profile])
            return {"scorecard_id": self.card_id}
        return {}

    def start_task(self, task_id):
        if self.benchmark == "arc-agi-3":
            self.env = self.arc.make(
                task_id, seed=self.seed, scorecard_id=self.card_id, save_recording=True
            )
            if self.env is None or self.env.observation_space is None:
                raise RuntimeError(f"ARC toolkit could not initialize {task_id}")
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
        # Reload the SAME complete world save for each comparison, then keep it alive throughout.
        if sha(self.world) != self.provenance["dataset_environment_version"]["world_sha256"]:
            raise ValueError("Factorio starting world changed after the comparison was pinned")
        command("docker", "stop", self.container)
        command(
            "docker",
            "cp",
            str(self.world),
            f"{self.container}:{self.options['container_save_path']}",
        )
        command("docker", "start", self.container)
        self.instance = None
        error = None
        for _ in range(60):
            try:
                self.instance = self.FactorioInstance(
                    address=self.options.get("address", "localhost"),
                    tcp_port=self.options.get("tcp_port", 27000),
                    fast=True,
                    all_technologies_researched=False,
                    clear_entities=False,
                    inventory=self.options.get("inventory", {}),
                    reset_speed=1,
                    reset_paused=True,
                )
                break
            except self.connection_errors as exc:
                error = exc
                time.sleep(0.5)
        if self.instance is None:
            raise RuntimeError(f"Cannot connect FLE to headless Factorio: {error}")
        self.run_id = f"{self.profile}-{uuid.uuid4().hex}"
        self.steps = 0
        self.initial_research = self.research()
        return {
            "run_id": self.run_id,
            "world_id": self.options["world_id"],
            "initial_research": self.initial_research,
            "messages": [
                {"role": "system", "content": self.instance.get_system_prompt()},
                {
                    "role": "user",
                    "content": "Build a persistent factory and advance research for the entire run budget. "
                    "Use benchmark_action(code=...) to execute Python in the FLE namespace. "
                    "Use the documented player tools; do not edit engine state, grant items or unlock technologies. "
                    "Recover from failures and continue. World progress and Python state persist between calls.",
                },
            ],
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

    def action(self, action=None, data=None, code=None):
        if self.benchmark == "arc-agi-3":
            selected = self.GameAction[action]
            if selected not in self.env.action_space and action != "RESET":
                raise ValueError(f"Action unavailable: {action}")
            obs = self.env.step(selected, data=data or {})
            if obs is None:
                raise RuntimeError("ARC toolkit action returned no observation")
            return self.arc_observation()
        if not code:
            raise ValueError("Factorio requires code")
        self.steps += 1
        self.instance.game_control.unpause()
        try:
            reward, duration, output = self.instance.eval(code, timeout=60)
        finally:
            self.instance.game_control.pause()
        result = {
            "output": output,
            "execution_seconds": duration,
            "research": {k: v for k, v in self.research().items() if k != "raw"},
            "step": self.steps,
        }
        state = self.GameState.from_instance(self.instance)
        checkpoint = self.output / f"{self.profile}-world-state.json"
        checkpoint.write_text(state.to_raw())
        with (self.output / f"{self.profile}-trajectory.jsonl").open("a") as stream:
            stream.write(json.dumps({"code": code, **result}) + "\n")
        return result

    def research(self):
        state = self.instance.first_namespace._save_research_state()
        return {
            "technologies_completed": sum(t.researched for t in state.technologies.values()),
            "current_technology": state.current_research,
            "current_research_progress_pct": 100 * (state.research_progress or 0),
            "raw": asdict(state),
        }

    def finish_profile(self):
        if self.benchmark == "arc-agi-3":
            card = self.arc.close_scorecard(self.card_id)
            if card is None:
                raise RuntimeError("Official ARC toolkit returned no final scorecard")
            raw = card.model_dump(mode="json")
            return {
                "primary_score": card.score,
                "metric": "RHAE (%)",
                "raw": raw,
                "scorecard_id": self.card_id,
            }
        result = self.research()
        checkpoint = self.output / f"{self.profile}-world-state.json"
        checkpoint.write_text(self.GameState.from_instance(self.instance).to_raw())
        self.instance.game_control.pause()
        save_name = f"buffalo-{self.run_id}"
        # FLE injects callable tool implementations into Factorio's persistent Lua storage.
        # Factorio cannot serialize Lua functions. Preserve the complete FLE checkpoint above,
        # then remove only runtime callables before saving the physical world. Clearing the
        # script checksums makes FLE reload these implementations when the save is reopened.
        self.instance.rcon_client.send_command(
            "/c local seen = {}; local function strip(t) "
            "if seen[t] then return end; seen[t] = true; "
            "for k,v in pairs(t) do if type(v) == 'function' then t[k] = nil "
            "elseif type(v) == 'table' then strip(v) end end end; "
            "strip(storage); storage.__lua_script_checksums = {}"
        )
        self.instance.rcon_client.send_command(f"/server-save {save_name}")
        saved_world = self.output / f"{self.profile}-final-world.zip"
        remote = str(Path(self.options["container_save_path"]).parent / f"{save_name}.zip")
        for attempt in range(20):
            try:
                command("docker", "cp", f"{self.container}:{remote}", str(saved_world))
                break
            except RuntimeError:
                if attempt == 19:
                    raise
                time.sleep(0.5)
        return {
            "primary_score": {
                k: result[k] for k in ("technologies_completed", "current_research_progress_pct")
            },
            "metric": "research progress",
            "current_technology": result["current_technology"],
            "world_id": self.options["world_id"],
            "run_id": self.run_id,
            "persistent_world": {
                "checkpoint": str(checkpoint),
                "server_save": save_name,
                "container": self.container,
                "world_save": str(saved_world),
                "sha256": sha(saved_world),
            },
            "raw": {
                "initial_research": self.initial_research,
                "final_research": result,
                "actions": self.steps,
            },
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
                "task",
                "grade",
                "summarize",
                "start_profile",
                "start_task",
                "action",
                "finish_profile",
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
