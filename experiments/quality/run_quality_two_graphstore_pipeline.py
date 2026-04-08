from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = REPO_ROOT / "experiments" / "quality" / "runs"
RUNNER_PATH = REPO_ROOT / "experiments" / "quality" / "run_quality_benchmark.py"
DEFAULT_WORKSPACE_ASSET_ROOT = REPO_ROOT / "experiments" / "quality" / "assets" / "article_workspaces"


@dataclass
class StageSpec:
    phase_name: str
    run_name: str
    command: List[str]
    log_path: Path
    report_path: Path


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(path)


def _resolve_path(raw: str, *, default: Optional[Path] = None) -> Path:
    text = str(raw or "").strip()
    if not text:
        if default is None:
            raise ValueError("Expected a path value")
        return default.resolve()
    path = Path(text)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


def _write_config_with_graph_store_override(
    *,
    base_config_path: Path,
    output_path: Path,
    graph_store_path: Path,
) -> Path:
    payload = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid YAML config: {base_config_path}")
    storage = payload.get("storage")
    if not isinstance(storage, dict):
        storage = {}
        payload["storage"] = storage
    storage["graph_store_path"] = str(graph_store_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return output_path


def _command_text(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(item) for item in command)


def _phase_run_root(run_name: str) -> Path:
    return RUNS_ROOT / run_name


def _build_stage_specs(
    *,
    python_bin: Path,
    manifest_path: Path,
    train_config_path: Path,
    eval_config_path: Path,
    run_name: str,
    workspace_asset_root: Path,
    exp1_1pass_parallel_settings: int,
    exp1_5pass_parallel_settings: int,
    exp2_parallel_settings: int,
    online_batch_size: int,
    eval_max_workers: int,
    offline_eval_max_workers: int,
    offline_training_max_workers: int,
    disable_sql_tools: bool,
    train_limit: int,
    eval_limit: int,
) -> List[StageSpec]:
    phase1_run_name = f"{run_name}__phase1_offline_memory_train10"
    phase2_run_name = f"{run_name}__phase2_exp1_eval50_1pass"
    phase3_run_name = f"{run_name}__phase3_exp1_eval50_5pass"
    phase4_run_name = f"{run_name}__phase4_exp2_eval50_1pass"
    shared_workspace = str(workspace_asset_root)
    runner = str(RUNNER_PATH)
    common_limit_args: List[str] = []
    if train_limit > 0:
        common_limit_args.extend(["--train-limit", str(train_limit)])
    if eval_limit > 0:
        common_limit_args.extend(["--eval-limit", str(eval_limit)])
    common_sql_args = ["--disable-sql-tools"] if disable_sql_tools else []

    phase1_command = [
        str(python_bin),
        "-u",
        runner,
        "--config",
        str(train_config_path),
        "--manifest",
        str(manifest_path),
        "--run-name",
        phase1_run_name,
        "--train-only",
        "--workspace-asset-root",
        shared_workspace,
        "--offline-training-max-workers",
        str(int(offline_training_max_workers)),
        *common_limit_args,
        *common_sql_args,
    ]
    phase2_command = [
        str(python_bin),
        "-u",
        runner,
        "--config",
        str(eval_config_path),
        "--manifest",
        str(manifest_path),
        "--run-name",
        phase2_run_name,
        "--settings",
        "no_strategy_agent,traditional_hybrid_rag_bm25",
        "--skip-offline-training",
        "--workspace-asset-root",
        shared_workspace,
        "--setting-repeats",
        "no_strategy_agent=1,traditional_hybrid_rag_bm25=1",
        "--max-parallel-settings",
        str(int(exp1_1pass_parallel_settings)),
        "--offline-eval-max-workers",
        str(int(offline_eval_max_workers)),
        "--skip-completed-settings",
        *common_limit_args,
        *common_sql_args,
    ]
    phase3_command = [
        str(python_bin),
        "-u",
        runner,
        "--config",
        str(eval_config_path),
        "--manifest",
        str(manifest_path),
        "--run-name",
        phase3_run_name,
        "--settings",
        "no_strategy_agent,traditional_hybrid_rag_bm25",
        "--skip-offline-training",
        "--workspace-asset-root",
        shared_workspace,
        "--setting-repeats",
        "no_strategy_agent=5,traditional_hybrid_rag_bm25=5",
        "--max-parallel-settings",
        str(int(exp1_5pass_parallel_settings)),
        "--offline-eval-max-workers",
        str(int(offline_eval_max_workers)),
        "--skip-completed-settings",
        *common_limit_args,
        *common_sql_args,
    ]
    phase4_command = [
        str(python_bin),
        "-u",
        runner,
        "--config",
        str(eval_config_path),
        "--manifest",
        str(manifest_path),
        "--run-name",
        phase4_run_name,
        "--settings",
        "offline_strategy_agent,offline_strategy_subagent,online_strategy_agent,online_strategy_subagent",
        "--skip-offline-training",
        "--offline-runtime-source-dir",
        str(_phase_run_root(phase1_run_name) / "runtime" / "offline"),
        "--workspace-source-root",
        shared_workspace,
        "--setting-repeats",
        "offline_strategy_agent=1,offline_strategy_subagent=1,online_strategy_agent=1,online_strategy_subagent=1",
        "--max-parallel-settings",
        str(int(exp2_parallel_settings)),
        "--eval-max-workers",
        str(int(eval_max_workers)),
        "--offline-eval-max-workers",
        str(int(offline_eval_max_workers)),
        "--online-batch-size",
        str(int(online_batch_size)),
        "--skip-completed-settings",
        *common_limit_args,
        *common_sql_args,
    ]
    return [
        StageSpec(
            phase_name="phase1_offline_memory_train10",
            run_name=phase1_run_name,
            command=phase1_command,
            log_path=_phase_run_root(phase1_run_name) / "logs" / "runner.log",
            report_path=_phase_run_root(phase1_run_name) / "reports" / "quality_benchmark_results.json",
        ),
        StageSpec(
            phase_name="phase2_exp1_eval50_1pass",
            run_name=phase2_run_name,
            command=phase2_command,
            log_path=_phase_run_root(phase2_run_name) / "logs" / "runner.log",
            report_path=_phase_run_root(phase2_run_name) / "reports" / "quality_benchmark_results.json",
        ),
        StageSpec(
            phase_name="phase3_exp1_eval50_5pass",
            run_name=phase3_run_name,
            command=phase3_command,
            log_path=_phase_run_root(phase3_run_name) / "logs" / "runner.log",
            report_path=_phase_run_root(phase3_run_name) / "reports" / "quality_benchmark_results.json",
        ),
        StageSpec(
            phase_name="phase4_exp2_eval50_1pass",
            run_name=phase4_run_name,
            command=phase4_command,
            log_path=_phase_run_root(phase4_run_name) / "logs" / "runner.log",
            report_path=_phase_run_root(phase4_run_name) / "reports" / "quality_benchmark_results.json",
        ),
    ]


def _launch_stage(stage: StageSpec) -> subprocess.Popen[str]:
    stage.log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = stage.log_path.open("a", encoding="utf-8")
    return subprocess.Popen(
        stage.command,
        cwd=str(REPO_ROOT),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _wait_for_parallel_stages(
    *,
    stages: Sequence[StageSpec],
    pipeline_status_path: Path,
    pipeline_state: Dict[str, Any],
) -> None:
    active: Dict[str, Dict[str, Any]] = {}
    for stage in stages:
        process = _launch_stage(stage)
        active[stage.phase_name] = {"stage": stage, "process": process}
        pipeline_state["phases"][stage.phase_name].update(
            {
                "status": "running",
                "pid": process.pid,
                "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
    _write_json_atomic(pipeline_status_path, pipeline_state)

    while active:
        finished: List[str] = []
        for phase_name, payload in active.items():
            process: subprocess.Popen[str] = payload["process"]
            stage: StageSpec = payload["stage"]
            return_code = process.poll()
            if return_code is None:
                continue
            finished.append(phase_name)
            pipeline_state["phases"][phase_name].update(
                {
                    "status": "completed" if return_code == 0 else "failed",
                    "return_code": int(return_code),
                    "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
            _write_json_atomic(pipeline_status_path, pipeline_state)
            if return_code != 0:
                for sibling_name, sibling_payload in active.items():
                    if sibling_name == phase_name:
                        continue
                    sibling_process: subprocess.Popen[str] = sibling_payload["process"]
                    if sibling_process.poll() is None:
                        sibling_process.terminate()
                        pipeline_state["phases"][sibling_name].update(
                            {
                                "status": "terminated",
                                "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                            }
                        )
                _write_json_atomic(pipeline_status_path, pipeline_state)
                raise RuntimeError(f"{phase_name} failed. See log: {stage.log_path}")
        for phase_name in finished:
            active.pop(phase_name, None)
        if active:
            time.sleep(5)


def _wait_for_single_stage(
    *,
    stage: StageSpec,
    pipeline_status_path: Path,
    pipeline_state: Dict[str, Any],
) -> None:
    process = _launch_stage(stage)
    pipeline_state["phases"][stage.phase_name].update(
        {
            "status": "running",
            "pid": process.pid,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    _write_json_atomic(pipeline_status_path, pipeline_state)
    return_code = process.wait()
    pipeline_state["phases"][stage.phase_name].update(
        {
            "status": "completed" if return_code == 0 else "failed",
            "return_code": int(return_code),
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    _write_json_atomic(pipeline_status_path, pipeline_state)
    if return_code != 0:
        raise RuntimeError(f"{stage.phase_name} failed. See log: {stage.log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the staged QUALITY pipeline with separate local NetworkX graph stores for train/eval."
    )
    parser.add_argument("--config", default="configs/config_openai_quality_stable.yaml")
    parser.add_argument("--manifest", default="experiments/quality/artifacts/split_manifest_train10_eval50_seed20260318.json")
    parser.add_argument("--run-name", default=f"quality_two_graphstores_pipeline_{time.strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--workspace-asset-root", default=str(DEFAULT_WORKSPACE_ASSET_ROOT))
    parser.add_argument("--graph-root", default="")
    parser.add_argument("--train-graph-store", default="")
    parser.add_argument("--eval-graph-store", default="")
    parser.add_argument("--exp1-1pass-max-parallel-settings", type=int, default=2)
    parser.add_argument("--exp1-5pass-max-parallel-settings", type=int, default=2)
    parser.add_argument("--exp2-max-parallel-settings", type=int, default=2)
    parser.add_argument("--online-batch-size", type=int, default=3)
    parser.add_argument("--eval-max-workers", type=int, default=4)
    parser.add_argument("--offline-eval-max-workers", type=int, default=5)
    parser.add_argument("--offline-training-max-workers", type=int, default=4)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument("--disable-sql-tools", action="store_true")
    args = parser.parse_args()

    base_config_path = _resolve_path(args.config)
    manifest_path = _resolve_path(args.manifest)
    python_bin = _resolve_path(args.python_bin)
    workspace_asset_root = _resolve_path(args.workspace_asset_root)

    pipeline_root = RUNS_ROOT / args.run_name
    config_dir = pipeline_root / "generated_configs"
    local_graph_root = _resolve_path(args.graph_root, default=pipeline_root / "local_graphs")
    train_graph_store = _resolve_path(
        args.train_graph_store,
        default=local_graph_root / "train" / "graph_runtime.pkl",
    )
    eval_graph_store = _resolve_path(
        args.eval_graph_store,
        default=local_graph_root / "eval" / "graph_runtime.pkl",
    )

    pipeline_status_path = pipeline_root / "pipeline_status.json"
    command_manifest_path = pipeline_root / "commands.json"
    pipeline_root.mkdir(parents=True, exist_ok=True)

    train_config_path = _write_config_with_graph_store_override(
        base_config_path=base_config_path,
        output_path=config_dir / "config_train_graph.yaml",
        graph_store_path=train_graph_store,
    )
    eval_config_path = _write_config_with_graph_store_override(
        base_config_path=base_config_path,
        output_path=config_dir / "config_eval_graph.yaml",
        graph_store_path=eval_graph_store,
    )

    stage_specs = _build_stage_specs(
        python_bin=python_bin,
        manifest_path=manifest_path,
        train_config_path=train_config_path,
        eval_config_path=eval_config_path,
        run_name=args.run_name,
        workspace_asset_root=workspace_asset_root,
        exp1_1pass_parallel_settings=args.exp1_1pass_max_parallel_settings,
        exp1_5pass_parallel_settings=args.exp1_5pass_max_parallel_settings,
        exp2_parallel_settings=args.exp2_max_parallel_settings,
        online_batch_size=args.online_batch_size,
        eval_max_workers=args.eval_max_workers,
        offline_eval_max_workers=args.offline_eval_max_workers,
        offline_training_max_workers=args.offline_training_max_workers,
        disable_sql_tools=bool(args.disable_sql_tools),
        train_limit=args.train_limit,
        eval_limit=args.eval_limit,
    )

    pipeline_state: Dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_name": args.run_name,
        "status": "preparing",
        "manifest": str(manifest_path),
        "workspace_asset_root": str(workspace_asset_root),
        "train_config_path": str(train_config_path),
        "eval_config_path": str(eval_config_path),
        "train_graph_store": str(train_graph_store),
        "eval_graph_store": str(eval_graph_store),
        "phases": {
            stage.phase_name: {
                "run_name": stage.run_name,
                "command": _command_text(stage.command),
                "log_path": str(stage.log_path),
                "report_path": str(stage.report_path),
                "status": "pending",
            }
            for stage in stage_specs
        },
    }

    try:
        pipeline_state["status"] = "running"
        _write_json_atomic(pipeline_status_path, pipeline_state)
        _write_json_atomic(
            command_manifest_path,
            {"phase_commands": {stage.phase_name: stage.command for stage in stage_specs}},
        )

        phase1, phase2, phase3, phase4 = stage_specs
        _wait_for_parallel_stages(
            stages=[phase1, phase2],
            pipeline_status_path=pipeline_status_path,
            pipeline_state=pipeline_state,
        )
        _wait_for_parallel_stages(
            stages=[phase3, phase4],
            pipeline_status_path=pipeline_status_path,
            pipeline_state=pipeline_state,
        )

        pipeline_state["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        pipeline_state["status"] = "completed"
        _write_json_atomic(pipeline_status_path, pipeline_state)
        print(
            json.dumps(
                {
                    "pipeline_status": str(pipeline_status_path),
                    "phase_reports": {stage.phase_name: str(stage.report_path) for stage in stage_specs},
                },
                ensure_ascii=False,
            )
        )
    except Exception as exc:
        pipeline_state["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        pipeline_state["status"] = "failed"
        pipeline_state["error"] = str(exc)
        _write_json_atomic(pipeline_status_path, pipeline_state)
        raise


if __name__ == "__main__":
    main()
