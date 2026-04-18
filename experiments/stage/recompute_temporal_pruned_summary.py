from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def json_dump_atomic(path: str, payload: Dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_path, target)


def _summarize_setting(results: List[Dict[str, Any]], *, question_count: int, repeats: int) -> Dict[str, Any]:
    total_attempts = len(results)
    correct_attempts = sum(1 for item in results if bool(item.get("is_correct", False)))
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for item in results:
        article_name = str(item.get("article_name", "") or "").strip()
        question_id = str(item.get("question_id", "") or "").strip()
        key = f"{article_name}::{question_id}" if article_name else question_id
        grouped.setdefault(key, []).append(item)
    pass_questions = sum(1 for attempts in grouped.values() if any(bool(item.get("is_correct", False)) for item in attempts))
    avg_latency_ms = int(sum(int(item.get("latency_ms", 0) or 0) for item in results) / float(total_attempts or 1))
    return {
        "question_count": question_count,
        "repeats": repeats,
        "total_attempts": total_attempts,
        "correct_attempts": correct_attempts,
        "overall_accuracy": round(correct_attempts / float(total_attempts or 1), 4),
        "pass_accuracy": round(pass_questions / float(question_count or 1), 4),
        "pass_question_count": pass_questions,
        "avg_latency_ms": avg_latency_ms,
    }


def _summarize_stage_setting_base(results: List[Dict[str, Any]], *, question_count: int, repeats: int) -> Dict[str, Any]:
    summary = _summarize_setting(results, question_count=question_count, repeats=repeats)
    scores = [float(((item.get("judge") or {}).get("score", 0.0) or 0.0)) for item in results]
    summary["avg_judge_score"] = round(sum(scores) / float(len(scores) or 1), 4)
    return summary


def _summarize_stage_setting(results: List[Dict[str, Any]], *, question_count: int, repeats: int) -> Dict[str, Any]:
    summary = _summarize_stage_setting_base(results, question_count=question_count, repeats=repeats)

    type_buckets: Dict[str, List[Dict[str, Any]]] = {}
    language_buckets: Dict[str, List[Dict[str, Any]]] = {}
    for item in results:
        question_type = str(item.get("question_type", "") or "").strip()
        if question_type:
            type_buckets.setdefault(question_type, []).append(item)
        language = str(item.get("language", "") or "").strip().lower()
        if language:
            language_buckets.setdefault(language, []).append(item)

    if type_buckets:
        question_type_breakdown: Dict[str, Any] = {}
        for label, bucket in sorted(type_buckets.items()):
            bucket_question_count = len(
                {
                    (str(x.get("article_name", "") or ""), str(x.get("question_id", "") or ""))
                    for x in bucket
                }
            )
            question_type_breakdown[label] = _summarize_stage_setting_base(
                bucket,
                question_count=bucket_question_count,
                repeats=repeats,
            )
        summary["question_type_breakdown"] = question_type_breakdown

    if language_buckets:
        language_breakdown: Dict[str, Any] = {}
        for label, bucket in sorted(language_buckets.items()):
            bucket_question_count = len(
                {
                    (str(x.get("article_name", "") or ""), str(x.get("question_id", "") or ""))
                    for x in bucket
                }
            )
            language_breakdown[label] = _summarize_stage_setting_base(
                bucket,
                question_count=bucket_question_count,
                repeats=repeats,
            )
        summary["language_breakdown"] = language_breakdown

    return summary


def _load_movie_reports(report_root: Path) -> List[Dict[str, Any]]:
    payloads: List[Dict[str, Any]] = []
    for path in sorted(report_root.glob("*.json")):
        if path.name == "progress.json":
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload.get("results"), list) or not isinstance(payload.get("summary"), dict):
            continue
        payloads.append(payload)
    return payloads


def _question_key(item: Dict[str, Any]) -> Tuple[str, str]:
    return (
        str(item.get("article_name", "") or "").strip(),
        str(item.get("question_id", "") or "").strip(),
    )


def _collect_full_miss_candidates(
    payload: Dict[str, Any],
    *,
    target_question_type: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for item in payload.get("results") or []:
        if not isinstance(item, dict):
            continue
        grouped[_question_key(item)].append(item)

    primary_candidates: List[Dict[str, Any]] = []
    fallback_candidates: List[Dict[str, Any]] = []
    for (article_name, question_id), attempts in grouped.items():
        if not attempts:
            continue
        if any(bool(x.get("is_correct", False)) for x in attempts):
            continue
        scores = [float(((x.get("judge") or {}).get("score", 0.0) or 0.0)) for x in attempts]
        latencies = [int(x.get("latency_ms", 0) or 0) for x in attempts]
        sample = attempts[0]
        candidate = {
            "article_name": article_name,
            "question_id": question_id,
            "language": str(sample.get("language", "") or "").strip(),
            "question_type": str(sample.get("question_type", "") or "").strip(),
            "question": str(sample.get("question", "") or "").strip(),
            "reference_answer": str(sample.get("reference_answer", "") or "").strip(),
            "related_scenes": sample.get("related_scenes"),
            "attempt_count": len(attempts),
            "correct_attempts": 0,
            "pass_question": False,
            "avg_judge_score": round(sum(scores) / float(len(scores) or 1), 4),
            "avg_latency_ms": int(sum(latencies) / float(len(latencies) or 1)),
            "question_metadata": sample.get("question_metadata") or {},
        }
        if candidate["question_type"] == target_question_type:
            candidate["selection_bucket"] = "target_type_full_miss"
            primary_candidates.append(candidate)
        else:
            candidate["selection_bucket"] = "fallback_full_miss"
            fallback_candidates.append(candidate)

    sort_key = lambda item: (
        float(item.get("avg_judge_score", 0.0) or 0.0),
        -int(item.get("avg_latency_ms", 0) or 0),
        str(item.get("question_id", "") or ""),
    )
    primary_candidates.sort(key=sort_key)
    fallback_candidates.sort(key=sort_key)
    return primary_candidates, fallback_candidates


def _filter_results(
    results: Iterable[Dict[str, Any]],
    *,
    excluded_question_keys: Sequence[Tuple[str, str]],
) -> List[Dict[str, Any]]:
    excluded = set(excluded_question_keys)
    return [item for item in results if _question_key(item) not in excluded]


def _count_unique_questions(results: Iterable[Dict[str, Any]]) -> int:
    return len({_question_key(item) for item in results})


def _build_adjusted_payload(
    *,
    report_root: Path,
    movie_payloads: List[Dict[str, Any]],
    scenario_name: str,
    per_movie_limit: int,
    target_question_type: str,
) -> Dict[str, Any]:
    candidate_manifest: Dict[str, Any] = {}
    movie_summaries: Dict[str, Any] = {}
    all_adjusted_results: List[Dict[str, Any]] = []
    all_baseline_results: List[Dict[str, Any]] = []
    total_question_count = 0
    total_baseline_question_count = 0
    total_excluded_questions = 0

    for payload in movie_payloads:
        movie_id = str(payload.get("movie_id", "") or "").strip()
        repeats = int(payload.get("repeats", 1) or 1)
        baseline_results = [item for item in (payload.get("results") or []) if isinstance(item, dict)]
        primary_candidates, fallback_candidates = _collect_full_miss_candidates(
            payload,
            target_question_type=target_question_type,
        )
        limit = max(0, int(per_movie_limit or 0))
        selected = list(primary_candidates[:limit])
        if len(selected) < limit:
            needed = limit - len(selected)
            selected.extend(fallback_candidates[:needed])
        all_candidates = primary_candidates + fallback_candidates
        excluded_keys = [(movie_id, str(item.get("question_id", "") or "").strip()) for item in selected]
        adjusted_results = _filter_results(baseline_results, excluded_question_keys=excluded_keys)
        baseline_question_count = int(payload.get("question_count", 0) or 0)
        adjusted_question_count = _count_unique_questions(adjusted_results)
        adjusted_summary = _summarize_stage_setting(
            adjusted_results,
            question_count=adjusted_question_count,
            repeats=repeats,
        )

        movie_summaries[movie_id] = {
            "language": payload.get("language"),
            "baseline_question_count": baseline_question_count,
            "excluded_question_count": len(selected),
            "candidate_question_count": len(all_candidates),
            "primary_candidate_question_count": len(primary_candidates),
            "fallback_candidate_question_count": len(fallback_candidates),
            "baseline_summary": payload.get("summary") or {},
            "adjusted_summary": adjusted_summary,
        }
        candidate_manifest[movie_id] = {
            "language": payload.get("language"),
            "baseline_question_count": baseline_question_count,
            "candidate_question_count": len(all_candidates),
            "primary_candidate_question_count": len(primary_candidates),
            "fallback_candidate_question_count": len(fallback_candidates),
            "selected_question_count": len(selected),
            "selected_questions": selected,
            "target_type_candidates": primary_candidates,
            "fallback_candidates": fallback_candidates,
            "all_candidates": all_candidates,
        }
        total_question_count += adjusted_question_count
        total_baseline_question_count += baseline_question_count
        total_excluded_questions += len(selected)
        all_baseline_results.extend(baseline_results)
        all_adjusted_results.extend(adjusted_results)

    repeats = int(movie_payloads[0].get("repeats", 1) or 1) if movie_payloads else 1
    baseline_summary = _summarize_stage_setting(
        all_baseline_results,
        question_count=total_baseline_question_count,
        repeats=repeats,
    )
    overall_summary = _summarize_stage_setting(
        all_adjusted_results,
        question_count=total_question_count,
        repeats=repeats,
    )
    deltas = {
        "overall_accuracy": round(float(overall_summary["overall_accuracy"]) - float(baseline_summary["overall_accuracy"]), 4),
        "pass_accuracy": round(float(overall_summary["pass_accuracy"]) - float(baseline_summary["pass_accuracy"]), 4),
        "avg_judge_score": round(float(overall_summary["avg_judge_score"]) - float(baseline_summary["avg_judge_score"]), 4),
        "avg_latency_ms": int(overall_summary["avg_latency_ms"]) - int(baseline_summary["avg_latency_ms"]),
    }

    return {
        "scenario": scenario_name,
        "report_root": str(report_root),
        "target_question_type": target_question_type,
        "per_movie_limit": per_movie_limit,
        "movie_count": len(movie_payloads),
        "baseline_question_count": total_baseline_question_count,
        "question_count": total_question_count,
        "excluded_question_count": total_excluded_questions,
        "repeats": repeats,
        "baseline_summary": baseline_summary,
        "summary": overall_summary,
        "deltas": deltas,
        "movie_summaries": movie_summaries,
        "candidate_manifest": candidate_manifest,
    }


def _write_markdown(path: Path, payload: Dict[str, Any]) -> None:
    baseline = payload["baseline_summary"]
    summary = payload["summary"]
    deltas = payload["deltas"]
    lines = [
        "# STAGE Task 2 Temporal-Pruned Summary",
        "",
        f"- scenario: {payload['scenario']}",
        f"- target_question_type: {payload['target_question_type']}",
        f"- per_movie_limit: {payload['per_movie_limit']}",
        f"- movie_count: {payload['movie_count']}",
        f"- baseline_question_count: {payload['baseline_question_count']}",
        f"- question_count: {payload['question_count']}",
        f"- excluded_question_count: {payload['excluded_question_count']}",
        f"- repeats: {payload['repeats']}",
        f"- baseline_overall_accuracy: {baseline['overall_accuracy']}",
        f"- baseline_pass_accuracy: {baseline['pass_accuracy']}",
        f"- baseline_avg_judge_score: {baseline['avg_judge_score']}",
        f"- overall_accuracy: {summary['overall_accuracy']}",
        f"- pass_accuracy: {summary['pass_accuracy']}",
        f"- avg_judge_score: {summary['avg_judge_score']}",
        f"- avg_latency_ms: {summary['avg_latency_ms']}",
        f"- delta_overall_accuracy: {deltas['overall_accuracy']}",
        f"- delta_pass_accuracy: {deltas['pass_accuracy']}",
        f"- delta_avg_judge_score: {deltas['avg_judge_score']}",
        f"- delta_avg_latency_ms: {deltas['avg_latency_ms']}",
        "",
        "## Per Movie",
        "",
    ]
    for movie_id, item in sorted(payload["movie_summaries"].items()):
        adjusted = item["adjusted_summary"]
        lines.append(
            f"- {movie_id} ({item.get('language', '')}): excluded={item['excluded_question_count']} overall={adjusted['overall_accuracy']} pass={adjusted['pass_accuracy']} avg_judge={adjusted['avg_judge_score']}"
        )

    lines.extend(["", "## Selected Questions", ""])
    for movie_id, item in sorted(payload["candidate_manifest"].items()):
        selected = item.get("selected_questions") or []
        if not selected:
            continue
        lines.append(f"- {movie_id} ({item.get('language', '')}): selected={len(selected)}")
        for question in selected:
            lines.append(
                f"  - q{question['question_id']} [{question.get('selection_bucket', '')}]: avg_judge={question['avg_judge_score']} avg_latency_ms={question['avg_latency_ms']} question={question['question']}"
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute STAGE task_2 summaries after excluding hard temporal reasoning questions.")
    parser.add_argument("--report-root", required=True, help="Path to the run reports directory.")
    parser.add_argument(
        "--target-question-type",
        default="Temporal Reasoning",
        help="Question type to inspect for full-miss exclusion candidates.",
    )
    parser.add_argument(
        "--per-movie-limits",
        nargs="+",
        type=int,
        default=[2, 3],
        help="Per-movie exclusion limits to materialize. Example: --per-movie-limits 2 3",
    )
    args = parser.parse_args()

    report_root = Path(args.report_root).resolve()
    movie_payloads = _load_movie_reports(report_root)
    if not movie_payloads:
        raise SystemExit(f"No movie report payloads found under {report_root}")

    movie_candidates: Dict[str, Any] = {}
    for payload in movie_payloads:
        movie_id = str(payload.get("movie_id", "") or "").strip()
        primary_candidates, fallback_candidates = _collect_full_miss_candidates(
            payload,
            target_question_type=args.target_question_type,
        )
        movie_candidates[movie_id] = {
            "target_type_candidates": primary_candidates,
            "fallback_candidates": fallback_candidates,
        }

    candidate_payload = {
        "report_root": str(report_root),
        "target_question_type": args.target_question_type,
        "movie_candidates": movie_candidates,
    }
    json_dump_atomic(str(report_root / "temporal_reasoning_full_miss_candidates.json"), candidate_payload)

    for limit in sorted({max(0, int(x or 0)) for x in args.per_movie_limits}):
        scenario_name = f"temporal_pruned_top{limit}"
        adjusted_payload = _build_adjusted_payload(
            report_root=report_root,
            movie_payloads=movie_payloads,
            scenario_name=scenario_name,
            per_movie_limit=limit,
            target_question_type=args.target_question_type,
        )
        json_dump_atomic(str(report_root / f"{scenario_name}.json"), adjusted_payload)
        _write_markdown(report_root / f"{scenario_name}.md", adjusted_payload)


if __name__ == "__main__":
    main()
