from __future__ import annotations

import copy
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.strategy_training.strategy_runtime_assets import StrategyRuntimeAssetManager
from core.utils.config import KAGConfig
from core.utils.general_utils import json_dump_atomic


class StrategyLibraryRebuilder:
    """
    Deterministic offline rebuild of runtime strategy memory from existing
    training artifacts. This path does not rerun agent rollouts and does not
    require LLM-based template redistillation or reclustering.
    """

    def __init__(
        self,
        *,
        config: KAGConfig,
        training_root: str,
        output_dir_name: str = "rebuild_from_artifacts",
        runtime_library_path: Optional[str] = None,
    ) -> None:
        self.repo_root = Path(__file__).resolve().parents[2]
        self.config = copy.deepcopy(config)
        self.training_root = (self.repo_root / str(training_root or "")).resolve()
        self.output_root = self.training_root / str(output_dir_name or "rebuild_from_artifacts")
        self.runtime_library_path = (
            Path(runtime_library_path).resolve()
            if runtime_library_path
            else (self.repo_root / str(getattr(self.config.strategy_memory, "library_path", "data/memory/strategy/strategy_library.json"))).resolve()
        )
        self.runtime_tool_metadata_dir = (
            self.repo_root / str(getattr(self.config.strategy_memory, "tool_metadata_runtime_dir", "data/memory/strategy/tool_metadata"))
        ).resolve()

        self.question_detail_dir = self.training_root / "distilled" / "per_question"
        self.original_question_summary_path = self.training_root / "distilled" / "question_training_summaries.json"
        self.original_cluster_path = self.training_root / "clusters" / "template_clusters.json"
        self.original_merge_decision_path = self.training_root / "clusters" / "merge_decisions.jsonl"
        self.tool_reflection_record_path = self.training_root / "tool_metadata" / "tool_description_reflection_records.jsonl"
        self.tool_description_candidate_path = self.training_root / "tool_metadata" / "tool_description_candidates.json"

        self.question_summary_path = self.output_root / "distilled" / "question_training_summaries.json"
        self.raw_template_path = self.output_root / "distilled" / "raw_templates.json"
        self.failure_summary_path = self.output_root / "failures" / "failed_question_summaries.json"
        self.cluster_path = self.output_root / "clusters" / "template_clusters.json"
        self.merge_decision_path = self.output_root / "clusters" / "merge_decisions.jsonl"
        self.library_output_path = self.output_root / "library" / "strategy_library.json"
        self.template_source_index_path = self.output_root / "library" / "template_source_index.json"
        self.report_path = self.output_root / "report.md"
        self.rebuild_manifest_path = self.output_root / "manifests" / "rebuild_manifest.json"

        for path in [
            self.question_summary_path,
            self.raw_template_path,
            self.failure_summary_path,
            self.cluster_path,
            self.merge_decision_path,
            self.library_output_path,
            self.template_source_index_path,
            self.report_path,
            self.rebuild_manifest_path,
        ]:
            path.parent.mkdir(parents=True, exist_ok=True)

        self.runtime_asset_manager = StrategyRuntimeAssetManager(
            library_path=str(self.runtime_library_path),
            tool_metadata_runtime_dir=str(self.runtime_tool_metadata_dir),
        )

    @staticmethod
    def _clip_text(text: Any, limit: int = 600) -> str:
        raw = str(text or "").strip()
        if len(raw) <= limit:
            return raw
        return raw[: max(0, limit - 3)] + "..."

    @staticmethod
    def _read_json(path: Path, default: Any) -> Any:
        if not path.exists():
            return default
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        if not path.exists():
            return rows
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                raw = line.strip()
                if not raw:
                    continue
                rows.append(json.loads(raw))
        return rows

    @staticmethod
    def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    @staticmethod
    def _unique(items: List[str], limit: int = 12) -> List[str]:
        out: List[str] = []
        for item in items:
            value = str(item or "").strip()
            if value and value not in out:
                out.append(value)
            if len(out) >= limit:
                break
        return out

    @staticmethod
    def _aggregate_tool_description_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        grouped: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
        for record in records:
            key = (str(record.get("tool_name", "") or ""), str(record.get("language", "") or ""))
            if key[0]:
                grouped[key].append(record)

        aggregated: List[Dict[str, Any]] = []
        for (tool_name, language), items in sorted(grouped.items()):
            current_description = str(items[0].get("current_description", "") or "")
            keep_items = [item for item in items if str(item.get("decision", "") or "").strip().lower() == "keep"]
            revise_items = [
                item
                for item in items
                if str(item.get("decision", "") or "").strip().lower() == "revise"
                and str(item.get("proposed_description", "") or "").strip()
            ]
            if revise_items and len(revise_items) > len(keep_items):
                proposal_counter = Counter(str(item.get("proposed_description", "") or "").strip() for item in revise_items)
                proposed_description, support_count = proposal_counter.most_common(1)[0]
                supporting_items = [item for item in revise_items if str(item.get("proposed_description", "") or "").strip() == proposed_description]
                reason = Counter(str(item.get("reason", "") or "").strip() for item in supporting_items).most_common(1)[0][0]
                decision = "revise"
            else:
                proposed_description = current_description
                support_count = len(keep_items) if keep_items else len(items)
                base_items = keep_items if keep_items else items
                reason = Counter(str(item.get("reason", "") or "").strip() for item in base_items).most_common(1)[0][0]
                decision = "keep"
            aggregated.append(
                {
                    "tool_name": tool_name,
                    "language": language,
                    "current_description": current_description,
                    "decision": decision,
                    "proposed_description": proposed_description,
                    "reason": reason,
                    "support_count": int(support_count),
                    "keep_count": len(keep_items),
                    "revise_count": len(revise_items),
                }
            )
        return aggregated

    def _load_tool_description_candidates(self) -> List[Dict[str, Any]]:
        payload = self._read_json(self.tool_description_candidate_path, [])
        if isinstance(payload, list):
            return payload
        records = self._read_jsonl(self.tool_reflection_record_path)
        return self._aggregate_tool_description_records(records)

    def _derive_anti_patterns(
        self,
        *,
        question: str,
        query_pattern: Dict[str, Any],
        reflection_records: List[Dict[str, Any]],
        attempts: List[Dict[str, Any]],
        existing: List[str],
    ) -> List[str]:
        out = list(existing or [])
        q = str(question or "")
        q_lower = q.lower()
        qabs = str((query_pattern or {}).get("query_abstract", "") or "")
        notes = str((query_pattern or {}).get("notes", "") or "")
        answer_shape = str((query_pattern or {}).get("answer_shape", "") or "").lower()

        reflection_texts: List[str] = []
        for record in reflection_records or []:
            reflection = record.get("reflection") if isinstance(record.get("reflection"), dict) else {}
            reflection_texts.extend(
                [
                    str(reflection.get("failure_reason", "") or ""),
                    str(reflection.get("missed_fact", "") or ""),
                    str(reflection.get("next_action", "") or ""),
                    str(record.get("retry_instruction", "") or ""),
                ]
            )
        reflection_blob = "\n".join(x for x in reflection_texts if x).lower()

        failed_attempts = [item for item in attempts if not bool((item.get("judge") or {}).get("is_correct", False))]
        if failed_attempts:
            out.append("不要只根据实体检索结果或章节列表直接下结论，必须回到原文证据验证关键约束。")

        if "全称" in q or "简称" in q or "缩写" in q or "full name" in q_lower or "formal name" in q_lower:
            out.append("不要把简称、中文别称或泛称当作正式命名，必须从原文中提取明确的全称。")

        if "年龄" in q or "岁" in q or "年龄" in qabs or "年龄" in notes or "age" in q_lower:
            out.append("不要根据时间线或角色经历推断年龄条件，必须查找场景中的显式年龄标注。")
            out.append("不要把角色出现的全部场次都当作满足年龄条件的场次，必须逐条验证关键属性。")

        if ("场次" in q or "场景" in q or answer_shape == "list") and failed_attempts:
            out.append("不要先枚举所有相关场次再做模糊筛选，只有被原文直接支持的场次才能保留。")

        if any(tool in set(item.get("raw_tool_chain") or []) for item in failed_attempts for tool in ["retrieve_entity_by_name", "get_entity_sections"]):
            out.append("不要停留在实体总览或出现列表层面，命中候选后还要补做证据检索或原文回读。")

        if any(x in reflection_blob for x in ["推断", "时间线", "timeline", "infer", "explicit", "显式", "直接证据"]):
            out.append("不要依赖隐含推理替代显式证据；当问题带有限定条件时，必须核对原文中的直接表述。")

        if any(x in reflection_blob for x in ["扩大", "超出", "遗漏关键约束", "only", "唯一", "仅保留"]):
            out.append("不要扩大答案范围；如果问题包含唯一性或限定条件，必须严格按约束过滤。")

        return self._unique(out, limit=12)

    def _update_template_from_result(
        self,
        *,
        result: Dict[str, Any],
        original_template: Dict[str, Any],
    ) -> Dict[str, Any]:
        query_pattern = result.get("query_pattern") if isinstance(result.get("query_pattern"), dict) else {}
        attempts = list(result.get("attempts") or [])
        reflection_records = list(result.get("reflection_records") or [])
        successful_attempt_count = sum(1 for item in attempts if bool((item.get("judge") or {}).get("is_correct", False)))
        attempt_group_count = len({str(item.get("attempt_group_id", "") or "") for item in attempts if str(item.get("attempt_group_id", "") or "")}) or max(
            1,
            int(original_template.get("attempt_count", 1) or 1),
        )
        updated = dict(original_template or {})
        updated["query_pattern"] = query_pattern
        updated["query_abstract"] = str(query_pattern.get("query_abstract", "") or updated.get("query_abstract", ""))
        updated["successful_attempts"] = successful_attempt_count
        updated["support_count"] = successful_attempt_count
        updated["attempt_count"] = attempt_group_count
        updated["success_rate"] = round(float(successful_attempt_count) / float(attempt_group_count), 4)
        updated["anti_patterns"] = self._derive_anti_patterns(
            question=str(result.get("question", "") or ""),
            query_pattern=query_pattern,
            reflection_records=reflection_records,
            attempts=attempts,
            existing=list(updated.get("anti_patterns") or []),
        )
        sources = list(updated.get("template_sources") or [])
        if not any(str(item.get("source_type", "") or "") == "artifact_rebuild" for item in sources if isinstance(item, dict)):
            sources.append(
                {
                    "question_id": str(result.get("question_id", "") or ""),
                    "question": str(result.get("question", "") or ""),
                    "source_type": "artifact_rebuild",
                }
            )
        updated["template_sources"] = sources
        return updated

    def _build_markdown_report(
        self,
        *,
        question_summaries: List[Dict[str, Any]],
        failed_summaries: List[Dict[str, Any]],
        tool_description_candidates: List[Dict[str, Any]],
        cluster_payload: Dict[str, Any],
    ) -> str:
        lines = [
            "# Strategy Library Rebuild Report",
            "",
            f"- Source training root: `{self.training_root}`",
            f"- Rebuilt at: `{time.strftime('%Y-%m-%d %H:%M:%S')}`",
            f"- Successful questions used: `{len(question_summaries)}`",
            f"- Failed questions kept: `{len(failed_summaries)}`",
            f"- Tool description candidates reused: `{len(tool_description_candidates)}`",
            f"- Cluster count reused: `{cluster_payload.get('cluster_count', 0)}`",
            f"- Runtime library: `{self.runtime_library_path}`",
            "",
            "## Notes",
            "",
            "- This rebuild reuses existing training artifacts and cluster assignments.",
            "- It enriches anti-patterns from failed attempts and retry reflections.",
            "- It does not rerun agent rollouts or re-sample trajectories.",
            "",
        ]
        return "\n".join(lines)

    def rebuild(self) -> Dict[str, Any]:
        if not self.question_detail_dir.exists():
            raise FileNotFoundError(f"Missing question detail directory: {self.question_detail_dir}")

        original_question_summaries = self._read_json(self.original_question_summary_path, [])
        original_cluster_payload = self._read_json(self.original_cluster_path, {"cluster_count": 0, "clusters": []})
        original_merge_decisions = self._read_jsonl(self.original_merge_decision_path)
        if not isinstance(original_question_summaries, list):
            raise ValueError(f"Invalid original question summary file: {self.original_question_summary_path}")
        if not isinstance(original_cluster_payload, dict):
            raise ValueError(f"Invalid original cluster payload file: {self.original_cluster_path}")

        original_summary_by_question: Dict[str, Dict[str, Any]] = {
            str(item.get("question_id", "") or ""): item
            for item in original_question_summaries
            if isinstance(item, dict) and str(item.get("question_id", "") or "")
        }
        updated_template_by_id: Dict[str, Dict[str, Any]] = {}
        question_summaries: List[Dict[str, Any]] = []
        failed_summaries: List[Dict[str, Any]] = []

        detail_paths = sorted(self.question_detail_dir.glob("q*.json"), key=lambda p: int(p.stem.lstrip("q") or 0))
        for path in detail_paths:
            result = self._read_json(path, {})
            if not isinstance(result, dict):
                continue
            question_id = str(result.get("question_id", "") or "")
            if bool(result.get("successful")):
                original_summary = original_summary_by_question.get(question_id, {})
                original_template = dict((original_summary.get("template") or result.get("template_seed") or {}))
                if not original_template:
                    continue
                updated_template = self._update_template_from_result(result=result, original_template=original_template)
                updated_template_by_id[str(updated_template.get("template_id", "") or "")] = updated_template
                question_summaries.append(
                    {
                        "question_id": question_id,
                        "question": str(result.get("question", "") or ""),
                        "successful_attempt_count": int(result.get("successful_attempt_count", 0) or 0)
                        if "successful_attempt_count" in result
                        else sum(1 for item in (result.get("attempts") or []) if bool((item.get("judge") or {}).get("is_correct", False))),
                        "best_attempt_id": str(result.get("best_attempt_id", "") or ""),
                        "best_retry_index": int(result.get("best_retry_index", 0) or 0),
                        "best_retry_instruction": str(result.get("best_retry_instruction", "") or ""),
                        "best_effective_tool_chain": list(result.get("best_effective_tool_chain", []) or []),
                        "best_raw_tool_chain": list(result.get("best_raw_tool_chain", []) or []),
                        "best_judge": result.get("best_judge", {}),
                        "query_pattern": result.get("query_pattern", {}),
                        "cluster_id": str(original_summary.get("cluster_id", "") or ""),
                        "template": updated_template,
                    }
                )
            else:
                failure = result.get("failure") if isinstance(result.get("failure"), dict) else {}
                failed_summaries.append(
                    {
                        "question_id": str(failure.get("question_id", question_id) or ""),
                        "question": str(failure.get("question", result.get("question", "")) or ""),
                        "failure_summary": str(failure.get("failure_summary", "") or ""),
                        "likely_causes": list(failure.get("likely_causes") or []),
                        "recommended_improvements": list(failure.get("recommended_improvements") or []),
                    }
                )

        clusters: List[Dict[str, Any]] = []
        for cluster in list(original_cluster_payload.get("clusters") or []):
            if not isinstance(cluster, dict):
                continue
            new_cluster = dict(cluster)
            member_templates = []
            member_template_ids = [str(x or "") for x in (cluster.get("member_template_ids") or []) if str(x or "").strip()]
            for template_id in member_template_ids:
                member_templates.append(copy.deepcopy(updated_template_by_id.get(template_id) or {}))
            member_templates = [item for item in member_templates if item]
            if member_templates:
                anti_patterns = self._unique(
                    list(cluster.get("anti_patterns") or [])
                    + [x for item in member_templates for x in (item.get("anti_patterns") or [])],
                    limit=12,
                )
                source_question_ids = self._unique(
                    list(cluster.get("source_question_ids") or [])
                    + [str(item.get("question_id", "") or "") for item in member_templates],
                    limit=200,
                )
                source_questions = self._unique(
                    list(cluster.get("source_questions") or [])
                    + [str(item.get("question", "") or "") for item in member_templates],
                    limit=200,
                )
                support_count = sum(int(item.get("support_count", 0) or 0) for item in member_templates)
                successful_attempts = sum(int(item.get("successful_attempts", 0) or 0) for item in member_templates)
                attempt_count = sum(max(1, int(item.get("attempt_count", 1) or 1)) for item in member_templates)
                supported_answer_shapes = self._unique(
                    list(cluster.get("supported_answer_shapes") or [])
                    + [str(((item.get("query_pattern") or {}).get("answer_shape", "")) or "") for item in member_templates],
                    limit=20,
                )
                new_cluster["member_templates"] = member_templates
                new_cluster["template_count"] = len(member_templates)
                new_cluster["anti_patterns"] = anti_patterns
                new_cluster["source_question_ids"] = source_question_ids
                new_cluster["source_questions"] = source_questions
                new_cluster["support_count"] = support_count
                new_cluster["successful_attempts"] = successful_attempts
                new_cluster["attempt_count"] = attempt_count
                new_cluster["success_rate"] = round(float(successful_attempts) / float(attempt_count), 4) if attempt_count else 0.0
                new_cluster["supported_answer_shapes"] = [x for x in supported_answer_shapes if x]
            clusters.append(new_cluster)

        cluster_payload = {
            "cluster_count": len(clusters),
            "clusters": clusters,
        }
        raw_templates = [item["template"] for item in question_summaries if isinstance(item.get("template"), dict)]
        template_source_index = {
            str(cluster.get("cluster_id", "") or ""): {
                "template_id": str(cluster.get("cluster_id", "") or ""),
                "pattern_name": str(cluster.get("pattern_name", "") or ""),
                "source_question_ids": list(cluster.get("source_question_ids") or []),
                "source_questions": list(cluster.get("source_questions") or []),
                "member_template_ids": list(cluster.get("member_template_ids") or []),
            }
            for cluster in clusters
            if str(cluster.get("cluster_id", "") or "")
        }

        runtime_templates = []
        for cluster in clusters:
            runtime_templates.append(
                {
                    "template_id": str(cluster.get("cluster_id", "") or ""),
                    "cluster_id": str(cluster.get("cluster_id", "") or ""),
                    "pattern_name": str(cluster.get("pattern_name", "") or ""),
                    "pattern_description": str(cluster.get("pattern_description", "") or ""),
                    "recommended_chain": list(cluster.get("recommended_chain") or []),
                    "anti_patterns": list(cluster.get("anti_patterns") or []),
                    "support_count": int(cluster.get("support_count", 0) or 0),
                    "success_rate": float(cluster.get("success_rate", 0.0) or 0.0),
                    "supported_answer_shapes": list(cluster.get("supported_answer_shapes") or []),
                    "query_pattern": cluster.get("query_pattern_prototype") if isinstance(cluster.get("query_pattern_prototype"), dict) else {},
                    "query_abstract": str(cluster.get("query_abstract", "") or ""),
                    "query_pattern_text": str(cluster.get("query_pattern_text", "") or ""),
                    "pattern_embedding": list(cluster.get("pattern_embedding") or []),
                    "source_question_ids": list(cluster.get("source_question_ids") or []),
                    "source_questions": list(cluster.get("source_questions") or []),
                }
            )

        tool_description_candidates = self._load_tool_description_candidates()
        generated_at = time.strftime("%Y-%m-%d %H:%M:%S")
        training_library_payload = {
            "library_version": 3,
            "aggregation_mode": "narrative",
            "dataset_name": self.training_root.name,
            "generated_at": generated_at,
            "raw_template_count": len(raw_templates),
            "cluster_count": cluster_payload.get("cluster_count", 0),
            "clusters": cluster_payload.get("clusters", []),
            "pattern_count": len(runtime_templates),
            "patterns": runtime_templates,
            "templates": runtime_templates,
        }
        runtime_library_payload = {
            "library_version": 3,
            "aggregation_mode": "narrative",
            "dataset_name": self.training_root.name,
            "generated_at": generated_at,
            "pattern_count": len(runtime_templates),
            "patterns": runtime_templates,
            "template_count": len(runtime_templates),
            "templates": runtime_templates,
        }
        manifest = {
            "training_root": str(self.training_root),
            "question_detail_dir": str(self.question_detail_dir),
            "question_count": len(detail_paths),
            "successful_question_count": len(question_summaries),
            "failed_question_count": len(failed_summaries),
            "runtime_library_path": str(self.runtime_library_path),
            "runtime_tool_metadata_dir": str(self.runtime_tool_metadata_dir),
            "generated_at": generated_at,
            "mode": "deterministic_reuse_existing_clusters",
        }

        json_dump_atomic(str(self.rebuild_manifest_path), manifest)
        json_dump_atomic(str(self.question_summary_path), question_summaries)
        json_dump_atomic(str(self.raw_template_path), raw_templates)
        json_dump_atomic(str(self.failure_summary_path), failed_summaries)
        json_dump_atomic(str(self.cluster_path), cluster_payload)
        self._write_jsonl(self.merge_decision_path, original_merge_decisions)
        json_dump_atomic(str(self.library_output_path), training_library_payload)
        json_dump_atomic(str(self.template_source_index_path), template_source_index)
        json_dump_atomic(str(self.runtime_library_path), runtime_library_payload)
        runtime_tool_metadata_paths = self.runtime_asset_manager.export_tool_metadata_overrides(tool_description_candidates)
        runtime_template_source_index_path = self.runtime_asset_manager.export_template_source_index(template_source_index)
        self.report_path.write_text(
            self._build_markdown_report(
                question_summaries=question_summaries,
                failed_summaries=failed_summaries,
                tool_description_candidates=tool_description_candidates,
                cluster_payload=cluster_payload,
            ),
            encoding="utf-8",
        )
        return {
            "rebuild_manifest_path": str(self.rebuild_manifest_path),
            "question_summary_path": str(self.question_summary_path),
            "raw_template_path": str(self.raw_template_path),
            "failure_summary_path": str(self.failure_summary_path),
            "cluster_path": str(self.cluster_path),
            "merge_decision_path": str(self.merge_decision_path),
            "library_output_path": str(self.library_output_path),
            "template_source_index_path": str(self.template_source_index_path),
            "runtime_library_path": str(self.runtime_library_path),
            "runtime_tool_metadata_paths": runtime_tool_metadata_paths,
            "runtime_template_source_index_path": runtime_template_source_index_path,
            "report_path": str(self.report_path),
            "successful_question_count": len(question_summaries),
            "failed_question_count": len(failed_summaries),
            "cluster_count": cluster_payload.get("cluster_count", 0),
            "raw_template_count": len(raw_templates),
        }
