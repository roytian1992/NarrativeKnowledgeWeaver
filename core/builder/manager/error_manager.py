import json
from typing import Dict, Any
from core.utils.config import KAGConfig
import os
from core.utils.prompt_loader import YAMLPromptLoader

from core.functions.error_fixing import (
    EntityErrorFixer,
    RelationErrorFixer,
    RelationDeduper,
    EntityTypeOverrideValidator,
    EntityScopeValidator,
    EntityRenamer,
    EntityDisambiguationJudger,
    RelationConflictResolver,
)
from core.functions.memory_management import (
    DistilledMemoryExtractor,
    extract_distilled_memories_with_guard,
    DistilledMemorySummarizer,
    summarize_distilled_memories_with_guard,
)

class ProblemSolver:
    """
    High-level information extractor facade.

    Wraps multiple tool interfaces (entity, relation, attribute, interaction) and
    exposes them as easy-to-use methods. Supports optional reflection
    for iterative quality improvements.
    """

    def __init__(self, config: KAGConfig, llm):
        self.config = config
        self.llm = llm
        prompt_dir = self.config.global_config.prompt_dir
        self.prompt_loader = YAMLPromptLoader(prompt_dir)
        schema_dir = self.config.global_config.schema_dir
        task_dir = self.config.global_config.task_dir
        entity_schema_path = os.path.join(schema_dir, "default_entity_schema.json")
        relation_schema_path = os.path.join(schema_dir, "default_relation_schema.json")
        entity_task_path = os.path.join(task_dir, "entity_extraction_task.json")
        relation_task_path = os.path.join(task_dir, "relation_extraction_task.json")

        self.entity_error_fixer = EntityErrorFixer(llm=self.llm, prompt_loader=self.prompt_loader, task_path=entity_task_path)
        self.relation_error_fixer = RelationErrorFixer(llm=self.llm, prompt_loader=self.prompt_loader)
        self.relation_deduper = RelationDeduper(llm=self.llm, prompt_loader=self.prompt_loader)
        self.entity_scope_checker = EntityScopeValidator(llm=self.llm, prompt_loader=self.prompt_loader)
        self.entity_type_checker = EntityTypeOverrideValidator(llm=self.llm, prompt_loader=self.prompt_loader)
        self.entity_renamer = EntityRenamer(llm=self.llm, prompt_loader=self.prompt_loader,  task_path=entity_task_path)
        self.entity_merger = EntityDisambiguationJudger(llm=self.llm, prompt_loader=self.prompt_loader)
        self.relation_conflict_resolver = RelationConflictResolver(llm=self.llm, prompt_loader=self.prompt_loader)
        self.distilled_memory_extractor = DistilledMemoryExtractor(llm=self.llm, prompt_loader=self.prompt_loader)
        self.distilled_memory_summarizer = DistilledMemorySummarizer(llm=self.llm, prompt_loader=self.prompt_loader)

    def fix_entity_error(
        self,
        text: str,
        candidate_entity_types: str,
        feedback: str,
    ) -> str:
        params = {
            "text": text,
            "candidate_entity_types": candidate_entity_types,
            "feedback": feedback
        }
        result = self.entity_error_fixer.call(params=json.dumps(params))
        return result
    
  
    def fix_relation_error(
        self,
        text: str,
        extracted_relation: str,
        allowed_relation_types: str,
        feedback: str,
    ) -> str:
        params = {
            "text": text,
            "extracted_relation": extracted_relation,
            "allowed_relation_types": allowed_relation_types,
            "feedback": feedback
        }
        result = self.relation_error_fixer.call(params=json.dumps(params))
        return result
    
    def dedup_relations(
        self,
        relations: str,
    ) -> str:
        """
        Deduplicate multiple relations between the same entity pair.

        Args:
            text: Original scene text.
            relations: string of relations (same subject-object pair).
        Returns:
            JSON string with:
            - decision: "keep" | "drop"
            - output:
                - {}  (if keep)
                - { drop_relation_types: [...], rationale: str } (if drop)
        """
        params = {
            "relations_json": relations,
        }
        result = self.relation_deduper.call(
            params=json.dumps(params),
        )
        return result

    def check_entity_scope(
        self,
        text: str,
        entity_name: str
    ) -> str:
        params = {
            "text": text,
            "entity_name": entity_name,

        }
        result = self.entity_scope_checker.call(
            params=json.dumps(params),
        )
        return result
    

    def check_entity_type(
        self,
        text: str,
        entity_name: str, 
        minority_type: str,
        majority_type: str,
        majority_percentage: str
    ) -> str:
        params = {
            "text": text,
            "entity_name": entity_name,
            "minority_type": minority_type,
            "majority_type": majority_type,
            "majority_percentage": majority_percentage,
        }
        result = self.entity_type_checker.call(
            params=json.dumps(params),
        )
        return result
    
    def rename_entity(
        self,
        text: str,
        entity_name: str, 
        entity_type: str
    ) -> str:
        params = {
            "text": text,
            "entity_name": entity_name,
            "entity_type": entity_type
        }
        result = self.entity_renamer.call(
            params=json.dumps(params),
        )
        return result
    
    def merge_entities(
        self,
        entity_descriptions: str
    ) -> str:
        params = {
            "entity_descriptions": entity_descriptions,
        }
        result = self.entity_merger.call(
            params=json.dumps(params),
        )
        return result

    
    def resolve_relation_conflict(
        self,
        subject_entity: str,
        object_entity: str,
        text: str
    ) -> str:
        params = {
            "subject_entity": subject_entity,
            "object_entity": object_entity,
            "text": text
        }
        result = self.relation_conflict_resolver.call(
            params=json.dumps(params),
        )
        return result

    def extract_distilled_memories(
        self,
        source_payload: str,
        requested_max: int,
    ) -> str:
        return extract_distilled_memories_with_guard(
            llm=self.llm,
            prompt_loader=self.prompt_loader,
            source_payload=source_payload,
            requested_max=int(requested_max),
        )

    def summarize_distilled_memories(
        self,
        grouped_memories_json: str,
        existing_summaries_json: str,
    ) -> str:
        return summarize_distilled_memories_with_guard(
            llm=self.llm,
            prompt_loader=self.prompt_loader,
            grouped_memories_json=grouped_memories_json,
            existing_summaries_json=existing_summaries_json,
        )
