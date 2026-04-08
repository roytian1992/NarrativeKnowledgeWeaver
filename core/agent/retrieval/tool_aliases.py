from __future__ import annotations

from typing import Dict


MEMORY_TOOL_ALIASES: Dict[str, str] = {
    "retrieve_scenes_by_entity": "get_entity_sections",
    "retrieve_scene_by_entity": "get_entity_sections",
    "retrieve_sections_by_entity": "get_entity_sections",
    "retrieve_scenes_by_character": "get_entity_sections",
    "retrieve_scenes_by_object": "get_entity_sections",
    "retrieve_scene_titles_by_document_ids": "lookup_titles_by_document_ids",
}


def normalize_memory_tool_name(tool_name: str) -> str:
    name = str(tool_name or "").strip()
    if not name:
        return ""
    return MEMORY_TOOL_ALIASES.get(name, name)
