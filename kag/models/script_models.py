"""
剧本特有的数据模型
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import re

from .entities import Entity, Relation # , EntityType, RelationType


class SceneMetadata(BaseModel):
    """场景元数据"""
    scene_id: str = Field(description="场景ID")
    scene_number: Optional[str] = Field(default=None, description="场景序号")
    sub_scene_number: Optional[str] = Field(default=None, description="子场景序号")
    scene_type: Optional[str] = Field(default=None, description="场景类型(INT/EXT)")
    time_of_day: Optional[str] = Field(default=None, description="时间(日/夜)")
    environment: Optional[str] = Field(default=None, description="宏观环境")
    location: Optional[str] = Field(default=None, description="地点")
    sub_location: Optional[str] = Field(default=None, description="具体位置")
    is_special_scene: Optional[bool] = Field(default=False, description="是否特殊场景")

    @classmethod
    def from_data(cls, scene_id: str, data: Dict[str, Any]) -> 'SceneMetadata':
        """从已提供数据字典构建SceneMetadata对象"""
        meta_data = data.get("meta_data", {})
        return cls(
            scene_id=scene_id,
            scene_number=data.get("scene_number"),
            sub_scene_number=data.get("sub_scene_number"),
            scene_type=meta_data.get("scene_type"),
            time_of_day=meta_data.get("time_of_day"),
            environment=meta_data.get("environment"),
            location=meta_data.get("location"),
            sub_location=meta_data.get("sub_location"),
            is_special_scene=meta_data.get("is_special_scene", False)
        )


class DialogueData(BaseModel):
    """对话数据"""
    dialogue_id: str = Field(description="对话ID")
    character: str = Field(description="角色名称")
    content: str = Field(description="对话内容")
    dialogue_type: Optional[str] = Field(default=None, description="对话类型(VO/OS等)")
    remarks: List[str] = Field(default_factory=list, description="备注信息")
    scene_id: Optional[str] = Field(default=None, description="所属场景ID")


class ScriptDocument(BaseModel):
    """剧本文档模型"""
    id: str = Field(description="文档ID")
    content: str = Field(description="完整内容")
    scene_name: str = Field(description="场景名称")
    sub_scene_name: Optional[str] = Field(default=None, description="子场景名称")
    conversations: List[DialogueData] = Field(default_factory=list, description="对话列表")
    scene_metadata: Optional[SceneMetadata] = Field(default=None, description="场景元数据")

    @classmethod
    def from_script_data(cls, script_data: Dict[str, Any]) -> 'ScriptDocument':
        """从剧本JSON数据创建文档"""
        doc_id = str(script_data.get('_id', script_data.get('id', '')))
        scene_name = script_data.get('scene_name', '')
        sub_scene_name = script_data.get('sub_scene_name', '')
        content = script_data.get('content', '')

        # 处理对话数据
        conversations = []
        for conv_data in script_data.get('conversation', []):
            dialogue = DialogueData(
                dialogue_id=str(conv_data.get('_id', conv_data.get('id', ''))),
                character=conv_data.get('character', ''),
                content=conv_data.get('content', ''),
                dialogue_type=conv_data.get('type', '').strip(),
                remarks=conv_data.get('remark', []),
                scene_id=doc_id
            )
            conversations.append(dialogue)

        # 解析场景元数据，优先使用 sub_scene_name
        parse_target = sub_scene_name.strip() if sub_scene_name and sub_scene_name.strip() else scene_name
        scene_metadata = SceneMetadata.from_data(scene_id=doc_id, data=script_data)

        return cls(
            id=doc_id,
            content=content,
            scene_name=scene_name,
            sub_scene_name=sub_scene_name,
            conversations=conversations,
            scene_metadata=scene_metadata
        )


class ScriptContentParser:
    """剧本内容解析器"""

    @staticmethod
    def parse_content_sections(content: str) -> Dict[str, List[str]]:
        """解析剧本内容中的不同部分

        返回:
            {
                'descriptions': [...],  # [描述]部分
                'dialogues': [...],     # [对话]部分  
                'stage_directions': [...] # [舞台提示]部分
            }
        """
        sections = {
            'descriptions': [],
            'dialogues': [],
            'stage_directions': []
        }

        lines = content.split('\n')
        current_section = None
        current_text = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith('[描述]'):
                if current_section and current_text:
                    sections[current_section].append('\n'.join(current_text).strip())
                current_section = 'descriptions'
                current_text = [line[4:]]
            elif line.startswith('[对话]') or line.startswith('[对白]'):
                if current_section and current_text:
                    sections[current_section].append('\n'.join(current_text).strip())
                current_section = 'dialogues'
                current_text = [line[4:]]
            elif line.startswith('[舞台提示]'):
                if current_section and current_text:
                    sections[current_section].append('\n'.join(current_text).strip())
                current_section = 'stage_directions'
                current_text = [line[6:]]
            else:
                if current_section:
                    current_text.append(line)

        # 添加最后一个部分
        if current_section and current_text:
            sections[current_section].append('\n'.join(current_text).strip())

        return sections

    @staticmethod
    def extract_character_mentions(content: str) -> List[str]:
        """从内容中提取角色提及"""
        import re

        # 匹配中文姓名模式
        chinese_name_pattern = r'[\u4e00-\u9fa5]{2,4}(?:先生|女士|老师|医生|教授|博士|科学家)?'
        # 可选：匹配英文角色名
        english_name_pattern = r'[A-Z][A-Za-z0-9()\.]{1,10}'

        matches_cn = re.findall(chinese_name_pattern, content)
        matches_en = re.findall(english_name_pattern, content)

        # 过滤常见词汇
        common_words = {'这个', '那个', '什么', '怎么', '为什么', '可以', '不是', '没有', '知道', '看到', '听到'}

        characters = [
            name for name in (matches_cn + matches_en)
            if name not in common_words and len(name) >= 2
        ]

        return list(set(characters))
