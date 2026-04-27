# functions/__init__.py
"""
Expose tool classes so they can be imported directly via:
    from functions import EntityRetrieverName, VDBDocsSearchTool, ...
"""

# Graph DB tools
from .graphdb_tools import (
    EntityRetrieverName,
    EntityRetrieverID,
    SearchCommunities,
    SearchEpisodes,
    SearchStorylines,
    SearchSections,
    SearchRelatedEntities,
    GetEntitySections,
    GetRelationsBetweenEntities,
    GetRelationSummary,
    GetCommonNeighbors,
    QuerySimilarEntities,
    QuerySimilarFacts,
    FindPathsBetweenNodes,
    TopKByCentrality,
    GetCoSectionEntities,
    GetKHopSubgraph,
)

# Vector DB tools
from .vectordb_tools import (
    VDBDocsSearchTool,
    VDBGetDocsByDocumentIDsTool,
    VDBSentencesSearchTool,
)

from .sqldb_tools import (
    SQLSearchDialogues,
    SQLSearchInteractions,
    SQLGetInteractionsByDocumentIDs,
)

from .composite_tools import (
    CommunityGraphRAGSearch,
    NarrativeHierarchicalSearch,
    NarrativeCausalTraceSearch,
    HybridEvidenceSearch,
    SectionEvidenceSearch,
    ChoiceGroundedEvidenceSearch,
    EntityEventTraceSearch,
)

from .native_tools import (
    BM25SearchDocsTool,
    LookupTitlesByDocumentIDsTool,
    LookupDocumentIDsByTitleTool,
    SearchRelatedContentTool,
)

__all__ = [
    # graphdb_tools
    "EntityRetrieverName",
    "EntityRetrieverID",
    "SearchCommunities",
    "SearchEpisodes",
    "SearchStorylines",
    "SearchSections",
    "SearchRelatedEntities",
    "GetEntitySections",
    "GetRelationsBetweenEntities",
    "GetRelationSummary",
    "GetCommonNeighbors",
    "QuerySimilarEntities",
    "QuerySimilarFacts",
    "FindPathsBetweenNodes",
    "TopKByCentrality",
    'GetCoSectionEntities',
    'GetKHopSubgraph',
    # vectordb_tools
    "VDBDocsSearchTool",
    "VDBGetDocsByDocumentIDsTool",
    "VDBSentencesSearchTool",
    # sqldb_tools
    "SQLSearchDialogues",
    "SQLSearchInteractions",
    "SQLGetInteractionsByDocumentIDs",
    "CommunityGraphRAGSearch",
    "NarrativeHierarchicalSearch",
    "NarrativeCausalTraceSearch",
    "HybridEvidenceSearch",
    "SectionEvidenceSearch",
    "ChoiceGroundedEvidenceSearch",
    "EntityEventTraceSearch",
    # keyword search tools
    "BM25SearchDocsTool",
    "LookupTitlesByDocumentIDsTool",
    "LookupDocumentIDsByTitleTool",
    "SearchRelatedContentTool",
]
