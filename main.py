#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KAG-Builder main entrypoint

Builds:
1) Knowledge graph (entities/relations, attributes, embeddings)
2) Relational DB (CMP info, scene info)
3) Event/plot graphs (causality, SABER, plot relations, embeddings)

Includes:
- Best-effort resource cleanup for common components
- Diagnostics for non-daemon alive threads (stack dump + short join + hard exit)
"""

import argparse
import sys
from pathlib import Path
import os
import threading
import traceback
import time
import inspect
import json

# Add project root to PYTHONPATH
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core import KAGConfig
from core.builder.graph_builder import KnowledgeGraphBuilder
from core.builder.database_builder import RelationalDatabaseBuilder
from core.builder.narrative_graph_builder import EventCausalityBuilder
from core.builder.supplementary_builder import SupplementaryBuilder

import logging

# ------- Base logging config (may be overridden by --verbose) -------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("neo4j").setLevel(logging.ERROR)
logging.getLogger("neo4j.io").setLevel(logging.ERROR)
logging.getLogger("neo4j.bolt").setLevel(logging.ERROR)
logging.getLogger("core.memory.vector_memory").setLevel(logging.INFO)
logging.getLogger("core.storage.vector_store").setLevel(logging.INFO)
logging.getLogger("core.storage.vector_store").setLevel(logging.INFO)
# logging.getLogger("core.functions.tool_calls.sqldb_tools").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.INFO)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# ================== Resource closing (component level) ==================

def _try_call(fn, **kwargs):
    """Call a function with only the kwargs it accepts (safe best-effort)."""
    if not callable(fn):
        return
    sig = inspect.signature(fn)
    accepted = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return fn(**accepted) if accepted else fn()

def _best_effort_close(obj):
    """
    Best-effort close for common resources; never raises.
    Handles:
      - concurrent.futures executors (with cancel_futures if available)
      - generic .close/.shutdown/.dispose/.stop/.terminate
    """
    if obj is None:
        return
    try:
        import concurrent.futures as _cf
        if isinstance(obj, (_cf.ThreadPoolExecutor, _cf.ProcessPoolExecutor)):
            try:
                obj.shutdown(wait=True, cancel_futures=True)
            except TypeError:
                obj.shutdown(wait=True)
            return
    except Exception:
        pass

    for name in ("close", "shutdown", "dispose", "stop", "terminate"):
        m = getattr(obj, name, None)
        if callable(m):
            try:
                _try_call(m, wait=True, cancel_futures=True)
                return
            except Exception:
                try:
                    m()
                    return
                except Exception:
                    pass

def _close_component(obj):
    """
    Close an object and its common resource attributes if present.
    """
    if obj is None:
        return
    _best_effort_close(obj)
    for attr in (
        "neo4j_utils", "driver", "session", "tx", "engine",
        "executor", "thread_pool", "pool",
        "vector_store", "graph_store",
        "client", "db", "conn",
        "http", "transport", "producer", "consumer",
    ):
        _best_effort_close(getattr(obj, attr, None))

# ================== Alive thread handling (diagnostics + fallback) ==================

def _alive_non_daemon_threads():
    """Return all alive, non-daemon threads except the main thread."""
    return [
        t for t in threading.enumerate()
        if t is not threading.main_thread() and not t.daemon
    ]

def _dump_alive_threads(prefix="[diag] "):
    """Log alive non-daemon threads and their stacks."""
    alive = _alive_non_daemon_threads()
    if not alive:
        return alive
    logger.warning(prefix + "Non-daemon threads still alive: %d", len(alive))
    frames = sys._current_frames()
    for t in alive:
        logger.warning(
            prefix + "Thread: name=%s, ident=%s, cls=%s, alive=%s",
            t.name, t.ident, t.__class__.__name__, t.is_alive()
        )
        frame = frames.get(t.ident)
        if frame:
            stack_str = "".join(traceback.format_stack(frame))
            logger.warning(prefix + "Stack trace:\n%s", stack_str)
    return alive

def _join_alive_threads(timeout=0.5, prefix="[diag] "):
    """Attempt to join alive non-daemon threads briefly."""
    alive = _alive_non_daemon_threads()
    for t in alive:
        try:
            t.join(timeout=timeout)
        except Exception:
            pass
    alive2 = _alive_non_daemon_threads()
    if alive2:
        logger.warning(prefix + "Still alive after short join: %s", ", ".join([t.name for t in alive2]))
    return alive2

def _handle_alive_threads(grace_seconds=0.3):
    """
    Handle alive threads only:
      - Wait briefly
      - Dump thread info & stacks
      - Short join
      - If still alive, hard exit
    """
    time.sleep(grace_seconds)
    _dump_alive_threads(prefix="[diag] ")
    still = _join_alive_threads(timeout=0.5, prefix="[diag] ")
    if still:
        logger.error("[diag] Non-daemon threads remain alive; performing hard exit (os._exit(0)).")
        os._exit(0)

# ================== Main pipeline ==================

def _write_stats_if_requested(path: str, payload: dict) -> None:
    """Write a JSON stats payload to file if a path is provided."""
    if not path:
        return
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        logger.info("Wrote stats to %s", path)
    except Exception as e:
        logger.warning("Failed to write stats to %s: %s", path, e)

def main():
    parser = argparse.ArgumentParser(description="KAG-Builder: Knowledge Graph Constructor")
    parser.add_argument("--input", "-i", required=True, help="Path to the input JSON file")
    parser.add_argument("--config", "-c", default="configs/default.yaml", help="Path to the config YAML")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose (DEBUG) logging")
    parser.add_argument("--output-stats", "-s", help="Write summary stats to the given JSON file")
    args = parser.parse_args()

    # Adjust logging level for verbosity
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose mode enabled (DEBUG). Config: %s", args.config)

    # Load config
    config = KAGConfig.from_yaml(args.config)
    if args.verbose:
        logger.debug("Loaded config from %s", args.config)

    # ---------- Knowledge graph phase ----------
    # builder = KnowledgeGraphBuilder(config)
    # try:
    #     builder.prepare_chunks(args.input, verbose=args.verbose, extract_timelines=False)
    #     builder.store_chunks(verbose=args.verbose)
    # finally:
    #     _close_component(builder)
    #     builder = None
        
    # builder = KnowledgeGraphBuilder(config)
    # try:
    #     builder.run_graph_probing(verbose=args.verbose, sample_ratio=0.35)
    # finally:
    #     _close_component(builder)
    #     builder = None
        
    # builder = KnowledgeGraphBuilder(config)
    # try:
    #     builder.initialize_agents()
    #     builder.extract_entity_and_relation(verbose=args.verbose)
    # finally:
    #     _close_component(builder)
    #     builder = None

    # builder = KnowledgeGraphBuilder(config)
    # try:
    #     builder.initialize_agents()
    #     builder.run_extraction_refinement(verbose=args.verbose)
    # finally:
    #     _close_component(builder)
    #     builder = None
        
    # builder = KnowledgeGraphBuilder(config)
    # try:
    #     builder.initialize_agents()
    #     builder.extract_entity_attributes(verbose=args.verbose)
    # finally:
    #     _close_component(builder)
    #     builder = None

          
    builder = KnowledgeGraphBuilder(config)
    try:
        builder.graph_store.reset_knowledge_graph()
        _ = builder.build_graph_from_results(verbose=args.verbose)
        builder.prepare_graph_embeddings()
    finally:
        _close_component(builder)
        builder = None

    # # ---------- Relational DB phase ----------
    # sql_builder = RelationalDatabaseBuilder(config)
    # try:
    #     sql_builder.extract_cmp_information()
    #     sql_builder.build_relational_database()
    #     sql_builder.build_scene_info()
        
    # finally:
    #     _close_component(sql_builder)
    #     sql_builder = None

    event_graph_builder = EventCausalityBuilder(config)
    try:
        # event_graph_builder.initialize(keep_event_cards=False)
        # event_graph_builder.produce_causality_artifacts(limit_events=None)   # 计算 & 落JSON（事件清单/候选对/因果结果等）
        event_graph_builder.materialize_causality_graph()                    # 读取JSON & 写入事件-事件因果边
    finally:
        _close_component(event_graph_builder)
        event_graph_builder = None
    
    # event_graph_builder = EventCausalityBuilder(config)
    # try:
    #     event_graph_builder.extract_event_chains()
    #     event_graph_builder.build_and_save_plot_nodes()
    # finally:
    #     _close_component(event_graph_builder)
    #     event_graph_builder = None

    event_graph_builder = EventCausalityBuilder(config)
    try:
        # event_graph_builder.build_and_save_plot_relations()
        event_graph_builder.materialize_plot_graph()
    finally:
        _close_component(event_graph_builder)
        event_graph_builder = None

    # supplementary_builder = SupplementaryBuilder(config)
    # supplementary_builder.extract_character_status_for_all_scenes()
    # supplementary_builder.build_character_status_database()
    # supplementary_builder.check_scene_continuity(only_true=True)

    # logger.info("✅ Knowledge graph and event/plot graph pipeline completed.")

    _handle_alive_threads(grace_seconds=0.3)

    sys.exit(0)


if __name__ == "__main__":
    main()