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

def main():
    parser = argparse.ArgumentParser(description="KAG-Builder: Knowledge Graph Constructor")
    parser.add_argument("--config", "-c", default="configs/config_openai.yaml", help="Path to the config YAML")
    args = parser.parse_args()

    # Load config
    config = KAGConfig.from_yaml(args.config)

    supplementary_builder = SupplementaryBuilder(config)
    # supplementary_builder.extract_character_status_for_all_scenes()
    # supplementary_builder.build_character_status_database()
    # supplementary_builder.check_scene_continuity(only_true=True)
    supplementary_builder.generate_continuity_chains()

    # logger.info("✅ Knowledge graph and event/plot graph pipeline completed.")
    _handle_alive_threads(grace_seconds=0.3)

    sys.exit(0)


if __name__ == "__main__":
    main()
