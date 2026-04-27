# core/utils/function_manager.py
from __future__ import annotations
import json
import logging
from typing import Dict, Any, List, Optional, Callable, Tuple, Set
from enum import Enum
from core.utils.format import correct_json_format, is_valid_json
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

import time
from collections import deque
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED, TimeoutError as FuturesTimeoutError
from typing import Any, Awaitable, Callable, Dict, Hashable, Iterable, List, Optional, Tuple, Set
import random
import traceback
import multiprocessing as mp


def _child_entry(fn, args, out_q: mp.Queue) -> None:
    try:
        out = fn(*args)
        out_q.put(("ok", out))
    except Exception as e:
        out_q.put((
            "err",
            f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        ))


def run_with_hard_timeout(
    *,
    fn,
    args: Tuple[Any, ...],
    timeout_s: float,
    start_method: str = "spawn",
) -> Tuple[bool, Any]:
    """
    Run fn(*args) in a separate spawned process.
    If timeout, terminate it and return (False, "timeout").
    """
    ctx = mp.get_context(start_method)
    out_q: mp.Queue = ctx.Queue()
    p = ctx.Process(target=_child_entry, args=(fn, args, out_q), daemon=True)
    p.start()
    p.join(timeout_s)

    if p.is_alive():
        p.terminate()
        p.join(5)
        return False, "timeout"

    try:
        status, payload = out_q.get_nowait()
    except queue.Empty:
        return False, "no result (child exited without output)"

    if status == "ok":
        return True, payload
    return False, payload


def run_concurrent_with_retries_hard_timeout(
    *,
    items: List[Any],
    work_fn: Callable[[Any], Any],
    key_fn: Callable[[Any], str],
    is_success_fn: Callable[[Any], bool],
    per_task_timeout: float,
    max_rounds: int,
    concurrency: int,
    retry_backoff_seconds: float = 0.0,
    start_method: str = "spawn",
    desc_prefix: str = "Running tasks",
    on_result: Optional[Callable[[str, Any], None]] = None,
    debug_print_fail: bool = False,
    fail_print_max_chars: int = 1200,
    poll_interval_s: float = 0.05,
) -> Tuple[Dict[str, Any], List[str]]:
    """
    True per-task hard-timeout runner.

    - Spawns one process per task.
    - Records each task's own start timestamp.
    - Kills only the tasks that individually exceed per_task_timeout.
    - Multi-round retries: only retry failed keys.
    - Calls on_result(key, result) immediately in parent when a task finishes.
    """
    if not items:
        return {}, []

    ctx = mp.get_context(start_method)

    max_rounds = max(1, int(max_rounds))
    concurrency = max(1, int(concurrency))

    # key -> item
    pending_by_key: Dict[str, Any] = {key_fn(it): it for it in items}
    final_map: Dict[str, Any] = {}

    for round_id in range(1, max_rounds + 1):
        if not pending_by_key:
            break

        keys = list(pending_by_key.keys())

        ok_cnt = 0
        fail_cnt = 0
        timeout_cnt = 0

        use_tqdm = tqdm is not None
        pbar = None
        if use_tqdm:
            pbar = tqdm(
                total=len(keys),
                desc=f"{desc_prefix} [round {round_id}/{max_rounds}]",
                leave=True,
                dynamic_ncols=True,
            )

        # running: key -> (proc, out_q, start_ts)
        running: Dict[str, Tuple[mp.Process, mp.Queue, float]] = {}
        next_pending_by_key: Dict[str, Any] = {}

        # queue of keys to start
        to_start = keys[:]
        idx = 0

        def _start_one(k: str) -> None:
            item = pending_by_key[k]
            out_q: mp.Queue = ctx.Queue()
            p = ctx.Process(
                target=_child_entry,
                args=(work_fn, (item,), out_q),
                daemon=True,
            )
            p.start()
            running[k] = (p, out_q, time.time())

        # fill initial
        while idx < len(to_start) and len(running) < concurrency:
            _start_one(to_start[idx])
            idx += 1

        # main loop: poll running tasks, start new ones as slots free
        while running:
            now = time.time()

            done_keys: List[str] = []

            for k, (p, out_q, start_ts) in list(running.items()):
                # 1) finished
                if not p.is_alive():
                    p.join(0.1)
                    try:
                        status, payload = out_q.get_nowait()
                    except queue.Empty:
                        payload = "no result (child exited without output)"
                        status = "err"

                    if status == "ok":
                        result = payload
                    else:
                        result = payload

                    final_map[k] = result

                    # callback immediately in parent
                    if on_result is not None:
                        try:
                            on_result(k, result)
                        except Exception:
                            pass

                    # success 판단은 is_success_fn(result)
                    if is_success_fn(result):
                        ok_cnt += 1
                    else:
                        fail_cnt += 1
                        next_pending_by_key[k] = pending_by_key[k]

                        if debug_print_fail:
                            msg = result
                            if not isinstance(msg, str):
                                try:
                                    msg = json.dumps(msg, ensure_ascii=False)
                                except Exception:
                                    msg = str(msg)
                            msg = (msg or "")[:fail_print_max_chars]
                            print(f"[FAIL] round={round_id} key={k} payload={msg}")

                    done_keys.append(k)

                    if pbar is not None:
                        pbar.update(1)
                        pbar.set_postfix({"fail": fail_cnt, "ok": ok_cnt, "timeout": timeout_cnt})

                    continue

                # 2) still running, check per-task timeout
                if (now - start_ts) > per_task_timeout:
                    p.terminate()
                    p.join(5)

                    result = "timeout"
                    final_map[k] = result

                    if on_result is not None:
                        try:
                            on_result(k, result)
                        except Exception:
                            pass

                    fail_cnt += 1
                    timeout_cnt += 1
                    next_pending_by_key[k] = pending_by_key[k]

                    done_keys.append(k)

                    if debug_print_fail:
                        print(f"[TIMEOUT] round={round_id} key={k} elapsed={now - start_ts:.1f}s")

                    if pbar is not None:
                        pbar.update(1)
                        pbar.set_postfix({"fail": fail_cnt, "ok": ok_cnt, "timeout": timeout_cnt})

            # remove completed/terminated
            for k in done_keys:
                running.pop(k, None)

            # start more if slots free
            while idx < len(to_start) and len(running) < concurrency:
                _start_one(to_start[idx])
                idx += 1

            if running:
                time.sleep(poll_interval_s)

        if pbar is not None:
            pbar.close()

        pending_by_key = next_pending_by_key

        if pending_by_key and retry_backoff_seconds > 0:
            time.sleep(float(retry_backoff_seconds))

    still_failed_keys = list(pending_by_key.keys())
    return final_map, still_failed_keys


# =========================
# NEW: unify llm.run() output into a single content string
# =========================
def unwrap_llm_content(result: Any) -> str:
    """
    Normalize different llm_client.run() return formats into a single assistant content string.

    Supports:
    - list[Message] (Message behaves like dict via .get("content"))
    - list[dict]
    - dict
    - nested list (defensive)
    - plain string
    """
    if result is None:
        return ""

    # list case
    if isinstance(result, list):
        if not result:
            return ""
        first = result[0]

        # nested list (defensive)
        if isinstance(first, list):
            return unwrap_llm_content(first)

        # dict
        if isinstance(first, dict):
            return first.get("content", "") or ""

        # Message-like (has .get)
        if hasattr(first, "get"):
            try:
                return first.get("content") or ""
            except Exception:
                return str(first)

        return str(first)

    # dict case
    if isinstance(result, dict):
        return result.get("content", "") or ""

    # Message-like (has .get)
    if hasattr(result, "get"):
        try:
            return result.get("content") or ""
        except Exception:
            return str(result)

    # string / fallback
    return str(result)


def run_with_soft_timeout_and_retries(
    items: List[Any],
    *,
    work_fn,
    key_fn,
    desc_label: str,
    per_task_timeout: float = 600.0,
    retries: int = 2,
    retry_backoff: float = 1.0,
    allow_placeholder_first_round: bool = False,
    placeholder_fn=None,
    should_retry=None,
    max_workers: Optional[int] = 16,
) -> Tuple[Dict[Any, Any], Set[Any]]:
    import threading
    total = len(items)
    if total == 0:
        return {}, set()

    results: Dict[Any, Any] = {}
    remaining_items = items[:]
    max_rounds = max(1, retries)

    for round_id in range(1, max_rounds + 1):
        if not remaining_items:
            break

        round_failures: Set[Any] = set()
        round_timeouts: Set[Any] = set()

        executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix=f"concur-r{round_id}")
        try:
            fut_info: Dict[Any, Dict[str, Any]] = {}
            for it in remaining_items:
                f = executor.submit(work_fn, it)
                fut_info[f] = {"start": time.monotonic(), "item": it, "key": key_fn(it)}

            pbar = tqdm(total=len(fut_info), desc=f"{desc_label}（第{round_id}/{max_rounds}轮）", ncols=100)
            pending = set(fut_info.keys())

            while pending:
                done, pending = wait(pending, timeout=0.25, return_when=FIRST_COMPLETED)

                for f in done:
                    meta = fut_info.pop(f, None)
                    key = meta["key"]
                    item = meta["item"]
                    try:
                        res = f.result()
                        results[key] = res
                        if callable(should_retry) and should_retry(res):
                            round_failures.add(key)
                    except Exception as e:
                        if allow_placeholder_first_round and round_id == 1 and callable(placeholder_fn):
                            try:
                                results[key] = placeholder_fn(item, exc=e)
                            except Exception:
                                pass
                        round_failures.add(key)
                    pbar.update(1)

                now = time.monotonic()
                to_forget = []
                for f in list(pending):
                    meta = fut_info[f]
                    if now - meta["start"] >= per_task_timeout:
                        key = meta["key"]
                        item = meta["item"]
                        try:
                            f.cancel()
                        except Exception:
                            pass
                        if allow_placeholder_first_round and round_id == 1 and callable(placeholder_fn):
                            try:
                                results[key] = placeholder_fn(item, exc=FuturesTimeoutError(f"soft-timeout {per_task_timeout}s"))
                            except Exception:
                                pass
                        round_timeouts.add(key)
                        pbar.update(1)
                        to_forget.append(f)

                for f in to_forget:
                    pending.remove(f)
                    fut_info.pop(f, None)

            pbar.close()
        finally:
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass

        keys_to_retry = round_failures | round_timeouts
        key_set = set(keys_to_retry)

        remaining_items = [it for it in items if key_fn(it) in key_set]

        if remaining_items and round_id < max_rounds and retry_backoff > 0:
            try:
                time.sleep(retry_backoff)
            except Exception:
                pass

    still_failed_keys: Set[Any] = {key_fn(it) for it in remaining_items}
    return results, still_failed_keys


def run_concurrent_with_retries(
    items: List[Any],
    task_fn: Callable[[Any], Any],
    *,
    per_task_timeout: float = 120.0,
    max_retry_rounds: int = 1,
    max_in_flight: Optional[int] = None,
    max_workers: Optional[int] = None,
    thread_name_prefix: str = "pool",
    desc_prefix: str = "Concurrent jobs",
    treat_empty_as_failure: bool = False,
    is_empty_fn: Optional[Callable[[Any], bool]] = None,
    show_progress: bool = True,
    on_attempt_result: Optional[Callable[[int, Any, bool, int], None]] = None,
) -> Tuple[Dict[int, Any], List[int]]:
    assert max_retry_rounds >= 1, "max_retry_rounds must be >= 1"
    if max_workers is None and max_in_flight is None:
        max_in_flight = 4
    if max_in_flight is None:
        max_in_flight = max_workers or 4

    def _run_one_round(round_indices: List[int], attempt_idx: int) -> Tuple[Dict[int, Any], List[int]]:
        results_round: Dict[int, Any] = {}
        failures_round: List[int] = []

        todo = deque(round_indices)
        fut_info: Dict[Any, Dict[str, Any]] = {}
        pending = set()

        executor = ThreadPoolExecutor(
            max_workers=max_workers or max_in_flight,
            thread_name_prefix=f"{thread_name_prefix}_a{attempt_idx}"
        )

        def _submit_one(idx: int):
            item = items[idx]
            start_box = {"t": None}

            def _wrapper(_item, _box):
                _box["t"] = time.monotonic()
                return task_fn(_item)

            f = executor.submit(_wrapper, item, start_box)
            fut_info[f] = {"idx": idx, "start_box": start_box}
            pending.add(f)

        while todo and len(pending) < max_in_flight:
            _submit_one(todo.popleft())

        pbar = tqdm(total=len(round_indices), desc=f"{desc_prefix} / round {attempt_idx+1}", ncols=100) if show_progress else None

        try:
            while pending:
                done, _ = wait(pending, timeout=0.25, return_when=FIRST_COMPLETED)

                for f in done:
                    info = fut_info.pop(f)
                    idx = info["idx"]
                    attempt_ok = False
                    attempt_res: Any = None
                    try:
                        res = f.result()
                        attempt_res = res
                        if treat_empty_as_failure:
                            empty = False
                            if is_empty_fn is not None:
                                empty = bool(is_empty_fn(res))
                            else:
                                empty = (res is None) or (res == []) or (res == {})
                            if empty:
                                failures_round.append(idx)
                            else:
                                results_round[idx] = res
                                attempt_ok = True
                        else:
                            results_round[idx] = res
                            attempt_ok = True
                    except Exception:
                        failures_round.append(idx)
                    if on_attempt_result is not None:
                        try:
                            on_attempt_result(idx, attempt_res, attempt_ok, attempt_idx)
                        except Exception:
                            pass
                    pending.remove(f)
                    if pbar:
                        pbar.update(1)

                    if todo and len(pending) < max_in_flight:
                        _submit_one(todo.popleft())

                now = time.monotonic()
                to_timeout = []
                for f in list(pending):
                    sb = fut_info[f]["start_box"]
                    t0 = sb.get("t")
                    if t0 is not None and (now - t0) >= per_task_timeout:
                        to_timeout.append(f)

                for f in to_timeout:
                    info = fut_info.pop(f)
                    idx = info["idx"]
                    try:
                        f.cancel()
                    finally:
                        failures_round.append(idx)
                        pending.remove(f)
                        if pbar:
                            pbar.update(1)
                        if todo and len(pending) < max_in_flight:
                            _submit_one(todo.popleft())
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
            if pbar:
                pbar.close()

        return results_round, failures_round

    indices = list(range(len(items)))
    all_results: Dict[int, Any] = {}
    round_results, failures = _run_one_round(indices, attempt_idx=0)
    all_results.update(round_results)

    attempt = 1
    while failures and attempt < max_retry_rounds:
        round_results, failures = _run_one_round(failures, attempt_idx=attempt)
        all_results.update(round_results)
        attempt += 1

    return all_results, failures


def apply_concurrent_with_retries(
    items: List[Any],
    task_fn: Callable[[Any], Any],
    *,
    per_task_timeout: float = 120.0,
    max_retry_rounds: int = 1,
    max_in_flight: Optional[int] = None,
    max_workers: Optional[int] = None,
    thread_name_prefix: str = "pool",
    desc_prefix: str = "Concurrent jobs",
    treat_empty_as_failure: bool = False,
    is_empty_fn: Optional[Callable[[Any], bool]] = None,
    default_on_fail: Any = None,
) -> List[Any]:
    results_map, still_failed = run_concurrent_with_retries(
        items,
        task_fn,
        per_task_timeout=per_task_timeout,
        max_retry_rounds=max_retry_rounds,
        max_in_flight=max_in_flight,
        max_workers=max_workers,
        thread_name_prefix=thread_name_prefix,
        desc_prefix=desc_prefix,
        treat_empty_as_failure=treat_empty_as_failure,
        is_empty_fn=is_empty_fn,
    )

    out: List[Any] = []
    for i in range(len(items)):
        if i in results_map:
            out.append(results_map[i])
        else:
            out.append(default_on_fail)
    return out


class JSONIssueType(Enum):
    VALID = "valid"
    INVALID_AFTER_CORRECTION = "invalid_after_correction"
    MISSING_REQUIRED_FIELDS = "missing_required_fields"
    INVALID_FIELD_VALUES = "invalid_field_values"
    EMPTY_RESPONSE = "empty_response"


class EnhancedJSONUtils:
    """Enhanced JSON handling utilities."""

    @staticmethod
    def _coerce_probability(value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        try:
            if isinstance(value, str):
                text = value.strip()
                if not text:
                    return None
                is_percent = text.endswith("%")
                text = text.rstrip("%").strip()
                prob = float(text)
                if is_percent:
                    prob = prob / 100.0
            else:
                prob = float(value)
        except Exception:
            return None
        if prob > 1.0 and prob <= 100.0:
            prob = prob / 100.0
        return max(0.0, min(1.0, prob))

    @staticmethod
    def _coerce_bool_like(value: Any) -> Optional[bool]:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and value in (0, 1):
            return bool(value)
        if isinstance(value, str):
            text = value.strip().lower()
            if text in {"true", "yes", "y", "1", "relevant", "likely", "是", "相关", "有关"}:
                return True
            if text in {"false", "no", "n", "0", "irrelevant", "unlikely", "否", "不相关", "无关"}:
                return False
        return None

    @staticmethod
    def _try_local_schema_normalization(
        parsed: Optional[Any],
        required_fields: Optional[List[str]],
        field_validators: Optional[Dict[str, Callable]],
    ) -> Optional[Dict[str, Any]]:
        """
        Normalize very small, high-frequency schemas locally before escalating to
        an LLM repair call. This is intentionally conservative: it only handles
        the retrieval relevance schema used heavily during QA.
        """
        if not isinstance(parsed, dict):
            return None

        required = set(required_fields or [])
        if not {"probability", "is_relevant", "reason"}.issubset(required):
            return None

        normalized = dict(parsed)
        probability = EnhancedJSONUtils._coerce_probability(normalized.get("probability"))
        is_relevant = EnhancedJSONUtils._coerce_bool_like(normalized.get("is_relevant"))

        if probability is None and is_relevant is not None:
            probability = 1.0 if is_relevant else 0.0
        if is_relevant is None and probability is not None:
            is_relevant = probability >= 0.5

        normalized["probability"] = 0.0 if probability is None else probability
        normalized["is_relevant"] = False if is_relevant is None else is_relevant
        normalized["reason"] = "" if normalized.get("reason") is None else str(normalized.get("reason", ""))

        for field in required:
            if field not in normalized:
                return None

        if field_validators:
            for field, validator in field_validators.items():
                if field not in normalized:
                    return None
                try:
                    if not validator(normalized[field]):
                        return None
                except Exception:
                    return None

        return normalized

    @staticmethod
    def analyze_json_response(
        content: str,
        required_fields: Optional[List[str]] = None,
        field_validators: Optional[Dict[str, Callable]] = None,
    ) -> Tuple[JSONIssueType, str, Optional[Dict]]:
        if not content or content.strip() == "":
            return JSONIssueType.EMPTY_RESPONSE, "Empty response", None

        corrected_content = correct_json_format(content)

        if not is_valid_json(corrected_content):
            return JSONIssueType.INVALID_AFTER_CORRECTION, "Still not valid JSON after format correction", None

        try:
            parsed = json.loads(corrected_content)
        except json.JSONDecodeError as e:
            return JSONIssueType.INVALID_AFTER_CORRECTION, f"JSON parse error: {e}", None

        if required_fields:
            if isinstance(parsed, dict):
                missing_fields = [f for f in required_fields if f not in parsed]
                if missing_fields:
                    return JSONIssueType.MISSING_REQUIRED_FIELDS, f"Missing required fields: {missing_fields}", parsed
            else:
                return JSONIssueType.MISSING_REQUIRED_FIELDS, "Parsed JSON is not an object; required_fields cannot be satisfied", None

        if field_validators and isinstance(parsed, dict):
            for field, validator in field_validators.items():
                if field in parsed:
                    try:
                        if not validator(parsed[field]):
                            return JSONIssueType.INVALID_FIELD_VALUES, f"Field '{field}' validation failed", parsed
                    except Exception as e:
                        return JSONIssueType.INVALID_FIELD_VALUES, f"Field '{field}' validation exception: {e}", parsed

        return JSONIssueType.VALID, "JSON is valid", parsed

    @staticmethod
    def enhanced_json_validation(
        content: str,
        required_fields: Optional[List[str]] = None,
        field_validators: Optional[Dict[str, Callable]] = None,
    ) -> Tuple[bool, str, Optional[Any], str]:
        corrected_content = correct_json_format(content)

        issue_type, error_msg, parsed = EnhancedJSONUtils.analyze_json_response(
            content, required_fields, field_validators
        )
        is_valid = issue_type == JSONIssueType.VALID
        return is_valid, error_msg, parsed, corrected_content

    @staticmethod
    def process_llm_response_with_retry(
        llm_client,
        initial_messages: List[Dict],
        required_fields: Optional[List[str]] = None,
        field_validators: Optional[Dict[str, Callable]] = None,
        max_retries: int = 3,
        repair_prompt_template: Optional[str] = None,
    ) -> Tuple[Any, str]:
        """
        Always returns a JSON string that has been passed through `correct_json_format` (even on failure).
        """
        # First attempt
        result = llm_client.run(initial_messages)
        content = unwrap_llm_content(result)
        issue_type, error_msg, parsed = EnhancedJSONUtils.analyze_json_response(
            content, required_fields, field_validators
        )
        corrected_content = correct_json_format(content)
        is_valid = issue_type == JSONIssueType.VALID
        if is_valid:
            return parsed, corrected_content

        normalized = EnhancedJSONUtils._try_local_schema_normalization(
            parsed,
            required_fields,
            field_validators,
        )
        if normalized is not None:
            return normalized, json.dumps(normalized, ensure_ascii=False)

        # Prefer the locally repaired JSON as the repair basis before escalating to LLM repair.
        current_content = corrected_content or content
        for attempt in range(max_retries):
            logger.warning("Attempting to repair JSON response (try %d/%d).", attempt + 1, max_retries)

            try:
                if repair_prompt_template:
                    repair_messages = list(initial_messages)
                    repair_prompt = repair_prompt_template.format(
                        original_response=current_content, error_message=error_msg
                    )
                    repair_messages.append({"role": "user", "content": repair_prompt})

                    repair_result = llm_client.run(repair_messages)
                    repair_content = unwrap_llm_content(repair_result)

                    is_valid, new_error_msg, parsed, corrected_content = EnhancedJSONUtils.enhanced_json_validation(
                        repair_content, required_fields, field_validators
                    )

                    if is_valid:
                        logger.warning("Repair succeeded on attempt %d.", attempt + 1)
                        return parsed, corrected_content

                    current_content = repair_content
                    error_msg = new_error_msg
                else:
                    repair_messages = list(initial_messages)
                    repair_messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Please fix the JSON in the following response.\n"
                                f"Issue: {error_msg}\n"
                                f"Original response: {current_content}\n"
                                "Return the complete, corrected JSON only."
                            ),
                        }
                    )

                    repair_result = llm_client.run(repair_messages)
                    repair_content = unwrap_llm_content(repair_result)

                    is_valid, new_error_msg, parsed, corrected_content = EnhancedJSONUtils.enhanced_json_validation(
                        repair_content, required_fields, field_validators
                    )

                    if is_valid:
                        logger.warning("Generic repair succeeded on attempt %d.", attempt + 1)
                        return parsed, corrected_content

                    current_content = repair_content
                    error_msg = new_error_msg

            except Exception as e:
                logger.exception("Repair attempt %d failed with exception: %s", attempt + 1, e)
                continue

        logger.error("All repair attempts failed; making one final call and returning corrected content.")
        try:
            result = llm_client.run(initial_messages)
            content = unwrap_llm_content(result)
        except Exception as e:
            logger.exception("Final call after failed repairs also failed: %s", e)
            content = ""

        final_corrected = correct_json_format(content)

        error_result = {
            "error": "Response processing failed",
            "original_content": content,
            "error_details": error_msg,
            "attempts": max_retries + 1,
        }
        return error_result, final_corrected


def is_valid_json_enhanced(
    content: str,
    required_fields: Optional[List[str]] = None,
    field_validators: Optional[Dict[str, Callable]] = None,
) -> bool:
    is_valid, _, _, _ = EnhancedJSONUtils.enhanced_json_validation(content, required_fields, field_validators)
    return is_valid


def get_corrected_json(content: str) -> str:
    return correct_json_format(content)


def analyze_json_issues(
    content: str,
    required_fields: Optional[List[str]] = None,
    field_validators: Optional[Dict[str, Callable]] = None,
) -> str:
    issue_type, error_msg, _ = EnhancedJSONUtils.analyze_json_response(content, required_fields, field_validators)
    return f"{issue_type.value}: {error_msg}"


def process_with_format_guarantee(
    llm_client,
    messages: List[Dict],
    required_fields: Optional[List[str]] = None,
    field_validators: Optional[Dict[str, Callable]] = None,
    max_retries: int = 3,
    repair_template: Optional[str] = None,
) -> Tuple[str, str]:
    result_dict, corrected_json = EnhancedJSONUtils.process_llm_response_with_retry(
        llm_client=llm_client,
        initial_messages=messages,
        required_fields=required_fields,
        field_validators=field_validators,
        max_retries=max_retries,
        repair_prompt_template=repair_template,
    )

    # IMPORTANT: parsed may be a list (e.g., event extraction output), so guard dict-only .get()
    status = "error" if (isinstance(result_dict, dict) and result_dict.get("error")) else "success"
    return corrected_json, status
