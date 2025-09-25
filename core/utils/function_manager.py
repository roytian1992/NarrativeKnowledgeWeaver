import json
import logging
from typing import Dict, Any, List, Optional, Callable, Tuple, Set
from enum import Enum
from core.utils.format import correct_json_format, is_valid_json

logger = logging.getLogger(__name__)

import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED, TimeoutError as FuturesTimeoutError

def run_with_soft_timeout_and_retries(
    items: List[Any],
    *,
    work_fn,                      # (item) -> result
    key_fn,                       # (item) -> hashable key
    desc_label: str,
    per_task_timeout: float = 600.0,
    retries: int = 2,
    retry_backoff: float = 1.0,
    allow_placeholder_first_round: bool = False,
    placeholder_fn=None,          # (item, exc=None) -> placeholder_result
    should_retry=None,             # (result) -> bool
    max_workers: Optional[int] = 16,
) -> Tuple[Dict[Any, Any], Set[Any]]:
    import threading
    total = len(items)
    if total == 0:
        return {}, set()

    results: Dict[Any, Any] = {}
    remaining_items = items[:]
    still_failed_keys: Set[Any] = set()
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
        still_failed_keys |= keys_to_retry
        key_set = set(keys_to_retry)
        remaining_items = [it for it in items if key_fn(it) in key_set]

        if remaining_items and round_id < max_rounds and retry_backoff > 0:
            try:
                time.sleep(retry_backoff)
            except Exception:
                pass

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
) -> Tuple[Dict[int, Any], List[int]]:
    """
    Windowed concurrency with soft timeouts (measured from when the task actually starts)
    and multi-round retries. Good default for all batch LLM / IO tasks.

    Returns: (results_by_index, still_failed_indices)
    - results_by_index: dict {idx -> result} for tasks that succeeded in any round
    - still_failed_indices: indices that failed/time-out in the last round

    Behavior:
    - Keeps at most `max_in_flight` tasks running at any time (default=max_workers).
    - Soft timeout: a future is considered timed out when (now - start_time) >= per_task_timeout.
      We try to cancel the future; regardless of cancel outcome, we mark it timeout and continue.
    - Retries: up to `max_retry_rounds` rounds total (round 1 + (max_retry_rounds-1) retries).
      Each later round only re-runs the failed/time-out indices.

    Failure policy:
    - If `treat_empty_as_failure=True`, a task returning None/[]/{} (or custom `is_empty_fn`) is treated as failure.
    """
    assert max_retry_rounds >= 1, "max_retry_rounds must be >= 1"
    if max_workers is None and max_in_flight is None:
        # reasonable default
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
            # store the actual start time in a small box (set inside the worker)
            start_box = {"t": None}

            def _wrapper(_item, _box):
                _box["t"] = time.monotonic()
                return task_fn(_item)

            f = executor.submit(_wrapper, item, start_box)
            fut_info[f] = {"idx": idx, "start_box": start_box}
            pending.add(f)

        # Prime the window
        while todo and len(pending) < max_in_flight:
            _submit_one(todo.popleft())

        pbar = tqdm(total=len(round_indices), desc=f"{desc_prefix} / round {attempt_idx+1}", ncols=100)
        try:
            while pending:
                done, _ = wait(pending, timeout=0.25, return_when=FIRST_COMPLETED)

                # 1) collect finished
                for f in done:
                    info = fut_info.pop(f)
                    idx = info["idx"]
                    try:
                        res = f.result()
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
                        else:
                            results_round[idx] = res
                    except Exception:
                        failures_round.append(idx)
                    pending.remove(f)
                    pbar.update(1)

                    # keep the window full
                    if todo and len(pending) < max_in_flight:
                        _submit_one(todo.popleft())

                # 2) soft timeout inspection
                now = time.monotonic()
                to_timeout = []
                for f in list(pending):
                    sb = fut_info[f]["start_box"]
                    t0 = sb.get("t")
                    # only time out after the task actually started
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
                        pbar.update(1)
                        if todo and len(pending) < max_in_flight:
                            _submit_one(todo.popleft())
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
            pbar.close()

        return results_round, failures_round

    # Round 1
    indices = list(range(len(items)))
    all_results: Dict[int, Any] = {}
    round_results, failures = _run_one_round(indices, attempt_idx=0)
    all_results.update(round_results)

    # Retries
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
    """
    Same engine as `run_concurrent_with_retries`, but returns a list in the original order.
    Failed/time-out indices are filled with `default_on_fail`.

    Useful when the caller just needs aligned outputs: output[i] corresponds to items[i].
    """
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
    """JSON问题类型"""
    VALID = "valid"
    INVALID_AFTER_CORRECTION = "invalid_after_correction"  
    MISSING_REQUIRED_FIELDS = "missing_required_fields"    
    INVALID_FIELD_VALUES = "invalid_field_values"          
    EMPTY_RESPONSE = "empty_response"                      


class EnhancedJSONUtils:
    """Enhanced JSON handling utilities."""

    @staticmethod
    def analyze_json_response(
        content: str,
        required_fields: Optional[List[str]] = None,
        field_validators: Optional[Dict[str, Callable]] = None,
    ) -> Tuple[JSONIssueType, str, Optional[Dict]]:
        """
        Analyze a JSON response and determine the issue type (if any).

        Args:
            content: Raw response content.
            required_fields: List of required top-level fields.
            field_validators: Mapping of field -> validator callable(value) -> bool.

        Returns:
            Tuple[JSONIssueType, message, parsed_or_none]
        """
        if not content or content.strip() == "":
            return JSONIssueType.EMPTY_RESPONSE, "Empty response", None

        # Try to normalize/repair formatting first
        corrected_content = correct_json_format(content)

        # Basic validity check
        if not is_valid_json(corrected_content):
            return JSONIssueType.INVALID_AFTER_CORRECTION, "Still not valid JSON after format correction", None

        # Parse JSON
        try:
            parsed = json.loads(corrected_content)
        except json.JSONDecodeError as e:
            return JSONIssueType.INVALID_AFTER_CORRECTION, f"JSON parse error: {e}", None

        # Required fields check
        if required_fields:
            missing_fields = [f for f in required_fields if f not in parsed]
            if missing_fields:
                return JSONIssueType.MISSING_REQUIRED_FIELDS, f"Missing required fields: {missing_fields}", parsed

        # Field value validation
        if field_validators:
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
    ) -> Tuple[bool, str, Optional[Dict], str]:
        """
        Enhanced JSON validation that also returns the corrected JSON string.

        Args:
            content: Raw JSON string (possibly malformed).
            required_fields: List of required fields.
            field_validators: Mapping of field -> validator callable(value) -> bool.

        Returns:
            (is_valid, message, parsed_or_none, corrected_json_string)
        """
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
        enable_thinking: bool = True,
        repair_prompt_template: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], str]:
        """
        Process an LLM response with retry and repair. Always returns a JSON string
        that has been passed through `correct_json_format` (even on failure).

        Args:
            llm_client: Your LLM client instance exposing `.run(messages, enable_thinking=...)`.
            initial_messages: The initial chat messages to send to the LLM.
            required_fields: Required JSON fields.
            field_validators: Field validators mapping.
            max_retries: Maximum number of repair attempts.
            enable_thinking: Whether to enable thinking mode for the initial call.
            repair_prompt_template: Optional template used for repair; must include
                                    `{original_response}` and `{error_message}` placeholders.

        Returns:
            (result_dict, final_json_string)
            - On success: result_dict is the parsed JSON.
            - On failure: result_dict contains an error summary; final_json_string is the corrected last response.
        """
        # First attempt
        result = llm_client.run(initial_messages, enable_thinking=enable_thinking)
        content = result[0]["content"] if isinstance(result, list) else result.get("content", "")

        # Validate
        is_valid, error_msg, parsed, corrected_content = EnhancedJSONUtils.enhanced_json_validation(
            content, required_fields, field_validators
        )
        if is_valid:
            # No info logging in normal path per requirement
            return parsed, corrected_content

        # Retry & repair loop (abnormal path -> allow logging)
        current_content = content
        for attempt in range(max_retries):
            logger.warning("Attempting to repair JSON response (try %d/%d).", attempt + 1, max_retries)

            try:
                if repair_prompt_template:
                    repair_messages = list(initial_messages)
                    repair_prompt = repair_prompt_template.format(
                        original_response=current_content, error_message=error_msg
                    )
                    repair_messages.append({"role": "user", "content": repair_prompt})

                    repair_result = llm_client.run(repair_messages, enable_thinking=False)
                    repair_content = (
                        repair_result[0]["content"] if isinstance(repair_result, list) else repair_result.get("content", "")
                    )

                    is_valid, new_error_msg, parsed, corrected_content = EnhancedJSONUtils.enhanced_json_validation(
                        repair_content, required_fields, field_validators
                    )

                    if is_valid:
                        logger.warning("Repair succeeded on attempt %d.", attempt + 1)
                        return parsed, corrected_content

                    current_content = repair_content
                    error_msg = new_error_msg
                else:
                    # Generic repair prompt (English)
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

                    repair_result = llm_client.run(repair_messages, enable_thinking=enable_thinking)
                    repair_content = (
                        repair_result[0]["content"] if isinstance(repair_result, list) else repair_result.get("content", "")
                    )

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

        # Final fallback: return corrected last response with error summary
        logger.error("All repair attempts failed; making one final call and returning corrected content.")
        try:
            result = llm_client.run(initial_messages, enable_thinking=enable_thinking)
            content = result[0]["content"] if isinstance(result, list) else result.get("content", "")
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


# Convenience wrappers (I/O compatible with your existing code)
def is_valid_json_enhanced(
    content: str,
    required_fields: Optional[List[str]] = None,
    field_validators: Optional[Dict[str, Callable]] = None,
) -> bool:
    """Enhanced boolean JSON validity check based on format utilities."""
    is_valid, _, _, _ = EnhancedJSONUtils.enhanced_json_validation(content, required_fields, field_validators)
    return is_valid


def get_corrected_json(content: str) -> str:
    """Get the JSON string after `correct_json_format` normalization."""
    return correct_json_format(content)


def analyze_json_issues(
    content: str,
    required_fields: Optional[List[str]] = None,
    field_validators: Optional[Dict[str, Callable]] = None,
) -> str:
    """Return a short string describing JSON issues."""
    issue_type, error_msg, _ = EnhancedJSONUtils.analyze_json_response(content, required_fields, field_validators)
    return f"{issue_type.value}: {error_msg}"


def process_with_format_guarantee(
    llm_client,
    messages: List[Dict],
    required_fields: Optional[List[str]] = None,
    field_validators: Optional[Dict[str, Callable]] = None,
    max_retries: int = 3,
    enable_thinking: bool = True,
    repair_template: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Process an LLM response and guarantee that the returned JSON string has passed
    through `correct_json_format`.

    Returns:
        (corrected_json_string, status) where status is "success" or "error".
    """
    result_dict, corrected_json = EnhancedJSONUtils.process_llm_response_with_retry(
        llm_client=llm_client,
        initial_messages=messages,
        required_fields=required_fields,
        field_validators=field_validators,
        max_retries=max_retries,
        enable_thinking=enable_thinking,
        repair_prompt_template=repair_template,
    )
    status = "error" if result_dict.get("error") else "success"
    return corrected_json, status