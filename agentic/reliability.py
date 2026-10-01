"""Chapter 10: retries, timeouts, and a guardrail layer around tools."""
import concurrent.futures
import functools
import json
import random
import time

import httpx


class RetryableError(Exception):
    """An error worth retrying: rate limits and transient server faults."""


class RateLimitError(RetryableError):
    pass


class ServerError(RetryableError):
    pass


class ToolTimeout(Exception):
    pass


def post_with_retry(fn, max_retries=5, base_delay=1.0, sleep=time.sleep):
    for attempt in range(max_retries):
        try:
            return fn()
        except (RetryableError, httpx.TransportError):
            if attempt == max_retries - 1:
                raise
            sleep(base_delay * (2**attempt) + random.uniform(0, 1))


_TOOL_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=16)


def with_timeout(fn, seconds):
    future = _TOOL_POOL.submit(fn)
    try:
        return future.result(timeout=seconds)
    except concurrent.futures.TimeoutError as err:
        raise ToolTimeout(f"Tool did not finish within {seconds}s") \
            from err


def truncate_output(output, max_chars):
    text = output if isinstance(output, str) else json.dumps(
        output, default=str
    )
    if len(text) <= max_chars:
        return output
    dropped = len(text) - max_chars
    return text[:max_chars] + f"... [truncated {dropped} chars]"


def guard_tool(fn, timeout_s=30, max_output_chars=4000):
    @functools.wraps(fn)
    def guarded(**kwargs):
        output = with_timeout(lambda: fn(**kwargs), timeout_s)
        return truncate_output(output, max_output_chars)

    return guarded


def guard_tools(tools, **limits):
    return {name: guard_tool(fn, **limits) for name, fn in tools.items()}
