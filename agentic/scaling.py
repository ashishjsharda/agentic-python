"""Chapter 13: parallel tool calls and caching."""
import concurrent.futures
import functools
import json
import time

from .agent import Agent


class ConcurrentAgent(Agent):
    """Runs the tool calls from one model turn in parallel."""

    def run_tools(self, tool_calls):
        if len(tool_calls) <= 1:
            return [self._run_tool(tc) for tc in tool_calls]
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(tool_calls)
        ) as pool:
            return list(pool.map(self._run_tool, tool_calls))


def cached_tool(fn, ttl_s=300, clock=time.monotonic):
    cache = {}

    @functools.wraps(fn)
    def wrapper(**kwargs):
        key = json.dumps(kwargs, sort_keys=True)
        hit = cache.get(key)
        if hit and clock() - hit[0] < ttl_s:
            return hit[1]
        result = fn(**kwargs)
        cache[key] = (clock(), result)
        return result

    return wrapper


def cached_embed(embed, maxsize=10_000):
    return functools.lru_cache(maxsize=maxsize)(embed)
