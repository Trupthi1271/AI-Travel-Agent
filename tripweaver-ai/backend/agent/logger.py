"""
Structured Logger & Performance Monitor
-----------------------------------------
Track B requirement: error logging and basic monitoring.

Provides:
  - Structured JSON logging to file + console
  - Tool call timing decorator
  - Agent run metrics (latency, tool usage, errors)
  - Simple in-memory metrics store for the Streamlit dashboard
"""

from __future__ import annotations

import json
import logging
import time
import functools
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

# ── Log file location ─────────────────────────────────────────────────────────
_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)
_LOG_FILE = _LOG_DIR / "tripweaver.log"


# ── JSON formatter ────────────────────────────────────────────────────────────
from datetime import datetime, timezone

class _JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level":     record.levelname,
            "logger":    record.name,
            "message":   record.getMessage(),
        }
        if hasattr(record, "extra"):
            log_obj.update(record.extra)
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj, ensure_ascii=False)


# ── Logger setup ──────────────────────────────────────────────────────────────
def get_logger(name: str = "tripweaver") -> logging.Logger:
    """Return a configured logger. Safe to call multiple times."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # File handler — JSON lines
    fh = logging.FileHandler(_LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(_JSONFormatter())

    # Console handler — human readable
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


logger = get_logger("tripweaver")


# ── In-memory metrics store ───────────────────────────────────────────────────
class MetricsStore:
    """
    Lightweight in-memory metrics for the monitoring dashboard.
    Resets on process restart — not persistent.
    """

    def __init__(self):
        self.tool_calls:    Dict[str, int]          = defaultdict(int)
        self.tool_errors:   Dict[str, int]          = defaultdict(int)
        self.tool_latency:  Dict[str, List[float]]  = defaultdict(list)
        self.agent_runs:    int                     = 0
        self.agent_errors:  int                     = 0
        self.agent_latency: List[float]             = []
        self.query_types:   Dict[str, int]          = defaultdict(int)

    def record_tool_call(self, tool_name: str, latency_ms: float, error: bool = False):
        self.tool_calls[tool_name] += 1
        self.tool_latency[tool_name].append(latency_ms)
        if error:
            self.tool_errors[tool_name] += 1

    def record_agent_run(self, latency_ms: float, query_type: str = "general", error: bool = False):
        self.agent_runs += 1
        self.agent_latency.append(latency_ms)
        self.query_types[query_type] += 1
        if error:
            self.agent_errors += 1

    def avg_latency(self, latencies: List[float]) -> float:
        return round(sum(latencies) / len(latencies), 1) if latencies else 0.0

    def summary(self) -> Dict[str, Any]:
        return {
            "agent_runs":        self.agent_runs,
            "agent_errors":      self.agent_errors,
            "avg_agent_ms":      self.avg_latency(self.agent_latency),
            "tool_calls":        dict(self.tool_calls),
            "tool_errors":       dict(self.tool_errors),
            "avg_tool_latency":  {
                k: self.avg_latency(v) for k, v in self.tool_latency.items()
            },
            "query_types":       dict(self.query_types),
        }

    def reset(self):
        self.__init__()


# Global metrics instance
metrics = MetricsStore()


# ── Decorators ────────────────────────────────────────────────────────────────
def log_tool_call(tool_name: str) -> Callable[[F], F]:
    """
    Decorator that logs tool calls with timing and error tracking.

    Usage:
        @log_tool_call("WeatherTool")
        def get_weather(city: str) -> str: ...
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            error = False
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as exc:
                error = True
                logger.error(
                    f"Tool error: {tool_name} — {exc}",
                    extra={"tool": tool_name, "error_msg": str(exc)},
                )
                raise
            finally:
                latency_ms = (time.perf_counter() - start) * 1000
                metrics.record_tool_call(tool_name, latency_ms, error)
                logger.debug(
                    f"Tool call: {tool_name} — {latency_ms:.0f}ms {'❌' if error else '✅'}",
                    extra={"tool": tool_name, "latency_ms": round(latency_ms, 1), "error": error},
                )
        return wrapper  # type: ignore
    return decorator


def log_agent_run(func: F) -> F:
    """
    Decorator that logs full agent runs with timing.

    Usage:
        @log_agent_run
        def chat(self, user_input: str) -> str: ...
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        error = False
        user_input = args[1] if len(args) > 1 else kwargs.get("user_input", "")
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as exc:
            error = True
            logger.error(
                f"Agent run error: {exc}",
                extra={"user_input": str(user_input)[:100], "error": str(exc)},
            )
            raise
        finally:
            latency_ms = (time.perf_counter() - start) * 1000
            metrics.record_agent_run(latency_ms, error=error)
            logger.info(
                f"Agent run: {latency_ms:.0f}ms {'❌' if error else '✅'} — {str(user_input)[:60]}",
                extra={"latency_ms": round(latency_ms, 1), "error": error},
            )
    return wrapper  # type: ignore
