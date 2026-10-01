"""Chapter 18: structured JSON logs, one object per line."""
import contextvars
import json
import logging
import sys

request_id = contextvars.ContextVar("request_id", default="-")


class JsonFormatter(logging.Formatter):
    def format(self, record):
        entry = {
            "ts": round(record.created, 3),
            "level": record.levelname,
            "event": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
            **getattr(record, "fields", {}),
        }
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


def configure_logging(level=logging.INFO):
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)


def log(logger, event, **fields):
    # Capture the request id now: a queued or deferred handler may
    # format this record later, outside the request's context.
    logger.info(event, extra={"fields": fields,
                              "request_id": request_id.get()})


def trace_to_log(logger, run_id):
    """An Agent trace callback that emits compact, searchable events."""

    def trace(event, data):
        if event == "tool_call":
            log(logger, "tool_call", run_id=str(run_id),
                tool=data["name"], ms=data["duration_ms"],
                error=data["is_error"])
        elif event == "model_response":
            log(logger, "model_call", run_id=str(run_id),
                step=data["step"], ms=data["duration_ms"],
                **data.get("usage", {}))
        elif event == "max_steps":
            log(logger, "max_steps", run_id=str(run_id))

    return trace
