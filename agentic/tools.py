"""Chapter 3: tool schemas, a dispatch table, and three first tools."""
import ast
import inspect
import operator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import jsonschema
from pydantic import ConfigDict, create_model

from .structured import model_schema

CALCULATOR_SCHEMA = {
    "name": "calculator",
    "description": (
        "Evaluate an arithmetic expression. Supports + - * / % ** and "
        "parentheses. Use for any math instead of computing in your head."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "expression": {"type": "string",
                           "description": "e.g. '0.15 * 348.20'"},
        },
        "required": ["expression"],
        "additionalProperties": False,
    },
}

WEB_SEARCH_SCHEMA = {
    "name": "web_search",
    "description": (
        "Search the web. Returns up to max_results items, each with a "
        "title, url, and short snippet. Use for current facts."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "max_results": {"type": "integer", "minimum": 1,
                            "maximum": 10, "default": 5},
        },
        "required": ["query"],
        "additionalProperties": False,
    },
}

READ_FILE_SCHEMA = {
    "name": "read_file",
    "description": "Read a text file from the agent's workspace.",
    "input_schema": {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
        "additionalProperties": False,
    },
}

WRITE_FILE_SCHEMA = {
    "name": "write_file",
    "description": "Write a text file into the agent's workspace.",
    "input_schema": {
        "type": "object",
        "properties": {"path": {"type": "string"},
                       "content": {"type": "string"}},
        "required": ["path", "content"],
        "additionalProperties": False,
    },
}


def execute_tool_call(tool_call, tools, schemas):
    name = tool_call["name"]
    fn = tools.get(name)
    if fn is None:
        return {"error": f"Unknown tool '{name}'. "
                         f"Available: {sorted(tools)}"}
    try:
        jsonschema.validate(tool_call["input"],
                            schemas[name]["input_schema"])
    except jsonschema.ValidationError as e:
        return {"error": f"Invalid arguments for {name}: {e.message}"}
    try:
        return fn(**tool_call["input"])
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


_OPERATORS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def calculator(expression):
    def evaluate(node):
        if isinstance(node, ast.Constant) and type(node.value) in (
            int, float
        ):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("Exponent too large")
            return _OPERATORS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
            return _OPERATORS[type(node.op)](evaluate(node.operand))
        raise ValueError(f"Unsupported syntax: {type(node).__name__}")

    return evaluate(ast.parse(expression, mode="eval").body)


class Workspace:
    """File tools confined to one directory."""

    def __init__(self, root, max_chars=20_000):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_chars = max_chars

    def _resolve(self, relative_path):
        path = (self.root / relative_path).resolve()
        if not path.is_relative_to(self.root):
            raise PermissionError(f"{relative_path} is outside workspace")
        return path

    def read_file(self, path):
        return self._resolve(path).read_text()[: self.max_chars]

    def write_file(self, path, content):
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        return f"Wrote {len(content)} characters to {path}"


def make_web_search(backend, max_snippet_chars=300):
    """backend(query) -> list of {"title", "url", "snippet"} dicts."""

    def web_search(query, max_results=5):
        results = backend(query)[:max_results]
        return [
            {"title": r["title"], "url": r["url"],
             "snippet": r["snippet"][:max_snippet_chars]}
            for r in results
        ]

    return web_search


@dataclass
class Tool:
    fn: Callable[..., Any]
    schema: dict

    @property
    def name(self):
        return self.schema["name"]

    def __call__(self, **kwargs):
        return self.fn(**kwargs)


def tool(fn=None, *, name=None, description=None):
    """Build a Tool whose schema comes from the function's signature."""

    def wrap(f):
        fields = {}
        for param in inspect.signature(f).parameters.values():
            annotation = (str if param.annotation is inspect.Parameter.empty
                          else param.annotation)
            default = (... if param.default is inspect.Parameter.empty
                       else param.default)
            fields[param.name] = (annotation, default)
        args_model = create_model(
            f"{f.__name__}_args", __config__=ConfigDict(extra="forbid"),
            **fields,
        )
        doc = description or inspect.getdoc(f)
        if not doc:
            raise ValueError(f"{f.__name__} needs a docstring")
        return Tool(f, {"name": name or f.__name__, "description": doc,
                        "input_schema": model_schema(args_model)})

    return wrap(fn) if fn else wrap


class Toolbox:
    def __init__(self, *tools):
        self.functions, self.schemas = {}, []
        for t in tools:
            self.add(t)

    def add(self, t):
        if t.name in self.functions:
            raise ValueError(f"Duplicate tool name: {t.name}")
        self.functions[t.name] = t.fn
        self.schemas.append(t.schema)
        return t


calculator_tool = Tool(calculator, CALCULATOR_SCHEMA)
