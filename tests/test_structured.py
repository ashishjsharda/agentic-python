"""Chapter 3 and 4: schemas, validation-and-repair, generated tools."""
from typing import Annotated, Literal

import pytest
from pydantic import BaseModel, Field

from agentic.structured import (ExtractionFailed, extract, inline_refs,
                                model_schema)
from agentic.testing import FakeClient, tool_use
from agentic.tools import Toolbox, execute_tool_call, tool


class Line(BaseModel):
    sku: str
    quantity: int = Field(ge=1)


class Order(BaseModel):
    customer: str
    priority: Literal["low", "normal", "urgent"]
    lines: list[Line] = Field(min_length=1)


def test_inline_refs_removes_defs():
    schema = model_schema(Order)
    assert "$defs" not in str(schema) and "$ref" not in str(schema)
    item = schema["properties"]["lines"]["items"]
    assert item["properties"]["quantity"]["minimum"] == 1


def test_inline_refs_leaves_plain_schemas_alone():
    plain = {"type": "object", "properties": {"a": {"type": "string"}}}
    assert inline_refs(plain) == plain


def test_extract_forces_the_tool_and_validates():
    fake = FakeClient([tool_use("record", {
        "customer": "Acme", "priority": "urgent",
        "lines": [{"sku": "A-1", "quantity": 2}]})])
    order = extract(fake, Order, "Acme needs 2 of A-1 ASAP")
    assert order.priority == "urgent" and order.lines[0].quantity == 2
    assert fake.calls[0]["tool_choice"] == {"type": "tool",
                                            "name": "record"}


def test_extract_repairs_after_validation_error():
    fake = FakeClient([
        tool_use("record", {"customer": "Acme", "priority": "asap",
                            "lines": []}, id="r1"),
        tool_use("record", {"customer": "Acme", "priority": "urgent",
                            "lines": [{"sku": "A-1", "quantity": 1}]},
                 id="r2"),
    ])
    assert extract(fake, Order, "...").priority == "urgent"
    feedback = fake.calls[1]["messages"][-1]["content"][0]
    assert feedback["is_error"] and "priority" in feedback["content"]
    assert feedback["tool_use_id"] == "r1"


def test_extract_gives_up():
    fake = FakeClient([tool_use("record", {"customer": 1}, id=f"r{i}")
                       for i in range(2)])
    with pytest.raises(ExtractionFailed):
        extract(fake, Order, "...", max_attempts=2)


def test_tool_decorator_builds_strict_schema_from_signature():
    @tool
    def lookup(city: Annotated[str, Field(description="City name")],
               units: Literal["c", "f"] = "c"):
        """Look up the weather."""
        return f"{city}:{units}"

    schema = lookup.schema["input_schema"]
    assert lookup.schema["description"] == "Look up the weather."
    assert schema["required"] == ["city"]
    assert schema["properties"]["units"]["enum"] == ["c", "f"]
    assert schema["additionalProperties"] is False

    box = Toolbox(lookup)
    schemas = {s["name"]: s for s in box.schemas}
    ok = execute_tool_call({"name": "lookup", "input": {"city": "Oslo"}},
                           box.functions, schemas)
    assert ok == "Oslo:c"
    bad = execute_tool_call(
        {"name": "lookup", "input": {"city": "Oslo", "units": "k"}},
        box.functions, schemas)
    assert "Invalid arguments" in bad["error"]


def test_tool_requires_docstring_and_unique_names():
    def undocumented(x: int):
        return x

    with pytest.raises(ValueError):
        tool(undocumented)

    @tool
    def a():
        """A."""

    with pytest.raises(ValueError):
        Toolbox(a, a)
