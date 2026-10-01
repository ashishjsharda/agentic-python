"""Chapter 3: structured outputs with schemas, validation, and repair."""
import copy
import json

from pydantic import BaseModel, ValidationError


class ExtractionFailed(Exception):
    pass


def inline_refs(schema):
    """Replace "$ref" pointers with their definitions.

    Pydantic puts nested models under "$defs". Inlining them gives every
    provider the same self-contained schema.
    """
    schema = copy.deepcopy(schema)
    defs = schema.pop("$defs", {})

    def resolve(node):
        if isinstance(node, dict):
            if "$ref" in node:
                name = node["$ref"].split("/")[-1]
                return resolve(copy.deepcopy(defs[name]))
            return {k: resolve(v) for k, v in node.items()}
        if isinstance(node, list):
            return [resolve(v) for v in node]
        return node

    return resolve(schema)


def model_schema(model_cls):
    schema = inline_refs(model_cls.model_json_schema())
    schema.pop("title", None)
    return schema


def schema_tool(model_cls, name, description):
    return {"name": name, "description": description,
            "input_schema": model_schema(model_cls)}


def parse_json_object(text):
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object in model output: {text[:200]}")
    return json.loads(text[start:end + 1])


def extract(client, model_cls: type[BaseModel], prompt, system=None,
            max_attempts=3, name="record",
            description="Record the extracted data."):
    tool = schema_tool(model_cls, name, description)
    messages = [{"role": "user", "content": prompt}]
    for _ in range(max_attempts):
        response = client.complete(
            messages, tools=[tool], system=system,
            tool_choice={"type": "tool", "name": name},
        )
        messages.append({"role": "assistant",
                         "content": response["content"]})
        call = next((b for b in response["content"]
                     if b["type"] == "tool_use"), None)
        if call is None:
            messages.append({"role": "user",
                             "content": f"Call the {name} tool."})
            continue
        try:
            return model_cls.model_validate(call["input"])
        except ValidationError as e:
            messages.append({"role": "user", "content": [{
                "type": "tool_result", "tool_use_id": call["id"],
                "content": f"Validation failed. Fix these errors:\n{e}",
                "is_error": True,
            }]})
    raise ExtractionFailed(
        f"No valid {model_cls.__name__} after {max_attempts} attempts"
    )
