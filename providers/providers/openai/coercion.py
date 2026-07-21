from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Mapping


def to_mapping(value: Any) -> Dict[str, Any]:
    """
    Convert raw dictionaries, Pydantic objects, OpenAI SDK models, and simple
    Python objects into a mutable dictionary.
    """
    if value is None:
        return {}

    if isinstance(value, dict):
        return dict(value)

    if isinstance(value, Mapping):
        return dict(value)

    if is_dataclass(value):
        return asdict(value)

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        return dict(dumped) if isinstance(dumped, Mapping) else {}

    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        dumped = to_dict()
        return dict(dumped) if isinstance(dumped, Mapping) else {}

    obj_dict = getattr(value, "__dict__", None)
    if isinstance(obj_dict, dict):
        return {
            key: item
            for key, item in obj_dict.items()
            if not str(key).startswith("_")
        }

    return {}


def get_value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)
