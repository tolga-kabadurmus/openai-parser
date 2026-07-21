from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Mapping


def sdk_object_to_python(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Mapping):
        return {
            str(key): sdk_object_to_python(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [sdk_object_to_python(item) for item in value]

    if is_dataclass(value):
        return sdk_object_to_python(asdict(value))

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return sdk_object_to_python(
                model_dump(mode="python", exclude_none=False)
            )
        except TypeError:
            return sdk_object_to_python(model_dump())

    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return sdk_object_to_python(to_dict())

    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, dict):
        return {
            key: sdk_object_to_python(item)
            for key, item in attributes.items()
            if not key.startswith("_")
        }

    return value
