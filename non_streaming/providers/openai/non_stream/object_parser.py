from __future__ import annotations

from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any, Dict, Mapping


class OpenAIObjectParser:
    """
    Converts OpenAI SDK response/request objects into plain Python structures.

    Supports:
      - dictionaries and mappings
      - Pydantic v2 `model_dump()`
      - Pydantic v1 `dict()`
      - OpenAI SDK `to_dict()`
      - dataclasses
      - simple Python objects with public attributes
    """

    def to_dict(self, value: Any) -> Dict[str, Any]:
        converted = self.to_python(value)
        return converted if isinstance(converted, dict) else {}

    def to_python(self, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value

        if isinstance(value, Enum):
            return value.value

        if isinstance(value, Mapping):
            return {
                str(key): self.to_python(item)
                for key, item in value.items()
            }

        if isinstance(value, (list, tuple, set)):
            return [self.to_python(item) for item in value]

        if is_dataclass(value):
            return self.to_python(asdict(value))

        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            try:
                return self.to_python(
                    model_dump(mode="python", exclude_none=False)
                )
            except TypeError:
                return self.to_python(model_dump())

        dict_method = getattr(value, "dict", None)
        if callable(dict_method):
            try:
                return self.to_python(dict_method(exclude_none=False))
            except TypeError:
                return self.to_python(dict_method())

        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            return self.to_python(to_dict())

        attributes = getattr(value, "__dict__", None)
        if isinstance(attributes, dict):
            return {
                str(key): self.to_python(item)
                for key, item in attributes.items()
                if not str(key).startswith("_")
            }

        return value
