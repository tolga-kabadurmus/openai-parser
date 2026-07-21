from dataclasses import dataclass,field
from typing import Any,Dict,Optional
from .base import SerializableModel
@dataclass
class NormalizedUsage(SerializableModel):
    input_tokens:Optional[int]=None; output_tokens:Optional[int]=None; total_tokens:Optional[int]=None
    cached_input_tokens:Optional[int]=None; reasoning_tokens:Optional[int]=None
    audio_input_tokens:Optional[int]=None; audio_output_tokens:Optional[int]=None
    raw:Dict[str,Any]=field(default_factory=dict)
    def __post_init__(self):
        if self.total_tokens is None and self.input_tokens is not None and self.output_tokens is not None:self.total_tokens=self.input_tokens+self.output_tokens
    @classmethod
    def from_openai(cls,usage:Optional[Dict[str,Any]]):
        if not usage:return cls()
        inp=usage.get("input_tokens",usage.get("prompt_tokens")); out=usage.get("output_tokens",usage.get("completion_tokens"))
        idet=usage.get("input_tokens_details") or usage.get("prompt_tokens_details") or {}
        odet=usage.get("output_tokens_details") or usage.get("completion_tokens_details") or {}
        return cls(_as_int(inp),_as_int(out),_as_int(usage.get("total_tokens")),_as_int(idet.get("cached_tokens")),_as_int(odet.get("reasoning_tokens")),_as_int(idet.get("audio_tokens")),_as_int(odet.get("audio_tokens")),dict(usage))
def _as_int(v):
    if isinstance(v,bool):return None
    try:return int(v) if v is not None else None
    except (TypeError,ValueError):return None
