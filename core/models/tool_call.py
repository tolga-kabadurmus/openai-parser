from dataclasses import dataclass
from typing import Any,Dict,Optional
from .base import SerializableModel
@dataclass
class NormalizedToolCall(SerializableModel):
    id:Optional[str]=None; name:Optional[str]=None; arguments_text:str=""; arguments:Optional[Dict[str,Any]]=None; call_type:str="function"; status:Optional[str]=None; raw:Any=None
