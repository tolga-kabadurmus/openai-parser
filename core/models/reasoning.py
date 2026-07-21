from dataclasses import dataclass
from typing import Any,Optional
from .base import SerializableModel
@dataclass
class NormalizedReasoning(SerializableModel):
    text:str=""; summary:Optional[str]=None; encrypted_content:Optional[str]=None; raw:Any=None
