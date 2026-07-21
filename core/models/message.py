from dataclasses import dataclass,field
from typing import Any,Dict,List,Optional
from .base import SerializableModel
@dataclass
class NormalizedMessage(SerializableModel):
    role:str; text:str=""; name:Optional[str]=None; content_parts:List[Dict[str,Any]]=field(default_factory=list); raw:Any=None
