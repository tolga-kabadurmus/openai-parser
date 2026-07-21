from dataclasses import dataclass,field
from typing import Any,List,Optional,Union
from .base import SerializableModel
EmbeddingValue=Union[float,int,str]
@dataclass
class NormalizedEmbedding(SerializableModel):
    index:Optional[int]=None; values:List[EmbeddingValue]=field(default_factory=list); encoding_format:Optional[str]=None; raw:Any=None
