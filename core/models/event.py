from dataclasses import dataclass,field
from typing import Any,Dict,Optional
from ..enums import EventType,Modality
from .base import SerializableModel
@dataclass
class NormalizedEvent(SerializableModel):
    event_type:EventType; modality:Modality=Modality.UNKNOWN; text:str=""; item_id:Optional[str]=None; sequence_number:Optional[int]=None
    tool_call_id:Optional[str]=None; tool_name:Optional[str]=None; metadata:Dict[str,Any]=field(default_factory=dict); raw:Any=None
