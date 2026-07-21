from dataclasses import dataclass
from typing import Any,Optional
from .base import SerializableModel
@dataclass
class NormalizedAudio(SerializableModel):
    transcript:str=""; audio_id:Optional[str]=None; format:Optional[str]=None; mime_type:Optional[str]=None; data:Optional[str]=None; duration_seconds:Optional[float]=None; raw:Any=None
