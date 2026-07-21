from dataclasses import dataclass
from typing import Any,Optional
from .base import SerializableModel
@dataclass
class NormalizedImage(SerializableModel):
    image_id:Optional[str]=None; url:Optional[str]=None; base64_data:Optional[str]=None; mime_type:Optional[str]=None; revised_prompt:Optional[str]=None; raw:Any=None
