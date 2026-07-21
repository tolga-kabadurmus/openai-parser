from dataclasses import dataclass,field
from typing import Any,Dict,List,Optional
from ..enums import Endpoint,Modality,Provider,ResponseKind,StreamMode
from ..utils.text_utils import join_non_empty
from .audio import NormalizedAudio
from .base import SerializableModel
from .embedding import NormalizedEmbedding
from .event import NormalizedEvent
from .image import NormalizedImage
from .reasoning import NormalizedReasoning
from .request import NormalizedRequest
from .tool_call import NormalizedToolCall
from .usage import NormalizedUsage
@dataclass
class NormalizedResponse(SerializableModel):
    provider:Provider=Provider.UNKNOWN; endpoint:Endpoint=Endpoint.UNKNOWN; response_kind:ResponseKind=ResponseKind.UNKNOWN; stream_mode:StreamMode=StreamMode.NON_STREAMING
    modalities:List[Modality]=field(default_factory=list); request:Optional[NormalizedRequest]=None; response_id:Optional[str]=None; model:Optional[str]=None; created_at:Optional[str]=None
    status:Optional[str]=None; finish_reason:Optional[str]=None; visible_text:str=""; reasoning:NormalizedReasoning=field(default_factory=NormalizedReasoning)
    tool_calls:List[NormalizedToolCall]=field(default_factory=list); audio:List[NormalizedAudio]=field(default_factory=list); images:List[NormalizedImage]=field(default_factory=list)
    embeddings:List[NormalizedEmbedding]=field(default_factory=list); events:List[NormalizedEvent]=field(default_factory=list); usage:NormalizedUsage=field(default_factory=NormalizedUsage)
    error:Optional[Dict[str,Any]]=None; metadata:Dict[str,Any]=field(default_factory=dict); raw_response:Any=None
    def normalized_text(self,include_reasoning:bool=False,include_tools:bool=False)->str:
        parts=[self.visible_text]
        if include_reasoning:parts.append(self.reasoning.text)
        if include_tools:parts.extend(c.arguments_text for c in self.tool_calls)
        if not self.visible_text and self.audio:parts.extend(a.transcript for a in self.audio)
        if not self.visible_text and self.embeddings:parts.extend(",".join(str(v) for v in e.values) for e in self.embeddings)
        return join_non_empty(parts)
NormalizedOpenAIResponse=NormalizedResponse
