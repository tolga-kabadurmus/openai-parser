from dataclasses import dataclass,field
from typing import Any,Dict,List,Optional
from ..enums import Endpoint,Modality,Provider,ResponseKind,StreamMode
from ..utils.text_utils import join_non_empty
from .base import SerializableModel
from .message import NormalizedMessage
@dataclass
class NormalizedRequest(SerializableModel):
    provider:Provider=Provider.UNKNOWN; endpoint:Endpoint=Endpoint.UNKNOWN; response_kind:ResponseKind=ResponseKind.UNKNOWN; stream_mode:StreamMode=StreamMode.NON_STREAMING
    modalities:List[Modality]=field(default_factory=list); model:Optional[str]=None; instructions:str=""; input_text:str=""; messages:List[NormalizedMessage]=field(default_factory=list)
    tools:List[Dict[str,Any]]=field(default_factory=list); tool_choice:Any=None; temperature:Optional[float]=None; top_p:Optional[float]=None; max_output_tokens:Optional[int]=None
    user:Optional[str]=None; metadata:Dict[str,Any]=field(default_factory=dict); raw_request:Any=None
    def normalized_text(self)->str:return join_non_empty([self.instructions,*(m.text for m in self.messages),self.input_text])
