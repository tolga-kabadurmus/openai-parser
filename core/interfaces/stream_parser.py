from abc import ABC,abstractmethod
from typing import Sequence,Union
from ..models import NormalizedEvent,NormalizedResponse
class StreamParser(ABC):
    @abstractmethod
    def feed(self,raw_chunk:Union[str,bytes])->None:raise NotImplementedError
    @abstractmethod
    def events(self)->Sequence[NormalizedEvent]:raise NotImplementedError
    @abstractmethod
    def finalize(self)->NormalizedResponse:raise NotImplementedError
