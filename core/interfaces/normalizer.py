from abc import ABC,abstractmethod
from typing import Any,List
from ..models import NormalizedEvent
class Normalizer(ABC):
    @abstractmethod
    def normalize(self,value:Any)->List[NormalizedEvent]:raise NotImplementedError
