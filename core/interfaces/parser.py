from abc import ABC,abstractmethod
from typing import Any
from ..models import NormalizedRequest,NormalizedResponse
class Parser(ABC):
    @abstractmethod
    def normalize_request(self,payload:Any)->NormalizedRequest:raise NotImplementedError
    @abstractmethod
    def normalize_response(self,payload:Any)->NormalizedResponse:raise NotImplementedError
