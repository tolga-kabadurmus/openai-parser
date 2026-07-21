from typing import Any,Dict,Protocol
from .enums import Provider
from .exceptions import ParserNotRegisteredError
class ParserConstructor(Protocol):
    def __call__(self,*args:Any,**kwargs:Any)->Any:...
class ParserFactory:
    _registry:Dict[Provider,ParserConstructor]={}
    @classmethod
    def register(cls,provider:Provider,parser:ParserConstructor)->None:
        if provider is Provider.UNKNOWN:raise ValueError("Cannot register Provider.UNKNOWN")
        cls._registry[provider]=parser
    @classmethod
    def unregister(cls,provider:Provider)->None:cls._registry.pop(provider,None)
    @classmethod
    def create(cls,provider:Provider,*args:Any,**kwargs:Any)->Any:
        parser=cls._registry.get(provider)
        if parser is None:raise ParserNotRegisteredError(f"No parser registered for {provider.value}")
        return parser(*args,**kwargs)
    @classmethod
    def registered_providers(cls):return tuple(cls._registry.keys())
