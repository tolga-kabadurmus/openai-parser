from typing import Any,Mapping,Optional,Union
from .enums import Provider
from .exceptions import ProviderDetectionError
class ProviderDispatcher:
    _ALIASES={"openai":Provider.OPENAI,"anthropic":Provider.ANTHROPIC,"cohere":Provider.COHERE}
    @classmethod
    def detect(cls,value:Any,explicit_provider:Optional[Union[Provider,str]]=None,strict:bool=False)->Provider:
        if explicit_provider is not None:
            p=explicit_provider if isinstance(explicit_provider,Provider) else cls._ALIASES.get(str(explicit_provider).strip().lower(),Provider.UNKNOWN)
            if p is not Provider.UNKNOWN:return p
        if isinstance(value,Mapping):
            hint=str(value.get("provider","")).lower()
            if hint in cls._ALIASES:return cls._ALIASES[hint]
            joined=" ".join(str(value.get(k,"")).lower() for k in ("object","type","api","endpoint"))
            if any(t in joined for t in ("chat.completion","response.","openai")):return Provider.OPENAI
            if "anthropic" in joined or "content_block_" in joined:return Provider.ANTHROPIC
            if "cohere" in joined:return Provider.COHERE
        ident=f"{type(value).__module__}.{type(value).__qualname__}".lower()
        for name,p in cls._ALIASES.items():
            if name in ident:return p
        if strict:raise ProviderDetectionError(f"Could not detect SDK provider from {type(value)!r}")
        return Provider.UNKNOWN
