import json
from typing import Any,Iterable,Mapping
_TEXT_KEYS=("text","content","input_text","output_text","transcript")
def join_non_empty(parts:Iterable[Any],separator:str=" ")->str:
    return separator.join(str(p).strip() for p in parts if p is not None and str(p).strip())
def extract_text(value:Any)->str:
    if value is None:return ""
    if isinstance(value,str):return value
    if isinstance(value,(int,float,bool)):return str(value)
    if isinstance(value,Mapping):
        xs=[extract_text(value[k]) for k in _TEXT_KEYS if k in value]
        return join_non_empty(xs) if xs else json.dumps(value,ensure_ascii=False,default=str)
    if isinstance(value,(list,tuple)):return join_non_empty(extract_text(x) for x in value)
    dump=getattr(value,"model_dump",None)
    return extract_text(dump()) if callable(dump) else str(value)
