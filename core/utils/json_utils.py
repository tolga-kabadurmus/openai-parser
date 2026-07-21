from dataclasses import asdict,is_dataclass
from enum import Enum
from typing import Any
def to_json_safe(value:Any)->Any:
    if value is None or isinstance(value,(str,int,float,bool)): return value
    if isinstance(value,Enum): return value.value
    if is_dataclass(value): return to_json_safe(asdict(value))
    if isinstance(value,dict): return {str(k):to_json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple,set)): return [to_json_safe(v) for v in value]
    model_dump=getattr(value,"model_dump",None)
    if callable(model_dump): return to_json_safe(model_dump())
    to_dict=getattr(value,"to_dict",None)
    if callable(to_dict): return to_json_safe(to_dict())
    d=getattr(value,"__dict__",None)
    if isinstance(d,dict): return {str(k):to_json_safe(v) for k,v in d.items() if not str(k).startswith("_")}
    return str(value)
