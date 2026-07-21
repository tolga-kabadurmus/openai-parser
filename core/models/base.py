from dataclasses import asdict,dataclass
from typing import Any,Dict
from ..utils.json_utils import to_json_safe
@dataclass
class SerializableModel:
    def to_dict(self,exclude_none:bool=True)->Dict[str,Any]:
        value=to_json_safe(asdict(self)); return _drop_none(value) if exclude_none else value
def _drop_none(value:Any)->Any:
    if isinstance(value,dict): return {k:_drop_none(v) for k,v in value.items() if v is not None}
    if isinstance(value,list): return [_drop_none(v) for v in value]
    return value
