from pydantic import BaseModel

class MaterialProperty(BaseModel):
    prop: str
    upper_bound: float
    lower_bound: float