from pydantic import BaseModel

class MixtureProperty(BaseModel):
    prop: str
    desired_prop: float