from pydantic import BaseModel

class MixtureProperty(BaseModel):
    """Represents a target property for an optimized mixture."""
    prop: str
    desired_prop: float