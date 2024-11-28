from pydantic import BaseModel

from typing import Optional

class GenerateRequest(BaseModel):
    model: Optional[str] = None
    prompts: list[str]

