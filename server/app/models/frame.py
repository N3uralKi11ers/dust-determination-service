from typing import List
from pydantic import BaseModel

class FrameBase(BaseModel):
  name: str
  time: float
  percent: float
  

class Frames(BaseModel):
  frames: List[FrameBase]
  