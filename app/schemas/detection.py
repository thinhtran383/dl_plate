from typing import Optional, List
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class DetectionRequest(BaseModel):
    image_base64: str
    conf_threshold: Optional[float] = None
    iou_threshold: Optional[float] = None


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class PlateResult(BaseModel):
    plate_text: str
    confidence: float
    bbox: List[float]          # [x1, y1, x2, y2] normalized


class DetectionResponse(BaseModel):
    success: bool
    plates: List[PlateResult] = []
    message: Optional[str] = None
    processing_time_ms: Optional[float] = None
