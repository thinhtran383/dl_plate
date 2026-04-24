import base64
import cv2
import numpy as np
from io import BytesIO

from PIL import Image
from fastapi.responses import JSONResponse
from starlette.datastructures import UploadFile


# ---------------------------------------------------------------------------
# Image conversion
# ---------------------------------------------------------------------------

def convert_base64_to_image(base64_string: str) -> np.ndarray:
    try:
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
            
        # Sửa lỗi bị null do URL decoded nhầm '+' thành ' '
        base64_string = base64_string.replace(' ', '+')
        
        img_data = base64.b64decode(base64_string)
        nparr = np.frombuffer(img_data, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Error decoding base64: {e}")
        return None


def im_2_b64(image_pil: Image.Image) -> str:
    """
    Chuyen PIL Image sang chuoi base64 JPEG.
    Ap dung OUTPUT_MAX_WIDTH va OUTPUT_JPEG_QUALITY neu duoc config trong .env.
    """
    try:
        from .config import settings

        # Option B: scale down neu anh rong hon gioi han
        max_w = settings.OUTPUT_MAX_WIDTH
        if max_w > 0 and image_pil.width > max_w:
            ratio     = max_w / image_pil.width
            new_size  = (max_w, int(image_pil.height * ratio))
            image_pil = image_pil.resize(new_size, Image.LANCZOS)

        # Option A: tuy chinh JPEG quality
        quality = settings.OUTPUT_JPEG_QUALITY if settings.OUTPUT_JPEG_QUALITY > 0 else 75

        buff = BytesIO()
        image_pil.save(buff, format="JPEG", quality=quality, optimize=True)
        return base64.b64encode(buff.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"Error encoding PIL to base64: {e}")
        return ""


# ---------------------------------------------------------------------------
# Image compositing
# ---------------------------------------------------------------------------

def paste_img_2_img(vehicle_img: np.ndarray, plate_img: np.ndarray) -> Image.Image:
    """Dán ảnh biển số gốc lên góc phải-dưới ảnh xe theo tỷ lệ."""
    veh_rgb = cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2RGB)
    plt_rgb = cv2.cvtColor(plate_img,   cv2.COLOR_BGR2RGB)

    veh_pil = Image.fromarray(veh_rgb)
    plt_pil = Image.fromarray(plt_rgb)

    from ..core.config import settings
    ratio  = veh_pil.size[0] / settings.PLATE_OVERLAY_DIVISOR / plt_pil.size[0]
    width  = int(plt_pil.size[0] * ratio)
    height = int(plt_pil.size[1] * ratio)

    plate_resized = plt_pil.resize((width, height), Image.LANCZOS)

    x = veh_pil.size[0] - width
    y = veh_pil.size[1] - height
    veh_pil.paste(plate_resized, (x, y))
    return veh_pil



def create_response(
    plate_number: str = "",
    base64_image: str = "",
    confident: float = 0.0,
) -> JSONResponse:
    return JSONResponse(
        content={
            "confident":     confident,
            "elapsed_time":  "",
            "model":         "xs-v2-global",
            "number":        plate_number,
            "plate_color":   "",
            "plate_type":    "",
            "request_id":    "",
            "base64_image":  base64_image,
            "response_code": 0,
        },
        status_code=200,
    )


# ---------------------------------------------------------------------------
# Form field helper
# ---------------------------------------------------------------------------

async def extract_image_from_form_field(field_data) -> np.ndarray:
    """Đọc ảnh từ UploadFile hoặc chuỗi base64."""
    if field_data is None:
        return None
    if isinstance(field_data, UploadFile):
        file_bytes = await field_data.read()
        if not file_bytes:
            return None
        nparr = np.frombuffer(file_bytes, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    elif isinstance(field_data, str):
        return convert_base64_to_image(field_data)
    return None


# ---------------------------------------------------------------------------
# ROI helpers
# ---------------------------------------------------------------------------

def parse_roi(roi_raw) -> list:
    """
    Parse ROI từ client thành list [[x,y],...] tọa độ chuẩn hóa 0-1.

    Chấp nhận:
        - JSON string: '[[0.1,0.1],[0.9,0.1],[0.9,0.9],[0.1,0.9]]'
        - List đã parse sẵn
        - None / chuoi rong -> tra None (khong ap ROI, xu ly toan anh)

    Returns:
        list[list[float]] hoặc None
    """
    import json
    if roi_raw is None:
        return None
    if isinstance(roi_raw, list):
        return roi_raw if len(roi_raw) >= 3 else None
    if isinstance(roi_raw, str):
        roi_raw = roi_raw.strip()
        if not roi_raw or roi_raw in ('null', '[]', ''):
            return None
        try:
            pts = json.loads(roi_raw)
            return pts if isinstance(pts, list) and len(pts) >= 3 else None
        except Exception:
            return None
    return None


def point_in_roi(nx: float, ny: float, roi_points: list) -> bool:
    """
    Kiểm tra điểm (nx, ny) tọa độ chuẩn hóa 0-1 có trong polygon ROI không.

    Args:
        nx, ny:     Tọa độ đã chuẩn hóa (0-1)
        roi_points: [[x,y],...] chuẩn hóa của polygon

    Returns:
        True nếu điểm trong hoặc trên cạnh polygon
    """
    pts = np.array(roi_points, dtype=np.float32)
    result = cv2.pointPolygonTest(pts, (float(nx), float(ny)), measureDist=False)
    return result >= 0

