import base64
import time
import cv2
import numpy as np
from enum import Enum
from fastapi import APIRouter, Request, File, Form
from fastapi import UploadFile as FUploadFile
from typing import Optional, Tuple, Union, List, Any
from fastapi.responses import JSONResponse
import logging
import json
import re

from ..core.utils import create_response, extract_image_from_form_field, parse_roi
from ..services.detection_service import detection_service
from ..services.ocr_service import ocr_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["License Plate Detection"])

def _log_in_out(endpoint: str, req_data: dict, response: JSONResponse):
    """Log request IN và OUT. Lọc các trường ảnh/base64 dài để tránh rác log."""
    # Sanitize Request IN
    cleaned_req = {}
    items = req_data.multi_items() if hasattr(req_data, 'multi_items') else req_data.items()
    for k, v in items:
        if hasattr(v, 'filename') and v.filename is not None:
            sz = getattr(v, 'size', 'unknown')
            cleaned_req[k] = f"<File: {v.filename} - Size: {sz}>"
        elif isinstance(v, str) and len(v) > 200:
            cleaned_req[k] = f"<String/Base64 length={len(v)}>"
        elif hasattr(v, 'filename'): # UploadFile but no filename or another type
             cleaned_req[k] = f"<File Object>"
        else:
            cleaned_req[k] = v

    # Sanitize Response OUT
    try:
        res_dict = json.loads(response.body.decode('utf-8'))
        if 'base64_image' in res_dict and isinstance(res_dict['base64_image'], str):
            len_b64 = len(res_dict['base64_image'])
            res_dict['base64_image'] = f"<Base64 length={len_b64}>" if len_b64 > 0 else ""
    except Exception:
        res_dict = "<Unparseable Response>"

    req_msg = f"[{endpoint}] REQUEST IN : {cleaned_req}"
    res_msg = f"[{endpoint}] RESPONSE OUT: {res_dict}"

    # Ghi ra file (.log handler đã config trong root)
    logger.info(req_msg)
    logger.info(res_msg)
    
    # Ghi ra terminal (stdout)
    print(req_msg)
    print(res_msg)




# ---------------------------------------------------------------------------
# Models & Enums
# ---------------------------------------------------------------------------

class RoiPreset(str, Enum):
    none        = "none"         # Không áp ROI – YOLO chạy toàn ảnh
    top_half    = "top_half"     # Nua tren (y: 0 -> 0.5)
    bottom_half = "bottom_half"  # Nua duoi (y: 0.5 -> 1)
    left_half   = "left_half"    # Nua trai  (x: 0 -> 0.5)
    right_half  = "right_half"   # Nua phai  (x: 0.5 -> 1)
    center      = "center"       # Vùng trung tâm (25%–75%)
    top_center  = "top_center"   # Trên-giữa (x:20%–80%, y:0–40%)
    custom      = "custom"       # Tự nhập JSON vào field roi_json bên dưới


# tọa độ chuẩn hóa 0-1: [[x,y], ...]
_ROI_PRESETS: dict = {
    "none":        None,
    "top_half":    [[0.0, 0.0], [1.0, 0.0], [1.0, 0.5], [0.0, 0.5]],
    "bottom_half": [[0.0, 0.5], [1.0, 0.5], [1.0, 1.0], [0.0, 1.0]],
    "left_half":   [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]],
    "right_half":  [[0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0]],
    "center":      [[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]],
    "top_center":  [[0.2, 0.0], [0.8, 0.0], [0.8, 0.4], [0.2, 0.4]],
    "custom":      None,  # sẽ đọc từ roi_json
}


def _resolve_roi(preset: RoiPreset, roi_json: Optional[str]) -> Optional[list]:
    """Chuyển preset + roi_json thành list tọa độ cuối cùng."""
    if preset == RoiPreset.custom:
        return parse_roi(roi_json)
    return _ROI_PRESETS.get(preset.value)


# ---------------------------------------------------------------------------
# Helper nội bộ
# ---------------------------------------------------------------------------

def _crop_to_b64(crop: np.ndarray) -> str:
    """Encode anh crop BGR -> base64 JPEG string."""
    _, buf = cv2.imencode(".jpg", crop)
    return base64.b64encode(buf.tobytes()).decode("utf-8")


async def _process_image(
    img: np.ndarray,
    only_crop: bool = False,
    roi_points: list = None,
) -> Tuple[str, str, float]:
    """
    Pipeline xu ly anh.

    Args:
        img:        Anh dau vao (OpenCV BGR)
        only_crop:  True -> chi crop bien so, khong OCR
        roi_points: [[x,y],...] toa do chuan hoa 0-1; None = toan anh

    Returns:
        (plate_text, base64_image, confidence)
    """
    if img is None:
        logger.warning("[PIPELINE] Anh dau vao la None - khong the xu ly.")
        return '', '', 0.0

    t_start = time.perf_counter()

    h, w = img.shape[:2]
    logger.debug(
        f"[PIPELINE] Nhan anh {w}x{h} px | only_crop={only_crop} | "
        f"roi={'on (' + str(len(roi_points)) + ' pts)' if roi_points else 'off'}"
    )

    plate_crops = detection_service.detect_plate(img, roi_points=roi_points)
    t_yolo = time.perf_counter()

    if not plate_crops:
        elapsed = int((t_yolo - t_start) * 1000)
        logger.warning(
            f"[PIPELINE] Khong co crop nao tu YOLO - pipeline dung lai, tra ve rong. "
            f"[YOLO={elapsed}ms]"
        )
        return '', '', 0.0

    logger.debug(f"[PIPELINE] YOLO tra ve {len(plate_crops)} crop - tiep tuc {'crop-only' if only_crop else 'OCR'}.")

    if only_crop:
        crop_b64 = _crop_to_b64(plate_crops[0])
        elapsed = int((time.perf_counter() - t_start) * 1000)
        logger.debug(f"[PIPELINE] only_crop=True - tra ve base64 crop, bo OCR. [total={elapsed}ms]")
        return '', crop_b64, 0.0

    plate_text, b64, conf = ocr_service.run(img, plate_crops)
    t_end = time.perf_counter()

    ms_yolo  = int((t_yolo - t_start) * 1000)
    ms_ocr   = int((t_end  - t_yolo)  * 1000)
    ms_total = int((t_end  - t_start) * 1000)

    logger.info(
        f"[PIPELINE] Done - plate='{plate_text}' conf={conf:.4f} | "
        f"YOLO={ms_yolo}ms | OCR={ms_ocr}ms | TOTAL={ms_total}ms"
    )
    return plate_text, b64, conf


def _create_crop_response(crop_b64: str) -> JSONResponse:
    """Response khi chỉ trả crop (onlyCrop=true)."""
    return JSONResponse(
        content={
            "confident":     0.0,
            "elapsed_time":  "",
            "model":         "crop-only",
            "number":        "",
            "plate_color":   "",
            "plate_type":    "",
            "request_id":    "",
            "base64_image":  crop_b64,
            "response_code": 500,
        },
        status_code=200,
    )


# ---------------------------------------------------------------------------
# Endpoint chính (tương thích hệ thống cũ, dùng JSON / form-data)
# ---------------------------------------------------------------------------

@router.post(
    "/detect_plate_base64", 
    summary="Nhận diện biển số (form-data, JSON hoặc Java toString)",
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "image": {"type": "string", "description": "Chuỗi base64 ảnh"},
                            "onlyCrop": {"type": "string", "description": "Truyền 'true' để chỉ crop biển số"},
                            "roi": {"type": "string", "description": "JSON string tọa độ ROI"}
                        }
                    }
                },
                "application/x-www-form-urlencoded": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "image": {"type": "string", "description": "Chuỗi base64 ảnh"},
                            "onlyCrop": {"type": "string", "description": "Truyền 'true' để crop"},
                            "roi": {"type": "string", "description": "JSON string tọa độ ROI"}
                        }
                    }
                }
            }
        }
    }
)
async def detect_plate_base64(request: Request):
    """
    Nhận diện biển số xe.
    Tương thích với định dạng `JSON`, `application/x-www-form-urlencoded`, 
    và hỗ trợ ngoại lệ khi Client Java gửi thẳng Object dạng `AnprRequest(image=..., onlyCrop=...)`.
    """
    try:
        body = await request.body()
        body_str = body.decode('utf-8', errors='ignore').strip()
        
        # LOG RAW BODY DE DEBUG
        logger.info(f"---- DEBUG RAW BODY ----\n{body_str[:500]} ...\n------------------------")
        
        form_data = {}
        
        if body_str.startswith("AnprRequest("):
          
            idx_img = body_str.find("image=")
            if idx_img != -1:
                start_img = idx_img + 6
                # Tìm field tiếp theo để cắt
                end_img = body_str.find(", onlyCrop=", start_img)
                if end_img == -1: end_img = body_str.find(", roi=", start_img)
                if end_img == -1: end_img = body_str.find(", rois=", start_img)
                if end_img == -1: end_img = body_str.rfind(")")
                
                if end_img != -1:
                    form_data['image'] = body_str[start_img:end_img].strip()
                    
            # onlyCrop
            idx_crop = body_str.find("onlyCrop=")
            if idx_crop != -1:
                start_crop = idx_crop + 9
                end_crop = body_str.find(",", start_crop)
                if end_crop == -1: end_crop = body_str.rfind(")")
                if end_crop != -1:
                    form_data['onlyCrop'] = body_str[start_crop:end_crop].strip()
                    
            # rois / roi
            idx_roi = body_str.find("rois=")
            if idx_roi == -1: idx_roi = body_str.find("roi=")
            if idx_roi != -1:
                start_roi = idx_roi + (5 if "rois=" in body_str else 4)
                end_roi = body_str.rfind(")")
                if end_roi != -1:
                    form_data['roi'] = body_str[start_roi:end_roi].strip()
        else:
            try:
                import json
                form_data = json.loads(body_str)
            except Exception:
                try:
                    # Nếu là form data chuẩn
                    from urllib.parse import parse_qsl
                    parsed = parse_qsl(body_str, keep_blank_values=True)
                    for k, v in parsed:
                        form_data[k] = v
                except Exception:
                    pass

        only_crop  = str(form_data.get('onlyCrop', '')).lower() in ('true', '1', 'yes')
        roi_points = parse_roi(form_data.get('roi') or form_data.get('rois'))

        response_out = None

        # Camera chính
        plate, b64, conf = await _process_image(
            await extract_image_from_form_field(form_data.get('image')),
            only_crop=only_crop,
            roi_points=roi_points,
        )
        if only_crop and b64:
            response_out = _create_crop_response(b64)
        elif len(plate) > 5:
            response_out = create_response(plate, b64, conf)

        if response_out is None:
            response_out = create_response()

        _log_in_out("detect_plate_base64", form_data, response_out)
        return response_out

    except Exception as e:
        print(f"API Error: {e}")
        logger.error(f"API Error [detect_plate_base64]: {e}")
        response_out = create_response()
        _log_in_out("detect_plate_base64", {"error": str(e)}, response_out)
        return response_out


# ---------------------------------------------------------------------------
# Endpoint Swagger UI – file upload + dropdown ROI preset
# ---------------------------------------------------------------------------

@router.post(
    "/detect_plate_test_ui",
    summary="Swagger UI Test – File Upload",
)
async def detect_plate_test_ui(
    image:      Optional[FUploadFile] = File(None,  description="Camera chính"),
    onlyCrop:   Optional[str]         = Form(None,  description="true = chỉ trả crop, false = full OCR"),
    roi_preset: RoiPreset             = Form(
        RoiPreset.none,
        description=(
            "Chọn vùng ROI preset. Nếu chọn **custom** thì điền tọa độ vào field `roi_json` bên dưới. "
            "Chọn **none** (mặc định) để YOLO chạy toàn ảnh."
        ),
    ),
    roi_json:   Optional[str]         = Form(
        None,
        description=(
            "Chỉ dùng khi roi_preset = custom. "
            "JSON polygon tọa độ chuẩn hóa 0-1. "
            "VD: [[0.1,0.1],[0.9,0.1],[0.9,0.9],[0.1,0.9]]"
        ),
    ),
):
    """
    Endpoint test trực tiếp trên `/docs`.

    - **`roi_preset`**: dropdown chọn vùng ROI có sẵn.
      - `none` -> YOLO chay toan anh (mac dinh).
      - Cac preset khac -> chi detect trong vung do.
      - `custom` -> dien JSON vao `roi_json`.
    - **`onlyCrop`**: `true` -> chi tra crop bien so, bo OCR.
    """
    only_crop  = str(onlyCrop or '').lower() in ('true', '1', 'yes')
    roi_points = _resolve_roi(roi_preset, roi_json)
    
    req_data = {
        "image": image,
        "onlyCrop": onlyCrop,
        "roi_preset": roi_preset,
        "roi_json": roi_json
    }
    
    response_out = None

    if image is not None:
        file_bytes = await image.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        plate, b64, conf = await _process_image(
            img, only_crop=only_crop, roi_points=roi_points
        )
        if only_crop and b64:
            response_out = _create_crop_response(b64)
        elif plate:
            response_out = create_response(plate, b64, conf)

    if response_out is None:
        response_out = create_response()
        
    _log_in_out("detect_plate_test_ui", req_data, response_out)
    return response_out
