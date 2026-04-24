import logging
import traceback
from typing import List
import numpy as np
import cv2

from ..core.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Detection Service (YOLOv8 singleton via ultralytics)
# ---------------------------------------------------------------------------

class DetectionService:
    def __init__(self):
        self._model = None
        self._load_model()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self):
        msg = f"Loading YOLOv8 model from: {settings.abs_detection_model}"
        print(msg)
        logger.info(msg)
        try:
            from ultralytics import YOLO
            self._model = YOLO(settings.abs_detection_model)
            logger.info("YOLOv8 model loaded OK.")
            print("YOLOv8 model loaded OK.")
        except Exception as e:
            logger.error(f"Error loading YOLOv8 model: {e}", exc_info=True)
            print(f"Error loading YOLOv8 model: {e}")
            traceback.print_exc()
            self._model = None

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def detect_plate(
        self,
        img: np.ndarray,
        roi_points: list = None,
    ) -> List[np.ndarray]:
        """
        Chạy YOLOv8 trên ảnh, trả về danh sách crop biển số (numpy BGR).

        Conf threshold đọc từ settings.DETECTION_CONF (.env: DETECTION_CONF).

        Args:
            img:        OpenCV BGR image
            roi_points: List [[x,y],...] tọa độ chuẩn hóa 0-1 của polygon ROI.
                        Neu None -> xu ly toan anh.
                        Neu co -> chi giu detection co tam bounding box nam trong ROI.

        Returns:
            list[np.ndarray] – các crop biển số tìm được
        """
        if self._model is None:
            logger.error("[DETECT] Model chua duoc load - bo qua inference.")
            return []

        from ..core.utils import point_in_roi

        conf_threshold = settings.DETECTION_CONF
        h, w = img.shape[:2]

        # --- Resize anh de YOLO chay nhanh hon ---
        target_w = settings.DETECTION_INPUT_WIDTH
        if target_w > 0 and w > target_w:
            scale = target_w / w
            det_img = cv2.resize(img, (target_w, int(h * scale)), interpolation=cv2.INTER_AREA)
            sx = w / det_img.shape[1]   # he so scale nguoc lai (x)
            sy = h / det_img.shape[0]   # he so scale nguoc lai (y)
            logger.debug(f"[DETECT] Resize {w}x{h} -> {det_img.shape[1]}x{det_img.shape[0]} cho YOLO (scale={scale:.2f})")
        else:
            det_img = img
            sx = sy = 1.0

        logger.debug(
            f"[DETECT] Bat dau inference - anh {w}x{h}, "
            f"conf={conf_threshold} iou={settings.DETECTION_IOU}, "
            f"roi={'co' if roi_points else 'khong'}"
        )

        results = self._model(
            det_img,
            conf=conf_threshold,
            iou=settings.DETECTION_IOU,
            verbose=False,
        )

        crops = []
        total_raw = 0

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            xyxy  = boxes.xyxy.cpu().numpy()   # shape (N, 4)
            confs = boxes.conf.cpu().numpy()   # shape (N,)
            total_raw += len(xyxy)

            for i, ((x_min, y_min, x_max, y_max), conf) in enumerate(zip(xyxy, confs)):
                # Scale toa do bbox tu anh nho ve anh goc
                x_min = x_min * sx
                y_min = y_min * sy
                x_max = x_max * sx
                y_max = y_max * sy

                bw = int(x_max - x_min)
                bh = int(y_max - y_min)
                logger.debug(
                    f"[DETECT] Box #{i}: xyxy=({int(x_min)},{int(y_min)},{int(x_max)},{int(y_max)}) "
                    f"size={bw}x{bh} conf={conf:.4f}"
                )

                # Kiểm tra ROI (tọa độ tâm chuẩn hóa)
                if roi_points is not None:
                    cx_norm = ((x_min + x_max) / 2) / w
                    cy_norm = ((y_min + y_max) / 2) / h
                    in_roi = point_in_roi(cx_norm, cy_norm, roi_points)
                    logger.debug(
                        f"[DETECT] Box #{i} - tam norm=({cx_norm:.3f},{cy_norm:.3f}) "
                        f"{'[IN-ROI]' if in_roi else '[OUT-ROI] bo qua'}"
                    )
                    if not in_roi:
                        continue

                # Mo rong bbox theo DETECTION_BBOX_PADDING
                pad = settings.DETECTION_BBOX_PADDING
                bw_pad = int((x_max - x_min) * pad)
                bh_pad = int((y_max - y_min) * pad)
                x1 = max(0,   int(x_min) - bw_pad)
                y1 = max(0,   int(y_min) - bh_pad)
                x2 = min(w,   int(x_max) + bw_pad)
                y2 = min(h,   int(y_max) + bh_pad)

                if pad > 0:
                    logger.debug(
                        f"[DETECT] Box #{i} padding {pad*100:.0f}%: "
                        f"({int(x_min)},{int(y_min)},{int(x_max)},{int(y_max)}) "
                        f"-> ({x1},{y1},{x2},{y2})"
                    )

                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    logger.warning(f"[DETECT] Box #{i} tao ra crop rong - bo qua.")
                    continue

                crops.append(crop)
                logger.debug(f"[DETECT] Box #{i} -> crop {x2-x1}x{y2-y1} px dua vao OCR.")

        # --- Tổng kết ---
        if total_raw == 0:
            logger.warning(
                f"[DETECT] YOLO không phát hiện được biển số nào trên ảnh {w}x{h}. "
                f"(conf={conf_threshold}, roi={'on' if roi_points else 'off'})"
            )
        elif not crops:
            logger.warning(
                f"[DETECT] YOLO tìm thấy {total_raw} box nhưng tất cả bị lọc bỏ "
                f"(ROI). Không có crop nào để OCR."
            )
        else:
            logger.debug(f"[DETECT] Done - {len(crops)}/{total_raw} box duoc giu lai de OCR.")

        return crops


# Singleton
detection_service = DetectionService()
