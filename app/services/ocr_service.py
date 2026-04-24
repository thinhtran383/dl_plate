import logging
import yaml
from typing import List, Tuple
import numpy as np
import onnxruntime as ort
import cv2

from ..core.config import settings
from ..core.utils import paste_img_2_img, im_2_b64

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config & ONNX helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def prepare_input(gray: np.ndarray, cfg: dict) -> np.ndarray:
    """Resize + convert theo config rồi thêm batch dim."""
    h    = cfg['img_height']
    w    = cfg['img_width']
    mode = cfg.get('image_color_mode', 'rgb')

    resized = cv2.resize(gray, (w, h), interpolation=cv2.INTER_LINEAR)

    if mode == 'rgb':
        img = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    else:
        img = resized[:, :, np.newaxis]

    return np.expand_dims(img.astype(np.uint8), axis=0)


def decode(outputs: list, cfg: dict) -> Tuple[str, float]:
    """Giai ma outputs ONNX -> (plate_text, confidence)."""
    alphabet = cfg['alphabet']
    pad_char = cfg['pad_char']
    n_slots  = cfg['max_plate_slots']

    chars: List[str]   = []
    confs: List[float] = []

    if len(outputs) == n_slots:
        for slot in outputs:
            idx  = int(np.argmax(slot[0]))
            conf = float(np.max(slot[0]))
            chars.append(alphabet[idx])
            confs.append(conf)
    else:
        arr = outputs[0]
        if arr.ndim == 3:
            arr = arr[0]
        elif arr.ndim == 2 and arr.shape[0] == 1:
            arr = arr[0].reshape(n_slots, len(alphabet))
        indices     = np.argmax(arr, axis=-1)
        conf_scores = np.max(arr, axis=-1)
        chars = [alphabet[i] for i in indices]
        confs = conf_scores.tolist()

    valid_confs = [c for ch, c in zip(chars, confs) if ch != pad_char]
    plate       = ''.join(ch for ch in chars if ch != pad_char)
    confidence  = sum(valid_confs) / len(valid_confs) if valid_confs else 0.0
    return plate, round(confidence, 4)


# ---------------------------------------------------------------------------
# OCR Service (singleton, giữ session ONNX)
# ---------------------------------------------------------------------------

class OCRService:
    def __init__(self):
        import os
        config_path = os.path.join(
            settings.abs_ocr_model_dir,
            'xs_v2_global_plate_config.yaml',
        )
        model_path = os.path.join(
            settings.abs_ocr_model_dir,
            'xs_v2_global.onnx',
        )

        print("Loading ONNX OCR model...")
        logger.info("Loading ONNX OCR model...")
        try:
            self._cfg     = load_config(config_path)
            self._session = ort.InferenceSession(
                model_path, providers=['CPUExecutionProvider']
            )
            logger.info("OCR model loaded OK.")
            print("OCR model loaded OK.")
        except Exception as e:
            logger.error(f"Error loading OCR model: {e}", exc_info=True)
            print(f"Error loading OCR model: {e}")
            self._session = None
            self._cfg     = None

    # ------------------------------------------------------------------

    def run(
        self,
        vehicle_image: np.ndarray,
        plate_images:  List[np.ndarray],
    ) -> Tuple[str, str, float]:
        """
        Chạy OCR trên danh sách crop biển số.

        Returns:
            (plate_number, base64_image, confidence)
        """
        if self._session is None or self._cfg is None:
            logger.error("[OCR] Session ONNX chua san sang - bo qua OCR.")
            return '', '', 0.0

        logger.debug(f"[OCR] Bat dau OCR - co {len(plate_images)} crop bien so.")

        plate_number    = ''
        base64_image    = ''
        best_confidence = 0.0

        for idx, plate_img in enumerate(plate_images):
            crop_h, crop_w = plate_img.shape[:2]
            logger.debug(f"[OCR] Crop #{idx}: kích thước {crop_w}x{crop_h} px")
            try:
                gray   = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
                tensor = prepare_input(gray, self._cfg)
                logger.debug(f"[OCR] Crop #{idx}: tensor shape={tensor.shape}, dtype={tensor.dtype}")

                input_name = self._session.get_inputs()[0].name
                outputs    = self._session.run(None, {input_name: tensor})
                logger.debug(f"[OCR] Crop #{idx}: ONNX trả về {len(outputs)} output(s)")

                text, conf = decode(outputs, self._cfg)
                logger.debug(f"[OCR] Crop #{idx}: decoded='{text}' conf={conf:.4f} (len={len(text)})")

                if 5 < len(text) < 10:
                    logger.debug(f"[OCR] Crop #{idx}: [OK] Bien so hop le - '{text}' conf={conf:.4f}")
                    plate_number    = text
                    best_confidence = conf
                    final_pil       = paste_img_2_img(vehicle_image, plate_img)
                    base64_image    = im_2_b64(final_pil)
                    break
                else:
                    logger.warning(
                        f"[OCR] Crop #{idx}: [FAIL] Ket qua khong hop le - '{text}' "
                        f"(len={len(text)}, can 6-9 ky tu). Bo qua crop nay."
                    )

            except Exception as e:
                logger.error(f"[OCR] Crop #{idx}: Loi xu ly - {e}", exc_info=True)
                print(f"Lỗi OCR cho 1 crop: {e}")
                continue

        if 5 < len(plate_number) < 10:
            logger.info(f"[OCR] [OK] Ket qua cuoi: '{plate_number}' conf={best_confidence:.4f}")
            return plate_number, base64_image, best_confidence

        logger.warning(
            f"[OCR] [FAIL] Khong doc duoc bien so hop le sau khi xu ly {len(plate_images)} crop(s)."
        )
        return '', '', 0.0


# Singleton
ocr_service = OCRService()
