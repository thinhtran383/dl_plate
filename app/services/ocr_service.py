import logging
import yaml
from typing import List, Optional, Tuple
import numpy as np
import onnxruntime as ort
import cv2

from ..core.config import settings
from ..core.preprocess import generate_variants
from ..core.utils import (
    detect_frame_brand_band,
    extract_vn_plate_text,
    has_frame_brand_in_text,
    is_valid_vn_plate,
    paste_img_2_img,
    im_2_b64,
    trim_plate_crop_for_ocr,
)

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


def _extract_prob_matrix(outputs: list, cfg: dict) -> np.ndarray:
    """Chuyen ONNX outputs -> probability matrix shape (n_slots, len(alphabet))."""
    alphabet = cfg['alphabet']
    n_slots  = cfg['max_plate_slots']
    n_chars  = len(alphabet)

    if len(outputs) == n_slots:
        rows = []
        for slot in outputs:
            row = slot[0].flatten()
            if len(row) < n_chars:
                row = np.pad(row, (0, n_chars - len(row)))
            rows.append(row[:n_chars])
        return np.array(rows, dtype=np.float32)

    arr = outputs[0]
    if arr.ndim == 3:
        arr = arr[0]
    elif arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr[0].reshape(n_slots, n_chars)
    return arr.astype(np.float32)


def decode_from_probs(probs: np.ndarray, cfg: dict) -> Tuple[str, float, List[float]]:
    """
    Giai ma tu probability matrix -> (plate_text, avg_confidence, per_char_confs).
    """
    alphabet = cfg['alphabet']
    pad_char = cfg['pad_char']

    indices     = np.argmax(probs, axis=-1)
    conf_scores = np.max(probs, axis=-1)
    chars = [alphabet[i] for i in indices]
    confs = conf_scores.tolist()

    valid_confs = [c for ch, c in zip(chars, confs) if ch != pad_char]
    plate       = ''.join(ch for ch in chars if ch != pad_char)
    confidence  = sum(valid_confs) / len(valid_confs) if valid_confs else 0.0
    return plate, round(confidence, 4), confs


def decode(outputs: list, cfg: dict) -> Tuple[str, float]:
    """Giai ma outputs ONNX -> (plate_text, confidence). Backward-compatible."""
    probs = _extract_prob_matrix(outputs, cfg)
    text, conf, _ = decode_from_probs(probs, cfg)
    return text, conf


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
    # Single-crop OCR (backward-compatible, dung khi voting = off)
    # ------------------------------------------------------------------

    def _ocr_single_crop(self, crop_img: np.ndarray) -> Tuple[str, float]:
        gray   = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        tensor = prepare_input(gray, self._cfg)
        input_name = self._session.get_inputs()[0].name
        outputs    = self._session.run(None, {input_name: tensor})
        return decode(outputs, self._cfg)

    # ------------------------------------------------------------------
    # Multi-variant OCR with per-character probability voting
    # ------------------------------------------------------------------

    def _run_onnx_probs(self, gray: np.ndarray) -> np.ndarray:
        """Chay ONNX tren 1 anh gray, tra ve prob matrix (n_slots, n_chars)."""
        tensor     = prepare_input(gray, self._cfg)
        input_name = self._session.get_inputs()[0].name
        outputs    = self._session.run(None, {input_name: tensor})
        return _extract_prob_matrix(outputs, self._cfg)

    def _ocr_multi_variant(
        self, crop_img: np.ndarray, idx: int, label: str = '',
    ) -> Tuple[str, float, List[float]]:
        """
        Chay OCR voi nhieu bien the tien xu ly.
        Ket hop: weighted-sum voting + best-individual selection.

        Returns: (plate_text, avg_conf, per_char_confs)
        """
        max_v = settings.OCR_VOTING_VARIANTS
        gray  = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

        if max_v <= 1:
            probs = self._run_onnx_probs(gray)
            text, conf, char_confs = decode_from_probs(probs, self._cfg)
            logger.debug(
                f"[OCR] Crop #{idx}/{label or 'single'}: "
                f"raw='{text}' conf={conf:.4f} (voting=off)"
            )
            return text, conf, char_confs

        variants     = generate_variants(gray, max_variants=max_v)
        prob_sum     = None
        weight_total = 0.0
        individual_results: List[Tuple[str, float, str]] = []

        for v_name, v_gray in variants:
            probs = self._run_onnx_probs(v_gray)
            text_v, conf_v, _ = decode_from_probs(probs, self._cfg)

            w = conf_v ** 3
            if prob_sum is None:
                prob_sum = probs * w
            else:
                prob_sum += probs * w
            weight_total += w

            individual_results.append((text_v, conf_v, v_name))

            logger.debug(
                f"[OCR] Crop #{idx}/{label or 'vote'}/{v_name}: "
                f"raw='{text_v}' conf={conf_v:.4f}"
            )

        if prob_sum is None or weight_total == 0:
            return '', 0.0, []

        avg_probs = prob_sum / weight_total
        voted_text, voted_conf, voted_chars = decode_from_probs(avg_probs, self._cfg)

        logger.debug(
            f"[OCR] Crop #{idx}/{label or 'vote'} VOTED: "
            f"'{voted_text}' conf={voted_conf:.4f} ({len(variants)} variants)"
        )

        # --- Chon ket qua tot nhat: voted vs best individual ---
        best_indiv = self._pick_best_individual(
            individual_results, voted_text, voted_conf, idx, label,
        )
        if best_indiv is not None:
            return best_indiv[0], best_indiv[1], voted_chars

        return voted_text, voted_conf, voted_chars

    def _pick_best_individual(
        self,
        results: List[Tuple[str, float, str]],
        voted_text: str,
        voted_conf: float,
        idx: int,
        label: str,
    ) -> Optional[Tuple[str, float]]:
        """
        Tim variant tot nhat so voi voted result.
        Uu tien: valid plate + confidence cao + 9 chars > 8 chars.
        Tra ve (text, conf) hoac None neu voted tot hon.
        """
        voted_norm  = self._normalize_ocr_text(voted_text)
        voted_valid = is_valid_vn_plate(voted_norm)
        voted_len   = len(voted_norm)

        best_text: Optional[str] = None
        best_conf  = 0.0
        best_score = -999.0
        best_vname = ''

        for v_text, v_conf, v_name in results:
            v_norm  = self._normalize_ocr_text(v_text)
            v_valid = is_valid_vn_plate(v_norm)
            v_len   = len(v_norm)

            # Score: validity + confidence + length preference (9 > 8)
            score = v_conf
            if v_valid:
                score += 1.0
            if v_len == 9:
                score += 0.1
            elif v_len == 8:
                score += 0.0

            if score > best_score:
                best_score = score
                best_text  = v_text
                best_conf  = v_conf
                best_vname = v_name

        if best_text is None:
            return None

        best_norm  = self._normalize_ocr_text(best_text)
        best_valid = is_valid_vn_plate(best_norm)

        # Score voted the same way
        voted_score = voted_conf
        if voted_valid:
            voted_score += 1.0
        if voted_len == 9:
            voted_score += 0.1

        should_override = False
        if best_valid and not voted_valid:
            should_override = True
        elif best_score > voted_score:
            should_override = True

        if should_override:
            logger.debug(
                f"[OCR] Crop #{idx}/{label or 'vote'}: "
                f"'{best_vname}' overrides voted: "
                f"'{best_norm}' score={best_score:.4f} > "
                f"voted '{voted_norm}' score={voted_score:.4f}"
            )
            return best_text, best_conf

        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _normalize_ocr_text(self, text: str) -> str:
        extracted = extract_vn_plate_text(text)
        if extracted:
            if extracted != text.upper().replace('_', ''):
                logger.debug(f"[OCR] Patch OCR: '{text}' -> '{extracted}'")
            return extracted
        return text

    def _score_candidate(self, text: str, conf: float) -> float:
        normalized = self._normalize_ocr_text(text)
        if is_valid_vn_plate(normalized):
            return conf + 1.0
        if extract_vn_plate_text(text):
            return conf + 0.5
        if 5 < len(normalized) < 10:
            return conf
        return conf - 1.0

    def _needs_trim_retry(self, text: str, plate_img: np.ndarray) -> bool:
        if has_frame_brand_in_text(text):
            return True
        if is_valid_vn_plate(self._normalize_ocr_text(text)):
            return False
        return detect_frame_brand_band(plate_img) is not None

    # ------------------------------------------------------------------
    # Main OCR per-crop (with trim fallback)
    # ------------------------------------------------------------------

    def _ocr_plate_with_fallback(
        self,
        plate_img: np.ndarray,
        idx: int,
    ) -> Tuple[Optional[Tuple[float, str, float]], np.ndarray]:
        """
        OCR full crop truoc (multi-variant voting); chi cat tem khi can.
        Returns: (best_candidate | None, source_plate_img)
        """
        best: Optional[Tuple[float, str, float]] = None

        def _try(crop: np.ndarray, label: str):
            nonlocal best
            text, conf, _ = self._ocr_multi_variant(crop, idx, label)
            normalized = self._normalize_ocr_text(text)
            score      = self._score_candidate(text, conf)
            logger.debug(
                f"[OCR] Crop #{idx}/{label}: normalized='{normalized}' "
                f"conf={conf:.4f} score={score:.4f}"
            )
            if best is None or score > best[0]:
                best = (score, normalized, conf)

        _try(plate_img, 'full')

        if best and is_valid_vn_plate(best[1]):
            return best, plate_img

        if not self._needs_trim_retry(best[1] if best else '', plate_img):
            return best, plate_img

        trim_ratio = detect_frame_brand_band(plate_img)
        if trim_ratio is None:
            logger.debug(
                f"[OCR] Crop #{idx}: co chu tem trong OCR nhung "
                f"khong phat hien vung tem trong anh."
            )
            return best, plate_img

        trimmed = trim_plate_crop_for_ocr(plate_img, trim_ratio)
        logger.debug(
            f"[OCR] Crop #{idx}: phat hien tem khung, "
            f"cat top {trim_ratio*100:.0f}% roi OCR lai."
        )
        _try(trimmed, f'trim_{trim_ratio:.2f}')

        return best, plate_img

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

        best_candidate = None  # (score, plate_number, confidence, source_plate_img)

        for idx, plate_img in enumerate(plate_images):
            crop_h, crop_w = plate_img.shape[:2]
            logger.debug(f"[OCR] Crop #{idx}: kích thước {crop_w}x{crop_h} px")
            try:
                candidate, source_img = self._ocr_plate_with_fallback(plate_img, idx)
                if candidate and (
                    best_candidate is None or candidate[0] > best_candidate[0]
                ):
                    best_candidate = (*candidate, source_img)
            except Exception as e:
                logger.error(f"[OCR] Crop #{idx}: Loi xu ly - {e}", exc_info=True)
                print(f"Lỗi OCR cho 1 crop: {e}")
                continue

        if best_candidate is None:
            logger.warning(
                f"[OCR] [FAIL] Khong doc duoc bien so sau khi xu ly {len(plate_images)} crop(s)."
            )
            return '', '', 0.0

        _, plate_number, best_confidence, source_plate_img = best_candidate
        if not is_valid_vn_plate(plate_number) and not (5 < len(plate_number) < 10):
            logger.warning(
                f"[OCR] [FAIL] Ket qua khong hop le sau khi loc tem - '{plate_number}'."
            )
            return '', '', 0.0

        final_pil    = paste_img_2_img(vehicle_image, source_plate_img)
        base64_image = im_2_b64(final_pil)
        logger.info(f"[OCR] [OK] Ket qua cuoi: '{plate_number}' conf={best_confidence:.4f}")
        return plate_number, base64_image, best_confidence


# Singleton
ocr_service = OCRService()
