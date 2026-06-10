"""
Tien xu ly anh bien so: tao nhieu bien the de voting OCR.

Moi ham nhan grayscale uint8, tra ve grayscale uint8.
"""
from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np


def _apply_gamma(gray: np.ndarray, gamma: float) -> np.ndarray:
    inv = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** inv) * 255 for i in range(256)], dtype=np.uint8,
    )
    return cv2.LUT(gray, table)


def original(gray: np.ndarray) -> np.ndarray:
    return gray


def clahe(gray: np.ndarray) -> np.ndarray:
    cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    return cl.apply(gray)


def denoise(gray: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoising(gray, None, h=12, templateWindowSize=7, searchWindowSize=21)


def sharpen(gray: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=2)
    return cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)


def gamma_bright(gray: np.ndarray) -> np.ndarray:
    return _apply_gamma(gray, 1.8)


def gamma_dark(gray: np.ndarray) -> np.ndarray:
    return _apply_gamma(gray, 0.55)


def bilateral(gray: np.ndarray) -> np.ndarray:
    return cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)


def threshold_adaptive(gray: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 4,
    )


_ALL_VARIANTS: List[Tuple[str, callable]] = [
    ('original',           original),
    ('clahe',              clahe),
    ('denoise',            denoise),
    ('sharpen',            sharpen),
    ('gamma_bright',       gamma_bright),
    ('gamma_dark',         gamma_dark),
    ('bilateral',          bilateral),
    ('threshold_adaptive', threshold_adaptive),
]


def generate_variants(
    gray: np.ndarray,
    max_variants: int = 0,
) -> List[Tuple[str, np.ndarray]]:
    """
    Tra ve list (name, processed_gray).

    Args:
        gray:          anh grayscale uint8
        max_variants:  0 = tat ca, >0 = gioi han so luong (luon gom original)
    """
    if max_variants <= 0:
        max_variants = len(_ALL_VARIANTS)

    results: List[Tuple[str, np.ndarray]] = []
    for name, fn in _ALL_VARIANTS:
        if len(results) >= max_variants:
            break
        try:
            out = fn(gray)
            if out is not None and out.size > 0:
                results.append((name, out))
        except Exception:
            continue

    if not results:
        results.append(('original', gray))
    return results
