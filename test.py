"""
test.py - Batch ANPR: nhan dien bien so cho toan bo anh trong thu muc test_image,
luu anh ket qua (composite vehicle+plate crop) voi ten = bien so nhan dien duoc.

Usage:
    python test.py                          # doc tu test_image/, luu vao test_image/output/
    python test.py --input path/to/images   # tuy chinh thu muc dau vao
    python test.py --output path/to/out     # tuy chinh thu muc dau ra
"""

import os
import sys
import base64
import argparse
import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Them project root vao sys.path de import tu app/
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Setup logging (chi ghi vao file, khong spam console)
import logging
from app.core.logging_config import setup_logging
setup_logging()
logging.getLogger("app").setLevel(logging.WARNING)  # tat INFO log khi chay batch

# Import services
from app.services.detection_service import detection_service
from app.services.ocr_service import ocr_service

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def _save_image(img_or_path, output_dir: str, name: str) -> str:
    """Luu anh vao output_dir voi ten chi dinh. Tra ve duong dan da luu."""
    out_path = os.path.join(output_dir, f"{name}.jpg")
    counter  = 1
    while os.path.exists(out_path):
        out_path = os.path.join(output_dir, f"{name}_{counter}.jpg")
        counter += 1

    if isinstance(img_or_path, str):           # la duong dan file goc
        import shutil
        src_ext = os.path.splitext(img_or_path)[1].lower()
        if src_ext in (".jpg", ".jpeg"):
            shutil.copy2(img_or_path, out_path)
        else:                                  # chuyen sang JPEG
            img = cv2.imread(img_or_path)
            if img is not None:
                cv2.imwrite(out_path, img)
    else:                                      # la numpy array
        cv2.imwrite(out_path, img_or_path)
    return out_path


def process_image(img_path: str, output_dir: str) -> str | None:
    """
    Xu ly 1 anh: detect bien so -> OCR -> luu anh composite.
    - Nhan dien duoc : luu voi ten = bien so.
    - Khong nhan dien: luu anh goc voi ten no_vehicle.
    """
    img = cv2.imread(img_path)
    if img is None:
        print(f"  [SKIP]   Cannot read image: {img_path}")
        return None

    # 1. YOLO detection
    crops = detection_service.detect_plate(img)
    if not crops:
        saved = _save_image(img_path, output_dir, "no_vehicle")
        print(f"  [NO DET] {os.path.basename(img_path):40s}  ->  no_vehicle")
        return None

    # 2. OCR
    plate_number, base64_image, confidence = ocr_service.run(img, crops)
    if not plate_number:
        saved = _save_image(img_path, output_dir, "no_vehicle")
        print(f"  [NO OCR] {os.path.basename(img_path):40s}  ->  no_vehicle")
        return None

    print(f"  [OK]     {os.path.basename(img_path):40s}  ->  {plate_number}  (conf={confidence:.4f})")

    # 3. Giai ma base64 thanh anh va luu voi ten bien so
    if base64_image:
        try:
            img_bytes = base64.b64decode(base64_image)
            img_array = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            _save_image(img_array, output_dir, plate_number)
        except Exception as e:
            print(f"           [WARN] Luu anh that bai: {e}")

    return plate_number


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Batch ANPR - Rename & save plate images")
    parser.add_argument("--input",  default="test_image", help="Thu muc chua anh dau vao")
    parser.add_argument("--output", default=None,         help="Thu muc luu ket qua (mac dinh: <input>/output)")
    args = parser.parse_args()

    input_dir  = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output) if args.output else os.path.join(input_dir, "output")

    if not os.path.isdir(input_dir):
        print(f"[ERROR] Khong tim thay thu muc: {input_dir}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # Lay danh sach anh
    images = sorted([
        f for f in os.listdir(input_dir)
        if os.path.splitext(f.lower())[1] in SUPPORTED_EXTS
    ])

    if not images:
        print(f"[ERROR] Khong co anh nao trong {input_dir}")
        sys.exit(0)

    print()
    print("=" * 60)
    print(f"  Batch ANPR Test")
    print(f"  Input : {input_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Images: {len(images)}")
    print("=" * 60)
    print()

    ok_count   = 0
    fail_count = 0

    for fname in images:
        img_path = os.path.join(input_dir, fname)
        result   = process_image(img_path, output_dir)
        if result:
            ok_count   += 1
        else:
            fail_count += 1

    print()
    print("=" * 60)
    print(f"  Ket qua: {ok_count} thanh cong / {fail_count} that bai / {len(images)} tong")
    print(f"  Anh da luu tai: {output_dir}")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
