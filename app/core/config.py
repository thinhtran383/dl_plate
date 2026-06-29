import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List

from .paths import BASE_DIR


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=os.path.join(BASE_DIR, ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # App
    APP_NAME: str = "DL Plate Recognition API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8080

    # CORS
    ALLOWED_ORIGINS: List[str] = ["*"]

    # Model paths (relative - se duoc resolve thanh absolute bang BASE_DIR)
    DETECTION_MODEL_PATH: str = "best_stripped.pt"
    OCR_MODEL_DIR: str = "xs-v2-global-model"

    # Detection
    DETECTION_CONF: float = 0.4
    DETECTION_IOU: float = 0.45
    DETECTION_BBOX_PADDING: float = 0.0
    DETECTION_INPUT_WIDTH: int = 960     # 0 = dung anh goc, >0 = resize truoc khi YOLO detect
    ROI_CONFIG_PATH: str = "roi_config.json"

    # Output
    PLATE_OVERLAY_DIVISOR: int = 4         # 4 = crop chiem 1/4 chieu rong anh xe
    OUTPUT_MAX_WIDTH: int = 0              # 0 = off, >0 = scale down anh output ve chieu rong nay
    OUTPUT_JPEG_QUALITY: int = 0           # 0 = off (dung PIL default=75), 1-95 = tuy chinh

    # Logging (rotation kieu Spring Boot / log4j)
    LOG_MAX_BYTES: int = 10 * 1024 * 1024  # 10 MB moi file
    LOG_RETENTION_DAYS: int = 30           # tu xoa log cu hon N ngay

    @property
    def abs_detection_model(self) -> str:
        return os.path.join(BASE_DIR, self.DETECTION_MODEL_PATH)

    @property
    def abs_ocr_model_dir(self) -> str:
        return os.path.join(BASE_DIR, self.OCR_MODEL_DIR)


settings = Settings()
