

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime

from .paths import BASE_DIR


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOG_FORMAT   = "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s"
DATE_FORMAT  = "%Y-%m-%d %H:%M:%S"
MAX_BYTES    = 10 * 1024 * 1024   # 10 MB mỗi file
BACKUP_COUNT = 5                  # giữ tối đa 5 file


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _get_app_root() -> Path:
 
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return BASE_DIR


def get_log_file() -> Path:
    """Trả về path file log của ngày hôm nay, tạo thư mục nếu chưa có."""
    log_dir = _get_app_root() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    return log_dir / f"dl_plate_{date_str}.log"


def setup_logging() -> dict:
    log_file = get_log_file()
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    is_frozen = getattr(sys, "frozen", False)

    # -- File handler (rotating) -------------------------------------------
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=MAX_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    # -- Console handler (chỉ khi chạy trong terminal, không dùng khi frozen exe) --
    console_handler = None
    if not is_frozen:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)  # INFO+ ra terminal, DEBUG chỉ vào file

    # -- Root logger -------------------------------------------------------
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for h in root.handlers[:]:
        root.removeHandler(h)
    root.addHandler(file_handler)
    if console_handler:
        root.addHandler(console_handler)

    # -- Giảm noise thư viện bên thứ 3 ------------------------------------
    for noisy in ("multipart", "PIL", "matplotlib", "urllib3", "ultralytics", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # -- Redirect stdout / stderr (chỉ khi exe, không ảnh hưởng dev terminal) --
    _redirect_std_streams(log_file)

    logging.info("=" * 60)
    logging.info("DL Plate Server – logging started")
    logging.info(f"Log file : {log_file}")
    logging.info("=" * 60)

    return _uvicorn_log_config(str(log_file), add_console=not is_frozen)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _redirect_std_streams(log_file: Path) -> None:
    try:
        log_stream = open(log_file, "a", encoding="utf-8", buffering=1)

        if sys.stdout is None or getattr(sys.stdout, 'fileno', lambda: -1)() < 0:
            sys.stdout = log_stream
        if sys.stderr is None or getattr(sys.stderr, 'fileno', lambda: -1)() < 0:
            sys.stderr = log_stream
    except Exception as exc:
        logging.warning(f"Could not redirect stdout/stderr: {exc}")


def _uvicorn_log_config(log_file_path: str, add_console: bool = True) -> dict:

    _file_handler_def = {
        "class": "logging.handlers.RotatingFileHandler",
        "filename": log_file_path,
        "maxBytes": MAX_BYTES,
        "backupCount": BACKUP_COUNT,
        "encoding": "utf-8",
        "formatter": "default",
    }
    _access_handler_def = {
        "class": "logging.handlers.RotatingFileHandler",
        "filename": log_file_path,
        "maxBytes": MAX_BYTES,
        "backupCount": BACKUP_COUNT,
        "encoding": "utf-8",
        "formatter": "access",
    }
    _console_handler_def = {
        "class": "logging.StreamHandler",
        "stream": "ext://sys.stdout",
        "formatter": "default",
    }
    _access_console_def = {
        "class": "logging.StreamHandler",
        "stream": "ext://sys.stdout",
        "formatter": "access",
    }

    handlers = {
        "file":        _file_handler_def,
        "access_file": _access_handler_def,
    }
    if add_console:
        handlers["console"]        = _console_handler_def
        handlers["access_console"] = _access_console_def

    uvicorn_handlers  = ["file", "console"] if add_console else ["file"]
    access_handlers   = ["access_file", "access_console"] if add_console else ["access_file"]

    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": "logging.Formatter",
                "fmt": LOG_FORMAT,
                "datefmt": DATE_FORMAT,
            },
            "access": {
                "()": "logging.Formatter",
                "fmt": "%(asctime)s [ACCESS  ] %(message)s",
                "datefmt": DATE_FORMAT,
            },
        },
        "handlers": handlers,
        "loggers": {
            "uvicorn":        {"handlers": uvicorn_handlers,  "level": "INFO", "propagate": False},
            "uvicorn.error":  {"handlers": uvicorn_handlers,  "level": "INFO", "propagate": False},
            "uvicorn.access": {"handlers": access_handlers,   "level": "INFO", "propagate": False},
            "fastapi":        {"handlers": uvicorn_handlers,  "level": "INFO", "propagate": False},
        },
    }
