
import logging
import re
import sys
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from .paths import BASE_DIR


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOG_FORMAT  = "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_PREFIX  = "dl_plate"
_LOG_NAME_RE = re.compile(rf"^{LOG_PREFIX}_(\d{{4}}-\d{{2}}-\d{{2}})(?:\.(\d+))?\.log$")

_file_handler: Optional["DailySizeRotatingFileHandler"] = None


# ---------------------------------------------------------------------------
# Daily + size rotating handler (Spring Boot / log4j style)
# ---------------------------------------------------------------------------

class DailySizeRotatingFileHandler(logging.Handler):
    """
    Ghi log theo pattern: dl_plate_{YYYY-MM-DD}.{N}.log
    - Sang ngay moi -> reset index ve 0
    - Trong cung ngay vuot max_bytes -> tang index
    """

    def __init__(
        self,
        log_dir: Path,
        max_bytes: int,
        retention_days: int,
        base_name: str = LOG_PREFIX,
    ):
        super().__init__()
        self.log_dir = log_dir
        self.max_bytes = max_bytes
        self.retention_days = retention_days
        self.base_name = base_name
        self._lock = threading.RLock()
        self._stream = None
        self._current_date: Optional[str] = None
        self._current_index = 0
        self._current_path: Optional[Path] = None
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, date_str: str, index: int) -> Path:
        return self.log_dir / f"{self.base_name}_{date_str}.{index}.log"

    def _resolve_open_target(self, date_str: str) -> tuple[int, Path]:
        """Tim file dang ghi cua hom nay (append neu con du cho)."""
        pattern = f"{self.base_name}_{date_str}.*.log"
        candidates: list[tuple[int, Path]] = []

        for path in self.log_dir.glob(pattern):
            match = _LOG_NAME_RE.match(path.name)
            if not match:
                continue
            idx = int(match.group(2) or 0)
            candidates.append((idx, path))

        if not candidates:
            return 0, self._path_for(date_str, 0)

        candidates.sort(key=lambda item: item[0])
        latest_index, latest_path = candidates[-1]
        if latest_path.stat().st_size < self.max_bytes:
            return latest_index, latest_path
        return latest_index + 1, self._path_for(date_str, latest_index + 1)

    def _open_stream(self, date_str: Optional[str] = None) -> None:
        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")

        index, path = self._resolve_open_target(date_str)
        self._current_date = date_str
        self._current_index = index
        self._current_path = path
        self._stream = open(path, "a", encoding="utf-8", buffering=1)

    def _close_stream(self) -> None:
        if self._stream is not None:
            try:
                self._stream.flush()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def _rollover_if_needed(self, nbytes: int) -> None:
        today = datetime.now().strftime("%Y-%m-%d")

        if self._current_date != today:
            self._close_stream()
            if self._current_date is not None:
                self._cleanup_old_logs()
            self._open_stream(today)
            return

        if self._stream is None:
            self._open_stream(today)
            return

        if self._stream.tell() + nbytes > self.max_bytes:
            self._close_stream()
            self._current_index += 1
            self._current_path = self._path_for(today, self._current_index)
            self._stream = open(self._current_path, "a", encoding="utf-8", buffering=1)

    def _cleanup_old_logs(self) -> None:
        if self.retention_days <= 0:
            return

        cutoff = datetime.now().date() - timedelta(days=self.retention_days)
        for path in self.log_dir.glob(f"{self.base_name}_*.log"):
            match = _LOG_NAME_RE.match(path.name)
            if not match:
                continue
            try:
                file_date = datetime.strptime(match.group(1), "%Y-%m-%d").date()
            except ValueError:
                continue
            if file_date < cutoff:
                try:
                    path.unlink()
                except OSError:
                    pass

    def write_raw(self, text: str) -> None:
        if not text:
            return
        data = text if isinstance(text, str) else str(text)
        encoded = data.encode("utf-8", errors="replace")
        with self._lock:
            self._rollover_if_needed(len(encoded))
            self._stream.write(data)
            self._stream.flush()

    def flush(self) -> None:
        with self._lock:
            if self._stream is not None:
                self._stream.flush()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self.write_raw(msg + "\n")
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        with self._lock:
            self._close_stream()
        super().close()

    @property
    def current_path(self) -> Optional[Path]:
        return self._current_path


class HandlerStream:
    """Stream wrapper de redirect stdout/stderr qua rotating handler."""

    def __init__(self, handler: DailySizeRotatingFileHandler):
        self._handler = handler

    def write(self, text: str) -> int:
        if text:
            self._handler.write_raw(text)
        return len(text) if text else 0

    def flush(self) -> None:
        self._handler.flush()

    def fileno(self) -> int:
        raise OSError("HandlerStream does not expose a file descriptor")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _get_app_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return BASE_DIR


def get_file_handler(**_kwargs) -> DailySizeRotatingFileHandler:
    global _file_handler
    if _file_handler is None:
        from .config import settings

        log_dir = _get_app_root() / "logs"
        _file_handler = DailySizeRotatingFileHandler(
            log_dir=log_dir,
            max_bytes=settings.LOG_MAX_BYTES,
            retention_days=settings.LOG_RETENTION_DAYS,
        )
    return _file_handler


def get_log_file() -> Path:
    """Tra ve path file log dang active (tao thu muc neu chua co)."""
    handler = get_file_handler()
    if handler.current_path is None:
        with handler._lock:
            if handler._stream is None:
                handler._open_stream()
    return handler.current_path or (
        _get_app_root() / "logs" / f"{LOG_PREFIX}_{datetime.now().strftime('%Y-%m-%d')}.0.log"
    )


def setup_logging() -> dict:
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    is_frozen = getattr(sys, "frozen", False)

    file_handler = get_file_handler()
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    console_handler = None
    if not is_frozen:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for h in root.handlers[:]:
        root.removeHandler(h)
    root.addHandler(file_handler)
    if console_handler:
        root.addHandler(console_handler)

    for noisy in ("multipart", "PIL", "matplotlib", "urllib3", "ultralytics", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _redirect_std_streams(file_handler)

    log_file = get_log_file()
    logging.info("=" * 60)
    logging.info("DL Plate Server – logging started")
    logging.info(f"Log file : {log_file}")
    logging.info("=" * 60)

    return _uvicorn_log_config(add_console=not is_frozen)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _redirect_std_streams(file_handler: DailySizeRotatingFileHandler) -> None:
    try:
        redirect_stream = HandlerStream(file_handler)

        if sys.stdout is None or getattr(sys.stdout, "fileno", lambda: -1)() < 0:
            sys.stdout = redirect_stream
        if sys.stderr is None or getattr(sys.stderr, "fileno", lambda: -1)() < 0:
            sys.stderr = redirect_stream
    except Exception as exc:
        logging.warning(f"Could not redirect stdout/stderr: {exc}")


def _uvicorn_log_config(add_console: bool = True) -> dict:
    _file_handler_def = {
        "()": "app.core.logging_config.get_file_handler",
        "formatter": "default",
    }
    _access_handler_def = {
        "()": "app.core.logging_config.get_file_handler",
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

    uvicorn_handlers = ["file", "console"] if add_console else ["file"]
    access_handlers  = ["access_file", "access_console"] if add_console else ["access_file"]

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
