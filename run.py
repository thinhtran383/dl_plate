"""
Entrypoint để start server.
Chạy: python run.py  hoặc  dl_plate_server.exe
"""

import sys
import os
import argparse
import logging
import traceback

# ---------------------------------------------------------------------------
# Single-instance guard (Windows Named Mutex + signal file)
# Phải chạy TRƯỚC KHI import bất kỳ thứ gì nặng (torch, uvicorn, ...)
# ---------------------------------------------------------------------------
def _get_signal_file() -> str:
    return os.path.join(os.environ.get("TEMP", os.path.expanduser("~")), "dl_plate_show.signal")


def _ensure_single_instance():
    """Nếu đã có 1 instance đang chạy → tạo signal file rồi thoát."""
    import ctypes

    MUTEX_NAME = "Global\\DLPlateServer_SingleInstance"
    mutex = ctypes.windll.kernel32.CreateMutexW(None, True, MUTEX_NAME)
    last_err = ctypes.windll.kernel32.GetLastError()

    if last_err == 183:  # ERROR_ALREADY_EXISTS — đã có instance chạy
        # Tạo signal file để instance cũ tự show lại cửa sổ
        try:
            signal_file = _get_signal_file()
            with open(signal_file, "w") as f:
                f.write("show")
        except Exception:
            pass
        sys.exit(0)  # Thoát instance mới ngay lập tức

    # Giữ mutex sống suốt vòng đời process (không release)
    return mutex




# Đưa root project vào sys.path
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ---------------------------------------------------------------------------
# Khởi tạo logging TRƯỚC KHI import bất kỳ thứ gì khác
# (để bắt print() từ YOLOv5, torch, v.v.)
# ---------------------------------------------------------------------------
from app.core.logging_config import setup_logging, get_log_file

log_config = setup_logging()
logger = logging.getLogger(__name__)

import asyncio
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="DL Plate Recognition API")
    parser.add_argument("--host",   default="0.0.0.0", help="Host")
    parser.add_argument("--port",   default=8000, type=int, help="Port")
    parser.add_argument("--reload", action="store_true", help="Auto-reload (chỉ dev)")
    parser.add_argument("--autostart", action="store_true", help="Auto start server nếu chạy từ registry")
    args, _ = parser.parse_known_args()
    return args
    
def launch_server(port):
    log_file = get_log_file()
    mode = "frozen exe" if getattr(sys, "frozen", False) else "python"
    banner = "\n".join([
        "=" * 50,
        "  DL Plate Recognition Server (HTTP/1.1 + HTTP/2)",
        "=" * 50,
        f"  URL    : http://0.0.0.0:{port}",
        f"  Swagger: http://localhost:{port}/docs",
        f"  Mode   : {mode}",
        f"  Log    : {log_file}",
        "=" * 50,
    ])
    logger.info(banner)

    from hypercorn.config import Config
    from hypercorn.asyncio import serve

    cfg = Config()
    cfg.bind = [f"0.0.0.0:{port}"]
    cfg.h2_max_concurrent_streams = 100
    # Ghi log hypercorn vao file theo logging_config hien tai
    cfg.accesslog  = "-"   # stdout (bi bat boi logging redirect)
    cfg.errorlog   = "-"
    cfg.loglevel   = "INFO"

    if getattr(sys, "frozen", False):
        from app.main import app
    else:
        from app.main import app

    asyncio.run(serve(app, cfg))





if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    is_frozen = getattr(sys, "frozen", False)

    # Single-instance guard — chỉ chạy ở main process, KHÔNG chạy ở child process
    _SINGLE_INSTANCE_MUTEX = _ensure_single_instance()

    args = parse_args()
    
    if '--no-gui' in sys.argv:
        launch_server(args.port)
        sys.exit(0)
        
    try:
        from app.gui_app import AppGUI
        gui = AppGUI(
            start_callback=launch_server,
            auto_run=args.autostart
        )
        gui.run()
    except Exception:
        logger.exception("GUI Boot Error")
        traceback.print_exc()
        if not is_frozen:
            input("\nNhấn Enter để thoát...")

