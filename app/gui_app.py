import tkinter as tk
from tkinter import messagebox
import threading
import sys
import os
import winreg as reg
import pystray
from PIL import Image


# ---------------------------------------------------------------------------
# Đường dẫn signal file (dùng để IPC giữa các instance)
# ---------------------------------------------------------------------------

def _get_signal_file() -> str:
    return os.path.join(os.environ.get("TEMP", os.path.expanduser("~")), "dl_plate_show.signal")


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------

def set_autostart(enable=True):
    key_path = r"Software\Microsoft\Windows\CurrentVersion\Run"
    app_name = "DLPlateServer"
    try:
        key = reg.OpenKey(reg.HKEY_CURRENT_USER, key_path, 0, reg.KEY_ALL_ACCESS)
        if enable:
            if getattr(sys, 'frozen', False):
                exe_path = f'"{sys.executable}" --autostart'
            else:
                exe_path = f'"{sys.executable}" "{os.path.abspath(sys.argv[0])}" --autostart'
            reg.SetValueEx(key, app_name, 0, reg.REG_SZ, exe_path)
        else:
            try:
                reg.DeleteValue(key, app_name)
            except FileNotFoundError:
                pass
        reg.CloseKey(key)
    except Exception as e:
        print("Lỗi chỉnh Registry:", e)


def check_autostart():
    key_path = r"Software\Microsoft\Windows\CurrentVersion\Run"
    app_name = "DLPlateServer"
    try:
        key = reg.OpenKey(reg.HKEY_CURRENT_USER, key_path, 0, reg.KEY_READ)
        reg.QueryValueEx(key, app_name)
        reg.CloseKey(key)
        return True
    except FileNotFoundError:
        return False


# ---------------------------------------------------------------------------
# AppGUI
# ---------------------------------------------------------------------------

class AppGUI:
    def __init__(self, start_callback, auto_run=False):
        self.start_callback   = start_callback
        self.server_started   = False
        self.tray_icon        = None
        self.server_process   = None
        self.auto_run_initial = auto_run
        self._signal_file     = _get_signal_file()

        self.root = tk.Tk()
        self.root.title("DL Plate Server Manager")
        self.root.geometry("400x300")
        self.root.resizable(False, False)

        icon_path = self.get_icon_path()
        if os.path.exists(icon_path):
            self.root.iconbitmap(icon_path)

        # Behavior khi an nut X (dong cua so -> xuong tray)
        self.root.protocol('WM_DELETE_WINDOW', self.hide_window)

        # Xóa signal file cũ (nếu còn sót từ lần trước)
        self._clear_signal()

        # Bắt đầu vòng poll tín hiệu mỗi 300ms
        self._poll_signal()

        # UI Components
        tk.Label(self.root, text="DL Plate Recognition Server",
                 font=("Helvetica", 14, "bold")).pack(pady=15)

        frame = tk.Frame(self.root)
        frame.pack(pady=5)
        tk.Label(frame, text="Port:").pack(side=tk.LEFT, padx=5)
        self.port_entry = tk.Entry(frame, width=10)
        self.port_entry.insert(0, "8000")
        self.port_entry.pack(side=tk.LEFT)

        self.auto_start_var = tk.BooleanVar(value=check_autostart())
        tk.Checkbutton(self.root, text="Khởi động ngầm cùng Windows (Registry)",
                       variable=self.auto_start_var).pack(pady=5)

        self.btn_start = tk.Button(
            self.root, text="Khởi động Server", command=self.on_start,
            width=20, bg="#4CAF50", fg="white", font=("Helvetica", 10, "bold")
        )
        self.btn_start.pack(pady=10)

        if self.auto_run_initial:
            self.root.after(500, self.auto_start_sequence)

    # ------------------------------------------------------------------
    # Signal file IPC - phat hien instance thu 2 muon show cua so
    # ------------------------------------------------------------------

    def _clear_signal(self):
        try:
            if os.path.exists(self._signal_file):
                os.remove(self._signal_file)
        except Exception:
            pass

    def _poll_signal(self):
        """Kiem tra signal file moi 300ms. Neu ton tai -> show cua so."""
        if os.path.exists(self._signal_file):
            self._clear_signal()
            self._show_from_signal()
        self.root.after(300, self._poll_signal)

    def _show_from_signal(self):
        """Hiển lại cửa sổ chính khi nhận tín hiệu từ instance thứ 2."""
        if self.tray_icon:
            self.tray_icon.stop()
            self.tray_icon = None
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()

    # ------------------------------------------------------------------
    # Server control
    # ------------------------------------------------------------------

    def auto_start_sequence(self):
        self.on_start()
        self.hide_window()

    def on_start(self):
        if self.server_started:
            self.on_stop()
            return

        port_str = self.port_entry.get().strip()
        try:
            port = int(port_str)
        except ValueError:
            messagebox.showerror("Lỗi", "Port phải là số nguyên!")
            return

        set_autostart(self.auto_start_var.get())

        self.btn_start.config(text="Dừng Server", bg="#f44336")
        self.port_entry.config(state=tk.DISABLED)
        self.server_started = True

        import multiprocessing
        self.server_process = multiprocessing.Process(
            target=self.start_callback, args=(port,), daemon=True
        )
        self.server_process.start()

        if not self.auto_run_initial:
            messagebox.showinfo(
                "Thành công",
                f"Server đang hoạt động trên port {port}.\n\n"
                "HƯỚNG DẪN:\n"
                "- Nhấn nút [ X ] góc phải trên cùng để thu nhỏ ứng dụng xuống góc màn hình (Chạy ngầm).\n"
                "- Khi chạy ngầm, bạn có thể click chuột phải vào biểu tượng dưới thanh taskbar để tắt hoàn toàn."
            )
        self.auto_run_initial = False

    def on_stop(self):
        if self.server_process and self.server_process.is_alive():
            self.server_process.terminate()
            self.server_process.join()

        self.server_started = False
        self.btn_start.config(text="Khởi động Server", bg="#4CAF50")
        self.port_entry.config(state=tk.NORMAL)

    # ------------------------------------------------------------------
    # System Tray
    # ------------------------------------------------------------------

    def hide_window(self):
        """Thay vì đóng app, giấu cửa sổ và tạo Icon góc phải dưới."""
        self.root.withdraw()
        if not self.tray_icon:
            image = self.create_image()
            menu = pystray.Menu(
                pystray.MenuItem('Mở lại', self.show_window, default=True),
                pystray.MenuItem('Thoát hoàn toàn', self.quit_window)
            )
            self.tray_icon = pystray.Icon("name", image, "DL Plate Server", menu)
            threading.Thread(target=self.tray_icon.run, daemon=True).start()

    def show_window(self, icon, item):
        self.tray_icon.stop()
        self.tray_icon = None
        self.root.deiconify()

    def quit_window(self, icon, item):
        self._clear_signal()
        if self.tray_icon:
            self.tray_icon.stop()
        if getattr(self, 'server_started', False):
            self.on_stop()
        self.root.destroy()
        os._exit(0)  # Ép thoát hoàn toàn

    # ------------------------------------------------------------------
    # Icon helpers
    # ------------------------------------------------------------------

    def get_icon_path(self):
        if getattr(sys, 'frozen', False):
            base_dir = getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
            icon_path = os.path.join(base_dir, "plate.ico")
            if not os.path.exists(icon_path):
                icon_path = os.path.join(os.path.dirname(sys.executable), "plate.ico")
        else:
            icon_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plate.ico"
            )
        return icon_path

    def create_image(self):
        icon_path = self.get_icon_path()
        if os.path.exists(icon_path):
            return Image.open(icon_path)
        from PIL import ImageDraw
        img = Image.new('RGB', (64, 64), color=(50, 50, 50))
        d = ImageDraw.Draw(img)
        d.rectangle([10, 20, 54, 44], fill=(255, 255, 255))
        return img

    def run(self):
        self.root.mainloop()
