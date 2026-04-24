# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_all
from PyInstaller.utils.hooks import copy_metadata

datas = [('app', 'app'), ('xs-v2-global-model', 'xs-v2-global-model'), ('best_stripped.pt', '.'), ('plate.ico', '.')]
binaries = []
hiddenimports = ['pystray', 'winreg', 'win32timezone', 'win32serviceutil', 'win32service', 'win32event', 'servicemanager', 'hypercorn', 'hypercorn.config', 'hypercorn.asyncio', 'hypercorn.protocol', 'hypercorn.protocol.h11', 'hypercorn.protocol.h2', 'hypercorn.middleware', 'hypercorn.utils', 'h2', 'h2.config', 'h2.connection', 'h2.events', 'wsproto', 'asyncio', 'fastapi', 'fastapi.routing', 'starlette', 'starlette.routing', 'pydantic', 'pydantic_settings', 'onnxruntime', 'cv2', 'yaml', 'PIL', 'PIL.Image', 'logging.handlers', 'app.core.logging_config', 'torch', 'torchvision', 'multiprocessing', 'multiprocessing.util', 'ultralytics']
datas += collect_data_files('torchvision')
datas += copy_metadata('torch')
datas += copy_metadata('torchvision')
tmp_ret = collect_all('onnxruntime')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('torch')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('ultralytics')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['run.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['onnxruntime.quantization', 'matplotlib', 'notebook', 'IPython'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='dl_plate_server',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['plate.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='dl_plate_server',
)
