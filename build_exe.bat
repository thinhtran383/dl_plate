@echo off
chcp 65001 > nul
setlocal EnableDelayedExpansion

set "APP_NAME=dl_plate_server"
set "PYTHON=venv\Scripts\python.exe"
set "PIP=venv\Scripts\pip.exe"
set "PYINST=venv\Scripts\pyinstaller.exe"

echo.
echo ==================================================
echo   DL Plate - Build Script (onedir)
echo ==================================================

if not exist "%PYTHON%" (
    echo [ERROR] Khong tim thay venv. Chay: py -3.12 -m venv venv
    pause & exit /b 1
)

"%PYTHON%" --version

echo.
echo [1/3] Cai PyInstaller...
"%PIP%" install pyinstaller --quiet
if errorlevel 1 ( echo [ERROR] Cai PyInstaller that bai! & pause & exit /b 1 )
echo [OK] PyInstaller san sang.

echo.
echo [2/3] Don dep output cu...
if exist "dist\%APP_NAME%" rmdir /s /q "dist\%APP_NAME%"
if exist "build"           rmdir /s /q "build"

echo.
echo [3/3] Dang build...

set CMD="%PYINST%"
set CMD=%CMD% --onedir
set CMD=%CMD% --noconfirm
set CMD=%CMD% --clean
set CMD=%CMD% --noconsole
set CMD=%CMD% --name "%APP_NAME%"
set CMD=%CMD% --add-data "app;app"
set CMD=%CMD% --add-data "xs-v2-global-model;xs-v2-global-model"
set CMD=%CMD% --add-data "plate_detection_v8.pt;."
set CMD=%CMD% --add-data "plate.ico;."
set CMD=%CMD% --icon=plate.ico
set CMD=%CMD% --hidden-import "pystray"
set CMD=%CMD% --hidden-import "winreg"
set CMD=%CMD% --hidden-import "win32timezone"
set CMD=%CMD% --hidden-import "win32serviceutil"
set CMD=%CMD% --hidden-import "win32service"
set CMD=%CMD% --hidden-import "win32event"
set CMD=%CMD% --hidden-import "servicemanager"
set CMD=%CMD% --hidden-import "hypercorn"
set CMD=%CMD% --hidden-import "hypercorn.config"
set CMD=%CMD% --hidden-import "hypercorn.asyncio"
set CMD=%CMD% --hidden-import "hypercorn.protocol"
set CMD=%CMD% --hidden-import "hypercorn.protocol.h11"
set CMD=%CMD% --hidden-import "hypercorn.protocol.h2"
set CMD=%CMD% --hidden-import "hypercorn.middleware"
set CMD=%CMD% --hidden-import "hypercorn.utils"
set CMD=%CMD% --hidden-import "h2"
set CMD=%CMD% --hidden-import "h2.config"
set CMD=%CMD% --hidden-import "h2.connection"
set CMD=%CMD% --hidden-import "h2.events"
set CMD=%CMD% --hidden-import "wsproto"
set CMD=%CMD% --hidden-import "asyncio"
set CMD=%CMD% --hidden-import "fastapi"
set CMD=%CMD% --hidden-import "fastapi.routing"
set CMD=%CMD% --hidden-import "starlette"
set CMD=%CMD% --hidden-import "starlette.routing"
set CMD=%CMD% --hidden-import "pydantic"
set CMD=%CMD% --hidden-import "pydantic_settings"
set CMD=%CMD% --hidden-import "onnxruntime"
set CMD=%CMD% --hidden-import "cv2"
set CMD=%CMD% --hidden-import "yaml"
set CMD=%CMD% --hidden-import "PIL"
set CMD=%CMD% --hidden-import "PIL.Image"
set CMD=%CMD% --hidden-import "logging.handlers"
set CMD=%CMD% --hidden-import "app.core.logging_config"
set CMD=%CMD% --hidden-import "torch"
set CMD=%CMD% --hidden-import "torchvision"
set CMD=%CMD% --hidden-import "multiprocessing"
set CMD=%CMD% --hidden-import "multiprocessing.util"
set CMD=%CMD% --hidden-import "ultralytics"
set CMD=%CMD% --collect-all "onnxruntime"
set CMD=%CMD% --collect-all "torch"
set CMD=%CMD% --collect-all "ultralytics"
set CMD=%CMD% --collect-data "torchvision"
set CMD=%CMD% --copy-metadata "torch"
set CMD=%CMD% --copy-metadata "torchvision"
set CMD=%CMD% --exclude-module "onnxruntime.quantization"
set CMD=%CMD% --exclude-module "matplotlib"
set CMD=%CMD% --exclude-module "notebook"
set CMD=%CMD% --exclude-module "IPython"
set CMD=%CMD% run.py

%CMD%

if errorlevel 1 (
    echo.
    echo [ERROR] Build that bai!
    pause & exit /b 1
)

echo.
echo ==================================================
echo   BUILD THANH CONG!
echo ==================================================
echo   Chay: dist\%APP_NAME%\%APP_NAME%.exe
echo   Docs: http://localhost:8080/docs
echo ==================================================
echo.
pause
endlocal
