import warnings
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

warnings.filterwarnings("ignore", category=FutureWarning)

from .core.config import settings
from .routers import detection, roi_tool

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title=settings.APP_NAME,
    description="API nhận diện biển số xe tự động.",
    version=settings.APP_VERSION,
)

# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(detection.router)
app.include_router(roi_tool.router)

# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "ok", "version": settings.APP_VERSION}


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=True)
