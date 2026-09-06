"""FastAPI application entrypoint."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import __version__
from .api import datasets as datasets_api
from .core.config import get_settings
from .core.database import init_db
from .exceptions import AppError
from .services import dataset_service


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    dataset_service.ensure_dirs()
    init_db()
    yield


app_logger = logging.getLogger("ai-data-scientist")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description=(
            "Upload a CSV and let the AI Data Scientist Agent profile it, clean it, "
            "visualize it, train and compare ML models, and explain the results. "
            "All computations run in Python (pandas / scikit-learn / Plotly)."
        ),
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(datasets_api.router)

    @app.get("/", tags=["meta"])
    def root() -> dict:
        return {
            "name": settings.app_name,
            "version": __version__,
            "status": "running",
            "api_base": "/api",
            "interactive_docs": "/docs",
            "health": "/api/health",
            "endpoints": [
                "POST /api/datasets/upload",
                "GET  /api/datasets",
                "POST /api/datasets/{id}/analyze",
                "POST /api/datasets/{id}/train",
                "GET  /api/datasets/{id}/models",
                "GET  /api/datasets/{id}/visualizations",
                "POST /api/datasets/{id}/chat",
                "POST /api/datasets/{id}/report",
            ],
            "frontend": "Start the Next.js app (npm run dev in frontend/) and open http://localhost:3000",
        }

    @app.get("/api/health", tags=["meta"])
    def health() -> dict:
        return {
            "status": "ok",
            "version": __version__,
            "llm_enabled": settings.llm_enabled,
        }

    @app.exception_handler(AppError)
    async def app_error_handler(_: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content=exc.to_dict())

    @app.exception_handler(Exception)
    async def unhandled_error_handler(_: Request, exc: Exception) -> JSONResponse:
        # structured 500s so the frontend can show the real cause
        app_logger.exception("Unhandled error", exc_info=exc)
        return JSONResponse(
            status_code=500,
            content={"error": {"code": "internal_error", "message": f"Unexpected error: {exc}"}},
        )

    return app


app = create_app()
