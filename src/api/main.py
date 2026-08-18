"""FastAPI application factory for the RAG agent backend."""

from collections.abc import Callable
from contextlib import AbstractContextManager, asynccontextmanager
import os
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.observability.logging import log
from src.api.routes import router
from src.api.service import (
    SessionBusyError,
    SessionNotFoundError,
    SessionService,
)


def create_app(
    *,
    application_factory: Callable[[], AbstractContextManager[Any]] | None = None,
) -> FastAPI:
    """Build the API without opening external resources during import."""
    if application_factory is None:
        from src.workflows.application import open_memory_application

        application_factory = open_memory_application

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        with application_factory() as runner:
            app.state.runner = runner
            app.state.session_service = SessionService(runner)
            yield

    app = FastAPI(
        title="Rag_Agent API",
        version="1.0.0",
        lifespan=lifespan,
    )
    configured_origins = os.getenv(
        "API_CORS_ORIGINS",
        "http://localhost:3000,http://localhost:5173",
    )
    allowed_origins = [
        origin.strip()
        for origin in configured_origins.split(",")
        if origin.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "X-User-ID", "X-Request-ID"],
    )

    @app.get("/health/live", tags=["health"])
    def health_live() -> dict[str, str]:
        return {"status": "ok"}

    @app.exception_handler(SessionNotFoundError)
    async def session_not_found_handler(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content={
                "detail": {
                    "code": "session_not_found",
                    "message": "session not found",
                }
            },
        )

    @app.exception_handler(SessionBusyError)
    async def session_busy_handler(
        request: Request,
        exc: SessionBusyError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=409,
            content={
                "detail": {
                    "code": "session_busy",
                    "message": "session is already processing a message",
                }
            },
        )

    async def dependency_unavailable_handler(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        log.error("[API] dependency unavailable | error={}", type(exc).__name__)
        return JSONResponse(
            status_code=503,
            content={
                "detail": {
                    "code": "service_unavailable",
                    "message": "a required service is unavailable",
                }
            },
        )

    app.add_exception_handler(ConnectionError, dependency_unavailable_handler)
    app.add_exception_handler(TimeoutError, dependency_unavailable_handler)

    @app.exception_handler(Exception)
    async def unexpected_error_handler(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        log.error("[API] unhandled request failure | error={}", type(exc).__name__)
        return JSONResponse(
            status_code=500,
            content={
                "detail": {
                    "code": "internal_error",
                    "message": "internal server error",
                }
            },
        )

    app.include_router(router)

    return app


app = create_app()
