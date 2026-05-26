from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import (
    routes_auth,
    routes_dashboard,
    routes_monitoring,
    routes_reports,
    routes_students,
    routes_users,
)
from app.core.config import settings
from app.websocket.monitoring import router as monitoring_ws_router


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.API_VERSION,
        openapi_url=f"{settings.API_PREFIX}/openapi.json",
        docs_url=f"{settings.API_PREFIX}/docs",
        redoc_url=f"{settings.API_PREFIX}/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get(f"{settings.API_PREFIX}/health", tags=["health"])
    def health_check() -> dict[str, str]:
        return {"status": "ok", "service": settings.PROJECT_NAME}

    app.include_router(routes_auth.router, prefix=settings.API_PREFIX)
    app.include_router(routes_students.router, prefix=settings.API_PREFIX)
    app.include_router(routes_monitoring.router, prefix=settings.API_PREFIX)
    app.include_router(routes_reports.router, prefix=settings.API_PREFIX)
    app.include_router(routes_dashboard.router, prefix=settings.API_PREFIX)
    app.include_router(routes_users.router, prefix=settings.API_PREFIX)
    app.include_router(monitoring_ws_router, prefix=settings.API_PREFIX)
    return app


app = create_app()
