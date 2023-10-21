import uvicorn
from fastapi import FastAPI

from .api import api_router
from .config import get_app_settings


# Setup application
def get_application() -> FastAPI:

    settings = get_app_settings()
    app = FastAPI(**settings.fastapi_kwargs)

    app.add_event_handler(
        "startup",
        on_startup,
    )

    app.add_event_handler(
        "shutdown",
        on_shutdown,
    )

    app.include_router(api_router)

    return app


# Process startup events
def on_startup():
    pass


# Process shutdown events
def on_shutdown():
    pass


app = get_application()
