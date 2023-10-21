import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import api_router
from .config import get_app_settings


# Setup application
def get_application() -> FastAPI:

    settings = get_app_settings()
    app = FastAPI(**settings.fastapi_kwargs)
    
    origins = [
        "http://localhost:8000",
        "http://localhost:5173",
        "localhost:5173",
        "localhost",
        "http://localhost"
    ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"]
    )

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

# if __name__ == '__main__':
#     app = get_application()
#     uvicorn.run(app, host="localhost", port=3005)