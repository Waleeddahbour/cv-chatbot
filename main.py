from fastapi import FastAPI

from app.api.routers.chat import router as chat_router
from app.api.routers.ingestion import router as ingestion_router
from app.api.routers.system import router as system_router
from app.config.app_startup import lifespan


def create_app() -> FastAPI:
    app = FastAPI(
        title="CV Bot API",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.include_router(system_router)
    app.include_router(chat_router)
    app.include_router(ingestion_router)
    return app


app = create_app()


def main() -> FastAPI:
    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
