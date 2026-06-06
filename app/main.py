from fastapi import FastAPI
from app.api.routes import router
from app.core.logging import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="Personal Knowledge Intelligence System",
    description="Query your personal documents using natural language with source citations.",
    version="0.1.0"
)

app.include_router(router)

@app.on_event("startup")
async def startup():
    logger.info("Starting Personal Knowledge Intelligence System")
    logger.info("API ready")

@app.on_event("shutdown")
async def shutdown():
    logger.info("Shutting down")