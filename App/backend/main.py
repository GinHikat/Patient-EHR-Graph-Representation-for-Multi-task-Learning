from fastapi import FastAPI
import os
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router as api_router
from core.database import close_db_driver
from contextlib import asynccontextmanager

from services.ensure_indices import ensure_indices

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic: ensure Neo4j indices are created
    try:
        ensure_indices()
    except Exception as e:
        print(f"Warning: Failed to ensure Neo4j indices on startup: {e}")
    yield
    # Shutdown logic
    close_db_driver()

app = FastAPI(title="Graph API", lifespan=lifespan)

origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173,*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False if "*" in origins else True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
