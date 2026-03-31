from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from database import create_db_and_tables
from api import jobs, files


@asynccontextmanager
async def lifespan(app: FastAPI):
    create_db_and_tables()
    yield


app = FastAPI(
    title="Resolution Converter",
    description="AI-powered image & video resolution conversion with content-aware resizing",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(jobs.router, prefix="/api", tags=["Jobs"])
app.include_router(files.router, prefix="/api", tags=["Files"])


@app.get("/api/health")
def health():
    return {"status": "ok"}
