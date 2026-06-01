from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import models, diff, weights

app = FastAPI(
    title="VisualQuantization API",
    description="神经网络量化差异可视化平台后端 API",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(models.router, prefix="/api/models", tags=["models"])
app.include_router(diff.router, prefix="/api/diff", tags=["diff"])
app.include_router(weights.router, prefix="/api/weights", tags=["weights"])


@app.get("/")
def root():
    return {"message": "VisualQuantization API", "version": "0.1.0"}


@app.get("/health")
def health():
    return {"status": "healthy"}
