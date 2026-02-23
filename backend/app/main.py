from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(title="ML Clinic API")

app.include_router(router)