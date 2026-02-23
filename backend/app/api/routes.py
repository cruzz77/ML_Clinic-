from fastapi import APIRouter
from app.schemas.prediction_schema import PredictionRequest
from app.services.prediction_service import predict

router = APIRouter()

@router.post("/predict")
def make_prediction(request: PredictionRequest):
    return predict(request.dict())