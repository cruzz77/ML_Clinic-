from pydantic import BaseModel
from datetime import datetime

class PredictionRequest(BaseModel):
    Gender: str
    ScheduledDay: datetime
    AppointmentDay: datetime
    Age: int
    Neighbourhood: str
    Scholoarship: int
    Hypertension: int
    Diabetes: int
    Alcoholism: int
    Handcap: int
    SMS_received: int