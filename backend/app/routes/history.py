from fastapi import APIRouter, Depends
from app.db import history
from app.security import current_user

router = APIRouter(prefix="/history", tags=["History"])

@router.get("")
def get_history(user=Depends(current_user)):
    docs = list(history.find({"user_email": user["email"]}, {"_id": 0}).sort("_id", -1).limit(30))
    return docs
