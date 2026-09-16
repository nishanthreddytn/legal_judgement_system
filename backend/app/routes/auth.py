from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, EmailStr
from app.db import users
from app.security import hash_password, verify_password, create_token

router = APIRouter(prefix="/auth", tags=["Authentication"])

class Register(BaseModel):
    email: EmailStr
    password: str
    role: str = "citizen"

class Login(BaseModel):
    email: EmailStr
    password: str

@router.post("/register")
def register(body: Register):
    if body.role not in {"citizen", "judiciary"}:
        raise HTTPException(400, "Invalid role")
    if users.find_one({"email": body.email}):
        raise HTTPException(409, "Account already exists")
    users.insert_one({"email": body.email, "password": hash_password(body.password), "role": body.role})
    return {"message": "Registered successfully"}

@router.post("/login")
def login(body: Login):
    u = users.find_one({"email": body.email})
    if not u or not verify_password(body.password, u["password"]):
        raise HTTPException(401, "Invalid email or password")
    return {"access_token": create_token(u["email"], u["role"]), "token_type": "bearer", "role": u["role"]}
