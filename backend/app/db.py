from pymongo import MongoClient
from .config import MONGODB_URL, MONGODB_DB

client = MongoClient(MONGODB_URL, serverSelectionTimeoutMS=3000)
db = client[MONGODB_DB]
users = db["users"]
history = db["history"]
