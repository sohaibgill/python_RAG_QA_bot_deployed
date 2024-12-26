__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
from src.schemas.config import settings
from src.api import api_router


app = FastAPI(title=settings.PROJECT_NAME, description="APP for generating responses to user queries related to Python",openapi_url=f"{settings.API_V1_STR}/openapi.json",version="0.1")

# Set all CORS enabled origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router,prefix=settings.API_V1_STR)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=True)


