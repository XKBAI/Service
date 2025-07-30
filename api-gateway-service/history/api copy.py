import os
import json
import uvicorn
import logging
import time
import asyncio
import urllib.parse
from datetime import timedelta
from typing import List, Dict, Any, Optional
import io
import tempfile
import mimetypes
import requests
from requests.auth import HTTPBasicAuth

from fastapi import FastAPI, HTTPException, status, Request, Response, BackgroundTasks, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel

from mylog import logger
# from LLM import LLM, SLM
# Assuming query.py exists and contains chat_rag_stream and deep_research_stream
# from query import chat_rag_stream, deep_research_stream 
from authentication.auth import (
    User, Token, get_current_api_user,
    verify_password, create_access_token,
    ACCESS_TOKEN_EXPIRE_MINUTES,
    FIXED_USERNAME, FIXED_PASSWORD_HASH,
    get_fixed_api_user
)

logger.setLevel(logging.INFO)

# --- Configuration ---
WEBDAV_BASE_URL: str = os.getenv("WEBDAV_BASE_URL", "http://192.168.2.4:58090").rstrip('/')
WEBDAV_USERNAME: str = os.getenv("WEBDAV_USERNAME", "xkb")
WEBDAV_PASSWORD: str = os.getenv("WEBDAV_PASSWORD", "Xkb@1234")

WEBDAV_IMAGE_ROOT_IN_PATH: str = os.getenv("WEBDAV_IMAGE_ROOT_IN_PATH", "/")
if not WEBDAV_IMAGE_ROOT_IN_PATH.startswith('/'):
    WEBDAV_IMAGE_ROOT_IN_PATH = '/' + WEBDAV_IMAGE_ROOT_IN_PATH

FASTAPI_BASE_URL: str = os.getenv("FASTAPI_BASE_URL", "https://localhost:60443").rstrip('/')

app = FastAPI()

# --- IP Rate Limiting and Concurrency Control (remains unchanged) ---
ip_data: Dict[str, Dict[str, Any]] = {}
ip_data_lock = asyncio.Lock()

QPS_LIMIT = 60
QPS_WINDOW_MS = 60 * 1000
CONCURRENT_REQUEST_LIMIT = 10
MAX_ERROR_COUNT = 30
BLOCK_DURATION_MS = 5 * 60 * 1000

class messages(BaseModel):
    messages: Optional[list[dict[str,str]]]=[]

class chat_ai_msg(BaseModel):
    prompt: str
    messages: Optional[list[dict[str,str]]]=[]
    chat_type:str

def get_client_ip(request: Request) -> str:
    x_forwarded_for = request.headers.get("x-forwarded-for")
    if x_forwarded_for:
        return x_forwarded_for.split(',')[0].strip()
    return request.client.host if request.client else "unknown_ip"

async def rate_limit_and_concurrency(request: Request):
    ip = get_client_ip(request)
    current_time = time.time() * 1000

    async with ip_data_lock:
        data = ip_data.get(ip)
        if not data:
            data = {
                'request_count': 0, 'last_request_time': current_time,
                'active_requests': 0, 'error_count': 0, 'blocked_until': 0
            }
            ip_data[ip] = data

        if data['blocked_until'] > current_time:
            remaining_seconds = int( (data['blocked_until'] - current_time) / 1000 )
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=f"IP ({ip}) is temporarily blocked. Please try again in {remaining_seconds} seconds.")

        if current_time - data['last_request_time'] > QPS_WINDOW_MS:
            data['request_count'] = 0
            data['last_request_time'] = current_time

        if data['request_count'] >= QPS_LIMIT:
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=f"IP ({ip}) is making requests too frequently. Please try again later.")

        if data['active_requests'] >= CONCURRENT_REQUEST_LIMIT:
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=f"IP ({ip}) has too many concurrent requests. Please wait for current requests to complete.")

        data['request_count'] += 1
        data['active_requests'] += 1
        logger.debug(f"IP: {ip}, Request Count: {data['request_count']}, Active Requests: {data['active_requests']}")

    def decrement_active_requests_callback_inner():
        async def _decrement():
            async with ip_data_lock:
                if ip in ip_data:
                    ip_data[ip]['active_requests'] = max(0, ip_data[ip]['active_requests'] - 1)
                    logger.debug(f"IP: {ip}, Active requests decreased to: {ip_data[ip]['active_requests']}")
        return _decrement

    request.state.decrement_callback = decrement_active_requests_callback_inner()

@app.middleware("http")
async def error_blocking_middleware(request: Request, call_next):
    ip = get_client_ip(request)
    current_time = time.time() * 1000
    decrement_needed = hasattr(request.state, 'decrement_callback')

    try:
        response = await call_next(request)
    except HTTPException as exc:
        async with ip_data_lock:
            data = ip_data.get(ip)
            if data and exc.status_code >= 400 and exc.status_code < 600:
                data['error_count'] += 1
                logger.info(f"IP: {ip}, Error count: {data['error_count']} (HTTPException: {exc.status_code})")
                if data['error_count'] >= MAX_ERROR_COUNT:
                    data['blocked_until'] = current_time + BLOCK_DURATION_MS
                    data['error_count'] = 0
                    logger.warning(f"IP: {ip} blocked due to too many consecutive errors.")
            elif data:
                 data['error_count'] = 0
        raise exc
    except Exception as exc:
        async with ip_data_lock:
            data = ip_data.get(ip)
            if data:
                data['error_count'] += 1
                logger.error(f"IP: {ip}, Error count: {data['error_count']} (Unhandled exception: {type(exc).__name__})", exc_info=True)
                if data['error_count'] >= MAX_ERROR_COUNT:
                    data['blocked_until'] = current_time + BLOCK_DURATION_MS
                    data['error_count'] = 0
                    logger.warning(f"IP: {ip} blocked due to too many consecutive errors.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal Server Error") from exc
    finally:
        if decrement_needed:
            await request.state.decrement_callback()

    async with ip_data_lock:
        data = ip_data.get(ip)
        if data:
            if response.status_code >= 400 and response.status_code < 600:
                data['error_count'] += 1
                logger.info(f"IP: {ip}, Error count: {data['error_count']} (Response status code: {response.status_code})")
                if data['error_count'] >= MAX_ERROR_COUNT:
                    data['blocked_until'] = current_time + BLOCK_DURATION_MS
                    data['error_count'] = 0
                    logger.warning(f"IP: {ip} blocked due to too many consecutive errors.")
            elif response.status_code < 400 :
                data['error_count'] = 0
    return response

async def cleanup_ip_data():
    while True:
        await asyncio.sleep(BLOCK_DURATION_MS * 2 / 1000)
        current_time = time.time() * 1000
        async with ip_data_lock:
            ips_to_delete = [
                ip for ip, data in ip_data.items()
                if data['active_requests'] == 0 and \
                   (current_time - data['last_request_time'] > QPS_WINDOW_MS * 2) and \
                   data['blocked_until'] < current_time
            ]
            for ip in ips_to_delete:
                del ip_data[ip]
                logger.info(f"Cleaned up inactive IP data: {ip}")

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(cleanup_ip_data())
    logger.info("Background IP data cleanup task started.")
    mimetypes.init()

@app.get("/")
async def read_root():
    return {"message": "Welcome to the FastAPI server!"}

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    if form_data.username != FIXED_USERNAME or not verify_password(form_data.password, FIXED_PASSWORD_HASH):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    fixed_user = get_fixed_api_user(FIXED_USERNAME)
    if not fixed_user or fixed_user.disabled:
         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="API access disabled")
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": FIXED_USERNAME}, expires_delta=access_token_expires
    )
    logger.info(f"User '{FIXED_USERNAME}' logged in successfully, token issued.")
    return {"access_token": access_token, "token_type": "bearer"}



# --- Uvicorn Launch Configuration ---
ssl_root="/home/xkb2/ACME.sh/https/acme.sh/cert/"
ssl_certfile = os.path.join(ssl_root, "fullchain.cer")
ssl_keyfile = os.path.join(ssl_root, "*.744204541.xyz.key")

if __name__ == "__main__":
    
    logger.info(f"WEBDAV_BASE_URL: {WEBDAV_BASE_URL}")
    logger.info(f"WEBDAV_USERNAME: {WEBDAV_USERNAME}")
    logger.info(f"WEBDAV_PASSWORD set: {'Yes' if WEBDAV_PASSWORD != 'default_webdav_password' else 'No (using default or empty)'}")
    logger.info(f"WEBDAV_IMAGE_ROOT_IN_PATH: {WEBDAV_IMAGE_ROOT_IN_PATH}")

    logger.info(f"Starting Uvicorn with SSL on port 60443. Cert: {ssl_certfile}, Key: {ssl_keyfile}")
    uvicorn.run(
        "__main__:app",
        host="0.0.0.0",
        port=60000,
        log_level="info",
        ssl_keyfile=ssl_keyfile,
        ssl_certfile=ssl_certfile,
        reload=True,
        reload_dirs=["/home/xkb2/Desktop/HQ/api_gateway"], 
        reload_excludes=[
            "logs/*", "__pycache__/*", ".git/*", "*.pyc"
        ]
    )