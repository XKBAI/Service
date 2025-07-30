# api_gateway_optimized.py
import httpx
from fastapi import (
    FastAPI, Request, Response, HTTPException, Depends, status,
    File, UploadFile, Form, Body, Path, Query, BackgroundTasks,
    APIRouter
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
from enum import Enum
import uvicorn
import asyncio
import logging
import os
import io
from datetime import datetime, timedelta
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request as StarletteRequest
from starlette.responses import Response as StarletteResponse
from typing import AsyncGenerator
# --- 全局功能开关和配置 ---
ENABLE_LOGIN_IP_LOCKOUT = False
LOGIN_ATTEMPTS_LIMIT = 10
LOGIN_LOCKOUT_WINDOW_SECONDS = 60
LOGIN_LOCKOUT_DURATION_SECONDS = 60

ENABLE_CONCURRENT_REQUEST_LIMIT = False
CONCURRENT_REQUEST_LIMIT_PER_IP = 10

# --- 日志配置 ---
logger = logging.getLogger(__name__)

# --- 认证组件 ---
try:
    from authentication.auth import (
        User, Token, get_current_api_user,
        verify_password, create_access_token,
        ACCESS_TOKEN_EXPIRE_MINUTES,
        FIXED_USERNAME, FIXED_PASSWORD_HASH,
        get_fixed_api_user
    )
    oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
    AUTH_CONFIGURED = True
    logger.info("成功导入认证组件")
except ImportError as e:
    AUTH_CONFIGURED = False
    logger.error(f"导入认证组件失败: {e}")
    async def get_current_api_user():
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="认证模块未配置")
    
    class OAuth2PasswordRequestFormPlaceholder:
        def __init__(self, grant_type: Optional[str] = None, username: Optional[str] = None, password: Optional[str] = None, scope: str = "", client_id: Optional[str] = None, client_secret: Optional[str] = None):
            self.grant_type = grant_type
            self.username = username
            self.password = password
            self.scope = scope
            self.client_id = client_id
            self.client_secret = client_secret
    OAuth2PasswordRequestForm = OAuth2PasswordRequestFormPlaceholder
    Token = Dict
    User = Dict

# --- 后端服务配置 ---
BACKEND_SERVICES = {
    "llm": {
        "base_url": os.getenv("LLM_SERVICE_URL", "http://192.168.2.3:61080"),
        "prefix": "/llm",
        "actual_health_endpoint": "/",
        "health_check_method": "GET"
    },
    "stt": {
        "base_url": os.getenv("STT_SERVICE_URL", "http://192.168.2.4:57001"),
        "prefix": "/stt",
        "actual_health_endpoint": "/stt/health",
        "health_check_method": "GET"
    },
    "tts": {
        "base_url": os.getenv("TTS_SERVICE_URL", "http://192.168.2.4:57002"),
        "prefix": "/tts",
        "actual_health_endpoint": "/tts/status",
        "health_check_method": "GET"
    },
    "ocr": {
        "base_url": os.getenv("OCR_SERVICE_URL", "http://192.168.2.4:57004"),
        "prefix": "/olmocr",
        "actual_health_endpoint": "/olmocr/",
        "health_check_method": "GET"
    },
    "user": {
        "base_url": os.getenv("USER_SERVICE_URL", "http://192.168.2.4:58000"),
        "prefix": "/user",
        "actual_health_endpoint": "/",
        "health_check_method": "GET"
    },
    "md2pdf": {
        "base_url": os.getenv("MD2PDF_SERVICE_URL", "http://192.168.2.3:55000"),
        "prefix": "/md2pdf",
        "actual_health_endpoint": "/",
        "health_check_method": "GET"
    }
}

# 全局HTTP客户端字典 - 连接池在这里
http_clients: Dict[str, httpx.AsyncClient] = {}

# --- 用于IP限制的数据结构 ---
login_failure_tracker: Dict[str, Dict[str, Any]] = {}
locked_out_ips: Dict[str, datetime] = {}
ip_concurrent_requests: Dict[str, int] = {}
ip_concurrent_requests_lock = asyncio.Lock()

# --- Pydantic 模型定义 ---
class MD2PDFConvertRequest(BaseModel):
    markdown_content: str = Field(..., description="要转换的Markdown内容")
    skip_images: bool = Field(False, description="是否跳过图片")

class MD2PDFConvertResponse(BaseModel):
    success: bool
    message: str
    download_url: str
    filename: str
    file_id: str
    expires_in_seconds: int

class MD2PDFFileInfo(BaseModel):
    file_id: str
    filename: str
    download_url: str
    remaining_seconds: int

class MD2PDFFileListResponse(BaseModel):
    files: List[MD2PDFFileInfo]
    total_count: int
    
class TranscriptionSegment(BaseModel): 
    start: float
    end: float
    text: str

class TranscriptionResponse(BaseModel): 
    language: str
    language_probability: float
    segments: List[TranscriptionSegment]
    execution_time: float
    task_id: str

class TTSRequest(BaseModel):
    text: str = Field(..., description="要转换的文本")
    language: str = Field("ZH", description="语言代码")
    instruct_text: Optional[str] = Field('', description="指令文本（可选）")
    speaker_id: Optional[str] = Field("ZH", description="说话人ID")
    speed: float = Field(1.0, description="语速 (0.5-2.0)")

class OLMOCRConfig(BaseModel): 
    model: Optional[str] = None
    model_max_context: Optional[int] = None

class JobStatus(str, Enum): 
    PENDING = "pending"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class Job(BaseModel): 
    id: str
    filename: str
    status: JobStatus = JobStatus.PENDING
    result_path: Optional[str] = None
    error: Optional[str] = None
    created_at: str
    queue_position: Optional[int] = None

class Message(BaseModel): 
    role: str
    content: str

class UserBase(BaseModel): 
    user_id: str

class ChatContentRequest(BaseModel): 
    chat_time: str
    messages: list
    chat_title: Optional[str] = '新聊天'

class ChatSessionIdRequest(BaseModel): 
    chat_session_id: str

class ChatSessionUpdateRequest(BaseModel): 
    chat_session_id: str
    chat_time: str
    chat_title: Optional[str] = None
    messages: List[Message]

class SuccessResponse(BaseModel): 
    message: str

class SuccessChangeTitleResponse(BaseModel): 
    message: str
    title: str

class ChatSessionIdResponse(BaseModel): 
    chat_session_id: str

class ChatSessionResponse(BaseModel): 
    chat_session_id: str
    chat_time: datetime
    chat_title: str
    messages: list

class UserListResponse(BaseModel): 
    users: list

class ChatSessionInfo(BaseModel): 
    chat_session_id: str
    chat_title: str
    chat_time: datetime

class LLMChatMessage(BaseModel): 
    role: str = Field(..., description="消息发送者角色")
    content: str = Field(..., description="消息内容")

class LLMChatHistory(BaseModel): 
    messages: Optional[List[LLMChatMessage]] = Field([], description="聊天历史记录")

class LLMChatCompletionsRequest(BaseModel): 
    prompt: str = Field(..., description="用户当前的输入提示")
    messages: Optional[List[LLMChatMessage]] = Field([], description="包含先前对话的消息列表")
    chat_type: str = Field(..., description="聊天类型")
    education_level: Optional[str] = 'junior'
    subject: Optional[str] = 'biology'
    collection_type: Optional[str] = 'content'

class LLMChatTitleResponse(BaseModel): 
    title: str = Field(..., description="生成的聊天标题")

# --- 辅助函数：获取客户端IP ---
def get_client_ip(request: StarletteRequest) -> Optional[str]:
    """获取客户端的真实IP地址"""
    x_forwarded_for = request.headers.get("x-forwarded-for")
    if x_forwarded_for:
        ip = x_forwarded_for.split(",")[0].strip()
        return ip
    
    x_real_ip = request.headers.get("x-real-ip")
    if x_real_ip:
        return x_real_ip.strip()
        
    if request.client and request.client.host:
        return request.client.host
        
    return None

# --- 中间件：IP并发请求限制 ---
class ConcurrentRequestLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: StarletteRequest, call_next: RequestResponseEndpoint) -> StarletteResponse:
        if not ENABLE_CONCURRENT_REQUEST_LIMIT:
            response = await call_next(request)
            return response

        client_ip = get_client_ip(request)
        if not client_ip:
            logger.warning("无法获取客户端IP以进行并发限制")
            response = await call_next(request)
            return response

        # 为流式端点设置更高的并发限制
        is_streaming_endpoint = (
            request.url.path.endswith('/chat/completions') or
            request.url.path.endswith('/transcribe/') or
            request.url.path.endswith('/synthesize')
        )
        
        limit = CONCURRENT_REQUEST_LIMIT_PER_IP * 2 if is_streaming_endpoint else CONCURRENT_REQUEST_LIMIT_PER_IP

        async with ip_concurrent_requests_lock:
            current_concurrency = ip_concurrent_requests.get(client_ip, 0)
            if current_concurrency >= limit:
                logger.warning(f"IP {client_ip} 的并发请求已达到上限 {limit}")
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={"detail": f"来自您IP的并发请求过多，请稍后再试。限制为 {limit}"}
                )
            ip_concurrent_requests[client_ip] = current_concurrency + 1
            logger.debug(f"IP {client_ip} 并发请求数增加到 {ip_concurrent_requests[client_ip]}")

        response_from_call_next = None
        try:
            response_from_call_next = await call_next(request)
        finally:
            async with ip_concurrent_requests_lock:
                if client_ip in ip_concurrent_requests:
                    ip_concurrent_requests[client_ip] -= 1
                    if ip_concurrent_requests[client_ip] <= 0:
                        del ip_concurrent_requests[client_ip]
                    logger.debug(f"IP {client_ip} 并发请求数减少到 {ip_concurrent_requests.get(client_ip, 0)}")
        
        return response_from_call_next

# --- FastAPI 应用 ---
app = FastAPI(
    title="统一 AI 服务网关 (优化版)",
    description="高性能的AI服务网关，支持流式传输和并发处理",
    version="2.0.0"
)

# 添加中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(ConcurrentRequestLimitMiddleware)

# --- 生命周期事件 ---
@app.on_event("startup")
async def startup_event():
    logger.info("API 网关启动中，正在初始化 HTTP 客户端...")
    
    # 连接池配置说明：
    # - max_connections: 总连接数上限（包括所有主机）
    # - max_keepalive_connections: 保持活跃的连接数
    # - 这些连接是异步的，不是多线程
    # - 连接池复用TCP连接，减少建立/关闭连接的开销
    
    for service_name, config in BACKEND_SERVICES.items():
        http_clients[service_name] = httpx.AsyncClient(
            base_url=config["base_url"],
            # 超时配置：连接超时5秒，总超时600秒
            timeout=httpx.Timeout(600.0, connect=5.0),
            # 连接池配置：异步连接池，不是多线程
            limits=httpx.Limits(
                max_connections=50,      # 最大连接数
                max_keepalive_connections=20  # 保持活跃连接数
            ),
            follow_redirects=False,
            # 启用HTTP/2以提高性能
            http2=True
        )
        logger.info(f"为服务 '{service_name}' 创建异步客户端: {config['base_url']}")
    
    logger.info("所有 HTTP 客户端已初始化 (使用异步连接池)")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("API 网关关闭中，正在关闭 HTTP 客户端...")
    await asyncio.gather(*(client.aclose() for client in http_clients.values()))
    logger.info("所有 HTTP 客户端已关闭")

# --- OAuth2 token 端点 ---
if AUTH_CONFIGURED:
    @app.post("/token", response_model=Token, tags=["Authentication"])
    async def login_for_access_token(request: Request, form_data: OAuth2PasswordRequestForm = Depends()):
        client_ip = get_client_ip(request)
        
        if ENABLE_LOGIN_IP_LOCKOUT and client_ip:
            if client_ip in locked_out_ips:
                lock_expiry_time = locked_out_ips[client_ip]
                if datetime.now() < lock_expiry_time:
                    logger.warning(f"IP {client_ip} 因登录失败次数过多而被锁定")
                    raise HTTPException(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS, 
                        detail=f"您的IP因尝试次数过多已被暂时锁定。请在 {LOGIN_LOCKOUT_DURATION_SECONDS // 60} 分钟后重试。"
                    )
                else:
                    del locked_out_ips[client_ip]
                    if client_ip in login_failure_tracker: 
                        del login_failure_tracker[client_ip]
        
        logger.debug(f"用户 {form_data.username} 尝试登录，IP: {client_ip}")
        
        login_successful = False
        if form_data.username == FIXED_USERNAME and verify_password(form_data.password, FIXED_PASSWORD_HASH):
            login_successful = True
        
        if not login_successful:
            logger.warning(f"用户 {form_data.username} 认证失败，IP: {client_ip}")
            if ENABLE_LOGIN_IP_LOCKOUT and client_ip:
                current_time = datetime.now()
                if client_ip not in login_failure_tracker or \
                   (current_time - login_failure_tracker[client_ip]["window_start"]).total_seconds() > LOGIN_LOCKOUT_WINDOW_SECONDS:
                    login_failure_tracker[client_ip] = {"count": 1, "window_start": current_time}
                else:
                    login_failure_tracker[client_ip]["count"] += 1
                    if login_failure_tracker[client_ip]["count"] >= LOGIN_ATTEMPTS_LIMIT:
                        lock_until = current_time + timedelta(seconds=LOGIN_LOCKOUT_DURATION_SECONDS)
                        locked_out_ips[client_ip] = lock_until
                        del login_failure_tracker[client_ip] 
                        raise HTTPException(
                            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                            detail=f"您的IP因尝试次数过多已被暂时锁定。请在 {LOGIN_LOCKOUT_DURATION_SECONDS // 60} 分钟后重试。"
                        )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, 
                detail="用户名或密码错误", 
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        if ENABLE_LOGIN_IP_LOCKOUT and client_ip and client_ip in login_failure_tracker:
            del login_failure_tracker[client_ip]

        user_obj = get_fixed_api_user(FIXED_USERNAME)
        user_name_for_token = FIXED_USERNAME 
        if user_obj and hasattr(user_obj, 'username') and user_obj.username:
             user_name_for_token = user_obj.username
        
        if not user_obj or (hasattr(user_obj, 'disabled') and user_obj.disabled):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="用户访问被禁用或用户不存在")
        
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(data={"sub": user_name_for_token}, expires_delta=access_token_expires)
        logger.info(f"用户 '{form_data.username}' 登录成功，IP: {client_ip}")
        return {"access_token": access_token, "token_type": "bearer"}
else:
    @app.post("/token", tags=["Authentication"])
    async def login_for_access_token_disabled():
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="认证服务不可用")

async def forward_request_optimized(
    service_name: str, 
    request: Request, 
    target_path: str,
    request_data: Optional[Union[Dict, bytes]] = None,
    files: Optional[Dict[str, Any]] = None,
    is_streaming: bool = False
):
    """
    通用的请求转发函数
    - 支持所有HTTP方法 (GET, POST, PUT, DELETE, PATCH, OPTIONS等)
    - 支持流式和非流式响应
    - 支持直接传递字典或bytes数据
    - 自动处理Content-Length问题
    """
    client = http_clients.get(service_name)
    if not client:
        raise HTTPException(status_code=500, detail=f"服务 '{service_name}' 的客户端未初始化")

    # 🔥 安全的头部处理：只保留必要的头部
    safe_headers = {}
    
    # 保留这些安全的头部
    safe_header_names = [
        'accept', 'accept-encoding', 'accept-language', 
        'authorization', 'x-api-key', 'x-auth-token',
        'user-agent', 'referer', 'origin'
    ]
    
    for header_name in safe_header_names:
        if header_name in request.headers:
            safe_headers[header_name] = request.headers[header_name]
    
    # 添加转发头
    client_ip = get_client_ip(request)
    if client_ip:
        safe_headers["X-Forwarded-For"] = client_ip
    if request.url:
        safe_headers["X-Forwarded-Proto"] = request.url.scheme

    params = dict(request.query_params)
    method = request.method.upper()
    
    try:
        if is_streaming:
            # 🔥 流式处理：直接使用stream方法
            return await _handle_streaming_request(
                client, method, target_path, safe_headers, params, request_data, files
            )
        else:
            # 🔥 非流式处理：统一处理所有HTTP方法
            return await _handle_regular_request(
                client, method, target_path, safe_headers, params, request_data, files
            )
                
    except httpx.ConnectError as e:
        logger.error(f"连接后端服务 {service_name} 失败: {e}")
        raise HTTPException(status_code=503, detail=f"无法连接到后端服务 '{service_name}'")
    except httpx.TimeoutException as e:
        logger.error(f"请求后端服务 {service_name} 超时: {e}")
        raise HTTPException(status_code=504, detail=f"后端服务 '{service_name}' 请求超时")
    except httpx.HTTPStatusError as e:
        error_detail = e.response.text[:500] if e.response else str(e)
        logger.error(f"后端服务 {service_name} 返回错误: {error_detail}")
        raise HTTPException(status_code=e.response.status_code if e.response else 500, detail=error_detail)
    except Exception as e:
        logger.exception(f"处理请求到 {service_name} 时发生错误: {e}")
        raise HTTPException(status_code=500, detail="处理请求时发生内部服务器错误")


async def _handle_streaming_request(
    client, method: str, target_path: str, headers: dict, 
    params: dict, request_data, files
):
    """处理流式请求"""
    
    async def stream_generator():
        try:
            # 根据数据类型选择合适的参数
            request_kwargs = {
                "method": method,
                "url": target_path,
                "headers": headers,
                "params": params,
                "timeout": 3600
            }
            
            # 🔥 智能选择数据传递方式
            if files:
                # 文件上传
                request_kwargs.update({"data": request_data, "files": files})
            elif isinstance(request_data, dict):
                # 字典数据，让httpx自动处理JSON编码和Content-Length
                request_kwargs["json"] = request_data
            elif isinstance(request_data, bytes):
                # 字节数据
                request_kwargs["content"] = request_data
                if "content-type" not in [h.lower() for h in headers.keys()]:
                    headers["Content-Type"] = "application/json"
            elif request_data is not None:
                # 其他数据类型，转为JSON
                request_kwargs["json"] = request_data
            
            async with client.stream(**request_kwargs) as response:
                # 检查响应状态
                if response.status_code >= 400:
                    error_content = await response.aread()
                    error_text = error_content.decode('utf-8')[:500]
                    yield f"data: {{\"error\": \"Backend error {response.status_code}: {error_text}\"}}\\n\\n".encode('utf-8')
                    return
                
                # 流式转发数据
                async for chunk in response.aiter_bytes(chunk_size=1024):
                    if chunk:
                        yield chunk
                        
        except Exception as e:
            logger.error(f"流式传输错误: {e}")
            yield f"data: {{\"error\": \"Stream error: {str(e)}\"}}\\n\\n".encode('utf-8')
    
    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )


async def _handle_regular_request(
    client, method: str, target_path: str, headers: dict, 
    params: dict, request_data, files
):
    """处理常规请求"""
    
    # 🔥 统一的请求参数构建
    request_kwargs = {
        "url": target_path,
        "headers": headers,
        "params": params,
        "timeout": 3600
    }
    
    # 🔥 智能选择数据传递方式
    if files:
        # 文件上传
        request_kwargs.update({"data": request_data, "files": files})
    elif isinstance(request_data, dict):
        # 字典数据，让httpx自动处理JSON编码
        request_kwargs["json"] = request_data
    elif isinstance(request_data, bytes):
        # 字节数据
        request_kwargs["content"] = request_data
        if "content-type" not in [h.lower() for h in headers.keys()]:
            headers["Content-Type"] = "application/json"
    elif request_data is not None:
        # 其他数据类型，转为JSON
        request_kwargs["json"] = request_data
    
    # 🔥 根据HTTP方法选择对应的客户端方法
    if method == "GET":
        # GET请求通常不包含body数据
        request_kwargs = {k: v for k, v in request_kwargs.items() 
                         if k not in ["json", "content", "data", "files"]}
        response = await client.get(**request_kwargs)
    elif method == "POST":
        response = await client.post(**request_kwargs)
    elif method == "PUT":
        response = await client.put(**request_kwargs)
    elif method == "PATCH":
        response = await client.patch(**request_kwargs)
    elif method == "DELETE":
        response = await client.delete(**request_kwargs)
    elif method == "HEAD":
        request_kwargs = {k: v for k, v in request_kwargs.items() 
                         if k not in ["json", "content", "data", "files"]}
        response = await client.head(**request_kwargs)
    elif method == "OPTIONS":
        request_kwargs = {k: v for k, v in request_kwargs.items() 
                         if k not in ["json", "content", "data", "files"]}
        response = await client.options(**request_kwargs)
    else:
        # 其他HTTP方法，使用通用request方法
        response = await client.request(method, **request_kwargs)
    
    # 🔥 智能处理响应
    content_type = response.headers.get("content-type", "")
    
    if "application/json" in content_type:
        # JSON响应
        return response.json()
    elif content_type.startswith("text/"):
        # 文本响应
        return Response(
            content=response.text, 
            media_type=content_type, 
            status_code=response.status_code,
            headers=dict(response.headers)
        )
    else:
        # 二进制或其他类型响应
        return Response(
            content=response.content,
            media_type=content_type,
            status_code=response.status_code,
            headers=dict(response.headers)
        )


# async def direct_stream_forward(
#     service_name: str,
#     request: Request, 
#     target_path: str,
#     payload_dict: dict = None  # 直接传字典，不是bytes
# ):
#     """直接转发，避免多次JSON编解码"""
#     client = http_clients.get(service_name)
#     if not client:
#         raise HTTPException(status_code=500, detail=f"服务 '{service_name}' 的客户端未初始化")

#     # 只保留必要的头部，避免Content-Length冲突
#     headers = {
#         "Accept": request.headers.get("accept", "*/*"),
#         "User-Agent": request.headers.get("user-agent", "FastAPI-Gateway"),
#     }
    
#     # 保留认证头部
#     for auth_header in ["authorization", "x-api-key", "x-auth-token"]:
#         if auth_header in request.headers:
#             headers[auth_header] = request.headers[auth_header]

#     async def stream_generator():
#         try:
#             # 🔥 关键：直接使用json参数，让httpx处理所有编码和Content-Length
#             async with client.stream(
#                 method="POST",
#                 url=target_path,
#                 headers=headers,
#                 json=payload_dict,  # 直接传字典
#                 params=dict(request.query_params),
#                 timeout=3600
#             ) as response:
#                 async for chunk in response.aiter_bytes():
#                     yield chunk
                        
#         except Exception as e:
#             logger.error(f"流式转发错误: {e}")
#             yield f"data: {{\"error\": \"Stream error: {str(e)}\"}}\\n\\n".encode('utf-8')
    
#     return StreamingResponse(
#         stream_generator(),
#         media_type="text/event-stream",
#         headers={
#             "Cache-Control": "no-cache",
#             "Connection": "keep-alive"
#         }
#     )
# # --- 认证依赖 ---
auth_dependency = [Depends(get_current_api_user)] if AUTH_CONFIGURED else []

# --- LLM 路由 ---
llm_router = APIRouter(prefix="/llm", tags=["LLM Service"], dependencies=auth_dependency)

@llm_router.get("/images/{image_path:path}", summary="代理LLM服务中的WebDAV图片")
async def llm_get_image_proxy(request: Request, image_path: str = Path(...)):
    logger.info(f"LLM 图片代理请求路径: {image_path}")
    return await forward_request_optimized("llm", request, f"/get_images/{image_path.lstrip('/')}")

@llm_router.post("/chat/completions", summary="与LLM进行流式聊天补全")
async def llm_chat_completions(payload: LLMChatCompletionsRequest, request: Request):
    logger.info(f"LLM 聊天补全请求类型: {payload.chat_type}")
    return await forward_request_optimized(
        "llm", 
        request, 
        "/chat/completions", 
        request_data=payload.model_dump_json().encode('utf-8'),
        is_streaming=True
    )
    
    
# async def llm_chat_completions(payload: LLMChatCompletionsRequest, request: Request):
#     return await direct_stream_forward(
#         "llm",
#         request,
#         "/chat/completions", 
#         payload_dict=payload.model_dump()  # 直接传字典，不编码
#     )
    


@llm_router.post("/get_chat_title", response_model=LLMChatTitleResponse, summary="获取LLM生成的聊天标题")
async def llm_get_chat_title(payload: LLMChatHistory, request: Request):
    logger.info("LLM 获取聊天标题请求")
    return await forward_request_optimized(
        "llm", 
        request, 
        "/get_chat_title/", 
        request_data=payload.model_dump_json().encode('utf-8'),
        is_streaming=False
    )

# --- STT 路由 ---
stt_router = APIRouter(prefix="/stt", tags=["STT Service"], dependencies=auth_dependency)

@stt_router.post("/transcribe/", response_model=TranscriptionResponse)
async def stt_transcribe(
    request: Request, 
    file: UploadFile = File(...), 
    language: Optional[str] = Form("zh"), 
    beam_size: int = Form(5), 
    vad_filter: bool = Form(True), 
    min_silence_duration_ms: Optional[int] = Form(1000)
):
    form_data = {
        'language': language, 
        'beam_size': str(beam_size), 
        'vad_filter': str(vad_filter)
    }
    if min_silence_duration_ms is not None: 
        form_data['min_silence_duration_ms'] = str(min_silence_duration_ms)
    
    files_data = {'file': (file.filename, await file.read(), file.content_type)}
    
    return await forward_request_optimized("stt", request, "/stt/transcribe/", request_data=form_data, files=files_data)

@stt_router.get("/health")
async def stt_health(request: Request): 
    return await forward_request_optimized("stt", request, "/stt/health")

@stt_router.get("/")
async def stt_root(request: Request): 
    return await forward_request_optimized("stt", request, "/stt/")

@stt_router.post("/clear-memory")
async def stt_clear_memory(request: Request): 
    return await forward_request_optimized("stt", request, "/stt/clear-memory")

# --- TTS 路由 ---
tts_router = APIRouter(prefix="/tts", tags=["TTS Service"], dependencies=auth_dependency)

@tts_router.post("/synthesize")
async def tts_synthesize(request: Request, tts_input: TTSRequest = Body(...)):
    return await forward_request_optimized("tts", request, "/tts/synthesize", request_data=tts_input.model_dump_json().encode('utf-8'))

@tts_router.get("/status")
async def tts_status(request: Request): 
    return await forward_request_optimized("tts", request, "/tts/status")

# --- OCR 路由 ---
ocr_router = APIRouter(prefix="/olmocr", tags=["OCR Service"], dependencies=auth_dependency)

@ocr_router.post("/process", response_model=Job)
async def ocr_process(request: Request, file: UploadFile = File(...)):
    files_data = {'file': (file.filename, await file.read(), file.content_type)}
    return await forward_request_optimized("ocr", request, "/olmocr/process", files=files_data)

@ocr_router.get("/results/{job_id}")
async def ocr_get_results(request: Request, job_id: str = Path(...)):
    return await forward_request_optimized("ocr", request, f"/olmocr/results/{job_id}")

# --- User 路由 ---
user_router = APIRouter(prefix="/user", tags=["User Service"], dependencies=auth_dependency)

@user_router.get("/get_all_users/", response_model=UserListResponse)
async def get_all_users(request: Request):
    return await forward_request_optimized("user", request, "/get_all_users/")

@user_router.post("/add_user/", response_model=SuccessResponse)
async def add_user(request: Request, user_payload: UserBase):
    return await forward_request_optimized("user", request, "/add_user/", request_data=user_payload.model_dump_json().encode('utf-8'))

@user_router.delete("/del_user/", response_model=SuccessResponse)
async def del_user(request: Request, user_payload: UserBase):
    return await forward_request_optimized("user", request, "/del_user/", request_data=user_payload.model_dump_json().encode('utf-8'))

@user_router.get("/get_user_chat/", response_model=List[ChatSessionInfo])
async def get_user_chat(request: Request, user_id: str):
    return await forward_request_optimized("user", request, f"/get_user_chat/?user_id={user_id}")

@user_router.post("/add_chat/", response_model=ChatSessionIdResponse)
async def add_chat(request: Request, user_id: str, chat_content: ChatContentRequest):
    # 特殊处理：使用JSON格式而不是字节
    client = http_clients.get("user")
    if not client:
        raise HTTPException(status_code=500, detail="User 客户端未初始化")
    
    headers = dict(request.headers)
    for h in ["host", "content-length", "content-type", "x-api-key", "authorization"]:
        headers.pop(h, None)
    headers["Content-Type"] = "application/json"
    
    try:
        response = await client.post(
            f"/add_chat/?user_id={user_id}",
            json=chat_content.model_dump(),
            headers=headers,
            timeout=30.0
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.exception(f"add_chat 错误: {e}")
        raise HTTPException(status_code=500, detail=f"add_chat 内部错误: {str(e)}")

@user_router.delete("/del_chat/", response_model=SuccessResponse)
async def del_chat(request: Request, chat_session: ChatSessionIdRequest):
    return await forward_request_optimized("user", request, "/del_chat/", request_data=chat_session.model_dump_json().encode('utf-8'))

@user_router.put("/edit_chat/", response_model=SuccessResponse)
async def edit_chat(request: Request, chat_session: ChatSessionUpdateRequest):
    return await forward_request_optimized("user", request, "/edit_chat/", request_data=chat_session.model_dump_json().encode('utf-8'))

@user_router.put("/update_chat/", response_model=SuccessResponse)
async def update_chat(request: Request, chat_session: ChatSessionUpdateRequest):
    return await forward_request_optimized("user", request, "/update_chat/", request_data=chat_session.model_dump_json().encode('utf-8'))

@user_router.get("/get_chat_by_session_id/", response_model=ChatSessionResponse)
async def get_chat_by_session_id(request: Request, chat_session_id: str):
    return await forward_request_optimized("user", request, f"/get_chat_by_session_id/?chat_session_id={chat_session_id}")

@user_router.put("/update_chat_title/", response_model=SuccessChangeTitleResponse)
async def update_chat_title(request: Request, chat_session: ChatSessionResponse):
    return await forward_request_optimized("user", request, "/update_chat_title/", request_data=chat_session.model_dump_json().encode('utf-8'))

@user_router.put("/edit_chat_title/", response_model=SuccessChangeTitleResponse)
async def edit_chat_title(request: Request, chat_session: ChatSessionInfo):
    return await forward_request_optimized("user", request, "/edit_chat_title/", request_data=chat_session.model_dump_json().encode('utf-8'))

# --- MD2PDF 路由 ---
md2pdf_router = APIRouter(prefix="/md2pdf", tags=["MD2PDF Service"], dependencies=auth_dependency)

@md2pdf_router.get("/api/download/{file_id}", summary="下载生成的PDF文件")
async def md2pdf_download(request: Request, file_id: str = Path(..., description="PDF文件ID")):
    """下载之前生成的PDF文件"""
    if not file_id.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="无效的文件ID")
    
    return await forward_request_optimized("md2pdf", request, f"/api/download/{file_id}")

# --- 网关自身路由 ---
@app.get("/", tags=["Gateway Info"])
async def get_root():
    return {
        "message": "欢迎使用统一 AI 服务网关 (优化版)",
        "version": app.version,
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "registered_services": list(BACKEND_SERVICES.keys()),
        "authentication_status": "已配置" if AUTH_CONFIGURED else "未配置",
        "connection_pool_info": {
            "type": "异步连接池 (Async Connection Pool)",
            "description": "使用httpx异步客户端，支持连接复用和HTTP/2",
            "max_connections_per_service": 50,
            "max_keepalive_connections_per_service": 20
        },
        "features": {
            "streaming_support": True,
            "connection_pooling": True,
            "http2_support": True,
            "concurrent_limiting": ENABLE_CONCURRENT_REQUEST_LIMIT,
            "login_ip_lockout": ENABLE_LOGIN_IP_LOCKOUT
        }
    }

@app.get("/health", tags=["Gateway Info"])
async def health_check():
    service_status = {}
    gateway_healthy = True
    auth_module_status = "已加载" if AUTH_CONFIGURED else "加载失败"

    for service_name, client_instance in http_clients.items():
        if service_name not in BACKEND_SERVICES:
            continue
        
        service_config = BACKEND_SERVICES[service_name]
        check_url = service_config.get('actual_health_endpoint', "/")
        check_method = service_config.get('health_check_method', "GET").upper()
        target_display_url = str(client_instance.base_url).rstrip('/') + check_url
        
        try:
            if check_method == "GET":
                response = await client_instance.get(check_url, timeout=15.0)
            elif check_method == "POST":
                response = await client_instance.post(check_url, timeout=15.0)
            else:
                response = await client_instance.head(check_url, timeout=15.0)
            
            if 200 <= response.status_code < 300:
                service_status[service_name] = {
                    "status": "健康",
                    "target": target_display_url,
                    "method": check_method,
                    "code": response.status_code,
                    "response_time_ms": response.elapsed.total_seconds() * 1000
                }
            else:
                error_text = response.text[:200] if response.content else "无响应内容"
                service_status[service_name] = {
                    "status": "不健康",
                    "target": target_display_url,
                    "method": check_method,
                    "code": response.status_code,
                    "detail": error_text
                }
                gateway_healthy = False
        except Exception as e:
            service_status[service_name] = {
                "status": "不可达",
                "target": target_display_url,
                "method": check_method,
                "error": type(e).__name__,
                "detail": str(e)
            }
            gateway_healthy = False
            
    return {
        "gateway_status": "健康" if gateway_healthy and AUTH_CONFIGURED else "降级",
        "authentication_module": auth_module_status,
        "backend_service_status": service_status,
        "connection_pool_stats": {
            "total_clients": len(http_clients),
            "pool_type": "异步连接池",
            "http2_enabled": True
        }
    }

# --- 包含所有路由 ---
app.include_router(stt_router)
app.include_router(tts_router)
app.include_router(ocr_router)
app.include_router(user_router)
app.include_router(llm_router)
app.include_router(md2pdf_router)

# --- Uvicorn 启动配置 ---
if __name__ == "__main__":
    # SSL 配置
    ssl_root_dir = os.getenv("SSL_ROOT_DIR", "/home/xkb2/ACME.sh/https/acme.sh/cert/")
    ssl_certfile_name = os.getenv("SSL_CERT_NAME", "fullchain.cer")
    ssl_keyfile_name = os.getenv("SSL_KEY_NAME", "*.744204541.xyz.key")
    ssl_certfile = os.path.join(ssl_root_dir, ssl_certfile_name)
    ssl_keyfile = os.path.join(ssl_root_dir, ssl_keyfile_name)
    
    # 启动参数
    gateway_port = int(os.getenv("GATEWAY_PORT", 60443))
    reload_mode = os.getenv("GATEWAY_RELOAD", "true").lower() == "true"
    log_level = os.getenv("LOG_LEVEL", "info").lower()
    
    # 日志配置
    logging.basicConfig(
        level=log_level.upper(),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info(f"启动优化版 API 网关，端口: {gateway_port}, 重载: {reload_mode}, 日志级别: {log_level.upper()}")
    
    # 连接池说明
    logger.info("连接池配置说明:")
    logger.info("- 类型: 异步连接池 (不是多线程)")
    logger.info("- 每个服务最大连接数: 50")
    logger.info("- 每个服务保持活跃连接数: 20")
    logger.info("- 支持 HTTP/2 和连接复用")
    logger.info("- 使用 asyncio 事件循环，单线程异步处理")
    
    if not AUTH_CONFIGURED:
        logger.critical("警告: 认证系统未配置，大多数端点将无法正常工作")
    
    # SSL 检查
    use_ssl = True
    if not os.path.exists(ssl_keyfile):
        logger.warning(f"SSL 密钥文件未找到: {ssl_keyfile}，HTTPS 将被禁用")
        use_ssl = False
    if not os.path.exists(ssl_certfile):
        logger.warning(f"SSL 证书文件未找到: {ssl_certfile}，HTTPS 将被禁用")
        use_ssl = False
    
    # Uvicorn 配置
    uvicorn_config = {
        "host": "0.0.0.0",
        "port": gateway_port,
        "reload": reload_mode,
        "log_level": log_level,
        # 优化 Uvicorn 性能
        "loop": "asyncio",  # 使用 asyncio 事件循环
        "access_log": True,
        "use_colors": True,
    }
    
    # 热重载配置
    if reload_mode:
        uvicorn_config["reload_dirs"] = ["/home/xkb2/Desktop/HQ/api_gateway"]
        uvicorn_config["reload_includes"] = ["*.py"]
        uvicorn_config["reload_excludes"] = [
            "*.pyc", "__pycache__/*", "*.log", "*.tmp", 
            ".git/*", "venv/*", "env/*"
        ]
        logger.info(f"热重载已启用，监控目录: {uvicorn_config['reload_dirs']}")
    
    # SSL 配置
    if use_ssl:
        uvicorn_config["ssl_keyfile"] = ssl_keyfile
        uvicorn_config["ssl_certfile"] = ssl_certfile
        logger.info(f"SSL 已启用，证书: {ssl_certfile}")
    else:
        logger.info("SSL 已禁用")
    
    # 启动服务器
    uvicorn.run("__main__:app", **uvicorn_config)