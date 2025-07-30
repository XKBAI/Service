# api_gateway_explicit_docs.py (或者您实际的文件名)
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
# 修正导入：将 RequestResponseCallNext 替换为 RequestResponseEndpoint
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request as StarletteRequest # 明确导入以避免歧义
from starlette.responses import Response as StarletteResponse # 明确导入以避免歧义


# --- 全局功能开关和配置 ---
# 登录IP锁定功能
ENABLE_LOGIN_IP_LOCKOUT = False  # True: 开启登录IP锁定, False: 关闭
LOGIN_ATTEMPTS_LIMIT = 10       # 同一IP在一分钟内允许的最大错误尝试次数
LOGIN_LOCKOUT_WINDOW_SECONDS = 60 # 错误尝试的统计窗口时间（秒）
LOGIN_LOCKOUT_DURATION_SECONDS = 60 # IP被锁定的持续时间（秒）

# IP并发请求限制功能
ENABLE_CONCURRENT_REQUEST_LIMIT = False # True: 开启并发请求限制, False: 关闭
CONCURRENT_REQUEST_LIMIT_PER_IP = 10   # 同一IP允许的最大并发请求数

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
    logger.info("成功导入认证组件 (Successfully imported authentication components from authentication.auth.)")
except ImportError as e:
    AUTH_CONFIGURED = False
    logger.error(f"导入认证组件失败 (Failed to import authentication components from authentication.auth): {e}")
    logger.error("OAuth2 认证将无法工作，请确保 'authentication/auth.py' 文件存在且配置正确。(OAuth2 authentication will not work. Please ensure 'authentication/auth.py' exists and is correctly configured.)")
    async def get_current_api_user(): # type: ignore
        if not AUTH_CONFIGURED:
             logger.critical("严重错误: 认证模块未加载，但有端点需要认证。(CRITICAL: Authentication module not loaded, but an endpoint requiring auth was called.)")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="认证模块未配置 (Authentication module not configured)")
    
    class OAuth2PasswordRequestFormPlaceholder: # type: ignore
        def __init__(self, grant_type: Optional[str] = None, username: Optional[str] = None, password: Optional[str] = None, scope: str = "", client_id: Optional[str] = None, client_secret: Optional[str] = None):
            self.grant_type = grant_type
            self.username = username
            self.password = password
            self.scope = scope
            self.client_id = client_id
            self.client_secret = client_secret
    OAuth2PasswordRequestForm = OAuth2PasswordRequestFormPlaceholder # type: ignore
    Token = Dict # type: ignore
    User = Dict # type: ignore

# --- 后端服务配置 ---
BACKEND_SERVICES = {
    "llm": {
        "base_url": os.getenv("LLM_SERVICE_URL", "http://192.168.2.3:61080"),
        "prefix": "/llm",
        "actual_health_endpoint": "/",
        "health_check_method": "GET"
    },
    "stt": {
        "base_url": os.getenv("STT_SERVICE_URL", "http://faster-whisper-stt:9000"),
        "prefix": "/stt",
        "actual_health_endpoint": "/stt/health",
        "health_check_method": "GET"
    },
    "tts": {
        "base_url": os.getenv("TTS_SERVICE_URL", "http://melotts-tts:9000"),
        "prefix": "/tts",
        "actual_health_endpoint": "/health",
        "health_check_method": "GET"
    },
    "ocr": {
        "base_url": os.getenv("OCR_SERVICE_URL", "http://olmocr-service:9000"),
        "prefix": "/olmocr",
        "actual_health_endpoint": "/",
        "health_check_method": "GET"
    },
    "user": {
        "base_url": os.getenv("USER_SERVICE_URL", "http://host.docker.internal:55003"),
        "prefix": "/user",
        "actual_health_endpoint": "/get_all_users/",
        "health_check_method": "GET"
    },
    "md2pdf": {
        "base_url": os.getenv("MD2PDF_SERVICE_URL", "http://md2pdf-service:9000"),
        "prefix": "/md2pdf",
        "actual_health_endpoint": "/",
        "health_check_method": "GET"
    },
    "vlm": {
        "base_url": os.getenv("VLM_SERVICE_URL", "http://vlm-service:9000"),
        "prefix": "/vlm",
        "actual_health_endpoint": "/vlm/health",
        "health_check_method": "GET"
    }
}

http_clients: Dict[str, httpx.AsyncClient] = {}

# --- 用于IP限制的数据结构 ---
# 记录登录失败尝试: {"ip": {"count": N, "window_start": datetime}}
login_failure_tracker: Dict[str, Dict[str, Any]] = {}
# 记录被锁定的IP: {"ip": expiry_datetime}
locked_out_ips: Dict[str, datetime] = {}
# 记录IP的并发请求数: {"ip": count}
ip_concurrent_requests: Dict[str, int] = {}
# 用于保护 ip_concurrent_requests 字典的异步锁
ip_concurrent_requests_lock = asyncio.Lock()


# --- Pydantic 模型定义 (保持不变) ---
# 在 Pydantic 模型定义部分添加：
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
    
class TranscriptionSegment(BaseModel): start: float; end: float; text: str
class TranscriptionResponse(BaseModel): language: str; language_probability: float; segments: List[TranscriptionSegment]; execution_time: float; task_id: str
class TTSRequest(BaseModel): text: str = Field(..., description="要转换的文本"); language: str = Field("ZH", description="语言代码"); speaker_id: Optional[str] = Field("ZH", description="说话人ID"); speed: float = Field(1.0, description="语速 (0.5-2.0)")
class OLMOCRConfig(BaseModel): model: Optional[str] = None; model_max_context: Optional[int] = None
class JobStatus(str, Enum): PENDING = "pending"; QUEUED = "queued"; PROCESSING = "processing"; COMPLETED = "completed"; FAILED = "failed"
class Job(BaseModel): id: str; filename: str; status: JobStatus = JobStatus.PENDING; result_path: Optional[str] = None; error: Optional[str] = None; created_at: str; queue_position: Optional[int] = None
class Message(BaseModel): role: str; content: str
class UserBase(BaseModel): user_id: str
class ChatContentRequest(BaseModel): chat_time: str; messages: list; chat_title: Optional[str] = '新聊天'
class ChatSessionIdRequest(BaseModel): chat_session_id: str
class ChatSessionUpdateRequest(BaseModel): chat_session_id: str; chat_time: str; chat_title: Optional[str] = None; messages: List[Message]
class SuccessResponse(BaseModel): message: str
class SuccessChangeTitleResponse(BaseModel): message: str; title: str
class ChatSessionIdResponse(BaseModel): chat_session_id: str
class ChatSessionResponse(BaseModel): chat_session_id: str; chat_time: datetime; chat_title: str; messages: list
class UserListResponse(BaseModel): users: list
class ChatSessionInfo(BaseModel): chat_session_id: str; chat_title: str; chat_time: datetime
class LLMChatMessage(BaseModel): role: str = Field(..., description="消息发送者角色 (例如 'user', 'assistant')"); content: str = Field(..., description="消息内容")
class LLMChatHistory(BaseModel): messages: Optional[List[LLMChatMessage]] = Field([], description="聊天历史记录 (用于获取标题)")
class LLMChatCompletionsRequest(BaseModel): prompt: str = Field(..., description="用户当前的输入提示"); messages: Optional[List[LLMChatMessage]] = Field([], description="包含先前对话的消息列表"); chat_type: str = Field(..., description="聊天类型")
class LLMChatTitleResponse(BaseModel): title: str = Field(..., description="生成的聊天标题")

# --- 辅助函数：获取客户端IP ---
def get_client_ip(request: StarletteRequest) -> Optional[str]: # 使用 StarletteRequest
    """
    获取客户端的真实IP地址。
    优先从 X-Forwarded-For 获取，然后是 X-Real-IP，最后是 request.client.host。
    """
    x_forwarded_for = request.headers.get("x-forwarded-for")
    if x_forwarded_for:
        # X-Forwarded-For 可能是一个逗号分隔的IP列表，第一个通常是原始客户端IP
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
    # 修正 dispatch 方法的参数类型注解
    async def dispatch(self, request: StarletteRequest, call_next: RequestResponseEndpoint) -> StarletteResponse:
        if not ENABLE_CONCURRENT_REQUEST_LIMIT:
            # FastAPI 的 Request 和 Response 类型与 Starlette 的兼容，但为了清晰，这里可以用 Starlette 类型
            response = await call_next(request)
            return response # type: ignore

        client_ip = get_client_ip(request)
        if not client_ip:
            # 如果无法获取IP，则不进行限制 (或者可以考虑拒绝请求)
            logger.warning("无法获取客户端IP以进行并发限制。(Could not get client IP for concurrency limiting.)")
            response = await call_next(request)
            return response # type: ignore

        async with ip_concurrent_requests_lock:
            current_concurrency = ip_concurrent_requests.get(client_ip, 0)
            if current_concurrency >= CONCURRENT_REQUEST_LIMIT_PER_IP:
                logger.warning(f"IP {client_ip} 的并发请求已达到上限 {CONCURRENT_REQUEST_LIMIT_PER_IP}，当前为 {current_concurrency}。(Concurrent request limit reached for IP {client_ip}. Limit: {CONCURRENT_REQUEST_LIMIT_PER_IP}, Current: {current_concurrency})")
                # 直接使用 FastAPI 的 JSONResponse 或 Starlette 的 JSONResponse 均可
                return JSONResponse( 
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={"detail": f"来自您IP的并发请求过多，请稍后再试。限制为 {CONCURRENT_REQUEST_LIMIT_PER_IP}。(Too many concurrent requests from your IP. Please try again later. Limit is {CONCURRENT_REQUEST_LIMIT_PER_IP}.)"}
                )
            ip_concurrent_requests[client_ip] = current_concurrency + 1
            logger.debug(f"IP {client_ip} 并发请求数增加到 {ip_concurrent_requests[client_ip]}")

        response_from_call_next = None
        try:
            response_from_call_next = await call_next(request)
        finally:
            async with ip_concurrent_requests_lock:
                if client_ip in ip_concurrent_requests: # 再次检查以防万一
                    ip_concurrent_requests[client_ip] -= 1
                    if ip_concurrent_requests[client_ip] <= 0:
                        del ip_concurrent_requests[client_ip] # 清理计数为0的IP
                    logger.debug(f"IP {client_ip} 并发请求数减少。新计数: {ip_concurrent_requests.get(client_ip, 0)}")
        return response_from_call_next # type: ignore

# --- FastAPI 应用 ---
app = FastAPI(
    title="统一 AI 服务网关 (OAuth2 认证与IP限制版)",
    description="将 STT, TTS, OCR, User, LLM 服务聚合到单一入口。使用 OAuth2 Bearer Token 进行认证，并包含IP登录锁定和并发请求限制功能。",
    version="1.4.1" # 版本号更新
)

# 添加中间件 - 注意：中间件的添加顺序很重要
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该设置具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(ConcurrentRequestLimitMiddleware)


# --- 生命周期事件 ---
@app.on_event("startup")
async def startup_event():
    """API网关启动事件 - 初始化所有HTTP客户端"""
    logger.info("=== API 网关启动中，正在初始化 HTTP 客户端... ===")
    logger.info("(API Gateway starting up, initializing HTTP clients...)")
    
    # 清空现有客户端字典（如果有的话）
    http_clients.clear()
    
    # 创建所有服务的HTTP客户端
    logger.info("正在创建新的HTTP客户端...")
    failed_services = []
    
    for service_name, config in BACKEND_SERVICES.items():
        try:
            # 创建异步HTTP客户端
            client = httpx.AsyncClient(
                base_url=config["base_url"],
                timeout=httpx.Timeout(
                    connect=10.0,    # 连接超时
                    read=300.0,      # 读取超时
                    write=30.0,      # 写入超时
                    pool=5.0         # 连接池超时
                ),
                limits=httpx.Limits(
                    max_connections=100,        # 最大连接数
                    max_keepalive_connections=20  # 最大保持连接数
                ),
                follow_redirects=False,
                verify=False  # 在内网环境中可以禁用SSL验证
            )
            
            http_clients[service_name] = client
            logger.info(f"✅ 为服务 '{service_name}' 创建客户端成功，目标: {config['base_url']}")
            
        except Exception as e:
            logger.error(f"❌ 创建服务 '{service_name}' 的HTTP客户端失败: {e}")
            failed_services.append(service_name)
    
    # 报告创建结果
    if failed_services:
        logger.warning(f"⚠️ 以下服务的客户端创建失败: {failed_services}")
    
    logger.info(f"📊 HTTP客户端创建完成。成功: {len(http_clients)}, 失败: {len(failed_services)}")
    
    # 验证关键服务的连接
    await verify_service_connections()
    
    # 检查认证模块状态
    if not AUTH_CONFIGURED:
        logger.warning("⚠️ 警告: 认证模块未能加载。API 网关可能无法正常提供认证保护。")
        logger.warning("(Warning: Authentication module failed to load. API Gateway may not provide authentication protection correctly.)")
    else:
        logger.info("✅ 认证模块已正确加载")
    
    logger.info(f"=== HTTP 客户端初始化完成。总计: {len(http_clients)} 个客户端 ===")
    logger.info("(HTTP clients initialization completed.)")


async def verify_service_connections():
    """验证关键服务的连接状态"""
    logger.info("🔍 开始验证服务连接...")
    
    # 定义关键服务和它们的健康检查端点
    critical_services = {
        'ocr': '/olmocr/',
        'stt': '/stt/health', 
        'tts': '/health',
        'user': '/get_all_users/',
        'llm': '/',
        'md2pdf': '/'
    }
    
    for service_name, health_endpoint in critical_services.items():
        if service_name not in http_clients:
            logger.warning(f"⚠️ 服务 '{service_name}' 的客户端未创建，跳过连接验证")
            continue
            
        client = http_clients[service_name]
        try:
            # 使用较短的超时时间进行健康检查
            response = await client.get(health_endpoint, timeout=5.0)
            if 200 <= response.status_code < 300:
                logger.info(f"✅ 服务 '{service_name}' 连接验证成功 (状态码: {response.status_code})")
            else:
                logger.warning(f"⚠️ 服务 '{service_name}' 响应异常 (状态码: {response.status_code})")
                
        except httpx.ConnectError:
            logger.warning(f"⚠️ 服务 '{service_name}' 连接失败 - 服务可能未启动")
        except httpx.TimeoutException:
            logger.warning(f"⚠️ 服务 '{service_name}' 连接超时")
        except Exception as e:
            logger.warning(f"⚠️ 服务 '{service_name}' 连接验证失败: {e}")
    
    logger.info("🔍 服务连接验证完成")


def create_http_client(service_config: dict, service_name: str) -> httpx.AsyncClient:
    """
    创建HTTP客户端的工厂函数
    
    Args:
        service_config: 服务配置字典
        service_name: 服务名称
        
    Returns:
        配置好的httpx.AsyncClient实例
    """
    # 根据不同服务设置不同的超时时间
    timeout_configs = {
        'llm': httpx.Timeout(connect=10.0, read=600.0, write=60.0, pool=10.0),  # LLM服务需要更长的读取时间
        'ocr': httpx.Timeout(connect=10.0, read=300.0, write=60.0, pool=5.0),   # OCR处理时间较长
        'stt': httpx.Timeout(connect=10.0, read=300.0, write=60.0, pool=5.0),   # STT处理时间较长
        'tts': httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=5.0),   # TTS处理时间中等
        'user': httpx.Timeout(connect=5.0, read=30.0, write=15.0, pool=5.0),    # 用户服务响应较快
        'md2pdf': httpx.Timeout(connect=10.0, read=180.0, write=30.0, pool=5.0) # PDF转换需要时间
    }
    
    # 根据不同服务设置不同的连接限制
    limit_configs = {
        'llm': httpx.Limits(max_connections=50, max_keepalive_connections=10),   # LLM可能需要较多连接
        'ocr': httpx.Limits(max_connections=30, max_keepalive_connections=5),    # OCR并发适中
        'stt': httpx.Limits(max_connections=30, max_keepalive_connections=5),    # STT并发适中
        'tts': httpx.Limits(max_connections=30, max_keepalive_connections=5),    # TTS并发适中
        'user': httpx.Limits(max_connections=100, max_keepalive_connections=20), # 用户服务高并发
        'md2pdf': httpx.Limits(max_connections=20, max_keepalive_connections=5)  # PDF转换并发较低
    }
    
    # 获取服务特定的配置，如果没有则使用默认值
    timeout = timeout_configs.get(service_name, httpx.Timeout(300.0, connect=10.0))
    limits = limit_configs.get(service_name, httpx.Limits(max_connections=100, max_keepalive_connections=20))
    
    return httpx.AsyncClient(
        base_url=service_config["base_url"],
        timeout=timeout,
        limits=limits,
        follow_redirects=False,
        verify=False  # 在内网环境中可以禁用SSL验证
    )
    
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("API 网关正在关闭，清理 HTTP 客户端... (API Gateway shutting down, cleaning up HTTP clients...)")
    for service_name, client in http_clients.items():
        try:
            await client.aclose()
            logger.info(f"已关闭服务 '{service_name}' 的客户端。(Closed client for service '{service_name}'.)")
        except Exception as e:
            logger.error(f"关闭服务 '{service_name}' 客户端时发生错误: {e} (Error closing client for service '{service_name}': {e})")
    logger.info("所有 HTTP 客户端已清理。(All HTTP clients cleaned up.)")

# --- OAuth2 token 端点 (包含登录IP锁定逻辑) ---
if AUTH_CONFIGURED:
    @app.post("/token", response_model=Token, tags=["Authentication"]) # type: ignore
    async def login_for_access_token(request: Request, form_data: OAuth2PasswordRequestForm = Depends()): # type: ignore
        # FastAPI 的 Request 类型在这里是合适的
        client_ip = get_client_ip(request) # get_client_ip 现在期望 StarletteRequest
        
        if ENABLE_LOGIN_IP_LOCKOUT and client_ip:
            # 检查IP是否已被锁定
            if client_ip in locked_out_ips:
                lock_expiry_time = locked_out_ips[client_ip]
                if datetime.now() < lock_expiry_time:
                    logger.warning(f"IP {client_ip} 因登录失败次数过多而被锁定，直到 {lock_expiry_time}。(IP {client_ip} is locked due to too many failed login attempts until {lock_expiry_time}.)")
                    raise HTTPException(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS, 
                        detail=f"您的IP因尝试次数过多已被暂时锁定。请在 {LOGIN_LOCKOUT_DURATION_SECONDS // 60} 分钟后重试。(Your IP has been temporarily locked due to too many attempts. Please try again in {LOGIN_LOCKOUT_DURATION_SECONDS // 60} minutes.)"
                    )
                else:
                    del locked_out_ips[client_ip]
                    if client_ip in login_failure_tracker: 
                        del login_failure_tracker[client_ip]
                    logger.info(f"IP {client_ip} 的登录锁定已过期并解除。(Login lockout for IP {client_ip} has expired and been lifted.)")
        
        logger.debug(f"用户 {form_data.username} 尝试登录，IP: {client_ip}。(Login attempt for user: {form_data.username}, IP: {client_ip})")
        
        login_successful = False
        if form_data.username == FIXED_USERNAME and verify_password(form_data.password, FIXED_PASSWORD_HASH): # type: ignore
            login_successful = True
        
        if not login_successful:
            logger.warning(f"用户 {form_data.username} 认证失败，IP: {client_ip}。(Authentication failed for user: {form_data.username}, IP: {client_ip})")
            if ENABLE_LOGIN_IP_LOCKOUT and client_ip:
                current_time = datetime.now()
                if client_ip not in login_failure_tracker or \
                   (current_time - login_failure_tracker[client_ip]["window_start"]).total_seconds() > LOGIN_LOCKOUT_WINDOW_SECONDS:
                    login_failure_tracker[client_ip] = {"count": 1, "window_start": current_time}
                    logger.info(f"IP {client_ip} 登录失败次数: 1。(Login failure count for IP {client_ip}: 1.)")
                else:
                    login_failure_tracker[client_ip]["count"] += 1
                    logger.info(f"IP {client_ip} 登录失败次数: {login_failure_tracker[client_ip]['count']} / {LOGIN_ATTEMPTS_LIMIT} (在 {LOGIN_LOCKOUT_WINDOW_SECONDS} 秒内)。(Login failure count for IP {client_ip}: {login_failure_tracker[client_ip]['count']} / {LOGIN_ATTEMPTS_LIMIT} (within {LOGIN_LOCKOUT_WINDOW_SECONDS}s).)")
                    if login_failure_tracker[client_ip]["count"] >= LOGIN_ATTEMPTS_LIMIT:
                        lock_until = current_time + timedelta(seconds=LOGIN_LOCKOUT_DURATION_SECONDS)
                        locked_out_ips[client_ip] = lock_until
                        del login_failure_tracker[client_ip] 
                        logger.warning(f"IP {client_ip} 因登录失败次数达到上限而被锁定，直到 {lock_until}。(IP {client_ip} locked due to reaching login attempt limit until {lock_until}.)")
                        raise HTTPException(
                            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                            detail=f"您的IP因尝试次数过多已被暂时锁定。请在 {LOGIN_LOCKOUT_DURATION_SECONDS // 60} 分钟后重试。(Your IP has been temporarily locked due to too many attempts. Please try again in {LOGIN_LOCKOUT_DURATION_SECONDS // 60} minutes.)"
                        )
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="用户名或密码错误 (Incorrect username or password)", headers={"WWW-Authenticate": "Bearer"})
        
        if ENABLE_LOGIN_IP_LOCKOUT and client_ip and client_ip in login_failure_tracker:
            del login_failure_tracker[client_ip]
            logger.info(f"IP {client_ip} 登录成功，清除了之前的失败尝试记录。(Login successful for IP {client_ip}, cleared previous failed attempt records.)")

        user_obj = get_fixed_api_user(FIXED_USERNAME) # type: ignore
        user_name_for_token = FIXED_USERNAME 
        if user_obj and hasattr(user_obj, 'username') and user_obj.username: # type: ignore
             user_name_for_token = user_obj.username # type: ignore
        
        if not user_obj or (hasattr(user_obj, 'disabled') and user_obj.disabled): # type: ignore
            logger.warning(f"用户 {FIXED_USERNAME} 访问被禁用或用户不存在。(Access disabled for user: {FIXED_USERNAME} or user not found.)")
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="用户访问被禁用或用户不存在 (User access disabled or user not found)")
        
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(data={"sub": user_name_for_token}, expires_delta=access_token_expires)
        logger.info(f"用户 '{form_data.username}' 登录成功，已颁发令牌，IP: {client_ip}。(User '{form_data.username}' logged in successfully. Token issued. IP: {client_ip})")
        return {"access_token": access_token, "token_type": "bearer"}
else:
    @app.post("/token", tags=["Authentication"])
    async def login_for_access_token_disabled():
        logger.error("登录尝试失败：认证模块未配置。(Login attempt failed: Authentication module not configured.)")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="由于配置错误，认证服务不可用。(Authentication service is not available due to misconfiguration.)")

# --- 辅助函数：通用请求转发 ---
async def forward_request(
    service_name: str, request: Request, target_path: str, # FastAPI Request
    request_data: Optional[Union[Dict, bytes]] = None,
    files: Optional[Dict[str, Any]] = None
):
    """通用请求转发函数，用于非流式传输的服务"""
    client = http_clients.get(service_name)
    if not client:
        logger.error(f"服务 '{service_name}' 的客户端未初始化。(Service client for '{service_name}' not initialized.)")
        raise HTTPException(status_code=500, detail=f"服务 '{service_name}' 的客户端未初始化")

    headers = dict(request.headers)
    common_headers_to_pop = ["host", "content-length", "content-type", "authorization"]
    for h in common_headers_to_pop:
        headers.pop(h, None)

    # Add API key for user service
    if service_name == "user":
        headers["X-API-Key"] = "xkbai"

    # FastAPI Request (request) is compatible with StarletteRequest for get_client_ip
    client_host = get_client_ip(request) 
    if client_host:
            headers["X-Forwarded-For"] = client_host
    if request.url: # Ensure request.url is not None
        headers["X-Forwarded-Proto"] = request.url.scheme

    params = dict(request.query_params)
    method = request.method
    full_target_url = f"{str(client.base_url).rstrip('/')}{target_path}"
    logger.info(f"转发请求 (Forwarding request): {method} {request.url.path if request.url else ''} -> {full_target_url}")
    logger.debug(f"转发请求头 (Forwarding with headers): {headers}, 参数 (params): {params}")

    if isinstance(request_data, bytes):
        logger.debug(f"转发JSON/bytes数据 (前100字节) (Forwarding with JSON/bytes data (first 100 bytes)): {request_data[:100]}")
    elif isinstance(request_data, dict):
         logger.debug(f"转发表单数据 (Forwarding with form data): {request_data}")

    try:
        if method == "GET":
            response = await client.get(target_path, headers=headers, params=params)
        elif method == "POST":
            if files:
                response = await client.post(target_path, headers=headers, params=params, data=request_data, files=files)
            elif isinstance(request_data, bytes):
                headers["Content-Type"] = "application/json"
                response = await client.post(target_path, headers=headers, params=params, content=request_data)
            elif isinstance(request_data, dict):
                response = await client.post(target_path, headers=headers, params=params, data=request_data)
            else:
                response = await client.post(target_path, headers=headers, params=params)
        elif method == "PUT":
            if isinstance(request_data, bytes):
                headers["Content-Type"] = "application/json"
                response = await client.put(target_path, headers=headers, params=params, content=request_data)
            elif isinstance(request_data, dict):
                response = await client.put(target_path, headers=headers, params=params, data=request_data)
            else:
                response = await client.put(target_path, headers=headers, params=params)
        elif method == "DELETE":
            if isinstance(request_data, bytes):
                headers["Content-Type"] = "application/json"
                # httpx.delete() doesn't accept data parameter, use generic request method
                response = await client.request("DELETE", target_path, headers=headers, params=params, content=request_data)
            else:
                response = await client.delete(target_path, headers=headers, params=params)
        else:
            response = await client.request(method, target_path, headers=headers, params=params)
        
        response.raise_for_status()
        
        # 根据响应内容类型返回适当的响应
        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            return response.json()
        elif content_type.startswith("text/"):
            return Response(content=response.text, media_type=content_type, status_code=response.status_code)
        else:
            # 对于二进制文件或其他类型，返回流式响应
            return StreamingResponse(
                io.BytesIO(response.content),
                media_type=content_type,
                status_code=response.status_code,
                headers=dict(response.headers)
            )
            
    except httpx.ConnectError as e:
        logger.error(f"连接后端服务 {service_name} ({full_target_url}) 失败: {e} (Connection to backend service {service_name} ({full_target_url}) failed: {e})")
        raise HTTPException(status_code=503, detail=f"无法连接到后端服务 '{service_name}'")
    except httpx.TimeoutException as e:
        logger.error(f"请求后端服务 {service_name} ({full_target_url}) 超时: {e} (Request to backend service {service_name} ({full_target_url}) timed out: {e})")
        raise HTTPException(status_code=504, detail=f"后端服务 '{service_name}' 请求超时")
    except httpx.HTTPStatusError as e:
        error_detail = e.response.text[:500] if e.response else str(e)
        logger.error(f"后端服务 {service_name} ({full_target_url}) 返回 HTTP {e.response.status_code if e.response else 'N/A'}: {error_detail} (Backend service {service_name} ({full_target_url}) returned HTTP {e.response.status_code if e.response else 'N/A'}: {error_detail})")
        raise HTTPException(status_code=e.response.status_code if e.response else 500, detail=f"来自 '{service_name}' 的后端错误: {error_detail}")
    except httpx.RequestError as e:
        logger.error(f"请求后端服务 {service_name} ({full_target_url}) 时发生错误: {e} (Error requesting backend service {service_name} ({full_target_url}): {e})")
        raise HTTPException(status_code=502, detail=f"代理请求到后端服务 '{service_name}' 时出错")
    except Exception as e:
        logger.exception(f"处理到 {service_name} ({full_target_url}) 的代理请求时发生意外错误: {e} (Unexpected error during proxy request to {service_name} ({full_target_url}): {e})")
        raise HTTPException(status_code=500, detail="处理请求时发生内部服务器错误")

# --- 辅助函数：LLM流式请求转发 ---
async def forward_llm_streaming_request(
    service_name: str, request: Request, target_path: str,
    request_data: Optional[Union[Dict, bytes]] = None
):
    """专门用于LLM服务的流式请求转发，能够智能处理流式和非流式响应"""
    client = http_clients.get(service_name)
    if not client:
        logger.error(f"服务 '{service_name}' 的客户端未初始化。(Service client for '{service_name}' not initialized.)")
        raise HTTPException(status_code=500, detail=f"服务 '{service_name}' 的客户端未初始化")

    headers = dict(request.headers)
    common_headers_to_pop = ["host", "content-length", "content-type", "authorization"]
    for h in common_headers_to_pop:
        headers.pop(h, None)

    # Add API key for user service
    if service_name == "user":
        headers["X-API-Key"] = "xkbai"

    client_host = get_client_ip(request) 
    if client_host:
        headers["X-Forwarded-For"] = client_host
    if request.url:
        headers["X-Forwarded-Proto"] = request.url.scheme

    params = dict(request.query_params)
    method = request.method
    full_target_url = f"{str(client.base_url).rstrip('/')}{target_path}"
    logger.info(f"LLM流式转发请求 (LLM Streaming forwarding request): {method} {request.url.path if request.url else ''} -> {full_target_url}")
    
    try:
        import requests
        import asyncio
        import json
        
        full_url = f"{str(client.base_url).rstrip('/')}{target_path}"
        
        def make_request():
            resp = requests.request(
                method=method,
                url=full_url,
                headers=headers,
                params=params,
                data=request_data if isinstance(request_data, (dict, bytes)) else None,
                stream=True,
                timeout=600
            )
            return resp
        
        # 在线程池中执行同步请求
        sync_response = await asyncio.to_thread(make_request)
        
        # 检查响应头来判断是否为流式响应
        content_type = sync_response.headers.get('content-type', '')
        transfer_encoding = sync_response.headers.get('transfer-encoding', '')
        
        # 判断是否为流式响应的几种情况：
        # 1. Content-Type 包含 text/event-stream
        # 2. Transfer-Encoding 为 chunked
        # 3. 响应头中没有 Content-Length（通常表示流式）
        is_streaming = (
            'text/event-stream' in content_type or
            'text/plain' in content_type or
            transfer_encoding == 'chunked' or
            'content-length' not in sync_response.headers
        )
        
        if is_streaming:
            logger.info(f"检测到流式响应，使用流式转发 (Detected streaming response, using streaming forward)")
            
            async def stream_generator():
                try:
                    for chunk in sync_response.iter_content(chunk_size=1024):
                        if chunk:
                            yield chunk
                except Exception as e:
                    logger.error(f"流式传输过程中发生错误: {e}")
                    yield f"data: {{\"error\": \"Stream interrupted: {str(e)}\"}}".encode('utf-8')
                finally:
                    # 确保连接关闭
                    sync_response.close()
            
            return StreamingResponse(
                stream_generator(),
                status_code=sync_response.status_code,
                headers={
                    key: value for key, value in sync_response.headers.items()
                    if key.lower() not in ['content-length', 'transfer-encoding']
                }
            )
        else:
            logger.info(f"检测到非流式响应，使用普通转发 (Detected non-streaming response, using normal forward)")
            
            # 非流式响应，读取全部内容后返回
            content = sync_response.content
            sync_response.close()
            
            # 尝试解析为JSON
            try:
                if content_type and 'application/json' in content_type:
                    return json.loads(content.decode('utf-8'))
                else:
                    return Response(
                        content=content,
                        status_code=sync_response.status_code,
                        headers=dict(sync_response.headers)
                    )
            except json.JSONDecodeError:
                return Response(
                    content=content,
                    status_code=sync_response.status_code,
                    headers=dict(sync_response.headers)
                )
                
    except Exception as e:
        logger.exception(f"LLM流式请求处理发生错误: {e} (Error in LLM streaming request: {e})")
        raise HTTPException(status_code=500, detail=f"LLM服务请求处理错误: {str(e)}")

# --- 认证依赖 ---
auth_dependency = [Depends(get_current_api_user)] if AUTH_CONFIGURED else []


# LLM Routes
llm_router = APIRouter(prefix="/llm", tags=["LLM Service"], dependencies=auth_dependency)
@llm_router.get("/images/{image_path:path}", summary="代理LLM服务中的WebDAV图片")
async def llm_get_image_proxy(request: Request, image_path: str = Path(...)):
    logger.info(f"LLM 图片代理请求路径: {image_path} (LLM Image Proxy request for path: {image_path})")
    return await forward_request("llm", request, target_path=f"/get_images/{image_path.lstrip('/')}")
@llm_router.post("/chat/completions", summary="与LLM进行流式聊天补全")
async def llm_chat_completions(payload: LLMChatCompletionsRequest, request: Request):
    logger.info(f"LLM 聊天补全请求类型: {payload.chat_type} (LLM Chat Completions request of type: {payload.chat_type})")
    return await forward_llm_streaming_request("llm", request, target_path="/chat/completions", request_data=payload.model_dump_json().encode('utf-8'))
@llm_router.post("/get_chat_title", response_model=LLMChatTitleResponse, summary="获取LLM生成的聊天标题")
async def llm_get_chat_title(payload: LLMChatHistory, request: Request):
    logger.info("LLM 获取聊天标题请求 (LLM Get Chat Title request)")
    return await forward_request("llm", request, target_path="/get_chat_title/", request_data=payload.model_dump_json().encode('utf-8'))





# --- STT, TTS, OCR, User, LLM 服务路由 (保持不变) ---
# STT Routes
stt_router = APIRouter(prefix="/stt", tags=["STT Service"], dependencies=auth_dependency)
@stt_router.post("/transcribe/", response_model=TranscriptionResponse)
async def stt_transcribe(request: Request, file: UploadFile = File(...), language: Optional[str] = Form("zh"), beam_size: int = Form(5), vad_filter: bool = Form(True), min_silence_duration_ms: Optional[int] = Form(1000)):
    form_data = {'language': language, 'beam_size': str(beam_size), 'vad_filter': str(vad_filter)}
    if min_silence_duration_ms is not None: form_data['min_silence_duration_ms'] = str(min_silence_duration_ms)
    files_data = {'file': (file.filename, await file.read(), file.content_type)}
    backend_target_path = "/stt/transcribe/"
    client = http_clients.get("stt")
    if not client: raise HTTPException(status_code=500, detail="STT 客户端未初始化")
    fwd_headers = dict(request.headers); [fwd_headers.pop(h,None) for h in ["host","content-length","content-type","x-api-key","authorization"]]
    try:
        backend_response = await client.post(backend_target_path, data=form_data, files=files_data, headers=fwd_headers, timeout=300.0)
        backend_response.raise_for_status()
        return backend_response.json()
    except httpx.HTTPStatusError as e: 
        error_detail = e.response.json().get("detail", e.response.text) if e.response and e.response.content else str(e)
        status_code = e.response.status_code if e.response else 500
        logger.error(f"STT 后端错误 {status_code} at {str(client.base_url).rstrip('/')}{backend_target_path}: {error_detail}")
        raise HTTPException(status_code=status_code, detail=error_detail)
    except Exception as e: 
        logger.exception(f"STT 请求发生意外错误 for {str(client.base_url).rstrip('/')}{backend_target_path}: {e}")
        raise HTTPException(status_code=500, detail="STT 请求内部错误")
@stt_router.get("/health")
async def stt_health(request: Request): return await forward_request("stt", request, "/stt/health") 
@stt_router.get("/")
async def stt_root(request: Request): return await forward_request("stt", request, "/stt/")
@stt_router.post("/clear-memory")
async def stt_clear_memory(request: Request): return await forward_request("stt", request, "/stt/clear-memory")

# TTS Routes
tts_router = APIRouter(prefix="/tts", tags=["TTS Service"], dependencies=auth_dependency)
@tts_router.post("/synthesize")
async def tts_synthesize(request: Request, tts_input: TTSRequest = Body(...)): return await forward_request("tts", request, "/tts/synthesize", request_data=tts_input.model_dump_json().encode('utf-8'))
@tts_router.get("/status")
async def tts_status(request: Request): return await forward_request("tts", request, "/tts/status")

# OCR Routes
ocr_router = APIRouter(prefix="/olmocr", tags=["OCR Service"], dependencies=auth_dependency)
# @ocr_router.post("/process_sync")
# async def ocr_process_sync(request: Request, file: UploadFile = File(...)):
#     files_data = {'file': (file.filename, await file.read(), file.content_type)}
#     backend_target_path = "/olmocr/process_sync"
#     client = http_clients.get("ocr"); 
#     if not client: raise HTTPException(status_code=500, detail="OCR 客户端未初始化")
#     fwd_headers = dict(request.headers); [fwd_headers.pop(h,None) for h in ["host","content-length","content-type","x-api-key","authorization"]]
#     try:
#         backend_response = await client.post(backend_target_path, files=files_data, headers=fwd_headers, timeout=300.0)
#         backend_response.raise_for_status(); return backend_response.json()
#     except httpx.HTTPStatusError as e: 
#         error_detail = e.response.json().get("detail", e.response.text) if e.response and e.response.content else str(e)
#         status_code = e.response.status_code if e.response else 500
#         logger.error(f"OCR 同步后端错误 {status_code} at {str(client.base_url).rstrip('/')}{backend_target_path}: {error_detail}")
#         raise HTTPException(status_code=status_code, detail=error_detail)
#     except Exception as e: 
#         logger.exception(f"OCR 同步请求错误 for {str(client.base_url).rstrip('/')}{backend_target_path}: {e}")
#         raise HTTPException(status_code=500, detail=f"OCR 同步处理内部错误: {type(e).__name__}")

@ocr_router.post("/process", response_model=Job)
async def ocr_process(request: Request, file: UploadFile = File(...)):
    files_data = {'file': (file.filename, await file.read(), file.content_type)}
    backend_target_path = "/olmocr/process"
    client = http_clients.get("ocr"); 
    if not client: raise HTTPException(status_code=500, detail="OCR 客户端未初始化")
    fwd_headers = dict(request.headers); [fwd_headers.pop(h,None) for h in ["host","content-length","content-type","x-api-key","authorization"]]
    try:
        backend_response = await client.post(backend_target_path, files=files_data, headers=fwd_headers, timeout=300.0)
        backend_response.raise_for_status(); return backend_response.json()
    except httpx.HTTPStatusError as e: 
        error_detail = e.response.json().get("detail", e.response.text) if e.response and e.response.content else str(e)
        status_code = e.response.status_code if e.response else 500
        logger.error(f"OCR 异步后端错误 {status_code} at {str(client.base_url).rstrip('/')}{backend_target_path}: {error_detail}")
        raise HTTPException(status_code=status_code, detail=error_detail)
    except Exception as e: 
        logger.exception(f"OCR 异步请求错误 for {str(client.base_url).rstrip('/')}{backend_target_path}: {e}")
        raise HTTPException(status_code=500, detail=f"OCR 异步处理内部错误: {type(e).__name__}")

# @ocr_router.get("/jobs/{job_id}", response_model=Job)
# async def ocr_get_job(request: Request, job_id: str = Path(...)): return await forward_request("ocr", request, f"/olmocr/jobs/{job_id}")

# @ocr_router.get("/jobs", response_model=List[Job])
# async def ocr_list_jobs(request: Request, status: Optional[JobStatus] = Query(None)): path = "/olmocr/jobs"; path += f"?status={status.value}" if status else ""; return await forward_request("ocr", request, path)

@ocr_router.get("/results/{job_id}")
async def ocr_get_results(request: Request, job_id: str = Path(...)): return await forward_request("ocr", request, f"/olmocr/results/{job_id}")

# @ocr_router.get("/queue")
# async def ocr_get_queue(request: Request): return await forward_request("ocr", request, "/olmocr/queue")

# @ocr_router.get("/")
# async def ocr_root(request: Request): return await forward_request("ocr", request, "/olmocr/")

# User Routes
user_router = APIRouter(prefix="/user", tags=["User Service"], dependencies=auth_dependency)
@user_router.get("/get_all_users/", response_model=UserListResponse)
async def get_all_users(request: Request): return await forward_request("user", request, "/get_all_users/")
@user_router.post("/add_user/", response_model=SuccessResponse)
async def add_user(request: Request, user_payload: UserBase): return await forward_request("user", request, "/add_user/", request_data=user_payload.model_dump_json().encode('utf-8'))
@user_router.delete("/del_user/", response_model=SuccessResponse)
async def del_user(request: Request, user_payload: UserBase): return await forward_request("user", request, "/del_user/", request_data=user_payload.model_dump_json().encode('utf-8'))
@user_router.get("/get_user_chat/", response_model=List[ChatSessionInfo])
async def get_user_chat(request: Request, user_id: str): return await forward_request("user", request, f"/get_user_chat/?user_id={user_id}")
@user_router.post("/add_chat/", response_model=ChatSessionIdResponse)
async def add_chat(request: Request, user_id: str, chat_content: ChatContentRequest):
    client = http_clients.get("user")
    if not client: raise HTTPException(status_code=500, detail="User 客户端未初始化")
    backend_target_path = f"/add_chat/?user_id={user_id}"
    fwd_headers = dict(request.headers); [fwd_headers.pop(h,None) for h in ["host","content-length","content-type","x-api-key","authorization"]]
    fwd_headers["Content-Type"] = "application/json" 
    try:
        backend_response = await client.post(backend_target_path, json=chat_content.model_dump(), headers=fwd_headers, timeout=300.0)
        backend_response.raise_for_status(); return backend_response.json()
    except httpx.HTTPStatusError as e: 
        error_detail = e.response.json().get("detail", e.response.text) if e.response and e.response.content else str(e)
        status_code = e.response.status_code if e.response else 500
        logger.error(f"User 服务 (add_chat) 错误 {status_code} at {str(client.base_url).rstrip('/')}{backend_target_path}: {error_detail}")
        raise HTTPException(status_code=status_code, detail=error_detail)
    except Exception as e: 
        logger.exception(f"add_chat 代理错误 for {str(client.base_url).rstrip('/')}{backend_target_path}: {e}")
        raise HTTPException(status_code=500, detail=f"add_chat 内部错误: {type(e).__name__}")
@user_router.delete("/del_chat/", response_model=SuccessResponse)
async def del_chat(request: Request, chat_session: ChatSessionIdRequest): return await forward_request("user", request, "/del_chat/", request_data=chat_session.model_dump_json().encode('utf-8'))
@user_router.put("/edit_chat/", response_model=SuccessResponse)
async def edit_chat(request: Request, chat_session: ChatSessionUpdateRequest): return await forward_request("user", request, "/edit_chat/", request_data=chat_session.model_dump_json().encode('utf-8'))
@user_router.put("/update_chat/", response_model=SuccessResponse)
async def update_chat(request: Request, chat_session: ChatSessionUpdateRequest): return await forward_request("user", request, "/update_chat/", request_data=chat_session.model_dump_json().encode('utf-8'))
@user_router.get("/get_chat_by_session_id/", response_model=ChatSessionResponse)
async def get_chat_by_session_id(request: Request, chat_session_id: str): return await forward_request("user", request, f"/get_chat_by_session_id/?chat_session_id={chat_session_id}")
@user_router.put("/update_chat_title/", response_model=SuccessChangeTitleResponse)
async def update_chat_title(request: Request, chat_session: ChatSessionResponse): return await forward_request("user", request, "/update_chat_title/", request_data=chat_session.model_dump_json().encode('utf-8'))
@user_router.put("/edit_chat_title/", response_model=SuccessChangeTitleResponse)
async def edit_chat_title(request: Request, chat_session: ChatSessionInfo): return await forward_request("user", request, "/edit_chat_title/", request_data=chat_session.model_dump_json().encode('utf-8'))



md2pdf_router = APIRouter(prefix="/md2pdf", tags=["MD2PDF Service"], dependencies=auth_dependency)

# @md2pdf_router.post("/api/convert/md-to-pdf", response_model=MD2PDFConvertResponse, summary="转换Markdown文本为PDF")
# async def md2pdf_convert_text(
#     request: Request,
#     markdown_content: str = Form(..., description="Markdown内容"),
#     skip_images: bool = Form(False, description="是否跳过图片")
# ):
#     """将Markdown文本内容转换为PDF文件"""
#     form_data = {
#         'markdown_content': markdown_content,
#         'skip_images': str(skip_images).lower()
#     }
    
#     return await forward_request("md2pdf", request, "/api/convert/md-to-pdf", request_data=form_data)

# @md2pdf_router.post("/api/convert/md-file-to-pdf", response_model=MD2PDFConvertResponse, summary="转换Markdown文件为PDF")
# async def md2pdf_convert_file(
#     request: Request,
#     file: UploadFile = File(..., description="Markdown文件 (.md, .markdown)"),
#     skip_images: bool = Form(False, description="是否跳过图片")
# ):
#     """将上传的Markdown文件转换为PDF文件"""
#     if not file.filename or not file.filename.lower().endswith(('.md', '.markdown')):
#         raise HTTPException(status_code=400, detail="仅支持Markdown文件(.md, .markdown)")
    
#     files_data = {'file': (file.filename, await file.read(), file.content_type or 'text/markdown')}
#     form_data = {'skip_images': str(skip_images).lower()}
    
#     backend_target_path = "/api/convert/md-file-to-pdf"
#     client = http_clients.get("md2pdf")
#     if not client:
#         raise HTTPException(status_code=500, detail="MD2PDF 客户端未初始化")
    
#     fwd_headers = dict(request.headers)
#     [fwd_headers.pop(h, None) for h in ["host", "content-length", "content-type", "x-api-key", "authorization"]]
    
#     try:
#         backend_response = await client.post(
#             backend_target_path, 
#             data=form_data, 
#             files=files_data, 
#             headers=fwd_headers, 
#             timeout=300.0
#         )
#         backend_response.raise_for_status()
#         return backend_response.json()
#     except httpx.HTTPStatusError as e:
#         error_detail = e.response.json().get("detail", e.response.text) if e.response and e.response.content else str(e)
#         status_code = e.response.status_code if e.response else 500
#         logger.error(f"MD2PDF 后端错误 {status_code} at {str(client.base_url).rstrip('/')}{backend_target_path}: {error_detail}")
#         raise HTTPException(status_code=status_code, detail=error_detail)
#     except Exception as e:
#         logger.exception(f"MD2PDF 请求发生意外错误 for {str(client.base_url).rstrip('/')}{backend_target_path}: {e}")
#         raise HTTPException(status_code=500, detail="MD2PDF 请求内部错误")

@md2pdf_router.get("/api/download/{file_id}", summary="下载生成的PDF文件")
async def md2pdf_download(
    request: Request,
    file_id: str = Path(..., description="PDF文件ID")
):
    """下载之前生成的PDF文件"""
    if not file_id.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="无效的文件ID")
    
    # 对于文件下载，我们需要使用 forward_request 进行流式传输
    return await forward_request("md2pdf", request, f"/api/download/{file_id}")

# @md2pdf_router.get("/api/files/list", response_model=MD2PDFFileListResponse, summary="列出可用的PDF文件")
# async def md2pdf_list_files(request: Request):
#     """列出当前可以下载的所有PDF文件"""
#     return await forward_request("md2pdf", request, "/api/files/list")

# @md2pdf_router.get("/", summary="MD2PDF服务根路径")
# async def md2pdf_root(request: Request):
#     """获取MD2PDF服务的基本信息"""
#     return await forward_request("md2pdf", request, "/")



# --- 网关自身路由 (不需要认证) ---
@app.get("/", tags=["Gateway Info"])
async def get_root():
    return {
        "message": "欢迎使用统一 AI 服务网关 (OAuth2 认证与IP限制版)",
        "version": app.version, "docs_url": "/docs", "redoc_url": "/redoc",
        "registered_services": list(BACKEND_SERVICES.keys()),
        "authentication_status": "已配置 (configured)" if AUTH_CONFIGURED else "未配置 - 请检查日志 (NOT CONFIGURED - CHECK LOGS)",
        "login_ip_lockout_enabled": ENABLE_LOGIN_IP_LOCKOUT,
        "concurrent_request_limit_enabled": ENABLE_CONCURRENT_REQUEST_LIMIT
    }

@app.get("/health", tags=["Gateway Info"])
async def health_check():
    service_status = {}
    gateway_healthy = True
    auth_module_status = "已加载 (loaded)" if AUTH_CONFIGURED else "加载失败 (FAILED_TO_LOAD)"

    for service_name, client_instance in http_clients.items():
        if service_name not in BACKEND_SERVICES:
            logger.warning(f"服务 '{service_name}' 在 http_clients 中但不在 BACKEND_SERVICES 配置中。(Service '{service_name}' found in http_clients but not in BACKEND_SERVICES config.)")
            continue
        service_config = BACKEND_SERVICES[service_name]
        check_url = service_config.get('actual_health_endpoint', "/")
        check_method = service_config.get('health_check_method', "HEAD").upper()
        target_display_url = str(client_instance.base_url).rstrip('/') + check_url
        try:
            response = None
            if check_method == "GET": response = await client_instance.get(check_url, timeout=5.0)
            elif check_method == "POST": response = await client_instance.post(check_url, timeout=5.0)
            else: response = await client_instance.head(check_url, timeout=5.0)
            
            if 200 <= response.status_code < 300:
                 service_status[service_name] = {"status": "可达 (reachable)", "target": target_display_url, "method": check_method, "code": response.status_code}
            else:
                 error_text = response.text[:200] if response.content else "错误响应中无内容"
                 service_status[service_name] = {"status": "不可达 (unreachable)", "target": target_display_url, "method": check_method, "code": response.status_code, "detail": error_text}
                 gateway_healthy = False
        except httpx.ConnectError as e:
            logger.warning(f"服务 '{service_name}' 健康检查 ({check_method} {target_display_url}) 失败: 连接错误 - {e} (Health check for '{service_name}' ({check_method} {target_display_url}) failed: ConnectError - {e})")
            service_status[service_name] = {"status": "不可达 (unreachable)", "target": target_display_url, "method": check_method, "error": "ConnectError", "detail": str(e)}
            gateway_healthy = False
        except (httpx.RequestError, asyncio.TimeoutError) as e:
            logger.warning(f"服务 '{service_name}' 健康检查 ({check_method} {target_display_url}) 失败: {type(e).__name__} - {e} (Health check for '{service_name}' ({check_method} {target_display_url}) failed: {type(e).__name__} - {e})")
            service_status[service_name] = {"status": "不可达 (unreachable)", "target": target_display_url, "method": check_method, "error": type(e).__name__, "detail": str(e)}
            gateway_healthy = False
        except Exception as e:
            logger.error(f"服务 '{service_name}' 健康检查 ({check_method} {target_display_url}) 时发生意外错误: {e} (Unexpected error during health check for '{service_name}' ({check_method} {target_display_url}): {e})", exc_info=True)
            service_status[service_name] = {"status": "检查错误 (check_error)", "target": target_display_url, "method": check_method, "error": type(e).__name__, "detail": str(e)}
            gateway_healthy = False
            
    # 计算核心服务健康状态 - 只要核心服务可达就认为网关健康
    core_services = ["stt", "tts", "user", "md2pdf"]  # 核心必需服务
    core_services_healthy = all(
        service_status.get(service, {}).get("status") == "可达 (reachable)" 
        for service in core_services if service in service_status
    )
    
    # 网关健康状态：只要核心服务健康且认证模块加载就认为健康
    gateway_status = "健康 (healthy)" if core_services_healthy and AUTH_CONFIGURED else "降级 (degraded)"
    
    return {
        "gateway_status": gateway_status,
        "authentication_module": auth_module_status,
        "core_services_status": {k: v for k, v in service_status.items() if k in core_services},
        "optional_services_status": {k: v for k, v in service_status.items() if k not in core_services},
        "backend_service_reachability": service_status
    }






# --- 包含所有路由 ---
app.include_router(stt_router)
app.include_router(tts_router)
app.include_router(ocr_router)
app.include_router(user_router)
app.include_router(llm_router)
app.include_router(md2pdf_router)

# --- Uvicorn 启动配置 ---
ssl_root_dir = os.getenv("SSL_ROOT_DIR", "/home/xkb2/ACME.sh/https/acme.sh/cert/")
ssl_certfile_name = os.getenv("SSL_CERT_NAME", "fullchain.cer")
ssl_keyfile_name = os.getenv("SSL_KEY_NAME", "*.744204541.xyz.key")
ssl_certfile = os.path.join(ssl_root_dir, ssl_certfile_name)
ssl_keyfile = os.path.join(ssl_root_dir, ssl_keyfile_name)

if __name__ == "__main__":
    gateway_port = int(os.getenv("GATEWAY_PORT", 60443))
    reload_mode = os.getenv("GATEWAY_RELOAD", "true").lower() == "true"
    log_level = os.getenv("LOG_LEVEL", "info").lower()
    
    logging.basicConfig(level=log_level.upper(), format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info(f"启动 API 网关 (OAuth2与IP限制版)，监听端口: {gateway_port}, Reload模式: {reload_mode}, 日志级别: {log_level.upper()} (Starting API Gateway (OAuth2 with IP Limiting Version) on port: {gateway_port}, Reload: {reload_mode}, Log Level: {log_level.upper()})")
    
    if not AUTH_CONFIGURED:
        logger.critical("严重警告: 认证系统未配置。如果依赖项不是有条件的，大多数端点将失败或不受保护。(CRITICAL: AUTHENTICATION SYSTEM IS NOT CONFIGURED. MOST ENDPOINTS WILL FAIL OR BE UNPROTECTED IF DEPENDENCIES ARE NOT CONDITIONAL.)")
    
    use_ssl = True
    if not os.path.exists(ssl_keyfile):
        logger.warning(f"SSL 密钥文件未找到: {ssl_keyfile}。HTTPS 将被禁用。(SSL key file not found: {ssl_keyfile}. HTTPS will be disabled.)")
        use_ssl = False
    if not os.path.exists(ssl_certfile):
        logger.warning(f"SSL 证书文件未找到: {ssl_certfile}。HTTPS 将被禁用。(SSL cert file not found: {ssl_certfile}. HTTPS will be disabled.)")
        use_ssl = False

    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    uvicorn_config = {
        "host": "0.0.0.0",
        "port": gateway_port,
        "reload": reload_mode,
        "log_level": log_level,
    }
    
    # 如果开启热重载，只监控特定目录和文件类型
    if reload_mode:
        uvicorn_config["reload_dirs"] = [
            "/home/xkb2/Desktop/HQ/api_gateway",  # 你的项目目录
            # 如果有其他需要监控的目录，可以添加在这里
        ]
        uvicorn_config["reload_includes"] = ["*.py"]  # 只监控 Python 文件
        uvicorn_config["reload_excludes"] = [
            "*.pyc",
            "__pycache__/*",
            "*.log",
            "*.tmp",
            ".git/*",
            "venv/*",
            "env/*",
        ]
        logger.info(f"热重载已启用，监控目录: {uvicorn_config['reload_dirs']}")
    
    if use_ssl:
        uvicorn_config["ssl_keyfile"] = ssl_keyfile
        uvicorn_config["ssl_certfile"] = ssl_certfile
        logger.info(f"SSL 已启用。密钥: {ssl_keyfile}, 证书: {ssl_certfile} (SSL is ENABLED. Key: {ssl_keyfile}, Cert: {ssl_certfile})")
    else:
        logger.info("SSL 已禁用，因为未找到密钥/证书文件。(SSL is DISABLED as key/cert files were not found.)")

    uvicorn.run("__main__:app", **uvicorn_config)