# api_gateway_explicit_docs.py
import httpx
from fastapi import (
    FastAPI, Request, Response, HTTPException, Depends, status,
    File, UploadFile, Form, Body, Path, Query, BackgroundTasks,
    APIRouter
)
from fastapi.security import APIKeyHeader
from fastapi.responses import StreamingResponse, FileResponse # Need StreamingResponse for files
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union # Import necessary types
from enum import Enum
import uvicorn
import asyncio
import logging
import os
import io # For handling file streams
from datetime import datetime

# --- 配置日志 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- API Key 配置 ---
API_KEY = "aK8sP2zQfT9rX7vYmWnJbE4gH6dC1uI0oZlMxKy" # 使用之前确定的复杂 Key
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# --- 验证 API Key 函数 ---
async def get_api_key(key: str = Depends(api_key_header)):
    """依赖项函数，用于验证请求头中的 X-API-Key"""
    if key == API_KEY:
        return key
    elif key is None:
         raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API Key in X-API-Key header",
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
        )

# --- 配置后端服务 ---
# 将chat改为user
BACKEND_SERVICES = {
    "stt": {
        "base_url": os.getenv("STT_SERVICE_URL", "http://localhost:57001"),
        "prefix": "/stt"
    },
    "tts": {
        "base_url": os.getenv("TTS_SERVICE_URL", "http://localhost:57002"),
        "prefix": "/tts"
    },
    "ocr": {
        "base_url": os.getenv("OCR_SERVICE_URL", "http://localhost:57004"), # OCR 默认在 57004
        "prefix": "/olmocr"
    },
    "user": {  # 改为user
        "base_url": os.getenv("USER_SERVICE_URL", "http://localhost:58000"),  # 环境变量也更新
        "prefix": "/user"  # 前缀改为user
    }
}

# --- 全局变量 ---
http_clients: Dict[str, httpx.AsyncClient] = {}

# --- Pydantic 模型定义 (从后端服务复制或重新定义) ---
# STT Models (来自 finaltest-api.py)
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

# TTS Models (来自 test_api.py)
class TTSRequest(BaseModel):
    text: str = Field(..., description="要转换的文本")
    language: str = Field("ZH", description="语言代码")
    speaker_id: Optional[str] = Field("ZH", description="说话人ID")
    speed: float = Field(1.0, description="语速 (0.5-2.0)")

# OCR Models (来自 olmocr_api.py)
class OLMOCRConfig(BaseModel): # 只定义结构，实际配置在后端
    model: Optional[str] = None
    model_max_context: Optional[int] = None
    # ... 可以根据需要添加其他字段，但网关通常不直接传递这些 ...

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

# 聊天服务模型
class Message(BaseModel):
    """消息基础模型，包含角色和内容"""
    role: str
    content: str

class UserBase(BaseModel):
    """用户基础模型，包含用户ID"""
    user_id: str

class ChatContentRequest(BaseModel):
    """聊天内容请求模型，用于添加新聊天"""
    chat_time: str
    messages: list
    chat_title: Optional[str] = '新聊天'

class ChatSessionIdRequest(BaseModel):
    """聊天会话ID请求模型，用于根据ID操作聊天会话"""
    chat_session_id: str

class ChatSessionUpdateRequest(BaseModel):
    """聊天会话更新请求模型，用于编辑或更新聊天内容"""
    chat_session_id: str
    chat_time: str
    chat_title: Optional[str] = None
    messages: List[Message]

class SuccessResponse(BaseModel):
    """操作成功响应模型"""
    message: str

class SuccessChangeTitleResponse(BaseModel):
    """操作成功响应模型，返回更新后的标题"""
    message: str
    title: str

class ChatSessionIdResponse(BaseModel):
    """聊天会话ID响应模型，用于返回新创建的聊天会话ID"""
    chat_session_id: str

class ChatSessionResponse(BaseModel):
    """聊天会话响应模型，用于返回聊天会话详情"""
    chat_session_id: str
    chat_time: datetime
    chat_title: str
    messages: list

class UserListResponse(BaseModel):
    """用户列表响应模型，返回所有用户的ID列表"""
    users: list

class ChatSessionInfo(BaseModel):
    """聊天会话信息模型，用于返回会话ID、标题和时间"""
    chat_session_id: str
    chat_title: str
    chat_time: datetime

# --- FastAPI 应用 ---
# 更新描述，告知这是显式定义的网关
app = FastAPI(
    title="统一 AI 服务网关 (显式路由版)",
    description="将 STT, TTS, OCR 服务聚合到单一入口 (端口 57000)。此版本在网关层面明确定义了后端服务接口，以便在 /docs 中展示。",
    version="1.1.0" # 版本号稍作区分
)

# --- 生命周期事件 ---
@app.on_event("startup")
async def startup_event():
    logger.info("API 网关启动中，正在初始化 HTTP 客户端...")
    for service_name, config in BACKEND_SERVICES.items():
        http_clients[service_name] = httpx.AsyncClient(
            base_url=config["base_url"],
            timeout=httpx.Timeout(300.0, connect=10.0),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
            follow_redirects=False
        )
        logger.info(f"为服务 '{service_name}' 创建客户端，目标: {config['base_url']}")
    logger.info("所有 HTTP 客户端已初始化。")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("API 网关关闭中，正在关闭 HTTP 客户端...")
    await asyncio.gather(*(client.aclose() for client in http_clients.values()))
    logger.info("所有 HTTP 客户端已关闭。")

# --- 辅助函数：通用请求转发 ---
async def forward_request(
    service_name: str,
    request: Request,
    target_path: str,
    request_data: Optional[Union[Dict, bytes]] = None,
    files: Optional[Dict[str, Any]] = None
):
    """通用函数，用于将请求转发给后端服务"""
    client = http_clients.get(service_name)
    if not client:
        raise HTTPException(status_code=500, detail=f"服务 '{service_name}' 的客户端未初始化")

    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("content-length", None) # httpx 会自动计算
    headers.pop("content-type", None) # httpx 会根据 data/files/json 设置
    headers.pop("x-api-key", None) # 移除 API Key

    if request.client:
        headers["X-Forwarded-For"] = request.client.host
        headers["X-Forwarded-Proto"] = request.url.scheme

    params = dict(request.query_params)
    method = request.method

    logger.info(f"转发请求: {method} {request.url.path} -> {client.base_url}{target_path}")

    try:
        backend_response = await client.request(
            method=method,
            url=target_path,
            headers=headers,
            params=params,
            content=request_data if isinstance(request_data, bytes) else None, # 直接传递 bytes
            data=request_data if isinstance(request_data, dict) else None, # 传递 form data
            files=files, # 传递文件
            timeout=300.0 # 单次请求超时也设置为较长
        )

        # 清理响应头
        response_headers = dict(backend_response.headers)
        excluded_headers = ["connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers", "transfer-encoding", "upgrade", "content-encoding", "server", "date"]
        for header in excluded_headers:
            response_headers.pop(header.lower(), None)

        # 对于文件流等特殊响应需要特殊处理
        if "content-disposition" in response_headers or response_headers.get("content-type", "").startswith(("audio/", "image/", "application/pdf", "application/octet-stream")):
             # 使用 StreamingResponse 返回文件流
             return StreamingResponse(
                 backend_response.aiter_bytes(), # 流式读取内容
                 status_code=backend_response.status_code,
                 headers=response_headers,
                 media_type=response_headers.get("content-type")
             )
        else:
            # 默认按普通 Response 返回
            return Response(
                content=await backend_response.aread(), # 读取完整内容
                status_code=backend_response.status_code,
                headers=response_headers,
                media_type=response_headers.get("content-type")
            )

    except httpx.ConnectError as e:
        logger.error(f"连接后端服务 {service_name} ({client.base_url}) 失败: {e}")
        raise HTTPException(status_code=503, detail=f"无法连接到后端服务 '{service_name}' (Service Unavailable)")
    except httpx.TimeoutException as e:
        logger.error(f"请求后端服务 {service_name} ({client.base_url}) 超时: {e}")
        raise HTTPException(status_code=504, detail=f"后端服务 '{service_name}' 请求超时 (Gateway Timeout)")
    except httpx.RequestError as e:
        logger.error(f"请求后端服务 {service_name} ({client.base_url}) 时发生错误: {e}")
        raise HTTPException(status_code=502, detail=f"代理请求到后端服务 '{service_name}' 时出错 (Bad Gateway)")
    except Exception as e:
        logger.exception(f"处理代理请求时发生意外错误: {e}")
        raise HTTPException(status_code=500, detail="处理请求时发生内部服务器错误")


# --- 显式定义的后端服务路由 ---

# --- STT Service Routes ---
stt_router = APIRouter(prefix="/stt", tags=["STT Service"], dependencies=[Depends(get_api_key)])

@stt_router.post("/transcribe/", response_model=TranscriptionResponse)
async def stt_transcribe(
    request: Request,
    file: UploadFile = File(...),
    language: Optional[str] = Form("zh"),
    beam_size: int = Form(5),
    # batch_size: int = Form(16), # Gateway 通常不关心 batch_size
    vad_filter: bool = Form(True),
    min_silence_duration_ms: Optional[int] = Form(1000)
):
    """转录音频文件"""
    # 准备 form data for backend
    form_data = {
        'language': language,
        'beam_size': str(beam_size), # Form data is typically string
        'vad_filter': str(vad_filter),
    }
    if min_silence_duration_ms is not None:
        form_data['min_silence_duration_ms'] = str(min_silence_duration_ms)

    # 准备文件数据
    files = {'file': (file.filename, await file.read(), file.content_type)}

    # 注意：这里不能直接用 forward_request 的 request_data 参数传递 form
    # httpx 需要明确的 data 和 files 参数来构建 multipart/form-data
    client = http_clients.get("stt")
    if not client:
        raise HTTPException(status_code=500, detail="STT 客户端未初始化")

    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("content-length", None)
    headers.pop("content-type", None) # Let httpx handle multipart Content-Type
    headers.pop("x-api-key", None)

    try:
        backend_response = await client.post(
            "/stt/transcribe/", # 后端服务的实际路径
            data=form_data,
            files=files,
            headers=headers,
            timeout=300.0
        )
        backend_response.raise_for_status() # Check for 4xx/5xx errors
        return backend_response.json()
    except httpx.HTTPStatusError as e:
         # 如果后端返回错误，尝试解析错误详情
        try:
            error_detail = e.response.json().get("detail", e.response.text)
        except:
            error_detail = e.response.text
        logger.error(f"STT 后端服务返回错误 {e.response.status_code}: {error_detail}")
        raise HTTPException(status_code=e.response.status_code, detail=error_detail)
    except httpx.RequestError as e:
        logger.error(f"请求 STT 服务时发生错误: {e}")
        raise HTTPException(status_code=502, detail=f"代理请求到 STT 服务时出错: {type(e).__name__}")
    except Exception as e:
        logger.exception(f"处理 STT 请求时发生意外错误: {e}")
        raise HTTPException(status_code=500, detail="处理 STT 请求时发生内部错误")


@stt_router.get("/health")
async def stt_health(request: Request):
    """检查 STT 服务健康状态"""
    return await forward_request("stt", request, "/stt/health")

@stt_router.get("/")
async def stt_root(request: Request):
    """STT 服务根路径信息"""
    return await forward_request("stt", request, "/stt/")

@stt_router.post("/clear-memory") # Assuming this might be useful
async def stt_clear_memory(request: Request):
    """(可能存在) 清理 STT 服务内存"""
    return await forward_request("stt", request, "/stt/clear-memory")


# --- TTS Service Routes ---
tts_router = APIRouter(prefix="/tts", tags=["TTS Service"], dependencies=[Depends(get_api_key)])

@tts_router.post("/synthesize") # Returns FileResponse, handle with StreamingResponse
async def tts_synthesize(request: Request, tts_input: TTSRequest = Body(...)):
    """合成语音"""
    # 直接将 Pydantic 模型转为 JSON bytes 转发
    return await forward_request("tts", request, "/tts/synthesize", request_data=tts_input.model_dump_json().encode('utf-8'))


@tts_router.get("/status")
async def tts_status(request: Request):
    """检查 TTS 服务状态"""
    return await forward_request("tts", request, "/tts/status")


# --- OCR Service Routes ---
ocr_router = APIRouter(prefix="/olmocr", tags=["OCR Service"], dependencies=[Depends(get_api_key)])

@ocr_router.post("/process_sync") # Returns JSON
async def ocr_process_sync(request: Request, file: UploadFile = File(...)):
    """同步处理 OCR 文件"""
    files = {'file': (file.filename, await file.read(), file.content_type)}
    # 同步处理类似 STT transcribe，需要手动构造请求
    client = http_clients.get("ocr")
    if not client:
        raise HTTPException(status_code=500, detail="OCR 客户端未初始化")

    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("content-length", None)
    headers.pop("content-type", None)
    headers.pop("x-api-key", None)

    try:
        backend_response = await client.post(
            "/olmocr/process_sync", # 后端路径
            files=files,
            headers=headers,
            timeout=300.0
        )
        backend_response.raise_for_status()
        # 假设后端直接返回 JSON
        return backend_response.json()
    except httpx.HTTPStatusError as e:
        try:
            error_detail = e.response.json().get("detail", e.response.text)
        except:
            error_detail = e.response.text
        logger.error(f"OCR 同步处理后端服务返回错误 {e.response.status_code}: {error_detail}")
        raise HTTPException(status_code=e.response.status_code, detail=error_detail)
    except httpx.RequestError as e:
        logger.error(f"请求 OCR 同步处理时发生错误: {e}")
        raise HTTPException(status_code=502, detail=f"代理请求到 OCR 服务时出错: {type(e).__name__}")
    except Exception as e:
        logger.exception(f"处理 OCR 同步请求时发生意外错误: {e}")
        raise HTTPException(status_code=500, detail="处理 OCR 同步请求时发生内部错误")


@ocr_router.post("/process", response_model=Job)
async def ocr_process(request: Request, file: UploadFile = File(...)):
    """异步提交 OCR 处理作业"""
    # 与 process_sync 类似，但路径不同，且返回 Job 模型
    files = {'file': (file.filename, await file.read(), file.content_type)}
    client = http_clients.get("ocr")
    if not client:
        raise HTTPException(status_code=500, detail="OCR 客户端未初始化")
    # ... (构造 headers 类似 process_sync) ...
    headers = dict(request.headers); headers.pop("host", None); headers.pop("content-length", None); headers.pop("content-type", None); headers.pop("x-api-key", None)

    try:
        backend_response = await client.post("/olmocr/process", files=files, headers=headers, timeout=300.0)
        backend_response.raise_for_status()
        return backend_response.json() # 返回 Job 对象
    # ... (错误处理类似 process_sync) ...
    except httpx.HTTPStatusError as e:
        try: error_detail = e.response.json().get("detail", e.response.text)
        except: error_detail = e.response.text
        logger.error(f"OCR 异步提交后端服务返回错误 {e.response.status_code}: {error_detail}")
        raise HTTPException(status_code=e.response.status_code, detail=error_detail)
    except httpx.RequestError as e:
        logger.error(f"请求 OCR 异步提交时发生错误: {e}")
        raise HTTPException(status_code=502, detail=f"代理请求到 OCR 服务时出错: {type(e).__name__}")
    except Exception as e:
        logger.exception(f"处理 OCR 异步提交时发生意外错误: {e}")
        raise HTTPException(status_code=500, detail="处理 OCR 异步提交时发生内部错误")


@ocr_router.get("/jobs/{job_id}", response_model=Job)
async def ocr_get_job(request: Request, job_id: str = Path(...)):
    """获取 OCR 作业状态"""
    return await forward_request("ocr", request, f"/olmocr/jobs/{job_id}")


@ocr_router.get("/jobs", response_model=List[Job])
async def ocr_list_jobs(request: Request, status: Optional[JobStatus] = Query(None)):
    """列出所有 OCR 作业 (可选按状态过滤)"""
    path = "/olmocr/jobs"
    if status:
        path += f"?status={status.value}"
    return await forward_request("ocr", request, path)


@ocr_router.get("/results/{job_id}") # Returns FileResponse (JSON content)
async def ocr_get_results(request: Request, job_id: str = Path(...)):
    """获取已完成的 OCR 作业结果"""
    # forward_request 会自动处理文件/流响应
    return await forward_request("ocr", request, f"/olmocr/results/{job_id}")


@ocr_router.get("/queue")
async def ocr_get_queue(request: Request):
    """获取 OCR 队列信息"""
    return await forward_request("ocr", request, "/olmocr/queue")

@ocr_router.get("/")
async def ocr_root(request: Request):
    """OCR 服务根路径信息"""
    return await forward_request("ocr", request, "/olmocr/")

# --- 创建用户服务路由 ---
user_router = APIRouter(prefix="/user", tags=["User Service"], dependencies=[Depends(get_api_key)])

@user_router.get("/get_all_users/", response_model=UserListResponse)
async def get_all_users(request: Request):
    """获取所有用户ID"""
    return await forward_request("user", request, "/get_all_users/")

@user_router.post("/add_user/", response_model=SuccessResponse)
async def add_user(request: Request, user: UserBase):
    """添加新用户"""
    return await forward_request("user", request, "/add_user/", request_data=user.model_dump_json().encode('utf-8'))

@user_router.delete("/del_user/", response_model=SuccessResponse)
async def del_user(request: Request, user: UserBase):
    """删除用户及其所有聊天记录"""
    return await forward_request("user", request, "/del_user/", request_data=user.model_dump_json().encode('utf-8'))

@user_router.get("/get_user_chat/", response_model=List[ChatSessionInfo])
async def get_user_chat(request: Request, user_id: str):
    """获取用户的所有聊天会话ID和标题"""
    return await forward_request("user", request, f"/get_user_chat/?user_id={user_id}")

@user_router.post("/add_chat/", response_model=ChatSessionIdResponse)
async def add_chat(request: Request, user_id: str, chat_content: ChatContentRequest):
    """添加新聊天会话"""
    # 需要特殊处理，因为需要传递user_id参数和请求体
    client = http_clients.get("user")
    if not client:
        raise HTTPException(status_code=500, detail="User 客户端未初始化")

    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("content-length", None)
    headers.pop("content-type", None)
    headers.pop("x-api-key", None)
    headers["Content-Type"] = "application/json"
    headers["X-API-Key"] = API_KEY

    try:
        backend_response = await client.post(
            f"/add_chat/?user_id={user_id}",
            json=chat_content.model_dump(),
            headers=headers,
            timeout=300.0
        )
        backend_response.raise_for_status()
        return backend_response.json()
    except httpx.HTTPStatusError as e:
        try:
            error_detail = e.response.json().get("detail", e.response.text)
        except:
            error_detail = e.response.text
        logger.error(f"User 服务返回错误 {e.response.status_code}: {error_detail}")
        raise HTTPException(status_code=e.response.status_code, detail=error_detail)
    except Exception as e:
        logger.exception(f"处理User请求时发生意外错误: {e}")
        raise HTTPException(status_code=500, detail=f"处理请求时发生内部错误: {str(e)}")

@user_router.delete("/del_chat/", response_model=SuccessResponse)
async def del_chat(request: Request, chat_session: ChatSessionIdRequest):
    """删除聊天会话"""
    return await forward_request("user", request, "/del_chat/", request_data=chat_session.model_dump_json().encode('utf-8'))

@user_router.put("/edit_chat/", response_model=SuccessResponse)
async def edit_chat(request: Request, chat_session: ChatSessionUpdateRequest):
    """完全替换聊天会话内容，可选更新标题"""
    return await forward_request("user", request, "/edit_chat/", request_data=chat_session.model_dump_json().encode('utf-8'))

@user_router.put("/update_chat/", response_model=SuccessResponse)
async def update_chat(request: Request, chat_session: ChatSessionUpdateRequest):
    """追加聊天会话内容，可选更新标题"""
    return await forward_request("user", request, "/update_chat/", request_data=chat_session.model_dump_json().encode('utf-8'))

@user_router.get("/get_chat_by_session_id/", response_model=ChatSessionResponse)
async def get_chat_by_session_id(request: Request, chat_session_id: str):
    """根据会话ID获取聊天内容，包括标题"""
    return await forward_request("user", request, f"/get_chat_by_session_id/?chat_session_id={chat_session_id}")

@user_router.put("/update_chat_title/", response_model=SuccessChangeTitleResponse)
async def update_chat_title(request: Request, chat_session: ChatSessionResponse):
    """自动更新聊天会话标题"""
    return await forward_request("user", request, "/update_chat_title/", request_data=chat_session.model_dump_json().encode('utf-8'))

@user_router.put("/edit_chat_title/", response_model=SuccessChangeTitleResponse)
async def edit_chat_title(request: Request, chat_session: ChatSessionInfo):
    """手动修改聊天会话标题"""
    return await forward_request("user", request, "/edit_chat_title/", request_data=chat_session.model_dump_json().encode('utf-8'))

# --- 网关自身路由 (不需要 API Key) ---
@app.get("/", tags=["Gateway Info"])
async def get_root():
    """网关根路径，提供基本信息"""
    return {
        "message": "欢迎使用统一 AI 服务网关 (显式路由版)",
        "version": app.version,
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "registered_services": list(BACKEND_SERVICES.keys())
    }

@app.get("/health", tags=["Gateway Info"])
async def health_check():
    """检查网关本身及尝试连接后端服务的状态"""
    # (健康检查逻辑可以保持和之前版本类似)
    service_status = {}
    gateway_healthy = True
    for service_name, client in http_clients.items():
        service_url = BACKEND_SERVICES[service_name]['base_url']
        check_url = "/" # 或者后端服务特定的健康检查路径
        try:
            response = await client.head(check_url, timeout=5.0)
            if 200 <= response.status_code < 500:
                 service_status[service_name] = {"status": "reachable", "target": service_url}
            else:
                 service_status[service_name] = {"status": "unreachable", "target": service_url, "code": response.status_code}
                 gateway_healthy = False
        except (httpx.RequestError, asyncio.TimeoutError) as e:
            logger.warning(f"健康检查连接 '{service_name}' 失败: {type(e).__name__}")
            service_status[service_name] = {"status": "unreachable", "target": service_url, "error": type(e).__name__}
            gateway_healthy = False
        except Exception as e:
            logger.error(f"健康检查时发生意外错误 '{service_name}': {e}")
            service_status[service_name] = {"status": "check_error", "target": service_url, "error": type(e).__name__}
            gateway_healthy = False
    return {
        "gateway_status": "healthy" if gateway_healthy else "degraded",
        "backend_service_reachability": service_status
    }

# --- 包含所有路由 ---
app.include_router(stt_router)
app.include_router(tts_router)
app.include_router(ocr_router)
app.include_router(user_router)  # 将chat_router改为user_router

# --- 运行服务器 ---
if __name__ == "__main__":
    gateway_port = int(os.getenv("GATEWAY_PORT", 57000))
    reload_mode = os.getenv("GATEWAY_RELOAD", "false").lower() == "true"

    logger.info(f"启动 API 网关 (显式路由版)，监听端口: {gateway_port}, Reload 模式: {reload_mode}")
    uvicorn.run(
        "__main__:app", # 或者 "api_gateway_explicit_docs:app"
        host="0.0.0.0",
        port=gateway_port,
        reload=reload_mode,
        log_level="info"
    )
