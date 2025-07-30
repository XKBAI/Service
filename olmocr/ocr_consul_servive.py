# OCR Service with Consul Integration - 队列模式并发优化版本
import asyncio
import datetime
import json
import logging
import os
import shutil
import tempfile
import socket
import atexit
import consul
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union

import uvicorn
from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.routing import APIRouter
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

# 导入olmocr pipeline中的必要组件
from olmocr.pipeline import (
    check_poppler_version,
    check_sglang_version,
    check_torch_gpu_available,
    convert_image_to_pdf_bytes,
    download_model,
    is_jpeg,
    is_png,
    process_pdf,
    sglang_server_ready,
    sglang_server_task,
)

# 服务配置
DEFAULT_CONFIG = {
    "SERVICE_NAME": "ocr-service",
    "SERVICE_ID": "ocr-service-001",
    "SERVICE_PORT": 47003,  # 避免端口冲突
    "CONSUL_HOST": "127.0.0.1",
    "CONSUL_PORT": 8500,
    "HEALTH_CHECK_INTERVAL": "10s",
    "HEALTH_CHECK_TIMEOUT": "5s",
    "HEALTH_CHECK_TTL": "30s",
    "MAX_CONCURRENT_TASKS": 100,
    "MAX_PARALLEL_PROCESSING": 3,
}

# 设置日志
logger = logging.getLogger("olmocr-consul-api")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)

# 全局变量
UPLOAD_DIR = Path("./uploads")
RESULTS_DIR = Path("./results")
MODEL_NAME = "./model/olmOCR-7B-0225-preview"

# 如果目录不存在，则创建
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# 全局变量
concurrency_semaphore = None
task_queue = asyncio.Queue()
processing_tasks = {}
sglang_server_task_obj = None
jobs = {}

class OCRConsulService:
    """OCR微服务的Consul集成类"""
    
    def __init__(self, config: dict = None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.service_name = self.config["SERVICE_NAME"]
        self.service_id = self.config["SERVICE_ID"]
        self.service_port = self.config["SERVICE_PORT"]
        self.service_address = self._get_local_ip()
        
        # 初始化Consul客户端
        try:
            self.client = consul.Consul(
                host=self.config["CONSUL_HOST"],
                port=self.config["CONSUL_PORT"]
            )
            logger.info(f"✅ Consul客户端连接成功: {self.config['CONSUL_HOST']}:{self.config['CONSUL_PORT']}")
        except Exception as e:
            logger.error(f"❌ Consul客户端连接失败: {e}")
            self.client = None
    
    def _get_local_ip(self) -> str:
        """获取本机IP地址"""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"
    
    def register_service(self) -> bool:
        """注册服务到Consul"""
        if not self.client:
            print("⚠️  Consul客户端未初始化，跳过服务注册")
            return False
        
        try:
            # 健康检查配置
            health_check = consul.Check.http(
                url=f"http://{self.service_address}:{self.service_port}/health",
                interval=self.config["HEALTH_CHECK_INTERVAL"],
                timeout=self.config["HEALTH_CHECK_TIMEOUT"],
                deregister=True
            )
            
            # 服务标签
            tags = [
                'ocr',
                'text-recognition',
                'pdf-processing',
                'ai-service',
                f'version-1.0',
                f'port-{self.service_port}',
                'fastapi',
                'olmocr'
            ]
            
            # 注册服务
            self.client.agent.service.register(
                name=self.service_name,
                service_id=self.service_id,
                address=self.service_address,
                port=self.service_port,
                tags=tags,
                check=health_check
            )
            
            logger.info(f"✅ 服务注册成功: {self.service_name} @ {self.service_address}:{self.service_port}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 服务注册失败: {e}")
            return False
    
    def deregister_service(self) -> bool:
        """注销服务"""
        if not self.client:
            return False
        
        try:
            self.client.agent.service.deregister(self.service_id)
            logger.info(f"✅ 服务注销成功: {self.service_id}")
            return True
        except Exception as e:
            logger.error(f"❌ 服务注销失败: {e}")
            return False
    
    def get_service_info(self) -> dict:
        """获取服务信息"""
        return {
            "service_name": self.service_name,
            "service_id": self.service_id,
            "service_address": self.service_address,
            "service_port": self.service_port,
            "consul_connected": self.client is not None
        }

# 全局服务实例
ocr_service = OCRConsulService()

# 查找可用端口函数
def find_available_port(start_port=30024, max_attempts=100):
    """查找可用端口，从start_port开始尝试"""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    return None

# 查找可用的SGLang服务器端口
SGLANG_SERVER_PORT = find_available_port(30024)
if not SGLANG_SERVER_PORT:
    raise RuntimeError("无法找到可用的SGLang服务器端口")

# API设置的配置模型
class OLMOCRConfig(BaseModel):
    model: str = Field(default=MODEL_NAME, description="Model to use for OCR")
    model_max_context: int = Field(default=8192, description="Maximum context length for the model")
    model_chat_template: str = Field(default="qwen2-vl", description="Chat template to use")
    target_longest_image_dim: int = Field(default=1024, description="Target longest dimension for rendered images")
    target_anchor_text_len: int = Field(default=6000, description="Maximum anchor text length")
    max_page_retries: int = Field(default=8, description="Maximum number of retries per page")
    max_page_error_rate: float = Field(default=0.004, description="Maximum allowable page error rate")
    apply_filter: bool = Field(default=True, description="Apply PDF filtering")
    port: int = Field(default=SGLANG_SERVER_PORT, description="Port for SGLang server")

# 作业状态枚举
class JobStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

# 作业模型
class Job(BaseModel):
    id: str
    filename: str
    status: JobStatus = JobStatus.PENDING
    result_path: Optional[str] = None
    error: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())
    queue_position: Optional[int] = None

# OCR结果响应模型
class OCRResultResponse(BaseModel):
    job_id: str = Field(description="作业ID")
    status: str = Field(description="作业状态")
    text: str = Field(description="OCR识别结果文本，未成功时为空字符串")

# 生命周期管理
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时
    global concurrency_semaphore, sglang_server_task_obj
    
    logger.info("🚀 启动OCR微服务...")
    
    # 注册服务到Consul
    if ocr_service.register_service():
        logger.info("✅ 服务已注册到Consul")
    else:
        logger.warning("⚠️  服务注册到Consul失败，服务仍会启动")
    
    try:
        # 检查系统要求
        check_poppler_version()
        check_sglang_version()
        
        # 初始化并发控制
        concurrency_semaphore = asyncio.Semaphore(DEFAULT_CONFIG["MAX_PARALLEL_PROCESSING"])
        
        # 启动任务处理器
        for _ in range(DEFAULT_CONFIG["MAX_CONCURRENT_TASKS"]):
            asyncio.create_task(task_processor())
        
        # 启动模型服务器（如果需要）
        config = OLMOCRConfig()
        args = Args(config)
        
        logger.info(f"✅ OCR微服务启动成功 - 端口: {ocr_service.service_port}")
        
    except Exception as e:
        logger.error(f"❌ 服务启动失败: {e}")
        ocr_service.deregister_service()
        raise
    
    yield  # 服务运行中
    
    # 关闭时
    logger.info("🛑 正在关闭OCR微服务...")
    
    # 注销服务
    if ocr_service.deregister_service():
        logger.info("✅ 服务已从Consul注销")
    
    # 取消任务
    if sglang_server_task_obj:
        sglang_server_task_obj.cancel()
    
    logger.info("✅ OCR微服务已关闭")

# 创建FastAPI应用
app = FastAPI(
    title="OCR微服务",
    description="基于OLMOCR的文档识别微服务，支持Consul服务发现",
    version="1.0.0",
    lifespan=lifespan
)

# 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 创建路由
ocr_router = APIRouter(prefix="/ocr", tags=["OCR"])

# Args类
class Args:
    def __init__(self, config: OLMOCRConfig):
        self.model = config.model
        self.model_max_context = config.model_max_context
        self.model_chat_template = config.model_chat_template
        self.target_longest_image_dim = config.target_longest_image_dim
        self.target_anchor_text_len = config.target_anchor_text_len
        self.max_page_retries = config.max_page_retries
        self.max_page_error_rate = config.max_page_error_rate
        self.apply_filter = config.apply_filter
        self.port = config.port
        self.workspace = str(RESULTS_DIR)
        self.workers = 1

async def get_config() -> OLMOCRConfig:
    return OLMOCRConfig()

# 队列处理相关函数
async def task_processor():
    """处理队列中的任务"""
    logger.info("任务处理器启动")
    
    while True:
        try:
            job_id = await task_queue.get()
            update_queue_positions()
            
            job = jobs.get(job_id)
            if not job or job.status == JobStatus.FAILED:
                task_queue.task_done()
                continue
            
            job.status = JobStatus.PROCESSING
            job.queue_position = None
            
            async with concurrency_semaphore:
                logger.info(f"处理作业 {job_id}")
                await process_job(job_id)
                
        except Exception as e:
            logger.exception(f"处理作业 {job_id} 时出错: {e}")
            if job_id in jobs:
                jobs[job_id].status = JobStatus.FAILED
                jobs[job_id].error = str(e)
        finally:
            task_queue.task_done()

def update_queue_positions():
    """更新队列中所有作业的位置"""
    queued_jobs = [job_id for job_id, job in jobs.items() if job.status == JobStatus.QUEUED]
    for position, job_id in enumerate(queued_jobs, 1):
        if job_id in jobs:
            jobs[job_id].queue_position = position

async def process_job(job_id: str):
    """处理作业的主函数"""
    job = jobs[job_id]
    file_path = UPLOAD_DIR / f"{job_id}_{job.filename}"
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到上传的文件: {file_path}")
    
    config = await get_config()
    args = Args(config)
    
    job_result_dir = RESULTS_DIR / job_id
    job_result_dir.mkdir(exist_ok=True)
    
    worker_id = 0
    pdf_orig_path = str(file_path)
    
    result = await process_pdf(args, worker_id, pdf_orig_path)
    
    if result is None:
        raise Exception("PDF处理失败或被过滤掉")
    
    result_path = job_result_dir / "result.json"
    with open(result_path, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    jobs[job_id].status = JobStatus.COMPLETED
    jobs[job_id].result_path = str(result_path)
    logger.info(f"作业 {job_id} 已完成")

# 健康检查端点
@app.get("/health")
async def health_check():
    """健康检查端点"""
    try:
        service_info = ocr_service.get_service_info()
        queue_info = {
            "queued_jobs": len([j for j in jobs.values() if j.status == JobStatus.QUEUED]),
            "processing_jobs": len([j for j in jobs.values() if j.status == JobStatus.PROCESSING])
        }
        
        return {
            "status": "healthy",
            "timestamp": datetime.datetime.now().isoformat(),
            "service": service_info,
            "queue": queue_info,
            "version": "1.0.0"
        }
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

# 服务信息端点
@app.get("/info")
async def service_info():
    """获取服务信息"""
    return {
        "service": ocr_service.get_service_info(),
        "config": DEFAULT_CONFIG,
        "endpoints": {
            "/health": "健康检查",
            "/info": "服务信息",
            "/ocr/process": "处理文档",
            "/ocr/jobs/{job_id}": "获取作业状态",
            "/ocr/results/{job_id}": "获取作业结果"
        }
    }

# OCR路由
@ocr_router.get("/")
async def ocr_root():
    """OCR服务根端点"""
    return {
        "message": "OCR微服务正在运行",
        "version": "1.0.0",
        "service_id": ocr_service.service_id
    }

@ocr_router.post("/process", response_model=Job)
async def process_file(
    file: UploadFile = File(...),
    config: OLMOCRConfig = Depends(get_config)
):
    """上传并处理文档文件"""
    if not file.filename.lower().endswith(('.pdf', '.png', '.jpg', '.jpeg')):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="仅支持PDF和图像文件（PNG、JPG）"
        )
    
    import uuid
    job_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{job_id}_{file.filename}"
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"保存文件失败: {str(e)}"
        )
    
    job = Job(id=job_id, filename=file.filename, status=JobStatus.QUEUED)
    jobs[job_id] = job
    
    await task_queue.put(job_id)
    update_queue_positions()
    
    return job

@ocr_router.get("/jobs/{job_id}", response_model=Job)
async def get_job(job_id: str):
    """获取作业状态"""
    if job_id not in jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"未找到作业 {job_id}"
        )
    return jobs[job_id]

@ocr_router.get("/results/{job_id}", response_model=OCRResultResponse)
async def get_results(job_id: str):
    """获取作业结果"""
    if job_id not in jobs:
        return OCRResultResponse(job_id=job_id, status="not_found", text="")
    
    job = jobs[job_id]
    
    if job.status == JobStatus.COMPLETED:
        if job.result_path and os.path.exists(job.result_path):
            try:
                with open(job.result_path, 'r', encoding='utf-8') as f:
                    result_data = json.load(f)
                    text_content = result_data.get('text', '')
                    return OCRResultResponse(
                        job_id=job_id,
                        status="completed",
                        text=text_content
                    )
            except Exception as e:
                logger.error(f"读取作业 {job_id} 结果文件失败: {e}")
                return OCRResultResponse(job_id=job_id, status="result_file_error", text="")
        else:
            return OCRResultResponse(job_id=job_id, status="result_file_missing", text="")
    
    return OCRResultResponse(
        job_id=job_id,
        status=job.status.value,
        text=""
    )

@ocr_router.get("/status")
async def service_status():
    """获取服务状态"""
    queue_info = {
        "queued": len([j for j in jobs.values() if j.status == JobStatus.QUEUED]),
        "processing": len([j for j in jobs.values() if j.status == JobStatus.PROCESSING]),
        "completed": len([j for j in jobs.values() if j.status == JobStatus.COMPLETED]),
        "failed": len([j for j in jobs.values() if j.status == JobStatus.FAILED])
    }
    
    return {
        "service": ocr_service.get_service_info(),
        "queue": queue_info,
        "total_jobs": len(jobs)
    }

# 注册路由
app.include_router(ocr_router)

# 程序退出时清理
def cleanup():
    """程序退出时的清理函数"""
    logger.info("正在清理资源...")
    ocr_service.deregister_service()

atexit.register(cleanup)

if __name__ == "__main__":
    import signal
    
    def handle_exit(signum, frame):
        logger.info(f"收到信号 {signum}，准备退出...")
        cleanup()
    
    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)
    
    logger.info(f"🚀 启动OCR微服务，端口: {ocr_service.service_port}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=ocr_service.service_port,
        log_level="info",
        timeout_keep_alive=120,
        timeout_graceful_shutdown=30
    )