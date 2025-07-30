import asyncio
import datetime
import json
import logging
import os
import shutil
import tempfile
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union

import uvicorn
from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.routing import APIRouter
from pydantic import BaseModel, Field

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

# 设置日志
logger = logging.getLogger("olmocr-api")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)

# 创建FastAPI应用实例
app = FastAPI(title="OLMOCR API", description="API for processing PDFs using OLMOCR pipeline")

# 创建APIRouter，所有请求路径前缀为/olmocr
ocr_router = APIRouter(prefix="/olmocr")

# 添加CORS中间件，允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量
UPLOAD_DIR = Path("./uploads")  # 上传文件保存目录
RESULTS_DIR = Path("./results")  # 结果保存目录
MODEL_NAME = "./model/olmOCR-7B-0225-preview"  # 默认模型名称
MAX_CONCURRENT_TASKS = 100  # 最大并发任务数
MAX_PARALLEL_PROCESSING = 3  # 最大并行处理数量 - 可以根据您的硬件调整

# 查找可用端口函数
def find_available_port(start_port=30024, max_attempts=100):
    """查找可用端口，从start_port开始尝试"""
    import socket

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

# 如果目录不存在，则创建
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# 全局信号量，用于控制并发数量
concurrency_semaphore = None

# 全局任务队列
task_queue = asyncio.Queue()

# 用于跟踪正在处理的任务
processing_tasks = {}

# SGLang服务器任务
sglang_server_task_obj = None

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
    PENDING = "pending"  # 等待处理
    QUEUED = "queued"    # 已加入队列
    PROCESSING = "processing"  # 处理中
    COMPLETED = "completed"  # 已完成
    FAILED = "failed"  # 失败

# 用于跟踪处理状态的作业模型
class Job(BaseModel):
    id: str  # 作业ID
    filename: str  # 文件名
    status: JobStatus = JobStatus.PENDING  # 作业状态，默认为等待处理
    result_path: Optional[str] = None  # 结果文件路径
    error: Optional[str] = None  # 错误信息
    created_at: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())  # 创建时间
    queue_position: Optional[int] = None  # 队列位置

# 新增：统一的OCR结果响应模型
class OCRResultResponse(BaseModel):
    job_id: str = Field(description="作业ID")
    status: str = Field(description="作业状态")
    text: str = Field(description="OCR识别结果文本，未成功时为空字符串")

# 内存中的作业存储
jobs = {}

# 任务处理队列
async def task_processor():
    """处理队列中的任务"""
    logger.info("任务处理器启动")

    while True:
        # 获取队列中的下一个任务
        job_id = await task_queue.get()

        # 更新队列中所有作业的位置
        update_queue_positions()

        try:
            # 获取作业信息
            job = jobs.get(job_id)
            if not job or job.status == JobStatus.FAILED:
                task_queue.task_done()
                continue

            # 更新作业状态
            job.status = JobStatus.PROCESSING
            job.queue_position = None

            # 获取信号量以限制并发
            async with concurrency_semaphore:
                logger.info(f"处理作业 {job_id}")

                # 处理作业
                await process_job(job_id)
        except Exception as e:
            logger.exception(f"处理作业 {job_id} 时出错: {e}")
            if job_id in jobs:
                jobs[job_id].status = JobStatus.FAILED
                jobs[job_id].error = str(e)
        finally:
            # 标记任务完成
            task_queue.task_done()

def update_queue_positions():
    """更新队列中所有作业的位置"""
    queued_jobs = [job_id for job_id, job in jobs.items() if job.status == JobStatus.QUEUED]

    for position, job_id in enumerate(queued_jobs, 1):
        if job_id in jobs:
            jobs[job_id].queue_position = position

# Args类，模拟argparse命名空间，以兼容现有代码
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

# 获取配置的依赖项
async def get_config() -> OLMOCRConfig:
    return OLMOCRConfig()

# 启动和关闭事件
@app.on_event("startup")
async def startup_event():
    """应用启动时执行的事件"""
    global concurrency_semaphore, sglang_server_task_obj

    # 检查系统要求
    try:
        check_poppler_version()
        check_sglang_version()
        #check_torch_gpu_available()
    except Exception as e:
        logger.error(f"初始化OLMOCR失败: {e}")
        raise RuntimeError(f"初始化OLMOCR失败: {e}")

    # 初始化信号量 - 允许多个任务并行处理
    concurrency_semaphore = asyncio.Semaphore(MAX_PARALLEL_PROCESSING)

    # 启动模型服务器
    config = await get_config()
    args = Args(config)

    # 在后台下载模型
    #logger.info("开始下载模型")
    #await download_model(args.model)
    #logger.info("模型下载完成")

    # 启动SGLang服务器
    logger.info("启动SGLang服务器")
    #sglang_server_task_obj = asyncio.create_task(sglang_server_task(args, concurrency_semaphore))

    # 等待服务器准备就绪
    # try:
    #     await sglang_server_ready()
    #     logger.info("SGLang服务器已准备就绪")
    # except Exception as e:
    #     logger.error(f"启动SGLang服务器失败: {e}")
    #     raise RuntimeError(f"启动SGLang服务器失败: {e}")

    # 启动任务处理器
    for _ in range(MAX_CONCURRENT_TASKS):
        asyncio.create_task(task_processor())

    logger.info(f"API服务已启动，最大并发任务数: {MAX_CONCURRENT_TASKS}，最大并行处理数: {MAX_PARALLEL_PROCESSING}")

@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时执行的事件"""
    global sglang_server_task_obj

    logger.info("正在关闭API服务...")

    # 取消SGLang服务器任务
    if sglang_server_task_obj:
        sglang_server_task_obj.cancel()
        try:
            await sglang_server_task_obj
        except asyncio.CancelledError:
            pass
        logger.info("SGLang服务器已停止")

    # 获取所有正在运行的任务
    for task in asyncio.all_tasks():
        if task is not asyncio.current_task():
            task.cancel()
            try:
                await asyncio.wait_for(task, timeout=1.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

    logger.info("所有任务已关闭，API服务已停止")

async def process_job(job_id: str):
    """处理作业的主函数"""
    job = jobs[job_id]
    file_path = UPLOAD_DIR / f"{job_id}_{job.filename}"

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到上传的文件: {file_path}")

    # 初始化参数
    config = await get_config()
    args = Args(config)

    # 为此作业创建结果目录
    job_result_dir = RESULTS_DIR / job_id
    job_result_dir.mkdir(exist_ok=True)

    # 处理PDF
    worker_id = 0
    pdf_orig_path = str(file_path)

    # 处理PDF
    result = await process_pdf(args, worker_id, pdf_orig_path)

    if result is None:
        raise Exception("PDF处理失败或被过滤掉")

    # 将结果保存为JSON文件
    result_path = job_result_dir / "result.json"
    with open(result_path, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # 更新作业状态为成功
    jobs[job_id].status = JobStatus.COMPLETED
    jobs[job_id].result_path = str(result_path)
    logger.info(f"作业 {job_id} 已完成")

# API端点
@ocr_router.get("/")
async def root():
    """根端点，提供基本的API信息"""
    return {
        "message": "OLMOCR API正在运行",
        "version": "1.0.0",
        "endpoints": {
            "/olmocr/process": "上传并处理PDF文件",
            "/olmocr/jobs/{job_id}": "获取作业状态",
            "/olmocr/jobs": "列出所有作业",
            "/olmocr/results/{job_id}": "获取作业结果",
        }
    }

@ocr_router.post("/process", response_model=Job)
async def process_file(
    file: UploadFile = File(...),
    config: OLMOCRConfig = Depends(get_config)
):
    """
    上传并用OLMOCR处理PDF文件。

    返回一个作业ID，可用于检查状态并检索结果。
    """
    # 验证文件类型
    if not file.filename.lower().endswith(('.pdf', '.png', '.jpg', '.jpeg')):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="仅支持PDF和图像文件（PNG、JPG）"
        )

    # 生成唯一作业ID
    import uuid
    job_id = str(uuid.uuid4())

    # 将文件保存到上传目录
    file_path = UPLOAD_DIR / f"{job_id}_{file.filename}"

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"保存文件失败: {str(e)}"
        )

    # 创建作业
    job = Job(
        id=job_id,
        filename=file.filename,
        status=JobStatus.QUEUED
    )
    jobs[job_id] = job

    # 将作业添加到队列
    await task_queue.put(job_id)

    # 更新队列位置
    update_queue_positions()

    return job

@ocr_router.post("/process_sync")
async def process_file_sync(
    file: UploadFile = File(...),
    config: OLMOCRConfig = Depends(get_config)
):
    """
    上传并用OLMOCR处理PDF文件，等待处理完成后直接返回结果。

    注意：处理大型文件可能需要较长时间，请确保客户端有足够的超时设置。
    """
    # 验证文件类型
    if not file.filename.lower().endswith(('.pdf', '.png', '.jpg', '.jpeg')):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="仅支持PDF和图像文件（PNG、JPG）"
        )

    # 生成临时文件路径
    import uuid
    temp_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{temp_id}_{file.filename}"

    try:
        # 保存上传的文件
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 初始化参数
        args = Args(config)

        # 处理PDF
        worker_id = 0
        pdf_orig_path = str(file_path)

        # 直接处理PDF并等待结果
        async with concurrency_semaphore:
            result = await process_pdf(args, worker_id, pdf_orig_path)

        # 清理临时文件
        if os.path.exists(file_path):
            os.remove(file_path)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="PDF处理失败或被过滤掉"
            )

        # 直接返回处理结果
        return result

    except Exception as e:
        # 确保清理临时文件
        if os.path.exists(file_path):
            os.remove(file_path)

        logger.exception(f"处理文件时出错: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"处理文件时出错: {str(e)}"
        )

@ocr_router.get("/jobs/{job_id}", response_model=Job)
async def get_job(job_id: str):
    """获取特定作业的状态"""
    if job_id not in jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"未找到作业 {job_id}"
        )

    return jobs[job_id]

@ocr_router.get("/jobs", response_model=List[Job])
async def list_jobs(
    status: Optional[JobStatus] = Query(None, description="按状态筛选作业")
):
    """列出所有作业，可选按状态筛选"""
    if status:
        return [job for job in jobs.values() if job.status == status]
    return list(jobs.values())

# 修改后的get_results函数
@ocr_router.get("/results/{job_id}", response_model=OCRResultResponse)
async def get_results(job_id: str):
    """获取作业结果，返回统一的JSON格式"""
    
    # 检查作业是否存在
    if job_id not in jobs:
        return OCRResultResponse(
            job_id=job_id,
            status="not_found",
            text=""
        )
    
    job = jobs[job_id]
    
    # 根据作业状态返回相应结果
    if job.status == JobStatus.COMPLETED:
        # 作业已完成，尝试读取结果
        if job.result_path and os.path.exists(job.result_path):
            try:
                with open(job.result_path, 'r', encoding='utf-8') as f:
                    result_data = json.load(f)
                    # 提取text字段，如果不存在则返回空字符串
                    text_content = result_data.get('text', '')
                    return OCRResultResponse(
                        job_id=job_id,
                        status="completed",
                        text=text_content
                    )
            except Exception as e:
                logger.error(f"读取作业 {job_id} 结果文件失败: {e}")
                return OCRResultResponse(
                    job_id=job_id,
                    status="result_file_error",
                    text=""
                )
        else:
            # 结果文件不存在
            return OCRResultResponse(
                job_id=job_id,
                status="result_file_missing",
                text=""
            )
    
    elif job.status == JobStatus.FAILED:
        return OCRResultResponse(
            job_id=job_id,
            status="failed",
            text=""
        )
    
    elif job.status == JobStatus.PROCESSING:
        return OCRResultResponse(
            job_id=job_id,
            status="processing",
            text=""
        )
    
    elif job.status == JobStatus.QUEUED:
        return OCRResultResponse(
            job_id=job_id,
            status="queued",
            text=""
        )
    
    elif job.status == JobStatus.PENDING:
        return OCRResultResponse(
            job_id=job_id,
            status="pending",
            text=""
        )
    
    else:
        # 未知状态
        return OCRResultResponse(
            job_id=job_id,
            status="unknown",
            text=""
        )

@ocr_router.get("/config", response_model=OLMOCRConfig)
async def get_current_config(config: OLMOCRConfig = Depends(get_config)):
    """获取当前配置"""
    return config

@ocr_router.post("/config", response_model=OLMOCRConfig)
async def update_config(new_config: OLMOCRConfig):
    """更新配置（需要重启才能应用更改）"""
    # 在实际实现中，你会持久化此配置
    # 在此示例中，我们只返回它并附上说明
    return {
        **new_config.dict(),
        "note": "配置已更新。需要重启服务器才能应用更改。"
    }

@ocr_router.get("/queue")
async def get_queue_info():
    """获取当前队列信息"""
    queued_jobs = [job for job in jobs.values() if job.status == JobStatus.QUEUED]
    processing_jobs = [job for job in jobs.values() if job.status == JobStatus.PROCESSING]

    return {
        "queue_size": len(queued_jobs),
        "processing": len(processing_jobs),
        "max_concurrent": MAX_CONCURRENT_TASKS,
        "max_parallel": MAX_PARALLEL_PROCESSING,
        "available_slots": max(0, MAX_PARALLEL_PROCESSING - len(processing_jobs))
    }

# 注册路由
app.include_router(ocr_router)

# 如果直接执行，则运行服务器
if __name__ == "__main__":
    import signal

    # 正确处理信号
    def handle_exit(signum, frame):
        logger.info(f"收到信号 {signum}，准备退出...")
        # Uvicorn会自动调用应用的shutdown事件

    # 注册信号处理器
    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    # 禁用uvicorn的默认信号处理
    uvicorn.run(
        "olmocr_api:app",
        host="0.0.0.0",
        port=57004,
        reload=False,
        use_colors=True,
        log_level="info",
        timeout_keep_alive=120,
        timeout_graceful_shutdown=30
    )