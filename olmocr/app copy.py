import argparse
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

import boto3
import uvicorn
from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

# 导入olmocr pipeline中的必要组件
from olmocr.pipeline import (
    check_poppler_version,  # 检查poppler版本
    check_sglang_version,   # 检查sglang版本
    check_torch_gpu_available,  # 检查是否有可用的GPU
    convert_image_to_pdf_bytes,  # 将图片转换为PDF字节
    download_model,  # 下载模型
    is_jpeg,  # 检查文件是否为JPEG格式
    is_png,   # 检查文件是否为PNG格式
    process_pdf,  # 处理PDF文件的核心函数
    sglang_server_ready,  # 检查sglang服务器是否准备就绪
    sglang_server_task,   # sglang服务器任务
)

# 设置日志
logger = logging.getLogger("olmocr-api")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)

# 创建FastAPI应用实例
app = FastAPI(title="OLMOCR API", description="API for processing PDFs using OLMOCR pipeline")

# 添加CORS中间件，允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有源
    allow_credentials=True,  # 允许凭证
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头
)

# 全局变量
UPLOAD_DIR = Path("./uploads")  # 上传文件保存目录
RESULTS_DIR = Path("./results")  # 结果保存目录
SGLANG_SERVER_PORT = 30024  # SGLang服务器端口
MODEL_NAME = "allenai/olmOCR-7B-0225-preview"  # 默认模型名称
sglang_server_task_obj = None  # SGLang服务器任务对象
semaphore = None  # 用于控制并发的信号量

# 如果目录不存在，则创建
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# API设置的配置模型
class OLMOCRConfig(BaseModel):
    model: str = Field(default=MODEL_NAME, description="Model to use for OCR")  # OCR使用的模型
    model_max_context: int = Field(default=8192, description="Maximum context length for the model")  # 模型的最大上下文长度
    model_chat_template: str = Field(default="qwen2-vl", description="Chat template to use")  # 使用的聊天模板
    target_longest_image_dim: int = Field(default=1024, description="Target longest dimension for rendered images")  # 渲染图像的目标最长尺寸
    target_anchor_text_len: int = Field(default=6000, description="Maximum anchor text length")  # 最大锚文本长度
    max_page_retries: int = Field(default=8, description="Maximum number of retries per page")  # 每页最大重试次数
    max_page_error_rate: float = Field(default=0.004, description="Maximum allowable page error rate")  # 最大可接受的页面错误率
    apply_filter: bool = Field(default=True, description="Apply PDF filtering")  # 是否应用PDF过滤
    port: int = Field(default=SGLANG_SERVER_PORT, description="Port for SGLang server")  # SGLang服务器端口

# 作业状态枚举
class JobStatus(str, Enum):
    PENDING = "pending"  # 等待处理
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

# 内存中的作业存储（生产环境中应替换为数据库）
jobs = {}

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
        self.workspace = str(RESULTS_DIR)  # 工作空间设为结果目录
        self.workers = 1  # 在此API中一次处理一个文档

# 获取配置的依赖项
async def get_config() -> OLMOCRConfig:
    return OLMOCRConfig()

# 启动和关闭事件
@app.on_event("startup")
async def startup_event():
    """应用启动时执行的事件"""
    global semaphore
    
    # 检查系统要求
    try:
        check_poppler_version()  # 检查poppler版本
        check_sglang_version()   # 检查sglang版本
        check_torch_gpu_available()  # 检查GPU可用性
    except Exception as e:
        logger.error(f"初始化OLMOCR失败: {e}")
        raise RuntimeError(f"初始化OLMOCR失败: {e}")
    
    # 初始化用于控制工作访问的信号量
    semaphore = asyncio.Semaphore(1)
    
    # 启动模型服务器
    config = await get_config()
    args = Args(config)
    
    # 在后台下载模型
    # asyncio.create_task(download_model(args.model))
    
    # 启动服务器
    asyncio.create_task(start_sglang_server(args))

@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时执行的事件"""
    global sglang_server_task_obj
    
    if sglang_server_task_obj:
        sglang_server_task_obj.cancel()  # 取消SGLang服务器任务
        logger.info("SGLang服务器已停止")

async def start_sglang_server(args):
    """启动SGLang服务器"""
    global sglang_server_task_obj
    
    # 启动服务器
    sglang_server_task_obj = asyncio.create_task(sglang_server_task(args, semaphore))
    
    # 等待服务器准备就绪
    try:
        await sglang_server_ready()
        logger.info("SGLang服务器已准备就绪")
    except Exception as e:
        logger.error(f"启动SGLang服务器失败: {e}")
        raise RuntimeError(f"启动SGLang服务器失败: {e}")

# API端点
@app.get("/")
async def root():
    """根端点，提供基本的API信息"""
    return {
        "message": "OLMOCR API正在运行",
        "version": "1.0.0",
        "endpoints": {
            "/process": "上传并处理PDF文件",
            "/jobs/{job_id}": "获取作业状态",
            "/jobs": "列出所有作业",
            "/results/{job_id}": "获取作业结果",
        }
    }

@app.post("/process", response_model=Job)
async def process_file(
    background_tasks: BackgroundTasks,  # 后台任务
    file: UploadFile = File(...),  # 上传的文件
    config: OLMOCRConfig = Depends(get_config)  # 配置
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
        status=JobStatus.PENDING
    )
    jobs[job_id] = job
    
    # 在后台处理
    background_tasks.add_task(process_file_task, job_id, file_path, config)
    
    return job

@app.post("/process_sync")
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

async def process_file_task(job_id: str, file_path: Path, config: OLMOCRConfig):
    """处理文件的后台任务"""
    try:
        # 更新作业状态
        jobs[job_id].status = JobStatus.PROCESSING
        
        # 初始化参数
        args = Args(config)
        
        # 处理PDF
        worker_id = 0
        pdf_orig_path = str(file_path)
        
        # 为此作业创建结果目录
        job_result_dir = RESULTS_DIR / job_id
        job_result_dir.mkdir(exist_ok=True)
        
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
        
    except Exception as e:
        logger.exception(f"处理作业 {job_id} 时出错: {e}")
        jobs[job_id].status = JobStatus.FAILED
        jobs[job_id].error = str(e)

@app.get("/jobs/{job_id}", response_model=Job)
async def get_job(job_id: str):
    """获取特定作业的状态"""
    if job_id not in jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"未找到作业 {job_id}"
        )
    
    return jobs[job_id]

@app.get("/jobs", response_model=List[Job])
async def list_jobs(
    status: Optional[JobStatus] = Query(None, description="按状态筛选作业")
):
    """列出所有作业，可选按状态筛选"""
    if status:
        return [job for job in jobs.values() if job.status == status]
    return list(jobs.values())

@app.get("/results/{job_id}")
async def get_results(job_id: str):
    """获取已完成作业的结果"""
    if job_id not in jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"未找到作业 {job_id}"
        )
    
    job = jobs[job_id]
    
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"作业 {job_id} 尚未完成（当前状态: {job.status}）"
        )
    
    if not job.result_path or not os.path.exists(job.result_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"未找到作业 {job_id} 的结果"
        )
    
    return FileResponse(job.result_path, media_type="application/json")

@app.get("/config", response_model=OLMOCRConfig)
async def get_current_config(config: OLMOCRConfig = Depends(get_config)):
    """获取当前配置"""
    return config

@app.post("/config", response_model=OLMOCRConfig)
async def update_config(new_config: OLMOCRConfig):
    """更新配置（需要重启才能应用更改）"""
    # 在实际实现中，你会持久化此配置
    # 在此示例中，我们只返回它并附上说明
    return {
        **new_config.dict(),
        "note": "配置已更新。需要重启服务器才能应用更改。"
    }

# 如果直接执行，则运行服务器
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)