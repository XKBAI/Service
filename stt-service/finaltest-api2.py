#队列模式并发 - 优化版本 - 批处理支持
import time
import os
import asyncio
import uuid
from io import BytesIO
from typing import Optional, List, Dict, Any, Tuple
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, BackgroundTasks, APIRouter
from fastapi.responses import JSONResponse
import uvicorn
from faster_whisper import WhisperModel, BatchedInferencePipeline
from concurrent.futures import ThreadPoolExecutor
import torch
from contextlib import asynccontextmanager

# 创建lifespan上下文管理器
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 在应用启动时运行的代码
    global models, batched_models, thread_pools, request_queues, workers

    # 检测可用GPU数量
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if gpu_count == 0:
        print("警告: 未检测到GPU，将使用CPU模式")
        device = "cpu"
        compute_type = "int8"
        model_count = 1
    else:
        device = "cuda"
        compute_type = "float16"
        # 使用3个模型实例
        print("@@@@@@@@")
        model_count = 1

    print(f"初始化 {model_count} 个模型实例")

    # 创建多个模型队列和线程池
    thread_pools = []
    request_queues = []
    for _ in range(model_count):
        # 每个模型2个线程，一个用于解码，一个用于处理
        thread_pools.append(ThreadPoolExecutor(max_workers=2))
        request_queues.append(asyncio.Queue())

    # 创建多个模型实例和对应的批处理模型
    models = []
    batched_models = []
    for i in range(model_count):
        print(f"加载模型 {i+1}/{model_count}...")
        model = WhisperModel(MODEL_SIZE, device=device, compute_type=compute_type)
        models.append(model)
        # 为每个模型创建批处理管道
        batched_model = BatchedInferencePipeline(model=model)
        batched_models.append(batched_model)

    # 启动工作线程 - 每个模型一个专用工作线程
    workers = []
    for i in range(model_count):
        worker = asyncio.create_task(process_queue_worker(i))
        workers.append(worker)

    # 预热模型
    await warmup_models()

    print(f"模型 {MODEL_SIZE} 加载成功! 创建了 {model_count} 个模型实例和 {model_count} 个工作线程")

    yield  # 这里FastAPI会启动服务

    # 在应用关闭时运行的代码
    # 取消所有工作线程
    for worker in workers:
        worker.cancel()

    # 关闭线程池
    for thread_pool in thread_pools:
        thread_pool.shutdown(wait=False)

    # 清空队列
    for queue in request_queues:
        while not queue.empty():
            try:
                task = queue.get_nowait()
                if not task.future.done():
                    task.future.set_exception(Exception("服务正在关闭"))
                queue.task_done()
            except asyncio.QueueEmpty:
                break

    print("API服务已关闭，资源已清理")
# 创建主FastAPI应用
app = FastAPI(
    title="Faster-Whisper API",
    description="使用faster-whisper转录音频的API",
    lifespan=lifespan
)

# 创建路由
stt_router = APIRouter(prefix="/stt", tags=["Speech-to-Text"])

# 全局变量
models = []
batched_models = []
thread_pools = []
request_queues = []
workers = []

# 模型配置
MODEL_SIZE = "large-v3"

class TranscriptionRequest(BaseModel):
    language: Optional[str] = None
    beam_size: int = 5
    batch_size: int = 16
    vad_filter: bool = True
    vad_parameters: Optional[Dict[str, Any]] = None
    compute_type: str = "float16"
    device: str = "cuda"

class BatchTranscriptionRequest(BaseModel):
    language: Optional[str] = None
    beam_size: int = 5
    batch_size: int = 16
    vad_filter: bool = True
    vad_parameters: Optional[Dict[str, Any]] = None

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

class BatchTranscriptionResponse(BaseModel):
    results: List[TranscriptionResponse]
    execution_time: float

class TranscriptionTask:
    def __init__(self, task_id, audio_data, params, future):
        self.task_id = task_id
        self.audio_data = audio_data
        self.params = params
        self.future = future
        self.start_time = time.time()

# 预热模型函数
async def warmup_models():
    """使用现有的音频文件预热所有模型实例"""
    print("开始预热模型...")

    # 使用现有的音频文件
    audio_file_path = "audio.mp3"  # 使用当前目录下的audio.mp3文件
    
    # 检查文件是否存在
    if not os.path.exists(audio_file_path):
        print(f"警告: 预热音频文件 {audio_file_path} 不存在，跳过预热")
        return
    
    print(f"使用音频文件进行预热: {audio_file_path}")
    
    # 为每个模型进行预热
    for i, model in enumerate(models):
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                thread_pools[i], 
                lambda: model.transcribe(
                    audio_file_path,
                    beam_size=5,
                    language="zh"
                )
            )
            print(f"模型 {i+1}/{len(models)} 预热完成")
        except Exception as e:
            print(f"模型 {i+1} 预热异常: {str(e)}")
            try:
                # 备用方法：使用batched_model来预热
                await loop.run_in_executor(
                    thread_pools[i], 
                    lambda: batched_models[i].transcribe(
                        audio_file_path,
                        beam_size=5,
                        language="zh",
                        batch_size=64
                    )
                )
                print(f"模型 {i+1} 使用备用方法预热完成")
            except Exception as e2:
                print(f"模型 {i+1} 备用预热也失败: {str(e2)}")

    print("所有模型预热完成")

# 处理音频的函数 (在线程池中运行)
def process_audio_with_model(batched_model, audio_data, params):
    """使用批处理模型处理音频数据"""
    try:
        # 使用BytesIO处理内存中的音频数据
        audio_buffer = BytesIO(audio_data)
            
        # 使用固定批处理大小 64
        batch_size = 64
            
        # 使用批处理模型处理
        segments, info = batched_model.transcribe(
            audio_buffer,
            beam_size=params.get("beam_size", 5),
            language=params.get("language", None),
            vad_filter=params.get("vad_filter", True),
            vad_parameters=params.get("vad_parameters", None),
            batch_size=batch_size
        )
        
        # 处理结果
        segments_list = [
            {"start": segment.start, "end": segment.end, "text": segment.text}
            for segment in segments
        ]
        
        return segments_list, info
    except Exception as e:
        # 记录异常并重新抛出
        print(f"音频处理错误: {str(e)}")
        raise e

# 新增处理多个音频的函数
def process_batch_audio_with_model(batched_model, audio_data_list, params):
    """批量处理多个音频文件"""
    try:
        all_results = []
        batch_info = None
        
        # 将多个音频数据整合为一个处理批次
        for audio_data in audio_data_list:
            # 使用BytesIO处理内存中的音频数据
            audio_buffer = BytesIO(audio_data)
                
            # 使用批处理模型处理
            segments, info = batched_model.transcribe(
                audio_buffer,
                beam_size=params.get("beam_size", 5),
                language=params.get("language", None),
                vad_filter=params.get("vad_filter", True),
                vad_parameters=params.get("vad_parameters", None),
                batch_size=64  # 使用固定批处理大小
            )
            
            # 处理结果
            segments_list = [
                {"start": segment.start, "end": segment.end, "text": segment.text}
                for segment in segments
            ]
            
            all_results.append((segments_list, info))
            
            # 记录上一个处理的结果信息
            batch_info = info
        
        return all_results, batch_info
    except Exception as e:
        print(f"批量音频处理错误: {str(e)}")
        raise e

# 处理队列中的任务 - 每个模型实例一个专用工作线程
async def process_queue_worker(worker_id):
    """工作线程处理对应队列中的任务"""
    print(f"工作线程 {worker_id} 已启动")

    # 获取该工作线程的专用资源
    queue = request_queues[worker_id]
    thread_pool = thread_pools[worker_id]
    batched_model = batched_models[worker_id]

    try:
        while True:
            # 从队列获取任务
            task = await queue.get()
            
            try:
                # 创建线程池执行器任务
                loop = asyncio.get_running_loop()
                
                # 判断是单个音频还是批量音频请求
                if isinstance(task.audio_data, list):
                    # 批量处理多个音频
                    all_results, info = await loop.run_in_executor(
                        thread_pool,
                        process_batch_audio_with_model,
                        batched_model,
                        task.audio_data,
                        task.params
                    )
                    
                    execution_time = time.time() - task.start_time
                    
                    if not task.future.done():
                        task.future.set_result({
                            "all_results": all_results,
                            "execution_time": execution_time
                        })
                    
                    print(f"批量任务 {task.task_id} 完成，处理了 {len(task.audio_data)} 个文件")
                else:
                    # 处理单个音频
                    segments_list, info = await loop.run_in_executor(
                        thread_pool,
                        process_audio_with_model,
                        batched_model,
                        task.audio_data,
                        task.params
                    )
                    
                    execution_time = time.time() - task.start_time
                    
                    if not task.future.done():
                        task.future.set_result({
                            "segments": segments_list,
                            "info": info,
                            "execution_time": execution_time
                        })
                    
                    print(f"任务 {task.task_id} 完成")
            except Exception as e:
                # 设置异常
                if not task.future.done():
                    task.future.set_exception(e)
                print(f"处理任务 {task.task_id} 失败: {str(e)}")
            finally:
                # 标记任务完成
                queue.task_done()
    except asyncio.CancelledError:
        print(f"工作线程 {worker_id} 已取消")
    except Exception as e:
        print(f"工作线程 {worker_id} 异常: {str(e)}")

@stt_router.post("/transcribe/", response_model=TranscriptionResponse)
async def transcribe_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    language: Optional[str] = Form("zh"),
    beam_size: int = Form(5),
    batch_size: int = Form(16),
    vad_filter: bool = Form(True),
    min_silence_duration_ms: Optional[int] = Form(1000)
):
    """转录音频文件API接口"""
    if not file.filename.endswith(('.mp3', '.wav', '.flac', '.ogg', '.m4a')):
        raise HTTPException(status_code=400, detail="不支持的文件格式")

    # 生成唯一的任务ID
    task_id = f"task_{uuid.uuid4()}"

    # 读取文件内容到内存
    audio_data = await file.read()

    # 设置VAD参数
    vad_parameters = None
    if vad_filter and min_silence_duration_ms is not None:
        vad_parameters = dict(min_silence_duration_ms=min_silence_duration_ms)

    # 创建参数字典
    params = {
        "beam_size": beam_size,
        "language": language,
        "vad_filter": vad_filter,
        "vad_parameters": vad_parameters,
        "batch_size": 64  # 使用固定批处理大小
    }

    # 创建Future对象
    future = asyncio.Future()

    # 查找空闲或最短的队列
    empty_queues = [i for i, q in enumerate(request_queues) if q.empty()]

    if empty_queues:
        # 有空闲队列，直接使用第一个空闲队列
        queue_idx = empty_queues[0]
    else:
        # 没有空闲队列，选择最短的队列
        queue_idx = min(range(len(request_queues)), key=lambda i: request_queues[i].qsize())

    # 创建任务并提交到选定的队列
    task = TranscriptionTask(task_id, audio_data, params, future)
    
    # 简化日志 - 只记录分配情况
    print(f"任务 {task_id} 分配到处理器 {queue_idx}")
    
    await request_queues[queue_idx].put(task)

    # 等待处理完成
    try:
        result = await asyncio.wait_for(future, timeout=300)  # 5分钟超时
        
        segments_list = result["segments"]
        info = result["info"]
        execution_time = result["execution_time"]
        
        # 转换为响应格式
        segments = [
            TranscriptionSegment(start=segment["start"], end=segment["end"], text=segment["text"])
            for segment in segments_list
        ]
        
        return TranscriptionResponse(
            language=info.language,
            language_probability=info.language_probability,
            segments=segments,
            execution_time=execution_time,
            task_id=task_id
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail=f"请求处理超时")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"转录过程中出错: {str(e)}")

# 新增批量转录端点
@stt_router.post("/batch-transcribe/", response_model=BatchTranscriptionResponse)
async def batch_transcribe_audio(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    language: Optional[str] = Form("zh"),
    beam_size: int = Form(5),
    batch_size: int = Form(16),
    vad_filter: bool = Form(True),
    min_silence_duration_ms: Optional[int] = Form(1000)
):
    """批量转录多个音频文件API接口"""
    # 检查文件格式
    for file in files:
        if not file.filename.endswith(('.mp3', '.wav', '.flac', '.ogg', '.m4a')):
            raise HTTPException(status_code=400, detail=f"不支持的文件格式: {file.filename}")
    
    # 生成唯一的批处理任务ID
    batch_task_id = f"batch_task_{uuid.uuid4()}"
    start_time = time.time()
    
    # 读取所有文件内容到内存
    audio_data_list = []
    for file in files:
        audio_data = await file.read()
        audio_data_list.append(audio_data)

    # 设置VAD参数
    vad_parameters = None
    if vad_filter and min_silence_duration_ms is not None:
        vad_parameters = dict(min_silence_duration_ms=min_silence_duration_ms)

    # 创建参数字典
    params = {
        "beam_size": beam_size,
        "language": language,
        "vad_filter": vad_filter,
        "vad_parameters": vad_parameters,
        "batch_size": 64  # 使用固定批处理大小
    }

    # 创建Future对象
    future = asyncio.Future()

    # 查找空闲或最短的队列
    empty_queues = [i for i, q in enumerate(request_queues) if q.empty()]

    if empty_queues:
        queue_idx = empty_queues[0]
    else:
        queue_idx = min(range(len(request_queues)), key=lambda i: request_queues[i].qsize())

    # 创建批处理任务并提交到选定的队列
    batch_task = TranscriptionTask(batch_task_id, audio_data_list, params, future)
    
    print(f"批量任务 {batch_task_id} 分配到处理器 {queue_idx}, 包含 {len(files)} 个文件")
    
    await request_queues[queue_idx].put(batch_task)

    # 等待处理完成
    try:
        result = await asyncio.wait_for(future, timeout=600)  # 10分钟超时
        
        all_results = []
        for i, (segments_list, info) in enumerate(result["all_results"]):
            # 为每个文件生成独立的任务ID
            file_task_id = f"{batch_task_id}_file_{i}"
            
            # 转换为响应格式
            segments = [
                TranscriptionSegment(start=segment["start"], end=segment["end"], text=segment["text"])
                for segment in segments_list
            ]
            
            all_results.append(
                TranscriptionResponse(
                    language=info.language,
                    language_probability=info.language_probability,
                    segments=segments,
                    execution_time=result["execution_time"],
                    task_id=file_task_id
                )
            )
        
        return BatchTranscriptionResponse(
            results=all_results,
            execution_time=time.time() - start_time
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail=f"批量请求处理超时")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批量转录过程中出错: {str(e)}")

@stt_router.get("/", description="STT服务根路径")
async def stt_root():
    return {"message": "Faster-Whisper API正在运行。使用/stt/transcribe/端点来转录音频。"}

@app.get("/", description="API根路径")
async def root():
    return {"message": "Faster-Whisper API正在运行。请访问/stt/端点获取STT服务。"}

@stt_router.get("/health", response_model=dict, description="健康检查端点")
async def health_check():
    """健康检查接口，返回服务状态信息"""
    if not models:
        raise HTTPException(status_code=503, detail="模型未加载")

    try:
        import torch
        gpu_available = torch.cuda.is_available()
        gpu_info = {
            "device_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device() if gpu_available else None,
            "device_name": torch.cuda.get_device_name(0) if gpu_available else None,
            "memory_allocated": {
                i: f"{torch.cuda.memory_allocated(i) / (1024**3):.2f} GB" 
                for i in range(torch.cuda.device_count())
            } if gpu_available else {},
            "memory_reserved": {
                i: f"{torch.cuda.memory_reserved(i) / (1024**3):.2f} GB"
                for i in range(torch.cuda.device_count())
            } if gpu_available else {}
        }
    except ImportError:
        gpu_available = False
        gpu_info = {}

    # 获取队列状态
    queue_status = {
        f"queue_{i}": q.qsize() for i, q in enumerate(request_queues)
    }

    # 获取线程池状态
    thread_info = {
        f"pool_{i}": {
            "max_workers": pool._max_workers,
            "active_threads": len([t for t in pool._threads if t.is_alive()])
        } for i, pool in enumerate(thread_pools)
    }

    return {
        "status": "健康",
        "model_loaded": len(models) > 0,
        "batched_model_loaded": len(batched_models) > 0,
        "model_instances": len(models),
        "model_size": MODEL_SIZE,
        "gpu_available": gpu_available,
        "gpu_info": gpu_info,
        "queue_status": queue_status,
        "thread_pool": thread_info,
    }

# 新增端点: 清理内存
@stt_router.post("/clear-memory", description="手动清理GPU内存")
async def clear_memory():
    """手动触发GPU内存清理"""
    if torch.cuda.is_available():
        before = {i: torch.cuda.memory_allocated(i) / (1024**3) for i in range(torch.cuda.device_count())}
        torch.cuda.empty_cache()
        after = {i: torch.cuda.memory_allocated(i) / (1024**3) for i in range(torch.cuda.device_count())}

        return {
            "status": "成功",
            "memory_before": {i: f"{mem:.2f} GB" for i, mem in before.items()},
            "memory_after": {i: f"{mem:.2f} GB" for i, mem in after.items()}
        }
    else:
        return {"status": "跳过", "reason": "GPU不可用"}

# 将路由器包含到应用中
app.include_router(stt_router)

if __name__ == "__main__":
    uvicorn.run("finaltest-api2:app", host="0.0.0.0", port=57001, reload=True)