import time
import os
import asyncio
import uuid
from io import BytesIO
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, APIRouter
from concurrent.futures import ThreadPoolExecutor
import torch
from contextlib import asynccontextmanager
import uvicorn
from faster_whisper import WhisperModel
import numpy as np
import wave

# 创建lifespan上下文管理器
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, worker_semaphore, thread_pool
    
    # 检测GPU
    if not torch.cuda.is_available():
        print("警告: 未检测到GPU，将使用CPU模式")
        device = "cpu"
        compute_type = "int8"
    else:
        device = "cuda"
        compute_type = "float16"
        print(f"检测到GPU: {torch.cuda.get_device_name(0)}")
        
    print("初始化高性能转录模型...")
    
    # 创建单一高性能模型实例
    # 优先使用本地模型路径，避免网络下载
    model_path = "/app/large-v3"  # 直接挂载到应用目录
    if os.path.exists(model_path) and os.path.exists(os.path.join(model_path, "model.bin")):
        print(f"使用本地模型路径: {model_path}")
        model = WhisperModel(
            model_path,
            device=device, 
            compute_type=compute_type,
            cpu_threads=16
        )
    else:
        print(f"本地模型不存在，尝试使用缓存模型或下载: {MODEL_SIZE}")
        try:
            model = WhisperModel(
                MODEL_SIZE,
                device=device, 
                compute_type=compute_type,
                cpu_threads=16,  
                download_root="/tmp/faster_whisper",
                local_files_only=False  # 允许网络下载
            )
        except Exception as e:
            print(f"模型加载失败: {e}")
            print("尝试使用较小的模型作为备用方案...")
            try:
                model = WhisperModel(
                    "base",  # 使用较小的base模型作为备用
                    device=device, 
                    compute_type=compute_type,
                    cpu_threads=16,  
                    download_root="/tmp/faster_whisper",
                    local_files_only=False
                )
                print("✅ 使用base模型作为备用方案")
            except Exception as e2:
                print(f"备用模型也加载失败: {e2}")
                raise e2
    
    # 创建线程池用于CPU处理任务
    thread_pool = ThreadPoolExecutor(max_workers=20)
    
    # 创建信号量控制GPU并发访问 - 设置为20表示最多20个并发任务
    worker_semaphore = asyncio.Semaphore(20)
    
    # 预热模型
    await warmup_model()
    
    print(f"模型 {MODEL_SIZE} 加载成功! 最大并发处理数: 20")
    
    yield
    
    # 清理资源
    thread_pool.shutdown(wait=False)
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
model = None
worker_semaphore = None
thread_pool = None
# 模型配置
MODEL_SIZE = "large-v3"

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

# 修改后的预热模型函数
async def warmup_model():
    """预热模型以减少首次推理延迟"""
    print("开始预热模型...")
    
    try:
        # 创建一段简单的正弦波音频作为预热数据
        sample_rate = 16000
        duration = 1.0  # 1秒
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = np.sin(2 * np.pi * 440 * t) * 0.5  # 440Hz音调
        audio_data = (audio_data * 32767).astype(np.int16)
        
        # 将NumPy数组转换为字节
        buffer = BytesIO()
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())
        
        buffer.seek(0)  # 重置缓冲区位置
        
        # 使用预热音频
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            thread_pool, 
            lambda: model.transcribe(
                buffer,
                beam_size=5,
                language="zh"
            )
        )
        print("模型预热完成")
    except Exception as e:
        print(f"模型预热异常: {str(e)}")
        import traceback
        traceback.print_exc()  # 打印详细异常信息

# 直接处理音频，不再使用队列和工作线程
async def process_audio_direct(audio_data, params):
    """直接处理音频，使用信号量控制并发"""
    async with worker_semaphore:  # 控制并发数
        start_time = time.time()
        
        try:
            # 创建内存流
            audio_buffer = BytesIO(audio_data)
            
            # 使用线程池执行CPU密集型任务
            loop = asyncio.get_running_loop()
            segments, info = await loop.run_in_executor(
                thread_pool,
                lambda: model.transcribe(
                    audio_buffer,
                    beam_size=params.get("beam_size", 5),
                    language=params.get("language", None),
                    vad_filter=params.get("vad_filter", True),
                    vad_parameters=params.get("vad_parameters", None)
                )
            )
            
            # 处理结果
            segments_list = [
                {"start": segment.start, "end": segment.end, "text": segment.text}
                for segment in segments
            ]
            
            execution_time = time.time() - start_time
            
            return {
                "segments": segments_list,
                "info": info,
                "execution_time": execution_time
            }
        except Exception as e:
            print(f"音频处理错误: {str(e)}")
            raise e

@stt_router.post("/transcribe/", response_model=TranscriptionResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    language: Optional[str] = Form("zh"),
    beam_size: int = Form(5),
    vad_filter: bool = Form(True),
    min_silence_duration_ms: Optional[int] = Form(1000)
):
    """简化的转录接口，直接处理请求"""
    if not file.filename.endswith(('.mp3', '.wav', '.flac', '.ogg', '.m4a')):
        raise HTTPException(status_code=400, detail="不支持的文件格式")
    
    # 生成任务ID
    task_id = f"task_{uuid.uuid4()}"
    
    # 读取文件内容
    audio_data = await file.read()
    file_size = len(audio_data)/1024/1024
    print(f"任务 {task_id} 开始处理，文件大小: {file_size:.2f}MB")
    
    # 设置VAD参数
    vad_parameters = None
    if vad_filter and min_silence_duration_ms is not None:
        vad_parameters = dict(min_silence_duration_ms=min_silence_duration_ms)
    
    # 创建参数
    params = {
        "beam_size": beam_size,
        "language": language,
        "vad_filter": vad_filter,
        "vad_parameters": vad_parameters
    }
    
    # 直接处理
    try:
        result = await process_audio_direct(audio_data, params)
        
        # 构建响应
        segments = [
            TranscriptionSegment(start=segment["start"], end=segment["end"], text=segment["text"])
            for segment in result["segments"]
        ]
        
        print(f"任务 {task_id} 完成，执行时间: {result['execution_time']:.2f}秒")
        
        return TranscriptionResponse(
            language=result["info"].language,
            language_probability=result["info"].language_probability,
            segments=segments,
            execution_time=result["execution_time"],
            task_id=task_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"转录过程中出错: {str(e)}")

@stt_router.get("/", description="STT服务根路径")
async def stt_root():
    return {"message": "Faster-Whisper API正在运行。使用/stt/transcribe/端点来转录音频。"}

@app.get("/", description="API根路径")
async def root():
    return {"message": "Faster-Whisper API正在运行。请访问/stt/端点获取STT服务。"}

@stt_router.get("/health", response_model=dict, description="健康检查端点")
async def health_check():
    """健康检查接口，返回服务状态信息"""
    if not model:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        gpu_info = {
            "device_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device() if gpu_available else None,
            "device_name": torch.cuda.get_device_name(0) if gpu_available else None,
            "memory_allocated": f"{torch.cuda.memory_allocated(0) / (1024**3):.2f} GB" if gpu_available else "N/A",
            "memory_reserved": f"{torch.cuda.memory_reserved(0) / (1024**3):.2f} GB" if gpu_available else "N/A"
        }
    except ImportError:
        gpu_available = False
        gpu_info = {}
    
    # 获取当前可用信号量数量
    available_workers = worker_semaphore._value if worker_semaphore else 0
    
    # 获取线程池状态
    thread_info = {
        "max_workers": thread_pool._max_workers,
        "active_threads": len([t for t in thread_pool._threads if t.is_alive()]),
    }
    
    return {
        "status": "健康",
        "model_loaded": model is not None,
        "model_size": MODEL_SIZE,
        "available_workers": available_workers,
        "max_concurrent_workers": 20,
        "gpu_available": gpu_available,
        "gpu_info": gpu_info,
        "thread_pool": thread_info,
    }

# 新增端点: 清理内存
@stt_router.post("/clear-memory", description="手动清理GPU内存")
async def clear_memory():
    """手动触发GPU内存清理"""
    if torch.cuda.is_available():
        before = torch.cuda.memory_allocated(0) / (1024**3)
        torch.cuda.empty_cache()
        after = torch.cuda.memory_allocated(0) / (1024**3)
        
        return {
            "status": "成功",
            "memory_before": f"{before:.2f} GB",
            "memory_after": f"{after:.2f} GB",
            "memory_freed": f"{before - after:.2f} GB"
        }
    else:
        return {"status": "跳过", "reason": "GPU不可用"}

# 将路由器包含到应用中
app.include_router(stt_router)

if __name__ == "__main__":
    uvicorn.run("app2:app", host="0.0.0.0", port=9000, reload=True)
