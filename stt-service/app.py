import time
import os
import asyncio
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, BackgroundTasks, APIRouter
from fastapi.responses import JSONResponse
import uvicorn
from faster_whisper import WhisperModel, BatchedInferencePipeline

# 创建主FastAPI应用
app = FastAPI(
    title="Faster-Whisper API", 
    description="使用faster-whisper转录音频的API"
)

# 创建路由而不是子应用
stt_router = APIRouter(prefix="/stt", tags=["Speech-to-Text"])

# 全局模型实例
model = None
batched_model = None
# 创建一个信号量来控制并发数
semaphore = None

class TranscriptionRequest(BaseModel):
    language: Optional[str] = None
    beam_size: int = 5
    batch_size: int = 16  # 添加batch_size参数
    vad_filter: bool = True
    vad_parameters: Optional[Dict[str, Any]] = None
    compute_type: str = "float16"
    device: str = "cuda"

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

@app.on_event("startup")
async def startup_event():
    global model, batched_model, semaphore
    model_size = "large-v3"
    # 应用启动时初始化模型
    model = WhisperModel(model_size, device="cuda", compute_type="float16")
    # 创建批处理模型
    batched_model = BatchedInferencePipeline(model=model)
    # 创建一个信号量来控制并发
    semaphore = asyncio.Semaphore(48)  # 假设最多允许3个并发请求
    print(f"模型 {model_size} 加载成功！")

async def process_transcription(
    temp_file_path: str,
    beam_size: int,
    language: Optional[str],
    vad_filter: bool,
    vad_parameters: Optional[Dict],
    batch_size: int = 16  # 添加batch_size参数
) -> tuple:
    """异步处理转录任务"""
    async with semaphore:
        # 使用ThreadPoolExecutor包装阻塞操作
        loop = asyncio.get_event_loop()
        segments, info = await loop.run_in_executor(
            None,
            lambda: batched_model.transcribe(  # 使用batched_model替代model
                temp_file_path, 
                beam_size=beam_size,
                language=language,
                vad_filter=vad_filter,
                vad_parameters=vad_parameters,
                batch_size=batch_size  # 添加batch_size参数
            )
        )
        
        # 将segments转换为列表
        segments_list = [
            TranscriptionSegment(start=segment.start, end=segment.end, text=segment.text)
            for segment in segments
        ]
        
        return segments_list, info

@stt_router.post("/transcribe/", response_model=TranscriptionResponse)
async def transcribe_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    language: Optional[str] = Form("zh"),
    beam_size: int = Form(5),
    batch_size: int = Form(16),  # 添加batch_size参数
    vad_filter: bool = Form(True),
    min_silence_duration_ms: Optional[int] = Form(1000)
):
    if not file.filename.endswith(('.mp3', '.wav', '.flac', '.ogg', '.m4a')):
        raise HTTPException(status_code=400, detail="不支持的文件格式")
    
    # 生成唯一的任务ID
    task_id = f"task_{int(time.time() * 1000)}"
    
    # 临时保存上传的文件
    temp_file_path = f"temp_{task_id}_{file.filename}"
    try:
        with open(temp_file_path, "wb") as buffer:
            contents = await file.read()
            buffer.write(contents)
        
        start = time.time()
        
        # 如果启用了VAD，设置VAD参数
        vad_parameters = None
        if vad_filter and min_silence_duration_ms is not None:
            vad_parameters = dict(min_silence_duration_ms=min_silence_duration_ms)
        
        # 异步处理转录
        segments_list, info = await process_transcription(
            temp_file_path, 
            beam_size, 
            language, 
            vad_filter, 
            vad_parameters,
            batch_size  # 添加batch_size参数
        )
        
        execution_time = time.time() - start
        
        # 添加背景任务来删除临时文件
        background_tasks.add_task(os.remove, temp_file_path)
        
        return TranscriptionResponse(
            language=info.language,
            language_probability=info.language_probability,
            segments=segments_list,
            execution_time=execution_time,
            task_id=task_id
        )
    
    except Exception as e:
        # 确保在异常情况下也删除临时文件
        if os.path.exists(temp_file_path):
            background_tasks.add_task(os.remove, temp_file_path)
        raise HTTPException(status_code=500, detail=f"转录过程中出错: {str(e)}")

@stt_router.get("/", description="STT服务根路径")
async def stt_root():
    return {"message": "Faster-Whisper API正在运行。使用/stt/transcribe/端点来转录音频。"}

@app.get("/", description="API根路径")
async def root():
    return {"message": "Faster-Whisper API正在运行。请访问/stt/端点获取STT服务。"}

@stt_router.get("/health", response_model=dict, description="健康检查端点")
async def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        gpu_info = {
            "device_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device(),
            "device_name": torch.cuda.get_device_name(0) if gpu_available else None,
        }
    except ImportError:
        gpu_available = False
        gpu_info = {}
    
    return {
        "status": "健康",
        "model_loaded": model is not None,
        "batched_model_loaded": batched_model is not None,  # 添加批处理模型状态
        "model_size": "large-v3",
        "gpu_available": gpu_available,
        "gpu_info": gpu_info,
        "concurrent_tasks": semaphore._value  # 当前可用的并发槽位
    }

# 将路由器包含到应用中
app.include_router(stt_router)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=57001, reload=True)