# cosyvoice_fastapi_wrapper.py
import asyncio
import time
import tempfile
import os
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import soundfile as sf
import grpc
from grpc import aio
import cosyvoice_pb2
import cosyvoice_pb2_grpc
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="CosyVoice TTS Service", version="1.0.0")

# 添加CORS支持
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === 请求模型 ===

class TTSRequest(BaseModel):
    text: str
    spk_id: str = "001"
    stream: bool = False
    speed: float = 1.0
    format: str = ""  # '', 'pcm'

class BatchTTSRequest(BaseModel):
    texts: List[str]
    spk_id: str = "001"
    max_conc: int = 4

# === 工具函数 ===

def convert_audio_bytes_to_ndarray(raw_audio: bytes, format: str = None) -> np.ndarray:
    """音频字节转numpy数组"""
    if not format:
        return np.frombuffer(raw_audio, dtype=np.float32).reshape(1, -1)
    elif format in {'pcm'}:
        return np.frombuffer(raw_audio, dtype=np.int16).reshape(1, -1)
    else:
        raise ValueError(f"Unsupported format: {format}")

# === gRPC客户端 ===

class CosyVoiceGRPCClient:
    def __init__(self, host: str = "localhost", port: int = 50000):
        self.host = host
        self.port = port
        
    def construct_request(self, text: str, spk_id: str, stream: bool, speed: float, format: str):
        """构造gRPC请求"""
        request = cosyvoice_pb2.Request()
        request.tts_text = text
        request.stream = stream
        request.speed = speed
        request.text_frontend = True
        request.format = format

        # 使用zero_shot_by_spk_id模式（最常用）
        zero_shot_by_spk_id_request = request.zero_shot_by_spk_id_request
        zero_shot_by_spk_id_request.spk_id = spk_id
            
        return request

    async def synthesize(self, text: str, spk_id: str = "001", stream: bool = False, 
                        speed: float = 1.0, format: str = "") -> bytes:
        """TTS合成"""
        async with aio.insecure_channel(f"{self.host}:{self.port}") as channel:
            stub = cosyvoice_pb2_grpc.CosyVoiceStub(channel)
            
            grpc_request = self.construct_request(text, spk_id, stream, speed, format)
            
            tts_audio = b''
            start_time = time.time()
            
            try:
                async for response in stub.Inference(grpc_request):
                    tts_audio += response.tts_audio
                    
                logger.info(f"TTS completed in {time.time() - start_time:.3f}s")
                return tts_audio
                
            except grpc.RpcError as e:
                logger.error(f"gRPC error: {e.code()}: {e.details()}")
                raise HTTPException(status_code=500, detail=f"gRPC error: {e.details()}")
            except Exception as e:
                logger.error(f"Processing failed: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

# 初始化gRPC客户端
grpc_client = CosyVoiceGRPCClient()

# === FastAPI接口 ===

@app.post("/synthesize")
async def synthesize(request: TTSRequest):
    """TTS语音合成接口"""
    try:
        # 调用gRPC服务
        audio_data = await grpc_client.synthesize(
            text=request.text,
            spk_id=request.spk_id,
            stream=request.stream,
            speed=request.speed,
            format=request.format
        )
        
        # 音频处理
        if request.format in {'', 'pcm'}:
            # 创建临时文件
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                output_path = tmp_file.name
                
            # 转换并保存音频
            tts_array = convert_audio_bytes_to_ndarray(audio_data, request.format)
            sf.write(output_path, tts_array.T, 24000)
            
            # 返回音频文件
            return FileResponse(
                path=output_path,
                media_type="audio/wav",
                filename=f"tts_{request.spk_id}.wav"
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {request.format}")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_synthesize")
async def batch_synthesize(request: BatchTTSRequest):
    """批量TTS处理"""
    try:
        start_time = time.time()
        results = []
        
        # 创建并发任务
        tasks = []
        for text in request.texts:
            tasks.append(grpc_client.synthesize(
                text=text,
                spk_id=request.spk_id
            ))
        
        # 并发执行
        audio_results = await asyncio.gather(*tasks)
        
        # 处理结果
        for i, audio_data in enumerate(audio_results):
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                output_path = tmp_file.name
                
            tts_array = convert_audio_bytes_to_ndarray(audio_data)
            sf.write(output_path, tts_array.T, 24000)
            
            results.append({
                "index": i,
                "text": request.texts[i],
                "audio_path": output_path,
                "audio_data": audio_data.hex()
            })
        
        total_time = time.time() - start_time
        logger.info(f"Batch processing completed in {total_time:.2f}s")
        
        return {
            "results": results,
            "total_time": total_time,
            "count": len(request.texts)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "healthy", "service": "CosyVoice TTS Service"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)  # 从55032改为8000