# client_fastapi.py - 根据utils.py完全修复版
import asyncio
from importlib import reload
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
import librosa
import grpc
from grpc import aio
import cosyvoice_pb2
import cosyvoice_pb2_grpc
import logging
import torch
import torchaudio

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="CosyVoice TTS Service with QC16K Voice", version="1.0.0")

# 添加CORS支持
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TTSRequest(BaseModel):
    text: str
    speed: float = 1.0
    stream: bool = False
    format: str = ""

# === 根据utils.py修复的音频处理函数 ===

def load_wav_for_server(wav_path: str, target_sr: int = 16000) -> np.ndarray:
    """加载音频并转换为server端期望的float32格式"""
    try:
        data, sample_rate = sf.read(wav_path, dtype='float32')
        
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        
        if sample_rate != target_sr:
            data = librosa.resample(
                data,
                orig_sr=sample_rate,
                target_sr=target_sr,
                res_type='kaiser_fast'
            )
        
        # 确保数据范围在[-1, 1]（float32格式）
        data = np.clip(data, -1.0, 1.0)
        
        # 保持float32格式（这是server端期望的）
        data = data.astype(np.float32)
        data = data.reshape(1, -1)  # (1, samples)
        
        logger.info(f"音频加载成功: shape={data.shape}, dtype={data.dtype}")
        return data
        
    except Exception as e:
        logger.error(f"音频加载失败 {wav_path}: {str(e)}")
        raise

def convert_audio_ndarray_to_bytes_for_server(array: np.ndarray) -> bytes:
    """将numpy数组转换为server端期望的float32 bytes"""
    # 确保数据类型为float32（server端期望的格式）
    if array.dtype != np.float32:
        array = array.astype(np.float32)
    
    # 确保数据范围在[-1, 1]
    array = np.clip(array, -1.0, 1.0)
    
    # 确保数组是连续的
    array = np.ascontiguousarray(array)
    
    if array.size == 0:
        raise ValueError("音频数组为空")
    
    # 转换为bytes（与server端的convert_audio_bytes_to_tensor匹配）
    audio_bytes = array.tobytes()
    
    logger.info(f"数组转bytes: shape={array.shape}, dtype={array.dtype}, bytes_len={len(audio_bytes)}")
    return audio_bytes

def reconstruct_audio_from_server_response(audio_chunks: List[bytes]) -> torch.Tensor:
    """根据server端的convert_audio_tensor_to_bytes重建音频"""
    try:
        all_audio_data = []
        
        for i, chunk in enumerate(audio_chunks):
            logger.info(f"处理chunk {i}: {len(chunk)} bytes")
            
            # server端返回的是float32格式（根据utils.py第12行）
            audio_array = np.frombuffer(chunk, dtype=np.float32)
            
            if len(audio_array) == 0:
                logger.warning(f"Chunk {i} 为空")
                continue
                
            # 转换为torch tensor
            audio_tensor = torch.from_numpy(audio_array).float()
            all_audio_data.append(audio_tensor)
            
            logger.info(f"Chunk {i} 处理完成: {len(audio_array)} samples")
        
        if not all_audio_data:
            raise ValueError("没有有效的音频数据")
        
        # 合并所有音频块
        final_audio = torch.cat(all_audio_data, dim=0)
        
        # 重新塑形为 (1, samples) 格式
        if final_audio.dim() == 1:
            final_audio = final_audio.unsqueeze(0)
        
        # 确保数据范围正确
        final_audio = torch.clamp(final_audio, -1.0, 1.0)
        
        duration = final_audio.shape[1] / 24000  # CosyVoice默认采样率
        logger.info(f"音频重建完成: shape={final_audio.shape}, 时长={duration:.2f}s")
        
        return final_audio
        
    except Exception as e:
        logger.error(f"音频重建失败: {str(e)}")
        raise

# === gRPC客户端类 ===

class CosyVoiceGRPCClient:
    def __init__(self, host="localhost", port=50000):
        self.host = host
        self.port = port
        self.qc16k_audio_bytes = None
        self.load_qc16k_audio()
        
    def load_qc16k_audio(self):
        """加载qc16k音频文件并转换为server端期望的格式"""
        qc16k_path = "/app/assets/qc16k.wav"
        try:
            # 加载为float32格式（server端期望）
            qc16k_audio = load_wav_for_server(qc16k_path, target_sr=16000)
            self.qc16k_audio_bytes = convert_audio_ndarray_to_bytes_for_server(qc16k_audio)
            logger.info(f"✅ qc16k音色加载成功: {qc16k_path}")
        except Exception as e:
            logger.error(f"❌ qc16k音色加载失败: {e}")
            self.qc16k_audio_bytes = None

    def construct_qc16k_request(self, text: str, stream: bool = False, speed: float = 1.0, format: str = ""):
        """构建qc16k请求"""
        if self.qc16k_audio_bytes is None:
            raise HTTPException(status_code=500, detail="qc16k音色未加载成功")
            
        request = cosyvoice_pb2.Request()
        request.tts_text = text
        request.stream = stream
        request.speed = speed
        request.text_frontend = True
        request.format = format  # 空字符串表示使用默认的float32格式
        
        cross_lingual_request = request.cross_lingual_request
        cross_lingual_request.prompt_audio = self.qc16k_audio_bytes
        
        logger.info(f"构建qc16k请求: text_len={len(text)}, audio_bytes_len={len(self.qc16k_audio_bytes)}")
        return request

    async def synthesize_with_qc16k(self, text: str, stream: bool = False, speed: float = 1.0, format: str = ""):
        """使用qc16k音色合成语音"""
        async with aio.insecure_channel(f"{self.host}:{self.port}") as channel:
            stub = cosyvoice_pb2_grpc.CosyVoiceStub(channel)
            
            grpc_request = self.construct_qc16k_request(text, stream, speed, format)
            
            audio_chunks = []
            start_time = time.time()
            
            try:
                logger.info(f"🎤 开始使用qc16k音色合成: {text[:50]}...")
                
                async for response in stub.Inference(grpc_request):
                    if response.tts_audio:
                        audio_chunks.append(response.tts_audio)
                        logger.info(f"收到音频chunk: {len(response.tts_audio)} bytes")
                    
                total_time = time.time() - start_time
                logger.info(f"✅ qc16k音色合成完成，用时: {total_time:.3f}s, 共{len(audio_chunks)}个chunks")
                
                return audio_chunks
                
            except grpc.RpcError as e:
                logger.error(f"gRPC error: {e.code()}: {e.details()}")
                raise HTTPException(status_code=500, detail=f"gRPC error: {e.details()}")
            except Exception as e:
                logger.error(f"Processing failed: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

# 初始化gRPC客户端
grpc_client = CosyVoiceGRPCClient()

# === FastAPI接口 ===

@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "CosyVoice TTS Service with QC16K Voice - Utils.py完全修复版",
        "status": "running",
        "description": "根据utils.py完全修复的版本，使用正确的float32数据格式",
        "qc16k_audio_loaded": grpc_client.qc16k_audio_bytes is not None,
        "qc16k_audio_size": len(grpc_client.qc16k_audio_bytes) if grpc_client.qc16k_audio_bytes else 0,
    }

@app.get("/tts/status")
async def tts_status():
    """TTS状态检查 - 匹配网关接口"""
    return {"status": "healthy", "qc16k_loaded": grpc_client.qc16k_audio_bytes is not None}

@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "healthy", "qc16k_loaded": grpc_client.qc16k_audio_bytes is not None}

@app.post("/tts/synthesize")
async def synthesize(request: TTSRequest):
    """语音合成接口（使用qc16k音色）- 完全修复版"""
    try:
        # 获取音频chunks
        audio_chunks = await grpc_client.synthesize_with_qc16k(
            text=request.text,
            stream=request.stream,
            speed=request.speed,
            format=request.format
        )
        
        if not audio_chunks:
            raise HTTPException(status_code=500, detail="没有接收到音频数据")
        
        # 重建完整音频
        final_audio = reconstruct_audio_from_server_response(audio_chunks)
        
        # 创建临时文件
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir="/app/audio_output")
        temp_file.close()
        
        # 使用torchaudio保存（与async_cosyvoice源码一致）
        torchaudio.save(temp_file.name, final_audio, 24000)
        
        return FileResponse(
            temp_file.name,
            media_type="audio/wav",
            filename="qc16k_synthesis_final.wav"
        )
        
    except Exception as e:
        logger.error(f"合成失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/test")
async def test_qc16k():
    """测试qc16k音色合成"""
    try:
        audio_chunks = await grpc_client.synthesize_with_qc16k("你好，这是完全修复的qc16k音色测试，现在应该有声音了。", stream=False, speed=1.0)
        
        final_audio = reconstruct_audio_from_server_response(audio_chunks)
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir="/app/audio_output")
        temp_file.close()
        
        torchaudio.save(temp_file.name, final_audio, 24000)
        
        return FileResponse(
            temp_file.name,
            media_type="audio/wav",
            filename="qc16k_test_final.wav"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    root_path = os.getenv("FASTAPI_ROOT_PATH", "")
    uvicorn.run("client_fastapi:app", host="0.0.0.0", port=9000, reload=True, root_path=root_path)