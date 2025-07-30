#!/bin/bash

# API网关服务启动脚本

set -e

echo "开始启动API网关服务..."

# 检查必要的目录
mkdir -p /app/shared /app/logs

# 设置权限
chmod -R 755 /app/shared /app/logs

# 等待后端服务启动
echo "等待后端服务启动..."
sleep 15

echo "启动API网关服务..."

# 启动服务前手动初始化HTTP客户端
echo "手动初始化HTTP客户端..."
python -c "
import asyncio
import httpx
import os
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 后端服务配置
BACKEND_SERVICES = {
    'llm': {
        'base_url': os.getenv('LLM_SERVICE_URL', 'http://192.168.2.3:61080'),
        'prefix': '/llm',
        'actual_health_endpoint': '/',
        'health_check_method': 'GET'
    },
    'stt': {
        'base_url': os.getenv('STT_SERVICE_URL', 'http://faster-whisper-stt:9000'),
        'prefix': '/stt',
        'actual_health_endpoint': '/stt/health',
        'health_check_method': 'GET'
    },
    'tts': {
        'base_url': os.getenv('TTS_SERVICE_URL', 'http://melotts-tts:9000'),
        'prefix': '/tts',
        'actual_health_endpoint': '/health',
        'health_check_method': 'GET'
    },
    'ocr': {
        'base_url': os.getenv('OCR_SERVICE_URL', 'http://olmocr-service:9000'),
        'prefix': '/olmocr',
        'actual_health_endpoint': '/',
        'health_check_method': 'GET'
    },
    'user': {
        'base_url': os.getenv('USER_SERVICE_URL', 'http://host.docker.internal:55003'),
        'prefix': '/user',
        'actual_health_endpoint': '/get_all_users/',
        'health_check_method': 'GET'
    },
    'md2pdf': {
        'base_url': os.getenv('MD2PDF_SERVICE_URL', 'http://md2pdf-service:9000'),
        'prefix': '/md2pdf',
        'actual_health_endpoint': '/',
        'health_check_method': 'GET'
    },
    'vlm': {
        'base_url': os.getenv('VLM_SERVICE_URL', 'http://vlm-service:9000'),
        'prefix': '/vlm',
        'actual_health_endpoint': '/vlm/health',
        'health_check_method': 'GET'
    }
}

async def init_clients():
    logger.info('API 网关启动中，正在初始化 HTTP 客户端...')
    global_clients = {}
    
    for service_name, config in BACKEND_SERVICES.items():
        global_clients[service_name] = httpx.AsyncClient(
            base_url=config['base_url'],
            timeout=httpx.Timeout(300.0, connect=10.0),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
            follow_redirects=False
        )
        logger.info(f'为服务 \"{service_name}\" 创建客户端，目标: {config[\"base_url\"]}')
    
    logger.info('所有 HTTP 客户端已初始化。')
    
    # 保存到全局作用域
    import pickle
    with open('/tmp/http_clients.pkl', 'wb') as f:
        pickle.dump(global_clients, f)
    
    logger.info('HTTP 客户端已保存到临时文件')

asyncio.run(init_clients())
"

# 启动服务 - 使用python api.py以启用SSL配置
exec python milvus_query_api.py 