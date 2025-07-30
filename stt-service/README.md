# Faster-Whisper STT 微服务

基于 faster-whisper 的高性能语音转录微服务，支持 GPU 加速和大规模并发处理。

## 🚀 特性

- ⚡ **高性能**: 基于 `app2.py` - 支持20个并发任务
- 🎯 **模型优化**: 使用 large-v3 模型，识别精度高
- 💾 **内存优化**: 直接处理音频数据，避免临时文件
- 🔥 **模型预热**: 减少首次请求延迟
- 📊 **完整监控**: 健康检查和性能监控
- 🐳 **容器化**: 生产级 Docker 配置

## 📦 文件结构

```
faster-whisper/
├── app2.py              # 🔥 主服务文件（高性能版本）
├── Dockerfile          # 容器构建配置
├── requirements.txt    # Python依赖
├── .dockerignore       # 构建忽略文件
├── large-v3/           # 本地模型文件
└── README.md           # 本文档
```

## 🛠️ 部署方式

### 方式1: 使用上级目录的 docker-compose.yml（推荐）

```bash
# 在 /home/xkb2/Desktop/QY 目录下
cd /home/xkb2/Desktop/QY

# 启动 STT 服务
docker-compose up -d faster-whisper-stt

# 查看日志
docker-compose logs -f faster-whisper-stt

# 停止服务
docker-compose down
```

### 方式2: 单独构建运行

```bash
cd faster-whisper

# 构建镜像
docker build -t faster-whisper-stt:latest .

# 运行容器
docker run -d \
  --name faster-whisper-stt \
  --gpus all \
  -p 57001:57001 \
  -v $(pwd)/large-v3:/tmp/faster_whisper/models--Systran--faster-whisper-large-v3/snapshots/main:ro \
  faster-whisper-stt:latest
```

## 🔧 API 使用

### 健康检查
```bash
curl http://localhost:57001/stt/health
```

### 音频转录
```bash
curl -X POST "http://localhost:57001/stt/transcribe/" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@audio.mp3" \
  -F "language=zh" \
  -F "beam_size=5"
```

### 清理GPU内存
```bash
curl -X POST "http://localhost:57001/stt/clear-memory"
```

## ⚙️ 配置说明

### 环境变量
- `MODEL_SIZE`: 模型大小，默认 `large-v3`
- `CUDA_VISIBLE_DEVICES`: GPU设备ID，默认 `0`

### 模型挂载
- 本地模型: `./large-v3` → `/tmp/faster_whisper/models--Systran--faster-whisper-large-v3/snapshots/main`
- 缓存目录: `./models/whisper` → `/tmp/faster_whisper`

### 资源限制
- 内存: 8GB
- CPU: 4核
- GPU: 1张显卡

## 📈 性能指标

| 指标 | 值 | 说明 |
|------|---|------|
| 并发处理 | 20个任务 | 同时处理的最大请求数 |
| 支持格式 | MP3, WAV, FLAC, OGG, M4A | 音频文件格式 |
| 响应时间 | 1-3秒 | 取决于音频长度 |
| 启动时间 | 60-90秒 | 包含模型加载时间 |

## 🔍 监控和调试

### 查看容器状态
```bash
# 容器运行状态
docker ps | grep faster-whisper

# 资源使用情况
docker stats faster-whisper-stt-service

# GPU使用情况
nvidia-smi
```

### 日志查看
```bash
# 实时日志
docker logs -f faster-whisper-stt-service

# 最近100行日志
docker logs --tail=100 faster-whisper-stt-service
```

## 🛠️ 故障排除

### 常见问题

1. **启动慢**
   - 原因: 需要加载3GB+的模型文件
   - 解决: 耐心等待，查看日志确认加载进度

2. **内存不足**
   - 原因: 模型太大或并发数过高
   - 解决: 减少并发数或使用更小的模型

3. **GPU不可用**
   - 原因: CUDA环境问题
   - 解决: 检查nvidia-docker安装，服务会自动降级到CPU

4. **模型文件缺失**
   - 原因: `large-v3` 目录不存在
   - 解决: 确保模型文件已下载到正确位置

### 性能优化

1. **预热模型**: 启动后发送一次测试请求
2. **批量处理**: 合并多个短音频文件
3. **缓存结果**: 对相同音频避免重复转录

## 🔐 安全配置

- ✅ 非root用户运行 (appuser:1000)
- ✅ 最小化基础镜像
- ✅ 只暴露必要端口
- ✅ 健康检查机制
- ✅ 日志轮转配置

## 📝 开发说明

此微服务基于 `app2.py` 构建，具有以下技术特点：

- **异步处理**: 使用 asyncio 和 ThreadPoolExecutor
- **内存优化**: 直接处理 BytesIO，避免临时文件
- **并发控制**: 信号量限制并发数，防止资源耗尽
- **模型管理**: 支持预热和内存清理
- **错误处理**: 完整的异常捕获和响应