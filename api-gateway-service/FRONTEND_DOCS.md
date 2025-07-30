# AI Services Gateway - 前端开发文档

## 项目概述

AI Services Gateway 前端是一个基于原生JavaScript的聊天界面，集成了多种AI服务，包括LLM对话、OCR文字识别、TTS语音合成等功能。

## 技术栈

- **核心技术**: 原生JavaScript (ES6+), HTML5, CSS3
- **UI框架**: 无框架，纯原生实现
- **依赖库**: 
  - [marked.js](https://cdn.jsdelivr.net/npm/marked/marked.min.js) - Markdown渲染
  - [highlight.js](https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js) - 代码高亮
  - [FontAwesome](https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css) - 图标库

## 项目结构

```
web/
├── index.html              # 主页面
├── css/
│   └── styles.css          # 主样式文件
└── js/
    ├── config.js           # 配置文件
    ├── auth.js             # 认证管理
    ├── api.js              # API客户端
    ├── chat.js             # 聊天管理
    └── app.js              # 主应用入口
```

## 启动和部署

### 开发环境启动

1. **启动API网关**:
   ```bash
   cd /home/xkb/api_gateway
   python api.py
   ```

2. **启动Web服务器**:
   ```bash
   cd /home/xkb/api_gateway/web
   python3 -m http.server 8081 > /dev/null 2>&1 &
   ```

3. **访问应用**:
   在浏览器中访问 `http://localhost:8081`

### 配置文件

主要配置在 `/web/js/config.js` 中：

```javascript
const CONFIG = {
    API_BASE_URL: 'http://localhost:60443',  // API网关地址
    DEFAULT_CREDENTIALS: {
        username: 'xkbAI',
        password: 'XuekuibangAI@2025'
    },
    // ... 其他配置
};
```

## 核心功能模块

### 1. 认证系统 (auth.js)

**主要功能**:
- JWT token管理
- 登录/登出处理
- 自动token过期检查

**核心类**:
- `AuthManager`: 认证管理器
- `LoginForm`: 登录表单处理

### 2. API客户端 (api.js)

**主要功能**:
- 统一API请求处理
- 自动添加认证头
- 用户管理API
- 聊天相关API
- OCR处理API
- TTS语音合成API

**核心方法**:
- `makeRequest()`: 基础API请求
- `getAllUsers()`: 获取用户列表
- `addChat()`: 创建聊天
- `processFileForChat()`: 文件处理（OCR）
- `synthesizeText()`: TTS语音合成

### 3. 聊天管理 (chat.js)

**主要功能**:
- 聊天消息管理
- 流式响应处理
- 文件上传处理
- Markdown渲染
- 用户界面交互

**核心类**:
- `ChatManager`: 聊天管理器

**重要特性**:
- 实时流式显示LLM回复
- 非阻塞文件处理
- Markdown格式渲染
- OCR异步处理和轮询

### 4. 应用入口 (app.js)

**主要功能**:
- 应用初始化
- 用户下拉菜单管理
- 全局状态协调

## 已实现的功能

### ✅ 完成的功能

1. **用户认证系统**
   - JWT token登录
   - 自动过期检查
   - 登出功能

2. **聊天界面**
   - 消息发送和接收
   - 聊天历史记录
   - 用户切换
   - 新建聊天

3. **流式传输**
   - LLM实时响应显示
   - 支持webui.md格式的流式数据
   - Markdown实时渲染

4. **文件处理**
   - 图片/PDF上传
   - OCR文字识别（异步处理）
   - 非阻塞用户界面
   - 处理进度指示

5. **Markdown渲染**
   - 完整Markdown语法支持
   - 代码块高亮
   - 表格、列表等格式

6. **TTS语音合成**
   - 文本转语音
   - 音频播放控制

7. **用户体验优化**
   - Toast通知系统
   - 加载状态指示
   - 响应式设计
   - 文件处理进度显示

### 🔧 技术实现细节

1. **流式传输处理**
   - 支持Server-Sent Events格式
   - 实时chunk显示
   - 完整Markdown重新渲染

2. **OCR异步处理**
   - 2秒间隔轮询
   - 最大60次尝试
   - 状态检查: `processing` → `completed`
   - 错误处理和超时保护

3. **文件上传流程**
   - 立即显示用户消息
   - 后台异步OCR处理
   - 处理完成后更新消息内容
   - 与用户输入合并作为完整提示词

4. **CORS配置**
   - API网关已添加CORS中间件
   - 支持跨域请求

## 配置说明

### API端点配置 (config.js)

```javascript
ENDPOINTS: {
    LOGIN: '/token',
    USER: {
        GET_ALL_USERS: '/user/get_all_users/',
        ADD_CHAT: '/user/add_chat/',
        // ... 其他端点
    },
    LLM: {
        CHAT_COMPLETIONS: '/llm/chat/completions',
        GET_CHAT_TITLE: '/llm/get_chat_title'
    },
    OCR: {
        PROCESS: '/olmocr/process',
        GET_RESULTS: '/olmocr/results/'
    },
    TTS: {
        SYNTHESIZE: '/tts/synthesize'
    }
}
```

### 文件上传限制

- 最大文件大小: 10MB
- 支持格式: PDF, PNG, JPEG, JPG, GIF, BMP, WebP

### UI设置

- 消息打字速度: 30ms/字符
- Toast通知持续时间: 3秒
- OCR轮询间隔: 2秒

## 开发注意事项

### 1. API调用模式

所有API调用都通过`APIClient`类进行，自动处理：
- 认证token添加
- 错误处理
- 响应状态检查

### 2. 流式数据格式

后端流式数据应符合以下格式：
```json
{"type": "chat", "name": "generate_question_tutorial", "content": "文本片段..."}
{"type": "citation_content", "name": "...", "content": [{"jpg_path": "...", "title": "...", "url": "..."}]}
{"type": "citation", "name": "...", "content": {"citations": [{"1": "文献ID"}]}}
{"type": "pdf", "name": "pdf_info", "content": {"status": "success", "filename": "...", "download_url": "..."}}
```

### 3. OCR处理流程

1. 用户上传文件 → 调用 `/olmocr/process`
2. 获得 `{job_id, status}` 响应
3. 轮询 `/olmocr/results/{job_id}` 
4. 直到 `status === 'completed'`
5. 提取 `text` 字段

### 4. 错误处理

- 网络错误自动重试
- API错误显示Toast通知
- 认证失败自动跳转登录

## 故障排除

### 常见问题

1. **CORS错误**
   - 确保API网关CORS中间件已启用
   - 检查API_BASE_URL配置

2. **认证失败**
   - 检查用户名密码配置
   - 验证token是否过期

3. **文件上传失败**
   - 检查文件大小和格式限制
   - 查看浏览器控制台错误信息

4. **OCR处理超时**
   - 检查后端OCR服务状态
   - 查看轮询日志

### 调试技巧

1. **开启浏览器开发者工具**
   - Network标签查看API请求
   - Console标签查看错误信息

2. **API调试**
   - 使用curl测试API端点
   - 检查请求头和响应格式

3. **OCR调试**
   - 查看控制台OCR轮询日志
   - 手工测试OCR API端点

## 待优化项目

1. **性能优化**
   - 添加请求缓存
   - 图片懒加载
   - 消息虚拟滚动

2. **用户体验**
   - 离线支持
   - 暗黑模式
   - 快捷键支持

3. **功能扩展**
   - 多文件批量上传
   - 导出聊天记录
   - 自定义主题

## 版本信息

- 当前版本: 1.4.1
- 最后更新: 2025年6月7日
- 维护者: Claude Code Assistant

---

*本文档记录了AI Services Gateway前端的开发和配置信息，用于项目维护和后续开发参考。*