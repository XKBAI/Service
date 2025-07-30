新建一个文件夹web, 写一个前端,先使用xkbAI和hash前的密码XuekuibangAI@2025登录,然后进入   │
│   界面后左侧是聊天记录,右侧是chat,右下方是输入框,类似openai webui,发送消息自动增聊天,右   │
│   上角选择用户uid,每条对话消息的气泡框要有按钮可以转语音,然后输入框要有加号,可以点击加号  │
│   上传文件(pdf或图片)上传文件后发送聊天是调用ocr识别然后拼到用户的prompt然后一起发给llm   │
│   chat completion接口       

输入输出需要对齐
## 基本格式

```
Content-Type: text/event-stream; charset=utf-8
```

每个数据块格式：
```json
{"type": "类型", "name": "名称", "content": "内容"}
```

## 响应阶段和类型

### 1. 问题解答教学阶段
```json
{
  "type": "chat",
  "name": "generate_question_tutorial", 
  "content": "问题解答的文本片段..."
}
```

### 2. 引用内容信息
```json
{
  "type": "citation_content",
  "name": "generate_question_tutorial",
  "content": [
    {
      "jpg_path": "http://localhost:60443/get_images/output_coreproblem/filename.jpg",
      "title": "original_relative_path",
      "url": "url"
    }
  ]
}
```

### 3. 原始引用数据
```json
{
  "type": "citation",
  "name": "generate_question_tutorial",
  "content": {
    "citations": [
      {"1": "文献ID1"}, 
      {"2": "文献ID2"}, 
      {"3": "文献ID3"}
    ]
  }
}
```

### 4. 知识点讲义阶段
```json
{
  "type": "chat",
  "name": "generate_knowledge_point_tutorial",
  "content": "知识点讲义的文本片段..."
}
```

对应的引用信息：
```json
{
  "type": "citation_content",
  "name": "generate_knowledge_point_tutorial",
  "content": [/* 引用内容数组 */]
}
```

```json
{
  "type": "citation",
  "name": "generate_knowledge_point_tutorial", 
  "content": {
    "citations": [{"1": "文献ID"}, {"2": "文献ID"}]
  }
}
```

### 5. 母题讲义阶段
```json
{
  "type": "chat", 
  "name": "generate_core_question_tutorial",
  "content": "母题讲义的文本片段..."
}
```

对应的引用信息：
```json
{
  "type": "citation_content",
  "name": "generate_core_question_tutorial",
  "content": [/* 引用内容数组 */]
}
```

```json
{
  "type": "citation",
  "name": "generate_core_question_tutorial",
  "content": {
    "citations": [{"1": "文献ID"}, {"2": "文献ID"}]
  }
}
```

### 6. 结论讲义阶段
```json
{
  "type": "chat",
  "name": "generate_conclusion", 
  "content": "结论讲义的文本片段..."
}
```

对应的引用信息：
```json
{
  "type": "citation_content",
  "name": "generate_conclusion",
  "content": [/* 引用内容数组 */]
}
```

```json
{
  "type": "citation",
  "name": "generate_conclusion",
  "content": {
    "citations": [{"1": "文献ID"}, {"2": "文献ID"}]
  }
}
```

### 7. PDF 生成完成
```json
{
  "type": "pdf",
  "name": "pdf_info",
  "content": {
    "status": "success",
    "filename": "生成的PDF文件名.pdf",
    "download_url": "PDF下载链接",
    "file_size": "文件大小信息"
  }
}
```




不要每次切换到一个用户就新增聊天,切换用户先get 这个用户的chat session  │
│   list然后进第一个聊天即可,用户手动点创建聊天再创建新聊天,每一个聊天记   │
│   录要有一个按钮,可以删掉这个聊天记录,检查音频播放按钮在播放后是否可以   │
│   暂停播放以及暂停后是否可以继续播放,播放直到一次播放完不再循环播放      │
╰────────────────────────────────────────────────────────────────────────

删除聊天记录使用这个
curl -X 'DELETE' \
  'http://localhost:58001/del_chat/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "chat_session_id": "2c9f246e428a44a982b5072d1e7e24cd"
}'

然后另外就是我发送一个hi过去，聊天记录里面显示了两个hi，然后chat completion也是针对两个hi进行响应的，你怎么处理的发送请求和user输入这一步，明显有错误

AI Chat Gateway大标题改成学魁榜AI

用户id如果有超长的，就只打印前6个字符，中文就前3个？反正保持打印的字符长度一致，超长的部分省略号一下


stt接口请求bug
chat.js:1590 
 STT processing error: Error: Failed to transcribe audio: 500 - {"detail":"转录过程中出错: 'zh-CN' is not a valid language code (accepted language codes: af, am, ar, as, az, ba, be, bg, bn, bo, br, bs, ca, cs, cy, da, de, el, en, es, et, eu, fa, fi, fo, fr, gl, gu, ha, haw, he, hi, hr, ht, hu, hy, id, is, it, ja, jw, ka, kk, km, kn, ko, la, lb, ln, lo, lt, lv, mg, mi, mk, ml, mn, mr, ms, mt, my, ne, nl, nn, no, oc, pa, pl, ps, pt, ro, ru, sa, sd, si, sk, sl, sn, so, sq, sr, su, sv, sw, ta, te, tg, th, tk, tl, tr, tt, uk, ur, uz, vi, yi, yo, zh, yue)"}
    at APIClient.transcribeAudio (api.js:393:23)
    at async ChatManager.processRecording (chat.js:1579:31)
    at async mediaRecorder.onstop (chat.js:1511:21)

清空对话这个没用，删掉