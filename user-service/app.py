from fastapi import FastAPI, HTTPException, Depends # Depends 仍然可能被其他地方使用，暂时保留
# from fastapi.security.api_key import APIKeyHeader # 修改: 移除 APIKeyHeader
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import mysql.connector
from datetime import datetime
import json
import uuid
import requests
import uvicorn
import os

app = FastAPI()

# 修改: 移除 API Key 设置与依赖项
# API_KEY = "xkbai"
# api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# 修改: 移除 API key 校验依赖函数
# async def verify_api_key(api_key: str = Depends(api_key_header)):
#     """
#     校验请求头中是否携带正确的 API Key
#     Curl 示例:
#     curl -X GET "http://localhost:8000/get_all_users/" -H "X-API-Key: apikey" # 旧示例
#     """
#     if api_key != API_KEY:
#         raise HTTPException(status_code=403, detail="无效的 API key")
#     return api_key

# 数据库连接函数
def get_db_connection():
    """
    建立与 MySQL 数据库的连接
    """
    return mysql.connector.connect(
        host=os.getenv("DB_HOST", "localhost"),
        user=os.getenv("DB_USER", "root"),
        password=os.getenv("DB_PASSWORD", "qwe123asd"),
        database=os.getenv("DB_NAME", "CHAT"),
        auth_plugin='mysql_native_password'
    )

# ===================== 数据模型定义 =====================

class Message(BaseModel):
    """消息基础模型，包含角色和内容"""
    role: str
    content: str

class UserBase(BaseModel):
    """用户基础模型，包含用户ID"""
    user_id: str

class ChatContentRequest(BaseModel):
    """
    聊天内容请求模型，用于添加新聊天
    Curl 示例:
    curl -X POST "http://localhost:58000/add_chat/?user_id=user123" \
         -H "Content-Type: application/json" \
         -d '{"chat_time": "2025-03-01 10:00:00", "chat_title": "新聊天", "messages": [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "您好，有什么可以帮助您的？"}]}'
    """
    chat_time: str
    messages: list # 在实际使用中，这里应该是 List[Message] 或者 List[Dict[str,str]] 以便更好地校验
    chat_title: Optional[str] = '新聊天'

class ChatSessionIdRequest(BaseModel):
    """
    聊天会话ID请求模型，用于根据 ID 操作聊天会话
    Curl 示例:
    curl -X DELETE "http://localhost:58000/del_chat/" \
         -H "Content-Type: application/json" \
         -d '{"chat_session_id": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"}'
    """
    chat_session_id: str

class ChatSessionUpdateRequest(BaseModel):
    """
    聊天会话更新请求模型，用于编辑或更新聊天内容
    Curl 示例:
    curl -X PUT "http://localhost:58000/edit_chat/" \
         -H "Content-Type: application/json" \
         -d '{"chat_session_id": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", "chat_time": "2025-03-01 11:00:00", "chat_title": "更新后的标题", "messages": [{"role": "user", "content": "更新后的消息"}]}'
    """
    chat_session_id: str
    chat_time: str
    chat_title: Optional[str] = None
    messages: List[Message]

# ===================== 响应数据模型 =====================

class SuccessResponse(BaseModel):
    """操作成功响应模型"""
    message: str

class SuccessChangeTitleResponse(BaseModel):
    """操作成功响应模型，返回更新后的标题"""
    message: str
    title: str

class ChatSessionIdResponse(BaseModel):
    """
    聊天会话 ID 响应模型，用于返回新创建的聊天会话 ID
    """
    chat_session_id: str

class ChatSessionResponse(BaseModel):
    """
    聊天会话响应模型，用于返回聊天会话详情
    """
    chat_session_id: str
    chat_time: datetime
    chat_title: str
    messages: list # 同样，这里可以是 List[Message] 或 List[Dict[str,str]]

class UserListResponse(BaseModel):
    """
    用户列表响应模型，返回所有用户的 ID 列表
    Curl 示例:
    curl -X GET "http://localhost:58000/get_all_users/"
    """
    users: list

class ChatSessionInfo(BaseModel):
    """
    聊天会话信息模型，用于返回会话 ID、标题和时间
    """
    chat_session_id: str
    chat_title: str
    chat_time: datetime

# ===================== 用户相关接口 =====================

@app.get("/get_all_users/", response_model=UserListResponse) # 修改: 移除 dependencies
async def get_all_users():
    """
    获取所有用户ID
    Curl 示例:
    curl -X GET "http://localhost:58000/get_all_users/"
    """
    db = get_db_connection()
    cursor = db.cursor()
    try:
        query = "SELECT user_id FROM users"
        cursor.execute(query)
        rows = cursor.fetchall()
        users = [row[0] for row in rows]
        return {"users": users}
    except mysql.connector.Error as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        if db.is_connected():
            cursor.close()
            db.close()

@app.post("/add_user/", response_model=SuccessResponse) # 修改: 移除 dependencies
async def add_user(user: UserBase):
    """
    添加新用户
    Curl 示例:
    curl -X POST "http://localhost:58000/add_user/" \
         -H "Content-Type: application/json" \
         -d '{"user_id": "user123"}'
    """
    db = get_db_connection()
    cursor = db.cursor()
    try:
        query = "SELECT user_id FROM users WHERE user_id = %s"
        cursor.execute(query, (user.user_id,))
        existing_user = cursor.fetchone()
        if existing_user:
            return {"message": f"已有用户 {user.user_id}"}
        query = "INSERT INTO users (user_id) VALUES (%s)"
        cursor.execute(query, (user.user_id,))
        db.commit()
        return {"message": f"添加新用户 {user.user_id} 成功"}
    except mysql.connector.Error as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        if db.is_connected():
            cursor.close()
            db.close()

@app.delete("/del_user/", response_model=SuccessResponse) # 修改: 移除 dependencies
async def del_user(user: UserBase):
    """
    删除用户及其所有聊天记录
    Curl 示例:
    curl -X DELETE "http://localhost:58000/del_user/" \
         -H "Content-Type: application/json" \
         -d '{"user_id": "user123"}'
    """
    db = get_db_connection()
    cursor = db.cursor()
    try:
        query = "SELECT user_id FROM users WHERE user_id = %s"
        cursor.execute(query, (user.user_id,))
        existing_user = cursor.fetchone()
        if not existing_user:
            return {"message": f"不存在用户 {user.user_id}"}
        # 注意：此操作仅删除 users 表中的用户，如果 chat_history 表设置了外键且ON DELETE CASCADE，则关联聊天记录也会被删除
        # 否则，您需要在此处添加逻辑来显式删除 chat_history 表中该用户的记录
        query = "DELETE FROM users WHERE user_id = %s"
        cursor.execute(query, (user.user_id,))
        db.commit()
        return {"message": f"删除用户 {user.user_id} 成功"}
    except mysql.connector.Error as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        if db.is_connected():
            cursor.close()
            db.close()

@app.get("/get_user_chat/", response_model=List[ChatSessionInfo]) # 修改: 移除 dependencies
async def get_user_chat(user_id: str):
    """
    获取用户的所有聊天会话ID和标题
    Curl 示例:
    curl -X GET "http://localhost:58000/get_user_chat/?user_id=user123"
    """
    db = get_db_connection()
    cursor = db.cursor(dictionary=True)
    try:
        query = "SELECT user_id FROM users WHERE user_id = %s"
        cursor.execute(query, (user_id,))
        existing_user = cursor.fetchone()
        if not existing_user:
            raise HTTPException(status_code=404, detail=f"不存在用户 {user_id}")
        
        query = """
        SELECT chat_session_id, chat_title, chat_time
        FROM chat_history 
        WHERE user_id = %s 
        ORDER BY chat_time DESC
        """
        cursor.execute(query, (user_id,))
        rows = cursor.fetchall()
        chat_sessions=[]
        for row in rows:
            # 确保 chat_title 和 chat_time 有默认值，如果数据库中为 NULL
            current_chat_title = row['chat_title'] if row['chat_title'] is not None else '新聊天'
            current_chat_time_str = row['chat_time'].strftime('%Y-%m-%d %H:%M:%S') if row['chat_time'] is not None else datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            needs_update_in_db = False
            if row['chat_title'] is None:
                row['chat_title'] = current_chat_title # 使用上面确定的 current_chat_title
                needs_update_in_db = True
            if row['chat_time'] is None:
                row['chat_time'] = datetime.strptime(current_chat_time_str, '%Y-%m-%d %H:%M:%S')
                needs_update_in_db = True

            if needs_update_in_db:
                update_query = """
                UPDATE chat_history 
                SET chat_time = %s, chat_title = %s
                WHERE chat_session_id = %s
                """
                # 使用转换后的 chat_time (datetime 对象) 和确定的 chat_title
                cursor.execute(update_query, (row['chat_time'], row['chat_title'], row['chat_session_id']))
                # db.commit() # 在循环中 commit 可能影响性能，最好在循环外一次性 commit (如果适用)
                            # 但此处是修复数据，每次修复后提交是安全的

            chat_sessions.append(
                ChatSessionInfo(
                    chat_session_id=row['chat_session_id'],
                    chat_title=current_chat_title, # 使用处理后的值
                    chat_time=datetime.strptime(current_chat_time_str, '%Y-%m-%d %H:%M:%S') # 使用处理后的值
                ) 
            )
        db.commit() # 如果在循环中有更新，确保在这里提交
        return chat_sessions
    except mysql.connector.Error as e:
        db.rollback() # 如果发生错误，回滚所有可能的更新
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        if db.is_connected():
            cursor.close()
            db.close()

# ===================== 聊天相关接口 =====================

@app.post("/add_chat/", response_model=ChatSessionIdResponse) # 修改: 移除 dependencies
async def add_chat(user_id: str, chat_content: ChatContentRequest):
    """
    添加新聊天会话
    Curl 示例:
    curl -X POST "http://localhost:58000/add_chat/?user_id=user123" \
         -H "Content-Type: application/json" \
         -d '{"chat_time": "2025-03-01 10:00:00", "messages": [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "您好，有什么可以帮助您的？"}]}'
    """
    db = get_db_connection()
    cursor = db.cursor()
    try:
        query = "SELECT user_id FROM users WHERE user_id = %s"
        cursor.execute(query, (user_id,))
        existing_user = cursor.fetchone()
        if not existing_user:
            raise HTTPException(status_code=404, detail=f"用户 {user_id} 不存在")
            
        chat_session_id = uuid.uuid4().hex
        chat_time = chat_content.chat_time
        chat_title = chat_content.chat_title if chat_content.chat_title else '新聊天'
        
        # 确保 messages 是 dict 列表
        valid_messages = []
        if isinstance(chat_content.messages, list):
            for msg_item in chat_content.messages:
                if isinstance(msg_item, dict) and "role" in msg_item and "content" in msg_item:
                    valid_messages.append({"role": msg_item["role"], "content": msg_item["content"]})
                # 可以选择在这里处理无效的 message item，例如跳过或引发错误
        messages_json = json.dumps(valid_messages)

        query = """
        INSERT INTO chat_history 
        (chat_session_id, user_id, chat_time, chat_title, messages) 
        VALUES (%s, %s, %s, %s, %s)
        """
        cursor.execute(query, (chat_session_id, user_id, chat_time, chat_title, messages_json))
        db.commit()
        return {"chat_session_id": chat_session_id}
    except mysql.connector.Error as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        if db.is_connected():
            cursor.close()
            db.close()

@app.delete("/del_chat/", response_model=SuccessResponse) # 修改: 移除 dependencies
async def del_chat(chat_session: ChatSessionIdRequest):
    """
    删除聊天会话
    Curl 示例:
    curl -X DELETE "http://localhost:58000/del_chat/" \
         -H "Content-Type: application/json" \
         -d '{"chat_session_id": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"}'
    """
    db = get_db_connection()
    cursor = db.cursor()
    try:
        query = "SELECT chat_session_id FROM chat_history WHERE chat_session_id = %s"
        cursor.execute(query, (chat_session.chat_session_id,))
        existing_session = cursor.fetchone()
        if not existing_session:
            # 返回成功信息，即使记录不存在，以实现幂等性（或者可以改为返回404）
            return {"message": f"聊天记录 {chat_session.chat_session_id} 不存在或已删除"}
        query = "DELETE FROM chat_history WHERE chat_session_id = %s"
        cursor.execute(query, (chat_session.chat_session_id,))
        db.commit()
        return {"message": f"删除聊天记录 {chat_session.chat_session_id} 成功"}
    except mysql.connector.Error as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        if db.is_connected():
            cursor.close()
            db.close()

@app.put("/edit_chat/", response_model=SuccessResponse) # 修改: 移除 dependencies
async def edit_chat(chat_session: ChatSessionUpdateRequest):
    """
    完全替换聊天会话内容，可选更新标题
    Curl 示例:
    curl -X PUT "http://localhost:58000/edit_chat/" \
         -H "Content-Type: application/json" \
         -d '{"chat_session_id": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", "chat_time": "2025-03-01 11:00:00", "chat_title": "更新后的标题", "messages": [{"role": "user", "content": "更新后的消息"}]}'
    """
    db = get_db_connection()
    cursor = db.cursor()
    try:
        query = "SELECT chat_session_id FROM chat_history WHERE chat_session_id = %s"
        cursor.execute(query, (chat_session.chat_session_id,))
        existing_session = cursor.fetchone()
        if not existing_session:
            raise HTTPException(status_code=404, detail=f"不存在聊天记录 {chat_session.chat_session_id}")
        
        chat_time = chat_session.chat_time
        messages_json = json.dumps([{"role": msg.role, "content": msg.content} for msg in chat_session.messages])
        
        if chat_session.chat_title is not None:
            query = """
            UPDATE chat_history 
            SET chat_time = %s, chat_title = %s, messages = %s 
            WHERE chat_session_id = %s
            """
            cursor.execute(query, (chat_time, chat_session.chat_title, messages_json, chat_session.chat_session_id))
        else:
            # 如果 chat_title 未提供，则不更新它
            query = """
            UPDATE chat_history 
            SET chat_time = %s, messages = %s 
            WHERE chat_session_id = %s
            """
            cursor.execute(query, (chat_time, messages_json, chat_session.chat_session_id))
        db.commit()
        return {"message": "操作：edit; 聊天记录更新成功"}
    except mysql.connector.Error as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        if db.is_connected():
            cursor.close()
            db.close()

@app.put("/update_chat/", response_model=SuccessResponse) # 修改: 移除 dependencies
async def update_chat(request_payload: ChatSessionUpdateRequest): # Renamed 'request' to avoid conflict with FastAPI's Request object
    """
    追加聊天会话内容，可选更新标题
    Curl 示例:
    curl -X PUT "http://localhost:58000/update_chat/" \
         -H "Content-Type: application/json" \
         -d '{"chat_session_id": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", "chat_time": "2025-03-01 11:30:00", "chat_title": "可选的新标题", "messages": [{"role": "user", "content": "追加的消息"}]}'
    """
    db = get_db_connection()
    cursor = db.cursor(dictionary=True)
    try:
        query = "SELECT chat_session_id, messages, chat_title FROM chat_history WHERE chat_session_id = %s"
        cursor.execute(query, (request_payload.chat_session_id,))
        existing_session = cursor.fetchone()
        if not existing_session:
            raise HTTPException(status_code=404, detail=f"不存在聊天记录 {request_payload.chat_session_id}")
        
        existing_messages = json.loads(existing_session['messages']) if isinstance(existing_session['messages'], str) else (existing_session['messages'] or [])
        new_messages = [{"role": msg.role, "content": msg.content} for msg in request_payload.messages]
        existing_messages.extend(new_messages)
        messages_json = json.dumps(existing_messages)
        
        chat_time = request_payload.chat_time
        chat_title = existing_session['chat_title'] # 默认使用旧标题
        if request_payload.chat_title is not None: # 如果请求中提供了标题
            if request_payload.chat_title == '' and chat_title == '新聊天': # 如果提供空字符串且旧标题是"新聊天"，则保持"新聊天"
                chat_title = '新聊天'
            elif request_payload.chat_title != '': # 如果提供了非空字符串，则更新标题
                chat_title = request_payload.chat_title
            # 如果提供了None，则 chat_title 保持 existing_session['chat_title'] 不变

        query = """
        UPDATE chat_history 
        SET messages = %s, chat_time = %s, chat_title = %s 
        WHERE chat_session_id = %s
        """
        cursor.execute(query, (messages_json, chat_time, chat_title, request_payload.chat_session_id))
        db.commit()
        return {"message": "操作：update; 聊天记录更新成功"}
    except (mysql.connector.Error, json.JSONDecodeError) as e: # Catch potential JSON errors
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        if db.is_connected():
            cursor.close()
            db.close()

@app.get("/get_chat_by_session_id/", response_model=ChatSessionResponse) # 修改: 移除 dependencies
async def get_chat_by_session_id(chat_session_id: str):
    """
    根据会话ID获取聊天内容，包括标题
    Curl 示例:
    curl -X GET "http://localhost:58000/get_chat_by_session_id/?chat_session_id=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    """
    db = get_db_connection()
    cursor = db.cursor(dictionary=True)
    try:
        query = """
        SELECT chat_session_id, chat_time, chat_title, messages 
        FROM chat_history 
        WHERE chat_session_id = %s
        """
        cursor.execute(query, (chat_session_id,))
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail=f"不存在聊天记录 {chat_session_id}")
        
        messages = json.loads(row['messages']) if isinstance(row['messages'], str) else (row['messages'] or [])
        
        return {
            "chat_session_id": row['chat_session_id'],
            "chat_time": row['chat_time'] if row['chat_time'] else datetime.now(), # 提供默认时间
            "chat_title": row['chat_title'] if row['chat_title'] else "新聊天", # 提供默认标题
            "messages": messages
        }
    except (mysql.connector.Error, json.JSONDecodeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        if db.is_connected():
            cursor.close()
            db.close()

def get_title_from_external_service(messages: List[Dict[str, str]], old_title: Optional[str]) -> Optional[str]: # 修改函数名和参数类型
    """
    调用外部接口获取新的聊天标题。
    注意：此函数现在不使用 API_KEY。如果外部服务需要认证，需另外处理。
    """
    url = "http://192.168.2.3:61080/get_chat_title/" # 这是您代码中提供的外部URL
    # headers = {"X-API-Key": API_KEY} # 修改: 移除 API_KEY
    headers = {"Content-Type": "application/json"} # 通常POST JSON需要这个头

    # 构建符合外部服务预期的payload，它似乎需要messages列表
    # 原始代码将 old_title 作为一条 user message 追加，如果外部服务确实这样设计，则保留
    payload_messages = list(messages) # 创建副本以避免修改原始传入的列表
    if old_title: # 仅当old_title存在时才添加
         payload_messages.append({"role": "user", "content": f"请基于以上对话以及旧标题“{old_title}”生成一个新标题。"})
    else:
         payload_messages.append({"role": "user", "content": "请基于以上对话生成一个新标题。"})

    payload = {"messages": payload_messages} # 外部服务预期的payload格式

    try:
        # print(f"调用外部标题服务 URL: {url}, Payload: {json.dumps(payload)}") # 调试信息
        res = requests.post(url, json=payload, headers=headers, timeout=10) # 添加超时
        res.raise_for_status()
        
        # 尝试解析JSON响应，如果外部服务返回JSON {"title": "..."}
        try:
            response_data = res.json()
            title = response_data.get("title")
            if title is None:
                # 如果 "title" 键不存在，但请求成功，尝试直接使用文本内容 (兼容原始实现)
                title = res.text.strip().replace('"', '').replace("'", '').replace('\n', ' ')
        except json.JSONDecodeError:
            # 如果不是JSON，直接使用文本内容
            title = res.text.strip().replace('"', '').replace("'", '').replace('\n', ' ')
        
        # print(f"外部服务返回标题: {title}") # 调试信息
        return title if title else None # 返回None如果标题为空
    except requests.exceptions.RequestException as e:
        print(f"调用外部标题服务错误: {e}") # 应该使用 logger
        return None


@app.put("/update_chat_title/", response_model=SuccessChangeTitleResponse) # 修改: 移除 dependencies
async def update_chat_title(chat_session_payload: ChatSessionResponse): # Renamed to avoid confusion
    """
    自动更新聊天会话标题 (通过调用外部服务)
    Curl 示例:
    curl -X PUT "http://localhost:58000/update_chat_title/" \
         -H "Content-Type: application/json" \
         -d '{"chat_session_id": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", "chat_time": "2025-03-01T10:00:00", "chat_title": "旧的标题", "messages": [{"role": "user", "content": "消息内容"}, {"role": "assistant", "content": "回复内容"}]}'
    """
    # 确保 messages 是 List[Dict[str, str]] 类型
    messages_for_title_service = []
    if isinstance(chat_session_payload.messages, list):
        for msg in chat_session_payload.messages:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                messages_for_title_service.append({"role": msg["role"], "content": msg["content"]})
            elif hasattr(msg, 'role') and hasattr(msg, 'content'): # 如果是Pydantic模型实例
                messages_for_title_service.append({"role": msg.role, "content": msg.content})


    new_chat_title = get_title_from_external_service(messages_for_title_service, chat_session_payload.chat_title)
    
    if new_chat_title is None:
        raise HTTPException(status_code=500, detail="无法从外部服务获取新标题")

    db = get_db_connection()
    cursor = db.cursor()
    try:
        query = "SELECT chat_session_id FROM chat_history WHERE chat_session_id = %s"
        cursor.execute(query, (chat_session_payload.chat_session_id,))
        existing_session = cursor.fetchone()
        if not existing_session:
            raise HTTPException(status_code=404, detail=f"不存在聊天记录 {chat_session_payload.chat_session_id}")
        
        # 使用从 payload 传入的时间或当前时间
        chat_time_to_update = chat_session_payload.chat_time if chat_session_payload.chat_time else datetime.now()

        query = """
        UPDATE chat_history 
        SET chat_time = %s, chat_title = %s
        WHERE chat_session_id = %s
        """
        cursor.execute(query, (chat_time_to_update, new_chat_title, chat_session_payload.chat_session_id))
        db.commit()
        return {"message": f"成功修改聊天 {chat_session_payload.chat_session_id} 标题为 {new_chat_title}", "title": new_chat_title}
    except mysql.connector.Error as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        if db.is_connected():
            cursor.close()
            db.close()


@app.put("/edit_chat_title/", response_model=SuccessChangeTitleResponse) # 修改: 移除 dependencies
async def edit_chat_title(chat_session_info: ChatSessionInfo): # Renamed to avoid confusion
    """
    手动修改聊天会话标题
    Curl 示例:
    curl -X PUT "http://localhost:58000/edit_chat_title/" \
         -H "Content-Type: application/json" \
         -d '{"chat_session_id": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", "chat_title": "手动设置的新标题", "chat_time": "2025-03-01T10:00:00"}'
    """
    new_chat_title = chat_session_info.chat_title
    if not new_chat_title: # 不允许设置空标题
        raise HTTPException(status_code=400, detail="标题不能为空")

    db = get_db_connection()
    cursor = db.cursor()
    try:
        query = "SELECT chat_session_id FROM chat_history WHERE chat_session_id = %s"
        cursor.execute(query, (chat_session_info.chat_session_id,))
        existing_session = cursor.fetchone()
        if not existing_session:
            raise HTTPException(status_code=404, detail=f"不存在聊天记录 {chat_session_info.chat_session_id}")
        
        # 使用从 payload 传入的时间或当前时间
        chat_time_to_update = chat_session_info.chat_time if chat_session_info.chat_time else datetime.now()

        query = """
        UPDATE chat_history 
        SET chat_time = %s, chat_title = %s
        WHERE chat_session_id = %s
        """
        cursor.execute(query, (chat_time_to_update, new_chat_title, chat_session_info.chat_session_id))
        db.commit()
        return {"message": f"成功修改聊天 {chat_session_info.chat_session_id} 标题为 {new_chat_title}", "title": new_chat_title}
    except mysql.connector.Error as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        if db.is_connected():
            cursor.close()
            db.close()
        
@app.get("/health", tags=["USERAPI Info"])
async def health_check():
    return {"message": "OK"}



@app.get("/", tags=["USERAPI Info"])
async def get_root():
    return {"message": "OK"}


if __name__ == "__main__":
    root_path = os.getenv("FASTAPI_ROOT_PATH", "")
    uvicorn.run(
        "app:app", # 确保这里的 "service_user_db_api" 是您的文件名 (不含 .py)
        host="0.0.0.0",
        port=9000,
        root_path=root_path,
        log_level="info",
        reload=False,
    )