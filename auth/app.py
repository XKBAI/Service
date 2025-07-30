from fastapi import FastAPI, Depends, HTTPException, status, Request, Form, Cookie
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, OAuth2PasswordRequestForm
from fastapi.templating import Jinja2Templates
from typing import Optional
import logging
from jose import JWTError, jwt
from datetime import datetime, timedelta
from pydantic import BaseModel
import os
import uvicorn

# 导入你现有的认证模块
from authentication.auth import (
    SECRET_KEY, 
    ALGORITHM, 
    FIXED_USERNAME, 
    FIXED_PASSWORD_HASH,
    fixed_api_user_entry,
    get_current_api_user, 
    User,
    Token,
    verify_password,
    create_access_token,
    get_fixed_api_user,
    ACCESS_TOKEN_EXPIRE_MINUTES
)

# 设置日志
logger = logging.getLogger(__name__)

# 创建 HTTPBearer 实例
security = HTTPBearer(auto_error=False)

# 登录失败跟踪相关配置
AUTH_CONFIGURED = True
ENABLE_LOGIN_IP_LOCKOUT = True
LOGIN_ATTEMPTS_LIMIT = 5
LOGIN_LOCKOUT_WINDOW_SECONDS = 300
LOGIN_LOCKOUT_DURATION_SECONDS = 900

# 内存存储登录失败跟踪
login_failure_tracker = {}
locked_out_ips = {}

app = FastAPI(
    title="Authentication Service",
    description="用户认证和授权服务",
    version="1.0.0",
    root_path=os.getenv("FASTAPI_ROOT_PATH", "")
)

# 模板设置（你需要创建 templates 目录）
templates = Jinja2Templates(directory="templates")

def get_client_ip(request: Request) -> str:
    """获取客户端真实IP地址"""
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip.strip()
    return request.client.host if request.client else "unknown"

def get_token_from_cookie_or_header(request: Request, credentials: Optional[HTTPAuthorizationCredentials] = None) -> Optional[str]:
    """从 Cookie 或 Authorization 头部获取 token"""
    # 优先从 Authorization 头部获取
    if credentials:
        return credentials.credentials
    
    # 从请求头获取（Traefik 转发的）
    auth_header = request.headers.get("authorization")
    if auth_header and auth_header.startswith("Bearer "):
        return auth_header.split(" ")[1]
    
    # 从 Cookie 获取
    token = request.cookies.get("access_token")
    if token:
        return token
    
    return None

# 🔥 Web 登录页面
@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, redirect: Optional[str] = None):
    """显示登录页面"""
    return templates.TemplateResponse("login.html", {
        "request": request,
        "redirect_url": redirect or "/auth/dashboard"
    })

# 🔥 Web 登录处理
@app.post("/login", response_class=HTMLResponse)
async def web_login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    redirect_url: str = Form(default="/auth/dashboard")
):
    """处理 Web 登录表单"""
    client_ip = get_client_ip(request)
    
    try:
        # IP锁定检查
        if ENABLE_LOGIN_IP_LOCKOUT and client_ip:
            if client_ip in locked_out_ips:
                lock_expiry_time = locked_out_ips[client_ip]
                if datetime.now() < lock_expiry_time:
                    return templates.TemplateResponse("login.html", {
                        "request": request,
                        "error": f"您的IP因尝试次数过多已被暂时锁定。请在 {LOGIN_LOCKOUT_DURATION_SECONDS // 60} 分钟后重试。",
                        "redirect_url": redirect_url
                    })
                else:
                    del locked_out_ips[client_ip]
                    if client_ip in login_failure_tracker:
                        del login_failure_tracker[client_ip]
        
        # 验证用户名和密码
        login_successful = False
        if username == FIXED_USERNAME and verify_password(password, FIXED_PASSWORD_HASH):
            login_successful = True
        
        if not login_successful:
            logger.warning(f"Web登录失败 - 用户: {username}, IP: {client_ip}")
            
            # 处理登录失败的IP跟踪
            if ENABLE_LOGIN_IP_LOCKOUT and client_ip:
                current_time = datetime.now()
                
                if client_ip not in login_failure_tracker or \
                   (current_time - login_failure_tracker[client_ip]["window_start"]).total_seconds() > LOGIN_LOCKOUT_WINDOW_SECONDS:
                    login_failure_tracker[client_ip] = {"count": 1, "window_start": current_time}
                else:
                    login_failure_tracker[client_ip]["count"] += 1
                    
                    if login_failure_tracker[client_ip]["count"] >= LOGIN_ATTEMPTS_LIMIT:
                        lock_until = current_time + timedelta(seconds=LOGIN_LOCKOUT_DURATION_SECONDS)
                        locked_out_ips[client_ip] = lock_until
                        del login_failure_tracker[client_ip]
                        return templates.TemplateResponse("login.html", {
                            "request": request,
                            "error": f"您的IP因尝试次数过多已被暂时锁定。请在 {LOGIN_LOCKOUT_DURATION_SECONDS // 60} 分钟后重试。",
                            "redirect_url": redirect_url
                        })
            
            return templates.TemplateResponse("login.html", {
                "request": request,
                "error": "用户名或密码错误",
                "redirect_url": redirect_url
            })
        
        # 登录成功，清理失败记录
        if ENABLE_LOGIN_IP_LOCKOUT and client_ip and client_ip in login_failure_tracker:
            del login_failure_tracker[client_ip]
        
        # 创建访问令牌
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(data={"sub": username}, expires_delta=access_token_expires)
        
        logger.info(f"Web登录成功 - 用户: {username}, IP: {client_ip}")
        
        # 🔥 重定向到目标页面并设置 Cookie
        response = RedirectResponse(url=redirect_url, status_code=status.HTTP_302_FOUND)
        response.set_cookie(
            key="access_token",
            value=access_token,
            max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,  # Cookie 过期时间（秒）
            httponly=True,  # 防止 XSS 攻击
            secure=True,    # 只在 HTTPS 下传输
            samesite="lax"  # CSRF 保护
        )
        return response
        
    except Exception as e:
        logger.error(f"Web登录错误: {str(e)}")
        return templates.TemplateResponse("login.html", {
            "request": request,
            "error": "登录服务内部错误，请稍后重试",
            "redirect_url": redirect_url
        })

# 🔥 Web 登出
@app.get("/logout")
async def web_logout():
    """Web 登出，清除 Cookie"""
    response = RedirectResponse(url="/auth/login", status_code=status.HTTP_302_FOUND)
    response.delete_cookie("access_token")
    return response

# 🔥 用户仪表板页面
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    """用户仪表板页面"""
    # 检查认证
    token = get_token_from_cookie_or_header(request)
    if not token:
        return RedirectResponse(url="/auth/login?redirect=/auth/dashboard")
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username != FIXED_USERNAME:
            return RedirectResponse(url="/auth/login?redirect=/auth/dashboard")
    except JWTError:
        return RedirectResponse(url="/auth/login?redirect=/auth/dashboard")
    
    # 显示仪表板
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "username": username,
        "services": [
            {"name": "STT 语音转文字", "url": "/stt/docs", "description": "语音转文字服务文档"},
            {"name": "TTS 文字转语音", "url": "/tts/docs", "description": "文字转语音服务文档"},
            {"name": "OCR 图像识别", "url": "/ocr/docs", "description": "光学字符识别服务文档"},
            {"name": "API Gateway", "url": "/api/docs", "description": "API网关服务文档"},
            {"name": "Traefik Dashboard", "url": "/traefik/", "description": "反向代理管理面板"},
            {"name": "Milvus 向量数据库", "url": "/attu/", "description": "向量数据库管理界面"},
        ]
    })

# 🔥 Traefik ForwardAuth 验证端点
# 🔥 Traefik ForwardAuth 验证端点
@app.get("/validate")
async def validate_auth(request: Request, credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    """Traefik ForwardAuth 验证端点"""
    try:
        token = get_token_from_cookie_or_header(request, credentials)
        
        if not token:
            # 🔥 判断是否是浏览器请求
            accept_header = request.headers.get("accept", "")
            user_agent = request.headers.get("user-agent", "").lower()
            
            # 检查是否是浏览器请求（而不是 API 请求）
            is_browser_request = (
                "text/html" in accept_header or 
                any(browser in user_agent for browser in ["mozilla", "chrome", "safari", "edge"])
            )
            
            # 构建登录 URL
            forwarded_proto = request.headers.get("X-Forwarded-Proto", "https")
            forwarded_host = request.headers.get("X-Forwarded-Host", request.headers.get("host", "localhost"))
            forwarded_uri = request.headers.get("X-Forwarded-Uri", "/")
            
            # 构建完整的原始 URL
            original_url = f"{forwarded_proto}://{forwarded_host}{forwarded_uri}"
            login_url = f"{forwarded_proto}://{forwarded_host}/auth/login?redirect={original_url}"
            
            # 🔥 对浏览器请求返回 302 重定向
            if is_browser_request:
                return RedirectResponse(url=login_url, status_code=302)
            
            # 对 API 请求返回 401 JSON
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Authentication required"},
                headers={
                    "WWW-Authenticate": "Bearer",
                    "Location": login_url
                }
            )
        
        # 验证 token
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username = payload.get("sub")
            if username != FIXED_USERNAME:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid user")
            
            if fixed_api_user_entry.disabled:
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User disabled")
                
        except JWTError:
            # Token 无效，同样处理
            accept_header = request.headers.get("accept", "")
            user_agent = request.headers.get("user-agent", "").lower()
            is_browser_request = (
                "text/html" in accept_header or 
                any(browser in user_agent for browser in ["mozilla", "chrome", "safari", "edge"])
            )
            
            forwarded_proto = request.headers.get("X-Forwarded-Proto", "https")
            forwarded_host = request.headers.get("X-Forwarded-Host", request.headers.get("host", "localhost"))
            forwarded_uri = request.headers.get("X-Forwarded-Uri", "/")
            original_url = f"{forwarded_proto}://{forwarded_host}{forwarded_uri}"
            login_url = f"{forwarded_proto}://{forwarded_host}/auth/login?redirect={original_url}"
            
            if is_browser_request:
                return RedirectResponse(url=login_url, status_code=302)
            
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
        
        # 认证成功
        return JSONResponse(
            status_code=200,
            content={"message": "Authentication successful"},
            headers={
                "X-User": username,
                "X-User-Role": "api_user",
                "X-Auth-Status": "success"
            }
        )
        
    except HTTPException as e:
        # 对于其他 HTTP 异常，也要判断是否需要重定向
        if e.status_code in [401, 403]:
            accept_header = request.headers.get("accept", "")
            user_agent = request.headers.get("user-agent", "").lower()
            is_browser_request = (
                "text/html" in accept_header or 
                any(browser in user_agent for browser in ["mozilla", "chrome", "safari", "edge"])
            )
            
            if is_browser_request:
                forwarded_proto = request.headers.get("X-Forwarded-Proto", "https")
                forwarded_host = request.headers.get("X-Forwarded-Host", request.headers.get("host", "localhost"))
                forwarded_uri = request.headers.get("X-Forwarded-Uri", "/")
                original_url = f"{forwarded_proto}://{forwarded_host}{forwarded_uri}"
                login_url = f"{forwarded_proto}://{forwarded_host}/auth/login?redirect={original_url}"
                return RedirectResponse(url=login_url, status_code=302)
        
        return JSONResponse(
            status_code=e.status_code,
            content={"detail": e.detail},
            headers={"WWW-Authenticate": "Bearer"} if e.status_code == 401 else {}
        )
    except Exception as e:
        logging.error(f"Authentication error: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal authentication error"}
        )
        
# 根路径重定向到仪表板
@app.get("/")
async def root():
    return RedirectResponse(url="/auth/dashboard")

# === 保持你原有的 API 端点 ===

# OAuth2 token 端点（用于 API 调用）
if AUTH_CONFIGURED:
    @app.post("/token", response_model=Token, tags=["Authentication"])
    async def login_for_access_token(request: Request, form_data: OAuth2PasswordRequestForm = Depends()):
        """API Token 登录端点（原有逻辑保持不变）"""
        # ... 你原有的登录逻辑代码 ...
        client_ip = get_client_ip(request)
        
        if ENABLE_LOGIN_IP_LOCKOUT and client_ip:
            if client_ip in locked_out_ips:
                lock_expiry_time = locked_out_ips[client_ip]
                if datetime.now() < lock_expiry_time:
                    logger.warning(f"IP {client_ip} 因登录失败次数过多而被锁定")
                    raise HTTPException(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS, 
                        detail=f"您的IP因尝试次数过多已被暂时锁定。请在 {LOGIN_LOCKOUT_DURATION_SECONDS // 60} 分钟后重试。"
                    )
                else:
                    del locked_out_ips[client_ip]
                    if client_ip in login_failure_tracker: 
                        del login_failure_tracker[client_ip]
        
        logger.debug(f"用户 {form_data.username} 尝试登录，IP: {client_ip}")
        
        login_successful = False
        if form_data.username == FIXED_USERNAME and verify_password(form_data.password, FIXED_PASSWORD_HASH):
            login_successful = True
        
        if not login_successful:
            logger.warning(f"用户 {form_data.username} 认证失败，IP: {client_ip}")
            if ENABLE_LOGIN_IP_LOCKOUT and client_ip:
                current_time = datetime.now()
                if client_ip not in login_failure_tracker or \
                   (current_time - login_failure_tracker[client_ip]["window_start"]).total_seconds() > LOGIN_LOCKOUT_WINDOW_SECONDS:
                    login_failure_tracker[client_ip] = {"count": 1, "window_start": current_time}
                else:
                    login_failure_tracker[client_ip]["count"] += 1
                    if login_failure_tracker[client_ip]["count"] >= LOGIN_ATTEMPTS_LIMIT:
                        lock_until = current_time + timedelta(seconds=LOGIN_LOCKOUT_DURATION_SECONDS)
                        locked_out_ips[client_ip] = lock_until
                        del login_failure_tracker[client_ip] 
                        raise HTTPException(
                            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                            detail=f"您的IP因尝试次数过多已被暂时锁定。请在 {LOGIN_LOCKOUT_DURATION_SECONDS // 60} 分钟后重试。"
                        )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, 
                detail="用户名或密码错误", 
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        if ENABLE_LOGIN_IP_LOCKOUT and client_ip and client_ip in login_failure_tracker:
            del login_failure_tracker[client_ip]

        user_obj = get_fixed_api_user(FIXED_USERNAME)
        user_name_for_token = FIXED_USERNAME 
        if user_obj and hasattr(user_obj, 'username') and user_obj.username:
             user_name_for_token = user_obj.username
        
        if not user_obj or (hasattr(user_obj, 'disabled') and user_obj.disabled):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="用户访问被禁用或用户不存在")
        
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(data={"sub": user_name_for_token}, expires_delta=access_token_expires)
        
        logger.info(f"用户 '{form_data.username}' 登录成功，IP: {client_ip}")
        return {"access_token": access_token, "token_type": "bearer"}

# 其他现有端点...
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "authentication-service",
        "timestamp": datetime.utcnow().isoformat(),
        "auth_configured": AUTH_CONFIGURED,
        "lockout_enabled": ENABLE_LOGIN_IP_LOCKOUT
    }

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    root_path = os.getenv("FASTAPI_ROOT_PATH", "")
    uvicorn.run(
        "app:app", 
        host="0.0.0.0", 
        port=9000,
        reload=True, 
        root_path=root_path
    )