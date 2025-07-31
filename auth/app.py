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
from urllib.parse import quote, unquote

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
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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

# 模板设置
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

def is_browser_request(request: Request) -> bool:
    """判断是否是浏览器请求"""
    accept_header = request.headers.get("accept", "")
    user_agent = request.headers.get("user-agent", "").lower()
    
    return (
        "text/html" in accept_header or 
        any(browser in user_agent for browser in ["mozilla", "chrome", "safari", "edge", "webkit"])
    )

def build_login_url(request: Request, original_path: Optional[str] = None) -> str:
    """构建登录URL"""
    forwarded_proto = request.headers.get("X-Forwarded-Proto", "https")
    forwarded_host = request.headers.get("X-Forwarded-Host", request.headers.get("host", "localhost:8443"))
    
    # 如果没有提供原始路径，从请求头获取
    if original_path is None:
        forwarded_uri = request.headers.get("X-Forwarded-Uri", "/")
        original_path = forwarded_uri
    
    # 检测无限循环：如果 original_path 中嵌套的 redirect 过多，直接重置为 dashboard
    if original_path and original_path.count('/auth/login?redirect=') > 2:  # 阈值可调整
        original_path = "/dashboard/#/"
    
    # 构建完整的原始 URL
    if original_path and original_path != "/":
        original_url = f"{forwarded_proto}://{forwarded_host}{original_path}"
    else:
        original_url = f"{forwarded_proto}://{forwarded_host}/"
    
    # URL编码redirect参数
    encoded_redirect = quote(original_url, safe='/:?#[]@!$&\'()*+,;=')
    login_url = f"{forwarded_proto}://{forwarded_host}/auth/login?redirect={encoded_redirect}"
    
    return login_url

# 🔥 Web 登录页面
@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, redirect: Optional[str] = None):
    """显示登录页面"""
    # 确保redirect参数被正确处理
    redirect_url = redirect or "/auth/dashboard"
    
    # 如果redirect是URL编码的，先解码
    try:
        redirect_url = unquote(redirect_url)
    except:
        pass
    
    logger.info(f"显示登录页面，重定向URL: {redirect_url}")
    
    return templates.TemplateResponse("login.html", {
        "request": request,
        "redirect_url": redirect_url
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
    
    # 调试日志
    logger.info(f"处理登录请求，用户: {username}, IP: {client_ip}, 重定向URL: {redirect_url}")
    
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
        
        logger.info(f"Web登录成功 - 用户: {username}, IP: {client_ip}, 重定向到: {redirect_url}")
        
        # 🔥 确保重定向URL格式正确
        if redirect_url and not redirect_url.startswith('http') and not redirect_url.startswith('/'):
            redirect_url = f"/{redirect_url}"
        
        # 重定向到目标页面并设置 Cookie
        response = RedirectResponse(url=redirect_url or "/auth/dashboard", status_code=status.HTTP_302_FOUND)
        
        # 🔥 修复Cookie设置 - 检测是否为本地开发环境
        host = request.headers.get("host", "")
        is_local_dev = "localhost" in host or "127.0.0.1" in host
        
        response.set_cookie(
            key="access_token",
            value=access_token,
            max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,  # Cookie 过期时间（秒）
            httponly=True,  # 防止 XSS 攻击
            secure=not is_local_dev,  # 🔥 本地开发时设为False，生产环境设为True
            samesite="lax",  # CSRF 保护
            path="/"  # 🔥 确保cookie在整个域名下有效
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
async def web_logout(request: Request):
    """Web 登出，清除 Cookie"""
    response = RedirectResponse(url=build_login_url(request, "/auth/dashboard"), status_code=status.HTTP_302_FOUND)
    response.delete_cookie("access_token", path="/")
    return response

# 🔥 用户仪表板页面
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    """用户仪表板页面"""
    logger.info(f"Dashboard访问请求，IP: {get_client_ip(request)}")
    
    # 检查认证
    token = get_token_from_cookie_or_header(request)
    if not token:
        logger.warning("Dashboard访问被拒绝：没有token")
        return RedirectResponse(url=build_login_url(request, "/auth/dashboard"))
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username != FIXED_USERNAME:
            logger.warning(f"Dashboard访问被拒绝：用户名不匹配 {username}")
            return RedirectResponse(url=build_login_url(request, "/auth/dashboard"))
        
        # 检查用户是否被禁用
        if fixed_api_user_entry.disabled:
            logger.warning("Dashboard访问被拒绝：用户已被禁用")
            return RedirectResponse(url=build_login_url(request, "/auth/dashboard"))
            
    except JWTError as e:
        logger.warning(f"Dashboard访问被拒绝：token无效 {str(e)}")
        return RedirectResponse(url=build_login_url(request, "/auth/dashboard"))
    
    logger.info(f"Dashboard访问成功，用户: {username}")
    
    # 显示仪表板
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "username": username,
        "services": [
            {"name": "STT 语音转文字", "url": "/stt/docs", "description": "语音转文字服务文档"},
            {"name": "TTS 文字转语音", "url": "/tts/docs", "description": "文字转语音服务文档"},
            {"name": "OCR 图像识别", "url": "/ocr/docs", "description": "光学字符识别服务文档"},
            {"name": "API Gateway", "url": "/api/docs", "description": "API网关服务文档"},
            {"name": "User Management", "url": "/user/docs", "description": "用户管理服务文档"},
            {"name": "Traefik Dashboard", "url": "/traefik/", "description": "反向代理管理面板"},
            {"name": "Milvus 向量数据库", "url": "/attu/", "description": "向量数据库管理界面"},
        ]
    })

# 🔥 Traefik ForwardAuth 验证端点 - 统一处理所有认证
@app.get("/validate")
async def validate_auth(request: Request, credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    """Traefik ForwardAuth 验证端点 - 处理所有服务的认证，包括 Traefik Dashboard"""
    # 记录请求详情用于调试
    forwarded_proto = request.headers.get("X-Forwarded-Proto", "https")
    forwarded_host = request.headers.get("X-Forwarded-Host", request.headers.get("host", "localhost:8443"))
    forwarded_uri = request.headers.get("X-Forwarded-Uri", "/")
    forwarded_method = request.headers.get("X-Forwarded-Method", "GET")
    
    logger.info(f"验证请求 - Method: {forwarded_method}, Proto: {forwarded_proto}, Host: {forwarded_host}, URI: {forwarded_uri}")
    
    try:
        token = get_token_from_cookie_or_header(request, credentials)
        
        if not token:
            logger.warning(f"认证失败：没有token，URI: {forwarded_uri}")
            
            # 🔥 判断是否是浏览器请求
            if is_browser_request(request):
                login_url = build_login_url(request, forwarded_uri)
                logger.info(f"浏览器请求重定向到: {login_url}")
                return RedirectResponse(url=login_url, status_code=302)
            
            # 对 API 请求返回 401 JSON
            login_url = build_login_url(request, forwarded_uri)
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
                
            logger.info(f"认证成功，用户: {username}, 访问: {forwarded_uri}")
                
        except JWTError as e:
            logger.warning(f"Token验证失败: {str(e)}, URI: {forwarded_uri}")
            
            # Token 无效，同样处理
            if is_browser_request(request):
                login_url = build_login_url(request, forwarded_uri)
                logger.info(f"Token无效，浏览器重定向到: {login_url}")
                return RedirectResponse(url=login_url, status_code=302)
            
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
        
        # 🔥 特殊处理：对于 Traefik Dashboard 的访问，可以添加额外的权限检查
        if forwarded_uri.startswith("/traefik"):
            logger.info(f"Traefik Dashboard 访问认证成功，用户: {username}")
            # 这里可以添加管理员权限检查
            # if not is_admin_user(username):
            #     raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")
        
        # 认证成功
        return JSONResponse(
            status_code=200,
            content={"message": "Authentication successful"},
            headers={
                "X-User": username,
                "X-User-Role": "admin" if forwarded_uri.startswith("/traefik") else "api_user",
                "X-Auth-Status": "success"
            }
        )
        
    except HTTPException as e:
        logger.warning(f"HTTP异常: {e.status_code} - {e.detail}, URI: {forwarded_uri}")
        
        # 对于其他 HTTP 异常，也要判断是否需要重定向
        if e.status_code in [401, 403]:
            if is_browser_request(request):
                login_url = build_login_url(request, forwarded_uri)
                return RedirectResponse(url=login_url, status_code=302)
        
        return JSONResponse(
            status_code=e.status_code,
            content={"detail": e.detail},
            headers={"WWW-Authenticate": "Bearer"} if e.status_code == 401 else {}
        )
    except Exception as e:
        logger.error(f"认证系统内部错误: {str(e)}, URI: {forwarded_uri}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal authentication error"}
        )

# 🔥 专门的管理员验证端点（如果需要更严格的权限控制）
@app.get("/validate-admin")
async def validate_admin_auth(request: Request, credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    """专门用于管理员权限验证的端点（如 Traefik Dashboard）"""
    # 基本认证检查（复用 validate_auth 的逻辑）
    auth_result = await validate_auth(request, credentials)
    
    # 如果基本认证失败，直接返回
    if auth_result.status_code != 200:
        return auth_result
    
    # 额外的管理员权限检查
    forwarded_uri = request.headers.get("X-Forwarded-Uri", "/")
    logger.info(f"管理员权限验证通过，URI: {forwarded_uri}")
    
    # 返回成功，添加管理员角色标识
    return JSONResponse(
        status_code=200,
        content={"message": "Admin authentication successful"},
        headers={
            "X-User": FIXED_USERNAME,
            "X-User-Role": "admin",
            "X-Auth-Status": "success",
            "X-Admin-Access": "granted"
        }
    )

# 根路径重定向到仪表板
@app.get("/")
async def root():
    """根路径重定向到仪表板"""
    return RedirectResponse(url="/auth/dashboard")

# 🔥 健康检查端点
@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "service": "authentication-service",
        "timestamp": datetime.utcnow().isoformat(),
        "auth_configured": AUTH_CONFIGURED,
        "lockout_enabled": ENABLE_LOGIN_IP_LOCKOUT
    }

# 🔥 调试端点 - 显示请求头信息（仅在开发环境使用）
@app.get("/debug-headers")
async def debug_headers(request: Request):
    """调试端点：显示所有请求头信息"""
    headers = dict(request.headers)
    return {
        "headers": headers,
        "client": str(request.client),
        "url": str(request.url)
    }

# === 保持原有的 API 端点 ===

# OAuth2 token 端点（用于 API 调用）
if AUTH_CONFIGURED:
    @app.post("/token", response_model=Token, tags=["Authentication"])
    async def login_for_access_token(request: Request, form_data: OAuth2PasswordRequestForm = Depends()):
        """API Token 登录端点"""
        client_ip = get_client_ip(request)
        
        logger.info(f"API Token登录请求，用户: {form_data.username}, IP: {client_ip}")
        
        # IP锁定检查
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
        
        # 验证用户名和密码
        login_successful = False
        if form_data.username == FIXED_USERNAME and verify_password(form_data.password, FIXED_PASSWORD_HASH):
            login_successful = True
        
        if not login_successful:
            logger.warning(f"API Token登录失败，用户: {form_data.username}, IP: {client_ip}")
            
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
                        raise HTTPException(
                            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                            detail=f"您的IP因尝试次数过多已被暂时锁定。请在 {LOGIN_LOCKOUT_DURATION_SECONDS // 60} 分钟后重试。"
                        )
            
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, 
                detail="用户名或密码错误", 
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        # 清理成功登录用户的失败记录
        if ENABLE_LOGIN_IP_LOCKOUT and client_ip and client_ip in login_failure_tracker:
            del login_failure_tracker[client_ip]

        # 获取用户对象
        user_obj = get_fixed_api_user(FIXED_USERNAME)
        user_name_for_token = FIXED_USERNAME 
        if user_obj and hasattr(user_obj, 'username') and user_obj.username:
             user_name_for_token = user_obj.username
        
        if not user_obj or (hasattr(user_obj, 'disabled') and user_obj.disabled):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="用户访问被禁用或用户不存在")
        
        # 创建访问令牌
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(data={"sub": user_name_for_token}, expires_delta=access_token_expires)
        
        logger.info(f"API Token登录成功，用户: {form_data.username}, IP: {client_ip}")
        return {"access_token": access_token, "token_type": "bearer"}

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