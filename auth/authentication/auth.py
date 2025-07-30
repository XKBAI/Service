import json
from datetime import datetime, timedelta
from typing import Optional, Dict

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

# --- 从配置文件加载配置 ---
try:
    with open("authentication/auth_config.json", "r") as f:
        config = json.load(f)
    SECRET_KEY = config["JWT_SECRET_KEY"]
    FIXED_USERNAME = config["FIXED_USERNAME"]
    FIXED_PASSWORD_HASH = config["FIXED_PASSWORD_HASH"]
except FileNotFoundError:
    raise RuntimeError("auth_config.json not found. Please create it with JWT_SECRET_KEY, FIXED_USERNAME, and FIXED_PASSWORD_HASH.")
except KeyError as e:
    raise RuntimeError(f"Missing key in auth_config.json: {e}. Please ensure all required keys are present.")


ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 1  # JWT令牌有效期，60分钟*24h*1天

# 初始化密码哈希上下文
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2PasswordBearer 用于FastAPI从请求头中提取Bearer Token
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token") # tokenUrl 指向获取令牌的路由

# --- Pydantic 模型 ---
class User(BaseModel):
    username: str = FIXED_USERNAME # 固定为配置中加载的用户名
    full_name: Optional[str] = "通用API访问者" # 可根据需要调整
    disabled: Optional[bool] = False

class UserInDB(User):
    hashed_password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

# 模拟的固定用户数据库条目（只有加载自配置的一个用户）
fixed_api_user_entry = UserInDB(
    username=FIXED_USERNAME,
    hashed_password=FIXED_PASSWORD_HASH,
    full_name="通用API访问者",
    disabled=False # 默认不禁用
)

def get_password_hash(password: str) -> str:
    """对密码进行哈希处理"""
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """验证明文密码和哈希密码是否匹配"""
    return pwd_context.verify(plain_password, hashed_password)

def get_fixed_api_user(username: str) -> Optional[UserInDB]:
    """获取固定API访问用户（如果用户名匹配）"""
    if username == FIXED_USERNAME:
        return fixed_api_user_entry
    return None

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """创建JWT Access Token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_api_user(token: str = Depends(oauth2_scheme)) -> User:
    """
    FastAPI依赖函数，用于验证JWT。
    此函数不区分客户端，只验证令牌是否由本服务颁发给 FIXED_USERNAME。
    如果令牌无效或用户不存在/被禁用，则抛出HTTPException。
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="无法验证凭据",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub") # JWT的 'sub' 字段存储用户名
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception

    # 令牌中的用户名必须是固定的 API 用户名
    if token_data.username != FIXED_USERNAME:
        raise credentials_exception # 令牌的主体不正确

    # 验证是否被禁用
    if fixed_api_user_entry.disabled:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="API访问已被禁用")
    
    # 鉴权成功，返回 User 模型
    return User(username=fixed_api_user_entry.username,
                full_name=fixed_api_user_entry.full_name,
                disabled=fixed_api_user_entry.disabled)