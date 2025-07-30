from passlib.context import CryptContext
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
# 将 "您的实际公共密码" 替换为您想要设置的明文密码
hashed_password = pwd_context.hash("XuekuibangAI@2025")
print(hashed_password)