import secrets # Python 3.6+ 推荐使用 secrets 模块来生成安全的随机数
import base64

# 生成一个随机的字节序列
# 32字节（256位）是 HMAC-SHA256 算法的推荐长度，非常安全
secret_bytes = secrets.token_bytes(32)

# 将字节序列编码为 Base64 URL 安全的字符串
# 这样可以直接在 JSON 文件和环境变量中使用，避免特殊字符问题
jwt_secret_key = base64.urlsafe_b64encode(secret_bytes).decode('utf-8')

print("您的 JWT 密钥 (请复制并妥善保管):")
print(jwt_secret_key)