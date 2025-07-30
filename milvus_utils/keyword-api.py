from openai import OpenAI

# 设置DeepSeek API密钥和端点
api_key = "sk-OSyEFmRhfr2pwr0K7E62BRZhoohKQ7Yum8044E1Kg9IW21Ul"
base_url = "http://192.168.2.4:3000/v1"

# 初始化OpenAI客户端
client = OpenAI(api_key=api_key, base_url=base_url)

# 调用DeepSeek API
try:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
        ],
        stream=False
    )
    print("API调用成功！")
    print("响应内容:", response.choices[0].message.content)
except Exception as e:
    print("API调用失败:", e)
