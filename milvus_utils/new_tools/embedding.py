# 测试embedding
from openai import OpenAI # 确保导入 OpenAI 客户端
import numpy as np # 用于可能查看维度

# --- 你的 OpenAIEmbeddingFunction (未修改版本) ---
class OpenAIEmbeddingFunction:
    def __init__(self, url="http://127.0.0.1:9997/v1", api_key='none', model="bge-large-zh-v1.5"): # model 参数有默认值
        self.client = OpenAI(
            base_url = url, # 硬编码的 base_url
            api_key = api_key # 将传入的 api_key 用于 OpenAI 客户端初始化
        )
        self.model = model # 使用传入的 model，如果没传则用默认值
        print(f"OpenAIEmbeddingFunction (用户指定版本) 已初始化。")
        print(f"  模型: {self.model}")
        print(f"  Base URL: {self.client.base_url}") # 注意：这里会显示你传入的 base_url
        print(f"  API Key (用于客户端): {api_key}")


    def __call__(self, input_texts: str | list[str]): # 添加了类型提示以便清晰
        # OpenAI API 通常期望输入是一个字符串列表，即使只有一个字符串。
        # 如果是单个字符串，将其包装在列表中。
        actual_input = [input_texts] if isinstance(input_texts, str) else input_texts
        
        # 基本的输入验证
        if not isinstance(actual_input, list) or not all(isinstance(i, str) for i in actual_input) or not actual_input:
            print(f"错误：嵌入函数的输入必须是非空字符串或非空字符串列表。收到: {input_texts}")
            # 返回与预期输出结构一致的空列表，表示失败
            return [[] for _ in range(len(actual_input) if isinstance(actual_input, list) else 1)]

        print(f"  正在调用嵌入 API，输入文本数量: {len(actual_input)}, 模型: {self.model}...")
        try:
            response = self.client.embeddings.create(
                input=actual_input,
                model=self.model
            )
            print("  嵌入 API 调用成功。")
            embeddings = [embedding.embedding for embedding in response.data]
            return embeddings
        except Exception as e:
            print(f"  调用嵌入 API 时出错 (模型: {self.model}): {e}")
            # 返回与输入数量相匹配的空列表，表示失败
            return [[] for _ in actual_input]

# --- 主测试逻辑 ---
if __name__ == "__main__":
    # 配置
    # 对于本地服务如Ollama，api_key可能不被验证，但OpenAI客户端通常需要一个非空字符串。
    test_api_key = "ollama" 
    # 确保这个模型名称能被你的 http://127.0.0.1:11434/v1 服务识别
    # 你可能需要移除 ":latest" 如果你的服务不处理这个标签
    test_model_name = "bge-large-zh-v1.5" 
    expected_dimension = 1024 # bge-large-zh-v1.5 的维度

    print("--- 开始嵌入功能测试 ---")
    
    # 1. 实例化嵌入函数
    try:
        embedder = OpenAIEmbeddingFunction(api_key=test_api_key, model=test_model_name)
    except Exception as e:
        print(f"创建 OpenAIEmbeddingFunction 实例失败: {e}")
        exit()

    # 2. 测试单个字符串
    print("\n--- 测试1: 单个字符串 ---")
    single_text = "你好，这是一个测试。"
    print(f"输入: \"{single_text}\"")
    single_embedding_list = embedder(single_text)

    if single_embedding_list and single_embedding_list[0] and isinstance(single_embedding_list[0], list):
        embedding_vector = single_embedding_list[0]
        print(f"成功获取嵌入！")
        print(f"  维度: {len(embedding_vector)}")
        print(f"  向量 (前5个值): {embedding_vector[:5]}...")
        if len(embedding_vector) != expected_dimension:
            print(f"  警告: 嵌入维度 ({len(embedding_vector)})与预期 ({expected_dimension}) 不符！")
    else:
        print(f"未能获取单个字符串的嵌入，或返回格式不正确。返回: {single_embedding_list}")

    # 3. 测试字符串列表
    print("\n--- 测试2: 字符串列表 ---")
    list_of_texts = ["第一句话。", "这是第二句话，它更长一些。", "最后一句。"]
    print(f"输入列表 (共 {len(list_of_texts)} 条): {list_of_texts}")
    list_embeddings = embedder(list_of_texts)

    if list_embeddings and len(list_embeddings) == len(list_of_texts) and all(isinstance(e, list) and e for e in list_embeddings):
        print("成功获取列表中所有文本的嵌入！")
        for i, (text, vec) in enumerate(zip(list_of_texts, list_embeddings)):
            print(f"  文本 {i+1}: \"{text}\" -> 维度: {len(vec)}, 前5值: {vec[:5]}...")
            if len(vec) != expected_dimension:
                 print(f"    警告: 文本 {i+1} 的嵌入维度 ({len(vec)})与预期 ({expected_dimension}) 不符！")
    else:
        print(f"未能获取字符串列表的嵌入，或部分失败，或返回格式不正确。返回: {list_embeddings}")
        
    # 4. 测试空字符串或不当输入 (可选，但有助于检查健壮性)
    print("\n--- 测试3: 不当输入 ---")
    empty_text = ""
    print(f"输入: 空字符串 \"{empty_text}\"")
    empty_embedding_list = embedder(empty_text) # 你的API可能不允许空字符串
    print(f"空字符串嵌入返回: {empty_embedding_list}")

    invalid_input = [123, "文本"]
    print(f"输入: 混合类型列表 {invalid_input}")
    # invalid_embedding_list = embedder(invalid_input) # 最好在调用前做类型检查
    # print(f"混合类型列表嵌入返回: {invalid_embedding_list}")


    print("\n--- 嵌入功能测试结束 ---")