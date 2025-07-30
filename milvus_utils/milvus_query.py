# 检索milvus数据库
import os
import numpy as np
from pymilvus import MilvusClient

# --- 全局配置 ---
# MILVUS_DB_PATH = "database/keyword_groups_from_csv.db" # 旧的本地文件配置
MILVUS_HOST = "127.0.0.1"  # <--- 新增/修改: Milvus 服务器 IP
MILVUS_PORT = 19530        # <--- 新增/修改: Milvus 服务器端口
MILVUS_URI = f"tcp://{MILVUS_HOST}:{MILVUS_PORT}" # <--- 新增/修改
DB_NAME = "math"           # <--- 新增/修改: 目标数据库名称

TARGET_COLLECTION_NAME = "math_knowledge_data" # <--- 修改: 确认与写入时一致
DIM_VALUE = 1024
TOP_K_RESULTS = 5

# Embedding 服务设置
EMBEDDING_API_KEY = "ollama_or_any_non_empty_string"
EMBEDDING_MODEL_OVERRIDE = None 

# --- 从 embedding.py 导入嵌入函数 ---
try:
    from embedding import OpenAIEmbeddingFunction 
    print("成功从 embedding.py 导入 OpenAIEmbeddingFunction。")
    if EMBEDDING_MODEL_OVERRIDE:
        embedding_function_impl = OpenAIEmbeddingFunction(
            api_key=EMBEDDING_API_KEY, model=EMBEDDING_MODEL_OVERRIDE
        )
    else:
        embedding_function_impl = OpenAIEmbeddingFunction(api_key=EMBEDDING_API_KEY)
except ImportError:
    print("错误: 无法从 embedding.py 导入 OpenAIEmbeddingFunction。")
    class FallbackPlaceholderEmbeddingFunction: 
        def __init__(self, *args, **kwargs): print(f"警告: 使用备用占位符嵌入函数。维度: {DIM_VALUE}")
        def __call__(self, text_or_texts):
            is_single = isinstance(text_or_texts, str)
            texts = [text_or_texts] if is_single else text_or_texts
            return [np.random.rand(DIM_VALUE).tolist() for _ in texts]
    embedding_function_impl = FallbackPlaceholderEmbeddingFunction()
except Exception as e:
    print(f"实例化从 embedding.py 导入的 OpenAIEmbeddingFunction 时出错: {e}")
    class FallbackPlaceholderEmbeddingFunction:
        def __init__(self, *args, **kwargs): print(f"警告: 使用备用占位符嵌入函数。维度: {DIM_VALUE}")
        def __call__(self, text_or_texts):
            is_single = isinstance(text_or_texts, str)
            texts = [text_or_texts] if is_single else text_or_texts
            return [np.random.rand(DIM_VALUE).tolist() for _ in texts]
    embedding_function_impl = FallbackPlaceholderEmbeddingFunction()


# --- Milvus 查询函数 ---
def search_keywords_in_milvus(query_text: str, top_k: int = TOP_K_RESULTS):
    print(f"\n--- 开始在 Milvus 中搜索与 '{query_text}' 相关的关键词组 ---")
    try:
        # 使用 URI 和 DB_NAME 连接到 Milvus 服务端
        client = MilvusClient(uri=MILVUS_URI, db_name=DB_NAME) # <--- 修改
        print(f"成功连接到 Milvus 服务: {MILVUS_URI}, 数据库: {DB_NAME}")
    except Exception as e:
        print(f"连接到 Milvus 失败: {e}")
        return

    # 检查集合是否存在于当前数据库
    if not client.has_collection(TARGET_COLLECTION_NAME):
        print(f"错误：集合 '{TARGET_COLLECTION_NAME}' 在数据库 '{DB_NAME}' 中不存在。请先运行数据写入脚本。")
        client.close() # 关闭连接
        return
    
    try:
        print(f"正在加载集合 '{TARGET_COLLECTION_NAME}' 以备搜索...")
        client.load_collection(TARGET_COLLECTION_NAME)
        print(f"集合 '{TARGET_COLLECTION_NAME}' 加载完成。")
    except Exception as e:
        print(f"加载集合 '{TARGET_COLLECTION_NAME}' 失败: {e}")
        print("警告：加载集合失败，但仍将尝试搜索。")


    print(f"正在为查询 '{query_text}' 生成嵌入向量...")
    try:
        query_embedding_list = embedding_function_impl(query_text) 
        if not query_embedding_list or not query_embedding_list[0] or not isinstance(query_embedding_list[0], list) or len(query_embedding_list[0]) == 0:
            print(f"错误：未能为查询文本 '{query_text}' 生成有效的嵌入向量。返回: {query_embedding_list}")
            return
        query_vector = query_embedding_list[0] 
        if len(query_vector) != DIM_VALUE:
            print(f"错误：查询向量的维度 ({len(query_vector)}) 与集合维度 ({DIM_VALUE}) 不匹配。")
            return
        print("查询向量生成成功。")
    except Exception as e:
        print(f"为查询文本 '{query_text}' 生成嵌入时出错: {e}")
        return

    print(f"正在搜索 Top {top_k} 个最相似的结果...")
    try:
        results = client.search(
            collection_name=TARGET_COLLECTION_NAME,
            data=[query_vector],
            anns_field="keyword_group_vector",
            metric_type="L2",
            limit=top_k,
            output_fields=["keyword_group_text", "md_files_list_str", "id",
                           "video_title", "video_chapter", "level",
                           "video_description", "original_relative_path"],
            consistency_level="Strong"
        )
        print("搜索完成。")
    except Exception as e:
        print(f"Milvus 搜索时发生错误: {e}")
        client.close()
        return

    if not results or not results[0]:
        print("没有找到相似的关键词组。")
        client.close()
        return

    print(f"\n--- 搜索结果 (Top {top_k}) ---")
    for i, hit in enumerate(results[0]):
        entity = hit.get('entity', {}) # 获取实体对象
        print(f"\n结果 {i+1}:")
        print(f"  ID (Milvus): {hit.get('id', 'N/A')}")
        print(f"  匹配关键词组: {entity.get('keyword_group_text', 'N/A')}")
        print(f"  关联MD文件列表: {entity.get('md_files_list_str', 'N/A')}")
        print(f"  视频标题: {entity.get('video_title', 'N/A')}")
        print(f"  视频章节: {entity.get('video_chapter', 'N/A')}")
        print(f"  视频描述: {entity.get('video_description', 'N/A')}")
        print(f"  级别: {entity.get('level', 'N/A')}")
        print(f"  原始相对路径: {entity.get('original_relative_path', 'N/A')}")
        print(f"  距离/相似度得分: {hit.get('distance', 'N/A'):.4f}")
    
    client.close() # 查询完毕后关闭连接

# --- 主程序入口 ---
if __name__ == "__main__":
    while True:
        user_query = input(f"\n请输入你要搜索的关键词或短语 (Top {TOP_K_RESULTS} 结果, 输入 'exit' 退出): ")
        if user_query.lower() == 'exit':
            break
        if not user_query.strip():
            print("查询不能为空，请重新输入。")
            continue
        search_keywords_in_milvus(user_query, top_k=TOP_K_RESULTS)
    print("查询程序已退出。")