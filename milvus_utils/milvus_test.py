# 构建milvus数据库
import os
import json
from pathlib import Path
import numpy as np
from pymilvus import MilvusClient, DataType
from tqdm import tqdm

# --- 全局配置 ---
JSON_DATA_SOURCE_PATH = "/home/xkb2/Desktop/QY/all_md_keywords_recursive.json"  # 你的 JSON 文件名
MILVUS_DB_PATH = "database/keyword_groups_from_json.db"  # Milvus DB 路径
KEYWORD_COLLECTION_NAME = "keyword_groups_collection"  # 集合名
DIM_VALUE = 1024  # !!! 向量维度，必须与你的嵌入模型输出一致 !!!

# Embedding 服务设置
EMBEDDING_API_KEY = "ollama_or_any_non_empty_string"
EMBEDDING_MODEL_OVERRIDE = None  # 或 "your_specific_model_for_embedding_py"

# --- 从 embedding.py 导入嵌入函数 (与之前相同) ---
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

# --- 主程序 ---
if __name__ == "__main__":
    # --- 步骤 1: 直接读取 JSON 文件 ---
    print(f"\n--- 开始从 JSON 文件 '{JSON_DATA_SOURCE_PATH}' 读取数据并存入 Milvus ---")

    if not os.path.exists(JSON_DATA_SOURCE_PATH):
        print(f"错误: 数据源 JSON 文件 '{JSON_DATA_SOURCE_PATH}' 未找到。")
        exit(1)

    # --- 步骤 2: 初始化 Milvus ---
    db_parent_dir = os.path.dirname(MILVUS_DB_PATH)
    if db_parent_dir and not os.path.exists(db_parent_dir):
        os.makedirs(db_parent_dir, exist_ok=True)
    try:
        milvus_client = MilvusClient(uri=MILVUS_DB_PATH)
        print(f"成功连接到 Milvus (或初始化 Milvus Lite数据库文件): {MILVUS_DB_PATH}")
    except Exception as e:
        print(f"连接或初始化 Milvus 失败: {e}")
        exit(1)

    # --- 步骤 3: 定义 Milvus Schema ---
    schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(field_name="keyword_group_text", datatype=DataType.VARCHAR, max_length=65535)  # 存储关键词组原文
    schema.add_field(field_name="md_files_list_str", datatype=DataType.VARCHAR, max_length=65535)  # 存储关联的 MD 文件信息
    schema.add_field(field_name="keyword_group_vector", datatype=DataType.FLOAT_VECTOR, dim=DIM_VALUE)  # 关键词组的嵌入向量

    # --- 步骤 4: 创建 Milvus 集合 ---
    if milvus_client.has_collection(collection_name=KEYWORD_COLLECTION_NAME):
        print(f"Milvus 集合 '{KEYWORD_COLLECTION_NAME}' 已存在。将删除并重建。")
        milvus_client.drop_collection(collection_name=KEYWORD_COLLECTION_NAME)
    
    try:
        print(f"正在创建 Milvus 集合: {KEYWORD_COLLECTION_NAME}，维度: {DIM_VALUE}")
        milvus_client.create_collection(collection_name=KEYWORD_COLLECTION_NAME, schema=schema, consistency_level="Strong")
        print(f"Milvus 集合 '{KEYWORD_COLLECTION_NAME}' 创建成功。")
    except Exception as e:
        print(f"创建 Milvus 集合 '{KEYWORD_COLLECTION_NAME}' 失败: {e}")
        exit(1)
        
    # --- 步骤 5: 创建索引 ---
    print(f"为集合 '{KEYWORD_COLLECTION_NAME}' 在字段 'keyword_group_vector' 上配置 AUTOINDEX。")
    index_params = milvus_client.prepare_index_params()
    index_params.add_index(field_name="keyword_group_vector", index_type="AUTOINDEX", metric_type="L2")
    try:
        milvus_client.create_index(KEYWORD_COLLECTION_NAME, index_params)
        print("索引创建/配置成功。")
    except Exception as e:
        print(f"创建索引失败: {e}")

    # --- 步骤 6: 读取 JSON，生成嵌入并准备插入数据 ---
    data_to_insert_milvus = []
    try:
        with open(JSON_DATA_SOURCE_PATH, 'r', encoding='utf-8') as json_file:
            json_data = json.load(json_file)
            print(f"从 JSON 文件 '{JSON_DATA_SOURCE_PATH}' 读取到 {len(json_data)} 条数据。")

            for item in tqdm(json_data, desc="处理JSON行并生成嵌入"):
                keywords_list = item.get("keywords", [])
                if not keywords_list:
                    continue
                keyword_group = "，".join(keywords_list)  # 用中文顿号拼接

                filename = item.get("filename", "")
                relative_path = item.get("relative_path", "")

                try:
                    embedding_list = embedding_function_impl(keyword_group)
                    if not embedding_list or not embedding_list[0] or not isinstance(embedding_list[0], list) or len(embedding_list[0]) == 0:
                        print(f"  警告：关键词组 '{keyword_group[:50]}...' 的嵌入结果无效，已跳过。")
                        continue

                    keyword_group_embedding = embedding_list[0]
                    if len(keyword_group_embedding) != DIM_VALUE:
                        print(f"  错误：关键词组 '{keyword_group[:50]}...' 的嵌入维度 ({len(keyword_group_embedding)}) 与 Milvus 集合维度 ({DIM_VALUE}) 不匹配，已跳过。")
                        continue

                    data_to_insert_milvus.append({
                        "keyword_group_text": keyword_group,    # 存储关键词组文本
                        "md_files_list_str": f"{filename} | {relative_path}",   # 存储文件名和路径
                        "keyword_group_vector": keyword_group_embedding  # 存储关键词组的向量
                    })
                except Exception as e:
                    print(f"  处理关键词组 '{keyword_group[:50]}...' (调用嵌入函数) 时出错: {e}")

    except FileNotFoundError:
        print(f"错误: JSON文件 '{JSON_DATA_SOURCE_PATH}' 在尝试读取时未找到。")
        exit(1)
    except Exception as e:
        print(f"读取或处理JSON文件 '{JSON_DATA_SOURCE_PATH}' 时发生错误: {e}")
        exit(1)

    # --- 步骤 7: 批量插入数据到 Milvus ---
    if data_to_insert_milvus:
        print(f"\n准备将 {len(data_to_insert_milvus)} 条数据从 JSON 插入 Milvus...")
        try:
            insert_response = milvus_client.insert(
                collection_name=KEYWORD_COLLECTION_NAME,
                data=data_to_insert_milvus
            )
            inserted_count = 0
            if isinstance(insert_response, dict) and 'ids' in insert_response:
                inserted_count = len(insert_response['ids'])
            elif isinstance(insert_response, list):
                inserted_count = len(insert_response)
            else:
                print(f"Milvus 插入操作已提交。返回类型: {type(insert_response)}")
                inserted_count = len(data_to_insert_milvus)

            if inserted_count > 0:
                print(f"成功提交了 {inserted_count} 条数据到 Milvus。")
            else:
                print(f"Milvus 插入操作已提交，但未能确认插入数量（返回: {insert_response}）。")
           
            print(f"正在执行 Flush 操作: {KEYWORD_COLLECTION_NAME}...")
            milvus_client.flush(collection_name=KEYWORD_COLLECTION_NAME) 
            print("Flush 操作完成。")
            stats_after_flush = milvus_client.get_collection_stats(collection_name=KEYWORD_COLLECTION_NAME)
            print(f"Flush后集合统计: {stats_after_flush}")

            print(f"正在加载集合 '{KEYWORD_COLLECTION_NAME}' 到内存...")
            milvus_client.load_collection(KEYWORD_COLLECTION_NAME)
            print(f"集合 '{KEYWORD_COLLECTION_NAME}' 加载完成。")

        except Exception as e:
            print(f"插入数据到 Milvus 或后续操作时发生错误: {e}")
    else:
        print("没有从 JSON 读取到有效数据，或处理嵌入时出错，无法插入 Milvus。")

    print("\nMilvus 处理完成。脚本执行结束。")