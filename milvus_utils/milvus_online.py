# 构建milvus数据库（服务端模式，Attu可见）
import os
import json
import numpy as np
from pymilvus import MilvusClient, DataType
from tqdm import tqdm

# --- 全局配置 ---
# 1. 更新 JSON 文件路径
JSON_DATA_SOURCE_PATH = "/home/xkb2/Desktop/QY/json_utils/math_knowledge_data_en_keys_smart_updated.json" # <--- 修改这里
MILVUS_HOST = "127.0.0.1"
MILVUS_PORT = 19530
MILVUS_URI = f"tcp://{MILVUS_HOST}:{MILVUS_PORT}"
DB_NAME = "math"  # <--- 新增：定义目标数据库名称
KEYWORD_COLLECTION_NAME = "math_knowledge_data_v2"  # 建议用新集合名，或删除旧集合
DIM_VALUE = 1024  # !!! 向量维度，必须与你的嵌入模型输出一致 !!!

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

# --- 主程序 ---
if __name__ == "__main__":
    print(f"\n--- 开始从 JSON 文件 '{JSON_DATA_SOURCE_PATH}' 读取数据并存入 Milvus 服务端 ---")

    if not os.path.exists(JSON_DATA_SOURCE_PATH):
        print(f"错误: 数据源 JSON 文件 '{JSON_DATA_SOURCE_PATH}' 未找到。")
        exit(1)

    # --- 步骤 1: 连接 Milvus 服务端并确保目标数据库存在 ---
    try:
        # 首先，连接到 Milvus 服务（默认数据库，通常是 'default'）以检查/创建目标数据库
        print(f"正在连接 Milvus 服务 ({MILVUS_URI}) 以管理数据库...")
        admin_client = MilvusClient(uri=MILVUS_URI) # 默认连接到 'default' 数据库
        print("成功连接到 Milvus 服务 (用于数据库管理)。")

        # 检查目标数据库是否存在，如果不存在则创建
        if DB_NAME not in admin_client.list_databases():
            print(f"数据库 '{DB_NAME}' 不存在，正在创建...")
            admin_client.create_database(db_name=DB_NAME)
            print(f"数据库 '{DB_NAME}' 创建成功。")
        else:
            print(f"数据库 '{DB_NAME}' 已存在。")
        
        # admin_client 的任务完成，可以让它自动关闭或显式关闭 (如果需要长时间运行且不希望占用连接)
        # admin_client.close() # MilvusClient 通常在对象销毁时自动处理

        # 现在，创建用于后续操作的 MilvusClient，并指定连接到目标数据库 DB_NAME
        print(f"正在连接到 Milvus 数据库 '{DB_NAME}' ({MILVUS_URI})...")
        milvus_client = MilvusClient(uri=MILVUS_URI, db_name=DB_NAME)
        print(f"成功连接到 Milvus，当前操作数据库: '{DB_NAME}'")

    except Exception as e:
        print(f"连接 Milvus 或准备数据库 '{DB_NAME}' 失败: {e}")
        exit(1)

    # --- 步骤 2: 定义 Milvus Schema ---
    print("正在定义 Milvus schema...")
    schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False) # 动态字段通常不推荐用于生产
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(field_name="keyword_group_text", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="md_files_list_str", datatype=DataType.VARCHAR, max_length=65535) # 之前JSON中的 filename 和 relative_path 组合
    schema.add_field(field_name="keyword_group_vector", datatype=DataType.FLOAT_VECTOR, dim=DIM_VALUE)

    # 2.1 添加新的字段到 schema
    schema.add_field(field_name="video_title", datatype=DataType.VARCHAR, max_length=1024) # <--- 新增
    schema.add_field(field_name="video_chapter", datatype=DataType.VARCHAR, max_length=512)  # <--- 新增
    schema.add_field(field_name="video_description", datatype=DataType.VARCHAR, max_length=2048) # <--- 新增 (长度按需调整)
    schema.add_field(field_name="video_link", datatype=DataType.VARCHAR, max_length=1024)      # <--- 新增
    schema.add_field(field_name="level", datatype=DataType.VARCHAR, max_length=255)       # <--- 新增
    # 如果还有原始 JSON 中的 filename 和 relative_path 也想单独存储，可以像下面这样添加
    schema.add_field(field_name="original_filename", datatype=DataType.VARCHAR, max_length=1024) # <--- 新增 (可选)
    schema.add_field(field_name="original_relative_path", datatype=DataType.VARCHAR, max_length=2048) # <--- 新增 (可选)


    # --- 步骤 3: 创建 Milvus 集合 ---
    # 建议对新的 schema 使用新的集合名，或者确保旧集合被删除
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

    # --- 步骤 4: 创建索引 ---
    print(f"为集合 '{KEYWORD_COLLECTION_NAME}' 在字段 'keyword_group_vector' 上配置 AUTOINDEX。")
    index_params = milvus_client.prepare_index_params()
    index_params.add_index(field_name="keyword_group_vector", index_type="AUTOINDEX", metric_type="L2") # 或者 COSINE，取决于你的嵌入模型和相似度度量需求
    # 如果其他 VARCHAR 字段也需要被高效过滤或查询，可以考虑为它们创建标量索引 (如 MARISA_TRIE)
    # index_params.add_index(field_name="video_title", index_type="MARISA_TRIE") # 示例
    try:
        milvus_client.create_index(KEYWORD_COLLECTION_NAME, index_params)
        print("索引创建/配置成功。")
    except Exception as e:
        print(f"创建索引失败: {e}")


    # --- 步骤 5: 读取 JSON，生成嵌入并准备插入数据 ---
    data_to_insert_milvus = []
    try:
        with open(JSON_DATA_SOURCE_PATH, 'r', encoding='utf-8') as json_file:
            json_data = json.load(json_file)
            print(f"从 JSON 文件 '{JSON_DATA_SOURCE_PATH}' 读取到 {len(json_data)} 条数据。")

            for item in tqdm(json_data, desc="处理JSON行并生成嵌入"):
                keywords_list = item.get("keywords", [])
                if not keywords_list:
                    # print(f"  警告：条目 {item.get('filename', '未知文件')} 的关键词列表为空，已跳过。")
                    continue
                keyword_group = "，".join(keywords_list) # 使用中文逗号连接

                # 从 JSON item 中获取原始文件名和相对路径 (这些仍在你的新JSON中)
                original_filename_val = item.get("filename", "")
                original_relative_path_val = item.get("relative_path", "")
                
                # 3.1 从 item 中获取新字段的值
                video_title_val = item.get("video_title", "")           # <--- 新增
                video_chapter_val = item.get("video_chapter", "")       # <--- 新增
                video_description_val = item.get("video_description", "") # <--- 新增
                video_link_val = item.get("video_link", "")             # <--- 新增
                video_level_val = item.get("level", "")           # <--- 新增


                try:
                    embedding_list = embedding_function_impl(keyword_group) # 嵌入函数现在只接收一个文本字符串
                    if not embedding_list or not embedding_list[0] or not isinstance(embedding_list[0], list) or len(embedding_list[0]) == 0:
                        print(f"  警告：关键词组 '{keyword_group[:50]}...' (来自 {original_filename_val}) 的嵌入结果无效，已跳过。")
                        continue
                    
                    keyword_group_embedding = embedding_list[0] # embedding_function_impl([text]) 返回 [[...]]

                    if len(keyword_group_embedding) != DIM_VALUE:
                        print(f"  错误：关键词组 '{keyword_group[:50]}...' (来自 {original_filename_val}) 的嵌入维度 ({len(keyword_group_embedding)}) 与 Milvus 集合维度 ({DIM_VALUE}) 不匹配，已跳过。")
                        continue
                    
                    # 3.2 将新字段添加到插入数据中
                    data_to_insert_milvus.append({
                        "keyword_group_text": keyword_group,
                        "md_files_list_str": f"{original_filename_val} | {original_relative_path_val}", # 保持这个组合字段
                        "keyword_group_vector": keyword_group_embedding,
                        "video_title": video_title_val,                 # <--- 新增
                        "video_chapter": video_chapter_val,             # <--- 新增
                        "video_description": video_description_val,     # <--- 新增
                        "video_link": video_link_val,                   # <--- 新增
                        "level": video_level_val,                 # <--- 新增
                        "original_filename": original_filename_val,         # <--- 新增 (可选)
                        "original_relative_path": original_relative_path_val # <--- 新增 (可选)
                    })
                except Exception as e:
                    print(f"  处理关键词组 '{keyword_group[:50]}...' (来自 {original_filename_val}, 调用嵌入函数或准备数据) 时出错: {e}")

    except FileNotFoundError:
        print(f"错误: JSON文件 '{JSON_DATA_SOURCE_PATH}' 在尝试读取时未找到。")
        exit(1)
    except json.JSONDecodeError:
        print(f"错误: JSON文件 '{JSON_DATA_SOURCE_PATH}' 格式无效。")
        exit(1)
    except Exception as e:
        print(f"读取或处理JSON文件 '{JSON_DATA_SOURCE_PATH}' 时发生错误: {e}")
        exit(1)

    # --- 步骤 6: 批量插入数据到 Milvus ---
    if data_to_insert_milvus:
        print(f"\n准备将 {len(data_to_insert_milvus)} 条数据从 JSON 插入 Milvus...")
        try:
            # Milvus Python SDK v2.3+ insert() 返回 MutationResult 对象
            # Milvus Python SDK v2.4+ insert() 返回 dict {'ids': [...], ...}
            insert_result = milvus_client.insert(
                collection_name=KEYWORD_COLLECTION_NAME,
                data=data_to_insert_milvus
            )
            
            inserted_count = 0
            # 检查 insert_result 的类型并获取实际插入的 ID 数量
            if hasattr(insert_result, 'insert_count'): # pymilvus < 2.4 (MutationResult)
                inserted_count = insert_result.insert_count
            elif isinstance(insert_result, dict) and 'ids' in insert_result: # pymilvus >= 2.4
                 inserted_count = len(insert_result['ids'])
            elif isinstance(insert_result, dict) and 'insert_count' in insert_result: # 兼容某些版本可能返回的dict
                 inserted_count = insert_result['insert_count']
            else:
                print(f"警告: 未能从 Milvus 返回结果中明确解析 'insert_count' 或 'ids'。返回类型: {type(insert_result)}, 内容: {str(insert_result)[:200]}...")
                # 作为后备，可以假设如果操作未抛出异常，则尝试插入的数量等于准备的数量
                #但这并不总是准确，因为部分数据可能因各种原因（如类型不匹配、长度超限等）插入失败而SDK未明确报错
                # inserted_count = len(data_to_insert_milvus) # 谨慎使用此回退

            if inserted_count > 0:
                 print(f"成功提交了 {inserted_count} 条数据到 Milvus (基于返回的 insert_count/ids)。")
            elif len(data_to_insert_milvus) > 0 :
                 print(f"Milvus 插入操作已提交，但未能确认插入数量或返回数量为0。请检查 Milvus 日志。数据已准备: {len(data_to_insert_milvus)}")
            else:
                 print("没有数据被提交到 Milvus (data_to_insert_milvus 为空)。")


            print(f"正在执行 Flush 操作: {KEYWORD_COLLECTION_NAME}...")
            milvus_client.flush(collection_name=KEYWORD_COLLECTION_NAME) 
            print("Flush 操作完成。")
            
            # 等待 flush 完成，对于大型数据集尤其重要
            # milvus_client.wait_for_flushed(collection_name=KEYWORD_COLLECTION_NAME) # pymilvus 2.4+
            # print("Flush 等待完成。")


            stats_after_flush = milvus_client.get_collection_stats(collection_name=KEYWORD_COLLECTION_NAME)
            print(f"Flush后集合统计: {stats_after_flush}")

            print(f"正在加载集合 '{KEYWORD_COLLECTION_NAME}' 到内存...")
            milvus_client.load_collection(KEYWORD_COLLECTION_NAME)
            print(f"集合 '{KEYWORD_COLLECTION_NAME}' 加载完成。")

        except Exception as e:
            print(f"插入数据到 Milvus 或后续操作时发生错误: {e}")
    else:
        print("没有从 JSON 读取到有效数据，或处理嵌入时出错，无法插入 Milvus。")

    # 关闭 Milvus 客户端连接 (可选, MilvusClient 通常在对象销毁时自动处理)
    # milvus_client.close()
    # print("Milvus 客户端连接已关闭。")

    print("\nMilvus 处理完成。脚本执行结束。")
