# 构建MD版数据的milvus数据库（服务端模式，Attu可见）- topics集合
import os
import json
import numpy as np
from pymilvus import MilvusClient, DataType
from tqdm import tqdm

# --- 全局配置 ---
# 1. 更新 JSON 文件路径为MD版数据
JSON_DATA_SOURCE_PATH = "/home/xkb2/Desktop/QY/new_json_util/json_data/初中/初中生物/初中生物母题数据_full_content.json"
MILVUS_HOST = "127.0.0.1"
MILVUS_PORT = 19530
MILVUS_URI = f"tcp://{MILVUS_HOST}:{MILVUS_PORT}"
DB_NAME = "junior"  # 使用您现有的数据库
KEYWORD_COLLECTION_NAME = "biology_junior_topic"  # 题型集合
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
    print(f"\n--- 开始从JSON文件 '{JSON_DATA_SOURCE_PATH}' 读取数据并存入 Milvus 服务端 (topics集合) ---")

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

        # 现在，创建用于后续操作的 MilvusClient，并指定连接到目标数据库 DB_NAME
        print(f"正在连接到 Milvus 数据库 '{DB_NAME}' ({MILVUS_URI})...")
        milvus_client = MilvusClient(uri=MILVUS_URI, db_name=DB_NAME)
        print(f"成功连接到 Milvus，当前操作数据库: '{DB_NAME}'")

    except Exception as e:
        print(f"连接 Milvus 或准备数据库 '{DB_NAME}' 失败: {e}")
        exit(1)

    # --- 步骤 2: 定义 Milvus Schema（使用英文字段名，包含MD版字段）---
    print("正在定义 Milvus schema（MD版题型数据）...")
    schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)
    
    # 主键ID
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
    
    # 数据的原有字段（使用英文字段名）
    schema.add_field(field_name="subject", datatype=DataType.VARCHAR, max_length=256)  # 科目
    schema.add_field(field_name="module", datatype=DataType.VARCHAR, max_length=512)   # 模块
    schema.add_field(field_name="topic_name", datatype=DataType.VARCHAR, max_length=512)  # 题型名称
    schema.add_field(field_name="topic_source", datatype=DataType.VARCHAR, max_length=512)  # 题型来源
    schema.add_field(field_name="course_name", datatype=DataType.VARCHAR, max_length=512)  # 题型来源课程名称
    schema.add_field(field_name="course_code", datatype=DataType.VARCHAR, max_length=256)  # 题型对应课程编号
    schema.add_field(field_name="index", datatype=DataType.VARCHAR, max_length=512)  # 题型关键词
    # 使用ARRAY类型存储图片路径列表
    schema.add_field(field_name="image_path", datatype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=100, max_length=1024)
    
    # MD版新增字段
    schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)  # OCR文本内容
    schema.add_field(field_name="path", datatype=DataType.VARCHAR, max_length=1024)  # 单个页面图片路径
    schema.add_field(field_name="length", datatype=DataType.INT64)  # 文本长度
    schema.add_field(field_name="page_number", datatype=DataType.INT64)  # 页面号
    
    # 新增full_content字段 - 存储同一题型下所有页面内容的列表
    schema.add_field(field_name="full_content", datatype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=200, max_length=65535)  # 同题型所有页面内容
    
    # index字段的向量表示（题型关键词的embedding）
    schema.add_field(field_name="index_vector", datatype=DataType.FLOAT_VECTOR, dim=DIM_VALUE)

    # --- 步骤 3: 创建 Milvus 集合 ---
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
    print(f"为集合 '{KEYWORD_COLLECTION_NAME}' 在字段 'index_vector' 上配置 AUTOINDEX。")
    index_params = milvus_client.prepare_index_params()
    index_params.add_index(field_name="index_vector", index_type="AUTOINDEX", metric_type="L2")
    
    # 不创建标量索引，避免MARISA_TRIE兼容性问题
    # index_params.add_index(field_name="subject", index_type="MARISA_TRIE")
    # index_params.add_index(field_name="module", index_type="MARISA_TRIE")
    # index_params.add_index(field_name="course_code", index_type="MARISA_TRIE")
    
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

            for item in tqdm(json_data, desc="处理MD版数据并生成嵌入"):
                # 提取所有字段（从JSON的中文字段名映射到英文字段名）
                subject_val = item.get("科目", "")
                module_val = item.get("模块", "")
                topic_name_val = item.get("题型名称", "")
                topic_source_val = item.get("题型来源", "")
                course_name_val = item.get("题型来源课程名称", "")
                course_code_val = item.get("题型对应课程编号", "")
                index_val = item.get("index", "")  # 题型关键词，用于embedding
                image_path_val = item.get("image_path", [])
                
                # MD版新增字段
                content_val = item.get("content", "")  # OCR文本内容
                path_val = item.get("path", "")  # 单个页面图片路径
                length_val = item.get("length", 0)  # 文本长度
                page_number_val = item.get("page_number", 0)  # 页面号
                
                # 新增full_content字段
                full_content_val = item.get("full_content", [])  # 同题型所有页面内容

                # 确保image_path是列表格式
                if not isinstance(image_path_val, list):
                    image_path_val = [str(image_path_val)] if image_path_val else []

                # 确保full_content是列表格式
                if not isinstance(full_content_val, list):
                    full_content_val = [str(full_content_val)] if full_content_val else []

                # 检查必要字段
                if not index_val:
                    print(f"  警告：记录的index字段为空，已跳过。题型名称：{topic_name_val}，页面：{page_number_val}")
                    continue

                try:
                    # 对index字段（题型关键词）进行embedding
                    embedding_list = embedding_function_impl(index_val)
                    if not embedding_list or not embedding_list[0] or not isinstance(embedding_list[0], list) or len(embedding_list[0]) == 0:
                        print(f"  警告：index '{index_val}' 的嵌入结果无效，已跳过。")
                        continue
                    
                    index_embedding = embedding_list[0] # embedding_function_impl([text]) 返回 [[...]]

                    if len(index_embedding) != DIM_VALUE:
                        print(f"  错误：index '{index_val}' 的嵌入维度 ({len(index_embedding)}) 与 Milvus 集合维度 ({DIM_VALUE}) 不匹配，已跳过。")
                        continue
                    
                    # 准备插入数据（使用英文字段名）
                    data_to_insert_milvus.append({
                        "subject": subject_val,           # 科目
                        "module": module_val,             # 模块
                        "topic_name": topic_name_val,     # 题型名称
                        "topic_source": topic_source_val, # 题型来源
                        "course_name": course_name_val,   # 题型来源课程名称
                        "course_code": course_code_val,   # 题型对应课程编号
                        "index": index_val,               # 题型关键词
                        "image_path": image_path_val,     # 图片路径列表
                        "content": content_val,           # OCR文本内容
                        "path": path_val,                 # 单个页面图片路径
                        "length": length_val,             # 文本长度
                        "page_number": page_number_val,   # 页面号
                        "full_content": full_content_val, # 同题型所有页面内容列表
                        "index_vector": index_embedding   # 题型关键词的向量表示
                    })
                    
                except Exception as e:
                    print(f"  处理index '{index_val}' (题型：{topic_name_val}，页面：{page_number_val}) 时出错: {e}")

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
        print(f"\n准备将 {len(data_to_insert_milvus)} 条MD版数据插入 Milvus...")
        try:
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

            if inserted_count > 0:
                 print(f"成功提交了 {inserted_count} 条MD版数据到 Milvus。")
            elif len(data_to_insert_milvus) > 0 :
                 print(f"Milvus 插入操作已提交，但未能确认插入数量或返回数量为0。请检查 Milvus 日志。数据已准备: {len(data_to_insert_milvus)}")
            else:
                 print("没有数据被提交到 Milvus。")

            print(f"正在执行 Flush 操作: {KEYWORD_COLLECTION_NAME}...")
            milvus_client.flush(collection_name=KEYWORD_COLLECTION_NAME) 
            print("Flush 操作完成。")

            stats_after_flush = milvus_client.get_collection_stats(collection_name=KEYWORD_COLLECTION_NAME)
            print(f"Flush后集合统计: {stats_after_flush}")

            print(f"正在加载集合 '{KEYWORD_COLLECTION_NAME}' 到内存...")
            milvus_client.load_collection(KEYWORD_COLLECTION_NAME)
            print(f"集合 '{KEYWORD_COLLECTION_NAME}' 加载完成。")

            # 显示一些插入的数据示例
            print("\n插入数据示例:")
            for i, data in enumerate(data_to_insert_milvus[:3]):
                print(f"记录 {i+1}:")
                print(f"  科目(subject): {data['subject']}")
                print(f"  模块(module): {data['module']}")
                print(f"  题型名称(topic_name): {data['topic_name'][:50]}...")
                print(f"  页面号(page_number): {data['page_number']}")
                print(f"  文本长度(length): {data['length']}")
                print(f"  index (题型关键词): {data['index']}")
                print(f"  content预览: {data['content'][:100]}...")
                print(f"  单页图片路径(path): {data['path']}")
                print(f"  图片路径数组长度: {len(data['image_path'])}")
                print(f"  full_content数组长度: {len(data['full_content'])}")
                print(f"  向量维度: {len(data['index_vector'])}")
                print("-" * 40)

        except Exception as e:
            print(f"插入数据到 Milvus 或后续操作时发生错误: {e}")
    else:
        print("没有从 JSON 读取到有效数据，或处理嵌入时出错，无法插入 Milvus。")

    print(f"\nMD版数据处理完成！")
    print(f"数据库: {DB_NAME}")
    print(f"集合: {KEYWORD_COLLECTION_NAME}")
    print(f"插入记录数: {len(data_to_insert_milvus)}")
    print("\n字段映射关系:")
    print("  科目 -> subject")
    print("  模块 -> module") 
    print("  题型名称 -> topic_name")
    print("  题型来源 -> topic_source")
    print("  题型来源课程名称 -> course_name")
    print("  题型对应课程编号 -> course_code")
    print("  index -> index (题型关键词，需要embedding)")
    print("  image_path -> image_path (图片路径数组)")
    print("  content -> content (OCR文本内容)")
    print("  path -> path (单个页面图片路径)")
    print("  length -> length (文本长度)")
    print("  page_number -> page_number (页面号)")
    print("  full_content -> full_content (同题型所有页面内容数组)")
    print("脚本执行结束。")