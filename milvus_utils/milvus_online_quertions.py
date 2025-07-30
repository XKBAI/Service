# ingest_questions_to_milvus.py
import os
import json
import numpy as np
from pymilvus import MilvusClient, DataType
from tqdm import tqdm

# --- 全局配置 ---
# 1. 更新 JSON 文件路径
JSON_DATA_SOURCE_PATH = "/home/xkb2/Desktop/QY/metadata_with_qpages_dict_and_qpaths_list.json" # <--- 修改这里：指向清洗后的题目JSON文件
MILVUS_HOST = "127.0.0.1"
MILVUS_PORT = 19530
MILVUS_URI = f"tcp://{MILVUS_HOST}:{MILVUS_PORT}"
DB_NAME = "math"  # 数据库名可以保持不变，如果题目和知识点在同一库不同集合
QUESTIONS_COLLECTION_NAME = "math_questions_v5"  # <--- 修改这里：新的集合名
DIM_VALUE = 1024  # !!! 向量维度，必须与你的嵌入模型输出一致 !!!

# Embedding 服务设置 (与原脚本一致)
EMBEDDING_API_KEY = "ollama_or_any_non_empty_string"
EMBEDDING_MODEL_OVERRIDE = None # "quentinz/bge-large-zh-v1.5" # 如果需要指定模型

# --- 从 embedding.py 导入嵌入函数 (与原脚本一致) ---
try:
    from embedding import OpenAIEmbeddingFunction
    print("成功从 embedding.py 导入 OpenAIEmbeddingFunction。")
    if EMBEDDING_MODEL_OVERRIDE:
        embedding_function_impl = OpenAIEmbeddingFunction(
            api_key=EMBEDDING_API_KEY, model_name=EMBEDDING_MODEL_OVERRIDE # 注意embedding.py中可能是model_name
        )
    else:
        embedding_function_impl = OpenAIEmbeddingFunction(api_key=EMBEDDING_API_KEY)
    print(f"嵌入模型维度 (DIM_VALUE) 设置为: {DIM_VALUE}")
    # 假设 embedding_function_impl 有一个获取维度的方法或属性，或者我们直接用DIM_VALUE
    # if hasattr(embedding_function_impl, 'dim') and embedding_function_impl.dim != DIM_VALUE:
    #     print(f"警告: 配置文件中的DIM_VALUE ({DIM_VALUE}) 与嵌入函数报告的维度 ({embedding_function_impl.dim}) 不符。请检查！")

except ImportError:
    print("错误: 无法从 embedding.py 导入 OpenAIEmbeddingFunction。将使用备用占位符。")
    class FallbackPlaceholderEmbeddingFunction:
        def __init__(self, *args, **kwargs): print(f"警告: 使用备用占位符嵌入函数。维度: {DIM_VALUE}")
        def __call__(self, text_or_texts):
            is_single = isinstance(text_or_texts, str)
            texts = [text_or_texts] if is_single else text_or_texts
            if not texts or all(not t for t in texts) : # 处理空输入
                 return [[] for _ in texts] if not is_single else []
            return [[np.random.rand(DIM_VALUE).tolist() for _ in texts]] # 保持 [[...]] 结构
    embedding_function_impl = FallbackPlaceholderEmbeddingFunction()
except Exception as e:
    print(f"实例化从 embedding.py 导入的 OpenAIEmbeddingFunction 时出错: {e}。将使用备用占位符。")
    class FallbackPlaceholderEmbeddingFunction:
        def __init__(self, *args, **kwargs): print(f"警告: 使用备用占位符嵌入函数。维度: {DIM_VALUE}")
        def __call__(self, text_or_texts):
            is_single = isinstance(text_or_texts, str)
            texts = [text_or_texts] if is_single else text_or_texts
            if not texts or all(not t for t in texts) :
                 return [[] for _ in texts] if not is_single else []
            return [[np.random.rand(DIM_VALUE).tolist() for _ in texts]]
    embedding_function_impl = FallbackPlaceholderEmbeddingFunction()

# --- 主程序 ---
if __name__ == "__main__":
    print(f"\n--- 开始从 JSON Lines 文件 '{JSON_DATA_SOURCE_PATH}' 读取题目数据并存入 Milvus 服务端 ---")

    if not os.path.exists(JSON_DATA_SOURCE_PATH):
        print(f"错误: 数据源 JSON 文件 '{JSON_DATA_SOURCE_PATH}' 未找到。")
        exit(1)

    # --- 步骤 1: 连接 Milvus 服务端并确保目标数据库存在 (与原脚本类似) ---
    try:
        print(f"正在连接 Milvus 服务 ({MILVUS_URI}) 以管理数据库...")
        admin_client = MilvusClient(uri=MILVUS_URI)
        print("成功连接到 Milvus 服务 (用于数据库管理)。")

        if DB_NAME not in admin_client.list_databases():
            print(f"数据库 '{DB_NAME}' 不存在，正在创建...")
            admin_client.create_database(db_name=DB_NAME)
            print(f"数据库 '{DB_NAME}' 创建成功。")
        else:
            print(f"数据库 '{DB_NAME}' 已存在。")
        
        print(f"正在连接到 Milvus 数据库 '{DB_NAME}' ({MILVUS_URI})...")
        milvus_client = MilvusClient(uri=MILVUS_URI, db_name=DB_NAME)
        print(f"成功连接到 Milvus，当前操作数据库: '{DB_NAME}'")

    except Exception as e:
        print(f"连接 Milvus 或准备数据库 '{DB_NAME}' 失败: {e}")
        exit(1)

    # --- 步骤 2: 定义新的 Milvus Schema 针对题目数据 ---
    print("正在定义题目数据的 Milvus schema...")
    schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(field_name="md_file_path", datatype=DataType.VARCHAR, max_length=2048) # 来自 JSON "title"
    schema.add_field(field_name="is_master_question", datatype=DataType.VARCHAR, max_length=32) # 来自 JSON "是否是母题"
    schema.add_field(field_name="question_number_str", datatype=DataType.VARCHAR, max_length=128) # 来自 JSON "编号"
    schema.add_field(field_name="question_type", datatype=DataType.VARCHAR, max_length=512)    # 来自 JSON "题型"
    schema.add_field(field_name="question_tag", datatype=DataType.VARCHAR, max_length=512)      # 来自 JSON "标记"
    schema.add_field(field_name="question_description", datatype=DataType.VARCHAR, max_length=65535) # 来自 JSON "description"
    schema.add_field(field_name="description_vector", datatype=DataType.FLOAT_VECTOR, dim=DIM_VALUE) # 向量字段
    # 新增字段：question_pages
    schema.add_field(field_name="question_pages", datatype=DataType.VARCHAR, max_length=65535) # 来自 JSON "question_pages"
    # 新增字段：question_related_paths (存储为 JSON 字符串)
    schema.add_field(field_name="question_related_paths", datatype=DataType.VARCHAR, max_length=65535) # 来自 JSON "question_related_paths"

    # --- 步骤 3: 创建 Milvus 集合 (使用新的集合名) ---
    if milvus_client.has_collection(collection_name=QUESTIONS_COLLECTION_NAME):
        print(f"Milvus 集合 '{QUESTIONS_COLLECTION_NAME}' 已存在。将删除并重建。")
        milvus_client.drop_collection(collection_name=QUESTIONS_COLLECTION_NAME)
    try:
        print(f"正在创建 Milvus 集合: {QUESTIONS_COLLECTION_NAME}，维度: {DIM_VALUE}")
        milvus_client.create_collection(collection_name=QUESTIONS_COLLECTION_NAME, schema=schema, consistency_level="Strong")
        print(f"Milvus 集合 '{QUESTIONS_COLLECTION_NAME}' 创建成功。")
    except Exception as e:
        print(f"创建 Milvus 集合 '{QUESTIONS_COLLECTION_NAME}' 失败: {e}")
        exit(1)

    # --- 步骤 4: 创建索引 (针对新的向量字段名) ---
    print(f"为集合 '{QUESTIONS_COLLECTION_NAME}' 在字段 'description_vector' 上配置 AUTOINDEX。")
    index_params = milvus_client.prepare_index_params()
    index_params.add_index(field_name="description_vector", index_type="AUTOINDEX", metric_type="L2") # 或 COSINE
    # 可以为常用于过滤的 VARCHAR 字段创建标量索引，如 question_type, is_master_question
    # index_params.add_index(field_name="question_type", index_type="MARISA_TRIE")
    # index_params.add_index(field_name="is_master_question", index_type="MARISA_TRIE") 
    try:
        milvus_client.create_index(QUESTIONS_COLLECTION_NAME, index_params)
        print("索引创建/配置成功。")
    except Exception as e:
        print(f"创建索引失败: {e}")


    # --- 步骤 5: 读取 JSON Lines，生成嵌入并准备插入数据 ---
    data_to_insert_milvus = []
    all_json_items = []
    try:
        with open(JSON_DATA_SOURCE_PATH, 'r', encoding='utf-8') as json_file:
            for line_num, line in enumerate(json_file):
                line_content = line.strip()
                if not line_content:
                    continue # 跳过空行
                try:
                    all_json_items.append(json.loads(line_content))
                except json.JSONDecodeError:
                    print(f"警告: JSON 解析错误，跳过行 {line_num + 1}: '{line_content[:100]}...'")
            print(f"从 JSON Lines 文件 '{JSON_DATA_SOURCE_PATH}' 读取到 {len(all_json_items)} 条记录。")

            for item in tqdm(all_json_items, desc="处理JSON记录并生成嵌入"):
                description_text = item.get("description", "")
                if not description_text: # 如果描述为空，可能不值得嵌入和存储
                    # print(f"  警告：记录 (md_file_path: {item.get('title', '未知')}) 的 description 为空，已跳过。")
                    continue

                md_file_path_val = item.get("title", "") # JSON "title" 字段
                is_master_val = item.get("是否是母题", "") # 默认为 "false"
                q_number_val = item.get("编号", "")
                q_type_val = item.get("题型", "") # 确保即使过滤后还有"无"的情况，或上游已处理
                q_tag_val = item.get("标记", "无")
                # 获取 question_pages 字段
                question_pages_val = item.get("question_pages", {})
                # 将 question_pages 转换为 JSON 字符串，以便存储
                question_pages_json = json.dumps(question_pages_val, ensure_ascii=False)
                
                # 获取新增的 question_related_paths 字段 (现在它直接是字符串)
                question_related_paths_str = item.get("question_related_paths", "") # 默认为空字符串
                                
                try:
                    # 嵌入函数期望单个文本或文本列表，返回列表的列表
                    embedding_result_list = embedding_function_impl(description_text) 
                    
                    if not embedding_result_list or \
                       not embedding_result_list[0] or \
                       not isinstance(embedding_result_list[0], list) or \
                       len(embedding_result_list[0]) == 0: # 检查 vector 本身是否有效
                        print(f"  警告：description '{description_text[:50]}...' (来自 {md_file_path_val}) 的嵌入结果无效或为空，已跳过。")
                        continue
                    
                    description_embedding = embedding_result_list[0] # <--- 修改此处

                    if len(description_embedding) != DIM_VALUE:
                        print(f"  错误：description '{description_text[:50]}...' (来自 {md_file_path_val}) 的嵌入维度 ({len(description_embedding)}) 与 Milvus 集合维度 ({DIM_VALUE}) 不匹配，已跳过。")
                        continue
                    
                    data_to_insert_milvus.append({
                        "md_file_path": md_file_path_val,
                        "is_master_question": str(is_master_val).lower(), # 确保是字符串 "true"/"false"
                        "question_number_str": str(q_number_val),
                        "question_type": q_type_val,
                        "question_tag": q_tag_val,
                        "question_description": description_text,
                        "description_vector": description_embedding,
                        "question_pages": question_pages_json,  # 添加 question_pages 字段
                        "question_related_paths": question_related_paths_str, # <--- 修改在这里：直接使用字符串
                    })
                except Exception as e:
                    print(f"  处理 description '{description_text[:50]}...' (来自 {md_file_path_val}, 调用嵌入函数或准备数据) 时出错: {e}")

    except FileNotFoundError:
        print(f"错误: JSON文件 '{JSON_DATA_SOURCE_PATH}' 在尝试读取时未找到。")
        exit(1)
    except Exception as e: # 其他可能的读取文件或初始解析错误
        print(f"读取或处理JSON文件 '{JSON_DATA_SOURCE_PATH}' 时发生一般性错误: {e}")
        exit(1)

    # --- 步骤 6: 批量插入数据到 Milvus (与原脚本类似，注意集合名) ---
    if data_to_insert_milvus:
        print(f"\n准备将 {len(data_to_insert_milvus)} 条数据从 JSON 插入 Milvus 集合 '{QUESTIONS_COLLECTION_NAME}'...")
        try:
            insert_result = milvus_client.insert(
                collection_name=QUESTIONS_COLLECTION_NAME, # <--- 使用新的集合名
                data=data_to_insert_milvus
            )
            
            inserted_count = 0
            if hasattr(insert_result, 'insert_count'): 
                inserted_count = insert_result.insert_count
            elif isinstance(insert_result, dict) and 'ids' in insert_result: 
                 inserted_count = len(insert_result['ids'])
            elif isinstance(insert_result, dict) and 'insert_count' in insert_result:
                 inserted_count = insert_result['insert_count']
            else:
                print(f"警告: 未能从 Milvus 返回结果中明确解析 'insert_count' 或 'ids'。返回类型: {type(insert_result)}, 内容: {str(insert_result)[:200]}...")

            if inserted_count > 0:
                 print(f"成功提交了 {inserted_count} 条数据到 Milvus (基于返回的 insert_count/ids)。")
            elif len(data_to_insert_milvus) > 0 :
                 print(f"Milvus 插入操作已提交，但未能确认插入数量或返回数量为0。请检查 Milvus 日志。数据已准备: {len(data_to_insert_milvus)}")
            else:
                 print("没有数据被提交到 Milvus (data_to_insert_milvus 为空)。")

            print(f"正在执行 Flush 操作: {QUESTIONS_COLLECTION_NAME}...")
            milvus_client.flush(collection_name=QUESTIONS_COLLECTION_NAME) 
            print("Flush 操作完成。")
            
            stats_after_flush = milvus_client.get_collection_stats(collection_name=QUESTIONS_COLLECTION_NAME)
            print(f"Flush后集合统计: {stats_after_flush}")

            print(f"正在加载集合 '{QUESTIONS_COLLECTION_NAME}' 到内存...")
            milvus_client.load_collection(QUESTIONS_COLLECTION_NAME)
            print(f"集合 '{QUESTIONS_COLLECTION_NAME}' 加载完成。")

        except Exception as e:
            print(f"插入数据到 Milvus 或后续操作时发生错误: {e}")
    else:
        print("没有从 JSON 读取到有效数据，或处理嵌入时出错，无法插入 Milvus。")

    # milvus_client.close() # MilvusClient 通常在对象销毁时自动处理
    print("\n题目数据 Milvus 处理完成。脚本执行结束。")