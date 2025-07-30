# milvus_query_api_full_fields_post.py
import os
import json
import numpy as np
from pymilvus import MilvusClient, FieldSchema, CollectionSchema
from fastapi import FastAPI, HTTPException, Depends, Body # Body 用于 POST
from pydantic import BaseModel, Field # BaseModel 用于请求体
import uvicorn
import traceback # 用于打印详细错误

# --- 全局配置 ---
MILVUS_HOST = os.getenv("MILVUS_HOST", "127.0.0.1")
MILVUS_PORT = 19530

EMBEDDING_URL= os.getenv("EMBEDDING_URL", "http://127.0.0.1:9997/v1")
print("EMBEDDING_URL=",EMBEDDING_URL)
MILVUS_URI = f"tcp://{MILVUS_HOST}:{MILVUS_PORT}"
DB_NAME = "math"
KNOWLEDGE_COLLECTION_NAME = "math_knowledge_data_v2" # 知识点库集合名称
# 假设知识点库向量维度和字段名如下 (请根据实际情况修改)
KNOWLEDGE_DIM_VALUE = 1024
KNOWLEDGE_VECTOR_FIELD = "keyword_group_vector"

# 新增: 题库集合相关配置
QUESTIONS_COLLECTION_NAME = "math_questions_v5"
QUESTIONS_DIM_VALUE = 1024 # 与知识库维度一致 (假设)
QUESTIONS_VECTOR_FIELD = "description_vector" # 题库的向量字段名

# Embedding 服务设置
EMBEDDING_API_KEY = "ollama_or_any_non_empty_string"
EMBEDDING_MODEL_OVERRIDE = None # 如果需要指定模型，在此处设置

# --- 初始化 FastAPI 应用 ---
app = FastAPI(
    title="Milvus 查询 API (POST, 包含全字段)",
    description="提供对 Milvus 知识点库和题库的查询接口（使用POST请求），返回匹配项的所有字段信息。",
    version="1.2.0" # 版本更新
)

# --- 定义请求体模型 ---
class SearchKnowledgeRequest(BaseModel):
    query: str = Field(..., description="要搜索的文本内容")
    top_k: int = Field(5, gt=0, description="返回最相似结果的数量")

class SearchQuestionsRequest(BaseModel):
    query: str = Field(..., description="要搜索的题目描述或相关文本")
    top_k: int = Field(5, gt=0, description="返回最相似结果的数量")


# --- 初始化嵌入函数 ---
embedding_function_impl = None
try:
    from embedding import OpenAIEmbeddingFunction
    print("成功从 embedding.py 导入 OpenAIEmbeddingFunction。")
    if EMBEDDING_MODEL_OVERRIDE:
        embedding_function_impl = OpenAIEmbeddingFunction(
            url=EMBEDDING_URL, api_key=EMBEDDING_API_KEY, model=EMBEDDING_MODEL_OVERRIDE
        )
    else:
        embedding_function_impl = OpenAIEmbeddingFunction( url=EMBEDDING_URL,api_key=EMBEDDING_API_KEY)
    print("--- 嵌入函数信息 ---")
    print(f"   类型: {type(embedding_function_impl).__name__}")
    if hasattr(embedding_function_impl, 'model'):
        print(f"   模型: {getattr(embedding_function_impl, 'model', 'N/A')}")
    if hasattr(embedding_function_impl, 'base_url'):
        print(f"   Base URL: {getattr(embedding_function_impl, 'base_url', 'N/A')}")
    print("--------------------")

except ImportError:
    print("错误: 无法从 embedding.py 导入 OpenAIEmbeddingFunction。将无法进行查询。")
except Exception as e:
    print(f"实例化 OpenAIEmbeddingFunction 时出错: {e}。将无法进行查询。")

# --- Milvus 客户端依赖 ---
def get_milvus_client():
    client = None
    try:
        client = MilvusClient(uri=MILVUS_URI, db_name=DB_NAME)
        print(f"Milvus client created for DB: {DB_NAME}")
        yield client # 提供给路径操作函数
    except Exception as e:
        print(f"创建 Milvus 客户端失败: {e}")
        raise HTTPException(status_code=503, detail=f"无法连接到 Milvus 服务: {e}")
    finally:
        if client:
            # MilvusClient in pymilvus 2.4+ doesn't require explicit close()
            # If using older versions, you might need: client.close()
            print("Milvus client dependency finished.")


# --- API 端点：搜索知识点库 (POST) ---
@app.post("/search_knowledge_points", summary="通过POST搜索知识点库") # 修改为 POST
async def search_knowledge_endpoint(
    payload: SearchKnowledgeRequest, # 使用请求体模型
    client: MilvusClient = Depends(get_milvus_client) # 注入 Milvus 客户端
):
    """
    根据输入的文本在知识点库 (`math_knowledge_data`) 中进行语义搜索 (使用 POST 请求)。

    请求体应包含:
    - **query**: 用于搜索的文本。
    - **top_k**: 返回多少个最相似的结果。

    返回一个包含多个结果的列表，每个结果是一个包含所有字段（除向量外）和距离得分的字典。
    """
    if not embedding_function_impl:
        raise HTTPException(status_code=500, detail="嵌入函数未成功初始化，无法执行查询。")

    query = payload.query
    top_k = payload.top_k

    try:
        # 1. 检查集合是否存在
        if not client.has_collection(KNOWLEDGE_COLLECTION_NAME):
            raise HTTPException(status_code=404, detail=f"集合 '{KNOWLEDGE_COLLECTION_NAME}' 在数据库 '{DB_NAME}' 中不存在。")

        # 2. 获取所有非向量字段名
        collection_info = client.describe_collection(KNOWLEDGE_COLLECTION_NAME)
        all_fields = [field['name'] for field in collection_info['fields'] if field['name'] != KNOWLEDGE_VECTOR_FIELD]
        primary_key_field_name = None
        for field in collection_info['fields']:
            if field.get('is_primary', False):
                primary_key_field_name = field['name']
                break
        if primary_key_field_name and primary_key_field_name not in all_fields:
            all_fields.append(primary_key_field_name)
        output_fields_list = list(set(all_fields))
        print(f"将查询知识点库字段: {output_fields_list}")

        # 3. 加载集合 (如果需要)
        try:
            client.load_collection(KNOWLEDGE_COLLECTION_NAME)
            print(f"集合 '{KNOWLEDGE_COLLECTION_NAME}' 已加载。")
        except Exception as load_err:
            print(f"加载集合 '{KNOWLEDGE_COLLECTION_NAME}' 时出现非致命错误: {load_err}")

        # 4. 生成查询向量
        print(f"为查询 '{query}' 生成嵌入向量...")
        try:
            query_embedding_list = embedding_function_impl(query)
            if not query_embedding_list or not isinstance(query_embedding_list, list) or not query_embedding_list[0] or not isinstance(query_embedding_list[0], list) or len(query_embedding_list[0]) == 0:
                raise ValueError(f"未能生成有效的嵌入向量列表。返回: {query_embedding_list}")
            query_vector = query_embedding_list[0]
            if len(query_vector) != KNOWLEDGE_DIM_VALUE:
                raise ValueError(f"查询向量维度 ({len(query_vector)}) 与集合维度 ({KNOWLEDGE_DIM_VALUE}) 不匹配。")
            print("查询向量生成成功。")
        except Exception as embed_err:
            print(f"生成嵌入向量时出错: {embed_err}")
            raise HTTPException(status_code=500, detail=f"生成查询向量失败: {embed_err}")

        # 5. 执行搜索
        print(f"正在搜索 Top {top_k} 个最相似的结果...")
        search_params = {
             "metric_type": "L2",
        }
        results = client.search(
            collection_name=KNOWLEDGE_COLLECTION_NAME,
            data=[query_vector],
            anns_field=KNOWLEDGE_VECTOR_FIELD,
            limit=top_k,
            output_fields=output_fields_list,
            search_params=search_params,
            consistency_level="Strong"
        )
        print("搜索完成。")

        # 6. 处理并格式化结果
        response_data = []
        if results and results[0]:
            for hit in results[0]:
                entity_data = hit.get('entity', {})
                result_item = {
                    "id": hit.get('id'),
                    "distance": hit.get('distance'),
                    "entity": entity_data
                }
                response_data.append(result_item)

        return response_data

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"处理 /search_knowledge 请求时发生未处理的错误: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {e}")


# --- 新增 API 端点：搜索题库 (POST) ---
@app.post("/search_core_questions", summary="通过POST搜索题库") # 修改为 POST
async def search_questions_endpoint(
    payload: SearchQuestionsRequest, # 使用请求体模型
    client: MilvusClient = Depends(get_milvus_client) # 注入 Milvus 客户端
):
    """
    根据输入的文本在题库 (`math_questions_v2`) 中进行语义搜索 (使用 POST 请求)。

    请求体应包含:
    - **query**: 用于搜索的文本。
    - **top_k**: 返回多少个最相似的结果。

    返回一个包含多个结果的列表，每个结果是一个包含所有字段（除向量外）和距离得分的字典。
    """
    if not embedding_function_impl:
        raise HTTPException(status_code=500, detail="嵌入函数未成功初始化，无法执行查询。")

    query = payload.query
    top_k = payload.top_k

    try:
        # 1. 检查集合是否存在
        if not client.has_collection(QUESTIONS_COLLECTION_NAME):
            raise HTTPException(status_code=404, detail=f"集合 '{QUESTIONS_COLLECTION_NAME}' 在数据库 '{DB_NAME}' 中不存在。")

        # 2. 获取所有非向量字段名
        collection_info = client.describe_collection(QUESTIONS_COLLECTION_NAME)
        all_fields = [field['name'] for field in collection_info['fields'] if field['name'] != QUESTIONS_VECTOR_FIELD]
        primary_key_field_name = None
        for field in collection_info['fields']:
            if field.get('is_primary', False):
                primary_key_field_name = field['name']
                break
        if primary_key_field_name and primary_key_field_name not in all_fields:
            all_fields.append(primary_key_field_name)
        output_fields_list = list(set(all_fields))
        print(f"将查询题库字段: {output_fields_list}")

        # 3. 加载集合 (如果需要)
        try:
            client.load_collection(QUESTIONS_COLLECTION_NAME)
            print(f"集合 '{QUESTIONS_COLLECTION_NAME}' 已加载。")
        except Exception as load_err:
            print(f"加载集合 '{QUESTIONS_COLLECTION_NAME}' 时出现非致命错误: {load_err}")
            # 通常可以继续搜索

        # 4. 生成查询向量
        print(f"为查询 '{query}' 生成嵌入向量...")
        try:
            query_embedding_list = embedding_function_impl(query)
            if not query_embedding_list or not isinstance(query_embedding_list, list) or not query_embedding_list[0] or not isinstance(query_embedding_list[0], list) or len(query_embedding_list[0]) == 0:
                raise ValueError(f"未能生成有效的嵌入向量列表。返回: {query_embedding_list}")
            query_vector = query_embedding_list[0]
            if len(query_vector) != QUESTIONS_DIM_VALUE: # 使用题库的维度
                raise ValueError(f"查询向量维度 ({len(query_vector)}) 与集合维度 ({QUESTIONS_DIM_VALUE}) 不匹配。")
            print("查询向量生成成功。")
        except Exception as embed_err:
            print(f"生成嵌入向量时出错: {embed_err}")
            raise HTTPException(status_code=500, detail=f"生成查询向量失败: {embed_err}")

        # 5. 执行搜索
        print(f"正在搜索 Top {top_k} 个最相似的结果...")
        search_params = {
             "metric_type": "L2", # 或 "IP", 应与建库时索引一致
        }
        results = client.search(
            collection_name=QUESTIONS_COLLECTION_NAME,
            data=[query_vector],
            anns_field=QUESTIONS_VECTOR_FIELD, # 使用题库的向量字段
            limit=top_k,
            output_fields=output_fields_list,
            search_params=search_params,
            consistency_level="Strong"
        )
        print("搜索完成。")

        # 6. 处理并格式化结果
        response_data = []
        if results and results[0]:
            for hit in results[0]:
                entity_data = hit.get('entity', {})
                # 尝试将 question_pages 从 JSON 字符串解析回对象
                if 'question_pages' in entity_data and isinstance(entity_data['question_pages'], str):
                    try:
                        entity_data['question_pages'] = json.loads(entity_data['question_pages'])
                    except json.JSONDecodeError:
                        print(f"警告: 解析 question_pages 字段失败 (ID: {hit.get('id')})。将保持为字符串。")

                result_item = {
                    "id": hit.get('id'),
                    "distance": hit.get('distance'),
                    "entity": entity_data # 包含所有 output_fields 的字典
                }
                response_data.append(result_item)

        return response_data

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"处理 /search_questions 请求时发生未处理的错误: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {e}")

# --- Uvicorn 启动入口 ---
if __name__ == "__main__":
    print("启动 FastAPI 应用 (POST)...")
    # 假设 Python 文件名为 milvus_query_api_full_fields_post.py
    # 如果你将此代码保存为不同的文件名，请相应修改 "milvus_query_api_full_fields_post:app"
    uvicorn.run("milvus_query_api:app", host="0.0.0.0", port=9000, reload=True)