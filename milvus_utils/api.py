# milvus_query_api_full_fields.py
import os
import json
import numpy as np
from pymilvus import MilvusClient, FieldSchema, CollectionSchema
from fastapi import FastAPI, Query, HTTPException, Depends
import uvicorn
import traceback # 用于打印详细错误

# --- 全局配置 ---
MILVUS_HOST = "127.0.0.1"
MILVUS_PORT = 19530
MILVUS_URI = f"tcp://{MILVUS_HOST}:{MILVUS_PORT}"
DB_NAME = "math"
KNOWLEDGE_COLLECTION_NAME = "math_knowledge_data" # 知识点库集合名称
# 假设知识点库向量维度和字段名如下 (请根据实际情况修改)
KNOWLEDGE_DIM_VALUE = 1024
KNOWLEDGE_VECTOR_FIELD = "keyword_group_vector"

# Embedding 服务设置
EMBEDDING_API_KEY = "ollama_or_any_non_empty_string"
EMBEDDING_MODEL_OVERRIDE = None # 如果需要指定模型，在此处设置

# --- 初始化 FastAPI 应用 ---
app = FastAPI(
    title="Milvus 查询 API (包含全字段)",
    description="提供对 Milvus 知识点库的查询接口，返回匹配项的所有字段信息。",
    version="1.0.0"
)

# --- 初始化嵌入函数 ---
embedding_function_impl = None
try:
    from embedding import OpenAIEmbeddingFunction
    print("成功从 embedding.py 导入 OpenAIEmbeddingFunction。")
    if EMBEDDING_MODEL_OVERRIDE:
        embedding_function_impl = OpenAIEmbeddingFunction(
            api_key=EMBEDDING_API_KEY, model=EMBEDDING_MODEL_OVERRIDE
        )
    else:
        embedding_function_impl = OpenAIEmbeddingFunction(api_key=EMBEDDING_API_KEY)
    print("--- 嵌入函数信息 ---")
    print(f"  类型: {type(embedding_function_impl).__name__}")
    if hasattr(embedding_function_impl, 'model'):
         print(f"  模型: {getattr(embedding_function_impl, 'model', 'N/A')}")
    if hasattr(embedding_function_impl, 'base_url'):
         print(f"  Base URL: {getattr(embedding_function_impl, 'base_url', 'N/A')}")
    print("--------------------")

except ImportError:
    print("错误: 无法从 embedding.py 导入 OpenAIEmbeddingFunction。将无法进行查询。")
except Exception as e:
    print(f"实例化 OpenAIEmbeddingFunction 时出错: {e}。将无法进行查询。")

# --- Milvus 客户端依赖 ---
# (每个请求创建一个新的客户端实例，对于低并发场景是可接受的)
# 更优化的方式是使用应用生命周期管理连接池，但会增加复杂度
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
            client.close()
            print("Milvus client closed.")

# --- API 端点 ---
@app.get("/search_knowledge", summary="搜索知识点库")
async def search_knowledge_endpoint(
    query: str = Query(..., description="要搜索的文本内容"),
    top_k: int = Query(5, gt=0, description="返回最相似结果的数量"),
    client: MilvusClient = Depends(get_milvus_client) # 注入 Milvus 客户端
):
    """
    根据输入的文本在知识点库 (`math_knowledge_data`) 中进行语义搜索。

    - **query**: 用于搜索的文本。
    - **top_k**: 返回多少个最相似的结果。

    返回一个包含多个结果的列表，每个结果是一个包含所有字段（除向量外）和距离得分的字典。
    """
    if not embedding_function_impl:
        raise HTTPException(status_code=500, detail="嵌入函数未成功初始化，无法执行查询。")

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
        # MilvusClient 的 search 会自动处理加载，但显式调用有时有助于调试
        try:
            client.load_collection(KNOWLEDGE_COLLECTION_NAME)
            print(f"集合 '{KNOWLEDGE_COLLECTION_NAME}' 已加载。")
        except Exception as load_err:
            print(f"加载集合 '{KNOWLEDGE_COLLECTION_NAME}' 时出现非致命错误: {load_err}")
            # 通常可以继续搜索

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
        search_params = { # 定义搜索参数 (如果需要微调)
             "metric_type": "L2", # 或 "IP"
             # "params": {"ef": 128} # 示例: HNSW 索引的搜索参数
        }
        results = client.search(
            collection_name=KNOWLEDGE_COLLECTION_NAME,
            data=[query_vector],
            anns_field=KNOWLEDGE_VECTOR_FIELD,
            limit=top_k,
            output_fields=output_fields_list,
            search_params=search_params,
            consistency_level="Strong" # 或根据需要调整
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
                    "entity": entity_data # 包含所有 output_fields 的字典
                }
                response_data.append(result_item)

        return response_data

    except HTTPException as http_exc:
        raise http_exc # 重新抛出已处理的 HTTP 异常
    except Exception as e:
        print(f"处理 /search_knowledge 请求时发生未处理的错误: {e}")
        traceback.print_exc() # 打印详细错误信息到服务器日志
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {e}")

# --- Uvicorn 启动入口 ---
if __name__ == "__main__":
    print("启动 FastAPI 应用...")
    # 使用之前选择的端口 57100
    uvicorn.run("api:app", host="0.0.0.0", port=57100, reload=True)