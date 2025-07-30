# universal_milvus_query_api.py
import os
import json
import numpy as np
from pymilvus import MilvusClient, FieldSchema, CollectionSchema
from fastapi import FastAPI, HTTPException, Depends, Body
from pydantic import BaseModel, Field
import uvicorn
import traceback
from typing import Optional, List, Dict, Any
from enum import Enum

# --- 全局配置 ---
MILVUS_HOST = os.getenv("MILVUS_HOST", "127.0.0.1")
MILVUS_PORT = 19530

EMBEDDING_URL= os.getenv("EMBEDDING_URL", "http://127.0.0.1:9997/v1")
MILVUS_URI = f"tcp://{MILVUS_HOST}:{MILVUS_PORT}"

# 教育阶段配置
class EducationLevel(str, Enum):
    JUNIOR = "junior"    # 初中
    SENIOR = "senior"    # 高中

# 学科配置
class Subject(str, Enum):
    CHINESE = "chinese"     # 语文
    MATH = "math"          # 数学
    ENGLISH = "english"    # 英语
    PHYSICS = "physics"    # 物理
    CHEMISTRY = "chemistry" # 化学
    BIOLOGY = "biology"    # 生物
    POLITICS = "politics"  # 政治
    HISTORY = "history"    # 历史
    GEOGRAPHY = "geography" # 地理

# 集合类型配置
class CollectionType(str, Enum):
    topic = "topic"      # 题型数据
    CONTENT = "content"    # 内容数据

# 数据库配置
DATABASE_CONFIG = {
    EducationLevel.JUNIOR: "junior",
    EducationLevel.SENIOR: "senior"
}

# 集合命名规则：{subject}_{education_level}_{collection_type}
COLLECTION_CONFIG = {
    Subject.BIOLOGY: {
        EducationLevel.JUNIOR: {
            CollectionType.topic: "biology_junior_topic",
            CollectionType.CONTENT: "biology_junior_content"
        },
        EducationLevel.SENIOR: {
            CollectionType.topic: "biology_senior_topic", 
            CollectionType.CONTENT: "biology_senior_content"
        }
    },
    Subject.MATH: {
        EducationLevel.JUNIOR: {
            CollectionType.topic: "math_junior_topic",
            CollectionType.CONTENT: "math_junior_content"
        },
        EducationLevel.SENIOR: {
            CollectionType.topic: "math_senior_topic",
            CollectionType.CONTENT: "math_senior_content"
        }
    },
    Subject.CHINESE: {
        EducationLevel.JUNIOR: {
            CollectionType.topic: "chinese_junior_topic",
            CollectionType.CONTENT: "chinese_junior_content"
        },
        EducationLevel.SENIOR: {
            CollectionType.topic: "chinese_senior_topic",
            CollectionType.CONTENT: "chinese_senior_content"
        }
    },
    Subject.ENGLISH: {
        EducationLevel.JUNIOR: {
            CollectionType.topic: "english_junior_topic",
            CollectionType.CONTENT: "english_junior_content"
        },
        EducationLevel.SENIOR: {
            CollectionType.topic: "english_senior_topic",
            CollectionType.CONTENT: "english_senior_content"
        }
    },
    Subject.PHYSICS: {
        EducationLevel.JUNIOR: {
            CollectionType.topic: "physics_junior_topic",
            CollectionType.CONTENT: "physics_junior_content"
        },
        EducationLevel.SENIOR: {
            CollectionType.topic: "physics_senior_topic",
            CollectionType.CONTENT: "physics_senior_content"
        }
    },
    Subject.CHEMISTRY: {
        EducationLevel.JUNIOR: {
            CollectionType.topic: "chemistry_junior_topic",
            CollectionType.CONTENT: "chemistry_junior_content"
        },
        EducationLevel.SENIOR: {
            CollectionType.topic: "chemistry_senior_topic",
            CollectionType.CONTENT: "chemistry_senior_content"
        }
    },
    Subject.POLITICS: {
        EducationLevel.JUNIOR: {
            CollectionType.topic: "politics_junior_topic",
            CollectionType.CONTENT: "politics_junior_content"
        },
        EducationLevel.SENIOR: {
            CollectionType.topic: "politics_senior_topic",
            CollectionType.CONTENT: "politics_senior_content"
        }
    },
    Subject.HISTORY: {
        EducationLevel.JUNIOR: {
            CollectionType.topic: "history_junior_topic",
            CollectionType.CONTENT: "history_junior_content"
        },
        EducationLevel.SENIOR: {
            CollectionType.topic: "history_senior_topic",
            CollectionType.CONTENT: "history_senior_content"
        }
    },
    Subject.GEOGRAPHY: {
        EducationLevel.JUNIOR: {
            CollectionType.topic: "geography_junior_topic",
            CollectionType.CONTENT: "geography_junior_content"
        },
        EducationLevel.SENIOR: {
            CollectionType.topic: "geography_senior_topic",
            CollectionType.CONTENT: "geography_senior_content"
        }
    }
}

# 当前已部署的学科（用于验证）
DEPLOYED_SUBJECTS = {
    Subject.CHINESE,
    Subject.MATH,
    Subject.ENGLISH,
    Subject.PHYSICS,
    Subject.CHEMISTRY,
    Subject.BIOLOGY,
    Subject.POLITICS,
    Subject.HISTORY,
    Subject.GEOGRAPHY
}

# 向量配置
DIM_VALUE = 1024
# 根据集合类型使用不同的向量字段
VECTOR_FIELD_CONFIG = {
    CollectionType.topic: "index_vector",    # topic集合使用index_vector
    CollectionType.CONTENT: "content_vector"  # content集合也使用index_vector
}

# Embedding 服务设置
EMBEDDING_API_KEY = "ollama_or_any_non_empty_string"
EMBEDDING_MODEL_OVERRIDE = None

# --- 初始化 FastAPI 应用 ---
app = FastAPI(
    title="通用教育资源 Milvus 查询 API",
    description="提供初高中全学科教育资源的统一查询接口，支持按教育阶段、学科和集合类型分类查询。",
    version="4.0.0"
)

# --- 定义请求体模型 ---
class UniversalSearchRequest(BaseModel):
    query: str = Field(..., description="要搜索的关键词或文本内容")
    education_level: EducationLevel = Field(..., description="教育阶段：junior（初中）或 senior（高中）")
    subject: Subject = Field(..., description="学科名称")
    collection_type: CollectionType = Field(..., description="集合类型：topic（题型）或 content（内容）")
    top_k: int = Field(5, gt=0, le=50, description="返回最相似结果的数量（1-50）")

class MultiCollectionSearchRequest(BaseModel):
    query: str = Field(..., description="要搜索的关键词或文本内容")
    education_level: EducationLevel = Field(..., description="教育阶段")
    subject: Subject = Field(..., description="学科名称")
    collection_types: Optional[List[CollectionType]] = Field(
        default=None,
        description="要搜索的集合类型列表，为空则搜索所有类型"
    )
    top_k: int = Field(5, gt=0, le=50, description="每种集合类型返回最相似结果的数量（1-50）")

class MultiLevelSearchRequest(BaseModel):
    query: str = Field(..., description="要搜索的关键词或文本内容")
    subject: Subject = Field(..., description="学科名称") 
    collection_type: CollectionType = Field(..., description="集合类型")
    education_levels: Optional[List[EducationLevel]] = Field(
        default=None, 
        description="要搜索的教育阶段列表，为空则搜索所有阶段"
    )
    top_k: int = Field(5, gt=0, le=50, description="每个教育阶段返回最相似结果的数量（1-50）")

class MultiSubjectSearchRequest(BaseModel):
    query: str = Field(..., description="要搜索的关键词或文本内容")
    education_level: EducationLevel = Field(..., description="教育阶段")
    collection_type: CollectionType = Field(..., description="集合类型")
    subjects: Optional[List[Subject]] = Field(
        default=None,
        description="要搜索的学科列表，为空则搜索所有已部署的学科"
    )
    top_k: int = Field(5, gt=0, le=50, description="每个学科返回最相似结果的数量（1-50）")

# --- 初始化嵌入函数 ---
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


# --- 工具函数 ---
def get_milvus_client(db_name: str):
    """获取指定数据库的 Milvus 客户端"""
    try:
        client = MilvusClient(uri=MILVUS_URI, db_name=db_name)
        return client
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"无法连接到 Milvus 数据库 {db_name}: {e}")

def validate_subject_deployment(subject: Subject):
    """验证学科是否已部署"""
    if subject not in DEPLOYED_SUBJECTS:
        available_subjects = [s.value for s in DEPLOYED_SUBJECTS]
        raise HTTPException(
            status_code=400,
            detail=f"学科 '{subject.value}' 暂未部署。当前可用学科: {available_subjects}"
        )

def get_collection_name(subject: Subject, education_level: EducationLevel, collection_type: CollectionType):
    """获取集合名称"""
    if subject not in COLLECTION_CONFIG:
        raise HTTPException(status_code=400, detail=f"不支持的学科: {subject.value}")
    
    if education_level not in COLLECTION_CONFIG[subject]:
        raise HTTPException(
            status_code=400, 
            detail=f"学科 '{subject.value}' 在 '{education_level.value}' 阶段暂未配置"
        )
    
    if collection_type not in COLLECTION_CONFIG[subject][education_level]:
        raise HTTPException(
            status_code=400,
            detail=f"学科 '{subject.value}' 在 '{education_level.value}' 阶段的 '{collection_type.value}' 集合暂未配置"
        )
    
    return COLLECTION_CONFIG[subject][education_level][collection_type]

def generate_query_vector(query: str):
    """生成查询向量"""
    if not embedding_function_impl:
        raise HTTPException(status_code=500, detail="嵌入函数未成功初始化，无法执行查询。")
    
    try:
        query_embedding_list = embedding_function_impl(query)
        if not query_embedding_list or not isinstance(query_embedding_list, list) or not query_embedding_list[0] or not isinstance(query_embedding_list[0], list) or len(query_embedding_list[0]) == 0:
            raise ValueError(f"未能生成有效的嵌入向量列表。返回: {query_embedding_list}")
        query_vector = query_embedding_list[0]
        if len(query_vector) != DIM_VALUE:
            raise ValueError(f"查询向量维度 ({len(query_vector)}) 与集合维度 ({DIM_VALUE}) 不匹配。")
        return query_vector
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成查询向量失败: {e}")

def search_in_collection(client: MilvusClient, collection_name: str, collection_type: CollectionType, query_vector: list, top_k: int):
    """在指定集合中执行搜索"""
    # 检查集合是否存在
    if not client.has_collection(collection_name):
        raise HTTPException(status_code=404, detail=f"集合 '{collection_name}' 不存在")

    # 获取向量字段名
    vector_field = VECTOR_FIELD_CONFIG.get(collection_type, "index_vector")

    # 获取所有非向量字段名
    collection_info = client.describe_collection(collection_name)
    all_fields = [field['name'] for field in collection_info['fields'] if field['name'] != vector_field]
    primary_key_field_name = None
    for field in collection_info['fields']:
        if field.get('is_primary', False):
            primary_key_field_name = field['name']
            break
    if primary_key_field_name and primary_key_field_name not in all_fields:
        all_fields.append(primary_key_field_name)
    output_fields_list = list(set(all_fields))

    print(f"集合 '{collection_name}' 字段: {output_fields_list}")
    print(f"使用向量字段: {vector_field}")

    # 加载集合
    try:
        client.load_collection(collection_name)
        print(f"集合 '{collection_name}' 已加载")
    except Exception as load_err:
        print(f"加载集合 '{collection_name}' 时出现警告: {load_err}")

    # 执行搜索
    search_params = {"metric_type": "L2"}
    results = client.search(
        collection_name=collection_name,
        data=[query_vector],
        anns_field=vector_field,
        limit=top_k,
        output_fields=output_fields_list,
        search_params=search_params,
        consistency_level="Strong"
    )

    print(f"搜索完成，找到 {len(results[0]) if results and results[0] else 0} 个结果")
    return results

def safe_convert_value(value):
    """安全地转换值为JSON可序列化的格式"""
    if value is None:
        return None
    elif isinstance(value, (str, int, float, bool)):
        return value
    elif isinstance(value, list):
        return [safe_convert_value(item) for item in value]
    elif isinstance(value, dict):
        return {str(k): safe_convert_value(v) for k, v in value.items()}
    else:
        # 对于其他类型，转换为字符串
        try:
            return str(value)
        except:
            return None

def format_topic_results(results, subject: Subject, education_level: EducationLevel):
    """格式化topic集合的搜索结果"""
    formatted_results = []
    
    if results and results[0]:
        for hit in results[0]:
            entity_data = hit.get('entity', {})
            
            # 安全地提取和转换所有字段
            safe_entity_data = {k: safe_convert_value(v) for k, v in entity_data.items()}
            
            result_item = {
                "id": safe_convert_value(hit.get('id')),
                "distance": safe_convert_value(hit.get('distance')),
                "similarity_score": round(1 / (1 + float(hit.get('distance', 0))), 4),
                "education_level": education_level.value,
                "subject": subject.value,
                "collection_type": "topic",
                "entity": safe_entity_data
            }
            formatted_results.append(result_item)
    
    return formatted_results

def format_content_results(results, subject: Subject, education_level: EducationLevel):
    """格式化content集合的搜索结果"""
    formatted_results = []
    
    if results and results[0]:
        for hit in results[0]:
            entity_data = hit.get('entity', {})
            
            # 安全地提取和转换所有字段
            safe_entity_data = {k: safe_convert_value(v) for k, v in entity_data.items()}
            
            result_item = {
                "id": safe_convert_value(hit.get('id')),
                "distance": safe_convert_value(hit.get('distance')),
                "similarity_score": round(1 / (1 + float(hit.get('distance', 0))), 4),
                "education_level": education_level.value,
                "subject": subject.value,
                "collection_type": "content",
                "entity": safe_entity_data
            }
            formatted_results.append(result_item)
    
    return formatted_results

def format_search_results(results, subject: Subject, education_level: EducationLevel, collection_type: CollectionType):
    """根据集合类型格式化搜索结果"""
    if collection_type == CollectionType.topic:
        return format_topic_results(results, subject, education_level)
    elif collection_type == CollectionType.CONTENT:
        return format_content_results(results, subject, education_level)
    else:
        raise HTTPException(status_code=400, detail=f"不支持的集合类型: {collection_type}")

# --- API 端点 ---

@app.post("/search", summary="通用搜索接口")
async def universal_search(payload: UniversalSearchRequest):
    """
    通用搜索接口，支持按教育阶段、学科和集合类型精确查询。
    
    这是主要的搜索接口，支持所有已部署的学科、教育阶段和集合类型。
    """
    # 验证学科是否已部署
    validate_subject_deployment(payload.subject)
    
    # 获取配置
    db_name = DATABASE_CONFIG[payload.education_level]
    collection_name = get_collection_name(payload.subject, payload.education_level, payload.collection_type)
    
    print(f"搜索参数: {payload.education_level.value} - {payload.subject.value} - {payload.collection_type.value} - '{payload.query}'")
    print(f"目标: 数据库={db_name}, 集合={collection_name}")
    
    try:
        # 生成查询向量
        query_vector = generate_query_vector(payload.query)
        print(f"查询向量生成成功，维度: {len(query_vector)}")
        
        # 创建客户端并搜索
        client = get_milvus_client(db_name)
        results = search_in_collection(client, collection_name, payload.collection_type, query_vector, payload.top_k)
        
        # 格式化结果
        formatted_results = format_search_results(results, payload.subject, payload.education_level, payload.collection_type)
        
        return {
            "query_info": {
                "query": payload.query,
                "education_level": payload.education_level.value,
                "subject": payload.subject.value,
                "collection_type": payload.collection_type.value,
                "database": db_name,
                "collection": collection_name,
                "top_k": payload.top_k
            },
            "results_count": len(formatted_results),
            "results": formatted_results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"搜索过程中发生错误: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"搜索失败: {e}")

@app.post("/search_multi_collection", summary="多集合类型搜索")
async def multi_collection_search(payload: MultiCollectionSearchRequest):
    """
    在指定学科和教育阶段的多个集合类型中搜索。
    """
    validate_subject_deployment(payload.subject)
    
    # 确定要搜索的集合类型
    search_types = payload.collection_types or list(CollectionType)
    
    query_vector = generate_query_vector(payload.query)
    all_results = {}
    
    for coll_type in search_types:
        try:
            db_name = DATABASE_CONFIG[payload.education_level]
            collection_name = get_collection_name(payload.subject, payload.education_level, coll_type)
            
            print(f"搜索集合类型 {coll_type.value}: {db_name}.{collection_name}")
            
            client = get_milvus_client(db_name)
            results = search_in_collection(client, collection_name, coll_type, query_vector, payload.top_k)
            formatted_results = format_search_results(results, payload.subject, payload.education_level, coll_type)
            
            all_results[coll_type.value] = formatted_results
            
        except Exception as e:
            print(f"在集合类型 {coll_type.value} 中搜索时出错: {e}")
            all_results[coll_type.value] = []
    
    # 合并结果
    merged_results = []
    for type_results in all_results.values():
        merged_results.extend(type_results)
    
    merged_results.sort(key=lambda x: x['similarity_score'], reverse=True)
    
    return {
        "query_info": {
            "query": payload.query,
            "education_level": payload.education_level.value,
            "subject": payload.subject.value,
            "searched_collection_types": [coll_type.value for coll_type in search_types],
            "top_k_per_type": payload.top_k
        },
        "results_by_collection_type": all_results,
        "total_results_count": len(merged_results),
        "merged_results": merged_results[:payload.top_k * len(search_types)]
    }

@app.post("/search_multi_level", summary="跨教育阶段搜索")
async def multi_level_search(payload: MultiLevelSearchRequest):
    """
    在指定学科和集合类型的多个教育阶段中搜索。
    """
    validate_subject_deployment(payload.subject)
    
    # 确定要搜索的教育阶段
    search_levels = payload.education_levels or list(EducationLevel)
    
    query_vector = generate_query_vector(payload.query)
    all_results = {}
    
    for level in search_levels:
        try:
            db_name = DATABASE_CONFIG[level]
            collection_name = get_collection_name(payload.subject, level, payload.collection_type)
            
            print(f"搜索 {level.value}: {db_name}.{collection_name}")
            
            client = get_milvus_client(db_name)
            results = search_in_collection(client, collection_name, payload.collection_type, query_vector, payload.top_k)
            formatted_results = format_search_results(results, payload.subject, level, payload.collection_type)
            
            all_results[level.value] = formatted_results
            
        except Exception as e:
            print(f"在 {level.value} 阶段搜索时出错: {e}")
            all_results[level.value] = []
    
    # 合并结果
    merged_results = []
    for level_results in all_results.values():
        merged_results.extend(level_results)
    
    merged_results.sort(key=lambda x: x['similarity_score'], reverse=True)
    
    return {
        "query_info": {
            "query": payload.query,
            "subject": payload.subject.value,
            "collection_type": payload.collection_type.value,
            "searched_levels": [level.value for level in search_levels],
            "top_k_per_level": payload.top_k
        },
        "results_by_level": all_results,
        "total_results_count": len(merged_results),
        "merged_results": merged_results[:payload.top_k * len(search_levels)]
    }

@app.post("/search_multi_subject", summary="跨学科搜索")
async def multi_subject_search(payload: MultiSubjectSearchRequest):
    """
    在指定教育阶段和集合类型的多个学科中搜索。
    """
    # 确定要搜索的学科（只搜索已部署的学科）
    search_subjects = payload.subjects or list(DEPLOYED_SUBJECTS)
    
    # 验证所有请求的学科都已部署
    for subject in search_subjects:
        validate_subject_deployment(subject)
    
    query_vector = generate_query_vector(payload.query)
    all_results = {}
    
    for subject in search_subjects:
        try:
            db_name = DATABASE_CONFIG[payload.education_level]
            collection_name = get_collection_name(subject, payload.education_level, payload.collection_type)
            
            print(f"搜索学科 {subject.value}: {db_name}.{collection_name}")
            
            client = get_milvus_client(db_name)
            results = search_in_collection(client, collection_name, payload.collection_type, query_vector, payload.top_k)
            formatted_results = format_search_results(results, subject, payload.education_level, payload.collection_type)
            
            all_results[subject.value] = formatted_results
            
        except Exception as e:
            print(f"在学科 {subject.value} 中搜索时出错: {e}")
            all_results[subject.value] = []
    
    # 合并结果
    merged_results = []
    for subject_results in all_results.values():
        merged_results.extend(subject_results)
    
    merged_results.sort(key=lambda x: x['similarity_score'], reverse=True)
    
    return {
        "query_info": {
            "query": payload.query,
            "education_level": payload.education_level.value,
            "collection_type": payload.collection_type.value,
            "searched_subjects": [subject.value for subject in search_subjects],
            "top_k_per_subject": payload.top_k
        },
        "results_by_subject": all_results,
        "total_results_count": len(merged_results),
        "merged_results": merged_results[:payload.top_k * len(search_subjects)]
    }

@app.get("/config", summary="获取系统配置")
async def get_system_config():
    """获取当前系统支持的配置信息"""
    return {
        "education_levels": [level.value for level in EducationLevel],
        "collection_types": [ctype.value for ctype in CollectionType],
        "all_subjects": [subject.value for subject in Subject],
        "deployed_subjects": [subject.value for subject in DEPLOYED_SUBJECTS],
        "databases": {level.value: db_name for level, db_name in DATABASE_CONFIG.items()},
        "collections": {
            subject.value: {
                level.value: {
                    ctype.value: collection_name
                    for ctype, collection_name in types.items()
                }
                for level, types in levels.items()
            }
            for subject, levels in COLLECTION_CONFIG.items()
        },
        "vector_config": {
            "dimension": DIM_VALUE,
            "vector_fields": VECTOR_FIELD_CONFIG
        }
    }

@app.get("/health", summary="健康检查")
async def health_check():
    """检查API服务状态"""
    return {
        "status": "healthy",
        "embedding_available": embedding_function_impl is not None,
        "deployed_subjects": [subject.value for subject in DEPLOYED_SUBJECTS],
        "total_supported_subjects": len(Subject),
        "collection_types": [ctype.value for ctype in CollectionType]
    }

# --- 调试端点：检查集合状态 ---
@app.get("/debug/collections", summary="调试：检查集合状态")
async def debug_collections():
    """调试用：检查所有数据库中的集合状态"""
    status = {}
    
    for level, db_name in DATABASE_CONFIG.items():
        try:
            client = get_milvus_client(db_name)
            collections = client.list_collections()
            status[level.value] = {
                "database": db_name,
                "collections": collections,
                "status": "connected"
            }
        except Exception as e:
            status[level.value] = {
                "database": db_name,
                "error": str(e),
                "status": "failed"
            }
    
    return status

# --- 启动应用 ---
if __name__ == "__main__":
    print("启动通用教育资源 Milvus 查询 API...")
    print(f"当前已部署学科: {[s.value for s in DEPLOYED_SUBJECTS]}")
    print(f"支持的教育阶段: {[l.value for l in EducationLevel]}")
    print(f"支持的集合类型: {[t.value for t in CollectionType]}")
    print("集合配置:")
    for subject, levels in COLLECTION_CONFIG.items():
        for level, types in levels.items():
            for ctype, collection_name in types.items():
                print(f"  {level.value}.{collection_name} ({ctype.value})")
    uvicorn.run("query_api:app", host="0.0.0.0", port=9000, reload=True)