# 智能分段存入脚本 - 简化版，不添加额外的分段字段
import os
import json
import numpy as np
from pymilvus import MilvusClient, DataType
from tqdm import tqdm
import re
from typing import List, Optional, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- 全局配置 ---
JSON_DATA_SOURCE_PATH = "/home/xkb2/Desktop/QY/new_json_util/json_data/初中/初中生物/初中生物母题数据_full_content.json"
MILVUS_HOST = "127.0.0.1"
MILVUS_PORT = 19530
MILVUS_URI = f"tcp://{MILVUS_HOST}:{MILVUS_PORT}"
DB_NAME = "junior"  # 使用数据库
KEYWORD_COLLECTION_NAME = "biology_junior_content"  # content集合
DIM_VALUE = 1024  # 向量维度
LENGTH_THRESHOLD = 1024  # 长度阈值，超过此值进行分段
CHUNK_SIZE = 800  # 分段大小
CHUNK_OVERLAP = 50  # 重叠字符数

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

# --- 使用LangChain中文分段逻辑 ---
def _split_text_with_regex_from_end(
        text: str, separator: str, keep_separator: bool
) -> List[str]:
    if separator:
        if keep_separator:
            _splits = re.split(f"({separator})", text)
            splits = ["".join(i) for i in zip(_splits[0::2], _splits[1::2])]
            if len(_splits) % 2 == 1:
                splits += _splits[-1:]
        else:
            splits = re.split(separator, text)
    else:
        splits = list(text)
    return [s for s in splits if s != ""]

class ChineseRecursiveTextSplitter(RecursiveCharacterTextSplitter):
    def __init__(
            self,
            separators: Optional[List[str]] = None,
            keep_separator: bool = True,
            is_separator_regex: bool = True,
            **kwargs: Any,
    ) -> None:
        super().__init__(keep_separator=keep_separator, **kwargs)
        self._separators = separators or [
            "\n\n",
            "\n",
            "。|！|？",  # 句号、感叹号、问号
            "\.\s|\!\s|\?\s",  # 英文句号、感叹号、问号加空格
            "；|;\s",   # 分号
            "，|,\s"    # 逗号
        ]
        self._is_separator_regex = is_separator_regex

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        final_chunks = []
        separator = separators[-1]
        new_separators = []
        for i, _s in enumerate(separators):
            _separator = _s if self._is_separator_regex else re.escape(_s)
            if _s == "":
                separator = _s
                break
            if re.search(_separator, text):
                separator = _s
                new_separators = separators[i + 1:]
                break

        _separator = separator if self._is_separator_regex else re.escape(separator)
        splits = _split_text_with_regex_from_end(text, _separator, self._keep_separator)

        _good_splits = []
        _separator = "" if self._keep_separator else separator
        for s in splits:
            if self._length_function(s) < self._chunk_size:
                _good_splits.append(s)
            else:
                if _good_splits:
                    merged_text = self._merge_splits(_good_splits, _separator)
                    final_chunks.extend(merged_text)
                    _good_splits = []
                if not new_separators:
                    final_chunks.append(s)
                else:
                    other_info = self._split_text(s, new_separators)
                    final_chunks.extend(other_info)
        if _good_splits:
            merged_text = self._merge_splits(_good_splits, _separator)
            final_chunks.extend(merged_text)
        return [re.sub(r"\n{2,}", "\n", chunk.strip()) for chunk in final_chunks if chunk.strip()!=""]

# 初始化中文文本分割器
text_splitter = ChineseRecursiveTextSplitter(
    keep_separator=True,
    is_separator_regex=True,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

# --- 主程序 ---
if __name__ == "__main__":
    print(f"\n--- 智能分段存入脚本 (保持字段统一性) ---")
    print(f"数据源: {JSON_DATA_SOURCE_PATH}")
    print(f"长度阈值: {LENGTH_THRESHOLD} 字符")
    print(f"分段大小: {CHUNK_SIZE} 字符")
    print(f"重叠大小: {CHUNK_OVERLAP} 字符")

    if not os.path.exists(JSON_DATA_SOURCE_PATH):
        print(f"错误: 数据源 JSON 文件 '{JSON_DATA_SOURCE_PATH}' 未找到。")
        exit(1)

    # --- 步骤 1: 连接 Milvus ---
    try:
        print(f"正在连接 Milvus 服务 ({MILVUS_URI}) 以管理数据库...")
        admin_client = MilvusClient(uri=MILVUS_URI)
        print("成功连接到 Milvus 服务。")

        if DB_NAME not in admin_client.list_databases():
            print(f"数据库 '{DB_NAME}' 不存在，正在创建...")
            admin_client.create_database(db_name=DB_NAME)
            print(f"数据库 '{DB_NAME}' 创建成功。")
        else:
            print(f"数据库 '{DB_NAME}' 已存在。")

        print(f"正在连接到 Milvus 数据库 '{DB_NAME}'...")
        milvus_client = MilvusClient(uri=MILVUS_URI, db_name=DB_NAME)
        print(f"成功连接到数据库: '{DB_NAME}'")

    except Exception as e:
        print(f"连接 Milvus 失败: {e}")
        exit(1)

    # --- 步骤 2: 定义 Milvus Schema（标准字段，不包含分段相关字段）---
    print("正在定义 Milvus schema（标准content集合字段）...")
    schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)
    
    # 主键ID
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
    
    # 标准字段（与其他collection保持一致）
    schema.add_field(field_name="subject", datatype=DataType.VARCHAR, max_length=256)
    schema.add_field(field_name="module", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="topic_name", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="topic_source", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="course_name", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="course_code", datatype=DataType.VARCHAR, max_length=256)
    schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)  # OCR文本内容
    schema.add_field(field_name="image_path", datatype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=100, max_length=1024)
    schema.add_field(field_name="path", datatype=DataType.VARCHAR, max_length=1024)
    schema.add_field(field_name="length", datatype=DataType.INT64)
    schema.add_field(field_name="page_number", datatype=DataType.INT64)
    
    # 新增full_content字段 - 存储同一题型下所有页面内容的列表
    schema.add_field(field_name="full_content", datatype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=200, max_length=65535)  # 同题型所有页面内容
    
    # content向量字段
    schema.add_field(field_name="content_vector", datatype=DataType.FLOAT_VECTOR, dim=DIM_VALUE)

    # --- 步骤 3: 创建 Milvus 集合 ---
    if milvus_client.has_collection(collection_name=KEYWORD_COLLECTION_NAME):
        print(f"删除已存在的集合 '{KEYWORD_COLLECTION_NAME}'...")
        milvus_client.drop_collection(collection_name=KEYWORD_COLLECTION_NAME)
    
    try:
        print(f"正在创建集合: {KEYWORD_COLLECTION_NAME}")
        milvus_client.create_collection(collection_name=KEYWORD_COLLECTION_NAME, schema=schema, consistency_level="Strong")
        print(f"集合 '{KEYWORD_COLLECTION_NAME}' 创建成功。")
    except Exception as e:
        print(f"创建集合失败: {e}")
        exit(1)

    # --- 步骤 4: 创建索引 ---
    print("正在创建索引...")
    index_params = milvus_client.prepare_index_params()
    index_params.add_index(field_name="content_vector", index_type="AUTOINDEX", metric_type="L2")
    
    try:
        milvus_client.create_index(KEYWORD_COLLECTION_NAME, index_params)
        print("索引创建成功。")
    except Exception as e:
        print(f"创建索引失败: {e}")

    # --- 步骤 5: 读取数据并根据长度智能处理 ---
    data_to_insert = []
    segmented_count = 0
    normal_count = 0
    total_segments_created = 0
    
    try:
        with open(JSON_DATA_SOURCE_PATH, 'r', encoding='utf-8') as json_file:
            json_data = json.load(json_file)
            print(f"从 JSON 文件读取到 {len(json_data)} 条数据。")

            for item in tqdm(json_data, desc="智能处理数据并生成嵌入"):
                # 提取所有字段
                subject_val = item.get("科目", "")
                module_val = item.get("模块", "")
                topic_name_val = item.get("题型名称", "")
                topic_source_val = item.get("题型来源", "")
                course_name_val = item.get("题型来源课程名称", "")
                course_code_val = item.get("题型对应课程编号", "")
                content_val = item.get("content", "")
                image_path_val = item.get("image_path", [])
                path_val = item.get("path", "")
                length_val = item.get("length", 0)
                page_number_val = item.get("page_number", 0)
                
                # 新增full_content字段
                full_content_val = item.get("full_content", [])  # 同题型所有页面内容

                # 确保image_path是列表格式
                if not isinstance(image_path_val, list):
                    image_path_val = [str(image_path_val)] if image_path_val else []

                # 确保full_content是列表格式
                if not isinstance(full_content_val, list):
                    full_content_val = [str(full_content_val)] if full_content_val else []

                # 检查必要字段
                if not content_val:
                    print(f"  警告：记录的content字段为空，已跳过。题型名称：{topic_name_val}，页面：{page_number_val}")
                    continue

                # 根据length字段判断是否需要分段
                if length_val > LENGTH_THRESHOLD:
                    # 需要分段处理
                    print(f"  检测到超长文本 (长度: {length_val})，进行分段处理: {topic_name_val}")
                    
                    try:
                        # 使用LangChain分割文本
                        text_chunks = text_splitter.split_text(content_val)
                        print(f"    LangChain分割完成，共 {len(text_chunks)} 段")
                        
                        # 为每个分段创建记录
                        for i, segment_text in enumerate(text_chunks):
                            try:
                                # 为当前分段生成embedding
                                embedding_list = embedding_function_impl(segment_text)
                                if not embedding_list or not embedding_list[0] or not isinstance(embedding_list[0], list):
                                    print(f"    警告：分段 {i+1} 的embedding生成失败，跳过")
                                    continue
                                
                                segment_embedding = embedding_list[0]
                                
                                if len(segment_embedding) != DIM_VALUE:
                                    print(f"    错误：分段 {i+1} 的embedding维度不匹配，跳过")
                                    continue
                                
                                # 创建分段记录（使用标准字段，不包含分段元数据）
                                segment_record = {
                                    "subject": subject_val,
                                    "module": module_val,
                                    "topic_name": topic_name_val,
                                    "topic_source": topic_source_val,
                                    "course_name": course_name_val,
                                    "course_code": course_code_val,
                                    "content": content_val,  # 保持原始完整文本
                                    "image_path": image_path_val,
                                    "path": path_val,
                                    "length": length_val,
                                    "page_number": page_number_val,
                                    "full_content": full_content_val,  # 同题型所有页面内容列表
                                    "content_vector": segment_embedding  # 基于分段文本的embedding
                                }
                                
                                data_to_insert.append(segment_record)
                                total_segments_created += 1
                                
                            except Exception as e:
                                print(f"    处理分段 {i+1} 时出错: {e}")
                        
                        segmented_count += 1
                        
                    except Exception as e:
                        print(f"  分段处理出错: {e}")
                else:
                    # 正常处理（不分段）
                    try:
                        # 对content字段进行embedding
                        embedding_list = embedding_function_impl(content_val)
                        if not embedding_list or not embedding_list[0] or not isinstance(embedding_list[0], list) or len(embedding_list[0]) == 0:
                            print(f"  警告：content '{content_val[:50]}...' 的嵌入结果无效，已跳过。")
                            continue
                        
                        content_embedding = embedding_list[0]

                        if len(content_embedding) != DIM_VALUE:
                            print(f"  错误：content '{content_val[:50]}...' 的嵌入维度不匹配，已跳过。")
                            continue
                        
                        # 创建普通记录（使用标准字段）
                        normal_record = {
                            "subject": subject_val,
                            "module": module_val,
                            "topic_name": topic_name_val,
                            "topic_source": topic_source_val,
                            "course_name": course_name_val,
                            "course_code": course_code_val,
                            "content": content_val,
                            "image_path": image_path_val,
                            "path": path_val,
                            "length": length_val,
                            "page_number": page_number_val,
                            "full_content": full_content_val,  # 同题型所有页面内容列表
                            "content_vector": content_embedding
                        }
                        
                        data_to_insert.append(normal_record)
                        normal_count += 1
                        
                    except Exception as e:
                        print(f"  处理content '{content_val[:50]}...' 时出错: {e}")

    except Exception as e:
        print(f"读取或处理JSON文件时出错: {e}")
        exit(1)

    # --- 步骤 6: 插入数据到 Milvus ---
    if data_to_insert:
        print(f"\n准备插入 {len(data_to_insert)} 条数据到 Milvus...")
        print(f"  普通处理: {normal_count} 条记录")
        print(f"  分段处理: {segmented_count} 条原始记录 -> {total_segments_created} 条分段记录")
        print(f"  总记录数: {len(data_to_insert)} 条")
        
        try:
            insert_result = milvus_client.insert(
                collection_name=KEYWORD_COLLECTION_NAME,
                data=data_to_insert
            )
            
            print(f"插入操作完成，结果: {insert_result}")
            
            # Flush和加载
            print("执行 Flush 操作...")
            milvus_client.flush(collection_name=KEYWORD_COLLECTION_NAME)
            
            print("加载集合到内存...")
            milvus_client.load_collection(KEYWORD_COLLECTION_NAME)
            
            # 检查插入结果
            stats = milvus_client.get_collection_stats(collection_name=KEYWORD_COLLECTION_NAME)
            print(f"集合统计信息: {stats}")
            
            print(f"\n✅ 智能分段存入成功完成！")
            print(f"数据库: {DB_NAME}")
            print(f"集合: {KEYWORD_COLLECTION_NAME}")
            print(f"处理策略:")
            print(f"  长度阈值: {LENGTH_THRESHOLD} 字符")
            print(f"  普通存入: {normal_count} 条")
            print(f"  分段存入: {segmented_count} 条原始记录 -> {total_segments_created} 条分段记录")
            print(f"  总计存入: {len(data_to_insert)} 条记录")
            print(f"字段说明:")
            print(f"  - 使用标准字段结构，与其他collection保持一致")
            print(f"  - 分段记录共享相同的原始字段值")
            print(f"  - content_vector基于不同文本片段生成")
            print(f"  - full_content包含同题型下所有页面内容")
            
        except Exception as e:
            print(f"插入数据时出错: {e}")
    else:
        print("没有数据需要插入")

    print("\n脚本执行完成")