# 测试超长文本分段存入Milvus的脚本 - 使用langchain分段逻辑
import os
import json
import numpy as np
from pymilvus import MilvusClient, DataType
from tqdm import tqdm
import re
from typing import List, Optional, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- 全局配置 ---
JSON_DATA_SOURCE_PATH = "/home/xkb2/Desktop/QY/milvus_utils/new_tools/test.json"
MILVUS_HOST = "127.0.0.1"
MILVUS_PORT = 19530
MILVUS_URI = f"tcp://{MILVUS_HOST}:{MILVUS_PORT}"
DB_NAME = "junior"  # 使用初中数据库
KEYWORD_COLLECTION_NAME = "biology_junior_content_segment_test"  # 测试用集合名
DIM_VALUE = 1024  # 向量维度
CHUNK_SIZE = 800  # 单段最大长度，为embedding留出安全边际
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

# --- 使用同事提供的中文分段逻辑 ---
def _split_text_with_regex_from_end(
        text: str, separator: str, keep_separator: bool
) -> List[str]:
    # Now that we have the separator, split the text
    if separator:
        if keep_separator:
            # The parentheses in the pattern keep the delimiters in the result.
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
        """Create a new TextSplitter."""
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
        """Split incoming text and return chunks."""
        final_chunks = []
        # Get appropriate separator to use
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

        # Now go merging things, recursively splitting longer texts.
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
    print(f"\n--- 测试超长文本分段存入 Milvus (使用LangChain分段逻辑) ---")
    print(f"数据源: {JSON_DATA_SOURCE_PATH}")
    print(f"分段大小: {CHUNK_SIZE} 字符")
    print(f"重叠大小: {CHUNK_OVERLAP} 字符")

    if not os.path.exists(JSON_DATA_SOURCE_PATH):
        print(f"错误: 数据源 JSON 文件 '{JSON_DATA_SOURCE_PATH}' 未找到。")
        exit(1)

    # --- 步骤 1: 连接 Milvus 服务端并确保目标数据库存在 ---
    try:
        print(f"正在连接 Milvus 服务 ({MILVUS_URI}) 以管理数据库...")
        admin_client = MilvusClient(uri=MILVUS_URI)
        print("成功连接到 Milvus 服务。")

        # 检查目标数据库是否存在
        if DB_NAME not in admin_client.list_databases():
            print(f"数据库 '{DB_NAME}' 不存在，正在创建...")
            admin_client.create_database(db_name=DB_NAME)
            print(f"数据库 '{DB_NAME}' 创建成功。")
        else:
            print(f"数据库 '{DB_NAME}' 已存在。")

        # 连接到目标数据库
        print(f"正在连接到 Milvus 数据库 '{DB_NAME}'...")
        milvus_client = MilvusClient(uri=MILVUS_URI, db_name=DB_NAME)
        print(f"成功连接到数据库: '{DB_NAME}'")

    except Exception as e:
        print(f"连接 Milvus 失败: {e}")
        exit(1)

    # --- 步骤 2: 定义 Milvus Schema ---
    print("正在定义 Milvus schema（分段测试集合）...")
    schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)
    
    # 主键ID
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
    
    # 原始字段（使用英文字段名）
    schema.add_field(field_name="subject", datatype=DataType.VARCHAR, max_length=256)
    schema.add_field(field_name="module", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="topic_name", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="topic_source", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="course_name", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="course_code", datatype=DataType.VARCHAR, max_length=256)
    schema.add_field(field_name="index", datatype=DataType.VARCHAR, max_length=65535)  # 保持原始完整文本
    schema.add_field(field_name="image_path", datatype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=100, max_length=1024)
    schema.add_field(field_name="path", datatype=DataType.VARCHAR, max_length=1024)
    schema.add_field(field_name="length", datatype=DataType.INT64)
    schema.add_field(field_name="page_number", datatype=DataType.INT64)
    
    # 分段相关字段
    schema.add_field(field_name="segment_text", datatype=DataType.VARCHAR, max_length=65535)  # 当前分段的文本
    schema.add_field(field_name="segment_id", datatype=DataType.INT64)  # 分段序号
    schema.add_field(field_name="total_segments", datatype=DataType.INT64)  # 总分段数
    schema.add_field(field_name="segment_length", datatype=DataType.INT64)  # 当前分段长度
    schema.add_field(field_name="chunk_overlap", datatype=DataType.INT64)  # 重叠字符数
    
    # 向量字段（对应分段文本的embedding）
    schema.add_field(field_name="index_vector", datatype=DataType.FLOAT_VECTOR, dim=DIM_VALUE)

    # --- 步骤 3: 创建 Milvus 集合 ---
    if milvus_client.has_collection(collection_name=KEYWORD_COLLECTION_NAME):
        print(f"删除已存在的测试集合 '{KEYWORD_COLLECTION_NAME}'...")
        milvus_client.drop_collection(collection_name=KEYWORD_COLLECTION_NAME)
    
    try:
        print(f"正在创建测试集合: {KEYWORD_COLLECTION_NAME}")
        milvus_client.create_collection(collection_name=KEYWORD_COLLECTION_NAME, schema=schema, consistency_level="Strong")
        print(f"集合 '{KEYWORD_COLLECTION_NAME}' 创建成功。")
    except Exception as e:
        print(f"创建集合失败: {e}")
        exit(1)

    # --- 步骤 4: 创建索引 ---
    print("正在创建索引...")
    index_params = milvus_client.prepare_index_params()
    index_params.add_index(field_name="index_vector", index_type="AUTOINDEX", metric_type="L2")
    index_params.add_index(field_name="subject", index_type="MARISA_TRIE")
    index_params.add_index(field_name="segment_id", index_type="STL_SORT")
    
    try:
        milvus_client.create_index(KEYWORD_COLLECTION_NAME, index_params)
        print("索引创建成功。")
    except Exception as e:
        print(f"创建索引失败: {e}")

    # --- 步骤 5: 读取测试数据并处理分段 ---
    data_to_insert = []
    
    try:
        with open(JSON_DATA_SOURCE_PATH, 'r', encoding='utf-8') as json_file:
            json_data = json.load(json_file)
            print(f"读取到测试数据，长度: {len(json_data.get('index', ''))} 字符")

            # 提取原始字段
            subject_val = json_data.get("科目", "")
            module_val = json_data.get("模块", "")
            topic_name_val = json_data.get("题型名称", "")
            topic_source_val = json_data.get("题型来源", "")
            course_name_val = json_data.get("题型来源课程名称", "")
            course_code_val = json_data.get("题型对应课程编号", "")
            index_val = json_data.get("index", "")
            image_path_val = json_data.get("image_path", [])
            path_val = json_data.get("path", "")
            length_val = json_data.get("length", 0)
            page_number_val = json_data.get("page_number", 0)

            # 确保image_path是列表格式
            if not isinstance(image_path_val, list):
                image_path_val = [str(image_path_val)] if image_path_val else []

            print(f"\n原始数据信息:")
            print(f"  题型名称: {topic_name_val}")
            print(f"  原始文本长度: {len(index_val)} 字符")
            print(f"  页面号: {page_number_val}")

            # 使用LangChain分割文本
            if len(index_val) > CHUNK_SIZE:
                print(f"文本长度 {len(index_val)} 超过限制 {CHUNK_SIZE}，使用LangChain进行分割...")
                text_chunks = text_splitter.split_text(index_val)
                print(f"LangChain分割完成，共 {len(text_chunks)} 段")
                
                # 显示分段预览
                for i, chunk in enumerate(text_chunks):
                    print(f"  分段 {i+1}: {len(chunk)} 字符 - {chunk[:50].replace(chr(10), ' ')}...")
                
                print(f"\n开始为每个分段生成embedding...")
                # 为每个分段创建记录
                for i, segment_text in enumerate(text_chunks):
                    print(f"处理分段 {i+1}/{len(text_chunks)}...")
                    
                    try:
                        # 为当前分段生成embedding
                        embedding_list = embedding_function_impl(segment_text)
                        if not embedding_list or not embedding_list[0] or not isinstance(embedding_list[0], list):
                            print(f"  警告：分段 {i+1} 的embedding生成失败，跳过")
                            continue
                        
                        segment_embedding = embedding_list[0]
                        
                        if len(segment_embedding) != DIM_VALUE:
                            print(f"  错误：分段 {i+1} 的embedding维度不匹配，跳过")
                            continue
                        
                        # 创建分段记录（保持所有原始字段不变）
                        segment_record = {
                            "subject": subject_val,
                            "module": module_val,
                            "topic_name": topic_name_val,
                            "topic_source": topic_source_val,
                            "course_name": course_name_val,
                            "course_code": course_code_val,
                            "index": index_val,  # 保持原始完整文本
                            "image_path": image_path_val,
                            "path": path_val,
                            "length": length_val,
                            "page_number": page_number_val,
                            # 分段相关字段
                            "segment_text": segment_text,  # 当前分段的文本
                            "segment_id": i + 1,  # 分段序号
                            "total_segments": len(text_chunks),  # 总分段数
                            "segment_length": len(segment_text),  # 当前分段长度
                            "chunk_overlap": CHUNK_OVERLAP,  # 重叠字符数
                            "index_vector": segment_embedding  # 分段文本的embedding
                        }
                        
                        data_to_insert.append(segment_record)
                        print(f"  分段 {i+1} 处理完成，长度: {len(segment_text)}")
                        
                    except Exception as e:
                        print(f"  处理分段 {i+1} 时出错: {e}")
            else:
                print("文本长度未超过限制，无需分段")
                # 直接处理单段
                try:
                    embedding_list = embedding_function_impl(index_val)
                    segment_embedding = embedding_list[0]
                    
                    segment_record = {
                        "subject": subject_val,
                        "module": module_val,
                        "topic_name": topic_name_val,
                        "topic_source": topic_source_val,
                        "course_name": course_name_val,
                        "course_code": course_code_val,
                        "index": index_val,
                        "image_path": image_path_val,
                        "path": path_val,
                        "length": length_val,
                        "page_number": page_number_val,
                        "segment_text": index_val,
                        "segment_id": 1,
                        "total_segments": 1,
                        "segment_length": len(index_val),
                        "chunk_overlap": 0,
                        "index_vector": segment_embedding
                    }
                    
                    data_to_insert.append(segment_record)
                    print("单段处理完成")
                except Exception as e:
                    print(f"处理单段时出错: {e}")

    except Exception as e:
        print(f"读取或处理JSON文件时出错: {e}")
        exit(1)

    # --- 步骤 6: 插入数据到 Milvus ---
    if data_to_insert:
        print(f"\n准备插入 {len(data_to_insert)} 条分段数据到 Milvus...")
        
        # 显示插入数据预览
        print("\n插入数据预览:")
        for i, record in enumerate(data_to_insert):
            print(f"分段 {i+1}:")
            print(f"  题型名称: {record['topic_name']}")
            print(f"  分段ID: {record['segment_id']}/{record['total_segments']}")
            print(f"  分段长度: {record['segment_length']}")
            print(f"  重叠设置: {record['chunk_overlap']}")
            print(f"  分段文本预览: {record['segment_text'][:100].replace(chr(10), ' ')}...")
            print(f"  原始index保持不变: {record['index'] == json_data.get('index', '')}")
            print(f"  向量维度: {len(record['index_vector'])}")
            print("-" * 50)
        
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
            
            print(f"\n✅ LangChain分段测试成功完成！")
            print(f"数据库: {DB_NAME}")
            print(f"集合: {KEYWORD_COLLECTION_NAME}")
            print(f"原始数据: 1条")
            print(f"分段后数据: {len(data_to_insert)}条")
            print(f"分段策略: 基于LangChain的中文递归分割")
            print(f"分段大小: {CHUNK_SIZE} 字符")
            print(f"重叠大小: {CHUNK_OVERLAP} 字符")
            
        except Exception as e:
            print(f"插入数据时出错: {e}")
    else:
        print("没有数据需要插入")

    print("\n脚本执行完成")