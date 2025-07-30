import numpy as np
from pymilvus import MilvusClient
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter, MarkdownHeaderTextSplitter
# from langchain.document_loaders import NotionDirectoryLoader
import re
import re
from typing import List, Optional, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging
import json
import random
import numpy as np
import os
import shutil
from openai import OpenAI
from tqdm import tqdm
from utils import ChineseRecursiveTextSplitter


logger = logging.getLogger(__name__)

class OpenAIEmbeddingFunction:
    def __init__(self, api_key, model="quentinz/bge-large-zh-v1.5:latest"):
        self.client = OpenAI(
            base_url = "http://127.0.0.1:11434/v1",
            api_key = "your_openai_api_key"
        )
        self.model = model

    def __call__(self, input):
        response = self.client.embeddings.create(
            input=input,
            model=self.model
        )
        return [embedding.embedding for embedding in response.data]
import os
import re

def extract_page_number_from_filename(filename):
    """
    从文件名中提取page编号。
    
    参数:
        filename (str): 包含完整路径或不含路径的文件名。
        
    返回:
        tuple: (基础文件名, page编号) 如果找到page编号；否则返回原始文件名和None。
    """
    # 获取文件名（不包含路径）
    base_name = os.path.basename(filename)
    
    # 使用正则表达式匹配文件名中的page编号
    match = re.search(r'(.+)_page_(\d+)\.md$', base_name)
    
    if match:
        # 提取基础文件名和page编号
        base_filename = match.group(1)
        page_number = int(match.group(2))
        return base_filename, page_number
    else:
        # 如果没有找到page编号，则返回原始文件名和None
        return base_name, None











def read_md(directory):
    # 使用Path对象和rglob来查找所有.md文件
    path = Path(directory)
    return list(path.rglob('*.md'))

def read_md_file(file_path):
    """读取单个.md文件的内容"""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


def init_db(DB_PATH=str()):
    shutil.rmtree('data',ignore_errors=True)
    os.makedirs(name='database',exist_ok=True)
    # DB_PATH="database/highschool.db"
    
    client=MilvusClient(uri=DB_PATH)
    return client


    

def main():
    data=list()
    # init db
    DB_PATH="database/highschool.db"
    TABLE_NAME="highschool_all"
    DIM_VALUE=1024
    client = init_db(DB_PATH)
    if client.has_collection(collection_name=TABLE_NAME):
        client.drop_collection(collection_name=TABLE_NAME)
    client.create_collection(
        collection_name=TABLE_NAME,
        dimension=DIM_VALUE
    )


    # init embedding
    embedding_function = OpenAIEmbeddingFunction(api_key="your_openai_api_key")
    
    



    text_splitter = ChineseRecursiveTextSplitter(
        keep_separator=True,
        is_separator_regex=True,
        chunk_size=512,
        chunk_overlap=50
    )
    directory = r"/home/xkb2/Desktop/QY/output"
    # directory = r"./test_md"

    # directory = r"C:\Users\74420\Desktop\RAG\md_output\“12345” 模型问题"

    md_files = read_md(directory)

    # print(md_files)
    i=0
    for md_file in tqdm(md_files):
        # print(md_file)
        data=[]
        filename, page_number = extract_page_number_from_filename(md_file)
        # print(filename, page_number)
        text = read_md_file(md_file)
        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            # print(f'chunk_{i}'+'_'*20)
            # print(chunk)
            # vectorize chunk
            embedding= embedding_function(chunk)[0]
            data.append({'id':i,'text':chunk,'vector':embedding,'filename':filename,'page_number':page_number})
            i=i+1

        # 数据落表
        res = client.insert(collection_name=TABLE_NAME,data=data)
        # print(res)
        
    
main()