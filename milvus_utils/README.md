# 启动milvus
cd ~/Desktop/QY/milvus
docker compose up -d

# 启动api
cd ~/Desktop/QY/milvus_utils
python milvus_query_api.py
localhost:57100/docs