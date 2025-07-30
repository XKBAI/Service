# 查看milvus数据库
from pymilvus import MilvusClient, connections, utility
import os

# --- 配置 (确保与你的写入脚本和目标数据库/集合一致) ---
MILVUS_DB_PATH = "database/keyword_groups_from_json.db" # <--- 确认这是正确的 DB 文件
TARGET_COLLECTION_NAME = "keyword_groups_collection" # <--- 确认这是正确的集合名
# DIM_VALUE = 1024 # 查询时通常不需要维度信息

# --- 主程序 ---
def inspect_milvus_collection():
    print(f"--- 开始检查 Milvus 数据库 ---")
    print(f"目标数据库文件 (Milvus Lite): {MILVUS_DB_PATH}")
    print(f"目标集合名称: {TARGET_COLLECTION_NAME}")

    # 1. 连接到 Milvus
    try:
        client = MilvusClient(uri=MILVUS_DB_PATH)
        print(f"\n成功连接到 Milvus (或指定 Milvus Lite数据库文件): {MILVUS_DB_PATH}")
    except Exception as e:
        print(f"连接到 Milvus 失败: {e}")
        return

    # 2. 列出所有集合
    try:
        collections = client.list_collections()
        print("\n--- 数据库中的所有集合 ---")
        if collections:
            for collection_name in collections:
                print(f"- {collection_name}")
        else:
            print("数据库中没有找到任何集合。")
    except Exception as e:
        print(f"列出集合时发生错误: {e}")
        return

    # 3. 检查目标集合是否存在
    if TARGET_COLLECTION_NAME not in collections:
        print(f"\n错误：目标集合 '{TARGET_COLLECTION_NAME}' 在数据库中不存在。")
        return

    print(f"\n--- 检查目标集合: '{TARGET_COLLECTION_NAME}' ---")

    # 4. 获取集合的统计信息
    try:
        if not client.has_collection(TARGET_COLLECTION_NAME):
            print(f"错误：目标集合 '{TARGET_COLLECTION_NAME}' 未找到 (在尝试获取统计信息前)。")
            return

        stats = client.get_collection_stats(collection_name=TARGET_COLLECTION_NAME)
        print("\n集合统计信息:")
        row_count = stats.get("row_count", "N/A")
        print(f"  行数 (实体数): {row_count}")
        for key, value in stats.items():
            if key != "row_count":
                 print(f"  {key}: {value}")

        if row_count == 0 or row_count == "N/A":
            print("集合中没有数据，或者无法获取行数。")
            return

    except Exception as e:
        print(f"获取集合 '{TARGET_COLLECTION_NAME}' 的统计信息时发生错误: {e}")
        return

    # 5. 检索并打印少量数据
    try:
        print(f"\n尝试确保集合 '{TARGET_COLLECTION_NAME}' 已加载...")
        client.load_collection(TARGET_COLLECTION_NAME)
        print(f"集合 '{TARGET_COLLECTION_NAME}' 加载请求已发送。")

        print("\n尝试检索前5条数据 (如果存在):")
        
        # --- 修改 output_fields ---
        results = client.query(
            collection_name=TARGET_COLLECTION_NAME,
            filter="",
            output_fields=["id", "keyword_group_text", "md_files_list_str"], # 使用正确的字段名
            limit=5 
        )
        # --- 结束修改 ---
        
        if results:
            print(f"\n成功检索到 {len(results)} 条数据:")
            for i, res in enumerate(results):
                print(f"\n--- 记录 {i+1} ---")
                print(f"  ID: {res.get('id', 'N/A')}")
                print(f"  Keyword Group Text: {res.get('keyword_group_text', 'N/A')}") # 匹配输出字段
                print(f"  MD Files List (str): {res.get('md_files_list_str', 'N/A')}")
        else:
            print("未能从集合中检索到任何数据，或者集合为空。")

    except Exception as e:
        print(f"检索数据或加载集合时发生错误: {e}")

    print("\n--- 数据库检查完成 ---")

if __name__ == "__main__":
    inspect_milvus_collection()