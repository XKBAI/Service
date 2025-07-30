#!/bin/bash

# User服务启动脚本

set -e

echo "开始启动User服务..."

# 检查必要的目录
mkdir -p /app/shared /app/logs

# 设置权限
chmod -R 755 /app/shared /app/logs

# 等待MySQL数据库准备就绪
echo "等待MySQL数据库连接..."
python3 -c "
import mysql.connector
import time
import os
import sys

def test_db_connection():
    max_retries = 30
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            conn = mysql.connector.connect(
                host=os.environ.get('DB_HOST', 'mysql-db'),
                user=os.environ.get('DB_USER', 'root'),
                password=os.environ.get('DB_PASSWORD', 'qwe123asd'),
                database=os.environ.get('DB_NAME', 'CHAT')
            )
            conn.close()
            print('数据库连接成功！')
            sys.exit(0)
        except Exception as e:
            print(f'等待MySQL数据库连接中... (尝试 {retry_count + 1}/{max_retries})')
            retry_count += 1
            time.sleep(3)
    
    print('数据库连接失败，超过最大重试次数')
    sys.exit(1)

test_db_connection()
"

# 等待一下让系统准备好
sleep 2

echo "启动User API服务..."

# 启动服务
exec python app.py