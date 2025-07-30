import os
import uuid
import subprocess
import asyncio
import re
from pathlib import Path
import urllib.parse

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse
import aiofiles
import uvicorn

# --- 配置 ---
PORT = 9000
HOST = "0.0.0.0"

BASE_DIR = Path(__file__).resolve().parent
STORAGE_DIR = BASE_DIR / "storage"
INPUT_MD_DIR = STORAGE_DIR / "input_md"
OUTPUT_PDF_DIR = STORAGE_DIR / "output_pdf"

# 图片根目录
import os
IMAGE_ROOT_DIR = Path(os.getenv("IMAGE_ROOT_DIR", "/app/shared"))

# 创建存储目录
INPUT_MD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PDF_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Markdown转PDF服务")

def cleanup_files(*paths: Path):
    """工具函数，用于删除文件，如果文件不存在则忽略错误。"""
    for path in paths:
        try:
            if path.exists():
                path.unlink()
        except OSError as e:
            print(f"清理文件 {path} 时出错: {e}")

def preprocess_markdown_content(content: str, skip_images: bool = False) -> str:
    """
    预处理Markdown内容
    """
    # 是否跳过图片处理
    if skip_images:
        # 移除双方括号图片 ![[...]]
        content = re.sub(r'!\[\[.*?\]\]', '', content)
        # 移除标准图片引用 ![...](...)
        content = re.sub(r'!\[.*?\]\(.*?\)', '', content)
    else:
        # 处理双方括号图片链接 ![[path/to/image.jpg]] -> ![图片](path/to/image.jpg)
        double_bracket_pattern = r'!\[\[(.*?)\]\]'
        content = re.sub(double_bracket_pattern, lambda m: f'![图片]({urllib.parse.unquote(m.group(1))})', content)
    
    # 完全移除"直达链接"行 - 这是关键修复
    content = re.sub(r'直达链接：\[.*?\]\(.*?\)', '', content)
    
    # 更彻底的处理方式 - 移除整行包含"直达链接："的内容
    lines = content.split('\n')
    filtered_lines = [line for line in lines if '直达链接：' not in line]
    content = '\n'.join(filtered_lines)
    
    return content

def run_pandoc_conversion(md_file_path: Path, pdf_file_path: Path, resource_paths: list[Path]) -> tuple[bool, str]:
    """
    运行Pandoc转换Markdown到PDF
    """
    # 使用非常简单的配置，避免不必要的LaTeX包
    pandoc_cmd = [
        "pandoc",
        str(md_file_path),
        "-o", str(pdf_file_path),
        "--pdf-engine=xelatex",
        # 使用标准Markdown格式，不添加额外的扩展
        "--from=markdown-raw_tex"  # 禁用原始TeX解析，防止误解链接中的内容
    ]
    
    # 处理资源路径
    if resource_paths:
        resource_path_str = os.pathsep.join(str(path) for path in resource_paths)
        pandoc_cmd.extend(["--resource-path", resource_path_str])
    
    # 添加基本的中文支持，但不指定字体，让系统自动选择
    pandoc_cmd.extend([
        "-V", "lang=zh-CN"
    ])
    
    print(f"执行Pandoc命令: {' '.join(pandoc_cmd)}")
    
    try:
        process = subprocess.run(
            pandoc_cmd,
            capture_output=True,
            text=True,
            check=False
        )
        
        if process.returncode != 0:
            error_message = (
                f"Pandoc转换失败 (返回码: {process.returncode}):\n"
                f"STDOUT:\n{process.stdout}\n"
                f"STDERR:\n{process.stderr}"
            )
            return False, error_message
        
        return True, None
    except FileNotFoundError:
        return False, "Pandoc命令未找到。请确保Pandoc已安装并添加到系统PATH中。"
    except Exception as e:
        return False, f"Pandoc执行期间发生错误: {e}"

@app.post("/api/convert/md-to-pdf")
async def convert_md_to_pdf(
    background_tasks: BackgroundTasks,
    markdown_content: str = Form(...),
    skip_images: bool = Form(False),  # 是否跳过图片处理
):
    """
    将Markdown内容转换为PDF
    
    - markdown_content: Markdown文本内容
    - skip_images: 是否跳过图片处理
    
    返回PDF文件
    """
    if not markdown_content.strip():
        raise HTTPException(status_code=400, detail="Markdown内容不能为空")
    
    # 生成任务ID
    task_id = str(uuid.uuid4())
    temp_md_filename = f"{task_id}.md"
    output_pdf_filename = f"{task_id}.pdf"
    
    temp_md_file_path = INPUT_MD_DIR / temp_md_filename
    output_pdf_file_path = OUTPUT_PDF_DIR / output_pdf_filename
    
    # 修改：只清理临时MD文件，保留PDF
    background_tasks.add_task(cleanup_files, temp_md_file_path)
    
    try:
        # 预处理Markdown内容
        processed_markdown = preprocess_markdown_content(markdown_content, skip_images)
        
        # 写入临时Markdown文件
        async with aiofiles.open(temp_md_file_path, "w", encoding="utf-8") as f:
            await f.write(processed_markdown)
        
        # 设置资源路径列表
        resource_paths = [
            IMAGE_ROOT_DIR,           # 主图片根目录
            IMAGE_ROOT_DIR.parent,    # 父目录
            Path.cwd(),               # 当前工作目录
            BASE_DIR                  # 脚本目录
        ]
        
        # 运行Pandoc转换
        success, error_message = await asyncio.to_thread(
            run_pandoc_conversion, 
            temp_md_file_path, 
            output_pdf_file_path,
            resource_paths
        )
        
        if not success:
            raise HTTPException(status_code=500, detail=f"PDF转换失败: {error_message}")
        
        # 打印PDF路径，方便在服务器上找到
        print(f"PDF已生成：{output_pdf_file_path}")
        
        # 返回PDF文件
        return FileResponse(
            path=output_pdf_file_path,
            filename="output.pdf",
            media_type="application/pdf",
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {e}")

@app.post("/api/convert/md-file-to-pdf")
async def convert_md_file_to_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    skip_images: bool = Form(False),  # 是否跳过图片处理
):
    """
    将上传的Markdown文件转换为PDF
    
    - file: 上传的Markdown文件
    - skip_images: 是否跳过图片处理
    
    返回PDF文件
    """
    # 验证文件类型
    if not file.filename.lower().endswith(('.md', '.markdown')):
        raise HTTPException(status_code=400, detail="仅支持Markdown文件(.md, .markdown)")
    
    # 生成任务ID
    task_id = str(uuid.uuid4())
    temp_md_filename = f"{task_id}_{file.filename}"
    output_pdf_filename = f"{task_id}.pdf"
    
    temp_md_file_path = INPUT_MD_DIR / temp_md_filename
    output_pdf_file_path = OUTPUT_PDF_DIR / output_pdf_filename
    
    # 修改：只清理临时MD文件，保留PDF
    background_tasks.add_task(cleanup_files, temp_md_file_path)
    
    try:
        # 读取上传的Markdown文件内容
        content = await file.read()
        markdown_content = content.decode("utf-8")
        
        # 预处理Markdown内容
        processed_markdown = preprocess_markdown_content(markdown_content, skip_images)
        
        # 写入临时Markdown文件
        async with aiofiles.open(temp_md_file_path, "w", encoding="utf-8") as f:
            await f.write(processed_markdown)
        
        # 设置资源路径列表
        resource_paths = [
            IMAGE_ROOT_DIR,           # 主图片根目录
            IMAGE_ROOT_DIR.parent,    # 父目录
            Path.cwd(),               # 当前工作目录
            BASE_DIR                  # 脚本目录
        ]
        
        # 运行Pandoc转换
        success, error_message = await asyncio.to_thread(
            run_pandoc_conversion, 
            temp_md_file_path, 
            output_pdf_file_path,
            resource_paths
        )
        
        if not success:
            raise HTTPException(status_code=500, detail=f"PDF转换失败: {error_message}")
        
        # 打印PDF路径，方便在服务器上找到
        print(f"PDF已生成：{output_pdf_file_path}")
        
        # 返回PDF文件
        return FileResponse(
            path=output_pdf_file_path,
            filename=f"{os.path.splitext(file.filename)[0]}.pdf",
            media_type="application/pdf",
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {e}")

@app.get("/")
async def root():
    return {"message": "Markdown转PDF转换服务已启动，使用POST /api/convert/md-to-pdf或/api/convert/md-file-to-pdf进行转换"}

if __name__ == "__main__":
    root_path = os.getenv("FASTAPI_ROOT_PATH", "")
    uvicorn.run("app:app", host="0.0.0.0", port=PORT, reload=True, root_path=root_path)