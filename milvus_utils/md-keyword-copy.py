import os
import json
import time
import re
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading 

# --- 配置 ---
API_KEY = "sk-OSyEFmRhfr2pwr0K7E62BRZhoohKQ7Yum8044E1Kg9IW21Ul" # 请替换为您的有效API Key
BASE_URL = "http://192.168.2.4:3000/v1"  # 请替换为您的模型服务地址
MODEL_NAME = "gpt-3.5-turbo"             # 请替换为您要使用的模型名称

# ! 请务必修改为您的实际 .md 文件根目录 !
TARGET_MD_DIRECTORY = "/home/xkb2/Desktop/QY/output" 
# 所有结果将合并保存到此JSON文件
SINGLE_OUTPUT_JSON_FILE = "./all_md_keywords_recursive.json" 
MAX_WORKERS = 64 # 并发处理的最大线程数

# --- 初始化 OpenAI 客户端 ---
try:
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    print(f"OpenAI 客户端初始化成功，目标 API: {BASE_URL}, 模型: {MODEL_NAME}")
except Exception as e:
    print(f"OpenAI 客户端初始化失败: {e}")
    exit()

# --- 更鲁棒的JSON提取函数 ---
def extract_json_candidate(text):
    matches = re.findall(r'\{[\s\S]*?\}', text)
    for m in matches[::-1]:  # 从后往前找
        if '"keywords"' in m or "'keywords'" in m:
            return m
    return text.strip()

# --- 核心功能：从 LLM 获取关键词 ---
def get_keywords_from_llm(md_content: str, md_filename: str) -> dict:
    system_prompt = (
        "你是一位高度智能的助手，专注于文档分析和关键词提取。"
        "你的任务是从提供的文本中识别出最重要和最相关的中文关键词。"
        "这些关键词应准确代表文本的核心主题和概念。"
        "请严格按照JSON格式返回提取的关键词。"
        "JSON对象必须包含一个名为 \"keywords\" 的键，其值必须是一个字符串列表，关键词数量最多不超过7个，不少于1个。"
        "列表中的每个字符串都是一个中文关键词。不要在JSON结构之外添加任何解释性文字或介绍性文本，也不要包含任何思考过程或XML/HTML标签。"
        "你的回答必须直接是JSON对象本身，例如：{\"keywords\": [\"关键词1\", \"关键词2\"]}"
    )
    user_prompt_template = "请从以下标题为 '{filename}' 的文档中提取中文关键词：\n\n{content}"
    
    base_result = {"filename": md_filename, "keywords": []} 

    try:
        thread_id = threading.get_ident()
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt_template.format(filename=md_filename, content=md_content)},
            ],
            temperature=0.2,
            stream=False
        )
        llm_output = response.choices[0].message.content
        print(f"DEBUG: LLM对'{md_filename}'的原始输出: '''{llm_output}'''")
        
        cleaned_llm_output = llm_output.strip()
        json_string_candidate = extract_json_candidate(cleaned_llm_output)
        print(f"DEBUG: 用正则提取的JSON候选: '''{json_string_candidate}'''")

        # 处理代码块包裹
        if json_string_candidate.startswith("```json"):
            json_string_candidate = json_string_candidate[len("```json"):].strip()
        if json_string_candidate.startswith("```"):
            json_string_candidate = json_string_candidate[len("```"):].strip()
        if json_string_candidate.endswith("```"):
            json_string_candidate = json_string_candidate[:-len("```")].strip()

        try:
            keywords_data = json.loads(json_string_candidate)
            if isinstance(keywords_data, dict) and "keywords" in keywords_data and isinstance(keywords_data["keywords"], list):
                base_result["keywords"] = keywords_data["keywords"]
                return base_result
            else:
                base_result["error"] = f"LLM对'{md_filename}'的响应是有效JSON但格式不符(缺'keywords'列表或非字典)"
                base_result["raw_response"] = llm_output
                return base_result
        except json.JSONDecodeError as json_e:
            base_result["error"] = f"无法解析LLM对'{md_filename}'的JSON响应.错误:{json_e}.候选JSON串:'''{json_string_candidate}'''"
            base_result["raw_response"] = llm_output
            print(f"ERROR-JSONDecode: LLM对'{md_filename}'的原始输出导致JSON解析失败: '''{llm_output}'''. 候选JSON串: '''{json_string_candidate}'''")
            return base_result
    except Exception as e:
        base_result["error"] = f"调用API处理'{md_filename}'时发生错误:{e}"
        base_result["raw_response"] = None
        return base_result

# --- 单个文件处理任务函数 ---
def process_single_md_file_task(md_filename: str, md_filepath: str) -> dict:
    """
    读取、处理单个MD文件。
    返回一个字典，包含文件名、其相对路径（相对于TARGET_MD_DIRECTORY）、关键词列表，或者错误信息。
    支持最多3次重试，只有连续3次都失败才记录错误。
    """
    # 计算相对于TARGET_MD_DIRECTORY的路径，用于结果中更好地定位文件
    try:
        relative_path = os.path.relpath(md_filepath, TARGET_MD_DIRECTORY)
    except ValueError: # 如果md_filepath不在TARGET_MD_DIRECTORY下（理论上不应发生）
        relative_path = md_filename 

    base_info = {"filename": md_filename, "relative_path": relative_path, "keywords": []}

    try:
        with open(md_filepath, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        if not md_content.strip():
            base_info["status"] = "File is empty"
            return base_info

        max_retries = 3
        last_result = None
        for attempt in range(1, max_retries + 1):
            extracted_data = get_keywords_from_llm(md_content, md_filename)
            last_result = extracted_data
            if "error" not in extracted_data:
                break  # 成功，跳出循环
            else:
                print(f"警告: 第{attempt}次解析'{md_filename}'失败，错误信息: {extracted_data.get('error')}")
                if attempt < max_retries:
                    time.sleep(1)  # 可选：重试前稍作等待

        # 将提取的数据合并到 base_info 中
        base_info["keywords"] = last_result.get("keywords", [])
        if "error" in last_result:
            base_info["error"] = last_result["error"]
            base_info["status"] = "Retry failed"
        if "raw_response" in last_result:
            base_info["raw_response"] = last_result["raw_response"]
        return base_info

    except FileNotFoundError:
        base_info["error"] = f"File not found at {md_filepath}"
        base_info["status"] = "File not found"
        return base_info
    except Exception as e:
        base_info["error"] = f"Unexpected processing error in task for {md_filepath}: {str(e)}"
        base_info["status"] = "Task error"
        return base_info

# --- 主脚本逻辑 (支持递归扫描并将所有结果保存到一个JSON文件) ---
def process_md_files_and_extract_keywords():
    print(f"--- 开始关键词提取流程 (并发模式, 支持递归扫描) ---")
    print(f"扫描目标根文件夹: {TARGET_MD_DIRECTORY}")
    print(f"所有结果将合并保存到: {SINGLE_OUTPUT_JSON_FILE}")
    print(f"最大并发线程数: {MAX_WORKERS}")

    if not os.path.isdir(TARGET_MD_DIRECTORY):
        print(f"错误: 目标文件夹 '{TARGET_MD_DIRECTORY}' 不存在或不是一个目录。请检查路径。")
        return

    output_file_dir = os.path.dirname(SINGLE_OUTPUT_JSON_FILE)
    if output_file_dir and not os.path.exists(output_file_dir): # 确保输出JSON文件的目录存在
        try:
            os.makedirs(output_file_dir)
            print(f"已创建输出目录: {output_file_dir}")
        except Exception as e:
            print(f"错误: 无法创建输出目录 '{output_file_dir}': {e}")
            return

    md_files_to_process_info = [] 
    print(f"正在从 '{TARGET_MD_DIRECTORY}' (及其子目录) 递归扫描 .md 文件...")
    for root, dirs, files in os.walk(TARGET_MD_DIRECTORY):
        for filename in files:
            if filename.lower().endswith(".md"): 
                filepath = os.path.join(root, filename)
                md_files_to_process_info.append({"name": filename, "path": filepath})
    
    if not md_files_to_process_info:
        print(f"在 '{TARGET_MD_DIRECTORY}' 及其子目录中未找到任何 .md 文件。")
        return

    total_files = len(md_files_to_process_info)
    print(f"扫描完成。找到 {total_files} 个 .md 文件准备处理。")

    start_time = time.time()
    completed_count = 0
    success_count = 0 
    empty_file_count = 0
    error_count = 0
    
    all_results_list = [] 

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for file_info in md_files_to_process_info:
            futures.append(executor.submit(process_single_md_file_task, file_info["name"], file_info["path"]))

        for future in as_completed(futures):
            completed_count += 1
            try:
                result_data = future.result() # result_data 是 process_single_md_file_task 返回的字典
                all_results_list.append(result_data) 
                
                filename_in_result = result_data.get("filename", "未知文件")
                relative_path_in_result = result_data.get("relative_path", "")

                progress = (completed_count / total_files) * 100
                # 打印时可以包含相对路径以便更好地区分同名文件
                display_name = os.path.join(relative_path_in_result) if relative_path_in_result != filename_in_result else filename_in_result
                print(f"\n[进度: {completed_count}/{total_files} ({progress:.2f}%)] 文件 '{display_name}' 处理完成。")

                if result_data.get("status") == "File is empty":
                    empty_file_count += 1
                    print(f"  -> 跳过: 文件为空。")
                elif result_data.get("status") == "Retry failed":
                    error_count += 1
                    print(f"  -> 跳过: 连续3次解析失败。错误: {result_data.get('error')}")
                elif "error" in result_data and result_data["error"]:
                    error_count += 1
                    print(f"  -> 错误: {result_data['error']}")
                else: 
                    success_count += 1
                    print(f"  -> 成功: 关键词已提取: {result_data.get('keywords', [])}")
            
            except Exception as e: 
                error_count += 1
                # 这种错误通常是 future.result() 本身抛出的，表示任务执行中发生了严重错误
                # 尝试从 future 对象中获取一些信息，如果任务函数有自定义的异常处理并返回了文件名
                # 但更可能的是 filename 无法获取，所以用占位符
                placeholder_error = {"filename": "未知文件(任务崩溃)", "relative_path": "", "keywords": [], "error": f"线程任务严重失败: {str(e)}"}
                all_results_list.append(placeholder_error)
                print(f"\n[进度: {completed_count}/{total_files} ({progress:.2f}%)] 处理某个文件时线程任务严重失败: {e}")

    try:
        with open(SINGLE_OUTPUT_JSON_FILE, 'w', encoding='utf-8') as outfile:
            json.dump(all_results_list, outfile, ensure_ascii=False, indent=4)
        print(f"\n所有结果已成功保存到文件: '{SINGLE_OUTPUT_JSON_FILE}'")
    except Exception as e:
        print(f"\n错误: 保存所有结果到 '{SINGLE_OUTPUT_JSON_FILE}' 失败: {e}")

    end_time = time.time()
    total_time = end_time - start_time

    print("\n--- 并发关键词提取流程结束 ---")
    print(f"总文件数: {total_files}")
    print(f"成功提取关键词: {success_count}")
    print(f"空文件被跳过: {empty_file_count}")
    print(f"重试失败被跳过: {error_count}")
    print(f"总耗时: {total_time:.2f} 秒")

# --- 运行主程序 ---
if __name__ == "__main__":
    print(f"开始处理，将从 '{os.path.abspath(TARGET_MD_DIRECTORY)}' 递归扫描 .md 文件。")
    print(f"结果将保存到 '{os.path.abspath(SINGLE_OUTPUT_JSON_FILE)}'.")
    process_md_files_and_extract_keywords()