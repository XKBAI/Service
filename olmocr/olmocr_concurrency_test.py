#!/usr/bin/env python3
# encoding: utf-8
"""
OLMOCR API并发测试脚本
测试API的并发处理能力，可配置并发数量
"""

import asyncio
import aiohttp
import os
import time
import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import statistics
from tqdm import tqdm


class OLMOCRConcurrencyTest:
    def __init__(self, api_base_url, concurrent_requests, test_files_dir):
        self.api_base_url = api_base_url
        self.concurrent_requests = concurrent_requests
        self.test_files_dir = Path(test_files_dir)
        self.results = []
        self.test_files = []
        
        # 查找测试文件
        self._find_test_files()
        
        # 检查测试文件
        if not self.test_files:
            raise ValueError(f"在 {test_files_dir} 中未找到测试文件")
        
        print(f"找到以下测试文件: {', '.join([os.path.basename(f) for f in self.test_files])}")
    
    def _find_test_files(self):
        """查找测试目录中的数字编号文件：1.pdf到10.png等从1到10编号的文件"""
        numbered_files = []
        for i in range(1, 11):  # 寻找编号为1到10的文件
            for ext in ['.pdf', '.png', '.jpg', '.jpeg']:
                file_path = self.test_files_dir / f"{i}{ext}"
                if file_path.exists():
                    numbered_files.append(str(file_path))
                    break  # 每个编号只取一个文件（按扩展名优先级）
        
        # 按照数字排序（1.pdf, 2.png, 3.jpg...）而不是字母顺序
        self.test_files = sorted(numbered_files, key=lambda x: int(os.path.basename(x).split('.')[0]))
        
        # 确保找到了足够的文件
        if len(self.test_files) < self.concurrent_requests:
            print(f"警告：找到的测试文件数量({len(self.test_files)})少于请求的并发数({self.concurrent_requests})。")
            print(f"将循环使用已有的{len(self.test_files)}个文件进行测试。")
    
    async def _send_request(self, session, file_path, request_id):
        """发送单个API请求"""
        start_time = time.time()
        file_name = os.path.basename(file_path)
        
        try:
            # 构建表单数据，包含文件
            form_data = aiohttp.FormData()
            form_data.add_field('file', 
                                open(file_path, 'rb'),
                                filename=file_name,
                                content_type=self._get_content_type(file_path))
            
            # 发送请求
            async with session.post(f"{self.api_base_url}/olmocr/process", data=form_data) as response:
                elapsed = time.time() - start_time
                
                # 读取响应
                status = response.status
                response_data = await response.json()
                job_id = response_data.get('id', 'unknown')
                
                # 打印进度信息
                print(f"请求 #{request_id}: 文件={file_name}, 状态={status}, 耗时={elapsed:.2f}秒, 任务ID={job_id}")
                
                # 如果是异步处理，我们需要轮询任务状态
                if status == 200 and 'id' in response_data:
                    job_result = await self._poll_job_status(session, job_id, request_id, file_name)
                    
                    # 明确区分三种状态:
                    # 1. 请求提交成功
                    # 2. 任务处理完成
                    # 3. 获取到有效结果
                    request_successful = status == 200
                    processing_completed = job_result.get('status') == 'completed'
                    result_valid = job_result.get('result_valid', False)
                    
                    return {
                        'request_id': request_id,
                        'file_name': file_name,
                        'initial_status': status,
                        'final_status': status if (processing_completed and result_valid) else 0,
                        'initial_response_time': elapsed,
                        'job_id': job_id,
                        'job_result': job_result,
                        'request_successful': request_successful,
                        'processing_completed': processing_completed,
                        'result_valid': result_valid
                    }
                
                return {
                    'request_id': request_id,
                    'file_name': file_name,
                    'status': status,
                    'initial_response_time': elapsed,
                    'job_id': job_id,
                    'result': response_data
                }
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"请求 #{request_id} 失败: {str(e)}")
            return {
                'request_id': request_id,
                'file_name': file_name,
                'status': -1,
                'initial_response_time': elapsed,
                'error': str(e)
            }
    
    async def _poll_job_status(self, session, job_id, request_id, file_name, max_polls=60, poll_interval=5):
        """轮询任务状态直到完成或超时"""
        start_time = time.time()
        polls = 0
        job_status_history = []
        
        while polls < max_polls:
            try:
                async with session.get(f"{self.api_base_url}/olmocr/jobs/{job_id}") as response:
                    if response.status != 200:
                        print(f"请求 #{request_id}: 轮询失败, 状态码={response.status}")
                        break
                    
                    job_data = await response.json()
                    status = job_data.get('status', 'unknown')
                    
                    # 记录状态变化历史
                    timestamp = time.time()
                    job_status_history.append({
                        "timestamp": timestamp,
                        "elapsed": timestamp - start_time,
                        "status": status,
                        "queue_position": job_data.get('queue_position')
                    })
                    
                    if status == 'completed':
                        total_time = time.time() - start_time
                        print(f"请求 #{request_id}: 文件={file_name}, 任务={job_id}, 已完成, 总耗时={total_time:.2f}秒")
                        
                        # 尝试获取结果数据
                        try:
                            result_response = await session.get(f"{self.api_base_url}/olmocr/results/{job_id}")
                            if result_response.status == 200:
                                result_data = await result_response.json()
                                
                                # 根据示例结果格式验证结果是否有效
                                is_valid = self._validate_result(result_data)
                                
                                job_data["result_data"] = result_data
                                job_data["result_valid"] = is_valid
                                
                                if is_valid:
                                    print(f"请求 #{request_id}: 获取到有效OCR结果，文本长度: {len(result_data.get('text', ''))}")
                                else:
                                    print(f"请求 #{request_id}: 获取到结果但无效")
                            else:
                                job_data["result_data"] = {"error": f"无法获取结果, 状态码: {result_response.status}"}
                                job_data["result_valid"] = False
                        except Exception as e:
                            job_data["result_data"] = {"error": f"获取结果时出错: {str(e)}"}
                            job_data["result_valid"] = False
                        
                        job_data["status_history"] = job_status_history
                        job_data["total_processing_time"] = total_time
                        return job_data
                    elif status == 'failed':
                        total_time = time.time() - start_time
                        print(f"请求 #{request_id}: 文件={file_name}, 任务={job_id}, 处理失败, 总耗时={total_time:.2f}秒")
                        print(f"请求 #{request_id}: 失败原因: {job_data.get('error', '未知')}")
                        job_data["status_history"] = job_status_history
                        job_data["total_processing_time"] = total_time
                        job_data["result_valid"] = False
                        return job_data
                    else:
                        queue_pos = job_data.get('queue_position', '未知')
                        print(f"请求 #{request_id}: 文件={file_name}, 任务={job_id}, 状态={status}, 队列位置={queue_pos}")
            
            except Exception as e:
                print(f"请求 #{request_id}: 轮询任务 {job_id} 时出错: {str(e)}")
            
            polls += 1
            await asyncio.sleep(poll_interval)
        
        print(f"请求 #{request_id}: 轮询任务 {job_id} 超时")
        return {"status": "timeout", "status_history": job_status_history, "result_valid": False}
    
    def _validate_result(self, result_data):
        """验证OCR结果是否有效"""
        # 基于示例结果格式验证结果有效性
        
        # 验证必要字段是否存在
        required_fields = ["id", "text", "source", "metadata", "attributes"]
        for field in required_fields:
            if field not in result_data:
                return False
        
        # 验证文本内容是否有实质内容
        text = result_data.get("text", "")
        if not text or len(text.strip()) < 20:  # 至少20个字符才算有效内容
            return False
        
        # 验证元数据是否包含处理信息
        metadata = result_data.get("metadata", {})
        if not metadata:
            return False
        
        # 检查是否包含页面总数、token计数等关键指标
        required_metadata = ["pdf-total-pages", "total-input-tokens", "total-output-tokens"]
        for field in required_metadata:
            if field not in metadata:
                return False
        
        # 验证页面信息是否存在
        attributes = result_data.get("attributes", {})
        if "pdf_page_numbers" not in attributes or not attributes["pdf_page_numbers"]:
            return False
        
        return True
    
    def _get_content_type(self, file_path):
        """根据文件后缀获取内容类型"""
        suffix = os.path.splitext(file_path)[1].lower()
        content_types = {
            '.pdf': 'application/pdf',
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg'
        }
        return content_types.get(suffix, 'application/octet-stream')
    
    async def run_test(self):
        """运行并发测试"""
        print(f"\n开始并发测试: {self.concurrent_requests} 个并发请求")
        print(f"API基础URL: {self.api_base_url}")
        print(f"测试文件目录: {self.test_files_dir}")
        print("-" * 80)
        
        start_time = time.time()
        
        # 创建请求列表 - 循环使用测试文件来达到所需的并发数
        requests = []
        for i in range(self.concurrent_requests):
            # 循环使用测试文件
            file_index = i % len(self.test_files)
            file_path = self.test_files[file_index]
            requests.append((file_path, i+1))
        
        # 设置进度条
        progress_bar = tqdm(total=len(requests), desc="发送请求", unit="请求")
        
        # 使用aiohttp.ClientSession进行并发请求
        async with aiohttp.ClientSession() as session:
            # 创建异步任务并添加回调以更新进度条
            tasks = []
            for file_path, req_id in requests:
                task = asyncio.create_task(self._send_request(session, file_path, req_id))
                task.add_done_callback(lambda _: progress_bar.update(1))
                tasks.append(task)
            
            # 等待所有任务完成
            results = await asyncio.gather(*tasks)
            self.results = results
        
        progress_bar.close()
        
        # 计算总耗时
        total_time = time.time() - start_time
        
        # 输出测试汇总信息
        self._print_summary(total_time)
        
        # 收集任务状态分布
        self.job_status_summary = {}
        for result in self.results:
            # 使用job_result中的status
            if "job_result" in result and isinstance(result["job_result"], dict) and "status" in result["job_result"]:
                status = result["job_result"]["status"]
                self.job_status_summary[status] = self.job_status_summary.get(status, 0) + 1
        
        # 生成可视化
        self.generate_visualizations()
        
        # 打印详细报告
        self.print_detailed_report()
        
        return self.results
    
    def _print_summary(self, total_time):
        """打印测试结果摘要"""
        # 计算统计信息
        response_times = [r.get('initial_response_time', 0) for r in self.results if 'initial_response_time' in r]
        success_count = sum(1 for r in self.results if r.get('result_valid', False))
        error_count = len(self.results) - success_count
        
        # 计算三个阶段的成功率
        request_successful_count = sum(1 for r in self.results if r.get('request_successful', False))
        processing_completed_count = sum(1 for r in self.results if r.get('processing_completed', False))
        result_valid_count = sum(1 for r in self.results if r.get('result_valid', False))
        
        self.summary_data = {
            "总请求数": len(self.results),
            "并发数": self.concurrent_requests,
            "请求提交成功数": request_successful_count,
            "请求提交成功率": f"{request_successful_count/len(self.results)*100:.1f}%",
            "处理完成数": processing_completed_count,
            "处理完成率": f"{processing_completed_count/len(self.results)*100:.1f}%",
            "有效结果数": result_valid_count,
            "有效结果率": f"{result_valid_count/len(self.results)*100:.1f}%",
            "总耗时": f"{total_time:.2f}秒"
        }
        
        if response_times:
            self.summary_data.update({
                "平均响应时间": f"{statistics.mean(response_times):.2f}秒",
                "最小响应时间": f"{min(response_times):.2f}秒",
                "最大响应时间": f"{max(response_times):.2f}秒"
            })
            if len(response_times) > 1:
                self.summary_data["响应时间标准差"] = f"{statistics.stdev(response_times):.2f}秒"
        
        # 按文件类型分组的统计
        file_stats = {}
        for result in self.results:
            file_name = result.get('file_name', 'unknown')
            if file_name not in file_stats:
                file_stats[file_name] = {
                    "请求数": 0,
                    "成功数": 0,
                    "响应时间总和": 0
                }
            
            file_stats[file_name]["请求数"] += 1
            # 使用result_valid判断是否成功
            if result.get('result_valid', False):
                file_stats[file_name]["成功数"] += 1
                file_stats[file_name]["响应时间总和"] += result.get('initial_response_time', 0)
        
        # 计算每个文件的平均响应时间
        for file_name, stats in file_stats.items():
            if stats["成功数"] > 0:
                stats["平均响应时间"] = f"{stats['响应时间总和'] / stats['成功数']:.2f}秒"
            else:
                stats["平均响应时间"] = "N/A"
        
        self.file_stats = file_stats
        
        # 生成详细的失败请求信息
        self.failed_requests = [r for r in self.results if not r.get('result_valid', False)]
        
        # 简单打印摘要信息
        print("\n" + "=" * 80)
        print(f"并发测试完成 - {self.concurrent_requests} 个并发请求")
        print(f"总耗时: {total_time:.2f}秒")
        print(f"成功请求: {success_count}/{len(self.results)} ({success_count/len(self.results)*100:.1f}%)")
        print(f"失败请求: {error_count}/{len(self.results)} ({error_count/len(self.results)*100:.1f}%)")
        
        if response_times:
            print(f"平均响应时间: {statistics.mean(response_times):.2f}秒")
            print(f"最小响应时间: {min(response_times):.2f}秒")
            print(f"最大响应时间: {max(response_times):.2f}秒")
            if len(response_times) > 1:
                print(f"响应时间标准差: {statistics.stdev(response_times):.2f}秒")
        
        print("\n详细报告将在测试完成后显示")
        print("=" * 80)
        
    def generate_visualizations(self):
        """生成测试结果可视化图表"""
        # 如果没有足够的结果数据，跳过可视化
        if len(self.results) < 2:
            print("数据不足，无法生成可视化图表")
            return
        
        # 创建结果目录
        results_dir = "olmocr_test_results"
        os.makedirs(results_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 转换结果为DataFrame以便于分析
        df = pd.DataFrame(self.results)
        
        # 保存原始数据
        df.to_csv(f"{results_dir}/raw_data_{timestamp}.csv", index=False)
        
        # 只对成功的请求进行分析
        successful_df = df[df["result_valid"] == True]
        if successful_df.empty:
            print("没有成功的请求，跳过可视化")
            return
        
        # 设置图表样式
        plt.style.use('ggplot')
        
        # 图1: 响应时间分布
        plt.figure(figsize=(10, 6))
        plt.hist(successful_df["initial_response_time"], bins=20, alpha=0.7, color='blue')
        mean_time = successful_df["initial_response_time"].mean()
        plt.axvline(mean_time, color='red', linestyle='dashed', linewidth=1)
        plt.title(f'API响应时间分布 (平均: {mean_time:.2f}秒)')
        plt.xlabel('响应时间 (秒)')
        plt.ylabel('请求数量')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{results_dir}/response_time_histogram_{timestamp}.png")
        
        # 图2: 按文件类型的响应时间对比
        plt.figure(figsize=(10, 6))
        file_groups = successful_df.groupby("file_name")["initial_response_time"]
        
        if len(file_groups) > 1:  # 只有当有多个文件类型时才生成此图
            plt.boxplot([group for _, group in file_groups], 
                      labels=[filename for filename, _ in file_groups])
            plt.title('各文件类型的响应时间对比')
            plt.ylabel('响应时间 (秒)')
            plt.xlabel('文件类型')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{results_dir}/file_response_time_comparison_{timestamp}.png")
        
        # 图3: 请求响应时间序列
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(successful_df)), successful_df["initial_response_time"], marker='o', 
               markersize=3, linestyle='-', alpha=0.7)
        plt.axhline(mean_time, color='red', linestyle='dashed', 
                  label=f'平均响应时间: {mean_time:.2f}秒')
        plt.title('请求响应时间序列')
        plt.xlabel('请求序号')
        plt.ylabel('响应时间 (秒)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{results_dir}/response_time_sequence_{timestamp}.png")
        
        print(f"\n可视化图表已保存到 {results_dir} 目录")
        
    def print_detailed_report(self):
        """打印详细的测试报告"""
        print("\n" + "="*60)
        print(" "*20 + "OLMOCR API 并发测试报告")
        print("="*60 + "\n")
        
        # 打印总体统计
        print("【总体统计】")
        for key, value in self.summary_data.items():
            print(f"  {key}: {value}")
        
        # 打印按文件类型统计
        print("\n【按文件类型统计】")
        for file_name, stats in self.file_stats.items():
            print(f"  文件: {file_name}")
            print(f"    请求数: {stats['请求数']}")
            print(f"    成功数: {stats['成功数']}")
            print(f"    成功率: {stats['成功数']/stats['请求数']*100:.1f}%")
            print(f"    平均响应时间: {stats['平均响应时间']}")
            print()
        
        # 打印队列状态信息（如果有）
        if hasattr(self, 'queue_info') and self.queue_info:
            print("\n【队列状态信息】")
            for key, value in self.queue_info.items():
                print(f"  {key}: {value}")
        
        # 打印任务处理情况
        if hasattr(self, 'job_status_summary'):
            print("\n【任务处理状态】")
            for status, count in sorted(self.job_status_summary.items()):
                print(f"  {status}: {count} 个任务")
        
        # 打印失败请求（如果有）
        if self.failed_requests:
            print(f"\n【失败请求详情】(显示前 {min(10, len(self.failed_requests))} 条)")
            for i, failure in enumerate(self.failed_requests[:10]):
                print(f"  {i+1}. 请求ID: {failure.get('request_id', '未知')}, 文件: {failure.get('file_name', '未知')}")
                # 增加更多失败信息
                job_result = failure.get('job_result', {})
                status = job_result.get('status', '未知')
                error = job_result.get('error', '未知')
                print(f"     状态: {status}, 错误: {error}")
        
        print("\n" + "="*60)
    
    async def check_queue_info(self, session):
        """检查队列信息"""
        try:
            async with session.get(f"{self.api_base_url}/olmocr/queue") as response:
                if response.status == 200:
                    queue_data = await response.json()
                    self.queue_info = {
                        "队列大小": queue_data.get('queue_size', '未知'),
                        "处理中任务": queue_data.get('processing', '未知'),
                        "最大并发": queue_data.get('max_concurrent', '未知'),
                        "最大并行": queue_data.get('max_parallel', '未知'),
                        "可用槽位": queue_data.get('available_slots', '未知')
                    }
                    print("\n当前队列信息:")
                    for key, value in self.queue_info.items():
                        print(f"  {key}: {value}")
                    return self.queue_info
                else:
                    print(f"获取队列信息失败, 状态码: {response.status}")
                    return None
        except Exception as e:
            print(f"获取队列信息时出错: {str(e)}")
            return None
    
    def save_results(self, output_file=None):
        """将测试结果保存到文件"""
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"olmocr_concurrency_test_{self.concurrent_requests}_{timestamp}.json"
        
        result_data = {
            "test_config": {
                "api_base_url": self.api_base_url,
                "concurrent_requests": self.concurrent_requests,
                "test_files_dir": str(self.test_files_dir),
                "test_files": [os.path.basename(f) for f in self.test_files],
                "timestamp": datetime.now().isoformat()
            },
            "results": self.results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n测试结果已保存到: {output_file}")


async def main():
    parser = argparse.ArgumentParser(description='OLMOCR API并发测试工具')
    parser.add_argument('--api-url', default='http://localhost:57004', help='OLMOCR API基础URL')
    parser.add_argument('--concurrent', type=int, default=10, help='并发请求数量')
    parser.add_argument('--test-dir', default='./testdata', help='测试文件目录')
    parser.add_argument('--output', help='输出结果文件名')
    parser.add_argument('--poll-interval', type=int, default=5, help='轮询任务状态的间隔时间(秒)')
    parser.add_argument('--max-polls', type=int, default=60, help='轮询任务状态的最大次数')
    parser.add_argument('--no-visualize', action='store_true', help='禁用图表生成')
    
    args = parser.parse_args()
    
    print(f"\n{'='*50}")
    print(f" OLMOCR API 并发测试工具 v1.0 ")
    print(f"{'='*50}")
    print(f"API地址: {args.api_url}")
    print(f"并发请求数: {args.concurrent}")
    print(f"测试文件目录: {args.test_dir}")
    print(f"轮询间隔: {args.poll_interval}秒")
    print(f"最大轮询次数: {args.max_polls}")
    
    try:
        # 创建测试实例
        test = OLMOCRConcurrencyTest(args.api_url, args.concurrent, args.test_dir)
        
        # 在测试开始前检查队列信息
        print("\n检查API队列状态...")
        async with aiohttp.ClientSession() as session:
            await test.check_queue_info(session)
        
        # 运行测试
        print("\n开始执行并发测试...")
        await test.run_test()
        
        # 测试结束后再次检查队列信息
        print("\n测试完成，再次检查API队列状态...")
        async with aiohttp.ClientSession() as session:
            await test.check_queue_info(session)
        
        # 保存结果
        if args.output:
            test.save_results(args.output)
        else:
            # 使用默认文件名保存
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"olmocr_concurrency_test_{args.concurrent}_{timestamp}.json"
            test.save_results(output_file)
        
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        print(f"\n测试过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())