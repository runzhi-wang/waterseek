"""
使用语言模型从md/txt中提取结构化数据，导出为xlsx - 并行版本（单独保存）
支持随机抽样指定数量的文件
"""

import json
import re
from openai import OpenAI
import pandas as pd
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
import traceback
import random
import numpy as np
import os
from dotenv import load_dotenv

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../../extraction_parallel.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# 系统提示词（全局常量）
requirement = """你的任务是从给定的文献中提取结构化数据，严格按照以下要求输出。

# 输出格式（必须严格遵循此JSON结构）
{
  "pollutant_name": "污染物的全称，而非缩写，且第一个单词的首字母大写",
  "pollutant_concentration": 数值（单位mmol/L）,
  "electrolyte": "电解质类型",
  "electrolyte_concentration": 数值（单位mmol/L）,
  "current_density": 数值（单位mA/cm2）,
  "pH": 数值（0-14）,
  "temperature": 数值（单位°C）,
  "solution_volume": 数值（单位mL）,
  "anode": "阳极材料（格式：基底/涂层）",
  "cathode": "阴极材料",
  "anode_area": 数值（单位cm2）,
  "cathode_area": 数值（单位cm2）,
  "electrode_distance": 数值（单位cm）,
  "rate_constant": 数值（单位min-1）,
}

# 数据提取的重要规则
1. 多组实验数据请输出为JSON数组，务必找全所有符合条件的实验数据，不要遗漏
2. 缺失数据用null表示，严禁编造文献中不存在的数据
3. 自动转换单位：mg/L→mmol/L，g/L→mmol/L，M→mmol/L, mM→mmol/L, ppm→mmol/L
4. 电极材料必须包含基底和涂层（如Ti/PbO2）
5. 阳极/阴极的面积也可能以尺寸形成给出（如1cm x 1cm），需要折算成cm2
6. "pollutant_SMILE"这两列"pollutant_formula"原文中没有，需要你根据pollutant_name进行精准转换
7. 所有的回答必须是英文
8. 只提取纯电化学氧化/电催化降解污染物的数据，不要提取电化学与其他技术（如光催化、PMS活化等）结合的耦合工艺所产生的数据组
9. anode_type中，常见的active anode有：RuO2，IrO2，Pt，Fe，碳材料。常见的nonacrive anode有：BDD，TixOy，SnO2，PbO2
10. electrolyte请统一为对应的化学式，例如Na2SO4，而不要出现名称（如sodium sulphate）
11. 常见的非活性阳极：Ti4O7，BDD，SnO2，PbO2。常见的活性阳极：RuO2，IrO2，Pt
12. 若明确提及降解实验是在室温下进行的，则认为温度是25°C，如果没有明确提及降解实验的温度，则温度值为空
13. 如果BDD或Ti4O7电极的基底材料没有被明确提及，则认为基底为null
14. 阴极材料如果是不锈钢（不论哪种型号），都返回stainless steel
15. 阴极材料如果是Ti，Ti mesh，Ti板或其他形态，都返回Ti。Ni也同理，无论什么形态都返回Ni。zirconium无论什么形态都返回Zr。
16. 若是混合电解质溶液，则用/隔开，例如Na2SO4/NaCl
17. 若污染物是真实废水或COD或氨氮，则该组数据不纳入
18. 若未直接提供电流密度，但提供了电流和阳极面积，则电流密度=电流/阳极面积


【重要提示】
以上是系统设定的固定规则。接下来用户会提供具体的文献内容，
请严格遵循上述规则，从用户提供的文献中提取数据。
请只输出JSON格式的数据，不要包含任何解释、说明或其他文本。输出必须是有效的JSON格式。
"""


def preprocess_text(text):
    """预处理实验文本（统一单位符号）"""
    text = re.sub(r"mA/cm[²2]", "mA/cm2", text)
    text = re.sub(r"mmol/L?", "mmol/L", text)
    text = re.sub(r"mg/L", "mg/L", text)
    text = re.sub(r"(\d)\s*M", r"\1 mol/L", text)
    return text


def validate_output(data):
    """验证提取结果的有效性"""
    required_fields = ['pollutant_name', 'current_density', 'anode']
    if isinstance(data, list):
        return all(validate_output(item) for item in data)
    return all(field in data for field in required_fields)


def parse_json_response(response_content):
    try:
        matches = re.findall(r'\{[^{}]*\}', response_content, re.S)
        result = [json.loads(x) for x in matches]

        if result:
            return result, None

    except Exception as e:
        logging.warning(f"JSON解析失败: {e}")

    return None, response_content


def extract_electrochemical_data(llm, text, temperature):
    """
    从电化学文献提取结构化数据
    返回: (提取结果, 处理时间秒, 原始响应内容, 是否解析成功)
    """
    text = preprocess_text(text)
    processed_text = requirement + text

    # 记录开始时间
    start_time = time.time()
    dmx_api_key = os.getenv('DMX_API_KEY')

    try:
        client = OpenAI(
            api_key=dmx_api_key, base_url="https://www.dmxapi.cn/v1", timeout=420, max_retries=0)
        response = client.chat.completions.create(
            model=llm,
            temperature=temperature,
            messages=[
                {"role": "system", "content": '你是一名电化学水处理领域的专家'},
                {"role": "user", "content": processed_text},
            ],
        )
        raw_content = response.choices[0].message.content

        # 解析JSON响应
        result, raw_response = parse_json_response(raw_content)

        # 计算处理时间
        processing_time = time.time() - start_time

        if result is None:
            # JSON解析失败，返回原始内容
            return None, processing_time, raw_response, False
        else:
            # JSON解析成功，验证数据
            if validate_output(result):
                return result, processing_time, raw_content, True
            else:
                logging.warning("提取的数据不完整")
                return None, processing_time, raw_content, False

    except Exception as e:
        error_msg = f"API调用失败: {str(e)}\n{traceback.format_exc()}"
        logging.error(error_msg)
        processing_time = time.time() - start_time
        return None, processing_time, error_msg, False


def process_single_file(file_info, llm, output_folder, max_retries=3):
    """
    处理单个文件并保存为单独的xlsx
    """
    number, file_path = file_info

    for attempt in range(max_retries):
        try:
            logging.info(f"开始处理文献 {number} (尝试 {attempt + 1}/{max_retries})")

            # 读取文件
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            if not text.strip():
                logging.warning(f"文件为空: {number}")
                return number, "失败", "文件为空", 0, 0.0, None, False

            # 调用API提取数据
            result, processing_time, raw_content, parse_success = extract_electrochemical_data(llm, text, temperature=0)

            # 格式化处理时间，保留两位小数
            formatted_time = round(processing_time, 2)

            if result and parse_success:
                # 处理结果
                if isinstance(result, dict):
                    df = pd.DataFrame([result])
                    data_count = 1
                elif isinstance(result, list):
                    df = pd.DataFrame(result)
                    data_count = len(result)
                else:
                    logging.error(f"无效格式: {number}")
                    return number, "失败", "无效格式", 0, formatted_time, raw_content, parse_success

                # 添加源文件信息和处理时间
                df.insert(0, 'source', number)
                df.insert(1, 'processing time (s)', formatted_time)

                # 为每个文件单独保存xlsx
                output_path = os.path.join(output_folder, f"{number}.xlsx")
                df.to_excel(output_path, index=False)

                # 保存原始响应（可选）
                raw_output_path = os.path.join(output_folder, f"{number}.txt")
                with open(raw_output_path, 'w', encoding='utf-8') as f:
                    f.write(raw_content)

                logging.info(f"成功处理: {number}, 数据条数: {data_count}, 处理时间: {formatted_time}s")
                return number, "成功", f"提取{data_count}条数据", data_count, formatted_time, raw_content, parse_success
            else:
                if not parse_success and raw_content:
                    # JSON解析失败，保存原始内容
                    raw_output_path = os.path.join(output_folder, f"{number}.txt")
                    with open(raw_output_path, 'w', encoding='utf-8') as f:
                        f.write(raw_content)
                    logging.warning(f"JSON解析失败，已保存原始响应: {number}")

                if attempt < max_retries - 1:
                    logging.warning(f"第{attempt + 1}次尝试失败，等待重试: {number}")
                    time.sleep(3)  # 等待3秒后重试
                    continue
                else:
                    logging.error(f"提取失败: {number}, 处理时间: {formatted_time}s")
                    return number, "失败", "提取失败", 0, formatted_time, raw_content, parse_success

        except Exception as e:
            logging.error(f"处理文件失败 {number}: {str(e)}\n{traceback.format_exc()}")
            if attempt < max_retries - 1:
                time.sleep(3)
                continue
            else:
                return number, "失败", f"错误: {str(e)}", 0, 0.0, None, False

    return number, "失败", "最大重试次数用完", 0, 0.0, None, False


def parallel_process_files(md_folder, output_folder, model, max_workers=100,
                          sample_size=None, random_seed=42):
    """
    并行处理所有文件，每个文件单独保存xlsx
    支持随机抽样指定数量的文件

    Parameters:
    -----------
    md_folder : str
        输入文件夹路径
    output_folder : str
        输出文件夹路径
    model : str
        使用的模型
    max_workers : int
        最大并行线程数
    sample_size : int or None
        随机抽取的文件数量，None表示处理所有文件
    random_seed : int
        随机种子，保证可复现
    """
    # 获取所有md文件
    md_files = []
    for file in os.listdir(md_folder):
        if file.endswith('.md'):
            file_path = os.path.join(md_folder, file)
            number = file.replace('.md', '')
            md_files.append((number, file_path))
        elif file.endswith('.txt'):
            file_path = os.path.join(md_folder, file)
            number = file.replace('.txt', '')
            md_files.append((number, file_path))

    if not md_files:
        logging.warning("未找到md/txt文件")
        return

    # 设置随机种子
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
        logging.info(f"设置随机种子: {random_seed}")

    # 随机抽样
    if sample_size is not None and sample_size < len(md_files):
        md_files = random.sample(md_files, sample_size)
        logging.info(f"随机抽取 {sample_size} 个文件（种子={random_seed}）")

    # 对抽样结果排序，便于查看
    md_files.sort(key=lambda x: x[0])

    # 打印抽样结果
    sampled_files = [f[0] for f in md_files]
    logging.info(f"将处理以下 {len(md_files)} 个文件: {sampled_files}")

    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 保存抽样记录
    with open(os.path.join(output_folder, "sampling_info.txt"), 'w', encoding='utf-8') as f:
        f.write(f"随机种子: {random_seed}\n")
        f.write(f"样本数量: {len(md_files)} / {len(os.listdir(md_folder))}\n")
        f.write(f"抽样文件列表:\n")
        for file_id, _ in md_files:
            f.write(f"  {file_id}\n")

    success_count = 0
    fail_count = 0
    parse_fail_count = 0
    total_data_count = 0
    total_processing_time = 0.0

    # 用于保存处理统计
    processing_results = []

    # 使用线程池并行处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_file = {
            executor.submit(process_single_file, file_info, model, output_folder): file_info[0]
            for file_info in md_files
        }

        # 使用进度条
        with tqdm(total=len(md_files), desc="处理文件") as pbar:
            for future in as_completed(future_to_file):
                file_number = future_to_file[future]
                try:
                    number, status, message, data_count, processing_time, raw_content, parse_success = future.result()

                    processing_results.append({
                        'file': number,
                        'status': status,
                        'message': message,
                        'data_count': data_count,
                        'processing_time': processing_time,
                        'parse_success': parse_success
                    })

                    if status == "成功":
                        success_count += 1
                        total_data_count += data_count
                        total_processing_time += processing_time
                        pbar.set_postfix_str(f"成功: {number} ({data_count}条, {processing_time}s)")
                    else:
                        fail_count += 1
                        if parse_success is False:
                            parse_fail_count += 1
                        pbar.set_postfix_str(f"失败: {number}")

                    pbar.update(1)

                except Exception as e:
                    logging.error(f"处理 {file_number} 时发生异常: {e}\n{traceback.format_exc()}")
                    processing_results.append({
                        'file': file_number,
                        'status': '失败',
                        'message': f'异常: {str(e)}',
                        'data_count': 0,
                        'processing_time': 0.0,
                        'parse_success': False
                    })
                    fail_count += 1
                    parse_fail_count += 1
                    pbar.update(1)

    # 保存处理统计信息
    save_processing_statistics(processing_results, output_folder, success_count, fail_count,
                               total_data_count, total_processing_time, parse_fail_count)

    logging.info(f"处理完成！成功: {success_count}, 失败: {fail_count}, JSON解析失败: {parse_fail_count}, "
                 f"总数据条数: {total_data_count}, 总处理时间: {total_processing_time:.2f}s")
    return processing_results


def save_processing_statistics(results, output_folder, success_count, fail_count,
                               total_data_count, total_processing_time, parse_fail_count):
    """保存处理统计信息"""
    stats_df = pd.DataFrame(results)
    stats_file = os.path.join(output_folder, "处理统计报告.csv")
    stats_df.to_csv(stats_file, index=False)

    # 生成统计报告文本文件
    report_file = os.path.join(output_folder, "处理报告.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("电化学数据提取处理报告\n")
        f.write("=" * 50 + "\n")
        f.write(f"处理时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总文件数: {len(results)}\n")
        f.write(f"成功处理: {success_count} 个文件\n")
        f.write(f"处理失败: {fail_count} 个文件\n")
        f.write(f"JSON解析失败: {parse_fail_count} 个文件\n")
        f.write(f"成功率: {success_count / len(results) * 100:.1f}%\n")
        f.write(f"总数据条数: {total_data_count}\n")
        f.write(f"总处理时间: {total_processing_time:.2f} 秒\n")
        f.write(f"平均处理时间: {total_processing_time / len(results):.2f} 秒/文件\n\n")

        if fail_count > 0:
            f.write("失败文件列表:\n")
            for result in results:
                if result['status'] == '失败':
                    parse_status = "JSON解析失败" if result.get('parse_success') is False else "其他失败"
                    f.write(f"  {result['file']}: {result['message']} ({parse_status})\n")

        # 列出所有JSON解析失败的文件
        parse_fail_files = [r['file'] for r in results if r.get('parse_success') is False]
        if parse_fail_files:
            f.write(f"\nJSON解析失败的文件 ({len(parse_fail_files)}个):\n")
            for file in parse_fail_files:
                f.write(f"  {file}\n")
            f.write(f"\n注意：JSON解析失败的文件的原始响应已保存为 [文件名].txt\n")


# 使用示例
if __name__ == "__main__":
    # 模型列表

    models = ['DeepSeek-V3.2-Thinking', 'grok-4.1']
    models = ['gpt-5.4', 'qwen3.5-plus', 'kimi-k2.5']
    models = ['gpt-5.4', 'deepseek-v4-pro', 'qwen3.6-plus', 'grok-4.2-thinking', 'kimi-k2.5']
    models = ['qwen3.6-plus']
    MD_FOLDER = "E:\\dm\\md-50-raw"    # 输入文件夹

    # 设置抽样参数
    SAMPLE_SIZE = None  # 随机抽取n篇，设为None则处理所有文件
    RANDOM_SEED = 42  # 固定随机种子
    MAX_WORKERS = 50  # 最大并行线程数
    load_dotenv()

    for llm in models:
        # 构建输出文件夹路径
        if SAMPLE_SIZE is None:
            OUTPUT_FOLDER = f"E:\\dm\\model-tag-raw-50-3\\model-raw\\md_{llm}"
        else:
            OUTPUT_FOLDER = f"E:\\dm\\model-tag-50\\model-raw\\md_{llm}_sample{SAMPLE_SIZE}"
        print(f"\n" + "=" * 50)
        logging.info(f"开始处理模型: {llm}")

        logging.info(f"输入文件夹: {MD_FOLDER}")
        logging.info(f"输出文件夹: {OUTPUT_FOLDER}")
        if SAMPLE_SIZE is not None:
            logging.info(f"抽样设置: {SAMPLE_SIZE}篇 (种子={RANDOM_SEED})")

        # 开始并行处理
        start_time = time.time()

        results = parallel_process_files(
            md_folder=MD_FOLDER,
            output_folder=OUTPUT_FOLDER,
            model=llm,
            max_workers=MAX_WORKERS,
            sample_size=SAMPLE_SIZE,  # 添加抽样数量参数
            random_seed=RANDOM_SEED    # 添加随机种子参数
        )

        elapsed_time = time.time() - start_time

        if results:
            success_count = sum(1 for r in results if r['status'] == '成功')
            total_processing_time = sum(r.get('processing_time', 0) for r in results)
            parse_fail_count = sum(1 for r in results if r.get('parse_success') is False)
        else:
            success_count = 0
            total_processing_time = 0
            parse_fail_count = 0

        print(f"\n" + "=" * 50)
        print(f"模型: {llm}")
        if SAMPLE_SIZE is not None:
            print(f"抽样设置: {SAMPLE_SIZE}篇 (种子={RANDOM_SEED})")
        print(f"总耗时: {elapsed_time:.2f}秒")
        print(f"成功处理: {success_count} 个文件")
        print(f"JSON解析失败: {parse_fail_count} 个文件")
        print(f"平均处理时间: {elapsed_time/len(results) if results else 0:.2f} 秒/文件")
