import os
import numpy as np
import re
import csv
from collections import defaultdict

def parse_and_count(file_path):

    # 字典结构为 {阶段: {layer_idx: {数字: 次数}}}
    counts = {
        'prefill': defaultdict(lambda: defaultdict(int)),
        'decoder': defaultdict(lambda: defaultdict(int))
    }
    # 改进正则表达式：精确匹配完整的数据块结构
    block_pattern = re.compile(
        r'^layer_idx = (\d+), topk_ids = Tensor\(shape=\[(\d+),\s*8\].*?\[\[(.*?)\]\][ ]*\),[ ]*topk_ids_end',
        re.DOTALL | re.MULTILINE
    )

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

        # 查找所有匹配的数据块
        for match in block_pattern.finditer(content):
            layer_idx = int(match.group(1))  # 提取层id
            m = int(match.group(2))  # 提取M值
            numbers_str = match.group(3)  # 提取数值部分字符串

            # 清理数据并提取所有数字
            numbers = []
            for num_str in re.findall(r'\d+', numbers_str.replace(' ', '')):
                numbers.append(int(num_str))
            
            # 确定阶段
            phase = 'prefill' if m > 1 else 'decoder'
            target = counts[phase][layer_idx]

            for num in numbers:
                target[num] += 1

    return counts

def print_statistics(stats, csv_file_path):

    with open(csv_file_path, mode='w', newline='', encoding='utf-8-sig') as csvfile:
        csvwriter = csv.writer(csvfile)
        for phase in ['prefill', 'decoder']:
            phase_data = stats[phase]
            if not phase_data:
                continue

            # 收集所有layer_idx和数字
            all_layers = sorted(phase_data.keys())

            # 打印阶段标题
            # print(f"\n{'='*40}")
            # print(f"{phase.upper()} PHASE STATISTICS")
            # print(f"{'='*40}")

            # 打印表头
            header = [f"{phase}阶段"] + [f"专家{num}" for num in range(256)]
            # print(",".join(header))

            csvwriter.writerow(header)  # 写入CSV表头

            # 打印每行数据
            for layer_idx in all_layers:
                row = [f'Layer{layer_idx}']
                for i in range(256):
                    row += [str(phase_data[layer_idx].get(i, 0))]
                # print(",".join(row))
                csvwriter.writerow(row)  # 写入CSV数据行

                        

if __name__ == "__main__":
    file_path = '/root/paddlejob/workspace/env_run/output/chenkailun/deepseek_serving/output_yiyan/paddle_distributed_logs/workerlog.0_part'  # 替换为实际文件路径
    csv_file_path = "/root/paddlejob/workspace/env_run/output/chenkailun/deepseek_serving/expert_count_19.csv"
    counts = parse_and_count(file_path)
    print_statistics(counts, csv_file_path)
