import pandas as pd
import numpy as np
from scipy.stats import norm
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def analysis1(csv_path):
    # 假设 CSV 文件的路径为 'your_data.csv'
    # 并且 CSV 文件的格式是每一行代表一个网络层，每一列代表一个专家，值为出现的次数

    # 从 CSV 文件读取数据到 DataFrame
    df = pd.read_csv(csv_path, index_col=0)

    # 创建一个列表用于存储结果
    most_least_frequent = []

    part_count = 5
    # 遍历每一层
    for idx, row in df.iterrows():
        # 排除用于统计的标题行
        if idx == 0:
            continue
        # print(f'idx = {idx}, row = {row}')
        # 找到出现次数最多的5个专家
        most_experts = row.nlargest(part_count)
        
        # 找到出现次数最少的5个专家
        least_experts = row.nsmallest(part_count)
        
        most_least_frequent.append({
            'Layer': idx,
            'Most Frequent Experts': most_experts.index.tolist(),
            'Most Frequent Counts': most_experts.values.tolist(),
            'Least Frequent Experts': least_experts.index.tolist(),
            'Least Frequent Counts': least_experts.values.tolist()
        })

    # 转换为 DataFrame 以便查看
    result_df = pd.DataFrame(most_least_frequent)

    # 打印结果
    for index, row in result_df.iterrows():
        print(f"Layer {row['Layer']}:")
        print(f"  Most Frequent Experts: {row['Most Frequent Experts']} with counts {row['Most Frequent Counts']}")
        print(f"  Least Frequent Experts: {row['Least Frequent Experts']} with counts {row['Least Frequent Counts']}")
        print()


def analysis2(csv_path):
    # 假设 CSV 文件的路径为 'your_data.csv'
    # CSV 文件的第一列是行标题，第一行是列标题

    # 从 CSV 文件读取数据，并指定第一列作为索引
    df = pd.read_csv(csv_path, index_col=0)

    # 创建一个用于存储每层负载信息的列表
    layer_load_info = []

    # 遍历每一层
    for layer_name, row in df.iterrows():
        # 排除用于统计的标题行
        if layer_name == 'prefill阶段':
            continue
        
        # 计算当前层中每个专家的负载
        layer_loads = row.drop(labels='prefill阶段', errors='ignore')
        
        # 找出负载最高和最低的专家
        most_loaded_expert = layer_loads.idxmax()
        least_loaded_expert = layer_loads.idxmin()
        
        # 存储每层负载信息
        layer_load_info.append({
            'Layer': layer_name,
            'Most Loaded Expert': most_loaded_expert,
            'Most Loaded Count': layer_loads[most_loaded_expert],
            'Least Loaded Expert': least_loaded_expert,
            'Least Loaded Count': layer_loads[least_loaded_expert]
        })

    # 打印每层的负载分布结果
    for info in layer_load_info:
        print(f"Layer {info['Layer']}:")
        print(f"  Most Loaded Expert: {info['Most Loaded Expert']} with a load of {info['Most Loaded Count']}")
        print(f"  Least Loaded Expert: {info['Least Loaded Expert']} with a load of {info['Least Loaded Count']}")
        print()

    # 计算每个专家在所有层中的总负载（出现次数总和）
    expert_loads = df.sum(axis=0)

    # 排除用于统计的标题列
    expert_loads = expert_loads.drop(labels='prefill阶段', errors='ignore')

    # 计算每个专家的平均负载
    expert_avg_loads = df.mean(axis=0)

    # 排除用于统计的标题列
    expert_avg_loads = expert_avg_loads.drop(labels='prefill阶段', errors='ignore')

    # 打印整体负载分布结果
    print("Total Load Distribution for Each Expert:")
    print(expert_loads)

    print("\nAverage Load Distribution for Each Expert:")
    print(expert_avg_loads)

    # 分析整体结果
    most_loaded_expert = expert_loads.idxmax()
    least_loaded_expert = expert_loads.idxmin()

    print(f"\nExpert with the highest total load: {most_loaded_expert} with a load of {expert_loads[most_loaded_expert]}")
    print(f"Expert with the lowest total load: {least_loaded_expert} with a load of {expert_loads[least_loaded_expert]}")

def analysis3_sort(csv_path):
    # 读取数据
    df = pd.read_csv(csv_path, index_col=0)

    # 存储拟合参数
    fit_params = []

    # 拟合每层的专家负载
    for idx, row in df.iterrows():
        # 排除用于统计的标题行
        if idx == 0:
            continue

        # 获取当前层的负载数据
        layer_loads = row.values

        # 拟合正态分布
        mu, std = norm.fit(layer_loads)
        
        # 存储拟合参数
        fit_params.append((mu, std))

    # 转换为 numpy 数组
    fit_params = np.array(fit_params)

    # 标准化特征
    scaler = StandardScaler()
    fit_params_scaled = scaler.fit_transform(fit_params)

    # 使用 DBSCAN 聚类
    dbscan = DBSCAN(eps=0.5, min_samples=5)  # eps 和 min_samples 需要根据数据调整
    labels = dbscan.fit_predict(fit_params_scaled)

    # 输出每层的聚类结果
    layer_names = df.index
    clusters = {}
    for layer_name, label in zip(layer_names, labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(layer_name)

    # 打印聚类结果
    for cluster_id, layers in clusters.items():
        print(f"Cluster {cluster_id}: {layers}")

    # 可视化聚类结果
    plt.scatter(fit_params_scaled[:, 0], fit_params_scaled[:, 1], c=labels, cmap='viridis')
    plt.xlabel('Mean (mu)')
    plt.ylabel('Standard Deviation (std)')
    plt.title('DBSCAN Clusters of Layer Expert Load Distributions')
    # plt.show()
    # 保存图像到文件
    plt.savefig('cluster_visualization.png', format='png', dpi=300)


def analysis_data_std(csv_path):
    # 读取数据
    df = pd.read_csv(csv_path, index_col=0)

    # 初始化一个列表来存储统计结果
    stats_list = []

    # 计算每层网络的统计特征
    for layer_name, row in df.iterrows():
        # 获取当前层的专家负载数据
        expert_loads = row.values
        
        # 计算统计特征
        max_load = expert_loads.max()
        min_load = expert_loads.min()
        # mean_load = expert_loads.mean()
        variance_load = expert_loads.var()
        # stddev_load = expert_loads.std()

        # 计算分位数
        expert_loads_asc = pd.Series(expert_loads).sort_values(ascending=True)
        quantile_10 = expert_loads_asc.quantile(0.10)
        quantile_20 = expert_loads_asc.quantile(0.20)
        quantile_30 = expert_loads_asc.quantile(0.30)

        
        # 添加结果到 stats_list
        stats_list.append({
            'Layer': layer_name,
            'Max': max_load,
            'Min': min_load,
            # 'Mean': mean_load,
            'Variance': variance_load,
            # 'StdDev': stddev_load
            '10% Quantile': quantile_10,
            '20% Quantile': quantile_20,
            '30% Quantile': quantile_30,
        })

    # 将结果列表转换为 DataFrame
    stats_df = pd.DataFrame(stats_list)

    # 打印统计结果
    print(stats_df)

if __name__ == "__main__":
    # csv_path = "/root/paddlejob/workspace/env_run/output/chenkailun/deepseek_serving/script/moe/data/expert_count_19_prefill.csv"
    csv_path = "/root/paddlejob/workspace/env_run/output/chenkailun/deepseek_serving/script/moe/data/expert_count_19_decoder.csv"
    analysis_data_std(csv_path)