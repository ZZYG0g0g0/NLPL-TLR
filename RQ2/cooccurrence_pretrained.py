import os
import random
from scipy import stats


def calculate_cootesturrence(source, target):
    """ 根据新规则计算共现比例 """
    source_list = source.split()
    target_list = target.split()
    source_set = set(source_list)
    target_set = set(target_list)

    # 计算source→target方向的共现率
    source_score = 0
    if len(source_list) > 0:
        matched = sum(1 for word in source_list if word in target_set)
        source_score = matched / len(source_list)

    # 计算target→source方向的共现率
    target_score = 0
    if len(target_list) > 0:
        matched = sum(1 for word in target_list if word in source_set)
        target_score = matched / len(target_list)

    # 返回双向平均值
    return (source_score + target_score) / 2


if __name__ == '__main__':
    # 配置数据集名称
    dataset_name = "smos"  # 替换为实际数据集名称

    # 路径配置
    type_a = 'uc' #修改
    type_b = 'cc' #修改0
    base_dir = f"./datasets/{dataset_name}"
    source_dir = os.path.join(base_dir, type_a)
    target_dir = os.path.join(base_dir, type_b)
    true_set_path = os.path.join(base_dir, "true_set.txt")#修改

    # 读取真集对
    true_pairs = set()
    with open(true_set_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                source_file, target_file = parts[0], parts[1]
                true_pairs.add((source_file, target_file))

    sources = []
    targets = []
    # 读取uc
    with open(os.path.join(base_dir, f'{type_a}_doc.txt'), 'r', encoding='ISO8859-1') as f:
        sources = f.readlines()
    #读取test
    with open(os.path.join(base_dir, f'{type_b}_doc.txt'), 'r', encoding='ISO8859-1') as f:
        targets = f.readlines()

    # 获取所有文件列表，使用集合模式存储，键为文件名，值为下标索引
    source_files = {f: idx for idx, f in enumerate(os.listdir(source_dir)) if os.path.isfile(os.path.join(source_dir, f))}
    target_files = {f: idx for idx, f in enumerate(os.listdir(target_dir)) if os.path.isfile(os.path.join(target_dir, f))}

    # 计算真集对的共现比例
    true_scores = []
    for source, target in true_pairs:
        score = calculate_cootesturrence(
            sources[source_files[source]],
           targets[target_files[target]]
        )
        if score is not None:
            true_scores.append(score)
        else:
            print(f"Warning: Missing true pair ({source}, {target})")
    # 计算真集对的共现比例平均值
    true_avg_score = sum(true_scores) / len(true_scores) if true_scores else 0

    # 随机采样非真集对（100次）
    non_true_scores = []
    for _ in range(100):
        # 随机抽取与真集对数量相同的假对（非真集对）
        non_true_avg_scores = []
        sampled_pairs = set()

        while len(sampled_pairs) < len(true_pairs):
            uc = random.choice(list(source_files.keys()))
            test = random.choice(list(target_files.keys()))
            if (uc, test) not in true_pairs:  # 确保是非真集对
                sampled_pairs.add((uc, test))
        for uc, test in sampled_pairs:
            score = calculate_cootesturrence(
                sources[source_files[uc]],
                targets[target_files[test]]
            )
            if score is not None:
                non_true_avg_scores.append(score)
            else:
                print(f"Warning: Missing sampled pair ({uc}, {test})")

        if non_true_avg_scores:
            non_true_scores.append(sum(non_true_avg_scores) / len(non_true_avg_scores))

    # 执行假设检验（使用Mann-Whitney U检验）
    u_stat, p_value = stats.mannwhitneyu(true_avg_score, non_true_scores, alternative='greater')

    # 输出结果
    print(f"真集平均共现比例: {true_avg_score:.4f}")
    print(f"非真集平均共现比例: {sum(non_true_scores) / len(non_true_scores):.4f}")
    print(f"Mann-Whitney U检验p值: {p_value:.6f}")

    if p_value < 0.01:
        print("结论：真集的共现比例显著高于非真集（p < 0.01）")
    else:
        print("结论：没有显著差异（p >= 0.01）")
