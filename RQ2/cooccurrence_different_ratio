import random
import pandas as pd
from scipy import stats
def calculate_cooccurrence(issue_text, commit_text):
    """ 根据新规则计算共现比例 """
    # 确保输入文本是字符串类型
    issue_text = str(issue_text)
    commit_text = str(commit_text)

    issue_list = issue_text.split()
    commit_list = commit_text.split()
    issue_set = set(issue_list)
    commit_set = set(commit_list)

    # 计算 Issue→Commit 方向的共现率
    issue_score = 0
    if len(issue_list) > 0:
        matched = sum(1 for word in issue_list if word in commit_set)
        issue_score = matched / len(issue_list)

    # 计算 Commit→Issue 方向的共现率
    commit_score = 0
    if len(commit_list) > 0:
        matched = sum(1 for word in commit_list if word in issue_set)
        commit_score = matched / len(commit_list)

    # 返回双向平均值
    return (issue_score + commit_score) / 2

if __name__ == '__main__':
    # 配置数据集名称
    dataset_name = "jfreechart"  # 替换为实际数据集名称
    source_type = "test-fqn"
    target_type = "tested-method-fqn"

    # source_type = "Source_Issue_Text"
    # target_type = "Target_Issue_Text"

    # 路径配置
    # dataset_path = f"./datasets/{dataset_name}/{dataset_name}_issue_issue_preprocessing.csv"
    dataset_path = f"./datasets/{dataset_name}/{dataset_name}_preprocessing.csv"

    # 读取CSV文件
    df = pd.read_csv(dataset_path, encoding='ISO-8859-1')

    # 提取真集对
    true_pairs = set()

    for _, row in df.iterrows():
        if row['label'] == 1:  # 真集对，label为1
            true_pairs.add((row[f'{source_type}'], row[f'{target_type}']))

    # 获取所有文件中的Issue和Commit文本
    issues = df[f'{source_type}'].tolist()
    commits = df[f'{target_type}'].tolist()

    # 计算真集对的共现比例
    true_scores = []
    for issue_text, commit_text in true_pairs:
        score = calculate_cooccurrence(issue_text, commit_text)
        if score is not None:
            true_scores.append(score)
        else:
            print(f"Warning: Missing true pair ({issue_text}, {commit_text})")
    # 计算真集对的共现比例平均值
    true_avg_score = sum(true_scores) / len(true_scores) if true_scores else 0

    # 随机采样非真集对（100次）
    non_true_scores = []
    for _ in range(100):
        # 随机抽取与真集对数量相同的假对（非真集对）
        non_true_avg_scores = []
        sampled_pairs = set()
        while len(sampled_pairs) < len(true_pairs):
            # 随机选择一个非真集对
            idx = random.randint(0, len(issues) - 1)
            issue_text = issues[idx]
            commit_text = commits[idx]
            if (issue_text, commit_text) not in true_pairs:
                sampled_pairs.add((issue_text, commit_text))
        for issue_text, commit_text in sampled_pairs:
            score = calculate_cooccurrence(issue_text, commit_text)
            if score is not None:
                non_true_avg_scores.append(score)
            else:
                print(f"Warning: Missing sampled pair ({issue_text}, {commit_text})")
        if non_true_avg_scores:
            non_true_scores.append(sum(non_true_avg_scores) / len(non_true_avg_scores))

    # 执行假设检验（使用Mann-Whitney U检验）
    u_stat, p_value = stats.mannwhitneyu(true_avg_score, non_true_scores, alternative='greater')

    # 输出结果
    print(f"真集平均共现比例: {sum(true_scores) / len(true_scores):.4f}")
    print(f"非真集平均共现比例: {sum(non_true_scores) / len(non_true_scores):.4f}")
    print(f"Mann-Whitney U检验p值: {p_value:.6f}")

    if p_value < 0.01:
        print("结论：真集的共现比例显著高于非真集（p < 0.01）")
    else:
        print("结论：没有显著差异（p >= 0.01）")
