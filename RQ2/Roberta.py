import os
import torch
from transformers import RobertaTokenizer, RobertaModel
import pandas as pd
from tqdm import tqdm

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def preprocess_texts(texts, tokenizer, max_length):
    input_ids = []
    attention_masks = []

    for text in texts:
        encoded_dict = tokenizer.encode_plus(
            text,                      # 输入文本
            add_special_tokens=True,   # 添加特殊标记
            max_length=max_length,     # 截断的最大长度
            truncation=True,           # 显式截断
            pad_to_max_length=True,    # 填充到最大长度
            return_attention_mask=True,# 返回注意力掩码
            return_tensors='pt',       # 返回 PyTorch 张量
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    return torch.cat(input_ids, dim=0), torch.cat(attention_masks, dim=0)

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def process_column(texts, tokenizer, model, device, max_length, batch_size, desc=""):
    input_ids, attention_masks = preprocess_texts(texts, tokenizer, max_length)
    input_ids = input_ids.to(device)
    attention_masks = attention_masks.to(device)

    vectors = []
    with torch.no_grad():
        for i in tqdm(range(0, len(input_ids), batch_size), desc=desc):
            batch_input_ids = input_ids[i:i + batch_size]
            batch_attention_masks = attention_masks[i:i + batch_size]
            outputs = model(batch_input_ids, attention_mask=batch_attention_masks)
            batch_vectors = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            vectors.extend(batch_vectors)
    return vectors


if __name__ == '__main__':
    set_seed(42)  # 设置随机种子以确保可重复性

    # 检查GPU是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 定义要处理的数据集列表
    # datasets = ['pgcli']
    datasets = ['commons-io']
    max_length = 512  # RoBERTa的最大输入长度
    batch_size = 4  # 减小批次大小以减少内存使用

    # 初始化 RoBERTa 模型和 tokenizer
    model_name = 'roberta-base'
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    for dataset in datasets:
        try:
            print(f"\nProcessing dataset: {dataset}")

            # 构建输入文件路径
            input_file = f'../datasets/{dataset}/{dataset}_preprocessing.csv'  # 假设每个数据集目录下有一个data.xlsx文件

            # 读取Excel文件
            df = pd.read_csv(input_file)

            df['test-fqn'] = df['test-fqn'].fillna('')
            df['tested-method-fqn'] = df['tested-method-fqn'].fillna('')

            # 处理test-fqn列
            if 'test-fqn' in df.columns:
                issue_vectors = process_column(
                    df['test-fqn'].tolist(),
                    tokenizer, model, device,
                    max_length, batch_size,
                    desc=f"Processing {dataset} test-fqn"
                )

                # 保存Issue向量
                issue_df = pd.DataFrame(issue_vectors)
                issue_output = f'../datasets/{dataset}/preprocessing_test_roberta.xlsx'
                ensure_dir(issue_output)
                issue_df.to_excel(issue_output, index=False)
                print(f"Issue vectors saved to {issue_output}")

            # 处理tested-method-fqn列
            if 'tested-method-fqn' in df.columns:
                commit_vectors = process_column(
                    df['tested-method-fqn'].tolist(),
                    tokenizer, model, device,
                    max_length, batch_size,
                    desc=f"Processing {dataset} tested-method-fqn"
                )

                # 保存Commit向量
                commit_df = pd.DataFrame(commit_vectors)
                commit_output = f'../datasets/{dataset}/preprocessing_tested_code_roberta.xlsx'
                ensure_dir(commit_output)
                commit_df.to_excel(commit_output, index=False)
                print(f"Commit vectors saved to {commit_output}")

        except Exception as e:
            print(f"Error processing dataset {dataset}: {e}")