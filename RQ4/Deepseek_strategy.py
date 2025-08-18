import pandas as pd
import os
import random
from tqdm import tqdm
from openai import OpenAI
from sklearn.metrics import precision_score, recall_score, f1_score

# Initialize DeepSeek client
client = OpenAI(
    api_key="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",  # Replace with your actual API key
    base_url="https://openrouter.ai/api/v1"  # DeepSeek API base URL
)

def prompt_for_relation(requirements_text, code_text, extra_info=""):
    """Construct a prompt following DeepSeek best practices"""
    return f"""Determine if the following Requirements and Code are related. Answer only "Yes" or "No":

Requirements: {requirements_text}
Code: {code_text}

Additional Information: {extra_info}

Answer:"""


def generate_response(prompt):
    """Generate response using DeepSeek R1"""
    try:
        response = client.chat.completions.create(
            model="google/gemini-2.5-pro-preview",
            messages=[{
                "role": "system",
                "content": "You are a judge in the field of software traceability"
            },{"role": "user", "content": prompt},],
            # max_tokens=3,  # Only need "Yes" or "No"
            temperature=1.0,
            # stop=["\n"]  # Stop at newline
        )

        # Standard DeepSeek API response structure
        answer = response.choices[0].message.content.strip().lower()
        print(answer)
        return "yes" if "yes" in answer else "no"

    except Exception as e:
        print(f"API Error: {e}")
        return "no"  # Default to "no" on error


def is_related(uc_text, cc_text, extra_info=""):
    """Determine if two texts are related"""
    prompt = prompt_for_relation(uc_text, cc_text, extra_info)
    response = generate_response(prompt)
    return 1 if response == "yes" else 0


def load_code_dependency(dataset):
    """Load code dependency from Excel into a {(class1, class2): relationship} dict"""
    path = f'./strategy/CodeDependency/{dataset}.xlsx'
    if not os.path.exists(path):
        return {}

    df = pd.read_excel(path)
    code_dependency = {}

    for _, row in df.iterrows():
        class1 = str(row['Class 1']).strip()
        class2 = str(row['Class 2']).strip()
        relation = str(row['Relationship']).strip()
        code_dependency[(class1, class2)] = relation

    return code_dependency


def load_user_feedback(dataset):
    """Load user feedback from existing file or randomly sample 10% of the data"""
    user_feedback_path = f'./strategy/UserFeedback/{dataset}.xlsx'

    # 如果用户反馈文件存在，则读取文件内容
    if os.path.exists(user_feedback_path):
        print(f"User feedback file found at {user_feedback_path}. Loading feedback...")
        feedback = pd.read_excel(user_feedback_path)
    else:
        # 如果用户反馈文件不存在，则随机选取10%的数据，并保存为新的Excel文件
        print(f"User feedback file not found. Sampling 10% of the data and saving to {user_feedback_path}...")
        df = pd.read_csv(f'../datasets/{dataset}/{dataset}_uc_cc.csv')
        feedback = df.sample(frac=0.1, random_state=520)
        feedback.to_excel(user_feedback_path, index=False)

    return feedback  # 返回 DataFrame


def check_user_feedback(uc_name, cc_name, user_feedback_df):
    """Check if user feedback exists for the given uc_name and cc_name"""
    feedback = user_feedback_df[
        (user_feedback_df['uc_name'] == uc_name) & (user_feedback_df['cc_name'] == cc_name)
    ]
    if not feedback.empty:
        return feedback.iloc[0]['label']
    return None


def load_fine_grained(dataset):
    """Load fine-grained requirement mapping"""
    fine_grained_path = f'./strategy/FineGrained/{dataset}/final_sim_result.xlsx'
    if not os.path.exists(fine_grained_path):
        return {}

    df = pd.read_excel(fine_grained_path)
    fine_grained = {}

    for _, row in df.iterrows():
        requirement_name = row['requirement_name']
        class_name = row['class_name']
        fine_grained.setdefault(requirement_name, []).append(class_name)

    return fine_grained


def check_fine_grained(uc_name, cc_name, fine_grained_data):
    """Check if a fine-grained relation exists"""
    for requirement, class_names in fine_grained_data.items():
        if uc_name in requirement and cc_name in class_names:
            return True
    return False


def apply_code_dependency_strategy(cc_name, code_dependency_dict, extra_info):
    """
    Given a cc_name and full code dependency dict, return a list of relationship strings
    indicating all classes related to cc_name and how.
    """
    cc_name = cc_name.split('.')[0]

    for (class1, class2), relation in code_dependency_dict.items():
        if cc_name == class1:
            extra_info.append(f"{cc_name} and {class2} have a {relation} relationship")
        elif cc_name == class2:
            extra_info.append(f"{cc_name} and {class1} have a {relation} relationship")

    return extra_info


def calculate_metrics(csv_file, strategies, dataset_name):
    """Calculate precision, recall, and F1 score for the dataset"""
    df = pd.read_csv(csv_file)
    # Randomly sample 10% of the data for evaluation
    eval_df = df.sample(frac=0.1, random_state=42)
    results = []

    # Load strategy data
    code_dependency = load_code_dependency(dataset_name)
    user_feedback_df = load_user_feedback(dataset_name)
    fine_grained_data = load_fine_grained(dataset_name)

    for _, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc="Processing rows"):
        uc_text = row['uc_text']
        cc_text = row['cc_text']
        uc_name = row['uc_name']
        cc_name = row['cc_name']
        extra_info = []

        # Apply strategies
        if 'code dependency' in strategies:
            # Apply code dependency strategy for cc_name
            extra_info = apply_code_dependency_strategy(cc_name, code_dependency, extra_info)

        if 'user feedback' in strategies:
            user_feedback_label = check_user_feedback(uc_name, cc_name, user_feedback_df)
            if user_feedback_label is not None:
                extra_info.append(f"User feedback indicates label is {user_feedback_label}")
            else:
                extra_info.append("No user feedback information")

        if 'fine grained' in strategies:
            fine_grained_relation = check_fine_grained(uc_name, cc_name, fine_grained_data)
            if fine_grained_relation:
                extra_info.append(f"Fine-grained relationship exists between {uc_name} and {cc_name}")
            else:
                extra_info.append(f"No fine-grained relationship between {uc_name} and {cc_name}")

        # Generate prompt and get prediction
        extra_info_str = ", ".join(extra_info)
        prediction = is_related(uc_text, cc_text, extra_info_str)
        label = row['label']

        # Calculate metrics for evaluation
        results.append([uc_name, cc_name, label, prediction])

    # Calculate precision, recall, f1
    true_labels = [r[2] for r in results]
    predicted_labels = [r[3] for r in results]

    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)

    return precision, recall, f1


def save_results_to_excel(results, output_file):
    """Save the results to an Excel file"""
    results_df = pd.DataFrame(results, columns=['uc_name', 'cc_name', 'label', 'predicted_label'])
    results_df.to_excel(output_file, index=False, float_format="%.4f")


if __name__ == '__main__':
    # Example usage
    datasets = ['Derby']
    # datasets = ['eAnci', 'Dronology', 'iTrust','Groovy', 'Infinispan', 'Derby', 'Drools', 'Seam2', 'maven', 'Pig']
    for dataset_name in datasets:
        csv_file = f"../datasets/{dataset_name}/{dataset_name}_uc_cc.csv"
        output_file = f"evaluation_results_{dataset_name}.xlsx"  # Output Excel file path

        # Define which strategies to apply
        # strategies = ['fine grained']
        strategies = ['code dependency', 'user feedback', 'fine grained']

        print(f"Evaluating {csv_file}...")

        # Calculate metrics
        precision, recall, f1 = calculate_metrics(csv_file, strategies, dataset_name)

        # Print the metrics
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        # Save results to Excel
        results = [(dataset_name, precision, recall, f1)]
        save_results_to_excel(results, output_file)
        print(f"Results saved to {output_file}")
