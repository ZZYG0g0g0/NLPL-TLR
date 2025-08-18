import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
from openai import OpenAI
import time

# Initialize DeepSeek client
client = OpenAI(
    # api_key = "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", #llama
    # api_key = "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", #Gemini
    # api_key = "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", #mistral
    # api_key = "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", #coder_large
    # api_key = "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", # codex
api_key = "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", #deepcoder
    base_url= "https://openrouter.ai/api/v1"
    # api_key="sk-82bdba6449304e02a5351e8e66995b2c",  # deepseek
    # base_url="https://api.deepseek.com"  # DeepSeek API
)

def prompt_for_relation(requirements_text, code_text):
    """Construct a prompt following DeepSeek best practices"""
    return f"""Determine if the following Requirements and Code are related. Answer only "Yes" or "No":

Requirements: {requirements_text}
Code: {code_text}

Answer:"""


def generate_response(prompt):
    """Generate response using DeepSeek R1"""
    try:
        response = client.chat.completions.create(
            model="agentica-org/deepcoder-14b-preview:free",
            # model = "deepseek-reasoner",
            messages=[{
                    "role": "system",
                    "content": "You are a judge in the field of software tracking"
                },{"role": "user", "content": prompt},],
            # max_tokens=3,  # Only need "Yes" or "No"
            temperature=1.0,
            # stop=["\n"]  # Stop at newline
        )
        # answer = response.choices[0].message
        answer = response.choices[0].message.content.strip().lower()
        print(answer)
        # time.sleep(3)
        return "yes" if "yes" in answer else "no"

    except Exception as e:
        print(f"API Error: {e}")
        return "no"  # Default to "no" on error


def is_related(issue_text, commit_text):
    """Determine if two texts are related"""
    prompt = prompt_for_relation(issue_text, commit_text)
    response = generate_response(prompt)
    return 1 if response == "yes" else 0


def calculate_metrics(csv_file):
    """Calculate precision, recall, and F1 score for the dataset"""
    df = pd.read_csv(csv_file)

    # Randomly sample 10% of the data for evaluation
    eval_df = df.sample(frac=0.1, random_state=42)

    true_labels = eval_df['label'].tolist()
    predicted_labels = []

    for _, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc="Processing evaluation rows"):
        predicted_labels.append(is_related(row['uc_text'], row['cc_text']))

    # Calculate metrics
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)

    return precision, recall, f1


def save_results_to_excel(results, output_file):
    """Save the results to an Excel file"""
    results_df = pd.DataFrame(results, columns=['dataset', 'precision', 'recall', 'f1'])
    results_df.to_excel(output_file, index=False, float_format="%.4f")


if __name__ == '__main__':

    # Example usage
    # datasets = ['iTrust']
    # datasets = ['albergate',  'eAnci', 'Dronology','Groovy', 'Infinispan', 'Derby', 'Drools', 'Seam2', 'maven', 'Pig', 'smos']
    datasets = ['Pig', 'smos']
    for dataset_name in datasets:
 # Replace with the dataset name
        csv_file = f"../datasets/{dataset_name}/{dataset_name}_uc_cc.csv"  # Replace with your CSV file
        output_file = f"evaluation_results_{dataset_name}.xlsx"  # The output Excel file path

        print(f"Evaluating {csv_file}...")

        # Calculate metrics
        precision, recall, f1 = calculate_metrics(csv_file)

        # Print the metrics
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        # Save results to Excel
        results = [(dataset_name, precision, recall, f1)]
        save_results_to_excel(results, output_file)
        print(f"Results saved to {output_file}")
