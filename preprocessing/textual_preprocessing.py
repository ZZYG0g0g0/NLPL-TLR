"""
Author: 邹致远
Email: www.pisyongheng@foxmail.com
Date Created: 2024/1/24
Last Updated: 2024/1/24
Version: 1.0.0
"""

import os
import re

import stanfordcorenlp
from nltk import word_tokenize
from nltk.corpus import stopwords
import string
import json
import nltk
from nltk.stem import PorterStemmer

nlp = stanfordcorenlp.StanfordCoreNLP(r'../CoreNLP/stanford-corenlp-4.5.4')
# nltk.download('stopwords')

#1.去除驼峰命名
def split_camel_case(s):
    # 将字符串s切分为单词列表
    words = s.split()
    # 对于列表中的每一个单词，如果它是驼峰命名的，则拆分它
    for index, word in enumerate(words):
        splitted = ' '.join(re.findall(r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))', word))
        if splitted:  # 如果成功拆分，则替换原单词
            words[index] = splitted

    # 将处理后的单词列表连接起来，并返回
    return ' '.join(words)

#2.分词
def tokenize_text(text):
    return word_tokenize(text)

#3.将词转为小写
def words_to_lowercase(words):
    return [word.lower() for word in words]

#4.去除停用词，标点和数字
def filter_words(word_list):
    # nltk.download('stopword')

    stop_words = set(stopwords.words('english'))  # 停用词列表
    punctuation_symbols = set(string.punctuation)  # 标点符号列表
    words = []
    for word in word_list:
        if '.' in word:
            for each in word.split('.'):
                words.append(each)
        words.append(word)
    # 过滤停用词、标点符号和数字
    filtered_words = [word for word in words if word.lower() not in stop_words and
                      not any(char in punctuation_symbols for char in word) and
                      word.isalpha()]
    return filtered_words

#5.词形还原
def extract_restore(text):
    # Initialize NLTK's PorterStemmer
    stemmer = PorterStemmer()

    def split_text(text, max_length):
        return [text[i:i + max_length] for i in range(0, len(text), max_length)]

    chunks = split_text(text, 50000)
    all_stems = []
    for chunk in chunks:
        doc = nlp.annotate(chunk, properties={
            'annotators': 'lemma',
            'pipelineLanguage': 'en',
            'outputFormat': 'json'
        })
        doc = json.loads(doc)
        lemmas = [word['lemma'] for sentence in doc['sentences'] for word in sentence['tokens']]
        stems = [stemmer.stem(token) for token in lemmas]
        all_stems.extend(stems)

    return ' '.join(all_stems)

#预处理
def preprocessing(dataset_name, type_a, type_b):
    file_names_source = os.listdir(f'../datasets/{dataset_name}/{type_a}')
    file_name_target = os.listdir(f'../datasets/{dataset_name}/{type_b}')
    source_dir = '../datasets/' + dataset_name
    target_dir = '../datasets/' + dataset_name
    if not os.path.exists(source_dir):
        os.makedirs(source_dir)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    open(f'{source_dir}/{type_a}_doc.txt', 'w').close()
    open(f'{target_dir}/{type_b}_doc.txt', 'w').close()
    for file_name in file_names_source:
        with open(f'../datasets/{dataset_name}/{type_a}/{file_name}','r',encoding='ISO8859-1')as sf:
            text = ""
            lines = sf.readlines()
            for line in lines:
                text += line.strip()
            res = split_camel_case(text)
            res = tokenize_text(res)
            res = words_to_lowercase(res)
            res = filter_words(res)
            res = extract_restore(' '.join(res))
        with open(f'../datasets/{dataset_name}/{type_a}_doc.txt','a',encoding='ISO8859-1')as swf:
            swf.write(res)
            swf.write('\n')
    for file_name in file_name_target:
        with open(f'../datasets/{dataset_name}/{type_b}/{file_name}','r',encoding='ISO8859-1')as tf:
            text = ""
            lines = tf.readlines()
            for line in lines:
                text += line.strip()
            res = split_camel_case(text)
            res = tokenize_text(res)
            res = words_to_lowercase(res)
            res = filter_words(res)
            res = extract_restore(' '.join(res))
        with open(f'../datasets/{dataset_name}/{type_b}_doc.txt','a',encoding='ISO8859-1')as twf:
            twf.write(res)
            twf.write('\n')

import os

def preprocessing_single_file(dataset_name, data_type):
    input_file = f'../RQ4/strategy/FineGrained/{dataset_name}/{data_type}.txt'
    output_file = f'../RQ4/strategy/FineGrained/{dataset_name}/{data_type}_doc.txt'


    with open(input_file, 'r', encoding='ISO8859-1') as infile, \
         open(output_file, 'w', encoding='ISO8859-1') as outfile:
        for line in infile:
            text = line.strip()
            if not text:
                outfile.write('\n')
                continue
            res = split_camel_case(text)
            res = tokenize_text(res)
            res = words_to_lowercase(res)
            res = filter_words(res)
            res = extract_restore(' '.join(res))
            outfile.write(res + '\n')

import pandas as pd

def preprocess_csv_columns(file_path, column_a, column_b):
    """
    对指定的 CSV 文件的指定列进行预处理，并将结果写入新文件。

    参数:
        file_path (str): CSV 文件的路径
        column_a (str): 需要处理的第一列列名
        column_b (str): 需要处理的第二列列名
    """
    def process_text(text):
        # print(text)
        """
        对单个文本进行预处理。
        """
        # 假设这些函数已经定义
        text = split_camel_case(text)  # 拆分驼峰命名
        text = tokenize_text(text)     # 分词
        text = words_to_lowercase(text) # 转为小写
        text = filter_words(text)      # 过滤单词
        text = extract_restore(' '.join(text)) # 恢复提取
        return text

    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"文件 {file_path} 不存在，请检查路径。")
        return

    # 读取 CSV 文件
    df = pd.read_csv(file_path)

    # 检查列是否存在
    if column_a not in df.columns or column_b not in df.columns:
        print(f"文件 {file_path} 中缺少指定列 {column_a} 或 {column_b}，跳过处理。")
        return

    # 对指定列进行预处理
    df[column_a] = df[column_a].astype(str).apply(process_text)
    df[column_b] = df[column_b].astype(str).apply(process_text)

    # 生成新文件名
    file_dir, file_name = os.path.split(file_path)
    new_file_name = f"{os.path.splitext(file_name)[0]}_preprocessing.csv"
    new_file_path = os.path.join(file_dir, new_file_name)

    # 保存处理后的数据到新文件
    df.to_csv(new_file_path, index=False)
    print(f"处理后的文件已保存为: {new_file_path}")

if __name__ == '__main__':
    # datasets = ['Albergate']
    datasets = ['Derby', 'Dronology', 'Drools', 'eAnci', 'Groovy', 'Infinispan', 'iTrust', 'maven', 'Pig', 'Seam2', 'smos']
    # data_types = ['class_attribute', 'class_comment', 'class_name', 'method_comment', 'method_parameter', 'method_return']
    data_types = ['method_name']
    for dataset in datasets:
        for data_type in data_types:
            preprocessing_single_file(dataset, data_type)
            # preprocessing('CCHIT', 'Requirements', 'code')
            # folder_path = '../datasets/zookeeper/zookeeper_code_change_issue.csv'  # 替换为你的文件夹路径
            # folder_path = '../datasets/jfreechart/jfreechart_test_method.csv'
            # column_a = 'test-fqn'  # 替换为需要/处理的第一列列名
            # column_b = 'tested-method-fqn'  # 替换为需要处理的第二列列名
            # column_a = 'Commit_Text'
            # column_b = 'Issue_Text'
            # preprocess_csv_columns(folder_path, column_a, column_b)
    nlp.close()  # 确保服务被关闭