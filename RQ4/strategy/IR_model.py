# -*-coding:ISO8859-1-*-
# @Time : 2022/10/10 20:44
# @Author : 邓洋、李幸阜
import os

import scipy
from gensim import corpora, similarities
from gensim import models
import re
import numpy as np
import pandas as pd
import scipy.stats

from rank_bm25 import BM25Okapi

import warnings

warnings.filterwarnings(action='ignore')

# 生成查询集或被查询集(生成数据集)
def set_generation(query_file):
    """
    生成查询集或被查询集(生成数据集)
    :param query_file: 分词和去除停用词后的数据集
    :return: 返回一个列表，列表的每个元素也是一个列表，后者中的列表的每个元素都是每一条数据中的单词。
    """
    with open(query_file, "r", encoding="ISO8859-1") as ft:
        lines_T = ft.readlines()
    setline = []
    for line in lines_T:
        word = line.split(' ')
        word = [re.sub('\s', '', i) for i in word]
        word = [i for i in word if len(i) > 0]
        setline.append(word)
    return setline


# VSM相似度计算
def vsm_similarity(queried_file, query_file, output_fname=None):
    # 生成被查询集
    queried_line = set_generation(queried_file)
    # 生成查询集
    query_line = set_generation(query_file)

    # 被查询集生成词典和corpus
    dictionary = corpora.Dictionary(queried_line)
    corpus = [dictionary.doc2bow(text) for text in queried_line]

    # 计算tfidf值
    tfidf_model = models.TfidfModel(corpus)
    corpus_tfidf = tfidf_model[corpus]

    # 待检索的文档向量初始化一个相似度计算的对象
    corpus_sim = similarities.MatrixSimilarity(corpus_tfidf)

    # 查询集生成corpus和tfidf值
    query_corpus = [dictionary.doc2bow(text) for text in query_line]  # 在每句话中每个词语出现的频率
    query_tfidf = tfidf_model[query_corpus]
    sim = pd.DataFrame(corpus_sim[query_tfidf])
    if output_fname is not None:
        sim.to_excel(output_fname)
    return sim

# def IR_based_feature_generation(dataset_name):
#     base_path = f'../../datasets/{dataset_name}'
#     finegrained_cc_path = f'./FineGrained/{dataset_name}'
#     output_path = f'./FineGrained/{dataset_name}'
#     os.makedirs(output_path, exist_ok=True)
#
#     uc_path = os.path.join(base_path, 'uc_doc.txt')
#     cc_subfiles = [
#         'class_attribute_doc.txt',
#         'class_comment_doc.txt',
#         'class_name_doc.txt',
#         'method_comment_doc.txt',
#         'method_name_doc.txt',
#         'method_parameter_doc.txt',
#         'method_return_doc.txt'
#     ]
#     sheet_names = [f.split('_doc.txt')[0] for f in cc_subfiles]
#
#     # 获取原始文件名对应关系
#     uc_names = os.listdir(os.path.join(base_path, 'uc'))
#     cc_names = os.listdir(os.path.join(base_path, 'cc'))
#     result_writer = pd.ExcelWriter(os.path.join(output_path, 'sim_result.xlsx'), engine='xlsxwriter')
#
#     for cc_file, sheet_name in zip(cc_subfiles, sheet_names):
#         cc_path = os.path.join(finegrained_cc_path, cc_file)
#
#         sim = vsm_similarity(cc_path, uc_path)  # uc 是查询集
#         num_links = int(sim.size * 0.05)  # top 5%
#         sim_flat = sim.stack().reset_index()
#         sim_flat.columns = ['uc_idx', 'cc_idx', 'score']
#         top_sim = sim_flat.nlargest(num_links, 'score')
#
#         # 映射文件名，并添加 'sim'
#         top_sim['requirement_name'] = top_sim['uc_idx'].apply(lambda i: uc_names[i])
#         top_sim['class_name'] = top_sim['cc_idx'].apply(lambda i: cc_names[i])
#         top_sim['sim'] = 'sim'
#
#         result_df = top_sim[['requirement_name', 'class_name', 'sim']]
#         result_df.to_excel(result_writer, sheet_name=sheet_name, index=False)
#
#     result_writer.close()

def IR_based_feature_generation(dataset_name):
    base_path = f'../../datasets/{dataset_name}'
    finegrained_cc_path = f'./FineGrained/{dataset_name}'
    output_path = f'./FineGrained/{dataset_name}'
    os.makedirs(output_path, exist_ok=True)

    uc_path = os.path.join(base_path, 'uc_doc.txt')
    cc_subfiles = [
        'class_attribute_doc.txt',
        'class_comment_doc.txt',
        'class_name_doc.txt',
        'method_comment_doc.txt',
        'method_name_doc.txt',
        'method_parameter_doc.txt',
        'method_return_doc.txt'
    ]
    sheet_names = [f.split('_doc.txt')[0] for f in cc_subfiles]

    # 获取原始文件名对应关系
    uc_names = os.listdir(os.path.join(base_path, 'uc'))
    cc_names = os.listdir(os.path.join(base_path, 'cc'))

    # 存储每个细粒度方法的前20%相似度链接
    all_top_sim_results = {}

    # 计算每个细粒度方法的相似度
    for cc_file, sheet_name in zip(cc_subfiles, sheet_names):
        cc_path = os.path.join(finegrained_cc_path, cc_file)

        # 假设你已有函数来计算相似度，vsm_similarity() 返回一个 DataFrame 或矩阵
        sim = vsm_similarity(cc_path, uc_path)  # uc 是查询集

        # 计算前 20% 的链接数目
        num_links = int(sim.size * 0.2)  # top 20%
        sim_flat = sim.stack().reset_index()
        sim_flat.columns = ['uc_idx', 'cc_idx', 'score']
        top_sim = sim_flat.nlargest(num_links, 'score')

        # 映射文件名，并添加 'sim'
        top_sim['requirement_name'] = top_sim['uc_idx'].apply(lambda i: uc_names[i])
        top_sim['class_name'] = top_sim['cc_idx'].apply(lambda i: cc_names[i])
        top_sim['sim'] = 'sim'

        # 保存每个细粒度方法的前20%结果
        all_top_sim_results[sheet_name] = top_sim[['requirement_name', 'class_name', 'sim']]

    # 找到在所有细粒度方法中都排在前20%的链接
    # 将所有的 top_sim 放在一起
    all_results = pd.concat(all_top_sim_results.values())

    # 统计每个链接在所有细粒度方法中的出现次数
    count_series = all_results.groupby(['requirement_name', 'class_name']).size()

    # 筛选出在所有7个细粒度方法中都出现在前20%的链接
    final_results = count_series[count_series == len(cc_subfiles)].reset_index()
    final_results.columns = ['requirement_name', 'class_name', 'count']

    # 写入到 Excel 文件
    result_writer = pd.ExcelWriter(os.path.join(output_path, 'final_sim_result.xlsx'), engine='xlsxwriter')
    final_results.to_excel(result_writer, sheet_name="FinalResults", index=False)
    result_writer.close()

if __name__ == '__main__':
    datasets = ['Albergate', 'Derby', 'Drools', 'Dronology', 'eAnci', 'Groovy', 'Infinispan', 'iTrust', 'maven', 'Pig', 'Seam2', 'smos']
    # datasets = ['Derby', 'Drools', 'Infinispan', 'iTrust', 'maven', 'Pig', 'Seam2']
    for dataset in datasets:
        IR_based_feature_generation(dataset)