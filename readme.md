<div align="center">
  <h1 align="center">Natural Language–Programming Language Software Traceability Link Recovery Needs More than Textual Similarity</h1>
</div>

### 🔍 Overview of this study

We found that methods relying solely on text similarity perform poorly in the natural language–programming language (NL–PL) traceability link recovery task. Therefore, this study first verifies this phenomenon through a systematic literature review and an empirical study. Subsequently, we introduce multiple auxiliary strategy signals—including code dependency strategies—into the models in two different ways: by representing them as edges in a heterogeneous graph neural network (HGT) and by incorporating them as additional information into a large language model (LLM). Based on these approaches, we construct HGT-ALL and Gemini-ALL, with HGT and Gemini 2.5 Pro serving as the representative models for heterogeneous graph neural networks and LLMs, respectively. Both methods leverage auxiliary strategy information to enhance the performance of the original models.

![1755574087379](https://github.com/ZZYG0g0g0/nlpl/blob/main/fig/QQ20250819-125022.png)

You can run HGT-ALL and Gemini-ALL locally using the following instructions.

### 📂 Dataset

You can obtain the dataset of this study [here](https://drive.google.com/drive/folders/1-0MJEreOJr6F5lDQtJnCV5aNjQn_PDJX?dmr=1&ec=wgc-drive-hero-goto). 

### 📂 Repository list
```
│  readme.md                                    # Description of this replication package
│  requirements.txt                             # 环境配置内容
│  statistic.xlsx                               # 系统文献综述搜寻到的文献
├─fig                                           # readme文件中的图片
│      
├─preprocessing                                 
│      textual_preprocessing.py                  # 对需求和代码制品进行预处理
│      
├─RQ2
│      cooccurrence_different_ratio.py          # 计算源制品和目标制品间的Different ratio比例
│      cooccurrence_pretrained.py               # 使用Roberta模型完成源制品和目标制品的TLR任务并进行评估
│      pretrain.py                              # 预训练模型代码
│      Roberta.py
│      
└─RQ4
    │  gemini.py
    │  Gemini_strategy.py
    │  HGT.py
    │  
    ├─HGNN
    │      GraphCodeBERT.py
    │      Roberta.py
    │      
    └─strategy
        │  IR_model.py
        │  rank_bm25.py
        │  
        ├─CodeDependency
        │      Albergate.xlsx
        │      ...
        │      
        ├─FineGrained
        │  ├─Albergate
        │  │      class_attribute.txt
        │  │      ...      
        │          
        ├─UserFeedback
        │      Albergate.xlsx
        │      ...
        │      
        └─__pycache__
```

### ⚙️ How to Use HGT-ALL and Gemini-ALL

#### 🛠️ Environment Setup

Follow these steps to set up the environment for HGT-ALL and Gemini-ALL

```shell
conda create -n hg python=3.12
conda activate hg
pip install -r requirements.txt
```

#### 📥 Download Dataset

Create a `datasets` folder in the root path, download the dataset you need [here](https://drive.google.com/drive/folders/1-0MJEreOJr6F5lDQtJnCV5aNjQn_PDJX?dmr=1&ec=wgc-drive-hero-goto), and then place the dataset into the folder you created. 

#### 🧮 Run HGT-ALL

Modify the datasets list in `RQ4/HGNN/GraphCodeBERT.py` to the datasets you need, for example：

```python
datasets = ['Albergate', 'Drools', 'eAnci', 'Groovy', 'Infinispan', 'iTrust', 'maven', 'Pig', 'Seam2', 'smos']
```

Use the following instructions to generate feature representations of code nodes. 

```shell
python RQ4/HGNN/GraphCodeBERT.py
```

Similarly, after modifying the datasets list in `RQ4/HGNN/Roberta.py`, use the following instructions to generate feature representations of requirement nodes.  

```shell
python RQ4/HGNN/Roberta.py
```

Finally, you can run `RQ4/HGT.py` to execute HGT-ALL. Similarly, you can select the datasets and edge type sets you need to execute by modifying the datasets list and the edge_type_combinations list.  For example:

```python
datasets = ['smos']
edge_type_combinations = [('import', 'extend', 'call', 'class_attribute', 'class_comment', 'class_name', 'method_comment', 'method_return', 'method_parameter', 'method_name', 'user_feedback')]
```

```shell
python RQ4/HGT.py
```



#### 🤖 Run Gemini-ALL

Modify the `api_key` and `base_url` to set the API key and select the model provider. For example:

```python
client = OpenAI(
    api_key="sk-or-v1-74f9********",  # Replace with your actual API key
    base_url="https://openrouter.ai/api/v1"  # openrouter API base URL
)
```

Select an appropriate LLM model; for example, this study uses Gemini 2.5 Pro.  For example:

```python
model="google/gemini-2.5-pro-preview"
```

Finally, you can run `RQ4/gemini.py` to execute Gemini-ALL. Similarly, you can select the datasets and strategies you need to execute by modifying the datasets list and the strategies list.  For example:

```python
 datasets = ['Derby']
 strategies = ['code dependency', 'user feedback', 'fine grained']
```



```shell
python RQ4/gemini.py
```
















