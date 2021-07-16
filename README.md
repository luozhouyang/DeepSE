# DeepSE

![Python package](https://github.com/luozhouyang/DeepSE/workflows/Python%20package/badge.svg)
[![PyPI version](https://badge.fury.io/py/deepse.svg)](https://badge.fury.io/py/deepse)
[![Python](https://img.shields.io/pypi/pyversions/deepse.svg?style=plastic)](https://badge.fury.io/py/deepse)

**DeepSE**: 面向生产环境的**Sentence Embedding**！

# 目录
1. [安装](#安装)
2. [实现的模型](#实现的模型)
    - 2.1 [BERT和RoBERTa](#BERT和RoBERTa)
    - 2.2 [SimCSE](#SimCSE)

## 安装

克隆仓库:

```bash
git clone https://github.com/luozhouyang/deepse.git
```

或者从`pypi`安装:

```bash
pip install -U deepse
```

## 实现的模型 

目前支持的模型如下：

* [x] 原始的BERT和RoBERTa
* [x] SimCSE
    - [x] Unsupervised SimCSE
    - [x] Supervised SimCSE
    - [x] Supervised SimCSE (with hard negative)

### BERT和RoBERTa

TODO: 补充文档

### SimCSE

对于不同的版本，训练数据的格式稍有不同，但是都是普通文本文件，每一行都是一个JSON格式的训练样本。

对于`Unsupervised SimCSE`，每个样本都需要含有`sequence`字段。举例如下：
```bash
{"sequence": "我很讨厌自然语言处理"}
{"sequence": "我对自然语言处理很感兴趣"}
```

对于`Supervised SimCSE`，每个样本都需要包含`sequence`和`positive_sequence`字段。举例如下：
```bash
{"sequence": "我很讨厌自然语言处理", "positive_sequence": "我不喜欢自然语言处理"}
{"sequence": "我对自然语言处理很感兴趣", "positive_sequence": "我想了解自然语言处理"}
```

对于`Supervised SimCSE with hard negative`，每个样本都需要包含`sequence`、`positive_sequence`和`negative_sequence`字段。如果`positive_sequence`字段为空，则会自动使用`sequence`作为自己的`positive_sequence`。举例如下：
```bash
{"sequence": "我很讨厌自然语言处理", "positive_sequence": "我不喜欢自然语言处理", "negative_sequence": "我想了解自然语言处理"}
{"sequence": "我对自然语言处理很感兴趣", "positive_sequence": "我想了解自然语言处理", "negative_sequence": "我很讨厌自然语言处理"}
```

然后，使用以下命令即可训练：

```bash
export PRETRAINED_MODEL_PATH=/path/to/your/pretrained/bert/dir 
nohup python run_simcse.py >> log/run_simcse.log 2>&1 &
tail -f log/run_simcse.log
```

> 参数可以到`run_simcse.py`直接修改。
> 
> 模型会同时保存成Checkpoint格式和SavedModel格式，后者你可以直接用tensorflow/serving部署在生产环境。


