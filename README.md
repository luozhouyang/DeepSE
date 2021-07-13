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

### BERT和RoBERTa

TODO: 补充文档

### SimCSE

**SimCSE**模型有多种形式，包括**有监督**和**无监督**版本，其中**有监督**版本又有**是否包含hard negative**之分。

目前实现列表如下：

* [x] 无监督SimCSE
* [x] 有监督SimCSE
* [ ] 有监督SimCSE with hard negative

训练一个**无监督SimCSE**模型，需要的训练数据格式是：**每行一个句子**。

然后，使用以下命令即可训练：

```bash
PRETRAINED_MODEL_PATH=/path/to/your/pretrained/bert/dir python run_simcse_unsup.py
```

> 参数可以到`run_simcse_unsup.py`直接修改。
> 
> 模型会同时保存成Checkpoint格式和SavedModel格式，后者你可以直接用tensorflow/serving部署在生产环境。


训练一个**有监督的SimCSE**模型，需要的训练数据格式是：**每行两个句子，使用任意的分隔符间隔开即可**(可以在Dataset的构建过程中指定分隔符`sep`)。

然后，使用以下命令即可训练：

```bash
PRETRAINED_MODEL_PATH=/path/to/your/pretrained/bert/dir python run_simcse.py
```

> 参数可以到`run_simcse.py`直接修改。
> 
> 模型会同时保存成Checkpoint格式和SavedModel格式，后者你可以直接用tensorflow/serving部署在生产环境。


