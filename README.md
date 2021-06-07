# DeepSE

![Python package](https://github.com/luozhouyang/DeepSE/workflows/Python%20package/badge.svg)
[![PyPI version](https://badge.fury.io/py/deepse.svg)](https://badge.fury.io/py/deepse)
[![Python](https://img.shields.io/pypi/pyversions/deepse.svg?style=plastic)](https://badge.fury.io/py/deepse)

**DeepSE**: **Sentence Embeddings** based on Deep Nerual Networks, designed for **PRODUCTION** enviroment!

## Installation

Clone git repo:

```bash
git clone https://github.com/luozhouyang/deepse.git
```

or install from pypi:

```bash
pip install -U deepse
```

## SimCSE

Train a **Unsup SimCSE** model in one line:
```bash
PRETRAINED_MODEL_PATH=/path/to/your/pretrained/bert/dir python run_simcse.py
```

> You can modify the parameters in `run_simcse.py` as you need.


