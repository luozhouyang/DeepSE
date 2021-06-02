import logging
import os

import tensorflow as tf
from transformers import BertTokenizer

from deepse.simcse import UnsupSimCSEModel
from deepse.simcse_dataset import UnsupSimCSEDataset

PRETRAINED_MODEL_PATH = os.environ['PRETRAINED_MODEL_PATH']
PRETRAINED_BERT_PATH = os.path.join(PRETRAINED_MODEL_PATH, 'chinese_roberta_wwm_ext_L-12_H-768_A-12')

MODEL_DIR = 'data/model'

EPOCHS = 10
LEARNINT_RATE = 1e-5
POOLING_STRATEGY = 'cls'  # embedding的pooling策略

TRAIN_INPUT_FILES = [
    'data/small.txt'
]
TRAIN_BATCH_SIZE = 32
TRAIN_BUFFER_SIZE = 100000  # 一般不小于训练样本总数
TRAIN_BUCKET_BOUNDARIES = [50, 100, 150, 200]  # 按照文本长度分桶，提升计算效率

VALID_INPUT_FILES = [
    'data/small.txt'
]
VALID_BATCH_SIZE = 32
VALID_BUFFER_SIZE = 2000  # 一般不小于验证样本总数
VALID_BUCKET_BOUNDARIES = [50, 100, 150, 200]  # 按照文本长度分桶，提升计算效率

model = UnsupSimCSEModel(
    pretrained_model_dir=PRETRAINED_BERT_PATH,
    strategy=POOLING_STRATEGY,
    lr=LEARNINT_RATE,
)

tokenizer = BertTokenizer.from_pretrained(PRETRAINED_BERT_PATH)
dataset = UnsupSimCSEDataset(tokenizer=tokenizer)
train_dataset = dataset(
    input_files=TRAIN_INPUT_FILES,
    batch_size=TRAIN_BATCH_SIZE,
    bucket_boundaries=TRAIN_BUCKET_BOUNDARIES,
    buffer_size=TRAIN_BUFFER_SIZE,
)
valid_dataset = dataset(
    input_files=VALID_INPUT_FILES,
    batch_size=VALID_BATCH_SIZE,
    bucket_boundaries=VALID_BUCKET_BOUNDARIES,
    buffer_size=VALID_BUFFER_SIZE,
)

# 开始训练模型
model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=EPOCHS,
    callbacks=[
        # 保存ckpt格式的模型
        tf.keras.callbacks.ModelCheckpoint(filepath=MODEL_DIR + '/ckpt', save_weights_only=False),
        # 保存SavedModel格式的模型，用于Serving部署
        tf.keras.callbacks.ModelCheckpoint(filepath=MODEL_DIR + '/export/{epoch}', save_weights_only=True)
        # TODO: 增加性能评估的回调
    ]
)
