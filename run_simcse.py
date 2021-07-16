import logging
import os

import tensorflow as tf
from tokenizers import BertWordPieceTokenizer

from deepse.simcse import SimCSE
from deepse.simcse_dataset import (HardNegativeSimCSEDataset,
                                   SupervisedSimCSEDataset, UnsupSimCSEDataset)

CONFIG = {
    'pretrained_model_dir': os.environ['PRETRAINED_MODEL_PATH'],
    'model_dir': 'models/simcse',
    'epochs': 10,
    'lr': 3e-5,
    'pooling_strategy': 'cls',  # ['cls', 'pooler']
    'mode': 'unsup',  # ['unsup', 'sup', 'hardneg']
    'max_sequence_length': 100
}

CONFIG.update({
    'train_input_files': [
        'data/simcse_unsup.jsonl',
    ],
    'train_batch_size': 80,
    'train_buffer_size': 1000000,  # 一般不小于训练样本总数
    'train_bucket_boundaries': [25, 50, 75],  # 按照文本长度分桶，提升计算效率
    'valid_input_files': [
        'data/simcse_unsup.jsonl',
    ],
    'valid_batch_size': 64,
    'valid_buffer_size': 2000,  # 一般不小于验证样本总数
    'valid_bucket_boundaries': [25, 50, 75],  # 按照文本长度分桶，提升计算效率
})

model = SimCSE(
    pretrained_model_dir=CONFIG['pretrained_model_dir'],
    lr=CONFIG['lr'],
    mode=CONFIG['mode'],
    pooling_strategy=CONFIG['pooling_strategy']
)

tokenizer = BertWordPieceTokenizer.from_file(os.path.join(CONFIG['pretrained_model_dir'], 'vocab.txt'))
if CONFIG['mode'] == 'sup':
    dataset = SupervisedSimCSEDataset(tokenizer=tokenizer)
elif CONFIG['mode'] == 'hardneg':
    dataset = HardNegativeSimCSEDataset(tokenizer=tokenizer)
else:
    dataset = UnsupSimCSEDataset(tokenizer=tokenizer)

train_dataset = dataset(
    input_files=CONFIG['train_input_files'],
    batch_size=CONFIG['train_batch_size'],
    bucket_boundaries=CONFIG['train_bucket_boundaries'],
    buffer_size=CONFIG['train_buffer_size'],
    max_sequence_length=CONFIG['max_sequence_length'],
)
valid_dataset = dataset(
    input_files=CONFIG['valid_input_files'],
    batch_size=CONFIG['valid_batch_size'],
    bucket_boundaries=CONFIG['valid_bucket_boundaries'],
    buffer_size=CONFIG['valid_buffer_size'],
    max_sequence_length=CONFIG['max_sequence_length'],
)

# 开始训练模型
model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=CONFIG['epochs'],
    callbacks=[
        # 保存ckpt格式的模型
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(CONFIG['model_dir'], 'ckpt/simcse-{epoch:04d}.ckpt'),
            save_weights_only=True),
        # 保存SavedModel格式的模型，用于Serving部署
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(CONFIG['model_dir'], 'export/{epoch}'),
            save_weights_only=False)
        # TODO: 增加性能评估的回调
    ]
)
