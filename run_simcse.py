import logging
import os

import tensorflow as tf
from tokenizers import BertWordPieceTokenizer

from deepse.simcse import SimCSE
from deepse.simcse_dataset import SimCSEDataset

CONFIG = {
    'pretrained_model_dir': os.environ['PRETRAINED_MODEL_PATH'],
    'model_dir': 'models/simcse',
    'epochs': 10,
    'lr': 3e-5,
    'pooling_strategy': 'cls'
}

CONFIG.update({
    'train_input_files': [
        'data/simcse.txt',
    ],
    'train_batch_size': 32,
    'train_buffer_size': 1000000,  # 一般不小于训练样本总数
    'train_bucket_boundaries': [50, 100, 150, 200],  # 按照文本长度分桶，提升计算效率
    'valid_input_files': [
        'data/simcse.txt',
    ],
    'valid_batch_size': 32,
    'valid_buffer_size': 2000,  # 一般不小于验证样本总数
    'valid_bucket_boundaries': [50, 100, 150, 200],  # 按照文本长度分桶，提升计算效率
})

model = SimCSE(
    pretrained_model_dir=CONFIG['pretrained_model_dir'],
    lr=CONFIG['lr'],
)

tokenizer = BertWordPieceTokenizer.from_file(os.path.join(CONFIG['pretrained_model_dir'], 'vocab.txt'))
dataset = SimCSEDataset(tokenizer=tokenizer)
train_dataset = dataset(
    input_files=CONFIG['train_input_files'],
    batch_size=CONFIG['train_batch_size'],
    bucket_boundaries=CONFIG['train_bucket_boundaries'],
    buffer_size=CONFIG['train_buffer_size'],
)
valid_dataset = dataset(
    input_files=CONFIG['valid_input_files'],
    batch_size=CONFIG['valid_batch_size'],
    bucket_boundaries=CONFIG['valid_bucket_boundaries'],
    buffer_size=CONFIG['valid_buffer_size'],
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
