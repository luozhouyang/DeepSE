import os
import unittest

import tensorflow as tf
from deepse.simcse import SimCSE
from deepse.simcse_dataset import (HardNegativeSimCSEDataset,
                                   SupervisedSimCSEDataset, UnsupSimCSEDataset)
from tokenizers import BertWordPieceTokenizer

PRETRAINED_MODEL_PATH = os.environ['PRETRAINED_MODEL_PATH']


class SimCSETest(unittest.TestCase):

    def test_unsup_simcse_train(self):
        model_path = os.path.join(PRETRAINED_MODEL_PATH, 'chinese_roberta_wwm_ext_L-12_H-768_A-12')
        model = SimCSE(model_path, mode='unsup')
        tokenizer = BertWordPieceTokenizer.from_file(os.path.join(model_path, 'vocab.txt'))
        dataset = UnsupSimCSEDataset(tokenizer)
        train_dataset = dataset(
            input_files=['data/simcse_unsup.jsonl'],
            batch_size=4,
            bucket_boundaries=[20],
            buffer_size=10,
            repeat=100,
        )
        model.fit(
            train_dataset,
            validation_data=train_dataset,
            epochs=2,
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint(
                    'data/simcse-unsup', monitor='loss', save_weights_only=False)
            ])

    def test_supervised_simcse_train(self):
        model_path = os.path.join(PRETRAINED_MODEL_PATH, 'chinese_roberta_wwm_ext_L-12_H-768_A-12')
        model = SimCSE(model_path, mode='sup')
        tokenizer = BertWordPieceTokenizer.from_file(os.path.join(model_path, 'vocab.txt'))
        dataset = SupervisedSimCSEDataset(tokenizer)
        train_dataset = dataset(
            input_files=['data/simcse_supervised.jsonl'],
            batch_size=4,
            bucket_boundaries=[20],
            buffer_size=10,
            repeat=100,
        )
        model.fit(
            train_dataset,
            validation_data=train_dataset,
            epochs=2,
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint(
                    'data/simcse-sup', monitor='loss', save_weights_only=False)
            ])

    def test_hardneg_simcse_train(self):
        model_path = os.path.join(PRETRAINED_MODEL_PATH, 'chinese_roberta_wwm_ext_L-12_H-768_A-12')
        model = SimCSE(model_path, mode='hardneg')
        tokenizer = BertWordPieceTokenizer.from_file(os.path.join(model_path, 'vocab.txt'))
        dataset = HardNegativeSimCSEDataset(tokenizer)
        train_dataset = dataset(
            input_files=['data/simcse_hardnegative.jsonl'],
            batch_size=4,
            bucket_boundaries=[20],
            buffer_size=10,
            repeat=100,
        )
        model.fit(
            train_dataset,
            validation_data=train_dataset,
            epochs=2,
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint(
                    'data/simcse-hardneg', monitor='loss', save_weights_only=False)
            ])


if __name__ == "__main__":
    unittest.main()
