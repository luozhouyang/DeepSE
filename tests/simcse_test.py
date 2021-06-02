import os
import unittest

import tensorflow as tf
from deepse.simcse import UnsupSimCSE, UnsupSimCSEModel, unsup_simcse_loss
from deepse.simcse_dataset import UnsupSimCSEDataset
from transformers import BertTokenizer

PRETRAINED_MODEL_PATH = os.environ['PRETRAINED_MODEL_PATH']
tf.get_logger().setLevel('INFO')


class SimCSETest(unittest.TestCase):

    def test_unsup_simcse_train(self):
        model_path = os.path.join(PRETRAINED_MODEL_PATH, 'chinese_roberta_wwm_ext_L-12_H-768_A-12')
        model = UnsupSimCSEModel(model_path)
        tokenizer = BertTokenizer.from_pretrained(model_path)
        dataset = UnsupSimCSEDataset(tokenizer)
        train_dataset = dataset(
            input_files=['data/small.txt'],
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
                    'data/model', monitor='loss', save_best_only=True, save_weights_only=False)
            ])


if __name__ == "__main__":
    unittest.main()
