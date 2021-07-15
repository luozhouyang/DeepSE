import os
import unittest

import numpy as np
import tensorflow as tf
from deepse.simcse_client import SimCSEClient

VOCAB_FILE = os.path.join(os.environ['PRETRAINED_BERT_PATH'], 'vocab.txt')


class SimCSEClientTest(unittest.TestCase):

    def test_simcse_client(self):
        c = SimCSEClient(
            url='http://localhost:8401/v1/models/simcse-atec:predict',
            vocab_file=VOCAB_FILE
        )
        sentences = [
            '为什么借呗提前还款显示不了额度',
            '借呗提前一个月还款',
            '蚂蚁借呗为什么到还款日没有自动扣款',
            '我的借呗怎么系统还没自动扣款'
        ]
        embeddings = c.predict(sentences)
        embeddings = [np.array(x) for x in embeddings]
        norm_embeddings = tf.linalg.normalize(embeddings, axis=-1)[0]
        sim = tf.matmul(norm_embeddings, norm_embeddings, transpose_b=True)
        print(sim)


if __name__ == "__main__":
    unittest.main()
