import os
import unittest

from tools.simcse_client import SimCSEClient


class SimCSEClientTest(unittest.TestCase):

    def test_simcse(self):
        model_path = os.path.join(os.environ['PRETRAINED_MODEL_PATH'], 'chinese_roberta_wwm_ext_L-12_H-768_A-12')
        c = SimCSEClient(vocab_file=os.path.join(model_path, 'vocab.txt'))
        c.call('I love NLP')


if __name__ == "__main__":
    unittest.main()
