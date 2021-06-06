import json
import os

import numpy as np
import requests
import tensorflow as tf
from tokenizers import BertWordPieceTokenizer


class SimCSEClient:

    def __init__(self, vocab_file):
        super().__init__()
        self.tokenizer = BertWordPieceTokenizer.from_file(vocab_file)

    def call(self, text):
        input_ids = self.tokenizer.encode(text)
        req = {
            'inputs': {
                'input_ids': [input_ids],
                'segment_ids': [[0] * len(input_ids)],
                'attention_mask': [[1] * len(input_ids)]
            }
        }
        resp = requests.post('http://localhost:8501/v1/models/simcse:predict', data=json.dumps(req)).json()
        outputs = resp['outputs']
        print(outputs[0])


if __name__ == "__main__":
    model_path = os.path.join(os.environ['PRETRAINED_MODEL_PATH'], 'chinese_roberta_wwm_ext_L-12_H-768_A-12')
    c = SimCSEClient(vocab_file=os.path.join(model_path, 'vocab.txt'))
    c.call('I love NLP')
