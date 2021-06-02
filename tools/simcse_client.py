import requests
from transformers import BertTokenizer
import tensorflow as tf

import numpy as np
import os
import json


class SimCSEClient:

    def __init__(self):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(
            os.path.join(os.environ['PRETRAINED_MODEL_PATH'], 'chinese_roberta_wwm_ext_L-12_H-768_A-12')
        )

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
    c = SimCSEClient()
    c.call('I love NLP')
