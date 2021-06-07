import json

import numpy as np
import requests
import tensorflow as tf
from tokenizers import BertWordPieceTokenizer


class SimCSEClient:

    def __init__(self, vocab_file, url='http://localhost:8601/v1/models/simcse:predict'):
        super().__init__()
        self.tokenizer = BertWordPieceTokenizer.from_file(vocab_file)
        self.url = url

    def call(self, text):
        encoding = self.tokenizer.encode(text)
        req = {
            'inputs': {
                'input_ids': [encoding.ids],
                'segment_ids': [encoding.type_ids],
                'attention_mask': [encoding.attention_mask]
            }
        }
        resp = requests.post(self.url, data=json.dumps(req)).json()
        outputs = resp['outputs']
        print(outputs[0])
