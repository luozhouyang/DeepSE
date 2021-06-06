import json
import os
import unittest

import requests
from deepse.bert_embedding import (BertAllInOneEmbeddingModel,
                                   BertAvgEmbeddingModel,
                                   BertCLSEmbeddingModel,
                                   BertFirstLastAvgEmbeddingModel,
                                   BertPoolerEmbeddingModel)
from tokenizers import BertWordPieceTokenizer

BERT_PATH = os.path.join(os.environ['PRETRAINED_MODEL_PATH'], 'chinese_roberta_wwm_ext_L-12_H-768_A-12')
tokenizer = BertWordPieceTokenizer.from_file(os.path.join(BERT_PATH, 'vocab.txt'))


class BertEmbeddingTest(unittest.TestCase):

    def _request_embedding(self, url, text):
        encoding = tokenizer.encode(text)
        req = {
            'inputs': {
                'input_ids': [encoding.ids],
                'segment_ids': [encoding.type_ids],
                'attention_mask': [encoding.attention_mask]
            }
        }
        resp = requests.post(url, data=json.dumps(req)).json()
        return resp

    def test_bert_cls_embedding(self):
        bert = BertCLSEmbeddingModel(BERT_PATH)
        bert.save('data/bert-cls/1', save_format='tf')

    def test_bert_cls_embedding_serving(self):
        resp = self._request_embedding(
            url='http://localhost:8501/v1/models/deepse:predict',
            text='我爱自然语言处理'
        )
        print(resp['outputs'][0])

    def test_bert_pooler_embedding(self):
        bert = BertPoolerEmbeddingModel(BERT_PATH)
        bert.save('data/bert-pooler/1')

    def test_bert_pooler_embedding_serving(self):
        resp = self._request_embedding(
            url='http://localhost:8501/v1/models/deepse:predict',
            text='我爱自然语言处理'
        )
        print(resp['outputs'][0])

    def test_bert_avg_embedding(self):
        bert = BertAvgEmbeddingModel(BERT_PATH)
        bert.save('data/bert-avg/1')

    def test_bert_avg_embedding_serving(self):
        resp = self._request_embedding(
            url='http://localhost:8501/v1/models/deepse:predict',
            text='我爱自然语言处理'
        )
        print(resp['outputs'][0])

    def test_bert_first_last_avg_embedding(self):
        bert = BertFirstLastAvgEmbeddingModel(BERT_PATH)
        bert.save('data/bert-first-last-avg/1')

    def test_bert_first_last_avg_embedding_serving(self):
        resp = self._request_embedding(
            url='http://localhost:8501/v1/models/deepse:predict',
            text='我爱自然语言处理'
        )
        print(resp['outputs'][0])

    def test_bert_all_in_one_embedding(self):
        bert = BertAllInOneEmbeddingModel(BERT_PATH)
        bert.save('data/bert-all-in-one/1')

    def test_bert_all_in_one_embedding_serving(self):
        resp = self._request_embedding(
            url='http://localhost:8501/v1/models/deepse:predict',
            text='我爱自然语言处理'
        )
        print(resp['outputs'].keys())
        cls_embedding = resp['outputs']['cls'][0]
        pooler_embedding = resp['outputs']['pooler'][0]
        avg_embedding = resp['outputs']['avg'][0]
        first_last_avg = resp['outputs']['first-last-avg'][0]
        print()
        print('cls embedding: \n', cls_embedding)
        print('avg embedding: \n', avg_embedding)
        print('pooler embedding: \n', pooler_embedding)
        print('first-last-avg embedding: \n', first_last_avg)


if __name__ == "__main__":
    unittest.main()
