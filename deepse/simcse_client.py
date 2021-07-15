import json

import requests
from tokenizers import BertWordPieceTokenizer


class SimCSEClient:

    def __init__(self, url, vocab_file, **kwargs):
        self.url = url
        self.tokenizer = BertWordPieceTokenizer(vocab_file)

    def predict(self, sentences, **kwargs):
        if not isinstance(sentences, list):
            sentences = [sentences]
        input_ids, segment_ids, attention_mask = self._tokenize(sentences, **kwargs)
        req = {
            'inputs': {
                'input_ids': input_ids,
                'segment_ids': segment_ids,
                'attention_mask': attention_mask
            }
        }
        resp = requests.post(self.url, data=json.dumps(req)).json()
        if not resp:
            return None
        outputs = resp['outputs']
        return outputs

    def _tokenize(self, sentences, **kwargs):
        input_ids, segment_ids, attention_mask = [], [], []
        for sent in sentences:
            encoding = self.tokenizer.encode(sent)
            if len(encoding.ids) > 512:
                raise ValueError('Sentence is too long!')
            input_ids.append(encoding.ids)
            segment_ids.append(encoding.type_ids)
            attention_mask.append(encoding.attention_mask)
        maxlen = max([len(x) for x in input_ids])
        padded_input_ids, padded_segment_ids, padded_attention_mask = [], [], []
        for ids, segids, mask in zip(input_ids, segment_ids, attention_mask):
            padded_input_ids.append(ids + [0] * (maxlen - len(ids)))
            padded_segment_ids.append(segids + [0] * (maxlen - len(segids)))
            padded_attention_mask.append(mask + [0] * (maxlen - len(mask)))
        return padded_input_ids, padded_segment_ids, padded_attention_mask
