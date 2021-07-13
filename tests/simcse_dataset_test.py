import os
import unittest

from deepse.simcse_dataset import SimCSEDataset, UnsupSimCSEDataset
from tokenizers import BertWordPieceTokenizer

PATH = os.environ['PRETRAINED_MODEL_PATH']


class SimCSEDatasetTest(unittest.TestCase):

    def test_unsup_simcse_dataset(self):
        tokenizer = BertWordPieceTokenizer.from_file(
            os.path.join(PATH, 'chinese_roberta_wwm_ext_L-12_H-768_A-12', 'vocab.txt'))
        dataset = UnsupSimCSEDataset(tokenizer)
        train_dataset = dataset(
            input_files=['data/small.txt'],
            batch_size=4,
            bucket_boundaries=[20],
            buffer_size=10,
        )
        print(next(iter(train_dataset)))

    def test_simcse_dataset(self):
        tokenizer = BertWordPieceTokenizer.from_file(
            os.path.join(PATH, 'chinese_roberta_wwm_ext_L-12_H-768_A-12', 'vocab.txt'))
        dataset = SimCSEDataset(tokenizer)
        train_dataset = dataset(
            input_files=['data/simcse.txt'],
            batch_size=4,
            bucket_boundaries=[5],
            buffer_size=10,
        )
        print(next(iter(train_dataset)))


if __name__ == "__main__":
    unittest.main()
