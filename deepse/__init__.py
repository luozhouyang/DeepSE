import logging

from .bert_embedding import (BertAllInOneEmbeddingModel, BertAvgEmbeddingModel,
                             BertCLSEmbeddingModel,
                             BertFirstLastAvgEmbeddingModel,
                             BertPoolerEmbeddingModel)
from .simcse import SimCSE, SimCSEModel
from .simcse_dataset import SimCSEDataset, UnsupSimCSEDataset
from .simcse_unsup import UnsupSimCSE, UnsupSimCSEModel

logging.basicConfig(format='%(asctime)s %(levelname)s %(filename)7s %(lineno)4d] %(message)s', level=logging.INFO)

__name__ = "deepse"
__version__ = "0.0.2"
