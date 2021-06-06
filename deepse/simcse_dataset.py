import logging
import os

import tensorflow as tf
from tokenizers import BertWordPieceTokenizer


class UnsupSimCSEDataset:

    def __init__(self, tokenizer: BertWordPieceTokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def __call__(self,
                 input_files,
                 batch_size=64,
                 bucket_boundaries=[50, 100, 150, 200, 250, 300, 350, 400],
                 buffer_size=100000,
                 reshuffle_each_iteration=True,
                 repeat=None,
                 **kwargs):
        input_ids, segment_ids, attn_mask = self._load_files(input_files)

        def _to_dataset(x):
            x = tf.ragged.constant(x)
            x = tf.data.Dataset.from_tensor_slices(x)
            x = x.map(lambda x: x)
            return x

        input_ids = _to_dataset(input_ids)
        segment_ids = _to_dataset(segment_ids)
        attn_mask = _to_dataset(attn_mask)
        dataset = tf.data.Dataset.zip((input_ids, segment_ids, attn_mask))
        dataset = dataset.filter(lambda x, y, z: tf.size(x) <= 512)
        if repeat is not None and repeat > 0:
            dataset = dataset.repeat(repeat)
        dataset = dataset.shuffle(
            buffer_size=buffer_size,
            seed=None,
            reshuffle_each_iteration=reshuffle_each_iteration)
        pad = tf.constant(0, dtype=tf.int32)
        dataset = dataset.apply(tf.data.experimental.bucket_by_sequence_length(
            element_length_func=lambda x, y, z: tf.size(x),
            bucket_boundaries=bucket_boundaries,
            bucket_batch_sizes=[batch_size] * (1 + len(bucket_boundaries)),
            padded_shapes=([None, ], [None, ], [None]),
            padding_values=(pad, pad, pad),
        )).prefetch(tf.data.AUTOTUNE)
        dataset = dataset.map(
            lambda x, y, z: ({'input_ids': x, 'segment_ids': y, 'attention_mask': z}, None),
            num_parallel_calls=tf.data.AUTOTUNE
        ).prefetch(tf.data.AUTOTUNE)
        return dataset

    def _load_files(self, input_files, **kwargs):
        if isinstance(input_files, str):
            input_files = [input_files]
        input_token_ids, segment_ids, attn_masks = [], [], []
        for f in input_files:
            if not os.path.exists(f):
                logging.warning('File %s does not exist. Skipped.', f)
                continue
            with open(f, mode='rt', encoding='utf-8') as fin:
                for line in fin:
                    encoding = self.tokenizer.encode(line.strip())
                    input_token_ids.append(encoding.ids)
                    segment_ids.append(encoding.type_ids)
                    attn_masks.append(encoding.attention_mask)
        return input_token_ids, segment_ids, attn_masks
