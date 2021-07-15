import json
import logging
import os
import re

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
                 bucket_batch_sizes=None,
                 buffer_size=100000,
                 reshuffle_each_iteration=True,
                 repeat=None,
                 max_sequence_length=512,
                 auto_shard_policy=None,
                 **kwargs):
        examples = self._load_examples(input_files)
        logging.info('Load %d examples in total.', len(examples['input_ids']))

        def _to_dataset(x):
            x = tf.ragged.constant(x)
            x = tf.data.Dataset.from_tensor_slices(x)
            x = x.map(lambda x: x)
            return x

        dataset = tf.data.Dataset.zip((
            _to_dataset(examples['input_ids']),
            _to_dataset(examples['segment_ids']),
            _to_dataset(examples['attention_mask'])
        ))
        dataset = dataset.filter(lambda x, y, z: tf.size(x) <= max_sequence_length)
        if repeat is not None and repeat > 0:
            dataset = dataset.repeat(repeat)
        dataset = dataset.shuffle(
            buffer_size=buffer_size,
            seed=None,
            reshuffle_each_iteration=reshuffle_each_iteration)
        pad = tf.constant(0, dtype=tf.int32)
        if bucket_batch_sizes is None:
            bucket_batch_sizes = [batch_size] * (1 + len(bucket_boundaries))
        assert len(bucket_batch_sizes) == (1 + len(bucket_boundaries)), "Bucket batch sizes mismatch boundaries!"
        dataset = dataset.apply(tf.data.experimental.bucket_by_sequence_length(
            element_length_func=lambda x, y, z: tf.size(x),
            bucket_boundaries=bucket_boundaries,
            bucket_batch_sizes=bucket_batch_sizes,
            padded_shapes=([None, ], [None, ], [None]),
            padding_values=(pad, pad, pad),
        )).prefetch(tf.data.AUTOTUNE)
        dataset = dataset.map(
            lambda x, y, z: ({'input_ids': x, 'segment_ids': y, 'attention_mask': z}, None),
            num_parallel_calls=tf.data.AUTOTUNE
        ).prefetch(tf.data.AUTOTUNE)
        if auto_shard_policy is not None:
            options = tf.data.Options()
            options.experimental_distribute.auto_shard_policy = auto_shard_policy
            dataset = dataset.with_options(options)
        return dataset

    def _load_examples(self, input_files, **kwargs):
        if isinstance(input_files, str):
            input_files = [input_files]
        input_ids, segment_ids, attention_mask = [], [], []
        for f in input_files:
            if not os.path.exists(f):
                logging.warning('File %s does not exist. Skipped.', f)
                continue
            with open(f, mode='rt', encoding='utf-8') as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    if not data['sequence']:
                        continue
                    encoding = self.tokenizer.encode(data['sequence'])
                    input_ids.append(encoding.ids)
                    segment_ids.append(encoding.type_ids)
                    attention_mask.append(encoding.attention_mask)
        examples = {
            'input_ids': input_ids,
            'segment_ids': segment_ids,
            'attention_mask': attention_mask
        }
        return examples


class SupervisedSimCSEDataset:

    def __init__(self, tokenizer: BertWordPieceTokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def __call__(self,
                 input_files,
                 batch_size=64,
                 bucket_boundaries=[50, 100, 150, 200, 250, 300, 350, 400],
                 bucket_batch_sizes=None,
                 buffer_size=100000,
                 reshuffle_each_iteration=True,
                 repeat=None,
                 max_sequence_length=512,
                 auto_shard_policy=None,
                 **kwargs):
        examples = self._load_examples(input_files, **kwargs)
        logging.info('Load %d examples in total.', len(examples['input_ids']))

        def _to_dataset(x):
            x = tf.ragged.constant(x)
            x = tf.data.Dataset.from_tensor_slices(x)
            x = x.map(lambda x: x)
            return x

        dataset = tf.data.Dataset.zip((
            _to_dataset(examples['input_ids']),
            _to_dataset(examples['segment_ids']),
            _to_dataset(examples['attention_mask']),
            _to_dataset(examples['pos_input_ids']),
            _to_dataset(examples['pos_segment_ids']),
            _to_dataset(examples['pos_attention_mask'])
        ))
        dataset = dataset.filter(lambda a, b, c, e, f, g: tf.logical_and(
            tf.size(a) <= max_sequence_length, tf.size(e) <= max_sequence_length
        ))
        if repeat is not None and repeat > 0:
            dataset = dataset.repeat(repeat)
        dataset = dataset.shuffle(
            buffer_size=buffer_size,
            seed=None,
            reshuffle_each_iteration=reshuffle_each_iteration)
        pad = tf.constant(0, dtype=tf.int32)
        if bucket_batch_sizes is None:
            bucket_batch_sizes = [batch_size] * (1 + len(bucket_boundaries))
        assert len(bucket_batch_sizes) == (1 + len(bucket_boundaries)), "Bucket batch sizes mismatch boundaries!"
        dataset = dataset.apply(tf.data.experimental.bucket_by_sequence_length(
            element_length_func=lambda a, b, c, e, f, g: tf.maximum(tf.size(a), tf.size(e)),
            bucket_boundaries=bucket_boundaries,
            bucket_batch_sizes=bucket_batch_sizes,
            padded_shapes=([None, ], [None, ], [None, ], [None, ], [None, ], [None, ]),
            padding_values=(pad, pad, pad, pad, pad, pad),
        )).prefetch(tf.data.AUTOTUNE)
        dataset = dataset.map(
            lambda a, b, c, e, f, g: (
                {
                    'input_ids': a,
                    'segment_ids': b,
                    'attention_mask': c,
                    'pos_input_ids': e,
                    'pos_segment_ids': f,
                    'pos_attention_mask': g
                }, None),
            num_parallel_calls=tf.data.AUTOTUNE
        ).prefetch(tf.data.AUTOTUNE)
        if auto_shard_policy is not None:
            options = tf.data.Options()
            options.experimental_distribute.auto_shard_policy = auto_shard_policy
            dataset = dataset.with_options(options)
        return dataset

    def _load_examples(self, input_files, **kwargs):
        if isinstance(input_files, str):
            input_files = [input_files]
        input_ids, segment_ids, attention_mask = [], [], []
        pos_input_ids, pos_segment_ids, pos_attention_mask = [], [], []
        for f in input_files:
            if not os.path.exists(f):
                logging.warning('File %s does not exist. Skipped.', f)
                continue
            with open(f, mode='rt', encoding='utf-8') as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    sequence, pos_sequence = data['sequence'].strip(), data['positive_sequence'].strip()
                    if not sequence or not pos_sequence:
                        continue
                    encoding = self.tokenizer.encode(sequence)
                    pos_encoding = self.tokenizer.encode(pos_sequence)
                    if not encoding.ids or not pos_encoding.ids:
                        continue

                    input_ids.append(encoding.ids)
                    segment_ids.append(encoding.type_ids)
                    attention_mask.append(encoding.attention_mask)

                    pos_input_ids.append(pos_encoding.ids)
                    pos_segment_ids.append(pos_encoding.type_ids)
                    pos_attention_mask.append(pos_encoding.attention_mask)

        examples = {
            'input_ids': input_ids,
            'segment_ids': segment_ids,
            'attention_mask': attention_mask,
            'pos_input_ids': pos_input_ids,
            'pos_segment_ids': pos_segment_ids,
            'pos_attention_mask': pos_attention_mask
        }
        return examples


class HardNegativeSimCSEDataset:

    def __init__(self, tokenizer: BertWordPieceTokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def __call__(self,
                 input_files,
                 batch_size=64,
                 bucket_boundaries=[50, 100, 150, 200, 250, 300, 350, 400],
                 bucket_batch_sizes=None,
                 buffer_size=100000,
                 reshuffle_each_iteration=True,
                 repeat=None,
                 max_sequence_length=512,
                 auto_shard_policy=None,
                 **kwargs):
        examples = self._load_examples(input_files, **kwargs)
        logging.info('Load %d examples in total.', len(examples['input_ids']))

        def _to_dataset(x):
            x = tf.ragged.constant(x)
            x = tf.data.Dataset.from_tensor_slices(x)
            x = x.map(lambda x: x)
            return x

        dataset = tf.data.Dataset.zip((
            _to_dataset(examples['input_ids']),
            _to_dataset(examples['segment_ids']),
            _to_dataset(examples['attention_mask']),
            _to_dataset(examples['pos_input_ids']),
            _to_dataset(examples['pos_segment_ids']),
            _to_dataset(examples['pos_attention_mask']),
            _to_dataset(examples['neg_input_ids']),
            _to_dataset(examples['neg_segment_ids']),
            _to_dataset(examples['neg_attention_mask']),
        ))
        dataset = dataset.filter(lambda a, b, c, e, f, g, h, i, j: tf.logical_and(
            tf.size(a) <= max_sequence_length,
            tf.logical_and(tf.size(e) <= max_sequence_length, tf.size(h) <= max_sequence_length)
        ))
        if repeat is not None and repeat > 0:
            dataset = dataset.repeat(repeat)
        dataset = dataset.shuffle(
            buffer_size=buffer_size,
            seed=None,
            reshuffle_each_iteration=reshuffle_each_iteration)
        pad = tf.constant(0, dtype=tf.int32)
        if bucket_batch_sizes is None:
            bucket_batch_sizes = [batch_size] * (1 + len(bucket_boundaries))
        assert len(bucket_batch_sizes) == (1 + len(bucket_boundaries)), "Bucket batch sizes mismatch boundaries!"
        dataset = dataset.apply(tf.data.experimental.bucket_by_sequence_length(
            element_length_func=lambda a, b, c, e, f, g, h, i, j: tf.maximum(
                tf.size(a),
                tf.maximum(tf.size(e), tf.size(h))
            ),
            bucket_boundaries=bucket_boundaries,
            bucket_batch_sizes=bucket_batch_sizes,
            padded_shapes=([None, ], [None, ], [None, ], [None, ], [None, ], [None, ], [None, ], [None, ], [None, ]),
            padding_values=(pad, pad, pad, pad, pad, pad, pad, pad, pad),
        )).prefetch(tf.data.AUTOTUNE)
        dataset = dataset.map(
            lambda a, b, c, e, f, g, h, i, j: (
                {
                    'input_ids': a,
                    'segment_ids': b,
                    'attention_mask': c,
                    'pos_input_ids': e,
                    'pos_segment_ids': f,
                    'pos_attention_mask': g,
                    'neg_input_ids': h,
                    'neg_segment_ids': i,
                    'neg_attention_mask': j,
                }, None),
            num_parallel_calls=tf.data.AUTOTUNE
        ).prefetch(tf.data.AUTOTUNE)
        if auto_shard_policy is not None:
            options = tf.data.Options()
            options.experimental_distribute.auto_shard_policy = auto_shard_policy
            dataset = dataset.with_options(options)
        return dataset

    def _load_examples(self, input_files, **kwargs):
        if isinstance(input_files, str):
            input_files = [input_files]
        input_ids, segment_ids, attention_mask = [], [], []
        pos_input_ids, pos_segment_ids, pos_attention_mask = [], [], []
        neg_input_ids, neg_segment_ids, neg_attention_mask = [], [], []
        for f in input_files:
            if not os.path.exists(f):
                logging.warning('File %s does not exist. Skipped.', f)
                continue
            with open(f, mode='rt', encoding='utf-8') as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    sequence = data['sequence'].strip()
                    pos_sequence = data['positive_sequence'].strip()
                    neg_sequence = data['negative_sequence'].strip()
                    if not sequence or not neg_sequence:
                        continue
                    encoding = self.tokenizer.encode(sequence)
                    pos_encoding = self.tokenizer.encode(pos_sequence) if pos_sequence else encoding
                    neg_encoding = self.tokenizer.encode(neg_sequence)
                    if not encoding.ids or not pos_encoding.ids:
                        continue

                    input_ids.append(encoding.ids)
                    segment_ids.append(encoding.type_ids)
                    attention_mask.append(encoding.attention_mask)

                    pos_input_ids.append(pos_encoding.ids)
                    pos_segment_ids.append(pos_encoding.type_ids)
                    pos_attention_mask.append(pos_encoding.attention_mask)

                    neg_input_ids.append(neg_encoding.ids)
                    neg_segment_ids.append(neg_encoding.type_ids)
                    neg_attention_mask.append(neg_encoding.attention_mask)

        examples = {
            'input_ids': input_ids,
            'segment_ids': segment_ids,
            'attention_mask': attention_mask,
            'pos_input_ids': pos_input_ids,
            'pos_segment_ids': pos_segment_ids,
            'pos_attention_mask': pos_attention_mask,
            'neg_input_ids': neg_input_ids,
            'neg_segment_ids': neg_segment_ids,
            'neg_attention_mask': neg_attention_mask,
        }
        return examples
