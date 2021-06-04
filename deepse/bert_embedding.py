import tensorflow as tf
from transformers_keras import Bert


def dummy_bert_inputs():
    dummy_input_ids = tf.reshape(tf.range(0, 512, dtype=tf.int32), shape=(1, 512))
    dummy_segment_ids = tf.constant([0] * 512, dtype=tf.int32, shape=(1, 512))
    dummy_attn_mask = tf.constant([1] * 512, dtype=tf.int32, shape=(1, 512))
    return dummy_input_ids, dummy_segment_ids, dummy_attn_mask


class BertCLSEmbedding(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, sequence_outputs):
        return sequence_outputs[:, 0]


def BertCLSEmbeddingModel(pretrained_bert_dir, **kwargs):
    input_ids = tf.keras.layers.Input(shape=[None, ], dtype=tf.int32, name='input_ids')
    segment_ids = tf.keras.layers.Input(shape=[None, ], dtype=tf.int32, name='segment_ids')
    attn_mask = tf.keras.layers.Input(shape=[None, ], dtype=tf.int32, name='attention_mask')
    bert = Bert.from_pretrained(pretrained_model_dir=pretrained_bert_dir)

    sequence_outputs, _ = bert(inputs=(input_ids, segment_ids, attn_mask))
    embedding = BertCLSEmbedding(name='embedding')(sequence_outputs)

    model = tf.keras.Model(inputs=[input_ids, segment_ids, attn_mask], outputs=embedding)

    dummy_input_ids, dummy_segment_ids, dummy_attn_mask = dummy_bert_inputs()
    model(inputs=(dummy_input_ids, dummy_segment_ids, dummy_attn_mask), training=False)

    model.summary()
    return model


class BertPoolerEmbedding(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, pooled_outputs):
        return pooled_outputs


def BertPoolerEmbeddingModel(pretrained_bert_dir, **kwargs):
    input_ids = tf.keras.layers.Input(shape=[None, ], dtype=tf.int32, name='input_ids')
    segment_ids = tf.keras.layers.Input(shape=[None, ], dtype=tf.int32, name='segment_ids')
    attn_mask = tf.keras.layers.Input(shape=[None, ], dtype=tf.int32, name='attention_mask')
    bert = Bert.from_pretrained(pretrained_model_dir=pretrained_bert_dir)

    _, pooled_outputs = bert(inputs=(input_ids, segment_ids, attn_mask))
    embedding = BertPoolerEmbedding(name='embedding')(pooled_outputs)

    model = tf.keras.Model(inputs=[input_ids, segment_ids, attn_mask], outputs=embedding)

    dummy_input_ids, dummy_segment_ids, dummy_attn_mask = dummy_bert_inputs()
    model(inputs=(dummy_input_ids, dummy_segment_ids, dummy_attn_mask), training=False)

    model.summary()
    return model


class BertAvgEmbedding(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, sequence_outputs):
        return tf.reduce_mean(sequence_outputs, axis=1)


def BertAvgEmbeddingModel(pretrained_bert_dir, **kwargs):
    input_ids = tf.keras.layers.Input(shape=[None, ], dtype=tf.int32, name='input_ids')
    segment_ids = tf.keras.layers.Input(shape=[None, ], dtype=tf.int32, name='segment_ids')
    attn_mask = tf.keras.layers.Input(shape=[None, ], dtype=tf.int32, name='attention_mask')
    bert = Bert.from_pretrained(pretrained_model_dir=pretrained_bert_dir)

    sequence_outputs, _ = bert(inputs=(input_ids, segment_ids, attn_mask))
    embedding = BertAvgEmbedding(name='embedding')(sequence_outputs)

    model = tf.keras.Model(inputs=[input_ids, segment_ids, attn_mask], outputs=embedding)

    dummy_input_ids, dummy_segment_ids, dummy_attn_mask = dummy_bert_inputs()
    model(inputs=(dummy_input_ids, dummy_segment_ids, dummy_attn_mask), training=False)

    model.summary()
    return model


class BertFirstLastAvgEmbedding(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, hidden_states):
        first = tf.reduce_mean(hidden_states[:, 0, :, :], axis=1)
        last = tf.reduce_mean(hidden_states[:, -1, :, :], axis=1)
        return tf.reduce_mean(tf.stack([first, last]), axis=0)


def BertFirstLastAvgEmbeddingModel(pretrained_bert_dir, **kwargs):
    input_ids = tf.keras.layers.Input(shape=[None, ], dtype=tf.int32, name='input_ids')
    segment_ids = tf.keras.layers.Input(shape=[None, ], dtype=tf.int32, name='segment_ids')
    attn_mask = tf.keras.layers.Input(shape=[None, ], dtype=tf.int32, name='attention_mask')
    bert = Bert.from_pretrained(pretrained_model_dir=pretrained_bert_dir, return_states=True)

    _, _, hidden_states = bert(inputs=(input_ids, segment_ids, attn_mask))
    embedding = BertFirstLastAvgEmbedding(name='embedding')(hidden_states)

    model = tf.keras.Model(inputs=[input_ids, segment_ids, attn_mask], outputs=embedding)

    dummy_input_ids, dummy_segment_ids, dummy_attn_mask = dummy_bert_inputs()
    model(inputs=(dummy_input_ids, dummy_segment_ids, dummy_attn_mask), training=False)

    model.summary()
    return model


def BertAllInOneEmbeddingModel(pretrained_bert_dir, **kwargs):
    input_ids = tf.keras.layers.Input(shape=[None, ], dtype=tf.int32, name='input_ids')
    segment_ids = tf.keras.layers.Input(shape=[None, ], dtype=tf.int32, name='segment_ids')
    attn_mask = tf.keras.layers.Input(shape=[None, ], dtype=tf.int32, name='attention_mask')
    bert = Bert.from_pretrained(pretrained_model_dir=pretrained_bert_dir, return_states=True)

    sequence_outputs, pooled_outputs, hidden_states = bert(inputs=(input_ids, segment_ids, attn_mask))

    cls_embedding = BertCLSEmbedding(name='cls')(sequence_outputs)
    pooler_embedding = BertPoolerEmbedding(name='pooler')(pooled_outputs)
    avg_embedding = BertAvgEmbedding(name='avg')(sequence_outputs)
    first_last_avg_embedding = BertFirstLastAvgEmbedding(name='first-last-avg')(hidden_states)

    model = tf.keras.Model(
        inputs=[input_ids, segment_ids, attn_mask],
        outputs=[cls_embedding, pooler_embedding, avg_embedding, first_last_avg_embedding])

    dummy_input_ids, dummy_segment_ids, dummy_attn_mask = dummy_bert_inputs()
    model(inputs=(dummy_input_ids, dummy_segment_ids, dummy_attn_mask), training=False)

    model.summary()
    return model
