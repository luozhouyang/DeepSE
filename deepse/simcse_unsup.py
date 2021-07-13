import tensorflow as tf
from transformers_keras import Bert


def _unpack_data(data):
    x, y, sample_weight = None, None, None
    if len(data) == 3:
        x, y, sample_weight = data[0], data[1], data[2]
    elif len(data) == 2:
        x, y = data[0], data[1]
    elif len(data) == 1:
        x = data
    return x, y, sample_weight


# 参考苏神的实现：https://github.com/bojone/SimCSE/blob/bce7175e9d87f45e6123d77d2080667ffa2915b4/eval.py#L126
def unsup_simcse_loss(y_true, y_pred, sample_weight=None):
    # construct labels
    idxs = tf.range(0, tf.shape(y_pred)[0])
    idxs_1 = idxs[None, :]
    idxs_2 = (idxs + 1 - idxs % 2 * 2)[:, None]
    y_true = tf.equal(idxs_1, idxs_2)
    y_true = tf.cast(y_true, tf.float32)
    # coompute cosine similarity
    y_pred = tf.linalg.l2_normalize(y_pred, axis=1)
    similarities = tf.matmul(y_pred, tf.transpose(y_pred))
    similarities = similarities - tf.eye(tf.shape(y_pred)[0]) * 1e12
    similarities = similarities * 20
    loss = tf.keras.losses.categorical_crossentropy(y_true, similarities, from_logits=True)
    return tf.reduce_mean(loss)


class UnsupSimCSEModel(tf.keras.Model):

    def train_step(self, data):
        x, y, sample_weight = _unpack_data(data)
        with tf.GradientTape() as tape:
            input_ids, segment_ids, attn_mask = x['input_ids'], x['segment_ids'], x['attention_mask']
            a_embedding = self(inputs=[input_ids, input_ids, attn_mask], training=True)
            b_embedding = self(inputs=[input_ids, segment_ids, attn_mask], training=True)
            # b_embedding = a_embedding
            embeddings = tf.stack([a_embedding, b_embedding], axis=1)
            batch_size = tf.shape(embeddings)[0]
            # embeddings
            y_pred = tf.reshape(embeddings, shape=(2 * batch_size, -1))
            # compute loss
            # loss = self.compiled_loss(y, y_pred, sample_weight=sample_weight, regularization_losses=self.losses)
            loss = unsup_simcse_loss(y, y_pred)

        # Compute gradients
        trainable_vars = self.trainable_variables
        # print_variables(trainable_vars)
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        results = {m.name: m.result() for m in self.metrics}
        results.update({'loss': loss})
        return results

    def test_step(self, data):
        x, y, sample_weight = _unpack_data(data)
        input_ids, segment_ids, attn_mask = x['input_ids'], x['segment_ids'], x['attention_mask']
        a_embedding = self(inputs=[input_ids, segment_ids, attn_mask], training=False)
        b_embedding = self(inputs=[input_ids, segment_ids, attn_mask], training=False)
        embeddings = tf.stack([a_embedding, b_embedding], axis=1)
        batch_size = tf.shape(embeddings)[0]
        y_pred = tf.reshape(embeddings, shape=(2 * batch_size, -1))
        # self.compiled_loss(y, y_pred, sample_weight=sample_weight, regularization_losses=self.losses)
        loss = unsup_simcse_loss(y, y_pred)
        self.compiled_metrics.update_state(y, y_pred)
        results = {m.name: m.result() for m in self.metrics}
        results.update({'loss': loss})
        return results

    def dummy_inputs(self):
        input_ids, segment_ids, attn_mask = [0] * 128, [0] * 128, [0] * 128
        input_ids = tf.constant(input_ids, dtype=tf.int64, shape=(1, 128))
        segment_ids = tf.constant(segment_ids, dtype=tf.int64, shape=(1, 128))
        attn_mask = tf.constant(segment_ids, dtype=tf.int64, shape=(1, 128))
        return input_ids, segment_ids, attn_mask


def UnsupSimCSE(pretrained_model_dir, strategy='cls', lr=3e-5, **kwargs):
    input_ids = tf.keras.layers.Input(shape=[None, ], dtype=tf.int32, name='input_ids')
    segment_ids = tf.keras.layers.Input(shape=[None, ], dtype=tf.int32, name='segment_ids')
    attn_mask = tf.keras.layers.Input(shape=[None, ], dtype=tf.int32, name='attention_mask')
    bert = Bert.from_pretrained(
        pretrained_model_dir,
        return_states=True,
        return_attention_weights=False)
    sequence_outputs, pooled_outputs, hidden_states = bert(inputs=[input_ids, segment_ids, attn_mask])

    # take CLS embedding as sentence embedding
    embedding = tf.keras.layers.Lambda(lambda x: x[:, 0], name='embedding')(sequence_outputs)
    model = UnsupSimCSEModel(inputs=[input_ids, segment_ids, attn_mask], outputs=embedding)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
    )
    model.summary()
    return model
