import tensorflow as tf
from transformers_keras import Bert


class SimCSEModel(tf.keras.Model):

    def _compute_constrasive_loss(self, embedding, pos_embedding, labels):
        norm_embedding = tf.linalg.normalize(embedding, axis=-1)[0]
        norm_pos_embedding = tf.linalg.normalize(pos_embedding, axis=-1)[0]
        cosine = tf.matmul(norm_embedding, norm_pos_embedding, transpose_b=True)
        loss = tf.keras.losses.sparse_categorical_crossentropy(labels, cosine, from_logits=True)
        return loss, cosine

    def train_step(self, data):
        x, _ = data
        input_ids, segment_ids, attention_mask = x['input_ids'], x['segment_ids'], x['attention_mask']
        pos_input_ids, pos_segment_ids, pos_attention_mask = x['pos_input_ids'], x['pos_segment_ids'], x['pos_attention_mask']
        with tf.GradientTape() as tape:
            embedding = self(inputs=[input_ids, segment_ids, attention_mask], training=True)
            pos_embedding = self(inputs=[pos_input_ids, pos_segment_ids, pos_attention_mask], training=True)
            y_true = tf.range(0, tf.shape(input_ids)[0])
            loss, y_pred = self._compute_constrasive_loss(embedding, pos_embedding, y_true)

        # Compute gradients
        trainable_vars = self.trainable_variables
        # print_variables(trainable_vars)
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y_true, y_pred)
        # Return a dict mapping metric names to current value
        results = {m.name: m.result() for m in self.metrics}
        results.update({'loss': loss})
        return results

    def test_step(self, data):
        x, _ = data
        input_ids, segment_ids, attention_mask = x['input_ids'], x['segment_ids'], x['attention_mask']
        pos_input_ids, pos_segment_ids, pos_attention_mask = x['pos_input_ids'], x['pos_segment_ids'], x['pos_attention_mask']
        embedding = self(inputs=[input_ids, segment_ids, attention_mask], training=False)
        pos_embedding = self(inputs=[pos_input_ids, pos_segment_ids, pos_attention_mask], training=False)
        y_true = tf.range(0, tf.shape(input_ids)[0])
        loss, y_pred = self._compute_constrasive_loss(embedding, pos_embedding, y_true)
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y_true, y_pred)
        # Return a dict mapping metric names to current value
        results = {m.name: m.result() for m in self.metrics}
        results.update({'loss': loss})
        return results


def SimCSE(pretrained_model_dir=None, pretrained_model_config=None, hidden_size=768, lr=3e-5, **kwargs):
    input_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='input_ids')
    segment_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='segment_ids')
    attention_mask = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='attention_mask')

    if pretrained_model_config:
        bert = Bert(**pretrained_model_config)
    elif pretrained_model_dir:
        bert = Bert.from_pretrained(pretrained_model_dir, **kwargs)
    else:
        raise ValueError('pretrained_model_dir or pretrained_model_config must be provided!')

    _, pooled_output = bert(inputs=[input_ids, segment_ids, attention_mask])
    embedding = tf.keras.activations.tanh(tf.keras.layers.Dense(hidden_size)(pooled_output))
    embedding = tf.keras.layers.Lambda(lambda x: x, name='embedding')(embedding)

    model = SimCSEModel(inputs=[input_ids, segment_ids, attention_mask], outputs=[embedding])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=lr)
    )
    model.summary()
    return model
