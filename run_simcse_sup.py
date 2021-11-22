import os

import tensorflow as tf
from transformers_keras import DatapipeForSupervisedSimCSE, SpearmanForSentenceEmbedding, SupervisedSimCSE

MODEL_DIR = "models/atec-supervised"
VOCAB_PATH = os.path.join(os.environ["PRETRAINED_MODEL_PATH"], "vocab.txt")
CONFIG = {
    "model_dir": "models/simcse",
    "train_input_files": [
        "data/atec-train.jsonl",
    ],
    "valid_input_files": [
        "data/atec-dev.jsonl",
    ],
}

model = SupervisedSimCSE.from_pretrained(
    pretrained_model_dir=os.environ["PRETRAINED_MODEL_PATH"],
)
model.summary()
model.compile(
    optimizer=tf.keras.optimizers.Adam(3e-5),
)

# spearman = SpearmanForSentenceEmbedding.from_jsonl_files(
#     input_files=["data/atec-dev.jsonl"],
#     vocab_file=VOCAB_PATH,
#     sentence_a_key="sequence",
#     sentence_b_key="negative_sequence",
# )
train_dataset = DatapipeForSupervisedSimCSE.from_jsonl_files(
    input_files=CONFIG["train_input_files"],
    vocab_file=VOCAB_PATH,
    sequence_key="sequence",
    pos_sequence_key="positive_sequence",
    batch_size=80,
    bucket_boundaries=[25, 50, 75],
    buffer_size=1000000,
    max_sequence_length=100,
    drop_remainder=True,
)
valid_dataset = DatapipeForSupervisedSimCSE.from_jsonl_files(
    input_files=CONFIG["valid_input_files"],
    vocab_file=VOCAB_PATH,
    sequence_key="sequence",
    pos_sequence_key="positive_sequence",
    batch_size=32,
    bucket_boundaries=[25, 50, 75],
    buffer_size=2000,
    max_sequence_length=100,
    drop_remainder=True,
)

for d in train_dataset.take(1):
    print(d)

# 开始训练模型
model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=10,
    callbacks=[
        # 保存ckpt格式的模型
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(os.path.join(MODEL_DIR, "supervised-simcse-{epoch:04d}.ckpt")), save_weights_only=True
        ),
        # 保存SavedModel格式的模型，用于Serving部署
        # tf.keras.callbacks.ModelCheckpoint(
        #     filepath=os.path.join(CONFIG["model_dir"], "export/{epoch}"), save_weights_only=False
        # ),
        # spearman,
    ],
)
