import os
import pickle
import kashgari
from kashgari.tasks.labeling import BiLSTM_CRF_Model
from kashgari.embeddings import BERTEmbedding
from keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint

with open("data_train.pkl", "rb") as f:
    x_train, y_train = pickle.load(f)
with open("data_valid.pkl", "rb") as f:
    x_valid, y_valid = pickle.load(f)
with open("data_test.pkl", "rb") as f:
    x_test, y_test = pickle.load(f)
x_train, y_train = list(map(list, x_train)), list(map(list, y_train))
x_valid, y_valid = list(map(list, x_valid)), list(map(list, y_valid))
x_test, y_test = list(map(list, x_test)), list(map(list, y_test))
# Skip testing for now
x_train, y_train = x_train + x_test, y_train + y_test

model_dir = 'bert_tagger'
log_dir = os.path.join(model_dir, 'logs')
weights_path = os.path.join(log_dir, 'weights.h5')
BERT_PATH = '/mnt/DATA/data/embeddings/uncased_L-12_H-768_A-12'
EARLY_STOP = 10

bert_embed = BERTEmbedding(BERT_PATH, task=kashgari.LABELING)
model = BiLSTM_CRF_Model(bert_embed)
model.fit(x_train, y_train, x_valid, y_valid,
          epochs=10, batch_size=64,
          callbacks=[
              TensorBoard(log_dir=log_dir, write_graph=False),
              ModelCheckpoint(weights_path, save_weights_only=True),
              ReduceLROnPlateau()])
print('Saving the model...')
model.save(model_dir)

from kashgari.utils import load_model