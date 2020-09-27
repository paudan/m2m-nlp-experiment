import os
import pickle
import numpy as np
from anago.utils import load_glove, filter_embeddings
from anago.models import ELModel, save_model
from anago.trainer import Trainer
from anago.preprocessing import ELMoTransformer
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

EMBEDDING_PATH = '/mnt/DATA/data/embeddings/glove.6B.100d.txt'
EMBEDDING_DIM = 100
EARLY_STOP=10
log_dir = 'elmo_tagger'
weights_path = os.path.join(log_dir, 'weights.h5')

with open("data_train.pkl", "rb") as f:
    x_train, y_train = pickle.load(f)
with open("data_valid.pkl", "rb") as f:
    x_valid, y_valid = pickle.load(f)
with open("data_test.pkl", "rb") as f:
    x_test, y_test = pickle.load(f)

x_train = np.r_[x_train, x_valid]
y_train = np.r_[y_train, y_valid]

print('Transforming datasets...')
p = ELMoTransformer()
p.fit(x_train, y_train)

print('Loading word embeddings...')
embeddings = load_glove(EMBEDDING_PATH)
embeddings = filter_embeddings(embeddings, p._word_vocab.vocab, EMBEDDING_DIM)

print('Building a model.')
model = ELModel(char_embedding_dim=32,
                word_embedding_dim=EMBEDDING_DIM,
                char_lstm_size=32,
                word_lstm_size=EMBEDDING_DIM,
                char_vocab_size=p.char_vocab_size,
                word_vocab_size=p.word_vocab_size,
                num_labels=p.label_size,
                embeddings=embeddings)
model, loss = model.build()
model.compile(loss=loss, optimizer='adam')

print('Training the model...')
trainer = Trainer(model, preprocessor=p)
trainer.train(x_train, y_train, x_test, y_test,
              callbacks=[
                  TensorBoard(log_dir=log_dir, write_graph=False),
                  ModelCheckpoint(weights_path, save_weights_only=True),
                  ReduceLROnPlateau(),
                  EarlyStopping(patience=EARLY_STOP)]
              )

print('Saving the model...')
save_model(model, 'weights.h5', 'params.json')
p.save('preprocessor.pkl')
# model.save('weights.h5', 'params.json')