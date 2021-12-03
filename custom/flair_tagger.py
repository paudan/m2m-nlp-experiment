import os
import pickle
import torch
from flair.datasets.sequence_labeling import ColumnCorpus
from flair.embeddings import TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from elmo_embeddings import ELMoEmbeddings

CACHE_DIR = '../embeddings'
OUTPUT_PATH = 'datasets'
torch.set_default_tensor_type(torch.FloatTensor)
# torch.backends.cudnn.enabled = False

config = {
#    'flair-bert-tagger': TransformerWordEmbeddings('bert-base-uncased', cache_dir=CACHE_DIR),
    'flair-elmo-tagger': ELMoEmbeddings(model='medium')
}

def convert_data(input_path: str, output_path: str):
    with open(input_path, 'rb') as f:
        data, tags = pickle.load(f)
    with open(output_path, 'w', encoding='utf-8') as f:
        for sent, tag in zip(data, tags):
            for item in zip(sent, tag):
                f.writelines([item[0], ' ', item[1], '\n'])
            f.write("\n")

def convert_dataset():
    if not os.path.isdir(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)
    convert_data('data_train.pkl', os.path.join(OUTPUT_PATH, 'data_train.txt'))
    convert_data('data_valid.pkl', os.path.join(OUTPUT_PATH, 'data_valid.txt'))
    convert_data('data_test.pkl',  os.path.join(OUTPUT_PATH, 'data_test.txt'))


columns = {0 : 'text', 1 : 'pos'}
corpus = ColumnCorpus(OUTPUT_PATH, columns, train_file = 'data_train.txt',
                      test_file = 'data_test.txt', dev_file = 'data_valid.txt')
for tagger_name, emb in config.items():
    tagger = SequenceTagger(hidden_size=128, embeddings=emb,
            tag_dictionary=corpus.make_tag_dictionary(tag_type='pos'),
            tag_type='pos')
    trainer = ModelTrainer(tagger, corpus)
    trainer.train(tagger_name, learning_rate=0.1, mini_batch_size=8, max_epochs=10, embeddings_storage_mode='gpu')
