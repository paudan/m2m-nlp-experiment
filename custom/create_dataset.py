import os
import glob
import itertools
import pickle
import nltk
from nltk.stem import WordNetLemmatizer

NLTK_PATH = '/mnt/DATA/data/nltk'
ONTONOTES_PATH = '/mnt/DATA/Darbas/KTU/code/OntoNotes-5.0/conll-formatted-ontonotes-5.0'
DATA_TRAIN_PATH = os.path.join(ONTONOTES_PATH, 'data', 'train')
DATA_VALID_PATH = os.path.join(ONTONOTES_PATH, 'data', 'development')
DATA_TEST_PATH = os.path.join(ONTONOTES_PATH, 'data', 'test')

nltk.data.path.append(NLTK_PATH)
lemmatizer = WordNetLemmatizer()
files_train = glob.glob(DATA_TRAIN_PATH + '/**/*.gold_conll', recursive=True)
files_valid = glob.glob(DATA_VALID_PATH + '/**/*.gold_conll', recursive=True)
files_test = glob.glob(DATA_TEST_PATH + '/**/*.gold_conll', recursive=True)

def changed_infinitive(sent):
    verb_tags = ['VBZ', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    return tuple([(lemmatizer.lemmatize(pair[0], pos='v'), 'VB') if pair[1] in verb_tags else pair for pair in sent])

def process_conll_file(file):
    data = list()
    with open(file, "r") as f:
        for line in f:
            if line.startswith('#begin'):
                continue
            elif len(line.strip()) == 0:
                data.append(('', ''))
            else:
                split = line.split()
                if len(split) > 5:
                    data.append((split[3], split[4]))
    size = len(data)
    idx_list = [idx + 1 for idx, val in enumerate(data) if val == ('', '')]
    sentences = [tuple(data[i: j-1]) for i, j in zip([0] + idx_list, idx_list + ([size] if idx_list[-1] != size else []))]
    sentences.extend(list(map(changed_infinitive, sentences)))
    augmented = set(sentences)
    return list(zip(*[tuple(zip(*item)) for item in augmented]))

def create_dataset(files, output_file):
    output = list(map(process_conll_file, files))
    x_train = list(itertools.chain.from_iterable(x[0] for x in output))
    y_train = list(itertools.chain.from_iterable(x[1] for x in output))
    del output
    with open(output_file, 'wb') as f:
        pickle.dump((x_train, y_train), f)


if __name__ == '__main__':
    print('Creating training dataset')
    create_dataset(files_train, 'data_train.pkl')
    print('Creating validation dataset')
    create_dataset(files_valid, 'data_valid.pkl')
    print('Creating testing dataset')
    create_dataset(files_test, 'data_test.pkl')
