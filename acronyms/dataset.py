#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import itertools
import string
import pandas as pd
import numpy as np
import enchant
import nltk
from nltk.tokenize import wordpunct_tokenize
from tqdm import tqdm
import torch
import stanza

NLTK_PATH = '/mnt/DATA/data/nltk'
STANZA_DIR = os.path.join('..', 'stanza_resources')
nltk.data.path.append(NLTK_PATH)
tqdm.pandas()
torch.set_default_tensor_type(torch.FloatTensor)
stanza.download('en', model_dir=STANZA_DIR)
tagger = stanza.Pipeline('en', dir=STANZA_DIR, processors='tokenize,pos,lemma,depparse', tokenize_pretokenized=True)
wchecker = enchant.Dict("en_US")


def try_tokenize(tok):
    try:
        return wordpunct_tokenize(tok)
    except:
        return []

def abbrev_candidates(items):
    return list(filter(lambda x: 1 < len(x) <= 5, set(itertools.chain(*map(try_tokenize, items)))))

def word2ngrams(text, n=3):
  return [text[i:i+n] for i in range(len(text)-n+1)]

def create_features(candidates, entries):

    def word_features(input_):
        tok, context = input_
        doc = tagger([context])
        tagged = [(word.text, word.upos, word.deprel) for sent in doc.sentences for word in sent.words]
        tok_ind = [i for i, item in enumerate(tagged) if item[0] == tok]
        tok_ind = tok_ind[0]
        return {
            'token': tok,
            'has.vowels': len(set(tok.lower()) & set(tuple('aeiouy'))) > 0,
            'length': len(tok),
            'has.special': len(set(tok) & set(tuple('&.'))) > 0,
            'just.letters': tok.isalpha(),
            'all.upper': tok.isupper(),
            'roman': set('xvi').issuperset(tok.lower()),
            'english.word': wchecker.check(tok.lower()),
            'long.char.seq': any(len(set(a))==1 for a in word2ngrams(tok.lower(), n=3)),
            'starts.with.two': [len(set(a)) for a in word2ngrams(tok.lower(), n=2)][0] == 1,
            'pos': tagged[tok_ind][1],
            'prev.pos': tagged[tok_ind-1][1] if tok_ind > 0 else None,
            'prev.pos2': tagged[tok_ind-2][1] if tok_ind > 1 else None,
            'next.pos': tagged[tok_ind+1][1] if tok_ind < len(tagged)-1 else None,
            'next.pos2': tagged[tok_ind+2][1] if tok_ind < len(tagged)-2 else None,
            'prev.dep': tagged[tok_ind-1][2] if tok_ind > 0 else None,
            'prev.dep2': tagged[tok_ind-2][2] if tok_ind > 1 else None,
            'next.dep': tagged[tok_ind+1][2] if tok_ind < len(tagged)-1 else None,
            'next.dep2': tagged[tok_ind+2][2] if tok_ind < len(tagged)-2 else None
        }

    contexts = [(x,y) for y in entries for x in candidates if x in y]
    return pd.DataFrame(data=list(map(word_features, contexts)))

data = pd.read_csv(os.path.join('..', 'datasets', 'bpmn_dataset_cleaned.csv'), sep=';')
items = data.groupby('Model').agg({'Lane': list, 'Task': list})
items['entries'] = items.apply(lambda row: set(row['Lane'] + row['Task']), axis=1)
items['candidates'] = items['entries'].apply(abbrev_candidates)
items['tokenized'] = items['entries'].apply(lambda row: list(map(try_tokenize, row)))
features = pd.concat(items.progress_apply(lambda row: create_features(row['candidates'], row['tokenized']), axis=1).tolist()).reset_index(drop=True)
features = features[~features['token'].str.match('^\w\d+$')]  # Remove entries like A1, M10, etc.
valid_token = lambda x: len(set(x) & set(string.ascii_letters)) > 1   # At least two letters
features = features[features['token'].apply(valid_token)]
features['label'] = np.where(features['all.upper'] == True, 1, 0)     # Initial labelling, remove from features during classification if not changed!
features = features.drop_duplicates()
features.to_csv(os.path.join('acronyms', 'dataset.csv'), index=False)
