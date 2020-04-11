#!/bin/bash

python3 -m pip install -r requirements.txt
# Spacy setup
python3 -m spacy download en_core_web_sm
# Stanford Stanza and CoreNLP
python3 -c "import stanza; stanza.download('en', dir='stanza_resources')"
wget http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip | unzip
wget http://nlp.stanford.edu/software/stanford-english-corenlp-2018-10-05-models.jar -P stanford-corenlp-full-2018-10-05/
# Flair
mkdir flair
mkdir flair/models
wget https://s3.eu-central-1.amazonaws.com/alan-nlp/resources/models-v0.4/POS-ontonotes--h256-l1-b32-p3-0.5-%2Bglove%2Bnews-forward%2Bnews-backward-normal-locked0.5-word0.05--v0.4_0/en-pos-ontonotes-v0.4.pt -P flair/models
wget https://alan-nlp.s3.eu-central-1.amazonaws.com/resources/models-v0.4/NER-conll03-english/en-ner-conll03-v0.4.pt -P flair/models
