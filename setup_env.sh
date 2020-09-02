#!/bin/bash

python3 -m pip install -r requirements.txt
# Spacy setup
mkdir spacy
wget -O - https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-2.3.1/en_core_web_lg-2.3.1.tar.gz | tar xzf - -C spacy
mv spacy/en_core_web_lg-* spacy/en_core_web_lg
mv spacy/en_core_web_lg/en_core_web_lg/en_core_web_lg-* spacy/en_core_web_lg/en_core_web_lg/en_core_web_lg
# Stanford Stanza and CoreNLP
python3 -c "import stanza; stanza.download('en', dir='stanza_resources')"
wget http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip | unzip
wget http://nlp.stanford.edu/software/stanford-english-corenlp-2018-10-05-models.jar -P stanford-corenlp-full-2018-10-05/
# Flair
mkdir flair
mkdir flair/models
wget https://nlp.informatik.hu-berlin.de/resources/models/pos/en-pos-ontonotes-v0.4.pt -P flair/models
wget https://nlp.informatik.hu-berlin.de/resources/models/ner/en-ner-conll03-v0.4.pt -P flair/models
