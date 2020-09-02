import os
import re
from abc import abstractmethod
from collections import Iterable
import itertools
import spacy
import torch
import stanza
from stanza.server import CoreNLPClient
from flair.models import SequenceTagger
from flair.data import Sentence
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus.reader import wordnet
from nltk.stem import WordNetLemmatizer
from simplenlg.features import Feature, Tense
from simplenlg.framework import InflectedWordElement
from simplenlg.lexicon import Lexicon, LexicalCategory
from simplenlg.realiser.english import Realiser
from transformers import AutoTokenizer, AutoModelForTokenClassification, BertConfig
from transformers.pipelines import pipeline

STANZA_DIR = 'stanza_resources'
FLAIR_POS_MODEL = 'flair/models/en-pos-ontonotes-v0.4.pt'
FLAIR_NER_MODEL = 'flair/models/en-ner-conll03-v0.4.pt'
SPACY_MODEL = "spacy/en_core_web_lg/en_core_web_lg/en_core_web_lg"
BERT_POS_MODEL = "vblagoje/bert-english-uncased-finetuned-pos"
BERT_NER_MODEL = 'wietsedv/bert-base-multilingual-cased-finetuned-conll2002-ner'
BERT_MODEL_DIR = 'embeddings'
nltk.data.path.append('/mnt/DATA/data/nltk')


class AbstractNLPProcessor:
    
    def __init__(self, process_proper_nouns=False):
        self.lexicon = Lexicon.getDefaultLexicon()
        self.realiser = Realiser(self.lexicon)
        self.process_proper_nouns = process_proper_nouns

    def grammar(self):
        NP = '<ADV|ADJ>*<NOUN|PROPN>+<PART>?<NUM>?'
        if self.process_proper_nouns is True:
            return """
            NP: {{(<ADV|ADJ>*<NOUN>+<PART>?<NUM>?)+(<ADP>*<DET>?{NP})*}}
            VP: {{<VERB>+<ADP>?}}
            PNP: {{<PROPN>+}}
            """.format(NP=NP)
        else:
            return """
            NP: {{({NP})+(<ADP>*<DET>?{NP})*}}
            VP: {{<VERB>+<ADP>?}}
            PNP: {{<PROPN>+}}
            """.format(NP=NP)

    @abstractmethod
    def extract_named_entities(self, token):
        pass

    def get_named_entity(self, token, index = 0):
        if token is None:
            return token
        entities = self.extract_named_entities(token)
        if entities is None or len(entities) == 0 or len(entities) + 1 < index: return None
        return entities[index]

    @abstractmethod
    def get_named_entity_types(self, token):
        pass

    def get_named_entity_type(self, token, index=0):
        if token is None:
            return token
        types = self.get_named_entity_types(token)
        if types is None or len(types) == 0 or len(types) + 1 < index: return None
        return types[index]

    def _extract_phrase(self, tagged, chunk_label):
        cp = nltk.RegexpParser(self.grammar())
        tree = cp.parse(tagged)
        return [' '.join(s for s, t in subtree).replace(" '", "'") for subtree in tree.subtrees() if subtree.label() == chunk_label]

    @abstractmethod
    def extract_phrase_by_type(self, token, type):
        pass

    def extract_noun_phrases(self, token):
        if token is None:
            return None
        return self.extract_phrase_by_type(token, "NP")

    def extract_proper_nouns(self, token):
        if token is None:
            return None
        return self.extract_phrase_by_type(token, "PNP")

    def extract_verb_phrases(self, token):
        if token is None:
            return None
        return self.extract_phrase_by_type(token, "VP")

    def extract_verb_phrase(self, token):
        verbs = self.extract_verb_phrases(token)
        if verbs is not None and len(verbs) > 0:
            return verbs[0]
        return None

    def normalize_verb(self, verb):
        if verb is None:
            return verb

        def normalize(vb):
            word = self.lexicon.getWord(self.lemma(vb, pos=wordnet.VERB), LexicalCategory.VERB)
            infl = InflectedWordElement(word)
            infl.setFeature(Feature.TENSE, Tense.PRESENT)
            return self.realiser.realise(infl).getRealisation()

        return ' '.join(normalize(t) if ind == 0 else t for ind, t in enumerate(verb.split()))

    def lemma(self, token, pos=wordnet.NOUN):
        if token is None:
            return token
        lemmatizer = WordNetLemmatizer()
        return lemmatizer.lemmatize(token, pos)

    def is_synonym(self, first, second):
        synonyms = [[l.name() for l in sn.lemmas()] for sn in wn.synsets(second.replace(' ', '_'), 'n')]
        synonyms = list(itertools.chain.from_iterable(synonyms))
        synonyms = [s for s in synonyms if s != second]
        synonyms = [s.replace('_', ' ') for s in synonyms]
        return first in synonyms

    def is_meronym(self, first, second):
        meronyms = [list(itertools.chain.from_iterable([s.lemma_names('eng') for s in sn.part_meronyms()]))
                     for sn in wn.synsets(second.replace(' ', '_'), 'n')]
        meronyms = list(itertools.chain.from_iterable(meronyms))
        meronyms = [s.replace('_', ' ') for s in meronyms]
        return first in meronyms

    def is_holonym(self, first, second):
        holonyms = [list(itertools.chain.from_iterable([s.lemma_names('eng') for s in sn.part_holonyms()]))
                     for sn in wn.synsets(second.replace(' ', '_'), 'n')]
        holonyms = list(itertools.chain.from_iterable(holonyms))
        holonyms = [s.replace('_', ' ') for s in holonyms]
        return first in holonyms

    def is_hyponym(self, first, second):
        hyponyms = [list(itertools.chain.from_iterable([s.lemma_names('eng') for s in sn.hyponyms()]))
                     for sn in wn.synsets(second.replace(' ', '_'), 'n')]
        hyponyms = list(itertools.chain.from_iterable(hyponyms))
        hyponyms = [s.replace('_', ' ') for s in hyponyms]
        return first in hyponyms

    def is_hypernym(self, first, second, full_hierarchy=False):
        if full_hierarchy is False:
            hypernyms = [list(itertools.chain.from_iterable([s.lemma_names('eng') for s in sn.hypernyms()]))
                         for sn in wn.synsets(second.replace(' ', '_'), 'n')]
            hypernyms = list(itertools.chain.from_iterable(hypernyms))
        else:
            hypernyms = self._hypernyms_full(second)
        hypernyms = set(hypernyms)
        hypernyms = [s.replace('_', ' ') for s in hypernyms]
        return first in hypernyms

    def _flatten(self, list_):
        for el in list_:
            if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
                yield from self._flatten(el)
            else:
                yield el

    def _hypernyms_full(self, word):
        # Collect full hierarchy of possible hypernyms
        def collect_hypernyms(sn):

            def add_hypernyms(hypernyms, synsets):
                if synsets is None:
                    return hypernyms
                hypernyms.extend([s.lemma_names('eng') for s in synsets])
                for p in synsets:
                    add_hypernyms(hypernyms, p.hypernyms())
                return hypernyms

            hyplist = list()
            return add_hypernyms(hyplist, sn)

        hypernyms = list(map(lambda x: collect_hypernyms(x.hypernyms()), wn.synsets(word.replace(' ', '_'), 'n')))
        return list(self._flatten(hypernyms))


class SpacyNLPProcessor(AbstractNLPProcessor):

    def __init__(self, process_proper_nouns=False):
        super().__init__(process_proper_nouns)
        spacy.prefer_gpu()
        self.tagger = spacy.load(SPACY_MODEL)

    def extract_named_entities(self, token):
        doc = self.tagger(token)
        return [ent.text for ent in doc.ents]

    def get_named_entity_types(self, token):
        doc = self.tagger(token)
        label_mapping = { 'GPE': 'LOCATION', 'ORG': 'ORGANIZATION', 'MONEY': 'MONEY'}
        return [label_mapping.get(entity.label_)  or entity.label_ for entity in doc.ents]

    def extract_phrase_by_type(self, token, type):
        doc = self.tagger(token)
        tagged = [(token.text, token.pos_) for token in doc]
        return self._extract_phrase(tagged, type)


class StanzaNLPProcessor(AbstractNLPProcessor):

    def __init__(self, process_proper_nouns=False):
        super().__init__(process_proper_nouns)
        torch.set_default_tensor_type(torch.FloatTensor)
        stanza.download('en', dir=STANZA_DIR)
        self.tagger = stanza.Pipeline('en', dir=STANZA_DIR)

    def extract_named_entities(self, token):
        doc = self.tagger(token)
        return [ent.text for ent in doc.entities]

    def get_named_entity_types(self, token):
        doc = self.tagger(token)
        label_mapping = { 'GPE': 'LOCATION', 'ORG': 'ORGANIZATION'}
        return [label_mapping.get(entity.type) or entity.type for entity in doc.entities]

    def extract_phrase_by_type(self, token, type):
        doc = self.tagger(token)
        tagged = [(word.text, word.upos) for sent in doc.sentences for word in sent.words]
        return self._extract_phrase(tagged, type)


class FlairNLPProcessor(AbstractNLPProcessor):

    def __init__(self, process_proper_nouns=False, ner_model=FLAIR_NER_MODEL, pos_model=FLAIR_POS_MODEL):
        super().__init__(process_proper_nouns)
        torch.set_default_tensor_type(torch.FloatTensor)
        self.tagger = SequenceTagger.load(ner_model)
        self.pos_tagger = SequenceTagger.load(pos_model)

    def extract_named_entities(self, token):
        sentence = Sentence(token)
        self.tagger.predict(sentence)
        return [' '.join([t.text for t in entity.tokens]) for entity in sentence.get_spans('ner')]

    def get_named_entity_types(self, token):
        sentence = Sentence(token)
        self.tagger.predict(sentence)
        entities = sentence.get_spans('ner')
        label_mapping = { 'LOC': 'LOCATION', 'ORG': 'ORGANIZATION', 'PER': 'PERSON'}
        return [label_mapping.get(entity.tag) or entity.tag for entity in entities]

    def extract_phrase_by_type(self, token, type):
        sentence = Sentence(token)
        self.pos_tagger.predict(sentence)
        tagged = [(token.text, token.get_tag('pos').value) for token in sentence.tokens]
        return self._extract_phrase(tagged, type)


class CoreNLPProcessor(AbstractNLPProcessor):

    def grammar(self):
        ADP = '<RB|RBR|RP|TO|IN|PREP>'
        NP = '<JJ|ADJ>*<NN|VBG|RBS|FW|NNS>+<POS>?<CD>?'
        return """
        NP: {{({NP})+({ADP}?<DT>?{NP})*}}
        VP: {{<VB*>+{ADP}?}}
        PNP: {{<NNP|NNPS>+}}        
        """.format(NP=NP, ADP=ADP)

    def __init__(self):
        super().__init__()
        os.environ["CORENLP_HOME"] = os.path.join(os.getcwd(), 'stanford-corenlp-full-2018-10-05')
        self.tagger = CoreNLPClient(annotators=['tokenize', 'pos', 'ner'], timeout=30000, memory='4G')

    def __del__(self):
        self.tagger.stop()

    def _extract_ner(self, token):
        ann = self.tagger.annotate(token)
        sentence = ann.sentence[0]
        return [(n.entityMentionText, n.entityType) for n in sentence.mentions]

    def extract_named_entities(self, token):
        entities = self._extract_ner(token)
        entities = list(set(map(lambda x: x[0], entities)))
        return entities

    def get_named_entity_types(self, token):
        return [entity[1] for entity in self._extract_ner(token)]

    def extract_phrase_by_type(self, token, type):
        ann = self.tagger.annotate(token)
        sentence = ann.sentence[0]
        tagged = [(token.word, token.pos) for token in sentence.token]
        return self._extract_phrase(tagged, type)


class BertNLPProcessor(AbstractNLPProcessor):

    def __init__(self, process_proper_nouns=False):
        super().__init__(process_proper_nouns)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pos_tokenizer = AutoTokenizer.from_pretrained(BERT_POS_MODEL, cache_dir=BERT_MODEL_DIR)
        self.pos_model = AutoModelForTokenClassification.from_pretrained(BERT_POS_MODEL, cache_dir=BERT_MODEL_DIR).to(self.device)
        self.pos_config = BertConfig.from_pretrained(BERT_POS_MODEL, cache_dir=BERT_MODEL_DIR)
        self.ner_tokenizer = AutoTokenizer.from_pretrained(BERT_NER_MODEL, cache_dir=BERT_MODEL_DIR)
        self.ner_model = AutoModelForTokenClassification.from_pretrained(BERT_NER_MODEL, cache_dir=BERT_MODEL_DIR).to(self.device)
        self.ner_config = BertConfig.from_pretrained(BERT_NER_MODEL, cache_dir=BERT_MODEL_DIR)

    def _run_bert_model(self, text, tokenizer, model, config):
        if text is None or not isinstance(text, str) or len(text.strip()) == 0:
            return None
        tokens = tokenizer.tokenize(text)
        encoded_sample = tokenizer.encode_plus(text, add_special_tokens=True, return_token_type_ids=True,
                                               return_attention_mask=True, return_tensors='pt')
        input_ids = encoded_sample['input_ids'].to(self.device)
        attention_mask = encoded_sample['attention_mask'].to(self.device)
        output = model(input_ids, attention_mask)
        _, prediction = torch.max(output[0], dim=2)
        labels = [config.id2label[label] for label in prediction[0].cpu().numpy()]
        labels = labels[1:-1]  # Remove special tokens
        return list(zip(tokens, labels))

    def _extract_ner(self, token):
        if token is None or not isinstance(token, str) or len(token.strip()) == 0:
            return None
        nlp = pipeline('ner', model=self.ner_model, config=self.ner_config, tokenizer=self.ner_tokenizer,
                       grouped_entities=True, device=0 if torch.cuda.is_available() else -1)
        results = nlp(token)
        label_mapping = { 'B-loc': 'LOCATION', 'B-org': 'ORGANIZATION', 'B-per': 'PERSON'}
        return [(entity.get('word'), label_mapping.get(entity.get('entity_group')) or entity.get('entity_group')) for entity in results]

    def extract_named_entities(self, token):
        entities = self._extract_ner(token)
        return list(set(map(lambda x: x[0], entities)))

    def get_named_entity_types(self, token):
        return [entity[1] for entity in self._extract_ner(token)]

    def extract_phrase_by_type(self, token, type):
        tagged = self._run_bert_model(token, self.pos_tokenizer, self.pos_model, self.pos_config)
        return self._extract_phrase(tagged, type)

    def _extract_phrase(self, tagged, chunk_label):
        phrases = super()._extract_phrase(tagged, chunk_label)
        # Remove BERT masking symbols
        return [re.sub(r"\s+#+", '', phrase) for phrase in phrases]
