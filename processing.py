import os
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

STANZA_DIR = 'stanza_resources'
FLAIR_POS_MODEL = 'flair/models/en-pos-ontonotes-v0.4.pt'
FLAIR_NER_MODEL = 'flair/models/en-ner-conll03-v0.4.pt'

nltk.data.path.append('/mnt/DATA/data/nltk')


class AbstractNLPProcessor:

    def grammar(self):
        NP = '<ADV|ADJ>*<NOUN|PROPN>+<PART>?<NUM>?'
        return """
        NP: {{(<ADV|ADJ>*<NOUN>+<PART>?<NUM>?)+(<ADP>*<DET>?{NP})*}}
        VP: {{<VERB>+<ADP>?}}
        PNP: {{<PROPN>+}}
        """.format(NP=NP)

    @abstractmethod
    def extract_named_entities(self, token):
        pass

    def get_named_entity(self, token, index = 0):
        entities = self.extract_named_entities(token)
        if entities is None or len(entities) == 0 or len(entities) + 1 < index: return None
        return entities[index]

    @abstractmethod
    def get_named_entity_type(self, token, index=0):
        pass

    def _extract_phrase(self, tagged, chunk_label):
        cp = nltk.RegexpParser(self.grammar())
        tree = cp.parse(tagged)
        return [' '.join(s for s, t in subtree).replace(" '", "'") for subtree in tree.subtrees() if subtree.label() == chunk_label]

    @abstractmethod
    def extract_phrase_by_type(self, token, type):
        pass

    def extract_noun_phrases(self, token):
        return self.extract_phrase_by_type(token, "NP")

    def extract_proper_nouns(self, token):
        return self.extract_phrase_by_type(token, "PNP")

    def extract_verb_phrase(self, token):
        verbs = self.extract_phrase_by_type(token, "VP")
        if len(verbs) > 0:
            return verbs[0]
        return None

    def normalize_verb(self, verb):
        lexicon = Lexicon.getDefaultLexicon()
        realiser = Realiser(lexicon)

        def normalize(vb):
            word = lexicon.getWord(self.lemma(vb, pos=wordnet.VERB), LexicalCategory.VERB)
            infl = InflectedWordElement(word)
            infl.setFeature(Feature.TENSE, Tense.PRESENT)
            return realiser.realise(infl).getRealisation()

        return ' '.join(normalize(t) if ind == 0 else t for ind, t in enumerate(verb.split()))

    def lemma(self, token, pos=wordnet.NOUN):
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

    def __init__(self):
        spacy.prefer_gpu()
        self.tagger = spacy.load("en_core_web_sm")

    def extract_named_entities(self, token):
        doc = self.tagger(token)
        return [ent.text for ent in doc.ents]

    def get_named_entity_type(self, token, index=0):
        doc = self.tagger(token)
        entities = doc.ents
        if entities is None or len(entities) == 0 or len(entities)+1 < index: return None
        label_mapping = { 'GPE': 'LOCATION', 'ORG': 'ORGANIZATION', 'MONEY': 'MONEY'}
        return label_mapping.get(doc.ents[index].label_)

    def extract_phrase_by_type(self, token, type):
        doc = self.tagger(token)
        tagged = [(token.text, token.pos_) for token in doc]
        return self._extract_phrase(tagged, type)


class StanzaNLPProcessor(AbstractNLPProcessor):

    def __init__(self):
        torch.set_default_tensor_type(torch.FloatTensor)
        stanza.download('en', dir=STANZA_DIR)
        self.tagger = stanza.Pipeline('en', dir=STANZA_DIR)

    def extract_named_entities(self, token):
        doc = self.tagger(token)
        return [ent.text for ent in doc.entities]

    def get_named_entity_type(self, token, index=0):
        doc = self.tagger(token)
        entities = doc.entities
        if entities is None or len(entities) == 0 or len(entities)+1 < index: return None
        label_mapping = { 'GPE': 'LOCATION', 'ORG': 'ORGANIZATION'}
        type = entities[index].type
        return label_mapping.get(type) or type

    def extract_phrase_by_type(self, token, type):
        doc = self.tagger(token)
        tagged = [(word.text, word.upos) for sent in doc.sentences for word in sent.words]
        return self._extract_phrase(tagged, type)


class FlairNLPProcessor(AbstractNLPProcessor):

    def __init__(self):
        torch.set_default_tensor_type(torch.FloatTensor)
        self.tagger = SequenceTagger.load(FLAIR_NER_MODEL)
        self.pos_tagger = SequenceTagger.load(FLAIR_POS_MODEL)

    def extract_named_entities(self, token):
        sentence = Sentence(token)
        self.tagger.predict(sentence)
        return [' '.join([t.text for t in entity.tokens]) for entity in sentence.get_spans('ner')]

    def get_named_entity_type(self, token, index=0):
        sentence = Sentence(token)
        self.tagger.predict(sentence)
        entities = sentence.get_spans('ner')
        if entities is None or len(entities) == 0 or len(entities)+1 < index: return None
        label_mapping = { 'LOC': 'LOCATION', 'ORG': 'ORGANIZATION', 'PER': 'PERSON'}
        type = entities[index].tag
        return label_mapping.get(type) or type

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

    def get_named_entity_type(self, token, index=0):
        entities = self._extract_ner(token)
        if entities is None or len(entities) == 0 or len(entities)+1 < index: return None
        return entities[index][1]

    def extract_phrase_by_type(self, token, type):
        ann = self.tagger.annotate(token)
        sentence = ann.sentence[0]
        tagged = [(token.word, token.pos) for token in sentence.token]
        return self._extract_phrase(tagged, type)
