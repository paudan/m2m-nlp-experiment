import sys
sys.path.append('custom')

import os
import torch
from custom.pytorch_tagger.datasets import ElmoDataset, BertDataset
from custom.pytorch_tagger.utils import load_model
from transformers import AutoTokenizer
from nltk.tokenize import wordpunct_tokenize
from processing import AbstractNLPProcessor
from anago.models import load_model
from anago.preprocessing import IndexTransformer
from anago.tagger import Tagger
from processing import FlairNLPProcessor, FLAIR_NER_MODEL

CACHE_DIR = 'embeddings'
ELMO_TAGGER_PATH = 'custom/elmo_tagger'
BERT_TAGGER_PATH = 'custom/flair-tagger/best-model.pt'


class CustomProcessor(AbstractNLPProcessor):

    def grammar(self):
        ADP = '<RB|RBR|RP|TO|IN|PREP>'
        NP = '<JJ|ADJ>*<NN|VBG|RBS|FW|NNS|PRP|PRP$>+<POS>?<CD>?'
        return """
        NP: {{({NP})+({ADP}?<DT>?{NP})*}}
        VP: {{<VB*>+{ADP}?}}
        PNP: {{<NNP|NNPS>+}}        
        """.format(NP=NP, ADP=ADP)

    def get_named_entity_type(self, token, index=0):
        pass

    def extract_named_entities(self, token):
        pass

    def get_named_entity(self, token, index = 0):
        pass


# ElmoBiLSTMCRF
class ElmoBiLSTM_CRFProcessor(CustomProcessor):

    def __init__(self, process_proper_nouns=False):
        super().__init__(process_proper_nouns)
        model = load_model(os.path.join(ELMO_TAGGER_PATH, 'weights.h5'), os.path.join(ELMO_TAGGER_PATH, 'params.json'))
        it = IndexTransformer.load(os.path.join(ELMO_TAGGER_PATH, 'preprocessor.pkl'))
        self.pos_tagger = Tagger(model, preprocessor=it, tokenizer=wordpunct_tokenize)

    def extract_phrase_by_type(self, token, type):
        return self._extract_phrase(list(zip(self.pos_tagger.tokenizer(token), self.pos_tagger.predict(token))), type)

# BertBiLSTMCRF
class BertBiLSTM_CRFProcessor(FlairNLPProcessor):

    def grammar(self):
        ADP = '<RB|RBR|RP|TO|IN|PREP>'
        NP = '<JJ|ADJ>*<NN|VBG|RBS|FW|NNS|PRP|PRP$>+<POS>?<CD>?'
        return """
        NP: {{({NP})+({ADP}?<DT>?{NP})*}}
        VP: {{<VB*>+{ADP}?}}
        PNP: {{<NNP|NNPS>+}}        
        """.format(NP=NP, ADP=ADP)

    def __init__(self, process_proper_nouns=False):
        super().__init__(process_proper_nouns, FLAIR_NER_MODEL, pos_model=BERT_TAGGER_PATH)


class CustomElmoProcessor(CustomProcessor):

    def __init__(self, model_dir, process_proper_nouns=False):
        super().__init__(process_proper_nouns)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = load_model(model_dir, torch.device(device))
        self.labels_map = self.model.labels_map
        self.tokenizer = wordpunct_tokenize

    def extract_phrase_by_type(self, token, type):
        tokens = self.tokenizer(token)
        embed = ElmoDataset.process_example(token, self.labels_map, pad=False)
        tags = self.model.predict_tags(embed.input_ids.unsqueeze(0))[0]
        return self._extract_phrase(list(zip(tokens, tags)), type)


class CustomBertProcessor(CustomProcessor):

    def __init__(self, model_dir, process_proper_nouns=False):
        super().__init__(process_proper_nouns)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, do_lower_case=False, cache_dir=CACHE_DIR)
        self.model = load_model(model_dir, torch.device(device))
        self.labels_map = self.model.labels_map

    def extract_phrase_by_type(self, token, type):
        tokens = self.tokenizer(token)
        embed = BertDataset.process_example(token, self.labels_map, self.tokenizer)
        tags = self.model.predict_tags((embed.input_ids.unsqueeze(0),
                                        embed.input_mask.unsqueeze(0),
                                        embed.segment_ids.unsqueeze(0)))[0]
        return self._extract_phrase(list(zip(tokens, tags)), type)


# class ElmoBiLSTM_CRFProcessor(CustomElmoProcessor):
#
#     def __init__(self):
#         super(ElmoBiLSTM_CRFProcessor, self).__init__(model_dir='custom/elmo-pos-tagger-lstm')
#
# class ElmoTransformer_CRFProcessor(CustomElmoProcessor):
#
#     def __init__(self):
#         super(ElmoTransformer_CRFProcessor, self).__init__(model_dir='custom/elmo-pos-tagger-transformer')
#
#
# class BertBiLSTM_CRFProcessor(CustomBertProcessor):
#
#     def __init__(self):
#         super(BertBiLSTM_CRFProcessor, self).__init__(model_dir='custom/bert-pos-tagger-lstm')
#
#
# class BertTransformer_CRFProcessor(CustomBertProcessor):
#
#     def __init__(self):
#         super(BertTransformer_CRFProcessor, self).__init__(model_dir='custom/bert-pos-tagger-transformer')
