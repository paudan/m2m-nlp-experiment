#!/usr/bin/python3

__author__ = "Paulius Danenas"
__maintainer__ = "Paulius Danenas"
__email__ = "danpaulius@gmail.com"

import itertools
import random
import logging
from collections import namedtuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torchcrf import CRF
from transformers import BertPreTrainedModel, BertModel

BERT_MODEL = "bert-base-uncased"
BERT_MODEL_DIR = 'embeddings'
module = __import__('transformers')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def set_seed(seed, use_gpu=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_gpu:
        torch.cuda.manual_seed_all(seed)


class BERT_BiLSTM_CRF(BertPreTrainedModel):

    def __init__(self, config, model_config, labels_map=None, use_bilstm=False, rnn_dim=128):
        super(BERT_BiLSTM_CRF, self).__init__(config)
        self.labels_map = labels_map
        if labels_map is None:
            self.labels_map = config.id2label
            if labels_map is None:
                raise ValueError('Labels map is not set')
        if model_config is None or model_config['model'] is None:
            raise ValueError('Model type configuration is not set or is invalid')
        num_labels = len(labels_map)
        model_class_ = getattr(module, model_config['model'])
        self.bert = model_class_(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        out_dim = config.hidden_size
        self.use_bilstm = use_bilstm
        if use_bilstm:
            self.birnn = nn.LSTM(config.hidden_size, rnn_dim, num_layers=1, bidirectional=True, batch_first=True)
            out_dim = rnn_dim*2
        self.hidden2tag = nn.Linear(out_dim, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, tags, token_type_ids=None, input_mask=None):
        emissions = self.tag_outputs(input_ids, token_type_ids, input_mask)
        loss = -1*self.crf(emissions, tags, mask=input_mask.byte())
        return loss

    def tag_outputs(self, input_ids, token_type_ids=None, input_mask=None):
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=input_mask)
        sequence_output = outputs[0]
        if self.use_bilstm:
            sequence_output, _ = self.birnn(sequence_output)
        sequence_output = self.dropout(sequence_output)
        emissions = self.hidden2tag(sequence_output)
        return emissions

    def predict(self, input_ids, token_type_ids=None, input_mask=None):
        emissions = self.tag_outputs(input_ids, token_type_ids, input_mask)
        return self.crf.decode(emissions, input_mask.byte())

    def predict_tags(self, input: TensorDataset):
        results = []
        for _, (input_ids, input_mask, segment_ids) in enumerate(input):
            input_ids = input_ids.unsqueeze(0)
            input_mask = input_mask.unsqueeze(0)
            segment_ids = segment_ids.unsqueeze(0)
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)
            with torch.no_grad():
                logits = self.predict(input_ids, segment_ids, input_mask)
            tags = [[self.labels_map[idx] for idx in l] for l in logits][0]
            results.append(tags[1:-1])  # Strip CLS/SEP symbols
        return results


class BertDataset(object):

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @staticmethod
    def tags_list(labels):
        label_list = set(list(itertools.chain(*labels)))
        label_list.add("''")
        return label_list

    @staticmethod
    def labels_map(tags):
        return {label : i for i, label in enumerate(sorted(tags))}

    def process_example(self, example, labels_map, label=None, max_seq_length=256):
        InputFeatures = namedtuple('InputFeatures', ['input_ids', 'input_mask', 'segment_ids', 'label_id', 'tokens', 'labels'])
        example = list(example)
        if len(example) >= max_seq_length - 1:
            example = example[0:(max_seq_length - 2)]
            if label is not None:
                label = label[0:(max_seq_length - 2)]
        tokens = [self.tokenizer.cls_token] + example + [self.tokenizer.sep_token]
        segment_ids = [0] * len(tokens)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        if label is not None:
            orig_labels = [''] + label + ['']
            label_ids = [labels_map["''"]] + [labels_map[l] for l in label] + [labels_map["''"]]
        else:
            label_ids, orig_labels = None, None
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            if label is not None:
                label_ids.append(0)
        return InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids,
                             label_id=label_ids, tokens=tokens, labels=orig_labels)

    def transform(self, examples, labels, labels_map, max_seq_length=256):
        if labels is None:
            raise ValueError('labels cannot be None')
        features = [self.process_example(example, labels_map, label, max_seq_length) for example, label in zip(examples, labels)]
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        examples = [f.tokens for f in features]
        labels = [f.labels for f in features]
        data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        return examples, labels, data

    def transform_input(self, examples, labels_map, max_seq_length=256):
        features = [self.process_example(example, labels_map, None, max_seq_length) for example in examples]
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        return TensorDataset(all_input_ids, all_input_mask, all_segment_ids)


