#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from py_stringmatching.similarity_measure.dice import Dice
from py_stringmatching.similarity_measure.jaccard import Jaccard

logging.basicConfig(level=logging.INFO)


def equal_verbs(verb1, verb2):
    if pd.isnull(verb1) & pd.isnull(verb2): return True
    if (~pd.isnull(verb1) and pd.isnull(verb2)) or (pd.isnull(verb1) and ~pd.isnull(verb2)): return False
    return verb1.lower() == verb2.lower()

def equal_outputs(out1, out2):
    if pd.isnull(out1) & pd.isnull(out2): return True
    if (~pd.isnull(out1) & pd.isnull(out2)) | (pd.isnull(out1) & ~pd.isnull(out2)): return False
    out1 = out1.replace(';', '|').split('|')
    out2 = out2.replace(';', '|').split('|')
    return sorted(list(map(lambda x: x.lower(), out1))) == sorted(list(map(lambda x: x.lower(), out2)))

def f1_score_(precision, recall):
    try:
        result = 2*precision*recall/(precision+recall)
    except ZeroDivisionError:
        result = np.nan
    return result

def single_verb(x):
    if pd.isnull(x):
        return True
    return len(list(itertools.chain([v.split('|') for v in x.split(';')]))) == 1


def calculate_extraction_performance(file, original):
    ind_original = original['Verb phrases'].apply(single_verb)
    original = original[ind_original]
    ind_original_vp = ~pd.isnull(original['Verb phrases'])
    original['VerbRequired'] = ind_original_vp.astype(int)
    original['NounRequired'] = (~pd.isnull(original['Noun phrases'])).astype(int)

    df_pred = pd.read_csv(file, sep=';')
    df_pred = df_pred[ind_original]
    df_pred['VerbRequired'] = (~pd.isnull(df_pred['VerbPhrases'])).astype(int)
    df_pred['NounRequired'] = (~pd.isnull(df_pred['NounPhrases'])).astype(int)    
    # Extract matching statistics for cases when transformation output will be noun
    # No need to check for verb extraction statistics
    matches_outputn = pd.Series(map(lambda x, y: equal_outputs(x, y), original[~ind_original_vp]['Noun phrases'], df_pred[~ind_original_vp]['NounPhrases']), 
                                index=df_pred[~ind_original_vp].index)
    # Extract matching statistics for cases when transformation output will be as "verb-noun" type (e.g. association-class)
    matches_outputv_np = pd.Series(map(lambda x, y: equal_outputs(x, y), original[ind_original_vp]['Noun phrases'], df_pred[ind_original_vp]['NounPhrases']),
                                   index=df_pred[ind_original_vp].index)
    matches_outputv_vp = pd.Series(map(lambda x, y: equal_verbs(x, y), original[ind_original_vp]['Verb phrases'], df_pred[ind_original_vp]['VerbPhrases']),
                                   index=df_pred[ind_original_vp].index)
    matches_outputn_nenp = matches_outputn[~pd.isnull(df_pred[~ind_original_vp]['NounPhrases'])]
    prec_outputn = sum(matches_outputn_nenp)/len(matches_outputn_nenp)
    matches_outputv_ne = ~pd.isnull(df_pred[ind_original_vp]['NounPhrases'])
    matches_outputv_nenp = matches_outputv_np[matches_outputv_ne]
    prec_outputv_np = sum(matches_outputv_nenp)/len(matches_outputv_nenp)
    matches_outputv_nevp = matches_outputv_vp[matches_outputv_ne]
    prec_outputv_vp = sum(matches_outputv_nevp)/len(matches_outputv_nevp)  
    recall_outputn = sum(matches_outputn)/len(matches_outputn)
    recall_outputv_np = sum(matches_outputv_np)/len(matches_outputv_np)
    recall_outputv_vp = sum(matches_outputv_vp)/len(matches_outputv_vp)
    return {
        'Extractor': file.split('-')[0],
        # Metrics depicting if verb presence was successfully detected
        'AccuracyVerbRequired': accuracy_score(original['VerbRequired'], df_pred['VerbRequired']),
        'F1ScoreVerbRequired': f1_score(original['VerbRequired'], df_pred['VerbRequired']),
        # Metrics depicting if noun presence was successfully detected
        'AccuracyNounRequired': accuracy_score(original['NounRequired'], df_pred['NounRequired']),
        'F1ScoreNounRequired': f1_score(original['NounRequired'], df_pred['NounRequired']),
        # Metrics depicting performance when transformation output will use only noun phrases
        'PrecisionOutputNoun': prec_outputn,
        'RecallOutputNoun': recall_outputn,
        'F1ScoreOutputNoun': f1_score_(prec_outputn, recall_outputn), 
        # Metrics depicting performance when transformation output will use both noun and verb phrases
        'PrecisionOutputBoth_Nouns': prec_outputv_np,
        'RecallOutputBoth_Nouns': recall_outputv_np,
        'F1ScoreOutputBoth_Nouns': f1_score_(prec_outputv_np, recall_outputv_np),
        'PrecisionOutputBoth_Verbs': prec_outputv_vp,
        'RecallOutputBoth_Verbs': recall_outputv_vp,
        'F1ScoreOutputBoth_Verbs': f1_score_(prec_outputv_vp, recall_outputv_vp)        
    }


def remove_other_ner(x):
    if pd.isnull(x['Entities']) or pd.isnull(x['EntityType']):
        return x
    entities = x['Entities'].split('|')
    types = x['EntityType'].split('|')
    entities = [x for ind, x in enumerate(entities) if types[ind] in ['PERSON', 'ORGANIZATION', 'LOCATION']]
    types = [x for x in types if x in ['PERSON', 'ORGANIZATION', 'LOCATION']]
    x['Entities'] = '|'.join(entities)
    x['EntityType'] = '|'.join(types)
    if len(x['Entities']) == 0: x['Entities'] = None
    if len(x['EntityType']) == 0: x['EntityType'] = None
    return x

def calculate_ner_performance(file, original):
    ner_pred = pd.read_csv(file, sep=';')
    ner_pred = ner_pred.apply(remove_other_ner, axis=1)
    ind_has_entity = ~pd.isnull(original['Entities'])
    original['HasEntity'] = ind_has_entity.astype(int)
    ner_pred['HasEntity'] = (~pd.isnull(ner_pred['Entities'])).astype(int)
    ner_matches = pd.Series(map(lambda x, y: equal_outputs(x, y), original[ind_has_entity]['Entities'], ner_pred[ind_has_entity]['Entities']),
                            index=ner_pred[ind_has_entity].index)
    stanza_matches_ne_token = ner_matches[~pd.isnull(ner_pred[ind_has_entity]['Entities'])]
    prec_ner_token = sum(stanza_matches_ne_token)/len(stanza_matches_ne_token)
    recall_ner_token = sum(ner_matches)/len(ner_matches)
    ner_matches = pd.Series(map(lambda x, y: equal_outputs(x, y), original[ind_has_entity]['EntityType'], ner_pred[ind_has_entity]['EntityType']),
                            index=ner_pred[ind_has_entity].index)
    stanza_matches_ne_token = ner_matches[~pd.isnull(ner_pred[ind_has_entity]['EntityType'])]
    prec_ner_type = sum(stanza_matches_ne_token)/len(stanza_matches_ne_token)
    recall_ner_type = sum(ner_matches)/len(ner_matches)    
    return {
        'Extractor': file.split('-')[0],
        # Metrics depicting if named entity presence was successfully detected
        'AccuracyNEFound': accuracy_score(original['HasEntity'], ner_pred['HasEntity']),
        'F1ScoreNEFound': f1_score(original['HasEntity'], ner_pred['HasEntity']),
        # Named entity detection
        'PrecisionNEEntry': prec_ner_token,
        'RecallNEEntry': recall_ner_token,
        'F1ScoreNEEntry': f1_score_(prec_ner_token, recall_ner_token), 
        # Named entity type detection
        'PrecisionNEType': prec_ner_type,
        'RecallNEType': recall_ner_type,
        'F1ScoreNEType': f1_score_(prec_ner_type, recall_ner_type) 
    }

def create_phrase_pairs(data_df, verb_col='Verb', noun_col='NounPhrases'):

    def create_pairs(ind, vph, nph):
        if pd.isnull(vph) and pd.isnull(nph):
            logging.debug(f'Empty record at index {ind}')
            return None
        split = lambda x, sep: x.split(sep) if pd.isnull(x) is not True else [None]
        verbs = split(vph, ';')
        nouns = split(nph, ';')
        # No verbs or nouns are available as inputs
        if len(verbs) == 1 and verbs[0] is None:
            verbs = [None] * len(nouns)
        if len(nouns) == 1 and nouns[0] is None:
            nouns = [None] * len(verbs)
        if len(verbs) != len(nouns):
            logging.debug(f'Invalid entry at index {ind}')
            return None
        results = map(lambda _: list(itertools.product(split(_[0], '|'), split(_[1], '|'))), zip(verbs, nouns))
        return list(itertools.chain(*results))

    return list(map(lambda x: create_pairs(x[0], x[1][verb_col], x[1][noun_col]), data_df.iterrows()))


def conjunctives_extraction_performance(standard, extracted, src_column='Task', verb_col='VerbPhrases', noun_col='NounPhrases'):
    original = create_phrase_pairs(standard)
    outputs = create_phrase_pairs(extracted, verb_col=verb_col, noun_col=noun_col)
    dice = Dice()
    jaccard = Jaccard()
    results = pd.DataFrame(data = {
        'input': standard[src_column],
        'total_standard': list(map(len, original)),
        'total_extracted': list(map(lambda x: len(x) if x is not None else 0, outputs)),
        'matches': list(map(lambda x, y: len(set.intersection(set(x), set(y))) if y is not None else 0, original, outputs)),
        'full_match': list(map(lambda x, y: x == y, original, outputs)),
        'dice': list(map(lambda x, y: dice.get_raw_score(x, y) if y is not None else 0, original, outputs)),
        'jaccard': list(map(lambda x, y: jaccard.get_raw_score(x, y) if y is not None else 0, original, outputs))
    })
    precision = results['matches'].sum() / results['total_extracted'].sum()
    recall = results['matches'].sum() / results['total_standard'].sum()
    return {
        'Accuracy': sum(results['full_match']) / results.shape[0],
        'Precision': precision,
        'Recall': recall,
        'F1-Score': (2 * precision * recall)/ (precision + recall),
        'MeanDeviation': np.mean(np.abs(results['total_standard'] - results['matches'])/results['total_standard']),
        'MedianDeviation': np.median(np.abs(results['total_standard'] - results['matches'])/results['total_standard']),
        'MeanDice': results['dice'].mean(),
        'MedianDice': results['dice'].median(),
        'MeanJaccard': results['jaccard'].mean(),
        'MedianJaccard': results['jaccard'].median()
    }
