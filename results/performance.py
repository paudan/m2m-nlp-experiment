#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import itertools
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score


def equal_verbs(verb1, verb2):
    if pd.isnull(verb1) & pd.isnull(verb2): return True
    if (~pd.isnull(verb1) and pd.isnull(verb2)) or (pd.isnull(verb1) and ~pd.isnull(verb2)): return False
    return verb1.lower() == verb2.lower()

def equal_outputs(out1, out2):
    if pd.isnull(out1) & pd.isnull(out2): return True
    if (~pd.isnull(out1) & pd.isnull(out2)) | (pd.isnull(out1) & ~pd.isnull(out2)): return False
    out1 = out1.split('|')
    out2 = out2.split('|')
    return sorted(list(map(lambda x: x.lower(), out1))) == sorted(list(map(lambda x: x.lower(), out2)))

def f1_score_(precision, recall):
    return 2*precision*recall/(precision+recall)

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
        # Metrics depicting if verb presence was successfully detected
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

def calculate_ner_performance(file, original):
    ner_pred = pd.read_csv(file, sep=';')
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

