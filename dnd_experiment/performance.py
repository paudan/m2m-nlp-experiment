#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import warnings
import pandas as pd
import numpy as np
from py_stringmatching.similarity_measure.jaccard import Jaccard
from py_stringmatching.similarity_measure.overlap_coefficient import OverlapCoefficient

warnings.simplefilter(action='ignore', category=FutureWarning)
EXPERIMENT_DIR = '.'

# Results table

def process_results(file, dragged_element_col):
    results = pd.read_csv(os.path.join(EXPERIMENT_DIR, file), sep=';')
    element_count = lambda x: len(re.split(';|', x)) if isinstance(x, str) else 0
    results['ExpectedOutput'] = results[dragged_element_col].apply(element_count) + results['Verb Phrases'].apply(element_count) + \
        results['Noun Phrases'].apply(element_count)
    results['ActualOutput'] = results[dragged_element_col].apply(element_count) + results['VerbPhrases'].apply(element_count) + \
        results['NounPhrases'].apply(element_count)
    results['OutputDiff'] = np.abs(results['ExpectedOutput'] - results['ActualOutput'])
    return results


def df_results_stats(results_df, model_col, dragged_element_col):

    def generate_elements(xv, xn):
        return ([('Association', x.lower().strip()) for x in xv.split('|')] if isinstance(xv, str) else []) \
            + ([('Class', x.lower().strip()) for x in xn.split('|')] if isinstance(xn, str) else [])

    def evaluation_scores(x):
        elements_generated = list(map(generate_elements, x['VerbPhrases'], x['NounPhrases']))
        elements_truth = list(map(generate_elements, x['Verb Phrases'], x['Noun Phrases']))
        elements_eq = list(map(lambda x, y: sorted(x) == sorted(y), elements_generated, elements_truth))
        jaccard = Jaccard()
        jaccard = list(map(jaccard.get_raw_score, elements_generated, elements_truth))
        overlap = OverlapCoefficient()
        overlap = list(map(overlap.get_raw_score, elements_generated, elements_truth))
        return sum(elements_eq)/len(elements_eq), np.mean(jaccard), np.mean(overlap)

    manual_stats = results_df.groupby(model_col).agg({'ExpectedOutput': 'sum', 'ActualOutput': 'sum', 'OutputDiff': 'mean'})
    manual_stats.columns = ['ExpectedElements', 'OutputElements', 'MeanDiff']
    measures_stats = results_df.groupby(model_col).apply(evaluation_scores)
    measures_stats = pd.DataFrame([[*a] for a in measures_stats.values], index=measures_stats.index,
                                  columns=['Accuracy', 'MeanJaccard', 'MeanOverlap'])
    return manual_stats.join(measures_stats)


def create_stats(manual_df, nlp_df, model_col, dragged_element_col):
    stats = pd.concat([manual_df[[model_col, dragged_element_col]].drop_duplicates().groupby([model_col])[model_col].count(),
                       manual_df.groupby([model_col])[model_col].count()], axis=1)
    stats.columns = ['Num Transformations', 'Atomic Transformations']
    manual_stats = df_results_stats(manual_df, model_col, dragged_element_col)
    nlp_stats = df_results_stats(nlp_df, model_col, dragged_element_col)
    final_stats = stats.join(manual_stats).join(nlp_stats, lsuffix='Manual', rsuffix='NLP')\
        .drop(labels='ExpectedElementsNLP', axis=1)\
        .rename({'ExpectedElementsManual': 'Expected Elements'}, axis=1)
    final_stats = final_stats.reset_index()
    final_stats[model_col] = ['Model '+str(i) for i in range(1, final_stats.shape[0] + 1)]
    return final_stats

manual_bpmn = process_results("bpmn_dataset_manual.csv", 'Lane')
nlp_bpmn = process_results("bpmn_dataset_stanza.csv", 'Lane')
stats_bpmn = create_stats(manual_bpmn, nlp_bpmn, 'Model', 'Lane')

manual_ucd = process_results("ucd_dataset_manual.csv", 'UCD Actor')
nlp_ucd = process_results("ucd_dataset_stanza.csv", 'UCD Actor')
stats_ucd = create_stats(manual_ucd, nlp_ucd, 'Diagram', 'UCD Actor')



