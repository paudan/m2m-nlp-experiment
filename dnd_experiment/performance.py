#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import pandas as pd
import numpy as np

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

def create_stats(manual_df, nlp_df, model_col, dragged_element_col):

    def generate_elements(xv, xn):
        return ([('Association', x.lower().strip()) for x in xv.split('|')] if isinstance(xv, str) else []) \
            + ([('Class', x.lower().strip()) for x in xn.split('|')] if isinstance(xn, str) else [])

    def evaluation_scores(x, model_col):
        elements_generated = list(map(generate_elements, x['VerbPhrases'], x['NounPhrases']))
        elements_truth = list(map(generate_elements, x['Verb Phrases'], x['Noun Phrases']))
        elements_eq = list(map(lambda x, y: sorted(x) == sorted(y), elements_generated, elements_truth))
        return sum(elements_eq)/len(elements_eq)

    stats = pd.concat([manual_df[[model_col, dragged_element_col]].drop_duplicates().groupby([model_col])[model_col].count(),
                       manual_df.groupby([model_col])[model_col].count()], axis=1)
    stats.columns = ['NumTransformations', 'AtomicTransformations']
    manual_stats = manual_df.groupby(model_col).agg({'ExpectedOutput': 'sum', 'ActualOutput': 'sum', 'OutputDiff': 'mean'})
    manual_stats.columns = ['ExpectedElements', 'OutputElementsManual', 'MeanDiffManual']
    manual_scores = manual_df.groupby(model_col).apply(evaluation_scores, model_col=model_col).rename('AccuracyManual')
    manual_stats = manual_stats.join(manual_scores)
    nlp_stats = nlp_df.groupby(model_col).agg({'ExpectedOutput': 'sum', 'ActualOutput': 'sum', 'OutputDiff': 'mean'})
    nlp_stats.columns = ['ExpectedElements', 'OutputElementsNLP', 'MeanDiffNLP']
    nlp_scores = nlp_df.groupby(model_col).apply(evaluation_scores, model_col=model_col).rename('AccuracyNLP')
    nlp_stats = nlp_stats.join(nlp_scores)
    final_stats = stats.join(manual_stats).join(nlp_stats, rsuffix='_y').drop(labels='ExpectedElements_y', axis=1)
    final_stats = final_stats.reset_index()
    final_stats[model_col] = ['Model '+str(i) for i in range(1, final_stats.shape[0] + 1)]
    return final_stats

manual_bpmn = process_results("bpmn_dataset_manual.csv", 'Lane')
nlp_bpmn = process_results("bpmn_dataset_stanza.csv", 'Lane')
stats_bpmn = create_stats(manual_bpmn, nlp_bpmn, 'Model', 'Lane')

manual_ucd = process_results("ucd_dataset_manual.csv", 'UCD Actor')
nlp_ucd = process_results("ucd_dataset_stanza.csv", 'UCD Actor')
stats_ucd = create_stats(manual_ucd, nlp_ucd, 'Diagram', 'UCD Actor')



