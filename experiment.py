import os
from collections.abc import Iterable
import argparse
import pandas as pd
from tqdm import tqdm
from processing import (StanzaNLPProcessor, SpacyNLPProcessor, FlairNLPProcessor,
                        CoreNLPProcessor, BertNLPProcessor, AllenNLPProcessor)

tqdm.pandas()

def tag_activity_dataset(df, processor, column=1, normalize=True, set_default_verb=False):
    default_verb = 'perform'

    def extract_verbs(x):
        verb_phrases =  processor.extract_verb_phrases(x)
        if (verb_phrases is None or len(verb_phrases) == 0) and set_default_verb is True:
            return processor.normalize_verb(default_verb) if normalize is True else default_verb
        return '|'.join([processor.normalize_verb(phrase) if normalize is True else phrase for phrase in verb_phrases])

    df['VerbPhrases'] = df.iloc[:, column].progress_apply(extract_verbs)
    df['NounPhrases'] = df.iloc[:, column].progress_apply(processor.extract_noun_phrases).apply(lambda x: '|'.join(x))
    df['ProperNouns'] = df.iloc[:, column].progress_apply(processor.extract_proper_nouns).apply(lambda x: '|'.join(x))
    return df

def tag_activity_dataset_default(df, column=1):
    df['VerbPhrases'] = df.iloc[:, column].apply(lambda x: x.split()[0])
    df['NounPhrases'] = df.iloc[:, column].apply(lambda x: ' '.join(x.split()[1:]))
    df['ProperNouns'] = None
    return df

def tag_ner_dataset(df, processor):

    def extract_entities(x, func=processor.get_named_entity_types):
        try:
            results = func(x)
        except:
            results = None
        if results is None or not isinstance(x, Iterable):
            return results
        return '|'.join([x for x in results if x is not None])

    df['Entities'] = df.iloc[:, 0].progress_apply(lambda x: extract_entities(x, func=processor.extract_named_entities))
    df['EntityType'] = df.iloc[:, 0].progress_apply(lambda x: extract_entities(x, func=processor.get_named_entity_types))
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--processor", help="Used extraction processor", default='stanza',
                        choices=['spacy', 'stanza', 'flair', 'corenlp', 'bert', 'allen', 'simple'])
    parser.add_argument("--input-file", help="Input dataset file which will be processed", required=True)
    parser.add_argument("--task", help="Task type", choices=['phrases', 'ner'], default='phrases')
    parser.add_argument("--column", help="Column index for processing", type=int)
    parser.add_argument("--normalize-verbs", help="Use verb normalization", dest='normalize', action='store_true', default=False)
    parser.add_argument("--output-file", help="Output file")
    args = parser.parse_args()
    input_file = args.input_file
    if not os.path.isfile(input_file):
        print('File {} is not found'.format(input_file))
        exit()
    data_df = pd.read_csv(input_file, sep=';')
    if data_df.shape[0] == 0:
        print('File {} is empty'.format(input_file))
        exit()
    output_file = args.output_file
    if output_file is None:
        fname =  '{}-{}.csv'.format(args.processor, args.task)
        output_file = os.path.join('results', fname)
    if args.processor == 'spacy':
        processor = SpacyNLPProcessor()
    elif args.processor == 'stanza':
        processor = StanzaNLPProcessor()
    elif args.processor == 'flair':
        processor = FlairNLPProcessor()
    elif args.processor == 'corenlp':
        processor = CoreNLPProcessor()
    elif args.processor == 'bert':
        processor = BertNLPProcessor()
    elif args.processor == 'allen':
        processor = AllenNLPProcessor()
    elif args.processor == 'simple':
        processor = None
    else:
        raise Exception('Invalid processor option')
    if args.task == 'phrases':
        column = args.column or 1
        if args.processor == 'simple':
            data_df = tag_activity_dataset_default(data_df, column=column)
        else:
            data_df = tag_activity_dataset(data_df, processor, column=column, normalize=args.normalize)
    elif args.task == 'ner':
        data_df = data_df.iloc[:, args.column or 0].to_frame()
        data_df = tag_ner_dataset(data_df, processor)
    else:
        raise Exception('Invalid task option')
    data_df.to_csv(output_file, sep=';', index=False)
