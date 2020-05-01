import os
from collections.abc import Iterable
import argparse
import pandas as pd
from processing import StanzaNLPProcessor, SpacyNLPProcessor, FlairNLPProcessor, CoreNLPProcessor


def tag_activity_dataset(df, processor):
    def extract_verbs(x):
        return '|'.join([processor.normalize_verb(phrase) for phrase in processor.extract_verb_phrases(x)])

    df['VerbPhrases'] = df.iloc[:, 1].apply(extract_verbs)
    df['NounPhrases'] = df.iloc[:, 1].apply(processor.extract_noun_phrases).apply(lambda x: '|'.join(x))
    df['ProperNouns'] = df.iloc[:, 1].apply(processor.extract_proper_nouns).apply(lambda x: '|'.join(x))
    return df

def tag_ner_dataset(df, processor):
    def extract_entity_types(x):
        types = processor.get_named_entity_types(x)
        if types is None or not isinstance(x, Iterable):
            return types
        return '|'.join([x for x in types if x is not None])

    df['Entities'] = df.iloc[:, 0].apply(processor.extract_named_entities).apply(lambda x: '|'.join(x))
    df['EntityType'] = df.iloc[:, 0].apply(extract_entity_types)
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--processor", help="Text processor which is used", choices=['spacy', 'stanza', 'flair', 'corenlp'], default='spacy')
    parser.add_argument("--input-file", help="Input dataset file which will be processed", required=True)
    parser.add_argument("--task", help="Task type", choices=['phrases', 'ner'], default='phrases')
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
    else:
        raise Exception('Invalid processor option')
    if args.task == 'phrases':
        data_df = data_df.iloc[:, :2]
        data_df = tag_activity_dataset(data_df, processor)
    elif args.task == 'ner':
        data_df = data_df[['Entry']]
        data_df = tag_ner_dataset(data_df, processor)
    else:
        raise Exception('Invalid task option')
    data_df.to_csv(output_file, sep=';', index=False)
