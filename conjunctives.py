import os
import argparse
import pandas as pd
from tqdm import tqdm
from parsing import ConjunctiveParser

tqdm.pandas()

def process_dataset(nlp, df: pd.DataFrame, column: str) -> pd.DataFrame:
    parser = ConjunctiveParser(nlp)
    df[column] = df[column].str.replace('&', ' and ')
    outputs = df[column].apply(lambda x: parser.process_parsed(nlp(x)))
    results = pd.DataFrame(data={column: df[column]})
    results['VerbPhrases'] = outputs.apply(lambda res: ';'.join(map(lambda _: _[0] if _[0] is not None else '', res)))
    results['NounPhrases'] = outputs.apply(lambda res: ';'.join(map(lambda _: _[1] if _[1] is not None else '', res)))
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", help="Input dataset file which will be processed", required=True)
    parser.add_argument("--output-file", help="Output file")
    parser.add_argument("--processor", help="Base NLP processor", default='stanza', choices=['spacy', 'stanza'])
    parser.add_argument("--column", help="Column name for processing")
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
        import spacy
        nlp = spacy.load("spacy/en_core_web_lg/en_core_web_lg/en_core_web_lg")
    else:
        import spacy_stanza
        nlp = spacy_stanza.load_pipeline("en", dir='stanza_resources')
    data_df = pd.read_csv(input_file, sep=';')
    results = process_dataset(nlp, data_df, column=args.column)
    results.to_csv(output_file, sep=';', index=False)

