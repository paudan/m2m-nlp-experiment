import os
import json
import re
import itertools
from datetime import datetime
import pandas as pd
from processing import StanzaNLPProcessor, SpacyNLPProcessor

BPMN_DATASET_DIR = '../modelCollection_1559160740359/'
DATASETS_PATH = 'datasets'

control_chars = ''.join(map(chr, itertools.chain(range(0,32), range(127,160))))
control_char_re = re.compile('[%s]' % re.escape(control_chars))


def is_process_file(f, mdir):
    if not os.path.isfile(os.path.join(mdir, f)): 
        return False
    split_name = os.path.splitext(f)
    if not '_metadata' in split_name[0] and split_name[1].lower() == '.json':
        return True
    return False

def element_type(shape):
    stencil = shape.get('stencil')
    if stencil is None: return None
    return stencil.get('id')

def element_name(shape):
    props = shape.get('properties')
    if props is None: return None
    name = props.get('name')
    if name is None: return None
    name = control_char_re.sub(' ', name)
    return name.replace('  ', ' ').replace('"', '').strip()
    

def extract_signavio_dataset():
    directories = sorted(os.listdir(BPMN_DATASET_DIR))
    results = list()
    for sdir in directories:
        path = os.path.join(BPMN_DATASET_DIR, sdir, 'BPMN2.0_Process')
        if not os.path.exists(path):
            continue
        dir_proc = os.listdir(path)
        for mdir in dir_proc:
            mdir = os.path.join(path, mdir)
            print('Processing directory:', mdir)
            bpmn_files = [f for f in os.listdir(mdir) if is_process_file(f, mdir)]
            if len(bpmn_files) == 0:
                continue
            for bfile in bpmn_files:
                fpath = os.path.join(mdir, bfile)
                with open(fpath, 'r') as f:
                    model = json.load(f)
                parent = model.get('childShapes')
                if parent is None:  continue
                for shape in parent:
                    if element_type(shape).lower() != 'pool': continue
                    pool_name = element_name(shape)    
                    lanes = shape.get('childShapes')
                    if lanes is None: continue
                    for lane in lanes:
                        if element_type(lane).lower() != 'lane': continue
                        lane_name = element_name(lane)
                        lane_shapes = lane.get('childShapes')
                        if lane_shapes is None: continue
                        for lane_shape in lane_shapes:
                            if element_type(lane_shape).lower() != 'task': continue
                            task_name = element_name(lane_shape)
                            if task_name is None or len(task_name.strip()) <= 1: continue
                            results.append((bfile, pool_name, lane_name, task_name))
    pd.DataFrame(results, columns=['Model', 'Pool', 'Lane', 'Task'])\
       .to_csv(os.path.join('datasets', 'bpmn_dataset.csv'), index=False, sep=';')


def preprocess_signavio_dataset():
    data = pd.read_csv(os.path.join('datasets', 'bpmn_dataset.csv'), sep=';', header=0)
    data = data[data['Task'].str.len() > 2]
    data['Lane'] = data['Lane'].combine_first(data['Pool'])
    data.drop(labels='Pool', axis=1, inplace=True)
    data.drop_duplicates(inplace=True)
    data.to_csv(os.path.join('datasets', 'bpmn_dataset_cleaned.csv'), sep=';', index=None, header=True)
    
def tag_datasets():
    processor = StanzaNLPProcessor()
    df = pd.read_csv(os.path.join('datasets', 'atcs_actors_useCases_tagged.csv'), sep=';', header=0)
    df = tag_dataframe(df, processor)
    df.to_csv(os.path.join('datasets', 'atcs_actors_useCases_tagged_v2.csv'), sep=';', index=False)
    df2 = pd.read_csv(os.path.join('datasets', 'bpmn_diagrams_lanes_tasks_tagged.csv'), sep=';', header=0)
    df2 = tag_dataframe(df2, processor)
    df2.to_csv(os.path.join('datasets', 'bpmn_diagrams_lanes_tasks_tagged_v2.csv'), sep=';', index=False)    
    
def tag_signavio_dataset():
    data = pd.read_csv(os.path.join('datasets', 'bpmn_dataset_cleaned.csv'), sep=';', header=0)
    print('Number of models used:', len(data['Model'].unique()))
    data_dedup = data.drop_duplicates(subset=('Lane', 'Task')).drop(labels='Model', axis=1)
    processor = StanzaNLPProcessor()
    df = tag_dataframe(data_dedup, processor)
    df.to_csv(os.path.join('datasets', 'bpmn_dataset_tagged.csv'), sep=';', index=False)
    print('Finished: ', datetime.now())

def tag_dataframe(df, processor):
    df['Verb'] = df.iloc[:, 1].apply(processor.extract_verb_phrase)
    df['NounPhrases'] = df.iloc[:, 1].apply(processor.extract_noun_phrases).apply(lambda x: '|'.join(x))
    df['Entities'] = df.iloc[:, 1].apply(processor.extract_named_entities).apply(lambda x: '|'.join(x))
    df['EntityType'] = df.iloc[:, 1].apply(processor.get_named_entity_types).apply(lambda x: '|'.join(x))
    return df

def create_activity_dataset():
    processor = SpacyNLPProcessor()
    def process_verb(x): return None if pd.isnull(x) else processor.normalize_verb(x.lower())
    dst1 = pd.read_csv(os.path.join(DATASETS_PATH, 'atcs_actors_useCases_tagged.csv'))
    dst1.iloc[:,-2] = dst1.iloc[:,-2].apply(process_verb)
    dst1.columns = ['Subject', 'Activity', 'Verb phrases', 'Noun phrases']
    dst2 = pd.read_csv(os.path.join(DATASETS_PATH, 'bpmn_diagrams_lanes_tasks_tagged.csv'))
    dst2.iloc[:,-2] = dst2.iloc[:,-2].apply(process_verb)
    dst2.drop(labels=['Entities', 'EntityType'], axis=1, inplace=True)
    dst2.columns = ['Subject', 'Activity', 'Verb phrases', 'Noun phrases']
    dst3 = pd.read_csv(os.path.join(DATASETS_PATH, 'bpmn_dataset_tagged.csv'), sep=';')
    dst3.drop(labels=['Entities', 'EntityType'], axis=1, inplace=True)
    dst3 = dst3[:1500]
    
    def process_multiple_verbs(x):
        if pd.isnull(x):
            return x
        verbs = [s.split('|') for s in x.split(';')]
        verbs = [[None if pd.isnull(x) else processor.normalize_verb(x.lower()) for x in a] for a in verbs]
        return ';'.join('|'.join(s) for s in verbs)
        
    dst3.iloc[:,-2] = dst3.iloc[:,-2].apply(process_multiple_verbs)
    dst3.columns = ['Subject', 'Activity', 'Verb phrases', 'Noun phrases']
    final_df = pd.concat([dst1, dst2, dst3], axis=0)
    final_df.to_csv(os.path.join(DATASETS_PATH, 'activity_dataset_final.csv'), sep=';', index=False)
    
    
def create_named_entity_dataset():
    processor = StanzaNLPProcessor()
    dst_names = pd.read_csv(os.path.join(DATASETS_PATH, 'bpmn_dataset_tagged.csv'), sep=';')
    dst_names = pd.Series(dst_names['Lane'].unique())
    
    def extract_entities(x):
        if x is None or pd.isnull(x): return x
        try:
            return '|'.join(processor.extract_named_entities(x))
        except:
            return None
        
    def extract_entity_types(x):
        if x is None or pd.isnull(x): return x
        try:
            return processor.get_named_entity_type(x)
        except:
            return None
    
    extracted = pd.DataFrame(data={'Entry': dst_names,
        'Entities': dst_names.apply(extract_entities),
        'EntityType': dst_names.apply(extract_entity_types)
    })
    extracted = extracted[~pd.isnull(extracted['Entry'])]
    extracted.to_csv(os.path.join(DATASETS_PATH, 'ner_dataset_final.csv'), sep=';', index=False)
    
    
if __name__ == '__main__':
    tag_signavio_dataset()

