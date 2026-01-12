import json
import os

from bioc import biocxml, biocjson
from tqdm.auto import tqdm
from typing import List, Dict, Any


def load_file(path):
    with open(path, 'r') as fp:
        collection = biocxml.load(fp)
    
    json_string = biocjson.dumps(collection)
    return json.loads(json_string)

def process_cdr(dataset: List[Dict], output_dir: str, split: str) -> None:
    """
    Process CDR BioC data into the standard format and save to "{output-dir}/cdr/{split}.jsonl".
    """
    os.makedirs(output_dir, exist_ok=True)
    processed_data = []
    for doc in tqdm(dataset, desc=f"Processing {split} split"):
        doc_id = doc['id']

        # Combine title and abstract texts and entities
        passages = doc['passages']
        doc_text = passages[0]['text'] + " " + passages[1]['text']
        annotations = passages[0]['annotations'] + passages[1]['annotations']
        doc_ent_map: Dict[str, List] = {}
        doc_ents = []
        for ann in annotations:
            if 'CompositeRole' in ann['infons'] and ann['infons']['CompositeRole'] == 'CompositeMention':
                continue # skip composite mentions

            ent_id = ann['infons']['MESH']
            if ent_id not in doc_ent_map:
                if '|' in ent_id:
                    ent_ids = ent_id.split('|')
                    for id in ent_ids:
                        if id not in doc_ent_map:
                            doc_ent_map[id] = []
                
                doc_ent_map[ent_id] = []

            start = ann['locations'][0]['offset']
            end = ann['locations'][-1]['offset'] + ann['locations'][-1]['length']            
            ent = {
                'id': ent_id,
                'text': ann['text'],
                'type': ann['infons']['type'],
                'start': start,
                'end': end
            }
            doc_ents.append(ent)
            doc_ent_map[ent_id].append(ent)
            if '|' in ent_id:
                for id in ent_ids:
                    doc_ent_map[id].append(ent)

        # Get relations
        doc_rels = []
        for rel in doc['relations']:
            h_id, t_id = rel['infons']['Chemical'], rel['infons']['Disease']
            rel = {
                'head_id': h_id, 
                'tail_id': t_id,
                'type': rel['infons']['relation']
            }
            heads, tails = doc_ent_map.get(h_id, []), doc_ent_map.get(t_id, [])
            if split == 'test':
                rel['head'] = heads
                rel['tail'] = tails
            else:
                for head in heads:
                    for tail in tails:
                        rel['head'] = [head['start'], head['end']]
                        rel['tail'] = [tail['start'], tail['end']]
                        rel['head_text'] = head['text']
                        rel['tail_text'] = tail['text']
                        rel['head_type'] = head['type']
                        rel['tail_type'] = tail['type']
            
            doc_rels.append(rel)
        
        processed_data.append({
            'id': doc_id,
            'text': doc_text,
            'entities': doc_ents,
            'relations': doc_rels
        })

    # Save processed split to JSONL file
    output_path = f"{output_dir}/{split}.jsonl"
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in processed_data:
            f.write(json.dumps(item) + '\n') # dump each dictionary as its own line in the file
        
    print(f"Saved {len(dataset)} examples to {output_path}.")

def process_cdr_seq2seq(dataset: List[Dict], output_dir: str, split: str) -> None:
    """
    Process CDR BioC data into the standard format for seq2seq and save to "{output_dir}/cdr/{split}.jsonl".
    """
    os.makedirs(output_dir, exist_ok=True)
    processed_data = []
    for i, doc in enumerate(tqdm(dataset, desc=f"Processing {split} split")):
        doc_id = doc['id']
        
        # Combine title and abstract texts and entities
        passages = doc['passages']
        doc_text = passages[0]['text'] + " " + passages[1]['text']
        annotations = passages[0]['annotations'] + passages[1]['annotations']
        doc_ent_map: Dict[str, Any] = {}
        doc_ents = []
        for ann in annotations:
            if 'CompositeRole' in ann['infons'] and ann['infons']['CompositeRole'] == 'CompositeMention':
                continue # skip composite mentions
            
            ent_id = ann['infons']['MESH']
            if ent_id not in doc_ent_map:
                if '|' in ent_id:
                    ent_ids = ent_id.split('|')
                    for id in ent_ids:
                        if id not in doc_ent_map:
                            doc_ent_map[id] = {'texts': set(), 'ents': []}
                
                doc_ent_map[ent_id] = {'texts': set(), 'ents': []}
            
            if ann['text'].lower() in doc_ent_map[ent_id]['texts']:
                continue # skip similar text forms
            
            start = ann['locations'][0]['offset']
            end = ann['locations'][-1]['offset'] + ann['locations'][-1]['length']
            ent = {
                'id': ent_id,
                'text': ann['text'],
                'type': ann['infons']['type'],
                'start': start,
                'end': end
            }
            doc_ents.append(ent)
            if '|' in ent_id:
                for id in ent_ids:
                    doc_ent_map[id]['ents'].append(ent)
                    doc_ent_map[id]['texts'].add(ann['text'].lower())

            doc_ent_map[ent_id]['ents'].append(ent)
            doc_ent_map[ent_id]['texts'].add(ann['text'].lower())

        # Get relations
        doc_rels = []
        for rel in doc['relations']:
            h_id, t_id = rel['infons']['Chemical'], rel['infons']['Disease']
            rel = {
                'head_id': h_id, 
                'tail_id': t_id,
                'type': rel['infons']['relation']
            }
            heads, tails = doc_ent_map.get(h_id, {}).get('ents', []), doc_ent_map.get(t_id, {}).get('ents', [])
            if split == 'test':
                rel['head'] = heads
                rel['tail'] = tails
            else:
                for head in heads:
                    for tail in tails:
                        rel['head'] = [head['start'], head['end']]
                        rel['tail'] = [tail['start'], tail['end']]
                        rel['head_text'] = head['text']
                        rel['tail_text'] = tail['text']
                        rel['head_type'] = head['type']
                        rel['tail_type'] = tail['type']
            
            doc_rels.append(rel)
        
        processed_data.append({
            'id': doc_id,
            'text': doc_text,
            'entities': doc_ents,
            'relations': doc_rels
        })

    # Save processed split to JSONL file
    output_path = f"{output_dir}/{split}.jsonl"
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in processed_data:
            f.write(json.dumps(item) + '\n') # dump each dictionary as its own line in the file
        
    print(f"Saved {len(dataset)} examples to {output_path}.")

def main():
    train_path = "/home/bt19d200/Ayaan/raw-datasets/CDR/CDR_Data/CDR.Corpus.v010516/CDR_TrainingSet.BioC.xml"
    val_path = "/home/bt19d200/Ayaan/raw-datasets/CDR/CDR_Data/CDR.Corpus.v010516/CDR_DevelopmentSet.BioC.xml"
    test_path = "/home/bt19d200/Ayaan/raw-datasets/CDR/CDR_Data/CDR.Corpus.v010516/CDR_TestSet.BioC.xml"
    
    ds_train = load_file(train_path)['documents']
    ds_val = load_file(val_path)['documents']
    ds_test = load_file(test_path)['documents']
    
    extraction_dir = "bio-datasets/cdr"
    seq2seq_dir = "bio-datasets/cdr-seq2seq"
    
    print()
    print("---------- Preprocessing CDR dataset ----------\n")
    print("Processing extraction format ...")
    process_cdr(ds_train, extraction_dir, 'train')
    process_cdr(ds_val, extraction_dir, 'val')
    process_cdr(ds_test, extraction_dir, 'test')
    print()
    print("Processing seq2seq format ...")
    process_cdr_seq2seq(ds_train, seq2seq_dir, 'train')
    process_cdr_seq2seq(ds_val, seq2seq_dir, 'val')
    process_cdr_seq2seq(ds_test, seq2seq_dir, 'test')
    print()
    print("---------- Finished preprocessing CDR dataset ----------\n")


if __name__ == "__main__":
    main()