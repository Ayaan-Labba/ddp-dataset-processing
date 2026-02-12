import json
import os
import xml.etree.ElementTree as ET

from tqdm.auto import tqdm
from typing import List, Dict


def preprocess_ddi(dataset_dir, output_dir, split='train'):
    data_samples = []
    os.makedirs(output_dir, exist_ok=True)

    # Walk through all XML files in the directory
    for root_dir, _, files in tqdm(os.walk(dataset_dir), desc=f"Processing {split} files"):
        for file in files:
            if file.endswith(".xml"):
                file_path = os.path.join(root_dir, file)
                tree = ET.parse(file_path)
                root = tree.getroot()
                doc_id = root.get('id')

                # Iterate through sentences
                for sent in root.findall('sentence'):
                    sent_id = sent.get('id')
                    text = sent.get('text')
                    
                    # Extract Entities (NER)
                    ent_map: Dict[str, List] = {}
                    for ent in sent.findall('entity'):
                        e_id = ent.get('id')
                        e_type = ent.get('type')
                        e_text = ent.get('text')
                        offsets = ent.get('charOffset').split(';')
                        start, end = offsets[0].split('-')[0], offsets[-1].split('-')[-1] # include the entire span containing the entity
                        ent_map[e_id] = {'id': e_id, 'text': e_text, 'type': e_type, 'start': start, 'end': end}

                    entities = list(ent_map.values())

                    # Extract Relation Triplets (RE)
                    relations = []
                    doc_rel_set = set()
                    for pair in sent.findall('pair'):
                        h_id, t_id, ddi = pair.get('e1'), pair.get('e2'), pair.get('ddi')
                        if ddi != 'true':
                            continue
                        
                        rel_type = pair.get('type')
                        head, tail = ent_map[h_id], ent_map[t_id]
                        h_offset, h_text, h_type = [head['start'], head['end']], head['text'], head['type']
                        t_offset, t_text, t_type = [tail['start'], tail['end']], tail['text'], tail['type']
                        if (h_text, t_text, h_type, t_type, rel_type) in doc_rel_set:
                            continue

                        relation = {'head_id': h_id, 
                                    'tail_id': t_id,
                                    'head': h_offset, 
                                    'tail': t_offset, 
                                    'head_text': h_text, 
                                    'tail_text': t_text, 
                                    'head_type': h_type, 
                                    'tail_type': t_type, 
                                    'type': rel_type}    
                        relations.append(relation)
                        doc_rel_set.add((h_text, t_text, h_type, t_type, rel_type))

                    # Store parsed sample
                    data_samples.append({
                        'doc_id': doc_id,
                        'sent_id': sent_id,
                        'text': text,
                        'entities': entities,
                        'relations': relations
                    })
    
    output_path = os.path.join(output_dir, f"{split}.jsonl")
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in data_samples:
            f.write(json.dumps(sample) + '\n')
    
    print(f"Saved {len(data_samples)} examples to {output_path}")

def main():
    train_dir = "/home/bt19d200/Ayaan/raw-datasets/DDICorpus/Train"
    test_dir = "/home/bt19d200/Ayaan/raw-datasets/DDICorpus/Test/Test for DDI Extraction task"
    output_dir = "bio-datasets/ddi"
    
    print()
    print("---------- Preprocessing DDI Dataset ----------\n")
    preprocess_ddi(dataset_dir=train_dir, output_dir=output_dir, split='train')
    print()
    preprocess_ddi(dataset_dir=test_dir, output_dir=output_dir, split='test')
    print()
    print("---------- Finished preprocessing DDI Dataset ----------")
    print()


if __name__ == "__main__":
    main()