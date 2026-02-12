import json
import os
import sys
import glob
import csv

from datasets import load_dataset
from tqdm.auto import tqdm
from typing import List, Dict


csv.field_size_limit(sys.maxsize)

def load_dataset(split_directory):
    """
    Parses DrugProt data from a split directory (train/dev/test).
    Automatically locates abstracts, entities, and relations files based on suffixes.
    
    Args:
        split_directory (str): Path to the folder containing the TSV files.
        
    Returns:
        list: A list of document dictionaries.
    """
    
    # Auto-locate files using glob
    # We look for any file ending with the specific suffixes
    abs_files = glob.glob(os.path.join(split_directory, "*_abstracs.tsv"))
    if not abs_files:
        abs_files = glob.glob(os.path.join(split_directory, "*_abstracts.tsv"))
    ent_files = glob.glob(os.path.join(split_directory, "*_entities.tsv"))
    rel_files = glob.glob(os.path.join(split_directory, "*_relations.tsv"))

    # There is only be one set per directory
    abstracts_path = abs_files[0]
    entities_path = ent_files[0]

    # Relations are optional (does not exist for Test set)
    relations_path = rel_files[0] if rel_files else None

    print(f"Found files:\n  - Abstracts: {os.path.basename(abstracts_path)}\n  - Entities:  {os.path.basename(entities_path)}")
    if relations_path:
        print(f"  - Relations: {os.path.basename(relations_path)}")
    else:
        print("  - Relations: None found (skipping relations)")


    docs = {}
    # Load Abstracts
    with open(abstracts_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            doc_id, title, abstract = row[0], row[1], row[2]
            docs[doc_id] = {
                "doc_id": doc_id,
                "text": f"{title} {abstract}",
                "entities": [], 
                "relations": []
            }
    print(f"Extracted {len(docs)} abstracts")

    # Load Entities
    with open(entities_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            doc_id, ent_id, ent_type, start, end, text = row
            docs[doc_id]["entities"].append({
                "id": ent_id,
                "type": ent_type,
                "offset": [int(start), int(end)],
                "text": text
            })

    # Load Relations
    if relations_path:
        with open(relations_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                doc_id, rel_type, arg1_full, arg2_full = row
                arg1_id = arg1_full.split(':')[1]
                arg2_id = arg2_full.split(':')[1]
                docs[doc_id]["relations"].append({
                    "type": rel_type,
                    "arg1": arg1_id,
                    "arg2": arg2_id
                })
        
    return list(docs.values())

def preprocess_drugprot_seq2seq(dataset, output_dir, split):
    data_samples = []
    os.makedirs(output_dir, exist_ok=True)

    for doc in tqdm(dataset, desc=f"Processing {split} samples"):
        doc_id = doc['doc_id']
        doc_text = doc['text']
        doc_ents = []
        doc_ent_set = set()
        doc_ent_map = {}
        for ent in doc['entities']:
            ent_id = ent['id']
            ent_start = ent['offset'][0]
            ent_end = ent['offset'][1]
            ent_text = ent['text']
            ent_type = ent['type']
            doc_ent_map[ent_id] = {
                'id': ent_id,
                'start': ent_start,
                'end': ent_end,
                'text': ent_text,
                'type': ent_type
            }
            if (ent_text, ent_type) in doc_ent_set:
                continue

            doc_ents.append({
                'id': ent_id,
                'start': ent_start,
                'end': ent_end,
                'text': ent_text,
                'type': ent_type
            })
            doc_ent_set.add((ent_text, ent_type))

        doc_rel_set = set()
        doc_rels = []
        for rel in doc['relations']:
            h_id, t_id, rel_type = rel['arg1'], rel['arg2'], rel['type']
            h_ent, t_ent = doc_ent_map.get(h_id), doc_ent_map.get(t_id)
            h_text, h_type = h_ent['text'], h_ent['type']
            t_text, t_type = t_ent['text'], t_ent['type']
            if (h_text, h_type, t_text, t_type, rel_type) in doc_rel_set:
                continue
            
            h_offset = h_ent['start'], h_ent['end']
            t_offset = t_ent['start'], t_ent['end']
            doc_rels.append({
                'head_id': h_id,
                'tail_id': t_id,
                'head': h_offset, 
                'tail': t_offset, 
                'head_text': h_text, 
                'tail_text': t_text, 
                'head_type': h_type, 
                'tail_type': t_type, 
                'type': rel_type
            })

            doc_rel_set.add((h_text, h_type, t_text, t_type, rel_type))
        
        data_samples.append({
            "doc_id": doc_id,
            "text": doc_text,
            "entities": doc_ents,
            "relations": doc_rels
        })

    output_path = os.path.join(output_dir, f"{split}.jsonl")
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in data_samples:
            f.write(json.dumps(sample) + '\n')
    
    print(f"Saved {len(data_samples)} examples to {output_path}")


def main():
    print("Loading dataset ...")
    print()
    print("Loading train split:")
    train_ds = load_dataset("/home/bt19d200/Ayaan/raw-datasets/DrugProt/training")
    print("Loading validation split:")
    print()
    val_ds = load_dataset("/home/bt19d200/Ayaan/raw-datasets/DrugProt/development")
    print("Loading test split:")
    print()
    test_ds = load_dataset("/home/bt19d200/Ayaan/raw-datasets/DrugProt/test-background")
    output_dir = "bio-datasets/drugprot-seq2seq"
    
    print()
    print("---------- Preprocessing DrugProt Dataset for Seq2Seq ----------\n")
    preprocess_drugprot_seq2seq(dataset=train_ds, output_dir=output_dir, split='train')
    print()
    preprocess_drugprot_seq2seq(dataset=val_ds, output_dir=output_dir, split='val')
    print()
    preprocess_drugprot_seq2seq(dataset=test_ds, output_dir=output_dir, split='test')
    print()
    print("---------- Finished preprocessing DrugProt Dataset for Seq2Seq ----------")
    print()


if __name__ == "__main__":
    main()