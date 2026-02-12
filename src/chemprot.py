import json
import os
from datasets import load_dataset

from tqdm.auto import tqdm
from typing import List, Dict


def preprocess_chemprot_seq2seq(dataset, output_dir, split):
    data_samples = []
    os.makedirs(output_dir, exist_ok=True)

    for doc in tqdm(dataset, desc=f"Processing {split} samples"):
        doc_id = doc['pmid']
        doc_text = doc['text']
        doc_ents = []
        doc_ent_set = set()
        doc_ent_map = {}
        for i, ent_id in enumerate(doc['entities']['id']):
            ent_start = doc['entities']['offsets'][i][0]
            ent_end = doc['entities']['offsets'][i][1]
            ent_text = doc['entities']['text'][i]
            ent_type = doc['entities']['type'][i]
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
        for arg1_id, arg2_id, rel_type in zip(doc['relations']['arg1'], doc['relations']['arg2'], doc['relations']['type']):
            h_ent = doc_ent_map.get(arg1_id)
            t_ent = doc_ent_map.get(arg2_id)
            h_text, h_type = h_ent['text'], h_ent['type']
            t_text, t_type = t_ent['text'], t_ent['type']
            if (h_text, h_type, t_text, t_type, rel_type) in doc_rel_set:
                continue
            
            h_offset = h_ent['start'], h_ent['end']
            t_offset = t_ent['start'], t_ent['end']
            doc_rels.append({
                'head_id': arg1_id,
                'tail_id': arg2_id,
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
    full_ds = load_dataset("bigbio/chemprot", "chemprot_full_source")
    train_ds = full_ds['train']
    val_ds = full_ds['validation']
    test_ds = full_ds['test']
    output_dir = "bio-datasets/chemprot-seq2seq"
    
    print()
    print("---------- Preprocessing ChemProt Dataset for Seq2Seq ----------\n")
    preprocess_chemprot_seq2seq(dataset=train_ds, output_dir=output_dir, split='train')
    print()
    preprocess_chemprot_seq2seq(dataset=val_ds, output_dir=output_dir, split='val')
    print()
    preprocess_chemprot_seq2seq(dataset=test_ds, output_dir=output_dir, split='test')
    print()
    print("---------- Finished preprocessing ChemProt Dataset for Seq2Seq ----------")
    print()


if __name__ == "__main__":
    main()