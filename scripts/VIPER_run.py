import ankh
import torch
import numpy as np
from deps.molformer import compute_molformer_emb
from io import StringIO  
from deps.model import create_models
import os
import fire
import pandas as pd

SEQ_COL = 'sequence'
SMILES_COL = 'smiles'
models = []

def std_dev_to_confidence(std):
    low_thresh = 0.08
    high_thresh = 0.04
    if std < high_thresh:
        return "HIGH"
    elif std > low_thresh:
        return "LOW"
    else:
        return "MEDIUM"

def gen_ankh(seq):
    model, tokenizer = ankh.load_base_model()
    model.eval()
    model.cuda()
    outputs = tokenizer.batch_encode_plus([list(seq)], 
                                    add_special_tokens=True, 
                                    padding=True, 
                                    is_split_into_words=True, 
                                    return_tensors="pt")
    with torch.no_grad():
        embeddings = model(input_ids=outputs['input_ids'].cuda(), attention_mask=outputs['attention_mask'].cuda())
    embeddings = embeddings.last_hidden_state.squeeze()
    return embeddings

def run(row):
    global models
    ankh = gen_ankh(row[SEQ_COL])
    molformer = compute_molformer_emb(row[SMILES_COL])
    ankh = ankh.unsqueeze(0)
    molformer = molformer.unsqueeze(0)
    # Run Model
    results = []
    for model in models:
        result = model(ankh,molformer)
        result = result.cpu().item()
        results.append(result)
    results = np.array(results)
    std = np.std(results)
    row['result'] = np.mean(results)
    row['confidence'] = std_dev_to_confidence(std)
    return row

def exec(input='in.csv',output="out.csv"):
    global models
    df = pd.read_csv(input)
    models = create_models()
    df = df.apply(run,axis=1)
    df.to_csv(output)
    print("Output saved!")

if __name__ == '__main__':
  fire.Fire(exec)
