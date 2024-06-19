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

def run(row):
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
    df = pd.read_csv(input)
    models = create_models()
    df = df.apply(run,axis=1)
    df.to_csv(output)
    print("Output saved!")

if __name__ == '__main__':
  fire.exec(exec)