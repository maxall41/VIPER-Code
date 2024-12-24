# Virtual Interaction Predictor for Enzyme Reactions (VIPER)

## Abstract

Enzymes, nature's catalysts, possess remarkable properties such as high stereo-, regio-, and chemo-specificity. These properties allow enzymes to greatly simplify complex synthetic processes, resulting in improved yields and reduced manufacturing costs compared to traditional chemical methods. However, the lack of experimental characterization of enzyme substrates, with only a few thousand out of tens of millions of known enzymes in Uniprot having annotated substrates, severely limits the ability of chemists to repurpose enzymes for industrial applications. Previous machine learning models aimed at predicting enzyme substrates have been hampered by poor generalization to new substrates. Here, we introduce VIPER (Virtual Interaction Predictor for Enzyme Reactivity), a model that achieves an average 30% improvement over the previous state-of-the-art model (ProSmith) in reaction prediction for unseen substrates. Furthermore, we reveal flaws in previous enzyme-substrate reaction datasets, and introduce a novel high-quality enzyme-substrate reaction dataset to alleviate these issues.


Web Server: https://viperwebserver.com

Model Weights: https://zenodo.org/records/12151573

Paper: https://www.biorxiv.org/content/10.1101/2024.06.21.599972v3

License: CC-BY 4.0

## Local usage

If you want to use VIPER on more than 100 records at a time you can use the `scripts/VIPER_run.py` script.
To do so create a new CSV file in `scripts/` with the column: `sequence` for the protein amino acid sequence, and the column: `smiles` for the molecule SMILES string. Then download all of the ensemble weights from https://zenodo.org/records/12151573 and save them in the `scripts/` directory. You can then run the script by doing:

```bash
python scripts/VIPER_run.py --input input_csv_file.csv --output out.csv
```
