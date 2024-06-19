# Virtual Interaction Predictor for Enzyme Reactions (VIPER)

Web Server: https://viperwebserver.com

Model Weights: https://zenodo.org/records/12151573

## Local usage

If you want to use VIPER on more than 100 records at a time you can use the `scripts/VIPER_run.py` script.
Todo so create a new CSV file in `scripts/` with the column: `sequence` for the protein amino acid sequence, and the column: `smiles` for the molecule SMILES string. Then run:

```bash
python scripts/VIPER_run --input input_csv_file.csv --output out.csv
```
