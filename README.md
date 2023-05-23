# AttentionMGT-DTA

# AttentionMGT-DTA: A multi-modal drug-target affinity prediction using graph transformer and attention mechanism

# Requirements:
- python 3.9
- cudatoolkit 11.3.1
- pytorch 1.10.0
- rdkit 2022.03.5
- networkx 2.8.4
- dgl 0.9.1
- deepchem 2.6.1
- mdanalysis 2.3.0
- scipy 1.9.1

# How to run
## AttentionMGT-DTA
1. Run protein_process.py to generate the preprocessed protein data.
2. Run compound_process.py to generate the preprocessed compound data.
3. Run train_DTA.py to train and test the model.
