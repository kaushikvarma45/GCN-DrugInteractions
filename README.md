# GCN-DrugInteractions
Drug-drug Interaction prediction module : The GCN algorithm can be used to get latent represention of any drugs; using just their SMILES string, followed by predictions on Drug-drug adverse affect interactions.(Python is required to execute this code; the .ipynb files can be directly run on google colab by uploading the data files.)

Running GCN-Module

To run the .ipynb files just open in google colab and upload data files in local runtime to run. The .py files run main.py by uploading the data folders into same directory as present working directory of code(make sure names of code files and path changed in code so the csv files are read)

Folder Information

The data folder contains 3 different datasets for training and testing the module, each is independent run from the other. 
The totalData_{dataset} contains the SMILES string format of a set of drugs used for pre-training the encoder decoder module(Make sure this file is in a folder named 'raw' in working directory); on running first part gives a 'processed' folder containing Molecular graph information of each of these drugs.
The Train and test folder for drug-drug interactions are also present for each of these datasets, just upload to working directory and run code to do the training and testing results.

