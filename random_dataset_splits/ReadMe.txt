This folder contains the code to generate random train/validation/test splits on MedleyDB dataset and saves the generated splits as a dataset in the folder 

/medleydb_dataset_splits

Example usage:

# To create a dataset with number 1 and using the splitting used in the paper. 

>> %run random_splits.py -v --dataset-number 1 --splits-file-path dataset-ismir-splits.json

# If you want to create your own splits

>> %run random_splits.py -v --dataset-number 1 



