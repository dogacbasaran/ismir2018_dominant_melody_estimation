# Main Melody Extraction with Source-Filter NMF and CRNN

## Citation

If you are using this source code please consider citing the following paper:

> D. Basaran, S. Essid and G. Peeters, Main Melody Extraction with Source-Filter NMF and CRNN,  In
18th International Society for Music Information Retrieval Conference, ISMIR, 2018, Paris, France

Bibtex
```
	@inproceedings{basaran2018CRNN,
	Address = {Paris, France},
    	Author = {Basaran, D. and Essid, S. and Peeters, G.},
    	Booktitle = {19th Int.~Soc.~for Music Info.~Retrieval Conf.},
    	Month = {Sep.},
    	Title = {Main Melody Extraction with Source-Filter NMF and CRNN},
    	Year = {2018}
	}
```

## Prediction

To compute dominant melody estimation with the trained CRNN model, you can run the script 

> predict/predict_on_single_audio_CRNN.py 

An example usage exists inside the script. 

## Source-Filter NMF training

To create HF0 activation representation for a single track or the whole dataset, you can run the script

> SF_NMF/extract_HF0.py

An example usage exists inside the script.

## Creating Random Splits

To create random train/validation/test splits, you can run the script

> random_dataset_splits/random_splits.py

An example usage exists in the ReadMe.txt file. Note that random splitting requires HF0 representations, hence one has to first create HF0 representations then is able to use this script.

## Training

To train the model on the random splitted dataset, you can run the script

> CRNN/C-RNN_model1.py

Note that if you want to use a GPU for the training part (probably you should), you would need to adjust the code for that purpose!

## Requirements

The required packages for the environment in the CRNN experiments are given in the requirements.txt file. Note that the main packages needed are

> tensorflow_gpu, keras, pandas, numpy, scipy, scikit-learn, librosa, mir_eval, matplotlib, h5py, 


