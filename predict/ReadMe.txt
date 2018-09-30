This folder contains the prediction code with the trained C-RNN model. The main script is "predict_on_single_audio_CRNN.py". You can choose an audio file or an HF0 file to predict the results. 

- In case of audio file: The source-filter NMF will be applied first to extract activations HF0, then the prediction is applied. Once an audio file is trained with SF-NMF, HF0 is written in a file with the same filename but with an extension of .h5, so that you don't need to go through this everytime. 
- In case of HF0 file: It directly reads the HF0 from an .h5 file.

Example usage:

audio fpath = <your_path_to_audio_file>/<track_name>.wav
main_prediction(file_path=audio_fpath, evaluate_results=True) # Note that if the audio file is in MedleyDB set, then you can use evaluate_results=True otherwise it will give an error.
