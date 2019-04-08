#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 16:25:09 2017

@author: dbasaran
"""
import numpy as np
from keras.layers import Dense, Reshape, BatchNormalization, Bidirectional, GRU
from keras.layers import Conv2D, LSTM, Input, TimeDistributed, Lambda, ZeroPadding3D
from keras import backend as Kb, Model
import keras as K

import h5py
import os
import click
from sklearn.preprocessing import LabelBinarizer, normalize
import csv
import pandas as pd
import mir_eval

import extract_HF0

# Parameters of the model are globally defined for the trained network
number_of_patches = 20
patch_size = 25
segment_length = 500  # number_of_patches x patch_size
feature_size = 301
number_of_classes = 62
step_notes = 5
SR = 22050
hop_size = 256
RNN = 'GRU'

#########################################################
## GET PATH FUNCTIONS: Functions to return paths
def get_path():
    '''
    Gets the path of the main folder
    :return: path (string)
    '''
    path = os.getcwd()
    path = path[:path.rfind('/')]

    return path


def get_path_to_quantized_annotations():
    quantized_annotations_path = '{0}/quantized_annotations'.format(get_path())

    return quantized_annotations_path


def get_path_to_dataset_audio():
    audio_path = '{0}/medleydb_audio'.format(get_path())
    return audio_path


def get_path_to_pitch_estimations():
    # Wrapper function
    results_path = get_model_output_save_path()

    return results_path


def get_model_output_save_path():
    model_output_save_path = '{0}/medleydb_melody_results/C-RNN_results'.format(get_path())
    if not os.path.exists(model_output_save_path):
        os.makedirs(model_output_save_path)

    return model_output_save_path


def get_dataset_splits_save_path():
    dataset_splits_save_path = '{0}/medleydb_dataset_splits'.format(get_path())

    if not os.path.exists(dataset_splits_save_path):
        os.makedirs(dataset_splits_save_path)

    return dataset_splits_save_path


def get_hf0_path():
    path = '{0}/medleydb_features/HF0s_STFT'.format(get_path())

    return path


def get_dataset_test_load_path():
    dataset_test_load_path = get_hf0_path()
    return dataset_test_load_path


def get_dataset_load_path():
    dataset_load_path = get_dataset_splits_save_path()
    return dataset_load_path


def get_trained_model_save_path(dataset_name):
    trained_model_save_path = '{0}/trained_models'.format(get_path(dataset_name=dataset_name))
    if not os.path.exists(trained_model_save_path):
        os.makedirs(trained_model_save_path)

    return trained_model_save_path

#######################################################


def get_labels(track_name):
    '''
    Get labels for the track
    :param track_name: String - Name of the track in the MedleyDB dataset
    :return: labels: Numpy array - quantized labels of the track with -1 for non-melody and all other target classes starting from 0
    '''
    quantized_annotation_path = get_path_to_quantized_annotations() \
                          + '/{0}_quantized_labels_Fs-22050_hop-256.h5'.format(track_name)
    labels_file = h5py.File(quantized_annotation_path , 'r')
    labels = np.array(labels_file['labels'])

    return labels


def get_pitch_estimation_from_csv(track_name):
    '''
    Gets the pitch estimation of a track from the csv file
    :param track_name: String - Name of the track in the MedleyDB dataset
    :return: pitch_estimation: Numpy array - Estimations for each frame
    '''
    estimation_path = get_path_to_pitch_estimations() + '/{0}.csv'.format(track_name)

    data = pd.read_csv(estimation_path, delimiter=',', header=None)
    pitch_estimation = np.array(data)[:, 1]

    return pitch_estimation


def construct_model(number_of_patches, patch_size, feature_size, number_of_classes, step_notes, RNN='LSTM',
                    verbose=True):
    kernel_coeff = 0.00001

    number_of_channels = 1
    input_shape = (number_of_patches, patch_size, feature_size, number_of_channels)
    inputs = Input(shape=input_shape)

    zp = ZeroPadding3D(padding=(0, 0, 2))(inputs)

    #### CNN LAYERS ####
    cnn1 = TimeDistributed(Conv2D(64, (1, 5),
                                  padding='valid',
                                  activation='relu',
                                  strides=(1, np.int(step_notes)),
                                  kernel_regularizer=K.regularizers.l2(kernel_coeff),
                                  data_format='channels_last', name='cnn1'))(zp)

    cnn1a = BatchNormalization()(cnn1)

    zp = ZeroPadding3D(padding=(0, 1, 2))(cnn1a)

    cnn2 = TimeDistributed(
        Conv2D(64, (3, 5), padding='valid', activation='relu', data_format='channels_last', name='cnn2'))(zp)

    cnn2a = BatchNormalization()(cnn2)

    zp = ZeroPadding3D(padding=(0, 1, 1))(cnn2a)

    cnn3 = TimeDistributed(
        Conv2D(64, (3, 3), padding='valid', activation='relu', data_format='channels_last', name='cnn3'))(zp)

    cnn3a = BatchNormalization()(cnn3)

    zp = ZeroPadding3D(padding=(0, 1, 7))(cnn3a)

    cnn4 = TimeDistributed(
        Conv2D(16, (3, 15), padding='valid', activation='relu', data_format='channels_last', name='cnn4'))(zp)

    cnn4a = BatchNormalization()(cnn4)

    cnn5 = TimeDistributed(
        Conv2D(1, (1, 1), padding='same', activation='relu', data_format='channels_last', name='cnn5'))(cnn4a)

    #### RESHAPING LAYERS ####
    cnn5a = Lambda(lambda x: Kb.squeeze(x, axis=4))(cnn5)

    cnn5b = Reshape((number_of_patches * patch_size, -1), name='cnn5-reshape')(cnn5a)

    #### BIDIRECTIONAL RNN LAYERS ####
    if RNN == 'LSTM':
        rnn1 = Bidirectional(LSTM(128,
                                  kernel_regularizer=K.regularizers.l1_l2(0.0001),
                                  return_sequences=True), name='rnn1')(cnn5b)
    elif RNN == 'GRU':
        rnn1 = Bidirectional(GRU(128,
                                 kernel_regularizer=K.regularizers.l1_l2(0.0001),
                                 return_sequences=True), name='rnn1')(cnn5b)

        #### CLASSIFICATION (DENSE) LAYER ####
    classifier = TimeDistributed(Dense(number_of_classes,
                                       activation='softmax',
                                       kernel_regularizer=K.regularizers.l2(0.00001),
                                       bias_regularizer=K.regularizers.l2()), name='output')(rnn1)

    model = Model(inputs=inputs, outputs=classifier)

    if verbose == True:
        model.summary()

        print('{0} as RNN!'.format(RNN))

    return model


def class_to_freq(input_class, minF0=55, maxF0=1760, number_of_classes=62, step_notes=1):
    '''

    '''

    output_freq = np.zeros(input_class.shape)

    F0_list = minF0 * 2 ** (np.arange(number_of_classes - 1) / (12. * step_notes))
    F0_list = np.append(0, F0_list)

    output_freq = F0_list[input_class+1]

    return output_freq


def get_prediction(HF0, model):

    lb = LabelBinarizer()
    lb.fit(np.arange(-1,number_of_classes-1))

    length_of_sequence = HF0.shape[1]
    number_of_segments = np.int(np.floor(length_of_sequence/segment_length))
    # This is to ensure all the data is used and the length is a multiple of segment_length
    x_test = np.append(HF0[:,:number_of_segments*segment_length],HF0[:,-segment_length:],axis=1)
    x_test = normalize(x_test,norm='l1',axis=0)
    x_test = x_test.T

    # Arrange the shape of the input for the network
    number_of_samples = np.int(x_test.shape[0] / (number_of_patches * patch_size))
    x_test = np.reshape(x_test, (number_of_samples, number_of_patches, patch_size, feature_size))
    x_test = x_test[:, :, :, :, np.newaxis]

    y_predicted = model.predict(x_test, batch_size=16)
    y_predicted = np.reshape(y_predicted, (y_predicted.shape[0] * y_predicted.shape[1], y_predicted.shape[2]))

    y_pred = np.argmax(y_predicted[:, 1:], axis=1)
    pitch_estimates = class_to_freq(y_pred, number_of_classes=number_of_classes, step_notes=1)
    y_pred = np.argmax(y_predicted, axis=1)
    signs = np.ones(y_pred.shape)
    signs[y_pred == 0] = -1
    pitch_estimates = signs * pitch_estimates

    # Set the length of the output estimates back to original length
    pitch_estimates[length_of_sequence-segment_length:length_of_sequence] = pitch_estimates[-segment_length:]
    pitch_estimates = pitch_estimates[:length_of_sequence]

    return pitch_estimates


def save_output(pitch_estimates, output_path):
    """Save output to a csv file

    Parameters
    ----------
    pitch_estimates : np.ndarray
        array of frequency values
    output_path : str
        path to save output

    """
    times = np.arange(len(pitch_estimates)) * np.float(hop_size)/SR
    with open(output_path, 'w') as fhandle:
        csv_writer = csv.writer(fhandle, delimiter=',')
        for t, f in zip(times, pitch_estimates):
            csv_writer.writerow([t, f])


def print_evaluation_results_statistics(evaluation_results_statistics, args):
    print('\n**************************************************\n')

    print('Model {0} - Evaluation results:'.format(args.model_name))

    print('    voicing_recall: mean={0}, std={1}'.format(evaluation_results_statistics['voicing_recall'][0],
                                                         evaluation_results_statistics['voicing_recall'][1]))

    print('    voicing_false_alarm: mean={0}, std={1}'.format(evaluation_results_statistics['voicing_false_alarm'][0],
                                                              evaluation_results_statistics['voicing_false_alarm'][1]))

    print('    raw_pitch_accuracy: mean={0}, std={1}'.format(evaluation_results_statistics['raw_pitch_accuracy'][0],
                                                             evaluation_results_statistics['raw_pitch_accuracy'][1]))

    print('    raw_chroma_accuracy: mean={0}, std={1}'.format(evaluation_results_statistics['raw_chroma_accuracy'][0],
                                                              evaluation_results_statistics['raw_chroma_accuracy'][1]))

    print('    overall_accuracy: mean={0}, std={1}'.format(evaluation_results_statistics['overall_accuracy'][0],
                                                           evaluation_results_statistics['overall_accuracy'][1]))

    print('\n**************************************************\n')

def get_evaluation_results_statistics(voicing_recall,
                                      voicing_false_alarm,
                                      raw_pitch_accuracy,
                                      raw_chroma_accuracy,
                                      overall_accuracy):
    evaluation_results_statistics = {}
    evaluation_results_statistics['voicing_recall'] = [np.mean(voicing_recall), np.std(voicing_recall)]
    evaluation_results_statistics['voicing_false_alarm'] = [np.mean(voicing_false_alarm), np.std(voicing_false_alarm)]
    evaluation_results_statistics['raw_pitch_accuracy'] = [np.mean(raw_pitch_accuracy), np.std(raw_pitch_accuracy)]
    evaluation_results_statistics['raw_chroma_accuracy'] = [np.mean(raw_chroma_accuracy), np.std(raw_chroma_accuracy)]
    evaluation_results_statistics['overall_accuracy'] = [np.mean(overall_accuracy), np.std(overall_accuracy)]

    return evaluation_results_statistics


def evaluate_melody_prediction(track_name, pitch_estimates, verbose):

    try:
        if 'corrected_pitch' in track_name:
            track_name_original = track_name.split('_corrected_pitch')[0]
        else:
            track_name_original = track_name

        labels = get_labels(track_name=track_name_original)
    except:
        print('Error')
    if pitch_estimates is None:
        pitch_estimates = get_pitch_estimation_from_csv(track_name=track_name)

    labels = class_to_freq(labels)

    min_len = np.min((len(labels), len(pitch_estimates)))
    labels = labels[:min_len]
    pitch_estimation = pitch_estimates[:min_len]

    evaluation_results = {}

    (ref_v, ref_c, est_v, est_c) = mir_eval.melody.to_cent_voicing(np.arange(np.size(labels)),
                                                                   labels,
                                                                   np.arange(np.size(labels)),
                                                                   pitch_estimation)

    vr, vfa = mir_eval.melody.voicing_measures(ref_v, est_v)
    rpa = mir_eval.melody.raw_pitch_accuracy(ref_v, ref_c, est_v, est_c, cent_tolerance=80)
    rca = mir_eval.melody.raw_chroma_accuracy(ref_v, ref_c, est_v, est_c, cent_tolerance=80)
    oa = mir_eval.melody.overall_accuracy(ref_v, ref_c, est_v, est_c, cent_tolerance=80)

    evaluation_results['Voicing Recall'] = vr
    evaluation_results['Voicing False Alarm'] = vfa
    evaluation_results['Raw Pitch Accuracy'] = rpa
    evaluation_results['Raw Chroma Accuracy'] = rca
    evaluation_results['Overall Accuracy'] = oa

    if verbose:
        print('{0} - Evaluation results:'.format(track_name))
        print('    voicing_recall_rate = {0}'.format(vr))
        print('    voicing_false_alarm_rate = {0}'.format(vfa))
        print('    raw_pitch_accuracy = {0}'.format(rpa))
        print('    raw_chroma_accuracy = {0}'.format(rca))
        print('    overall_accuracy = {0}'.format(oa))

    return evaluation_results


def load_model(model_weights_path=None):
    model = construct_model(number_of_patches, patch_size, feature_size, number_of_classes, step_notes, RNN=RNN)
    if model_weights_path == None:
        model.load_weights('weights_C-RNN.h5')
    else:
        model.load_weights(filepath=model_weights_path)

    return model


def compute_output(HF0, save_dir, save_name):
    model = load_model()
    pitch_estimates = get_prediction(HF0, model)
    output_path = '{0}/{1}.csv'.format(save_dir, save_name)
    save_output(pitch_estimates, output_path)

    return pitch_estimates


def print_evaluation_results_statistics(evaluation_results_statistics):
    print('\n**************************************************\n')

    print('Evaluation results:')

    print('    voicing_recall: mean={0}, std={1}'.format(evaluation_results_statistics['voicing_recall'][0],
                                                         evaluation_results_statistics['voicing_recall'][1]))

    print('    voicing_false_alarm: mean={0}, std={1}'.format(evaluation_results_statistics['voicing_false_alarm'][0],
                                                              evaluation_results_statistics['voicing_false_alarm'][1]))

    print('    raw_pitch_accuracy: mean={0}, std={1}'.format(evaluation_results_statistics['raw_pitch_accuracy'][0],
                                                             evaluation_results_statistics['raw_pitch_accuracy'][1]))

    print('    raw_chroma_accuracy: mean={0}, std={1}'.format(evaluation_results_statistics['raw_chroma_accuracy'][0],
                                                              evaluation_results_statistics['raw_chroma_accuracy'][1]))

    print('    overall_accuracy: mean={0}, std={1}'.format(evaluation_results_statistics['overall_accuracy'][0],
                                                           evaluation_results_statistics['overall_accuracy'][1]))

    print('\n**************************************************\n')


def get_evaluation_results_statistics(voicing_recall,
                                      voicing_false_alarm,
                                      raw_pitch_accuracy,
                                      raw_chroma_accuracy,
                                      overall_accuracy):
    evaluation_results_statistics = {}
    evaluation_results_statistics['voicing_recall'] = [np.mean(voicing_recall), np.std(voicing_recall)]
    evaluation_results_statistics['voicing_false_alarm'] = [np.mean(voicing_false_alarm), np.std(voicing_false_alarm)]
    evaluation_results_statistics['raw_pitch_accuracy'] = [np.mean(raw_pitch_accuracy), np.std(raw_pitch_accuracy)]
    evaluation_results_statistics['raw_chroma_accuracy'] = [np.mean(raw_chroma_accuracy), np.std(raw_chroma_accuracy)]
    evaluation_results_statistics['overall_accuracy'] = [np.mean(overall_accuracy), np.std(overall_accuracy)]

    return evaluation_results_statistics


@click.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
def main_prediction(file_path, output_path, evaluate_results=False):
    '''
    main function to estimate melody from SF-NMF activations of a track. If the dataset is indicated, then the estimated
    pitch values are also evaluated. Note that the system here is trained with MedleyDB alone.
    :param HF0_fpath: (String) The path to the SF-NMF activations (HF0) of the target track
    :param dataset: (String) Indicates which dataset is the track from (medleydb / jazzomat). Default value None
    :return:
    '''

    ## Load the file: Either an audio file or HF0 estimation file
    try:
        if '.wav' == file_path[-4:]:
            HF0 = extract_HF0.main(audio_fpath=file_path)
        elif '.h5' == file_path[-3:]:
            feats = h5py.File(HF0_fpath, 'r')
            HF0 = np.array(feats['HF0'])

        # track_name = os.path.basename(HF0_fpath).split('.h5')[0]

    except:
        raise RuntimeError('Wav file or HF0 file could not be loaded!')

    ## Load the model
    try:
        model = load_model()
    except:
        raise RuntimeError('Model could not be loaded!')

    ## Estimate the dominant melody
    try:
        pitch_estimates = get_prediction(HF0, model)
    except:
        raise RuntimeError('An error occured in the melody estimation!')

    ## Save the estimations to a csv file
    try:
        # output_path = '{0}/{1}.csv'.format(get_model_output_save_path(),
        #                                    track_name)
        save_output(pitch_estimates, output_path)
    except:
        # output_path = '{0}.csv'.format(track_name)
        save_output(pitch_estimates, output_path)

    ## Evaluate the results if annotations are available
    # try:
    #     if evaluate_results:
    #         evaluation_results = evaluate_melody_prediction(track_name=track_name,
    #                                                         pitch_estimates=pitch_estimates,
    #                                                         verbose=True)
    #     return evaluation_results
    # except:
    #     raise RuntimeError('An error occured in the evaluation!')


if __name__ == '__main__':
    # Example usage:
    # track_name = 'AClassicEducation_NightOwl'
    # HF0_fpath = '{0}/{1}.h5'.format(get_hf0_path(),track_name)
    # audio_fpath = '{0}/{1}.wav'.format(get_path_to_dataset_audio(),track_name)

    main_prediction()
