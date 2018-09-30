#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Common functions that are utilized by neural network models

@author: dbasaran
"""
import numpy as np
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau,Callback
import keras as K

import logging
import h5py
import json
import os
import csv
import sys
import pandas as pd
import mir_eval

from sklearn.preprocessing import LabelBinarizer,normalize

import argparse


def parse_input(input_args):
    '''
    Parsing the input arguments
    :param input_args: Input arguments from the console
    :return: args: List of parsed arguments
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument("-v", "--verbose",
                        action="store_true",
                        help="Print the procedure")

    parser.add_argument("--gpu-segment",
                        action="store",
                        dest="gpu", type=int,
                        help="CUDA_VISIBLE_DEVICES setting",
                        default=0)

    parser.add_argument("--patch-size",
                        action="store",
                        dest="patch_size", type=int,
                        help="The width (length) of a CNN patch/image",
                        default=25)

    parser.add_argument("--number-of-patches",
                        action="store",
                        dest="number_of_patches", type=int,
                        help="The number of consecutive CNN patches to be fed into RNN layer",
                        default=20)

    parser.add_argument("--batch-size",
                        action="store",
                        dest="batch_size", type=int,
                        help="The number of samples in the training phase",
                        default=16)

    parser.add_argument("--epochs",
                        action="store",
                        dest="epochs", type=int,
                        help="The number epochs in the training phase",
                        default=100)

    parser.add_argument("--drop-out",
                        action="store",
                        dest="drop_out", type=float,
                        help="The dropout ratio in the network",
                        default=0.3)

    parser.add_argument("--number-of-classes",
                        action="store",
                        dest="number_of_classes", type=int,
                        help="The number of target note classes (Including the non-melody class)",
                        default=62)

    parser.add_argument("--step-notes",
                        action="store",
                        dest="step_notes", type=int,
                        help="The number of F0's between each semitone",
                        default=5)

    parser.add_argument("--sampling-rate",
                        action="store",
                        dest="SR", type=int,
                        help="Sampling rate for the signals",
                        default=22050)

    parser.add_argument("--hop-size",
                        action="store",
                        dest="hop_size", type=int,
                        help="Hopsize for the signals",
                        default=256)

    parser.add_argument("--dataset-number",
                        action="store",
                        dest="dataset_number", type=int,
                        help="The number of the dataset i.e., dataset 1, dataset 2 etc.",
                        default=1)

    parser.add_argument("--RNN-type",
                        action="store",
                        dest="RNN", type=str,
                        help="Type of the RNN LSTM/GRU",
                        default='GRU')

    parser.add_argument("--model-name",
                        action="store",
                        dest="model_name", type=str,
                        help="The name of the model",
                        default=None)

    parser.add_argument("--early-stopping-patience",
                        action="store",
                        dest="early_stopping_patience", type=int,
                        help="The patience value for the EarlyStopping callback",
                        default=20)

    parser.add_argument("--reduce-LR-patience",
                        action="store",
                        dest="reduce_LR_patience", type=int,
                        help="The patience value for the ReduceLROnPlateau callback",
                        default=10)

    parser.add_argument("--feature-size",
                        action="store",
                        dest="feature_size", type=int,
                        help="The feature size of the input (Default for step_size=5, minFreq=55, maxFreq=1760)",
                        default=301)

    parser.add_argument("--augment-data",
                        action="store_true",
                        default=False,
                        help="Use augmentation if this option is assigned to True")

    parser.add_argument("--dataset-name",
                        action="store",
                        dest="dataset_name", type=str,
                        help="The name of dataset medleydb/jazzomat",
                        default='medleydb')

    parser.add_argument("--use-part-of-training-set",
                        action="store_true",
                        default=False,
                        help="Use augmentation if this option is assigned to True")

    parser.add_argument("--training-amount-percentage",
                        action="store",
                        dest="training_amount_percentage", type=float,
                        help="The amount of the training data that is going to be used for training. Effective only if "
                             "--use-part-of-training indicator is True.",
                        default=80.)

    parser.add_argument("--use-part-of-training-set-per-epoch",
                        action="store_true",
                        default=False,
                        help="Use augmentation if this option is assigned to True")

    parser.add_argument("--training-amount-number-of-samples",
                        action="store",
                        dest="training_amount_number_of_samples", type=float,
                        help="The number of batches in the training data that is going to be used for training in one "
                             "epoch. Effective only if --use-part-of-training-set-per-epoch indicator is True.",
                        default=120.)

    args = parser.parse_args(input_args)

    return args


class Logger(object):
    def __init__(self, args):
        self.terminal = sys.stdout
        self.log = open('{0}/CRNN-model_{1}-dataset-{2}.log'.format(get_path(),
                                                                    args.model_name, args.dataset_number), "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass


def read_arguments(input_args):
    '''
    Reads the arguments from the user and makes the number_of_classes, step_notes and dataset_name parameters compatible
    with the chosen dataset.
    :param input_args: Input arguments
    :return: args: List of parsed arguments
    '''
    args = parse_input(input_args[1:])

    args.model_name = input_args[0].split('_')[-1].split('.')[0].split('del')[-1]

    with open('{0}/dataset-{1}_parameters.json'.format(get_dataset_load_path(), args.dataset_number), 'r') as f:
        parameters = json.load(f)

    args.number_of_classes = parameters['number_of_classes']
    args.step_notes = parameters['step_notes']
    args.dataset_name = parameters['dataset_name']

    return args


def print_arguments(args):
    print('\n\n******* Experiment *******')
    print('Model {0}: '.format(args.model_name))
    print('Parameters:')
    print('    number_of_classes: {0}'.format(args.number_of_classes))
    print('    step_notes: {0}'.format(args.step_notes))
    print('    patch_size: {0}'.format(args.patch_size))
    print('    number_of_patches: {0}'.format(args.number_of_patches))
    print('    feature_size: {0}'.format(args.feature_size))
    print('    segment length: {0}'.format(np.int(args.patch_size*args.number_of_patches)))
    print('    augment: {0}'.format(args.augment_data))
    print('    batch_size:{0}'.format(args.batch_size))
    print('    number of epochs: {0}'.format(args.epochs))
    print('    dropout value: {0}'.format(args.drop_out))
    print('    dataset_number: {0}'.format(args.dataset_number))
    print('    dataset_name: {0}'.format(args.dataset_name))
    print('    use_part_of_training_set: {0}'.format(args.use_part_of_training_set))
    if args.use_part_of_training_set:
        print('    training_amount_percentage: {0:.1f}'.format(args.training_amount_percentage))


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


def get_model_output_save_path(dataset_name, args):
    model_output_save_path = '{0}/{1}_melody_results/C-RNN_results/model-{2}_datasetNumber-{3}_batchSize-{4}_patchSize-{5}_numberOfPatches-{6}'.format(get_path(dataset_name=dataset_name),
                                                                            dataset_name,
                                                                            args.model_name,
                                                                            args.dataset_number,
                                                                            args.batch_size,
                                                                            args.patch_size,
                                                                            args.number_of_patches)
    if not os.path.exists(model_output_save_path):
        os.makedirs(model_output_save_path)

    return model_output_save_path

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


def train_model(model, args):
    '''
    The function that trains a certain neural network model with the given arguments.
    :param model: Keras.Model - Constructed model
    :param args: List - Input arguments
    :return:
    '''
    x_train, y_train, x_validation, y_validation = load_dataset_TD(dataset_number=args.dataset_number, args=args)

    dataset_train_size = x_train.shape[0]  # First dimension gives the number of samples
    dataset_validation_size = x_validation.shape[0]

    # Set the optimizers
    opt_ADAM = Adam(clipnorm=1., clipvalue=0.5)
    opt_SGD = SGD(lr=0.0005, decay=1e-4, momentum=0.9, nesterov=True)

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer=opt_ADAM, metrics=['accuracy'])

    # Use either a part of training set per epoch or all the set per epoch
    if args.use_part_of_training_set_per_epoch:
        number_of_batches_train = np.int(np.floor(args.training_amount_number_of_samples/args.batch_size))
    else:
        number_of_batches_train = np.max((np.floor((dataset_train_size) / args.batch_size), 1))

    number_of_batches_validation = np.max((np.floor(dataset_validation_size / args.batch_size), 1))

    if args.use_part_of_training_set:
        filename = 'model{0}_' \
                   'datasetNumber-{1}_' \
                   'augment-{2}_patchSize-{3}_' \
                   'numberOfPatches-{4}_' \
                   'batchSize-{5}_' \
                   'batchInOneEpoch-{6}_' \
                   'trainingAmountPercentage-{7}'.format(
            args.model_name, args.dataset_number, args.augment_data, args.patch_size, args.number_of_patches,
            args.batch_size, number_of_batches_train, np.int(args.training_amount_percentage))
    else:
        filename = 'model{0}_' \
                   'datasetNumber-{1}_' \
                   'augment-{2}_' \
                   'patchSize-{3}_' \
                   'numberOfPatches-{4}_' \
                   'batchSize-{5}_' \
                   'batchInOneEpoch-{6}'.format(
            args.model_name, args.dataset_number, args.augment_data, args.patch_size, args.number_of_patches,
            args.batch_size, number_of_batches_train)

    cb = set_callbacks(filename, args=args)

    model.fit_generator(data_generator_MedleyDB_train(x_train, y_train, dataset_train_size, batch_size=args.batch_size),
                        number_of_batches_train,
                        args.epochs,
                        validation_data=data_generator_MedleyDB_validation(x_validation, y_validation,
                                                                           dataset_validation_size,
                                                                           batch_size=args.batch_size),
                        validation_steps=number_of_batches_validation,
                        callbacks=cb,
                        verbose=2)

    model.load_weights('{0}/{1}.h5'.format(get_trained_model_save_path(dataset_name=args.dataset_name), filename))

    return model


def class_to_freq(input_class, minF0=55, maxF0=1760, number_of_classes=62, step_notes=1):
    '''
    Converts class labels to frequency in Hz
    :param input_class: A list of classes
    :param minF0: Minimum frequency
    :param maxF0: Maximum frequency
    :param number_of_classes: Number of classes
    :param step_notes: Number of F0s in one semitone
    :return: output_freq: Output frequencies for class symbols
    '''
    output_freq = np.zeros(input_class.shape)

    F0_list = minF0 * 2 ** (np.arange(number_of_classes - 1) / (12. * step_notes))
    F0_list = np.append(0, F0_list)

    output_freq = F0_list[input_class+1]

    return output_freq


def get_prediction(HF0, model, args):
    '''
    Predicts the melody estimations for a trained model and given HF0.
    :param HF0: Input HF0 saliency representation
    :param model: Trained model
    :param args: Input arguments
    :return: pitch_estimates: The estimated pitch values for each frame
    '''

    # Read the dataset split parameters
    with open('{0}/dataset-{1}_parameters.json'.format(get_dataset_load_path(),args.dataset_number),'r') as f:
        parameters = json.load(f)

    segment_length = parameters['segment_length']

    lb = LabelBinarizer()
    lb.fit(np.arange(-1,args.number_of_classes-1))

    length_of_sequence = HF0.shape[1]
    number_of_segments = np.int(np.floor(length_of_sequence/segment_length))
    # This is to ensure all the data is used and the length is a multiple of segment_length
    x_test = np.append(HF0[:,:number_of_segments*segment_length],HF0[:,-segment_length:],axis=1)
    x_test = normalize(x_test,norm='l1',axis=0)
    x_test = x_test.T

    # Arrange the shape of the input for the network
    number_of_samples = np.int(x_test.shape[0] / (args.number_of_patches * args.patch_size))
    x_test = np.reshape(x_test, (number_of_samples, args.number_of_patches, args.patch_size, args.feature_size))
    x_test = x_test[:, :, :, :, np.newaxis]

    y_predicted = model.predict(x_test, batch_size=16)
    y_predicted = np.reshape(y_predicted, (y_predicted.shape[0] * y_predicted.shape[1], y_predicted.shape[2]))

    y_pred = np.argmax(y_predicted[:, 1:], axis=1)
    pitch_estimates = class_to_freq(y_pred, number_of_classes=args.number_of_classes, step_notes=1)
    y_pred = np.argmax(y_predicted, axis=1)
    signs = np.ones(y_pred.shape)
    signs[y_pred == 0] = -1
    pitch_estimates = signs * pitch_estimates

    # Set the length of the output estimates back to original length
    pitch_estimates[length_of_sequence-segment_length:length_of_sequence] = pitch_estimates[-segment_length:]
    pitch_estimates = pitch_estimates[:length_of_sequence]

    return pitch_estimates


def save_output(pitch_estimates, output_path, args):
    """Save output to a csv file

    Parameters
    ----------
    pitch_estimates : np.ndarray
        array of frequency values
    output_path : str
        path to save output

    """
    times = np.arange(len(pitch_estimates)) * np.float(args.hop_size)/args.SR
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


def evaluate_melody_prediction(track_name, pitch_estimates, args):
    '''
    Evaluates the melody prediction for a single track

    :param track_name: Name of the track to be evaluated
    :param pitch_estimates: Pitch estimates for that track
    :param args: Input arguments
    :return: evaluation_results: A dictionary that holds the metrics
    '''

    if 'corrected_pitch' in track_name:
        track_name_original = track_name.split('_corrected_pitch')[0]
    else:
        track_name_original = track_name

    try:
        labels = get_labels(track_name=track_name_original)
    except:
        print('Error')
    if pitch_estimates is None:
        pitch_estimates = get_pitch_estimation_from_csv(track_name=track_name, args=args)

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

    if args.verbose:
        print('{0} - Evaluation results:'.format(track_name))
        print('    voicing_recall_rate = {0}'.format(vr))
        print('    voicing_false_alarm_rate = {0}'.format(vfa))
        print('    raw_pitch_accuracy = {0}'.format(rpa))
        print('    raw_chroma_accuracy = {0}'.format(rca))
        print('    overall_accuracy = {0}'.format(oa))

    return evaluation_results


def evaluate_model(model,args):
    '''
    Evaluates the trained model by testing every test track
    
    :param model: Trained model
    :param args: Input arguments
    :return: 
    '''

    with open('{0}/dataset-{1}_splits.json'.format(get_dataset_load_path(),args.dataset_number),'r') as f:
        dataset_splits = json.load(f)

    voicing_recall = []
    voicing_false_alarm = []
    raw_pitch_accuracy = []
    raw_chroma_accuracy = []
    overall_accuracy = []

    for i, track_name in enumerate(dataset_splits['test']):
        if args.verbose:
            print('{0} - {1}'.format(i, track_name))

        try:
            feats = h5py.File('{0}/{1}.h5'.format(get_dataset_test_load_path(), track_name), 'r')
            HF0 = np.array(feats['HF0_firstestimation'])
        except:
            raise ValueError('An error occured in loading HF0!!')

        try:
            pitch_estimates = get_prediction(HF0, model, args=args)
        except:
            print('An error occured in the melody estimation!!')

        try:
            output_path = '{0}/{1}.csv'.format(get_model_output_save_path(dataset_name=args.dataset_name, args=args),
                                               track_name)
            save_output(pitch_estimates, output_path, args)
        except:
            print('An error occured in saving the output prediction!!')

        try:
            evaluation_results = evaluate_melody_prediction(track_name=track_name,
                                                            pitch_estimates=pitch_estimates,
                                                            args=args)

            voicing_recall.append(evaluation_results['Voicing Recall'])
            voicing_false_alarm.append(evaluation_results['Voicing False Alarm'])
            raw_pitch_accuracy.append(evaluation_results['Raw Pitch Accuracy'])
            raw_chroma_accuracy.append(evaluation_results['Raw Chroma Accuracy'])
            overall_accuracy.append(evaluation_results['Overall Accuracy'])
        except:
            print('An error occured in the evaluation of the prediction: {0}'.format(track_name))

    evaluation_results_statistics = get_evaluation_results_statistics(voicing_recall=voicing_recall,
                                                                      voicing_false_alarm=voicing_false_alarm,
                                                                      raw_pitch_accuracy=raw_pitch_accuracy,
                                                                      raw_chroma_accuracy=raw_chroma_accuracy,
                                                                      overall_accuracy=overall_accuracy)

    if args.verbose:
        print_evaluation_results_statistics(evaluation_results_statistics=evaluation_results_statistics, args=args)


def load_dataset_TD(dataset_number, args):
    '''
    Loads the dataset split with the dataset_number 
    
    :param dataset_number: The split number
    :param args: Input arguments
    :return: x_train: The training data
    :return: y_train: The training annotations
    :return: x_validation: The validation data
    :return: y_validation: The validation annotations
    '''

    with open('{0}/dataset-{1}_parameters.json'.format(get_dataset_load_path(),dataset_number),'r') as f:
        parameters = json.load(f)

    number_of_classes = parameters['number_of_classes']

    patch_size = args.patch_size
    number_of_patches = args.number_of_patches

    train = h5py.File('{0}/dataset-{1}_train.h5'.format(get_dataset_load_path(),dataset_number),'r')
    x_train = train['x_train']
    y_train = train['y_train']

    validation = h5py.File('{0}/dataset-{1}_validation.h5'.format(get_dataset_load_path(), dataset_number), 'r')
    x_validation = validation['x_validation']
    y_validation = validation['y_validation']

    feature_size = x_train.shape[1]
    
    number_of_samples = np.int(x_train.shape[0]/(number_of_patches*patch_size))
    
    # If only a piece of the training set is used for training
    if args.use_part_of_training_set:
        if args.training_amount_percentage == None:
            print('Error: Please insert a percentage amount of training set between 0 and 100')
            return None
        # Number of samples is adjusted according to the percentage input.
        number_of_samples = np.int(args.training_amount_percentage / 100. * number_of_samples)
        x_train = x_train[:number_of_samples*(number_of_patches * patch_size),:]
        y_train = y_train[:number_of_samples * (number_of_patches * patch_size), :]

    x_train = np.reshape(x_train, (number_of_samples, number_of_patches, patch_size, feature_size))
    x_train = x_train[:,:,:,:,np.newaxis]
    y_train = np.reshape(y_train, (number_of_samples, np.int(number_of_patches*patch_size), number_of_classes))

    number_of_samples = np.int(x_validation.shape[0]/(number_of_patches*patch_size))
    x_validation = np.reshape(x_validation, (number_of_samples, number_of_patches, patch_size, feature_size))
    x_validation = x_validation[:,:,:,:,np.newaxis]
    y_validation = np.reshape(y_validation, (number_of_samples, np.int(number_of_patches*patch_size), number_of_classes))

    return x_train, y_train, x_validation, y_validation


def data_generator_MedleyDB_train(x_train, y_train, dataset_size, batch_size=100):
    '''
    The data generator for sending data to fit_generator for training
    
    :param x_train: The training data 
    :param y_train: The training annotations
    :param dataset_size: The size for one epoch
    :param batch_size: The number batches in a minibatch
    :return: 
    '''
    batch_size = np.min((batch_size, dataset_size))
    randomized_indices = np.random.choice(np.arange(dataset_size), size=dataset_size, replace=False)
    upper_limit = len(randomized_indices)
    
    i = 0
    while (True):
        sample_indices = randomized_indices[i:i + batch_size]
        yield x_train[sample_indices, :, :, :, :], y_train[sample_indices, :, :]

        i += batch_size
        if (i + batch_size > upper_limit):
            i = 0
            # Reshuffle data
            randomized_indices = np.random.choice(np.arange(dataset_size), size=dataset_size, replace=False)


def data_generator_MedleyDB_validation(x_validation, y_validation, dataset_size, batch_size=100):
    '''
    The data generator for sending data to fit_generator for validation

    :param x_train: The validation data 
    :param y_train: The validation annotations
    :param dataset_size: The size for one epoch
    :param batch_size: The number batches in a minibatch
    :return: 
    '''
    
    i = 0
    batch_size = np.min((batch_size, dataset_size))

    while (True):

        yield x_validation[i:i + batch_size, :, :, :, :], y_validation[i:i + batch_size, :, :]
        i += batch_size
        if (i + batch_size > dataset_size):
            i = 0


def set_callbacks(save_filename, args):
    '''
    Sets the callback functions for the network training
    
    :param save_filename: Filename to be used in ModelCheckpoint 
    :param args: Input arguments
    :return: cb: List of callbacks
    '''
    
    # Callbacks
    cb = [EarlyStopping(monitor='val_loss',
                        patience=args.early_stopping_patience,
                        verbose=True),
          ModelCheckpoint('{0}/{1}.h5'.format(get_trained_model_save_path(dataset_name=args.dataset_name), save_filename),
                          monitor='val_loss',
                          save_best_only=True,
                          verbose=False),
          ReduceLROnPlateau(monitor='val_loss',
                            patience=args.reduce_LR_patience,
                            verbose=True)]

    return cb

