#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 16:25:09 2017

@author: dbasaran
"""
import numpy as np
from keras.models import Model
from keras.layers import Dense, Reshape, BatchNormalization, Bidirectional, GRU,Dropout
from keras.layers import Conv2D, LSTM, Input, TimeDistributed, Lambda, ZeroPadding3D
from keras import backend as Kb
import keras as K
import tensorflow as tf

import os
import sys

from utils import read_arguments, print_arguments, train_model, evaluate_model, \
    get_GPU_lock, release_GPU_lock, Logger


def construct_model(args):
    '''
    Construcs the CRNN model
    :param args: Input arguments
    :return: model: Constructed Model object
    '''
    kernel_coeff = 0.00001

    number_of_channels = 1
    input_shape = (args.number_of_patches, args.patch_size, args.feature_size, number_of_channels)

    inputs = Input(shape=input_shape)

    zp = ZeroPadding3D(padding=(0, 0, 2))(inputs)

    #### CNN LAYERS ####
    cnn1 = TimeDistributed(Conv2D(64, (1, 5),
                                  padding='valid',
                                  activation='relu',
                                  strides=(1, np.int(args.step_notes)),
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

    cnn5b = Reshape((args.number_of_patches * args.patch_size, -1), name='cnn5-reshape')(cnn5a)

    #### BIDIRECTIONAL RNN LAYERS ####
    if args.RNN == 'LSTM':
        rnn1 = Bidirectional(LSTM(128,
                                  kernel_regularizer=K.regularizers.l1_l2(0.0001),
                                  return_sequences=True), name='rnn1')(cnn5b)
    elif args.RNN == 'GRU':
        rnn1 = Bidirectional(GRU(128,
                                 kernel_regularizer=K.regularizers.l1_l2(0.0001),
                                 return_sequences=True), name='rnn1')(cnn5b)

    #### CLASSIFICATION (DENSE) LAYER ####
    classifier = TimeDistributed(Dense(args.number_of_classes,
                                       activation='softmax',
                                       kernel_regularizer=K.regularizers.l2(0.00001),
                                       bias_regularizer=K.regularizers.l2()), name='output')(rnn1)

    model = Model(inputs=inputs, outputs=classifier)

    if args.verbose == True:
        model.summary()

        print('{0} as RNN!'.format(args.RNN))

    return model


def main(args):
    '''
    Main function that constructs, trains and evaluates the model
    :param args: Input arguments
    :return:
    '''
    if args.verbose:
        print_arguments(args=args)

    try:
        model = construct_model(args=args)
        model_trained = train_model(model=model, args=args)
        evaluate_model(model=model_trained, args=args)
    except:
        print('Error occured in training or evaluation!!')


if __name__ == '__main__':
    # Read the arguments
    args = read_arguments(sys.argv[:])

    # Start logging
    sys.stdout = Logger(args)

    # Lock the GPU
    gpu_id_locked, comp_device = get_GPU_lock()

    # Start the main training and evaluation procedure
    with tf.device(comp_device):
        main(args=args)

    # Release the GPU
    release_GPU_lock(gpu_id_locked=gpu_id_locked)

