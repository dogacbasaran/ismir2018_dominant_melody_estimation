#!/usr/bin/python
#

# Script implementing the multiplicative rules from the following
# article:
# 
# J.-L. Durrieu, G. Richard, B. David and C. Fevotte
# Source/Filter Model for Unsupervised Main Melody
# Extraction From Polyphonic Audio Signals
# IEEE Transactions on Audio, Speech and Language Processing
# Vol. 18, No. 3, March 2010
#
# with more details and new features explained in my PhD thesis:
#
# J.-L. Durrieu,
# Automatic Extraction of the Main Melody from Polyphonic Music Signals,
# EDITE
# Institut TELECOM, TELECOM ParisTech, CNRS LTCI

# copyright (C) 2010 Jean-Louis Durrieu
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
#import time, os, sys

from numpy.random import randn
# from string import join
import matplotlib.pyplot as plt1
from sklearn.preprocessing import normalize

def db(positiveValue):
    """
    db(positiveValue)

    Returns the decibel value of the input positiveValue
    """
    return 10 * np.log10(np.abs(positiveValue))


def ISDistortion(X,Y):
    """
    value = ISDistortion(X, Y)

    Returns the value of the Itakura-Saito (IS) divergence between
    matrix X and matrix Y. X and Y should be two NumPy arrays with
    same dimension.
    """
    return np.sum((-np.log(X / Y) + (X / Y) - 1))


def plot_annotation_vs_estimation(annotation, pitch_estimation):
    data_length = np.min((len(annotation), len(pitch_estimation)))
    time_index = np.arange(data_length) * 256. / 22050

    plt1.figure('Annotation vs Estimation'), plt1.gcf().clf()
    plt1.plot(time_index, annotation[:data_length], 'o', color='black', markersize=6),
    plt1.plot(time_index, pitch_estimation[:data_length], '.', color='red', markersize=2),
    plt1.pause(1)
    plt1.show()


def compute_raw_pitch_accuracy_HF0(HF0,labels):

    numberOfClasses = 62

    sequence_length = np.min((len(labels), HF0.shape[1]))

    labels = np.array(labels[:sequence_length])
    HF0 = HF0[:, :sequence_length]

    stepNotes_class= np.int((numberOfClasses-2)/(12.*np.log2(1760./ 55.)))
    
    stepNotes_dataset = np.int((HF0.shape[0]-1)/(12.*np.log2(1760./ 55.)))
    numberOfF0 = HF0.shape[0]
    F0labels = 55. * (2 ** (np.arange(numberOfF0,dtype=np.double) / (12 * stepNotes_dataset)))
    
    freqPeaks = F0labels[np.argmax(HF0,axis=0)]
    
    y_pred = []
    
    numberOfF0 = numberOfClasses-1
    
    F0labels = 55. * (2 ** (np.arange(numberOfF0,dtype=np.double) / (12 * stepNotes_class)))
    
    # Non-linear quantization levels
    nlq_levels = F0labels * (2 ** (1./(12 * (stepNotes_class+1))))    
    nlq_levels = nlq_levels[:-1]

    # labels[labels!=0] = labels[labels!=0] - 33  # The labels are in MIDI however quantization process results from 0 to sth

    for i, freq in enumerate(freqPeaks):
        if labels[i]==0 :
            y_pred.append(0)
        else: # Apply quantization
            label_idx = np.sum(nlq_levels<freq)
            y_pred.append(label_idx)
    
    y_pred = np.array(y_pred)
    raw_pitch_accuracy = np.sum(labels[labels!=0]==y_pred[labels!=0])/np.float(np.sum(labels!=0))
    raw_chroma_accuracy = np.sum((np.abs(labels[labels!=0]-y_pred[labels!=0]))%12==0)/np.float(np.sum(labels!=0))
#    conf_mat = confusion_matrix(labels,y_pred,labels=np.arange(-1,numberOfClasses-1))
#    pitch_accuracy = np.float(np.sum(np.diag(conf_mat[1:,1:])))/np.sum(conf_mat[1:,1:])

    # plot_annotation_vs_estimation(annotation=labels, pitch_estimation=y_pred)

    return raw_pitch_accuracy, raw_chroma_accuracy


def SIMM(# the data to be fitted to:
         SX,
         # the basis matrices for the spectral combs
         WF0,
         # and for the elementary filters:
         WGAMMA,
         # the true note labels for observing HF0 performance through iterations
         labels=None,
         # number of desired filters, accompaniment spectra:
         numberOfFilters=4, numberOfAccompanimentSpectralShapes=10,
         # if any, initial amplitude matrices for 
         HGAMMA0=None, HPHI0=None,
         HF00=None,
         WM0=None, HM0=None,
         # Some more optional arguments, to control the "convergence"
         # of the algo
         numberOfIterations=1000, updateRulePower=1.0,
         stepNotes=4, 
         alpha=0.01, beta = 0.01, 
         lambdaHF0=0.00,alphaHF0=0.99,
         WMstart=-1,
         displayEvolution=False, verbose=False, makeMovie=False,
#         update_HGAMMA=True,
         computeISDistortion=False):
    """
    HGAMMA, HPHI, HF0, HM, WM, recoError =
        SIMM(SX, WF0, WGAMMA, numberOfFilters=4,
             numberOfAccompanimentSpectralShapes=10, HGAMMA0=None, HPHI0=None,
             HF00=None, WM0=None, HM0=None, numberOfIterations=1000,
             updateRulePower=1.0, stepNotes=4, 
             lambdaHF0=0.00, alphaHF0=0.99, displayEvolution=False,
             verbose=True)

    Implementation of the Smooth-filters Instantaneous Mixture Model
    (SIMM). This model can be used to estimate the main melody of a
    song, and separate the lead voice from the accompaniment, provided
    that the basis WF0 is constituted of elements associated to
    particular pitches.

    Inputs:
        SX
            the F x N power spectrogram to be approximated.
            F is the number of frequency bins, while N is the number of
            analysis frames
        WF0
            the F x NF0 basis matrix containing the NF0 source elements
        WGAMMA
            the F x P basis matrix of P smooth elementary filters
        labels
            the annotations for HF0
        numberOfFilters
            the number of filters K to be considered
        numberOfAccompanimentSpectralShapes
            the number of spectral shapes R for the accompaniment
        HGAMMA0
            the P x K decomposition matrix of WPHI on WGAMMA
        HPHI0
            the K x N amplitude matrix of the filter part of the lead
            instrument
        HF00
            the NF0 x N amplitude matrix for the source part of the lead
            instrument
        WM0
            the F x R the matrix for spectral shapes of the
            accompaniment
        HM0
            the R x N amplitude matrix associated with each of the R
            accompaniment spectral shapes
        numberOfIterations
            the number of iterations for the estimatino algorithm
        updateRulePower
            the power to which the multiplicative gradient is elevated to
        stepNotes
            the number of elements in WF0 per semitone. stepNotes=4 means
            that there are 48 elements per octave in WF0.
        lambdaHF0
            Lagrangian multiplier for the octave control
        alphaHF0
            parameter that controls how much influence a lower octave
            can have on the upper octave's amplitude.

    Outputs:
        HGAMMA
            the estimated P x K decomposition matrix of WPHI on WGAMMA
        HPHI
            the estimated K x N amplitude matrix of the filter part 
        HF0
            the estimated NF0 x N amplitude matrix for the source part
        HM
            the estimated R x N amplitude matrix for the accompaniment
        WM
            the estimate F x R spectral shapes for the accompaniment
        recoError
            the successive values of the Itakura Saito divergence
            between the power spectrogram and the spectrogram
            computed thanks to the updated estimations of the matrices.

    Please also refer to the following article for more details about
    the algorithm within this function, as well as the meaning of the
    different matrices that are involved:
        J.-L. Durrieu, G. Richard, B. David and C. Fevotte
        Source/Filter Model for Unsupervised Main Melody
        Extraction From Polyphonic Audio Signals
        IEEE Transactions on Audio, Speech and Language Processing
        Vol. 18, No. 3, March 2010
    """
    eps = 10 ** (-20)

    # renamed for convenience:
    K = numberOfFilters
    R = numberOfAccompanimentSpectralShapes
    omega = updateRulePower
    
    F, N = SX.shape
    Fwf0, NF0 = WF0.shape
    Fwgamma, P = WGAMMA.shape
    
    # Checking the sizes of the matrices
    if Fwf0 != F:
        return False # A REVOIR!!!
    if HGAMMA0 is None:
        HGAMMA0 = np.abs(randn(P, K))
    else:
        if not(isinstance(HGAMMA0,np.ndarray)): # default behaviour
            HGAMMA0 = np.array(HGAMMA0)
        Phgamma0, Khgamma0 = HGAMMA0.shape
        if Phgamma0 != P or Khgamma0 != K:
            print("Wrong dimensions for given HGAMMA0, \n")
            print("random initialization used instead")
            HGAMMA0 = np.abs(randn(P, K))

    HGAMMA = np.copy(HGAMMA0)
    
    if HPHI0 is None: # default behaviour
        HPHI = np.abs(randn(K, N))
    else:
        Khphi0, Nhphi0 = np.array(HPHI0).shape
        if Khphi0 != K or Nhphi0 != N:
            print("Wrong dimensions for given HPHI0, \n")
            print("random initialization used instead")
            HPHI = np.abs(randn(K, N))
        else:
            HPHI = np.copy(np.array(HPHI0))

    if HF00 is None:
        HF00 = np.abs(randn(NF0, N))
    else:
        if np.array(HF00).shape[0] == NF0 and np.array(HF00).shape[1] == N:
            HF00 = np.array(HF00)
        else:
            print("Wrong dimensions for given HF00, \n")
            print("random initialization used instead")
            HF00 = np.abs(randn(NF0, N))
    HF0 = np.copy(HF00)

    if HM0 is None:
        HM0 = np.abs(randn(R, N))
    else:
        if np.array(HM0).shape[0] == R and np.array(HM0).shape[1] == N:
            HM0 = np.array(HM0)
        else:
            print("Wrong dimensions for given HM0, \n")
            print("random initialization used instead")
            HM0 = np.abs(randn(R, N))
    HM = np.copy(HM0)

    if WM0 is None:
        WM0 = np.abs(randn(F, R))
    else:
        if np.array(WM0).shape[0] == F and np.array(WM0).shape[1] == R:
            WM0 = np.array(WM0)
        else:
            print("Wrong dimensions for given WM0, \n")
            print("random initialization used instead")
            WM0 = np.abs(randn(F, R))
    WM = np.copy(WM0)
    
    # Iterations to estimate the SIMM parameters:
    lambda_diag = np.sum(HGAMMA, axis=0)
    HGAMMA[:, lambda_diag>0] = HGAMMA[:, lambda_diag>0] / \
                             np.outer(np.ones(P), \
                                      lambda_diag[lambda_diag>0])
    HPHI = HPHI * np.outer(lambda_diag, np.ones(N))
    
    WPHI = np.dot(WGAMMA, HGAMMA)
#    Fwphi = WPHI.shape[0]
#    lambda_diag = np.sum(WPHI, axis=0)
#    # Normalize eack column of WPHI with lambda_k
#    WPHI[:, lambda_diag>0] = WPHI[:, lambda_diag>0] / \
#                             np.outer(np.ones(Fwphi), lambda_diag[lambda_diag>0])
#    # Transfer the lambda_k coefficient to each row of HPHI
#    HPHI[lambda_diag>0,:] = HPHI[lambda_diag>0,:] * np.outer(lambda_diag[lambda_diag>0], np.ones(N))
    SPHI = np.dot(WPHI, HPHI)
    SF0 = np.dot(WF0, HF0)
    SM = np.dot(WM, HM)
    hatSX = SF0 * SPHI + SM

    ## SX = SX + np.abs(randn(F, N)) ** 2
                                       # should not need this line
                                       # which ensures that data is not
                                       # 0 everywhere. 
    # temporary matrices
    tempNumFbyN = np.zeros([F, N])
    tempDenFbyN = np.zeros([F, N])

    # Array containing the reconstruction error after the update of each 
    # of the parameter matrices:
    recoError = np.zeros([numberOfIterations * 5 * 2 + NF0 * 2 + 1])
    recoError[0] = ISDistortion(SX, hatSX)
    if verbose:
        print("Reconstruction error at beginning: ", recoError[0])
    counterError = 1

    raw_pitch_accuracy = []
    raw_chroma_accuracy = []
    if labels is not None:
        rpa,rca = compute_raw_pitch_accuracy_HF0(HF0,labels)
        raw_pitch_accuracy.append(rpa)
        raw_chroma_accuracy.append(rca)
            
    
    error_IS = np.zeros(numberOfIterations)
    # Main loop for multiplicative updating rules:
    for n in np.arange(numberOfIterations):
        # order of re-estimation: HF0, HPHI, HM, HGAMMA, WM
        #if verbose:
        print("iteration ", n, " over ", numberOfIterations)
        error_IS[n] = ISDistortion(SX, hatSX)

        ## normal update rules without checking octaves:
        # updating HF0:
        tempNumFbyN = (SPHI * SX) / np.maximum(hatSX ** 2, eps)
        tempDenFbyN = SPHI / np.maximum(hatSX, eps)

#                # This to enable octave control
#                HF0[np.arange(12 * stepNotes, NF0), :] \
#                   = HF0[np.arange(12 * stepNotes, NF0), :] \
#                     * (np.dot(WF0[:, np.arange(12 * stepNotes,
#                                                NF0)].T, tempNumFbyN) \
#                        / np.maximum(
#                    np.dot(WF0[:, np.arange(12 * stepNotes, NF0)].T,
#                           tempDenFbyN) \
#                    + lambdaHF0 * (- (alphaHF0 - 1.0) \
#                                   / np.maximum(HF0[
#                    np.arange(12 * stepNotes, NF0), :], eps) \
#                                   + HF0[
#                    np.arange(NF0 - 12 * stepNotes), :]),
#                    eps)) ** omega
#        
#                HF0[np.arange(12 * stepNotes), :] \
#                   = HF0[np.arange(12 * stepNotes), :] \
#                     * (np.dot(WF0[:, np.arange(12 * stepNotes)].T,
#                              tempNumFbyN) /
#                       np.maximum(
#                        np.dot(WF0[:, np.arange(12 * stepNotes)].T,
#                               tempDenFbyN), eps)) ** omega

        HF0 = HF0 * (np.dot(WF0.T, tempNumFbyN) /
                         np.maximum(np.dot(WF0.T, tempDenFbyN), eps)) ** omega
        if labels is not None:
            rpa,rca = compute_raw_pitch_accuracy_HF0(HF0,labels)
            raw_pitch_accuracy.append(rpa)
            raw_chroma_accuracy.append(rca)
                
        SF0 = np.maximum(np.dot(WF0, HF0),eps)

        hatSX = np.maximum(SF0 * SPHI + SM,eps)

        if computeISDistortion:
            recoError[counterError] = ISDistortion(SX, hatSX)

        if verbose:
            print("Reconstruction error difference after HF0   : ", recoError[counterError] - recoError[counterError - 1])
        counterError += 1

        # updating HPHI
        tempNumFbyN = (SF0 * SX) / np.maximum(hatSX ** 2, eps)
        tempDenFbyN = SF0 / np.maximum(hatSX, eps)


        HPHI = HPHI * (np.dot(WPHI.T, tempNumFbyN) / \
                       np.maximum(np.dot(WPHI.T, tempDenFbyN), eps)) ** omega

#################################################################################    
#            sumHPHI = np.sum(HPHI, axis=0)
#            HPHI[:, sumHPHI>0] = HPHI[:, sumHPHI>0] / \
#                                 np.outer(np.ones(K), sumHPHI[sumHPHI>0])
#            HF0 = HF0 * np.outer(np.ones(NF0), sumHPHI)
#    
#            SF0 = np.maximum(np.dot(WF0, HF0), eps)
#################################################################################
        SPHI = np.maximum(np.dot(WPHI, HPHI), eps)
        hatSX = np.maximum(SF0 * SPHI + SM, eps)

        if computeISDistortion:
            recoError[counterError] = ISDistortion(SX, hatSX)

        if verbose:
            print("Reconstruction error difference after HPHI  : ", \
                  recoError[counterError] - recoError[counterError - 1])
        counterError += 1
        
        # updating HM
        tempNumFbyN = SX / np.maximum(hatSX ** 2, eps)
        tempDenFbyN = 1 / np.maximum(hatSX, eps)
        HM = np.maximum(HM * (np.dot(WM.T, tempNumFbyN) / \
                              np.maximum(np.dot(WM.T, tempDenFbyN), eps)) ** \
                        omega, eps)

        SM = np.maximum(np.dot(WM, HM), eps)
        hatSX = np.maximum(SF0 * SPHI + SM, eps)

        if computeISDistortion:
            recoError[counterError] = ISDistortion(SX, hatSX)

        if verbose:
            print("Reconstruction error difference after HM    : ", \
                  recoError[counterError] - recoError[counterError - 1])
        counterError += 1

        # updating HGAMMA
        tempNumFbyN = (SF0 * SX) / np.maximum(hatSX ** 2, eps)
        tempDenFbyN = SF0 / np.maximum(hatSX, eps)
        HGAMMA = np.maximum(\
                 HGAMMA * (np.dot(WGAMMA.T, \
                                  np.dot(tempNumFbyN, HPHI.T)) / \
                           np.maximum(\
                               np.dot(WGAMMA.T, \
                                      np.dot(tempDenFbyN, HPHI.T)),
                               eps)) ** \
                 omega, eps)

#            WPHI = np.maximum(np.dot(WGAMMA, HGAMMA), eps)
#            lambda_diag = np.sum(WPHI, axis=0)
#            # Normalize eack column of WPHI with lambda_k
#            WPHI[:, lambda_diag>0] = WPHI[:, lambda_diag>0] / \
#                                     np.outer(np.ones(Fwphi), lambda_diag[lambda_diag>0])
#            # Transfer the lambda_k coefficient to each row of HPHI
#            HPHI[lambda_diag>0,:] = HPHI[lambda_diag>0,:] * np.outer(lambda_diag[lambda_diag>0], np.ones(N))

        lambda_diag = np.sum(HGAMMA, axis=0)
        HGAMMA[:, lambda_diag>0] = HGAMMA[:, lambda_diag>0] / \
                                 np.outer(np.ones(P), \
                                          lambda_diag[lambda_diag>0])
        HPHI = HPHI * np.outer(lambda_diag, np.ones(N))


#################################################################################
#            sumHPHI = np.sum(HPHI, axis=0)
#            HPHI[:, sumHPHI>0] = HPHI[:, sumHPHI>0] / np.outer(np.ones(K), sumHPHI[sumHPHI>0])
#            HF0 = HF0 * np.outer(np.ones(NF0), sumHPHI)
#            
#            SF0 = np.maximum(np.dot(WF0, HF0), eps)
#################################################################################

        WPHI = np.maximum(np.dot(WGAMMA, HGAMMA), eps)
        SPHI = np.maximum(np.dot(WPHI, HPHI), eps)
        hatSX = np.maximum(SF0 * SPHI + SM, eps)

        if computeISDistortion:
            recoError[counterError] = ISDistortion(SX, hatSX)

        if verbose:
            print("Reconstruction error difference after HGAMMA: ",recoError[counterError] - recoError[counterError - 1])

        counterError += 1

        # updating WM, after a certain number of iterations (here, after 1 iteration)
        if n > WMstart: # this test can be used such that WM is updated only
                  # after a certain number of iterations
            tempNumFbyN = SX / np.maximum(hatSX ** 2, eps)
            tempDenFbyN = 1 / np.maximum(hatSX, eps)
            WM = np.maximum(WM * (np.dot(tempNumFbyN, HM.T) /
                                  np.maximum(np.dot(tempDenFbyN, HM.T),
                                             eps)) ** omega, eps)

            sumWM = np.sum(WM, axis=0)
            WM[:, sumWM>0] = (WM[:, sumWM>0] /
                              np.outer(np.ones(F),sumWM[sumWM>0]))
            HM = HM * np.outer(sumWM, np.ones(N))

            SM = np.maximum(np.dot(WM, HM), eps)
            hatSX = np.maximum(SF0 * SPHI + SM, eps)

            if computeISDistortion:
                recoError[counterError] = ISDistortion(SX, hatSX)

            if verbose:
                print("Reconstruction error difference after WM    : ",recoError[counterError] - recoError[counterError - 1])
            counterError += 1

    activations = {}
    activations['HGAMMA'] = HGAMMA
    activations['HPHI'] = HPHI
    activations['HF0'] = HF0
    activations['HM'] = HM
    activations['WM'] = WM
    
    return activations, raw_pitch_accuracy, recoError