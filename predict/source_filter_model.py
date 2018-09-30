import sys
import numpy as np
import SIMM
import os
import h5py
from sklearn.metrics import confusion_matrix
import pandas as pd
import librosa

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


def db(val):
    """
    db(positiveValue)

    Returns the decibel value of the input positiveValue
    """
    return 10 * np.log10(val)


def nextpow2(i):
    """
    Find 2^n that is equal to or greater than.

    code taken from the website:
    http://www.phys.uu.nl/~haque/computing/WPark_recipes_in_python.html
    """
    n = 2
    while n < i:
        n = n * 2
    return n


def ISDistortion(X,Y):
    """
    value = ISDistortion(X, Y)

    Returns the value of the Itakura-Saito (IS) divergence between
    matrix X and matrix Y. X and Y should be two NumPy arrays with
    same dimension.
    """
    return sum((-np.log(X / Y) + (X / Y) - 1))


# DEFINING SOME WINDOW FUNCTIONS
def sinebell(lengthWindow):
    """
    window = sinebell(lengthWindow)

    Computes a "sinebell" window function of length L=lengthWindow

    The formula is:
        window(t) = sin(pi * t / L), t = 0..L-1
    """
    window = np.sin((np.pi * (np.arange(lengthWindow))) \
                    / (1.0 * lengthWindow))
    return window


def hann(args):
    """
    window = hann(args)

    Computes a Hann window, with NumPy's function hanning(args).
    """
    return np.hanning(args)


# FUNCTIONS FOR TIME-FREQUENCY REPRESENTATION
def stft(data, window=sinebell(2048), hopsize=256.0, nfft=2048.0, fs=44100.0):
    """
    X, F, N = stft(data, window=sinebell(2048), hopsize=1024.0,
                   nfft=2048.0, fs=44100)

    Computes the short time Fourier transform (STFT) of data.

    Inputs:
        data                  : one-dimensional time-series to be
                                analyzed
        window=sinebell(2048) : analysis window
        hopsize=1024.0        : hopsize for the analysis
        nfft=2048.0           : number of points for the Fourier
                                computation (the user has to provide an
                                even number)
        fs=44100.0            : sampling rate of the signal

    Outputs:
        X                     : STFT of data
        F                     : values of frequencies at each Fourier
                                bins
        N                     : central time at the middle of each
                                analysis window
    """

    # window defines the size of the analysis windows
    lengthWindow = window.size

    # !!! adding zeros to the beginning of data, such that the first
    # window is centered on the first sample of data
    data = np.concatenate((np.zeros(int(lengthWindow / 2.0)),data))
    lengthData = data.size

    # adding one window for the last frame (same reason as for the
    # first frame)
    numberFrames = int(np.ceil((lengthData - lengthWindow) / hopsize \
                           + 1) + 1)
    newLengthData = (numberFrames - 1) * hopsize + lengthWindow
    # zero-padding data such that it holds an exact number of frames
    data = np.concatenate((data, np.zeros([int(newLengthData - lengthData)])))

    # the output STFT has nfft/2+1 rows. Note that nfft has to be an
    # even number (and a power of 2 for the fft to be fast)
    numberFrequencies = int(nfft / 2.0 + 1)

    STFT = np.zeros([numberFrequencies, numberFrames], dtype=complex)

    for n in np.arange(numberFrames):
        beginFrame = int(n * hopsize)
        endFrame = beginFrame + lengthWindow
        frameToProcess = window * data[beginFrame:endFrame]
        STFT[:,n] = np.fft.rfft(frameToProcess, nfft);

    F = np.arange(numberFrequencies) / nfft * fs
    N = np.arange(numberFrames) * hopsize / fs

    return STFT, F, N


def istft(X, window=sinebell(2048), hopsize=256.0, nfft=2048.0):
    """
    data = istft(X, window=sinebell(2048), hopsize=256.0, nfft=2048.0)

    Computes an inverse of the short time Fourier transform (STFT),
    here, the overlap-add procedure is implemented.

    Inputs:
        X                     : STFT of the signal, to be "inverted"
        window=sinebell(2048) : synthesis window
                                (should be the "complementary" window
                                for the analysis window)
        hopsize=1024.0        : hopsize for the analysis
        nfft=2048.0           : number of points for the Fourier
                                computation
                                (the user has to provide an even number)

    Outputs:
        data                  : time series corresponding to the given
                                STFT the first half-window is removed,
                                complying with the STFT computation
                                given in the function 'stft'
    """
    lengthWindow = np.array(window.size)
    numberFrequencies, numberFrames = np.array(X.shape)
    lengthData = hopsize * (numberFrames - 1) + lengthWindow

    data = np.zeros(lengthData)
    for n in np.arange(numberFrames):
        beginFrame = n * hopsize
        endFrame = beginFrame + lengthWindow
        frameTMP = np.fft.irfft(X[:,n], nfft)
        frameTMP = frameTMP[:lengthWindow]
        data[beginFrame:endFrame] = data[beginFrame:endFrame] \
                                    + window * frameTMP

    # remove the extra bit before data that was - supposedly - added
    # in the stft computation:
    data = data[(lengthWindow / 2.0):]
    return data


# DEFINING THE FUNCTIONS TO CREATE THE 'BASIS' WF0
def generate_WF0_chirped(minF0, maxF0, Fs, Nfft=2048, stepNotes=4, \
                         lengthWindow=2048, Ot=0.5, perF0=2, \
                         depthChirpInSemiTone=0.5, loadWF0=True,
                         analysisWindow='hanning'):
    """
    F0Table, WF0 = generate_WF0_chirped(minF0, maxF0, Fs, Nfft=2048,
                                        stepNotes=4, lengthWindow=2048,
                                        Ot=0.5, perF0=2,
                                        depthChirpInSemiTone=0.5)

    Generates a 'basis' matrix for the source part WF0, using the
    source model KLGLOTT88, with the following I/O arguments:
    Inputs:
        minF0                the minimum value for the fundamental
                             frequency (F0)
        maxF0                the maximum value for F0
        Fs                   the desired sampling rate
        Nfft                 the number of bins to compute the Fourier
                             transform
        stepNotes            the number of F0 per semitone
        lengthWindow         the size of the window for the Fourier
                             transform
        Ot                   the glottal opening coefficient for
                             KLGLOTT88
        perF0                the number of chirps considered per F0
                             value
        depthChirpInSemiTone the maximum value, in semitone, of the
                             allowed chirp per F0

    Outputs:
        F0Table the vector containing the values of the fundamental
                frequencies in Hertz (Hz) corresponding to the
                harmonic combs in WF0, i.e. the columns of WF0
        WF0     the basis matrix, where each column is a harmonic comb
                generated by KLGLOTT88 (with a sinusoidal model, then
                transformed into the spectral domain)
    """
    # generating a filename to keep data:
    filename = str('').join(['wf0_',
                             '_minF0-', str(minF0),
                             '_maxF0-', str(maxF0),
                             '_Fs-', str(Fs),
                             '_Nfft-', str(Nfft),
                             '_stepNotes-', str(stepNotes),
                             '_Ot-', str(Ot),
                             '_perF0-', str(perF0),
                             '_depthChirp-', str(depthChirpInSemiTone),
                             '_analysisWindow-', analysisWindow,
                             '.npz'])

    if os.path.isfile(filename) and loadWF0:
        struc = np.load(filename)
        return struc['F0Table'], struc['WF0']


    # converting to double arrays:
    minF0=np.double(minF0)
    maxF0=np.double(maxF0)
    Fs=np.double(Fs)
    stepNotes=np.double(stepNotes)

    # computing the F0 table:
    numberOfF0 = np.int(np.ceil(12.0 * stepNotes * np.log2(maxF0 / minF0)) + 1)
    F0Table=minF0 * (2 ** (np.arange(numberOfF0,dtype=np.double) \
                           / (12 * stepNotes)))

    numberElementsInWF0 = np.int(numberOfF0 * perF0)

    # computing the desired WF0 matrix
    WF0 = np.zeros([Nfft, numberElementsInWF0],dtype=np.double)
    for fundamentalFrequency in np.arange(numberOfF0):
        odgd, odgdSpec = \
              generate_ODGD_spec(F0Table[fundamentalFrequency], Fs, \
                                 Ot=Ot, lengthOdgd=lengthWindow, \
                                 Nfft=Nfft, t0=0.0,\
                                 analysisWindowType=analysisWindow) # 20100924 trying with hann window
        WF0[:,fundamentalFrequency * perF0] = np.abs(odgdSpec) ** 2
        for chirpNumber in np.arange(perF0 - 1):
            F2 = F0Table[fundamentalFrequency] \
                 * (2 ** ((chirpNumber + 1.0) * depthChirpInSemiTone \
                          / (12.0 * (perF0 - 1.0))))
            # F0 is the mean of F1 and F2.
            F1 = 2.0 * F0Table[fundamentalFrequency] - F2
            odgd, odgdSpec = \
                  generate_ODGD_spec_chirped(F1, F2, Fs, \
                                             Ot=Ot, \
                                             lengthOdgd=lengthWindow, \
                                             Nfft=Nfft, t0=0.0)
            WF0[:,fundamentalFrequency * perF0 + chirpNumber + 1] = \
                                       np.abs(odgdSpec) ** 2

    np.savez(filename, F0Table=F0Table, WF0=WF0)

    return F0Table, WF0

def generate_ODGD_spec(F0, Fs, lengthOdgd=2048, Nfft=2048, Ot=0.5, \
                       t0=0.0, analysisWindowType='sinebell'):
    """
    generateODGDspec:

    generates a waveform ODGD and the corresponding spectrum,
    using as analysis window the -optional- window given as
    argument.
    """

    # converting input to double:
    F0 = np.double(F0)
    Fs = np.double(Fs)
    Ot = np.double(Ot)
    t0 = np.double(t0)

    # compute analysis window of given type:
    if analysisWindowType=='sinebell':
        analysisWindow = sinebell(lengthOdgd)
    else:
        if analysisWindowType=='hanning' or \
               analysisWindowType=='hanning':
            analysisWindow = hann(lengthOdgd)

    # maximum number of partials in the spectral comb:
    partialMax = np.floor((Fs / 2) / F0)

    # Frequency numbers of the partials:
    frequency_numbers = np.arange(1,partialMax + 1)

    # intermediate value
    temp_array = 1j * 2.0 * np.pi * frequency_numbers * Ot

    # compute the amplitudes for each of the frequency peaks:
    amplitudes = F0 * 27 / 4 \
                 * (np.exp(-temp_array) \
                    + (2 * (1 + 2 * np.exp(-temp_array)) / temp_array) \
                    - (6 * (1 - np.exp(-temp_array)) \
                       / (temp_array ** 2))) \
                       / temp_array

    # Time stamps for the time domain ODGD
    timeStamps = np.arange(lengthOdgd) / Fs + t0 / F0

    # Time domain odgd:
    odgd = np.exp(np.outer(2.0 * 1j * np.pi * F0 * frequency_numbers, \
                           timeStamps)) \
                           * np.outer(amplitudes, np.ones(lengthOdgd))
    odgd = np.sum(odgd, axis=0)

    # spectrum:
    odgdSpectrum = np.fft.fft(np.real(odgd * analysisWindow), n=Nfft)

    return odgd, odgdSpectrum


def generate_ODGD_spec_chirped(F1, F2, Fs, lengthOdgd=2048, Nfft=2048, \
                               Ot=0.5, t0=0.0, \
                               analysisWindowType='sinebell'):
    """
    generateODGDspecChirped:

    generates a waveform ODGD and the corresponding spectrum,
    using as analysis window the -optional- window given as
    argument.
    """

    # converting input to double:
    F1 = np.double(F1)
    F2 = np.double(F2)
    F0 = np.double(F1 + F2) / 2.0
    Fs = np.double(Fs)
    Ot = np.double(Ot)
    t0 = np.double(t0)

    # compute analysis window of given type:
    if analysisWindowType == 'sinebell':
        analysisWindow = sinebell(lengthOdgd)
    else:
        if analysisWindowType == 'hanning' or \
               analysisWindowType == 'hann':
            analysisWindow = hann(lengthOdgd)

    # maximum number of partials in the spectral comb:
    partialMax = np.floor((Fs / 2) / np.max(F1, F2))

    # Frequency numbers of the partials:
    frequency_numbers = np.arange(1,partialMax + 1)

    # intermediate value
    temp_array = 1j * 2.0 * np.pi * frequency_numbers * Ot

    # compute the amplitudes for each of the frequency peaks:
    amplitudes = F0 * 27 / 4 * \
                 (np.exp(-temp_array) \
                  + (2 * (1 + 2 * np.exp(-temp_array)) / temp_array) \
                  - (6 * (1 - np.exp(-temp_array)) \
                     / (temp_array ** 2))) \
                  / temp_array

    # Time stamps for the time domain ODGD
    timeStamps = np.arange(lengthOdgd) / Fs + t0 / F0

    # Time domain odgd:
    odgd = np.exp(2.0 * 1j * np.pi \
                  * (np.outer(F1 * frequency_numbers,timeStamps) \
                     + np.outer((F2 - F1) \
                                * frequency_numbers,timeStamps ** 2) \
                     / (2 * lengthOdgd / Fs))) \
                     * np.outer(amplitudes,np.ones(lengthOdgd))
    odgd = np.sum(odgd,axis=0)

    # spectrum:
    odgdSpectrum = np.fft.fft(np.real(odgd * analysisWindow), n=Nfft)

    return odgd, odgdSpectrum


def generateHannBasis(numberFrequencyBins, sizeOfFourier, Fs, \
                      frequencyScale='linear', numberOfBasis=20, \
                      overlap=.75):
    isScaleRecognized = False
    if frequencyScale == 'linear':
        # number of windows generated:
        numberOfWindowsForUnit = np.ceil(1.0 / (1.0 - overlap))
        # recomputing the overlap to exactly fit the entire
        # number of windows:
        overlap = 1.0 - 1.0 / np.double(numberOfWindowsForUnit)
        # length of the sine window - that is also to say: bandwidth
        # of the sine window:
        lengthSineWindow = np.ceil(numberFrequencyBins \
                                   / ((1.0 - overlap) \
                                      * (numberOfBasis - 1) + 1 \
                                      - 2.0 * overlap))
        # even window length, for convenience:
        lengthSineWindow = int(2.0 * np.floor(lengthSineWindow / 2.0))

        # for later compatibility with other frequency scales:
        mappingFrequency = np.arange(numberFrequencyBins)

        # size of the "big" window
        sizeBigWindow = int(2.0 * numberFrequencyBins)

        # centers for each window
        ## the first window is centered at, in number of window:
        firstWindowCenter = -numberOfWindowsForUnit + 1
        ## and the last is at
        lastWindowCenter = numberOfBasis - numberOfWindowsForUnit + 1
        ## center positions in number of frequency bins
        sineCenters = np.round(\
            np.arange(firstWindowCenter, lastWindowCenter) \
            * (1 - overlap) * np.double(lengthSineWindow) \
            + lengthSineWindow / 2.0)

        # For future purpose: to use different frequency scales
        isScaleRecognized = True

    # For frequency scale in logarithm (such as ERB scales)
    if frequencyScale == 'log':
        isScaleRecognized = False

    # checking whether the required scale is recognized
    if not(isScaleRecognized):
        print("The desired feature for frequencyScale is not recognized yet...")
        return 0

    # the shape of one window:
    prototypeSineWindow = hann(lengthSineWindow)
    # adding zeroes on both sides, such that we do not need to check
    # for boundaries
    bigWindow = np.zeros([sizeBigWindow * 2, 1])
    bigWindow[int(sizeBigWindow - lengthSineWindow / 2.0):\
              int(sizeBigWindow + lengthSineWindow / 2.0)] \
              = np.vstack(prototypeSineWindow)

    WGAMMA = np.zeros([numberFrequencyBins, numberOfBasis])

    for p in np.arange(numberOfBasis):
        WGAMMA[:, p] = np.hstack(bigWindow[np.int32(mappingFrequency \
                                                    - sineCenters[p] \
                                                    + sizeBigWindow)])

    return WGAMMA

def main(args,options):

    # Compulsory option: name of the input file:
    input_audio_file = args[0]

    if input_audio_file[-4:] != ".wav":
        raise ValueError("File not WAV file? Only WAV format support, for now...")

    data, Fs = librosa.load(input_audio_file, sr=options.Fs)

    tuning_fraction = librosa.core.estimate_tuning(y=data, sr=Fs)
    print('Tuning fraction: {0}'.format(tuning_fraction))

    track_name = input_audio_file.split('/')[-1].split('.')[0]
    if 'medleydb' in input_audio_file:
        labels = get_labels(track_name=track_name)
    else:
        dataset_name = None
        labels = None

    is_stereo = True
    if data.shape[0] == data.size: # data is multi-channel
        #print "The audio file is not stereo."
        #print "The audio file is not stereo. Making stereo out of mono."
        #print "(You could also try the older separateLead.py...)"
        is_stereo = False
        # data = np.vstack([data,data]).T
        # raise ValueError("number of dimensions of the input not 2")
    if is_stereo and data.shape[1] != 2:
        print("The data is multichannel, but not stereo... \n")
        print("Unfortunately this program does not scale well. Data is \n")
        print("reduced to its 2 first channels.\n")
        data = data[:,0:2]


    # Processing the options:
    windowSizeInSamples = nextpow2(np.round(options.windowSize * Fs))

    hopsize = np.round(options.hopsize * Fs)
    #if hopsize != windowSizeInSamples/8:
    #    #print "Overriding given hopsize to use 1/8th of window size"
    #    #hopsize = windowSizeInSamples/8
    #    warnings.warn("Chosen hopsize: "+str(hopsize)+\
    #                  ", while windowsize: "+str(windowSizeInSamples))

    options.hopsizeInSamples = hopsize
    if options.fourierSize is None:
        NFT = windowSizeInSamples
    else:
        NFT = options.fourierSize

    # number of iterations for each parameter estimation step:
    niter = options.nbiter
    # number of spectral shapes for the accompaniment
    R = int(options.R)

    eps = 10 ** -9

    if is_stereo:
        XR, F, N = stft(data[:,0], fs=Fs, hopsize=hopsize,window=sinebell(windowSizeInSamples), nfft=NFT)
        XL, F, N = stft(data[:,1], fs=Fs, hopsize=hopsize,window=sinebell(windowSizeInSamples), nfft=NFT)
        SX = np.maximum((0.5*np.abs(XR+XL)) ** 2, eps)
    else: # data is mono
        X, F, N = stft(data, fs=Fs, hopsize=hopsize,window=sinebell(windowSizeInSamples), nfft=NFT)
        SX = np.maximum(np.abs(X) ** 2, eps)

    del data, F, N

    # minimum and maximum F0 in glottal source spectra dictionary
    if options.pitch_corrected:
        minF0 = options.minF0 * 2**(tuning_fraction/12)
        maxF0 = options.maxF0 * 2**(tuning_fraction/12)
    else:
        minF0 = options.minF0
        maxF0 = options.maxF0

    F, N = SX.shape
    stepNotes = options.stepNotes # this is the number of F0s within one semitone

    K = int(options.K_numFilters) # number of spectral shapes for the filter part
    P = int(options.P_numAtomFilters) # number of elements in dictionary of smooth filters
    chirpPerF0 = 1 # number of chirped spectral shapes between each F0
    # this feature should be further studied before
    # we find a good way of doing that.

    # Create the harmonic combs, for each F0 between minF0 and maxF0:
    F0Table, WF0 = \
             generate_WF0_chirped(minF0, maxF0, Fs, Nfft=NFT, \
                                  stepNotes=stepNotes, \
                                  lengthWindow=windowSizeInSamples, Ot=0.25, \
                                  perF0=chirpPerF0, \
                                  depthChirpInSemiTone=.15, loadWF0=True,\
                                  analysisWindow='sinebell')
    WF0 = WF0[0:F, :] # ensure same size as SX
    NF0 = F0Table.size # number of harmonic combs
    # Normalization:
    WF0 = WF0 / np.outer(np.ones(F), np.amax(WF0, axis=0))

    # Create the dictionary of smooth filters, for the filter part of
    # the lead isntrument:
    WGAMMA = generateHannBasis(F, NFT, Fs=Fs, frequencyScale='linear', \
                               numberOfBasis=P, overlap=.75)


    ## section to estimate the melody, on monophonic algo:
    # First round of parameter estimation:

    activations, pitch_accuracy, recoError1 = SIMM.SIMM(
        # the data to be fitted to:
        SX,
        # the basis matrices for the spectral combs
        WF0,
        # and for the elementary filters:
        WGAMMA,
        # the true note labels for observing HF0 performance through iterations
        labels=labels,
        # number of desired filters, accompaniment spectra:
        numberOfFilters=K, numberOfAccompanimentSpectralShapes=R,
        # putting only 2 elements in accompaniment for a start...
        # if any, initial amplitude matrices for
        HGAMMA0=None, HPHI0=None,
        HF00=None,
        WM0=None, HM0=None,
        # Some more optional arguments, to control the "convergence"
        # of the algo
        numberOfIterations=niter, updateRulePower=1.,
        stepNotes=stepNotes,
        alpha = options.alpha,
        beta = options.beta,
        lambdaHF0 = 0.0 / (1.0 * SX.max()), alphaHF0=0.9,
        verbose=False)

    HGAMMA = activations['HGAMMA']
    HPHI = activations['HPHI']
    HF0 = activations['HF0']
    HM = activations['HM']
    WM = activations['WM']

    times = np.array([np.arange(N) * hopsize / np.double(Fs)])

    print("Done!")

    return times[0], HF0, HGAMMA, HPHI, WM, HM, pitch_accuracy, options

if __name__ == '__main__':
    main(sys.argv[1:])


