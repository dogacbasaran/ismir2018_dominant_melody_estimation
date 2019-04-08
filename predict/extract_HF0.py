import h5py
import parsing
import numpy as np
import os
import glob
import sys

import source_filter_model

# Global Parameters
Fs = 22050
hop = 256. / Fs
pitch_corrected = False

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


def get_path_to_dataset_audio():
    audio_path = '{0}/medleydb_audio'.format(get_path())
    return audio_path


def get_hf0_save_path(dataset_name):
    if dataset_name == 'medleydb':
        hf0_save_path = '{0}/medleydb_features/HF0s_STFT'.format(get_path())
    else:
        hf0_save_path = os.path.dirname(os.path.realpath(__file__))

    if not os.path.exists(hf0_save_path):
        os.makedirs(hf0_save_path)

    return hf0_save_path
########################################


def main(audio_fpath, verbose=True):
    '''
    The main function that calls the source filter model module to extract the parameters
    :param audio_fpath: Input audio path
    :param verbose: To print procedure
    :return:
    '''
    if 'medleydb' in audio_fpath:
        dataset_name = 'medleydb'
    else:
        dataset_name = None

    train_parameter = 'HF0_standard'

    track_name_original = os.path.basename(audio_fpath).split('.wav')[0]
    if pitch_corrected:
        track_name = track_name_original + '_corrected_pitch'
    else:
        track_name = track_name_original

    if os.path.isfile('{0}/{1}.h5'.format(get_hf0_save_path(dataset_name=dataset_name), track_name)) and False:
        print('{0} - Already processed'.format(track_name))
    else:
        print('{0} - Processing'.format(track_name_original))

        input_args = [u'{0}'.format(audio_fpath), \
                      u'--samplingRate={0}'.format(Fs), \
                      u'--hopsize={0}'.format(hop)]

        if pitch_corrected:
            input_args.append(u'--pitch-corrected')

        try:
            if train_parameter == 'HF0_standard':
                (pargs, options) = parsing.parseOptions(input_args)
                options.verbose=True

                times, HF0, HGAMMA, HPHI, WM, HM, pitch_accuracy, options = source_filter_model.main(pargs,options)

                pitch_accuracy = np.array(pitch_accuracy)

                filename = '{0}.h5'.format(track_name)
                out = h5py.File('{0}/{1}'.format(get_hf0_save_path(dataset_name=dataset_name), filename), 'w')
                out.create_dataset('HF0', HF0.shape, data=HF0)
                out.create_dataset('pitch_accuracy', pitch_accuracy.shape, data=pitch_accuracy)
                out.create_dataset('HGAMMA', HGAMMA.shape, data=HGAMMA)
                out.create_dataset('HPHI', HPHI.shape, data=HPHI)
                out.create_dataset('WM', WM.shape, data=WM)
                out.create_dataset('HM', HM.shape, data=HM)
                out.close()
            if verbose:
                print('{0} file is created'.format(track_name))
        except:
            if verbose:
                print('{0} file is not processed due to an error'.format(track_name))

        parameters_file = open('{0}/parameters.txt'.format(get_hf0_save_path(dataset_name=dataset_name)), 'w')
        parameters_file.write('Parameters:')
        parameters_file.write('    Fs: {0}'.format(Fs))
        parameters_file.write('    hop_size: {0}'.format(np.int(hop * Fs)))
        parameters_file.write('    step_notes: {0}'.format(options.stepNotes))
        parameters_file.write('    R: {0}'.format(options.R))
        parameters_file.write('    K: {0}'.format(options.K_numFilters))
        parameters_file.write('    nb_iterations: {0}'.format(options.nbiter))
        parameters_file.close()

    return np.array(HF0)


def get_medleydb_list():
    """Gets the list of songs in the Medley-dB dataset. Only the songs with annotations are chosen.

    **Returns**

    track_list: List
        The list of songs from the Medley-dB dataset."""

    path = get_path()

    track_list_full_path = glob.glob('{0}/quantized_annotations/*.h5'.format(path))
    track_list = [song.split('_quantized')[0] for song in track_list_full_path]

    return track_list


def extract_HF0_from_dataset():
    '''
    Function that computes HF0 for every track in MedleyDB
    :return:
    '''

    track_list = get_medleydb_list()
    for i, track_name in enumerate(track_list):
        audio_fpath = '{0}/{1}.wav'.format(get_path_to_dataset_audio(),track_name)
        extract_HF0_single_track(audio_fpath=audio_fpath, verbose=False)


def extract_HF0_single_track(audio_fpath, verbose=True):
    '''
    Wrapper to main function
    :param audio_fpath: Input audio path
    :param verbose: To print procedure
    :return:
    '''
    HF0 = main(audio_fpath=audio_fpath, verbose=verbose)

    return HF0

if __name__ == '__main__':

    ## Example to extract HF0 from a single track
    track_name = 'AClassicEducation_NightOwl'
    audio_fpath='{0}/{1}.wav'.format(get_path_to_dataset_audio(), track_name)
    extract_HF0_single_track(audio_fpath=audio_fpath)

    ## Example to extract HF0 from a dataset
    # extract_HF0_from_dataset()

