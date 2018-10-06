import mir_eval

from sklearn.preprocessing import LabelBinarizer, normalize

import argparse
import glob
import os
import numpy as np
import json
import h5py
import sys


def parse_input(input_args):
    parser = argparse.ArgumentParser()

    parser.add_argument("-v", "--verbose",
                        action="store_true",
                        help="Print the procedure")

    parser.add_argument("--sampling-rate",
                        action="store",
                        dest="SR", type=int,
                        help="Sampling rate",
                        default=22050)

    parser.add_argument("--hop-size",
                        action="store",
                        dest="hop_size", type=int,
                        help="Sampling rate",
                        default=256)

    parser.add_argument("--shuffle-data",
                        action="store_true",
                        default=False,
                        help="Shuffle the dataset")

    parser.add_argument("--splits-file-path",
                        action="store",
                        dest="split_file_path", type=str,
                        help="The path to a predefined split file",
                        default=None)

    parser.add_argument("--segment_length",
                        action="store",
                        dest="segment_length", type=int,
                        help="The segment length",
                        default=500)

    parser.add_argument("--train-set-ratio",
                        action="store",
                        dest="validation_set_ratio", type=float,
                        help="The ratio of the validation set",
                        default=0.17)

    parser.add_argument("--validation-set-ratio",
                        action="store",
                        dest="train_set_ratio", type=float,
                        help="The ratio of the training set",
                        default=0.63)

    parser.add_argument("--number-of-bins",
                        action="store",
                        dest="number_of_bins", type=int,
                        help="The number of frequency bins in the Fourier transform",
                        default=1025)

    parser.add_argument("--step-notes",
                        action="store",
                        dest="step_notes", type=int,
                        help="The number of F0s in between semitones",
                        default=5)

    parser.add_argument("--number-of-classes",
                        action="store",
                        dest="number_of_classes", type=int,
                        help="The number of output classes: 1 class for non-melody and other classes for target F0s",
                        default=62)

    parser.add_argument("--dataset-number",
                        action="store",
                        dest="dataset_number", type=int,
                        help="The number of the dataset i.e., dataset 1, dataset 2 etc.",
                        default=1)

    parser.add_argument("--feature-size",
                        action="store",
                        dest="feature_size", type=int,
                        help="The feature size of the input (Default for step_size=5, minFreq=55, maxFreq=1760)",
                        default=301)

    parser.add_argument("--feature-type",
                        action="store",
                        dest="feature_type", type=str,
                        help="The feature to be used for dataset generation HF0/CQT",
                        default='HF0')

    parser.add_argument("--augment-data",
                        action="store_true",
                        default=False,
                        help="Use augmentation if this option is assigned to True")

    parser.add_argument("--corrected-pitch",
                        action="store_true",
                        default=False,
                        help="Use HF0s that are extracted using pitch correction on the WF0 basis")

    parser.add_argument("--save-path",
                        action="store",
                        dest="split_save_path", type=str,
                        help="The path to save the dataset splits",
                        default=None)

    parser.add_argument("--dataset-name",
                        action="store",
                        dest="dataset_name", type=str,
                        help="The name of the dataset medleydb/jazzomat",
                        default='medleydb')

    args = parser.parse_args(input_args)

    return args


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


def get_medleydb_list():
    """Gets the list of songs in the Medley-dB dataset. Only the songs with annotations are chosen.

    **Returns**

    track_list: List
        The list of songs from the Medley-dB dataset."""

    path = get_path()

    track_list_full_path = glob.glob('{0}/quantized_annotations/*.h5'.format(path))
    track_list = [song.split('/')[-1].split('.')[0] for song in track_list_full_path]

    return track_list

def get_cqt_path():
    path = '{0}/medleydb_features/CQT'.format(get_path())

    return path


def get_dataset_track_list():
    track_list = get_medleydb_list()

    return track_list


def generate_train_validation_set_lists(dataset_name, train_set_ratio, test_set):
    """Creates random training and validation set lists from the Medley-dB
    dataset with the given ratios. No song from the same artist can exist
    in different sets!

    **Parameters**
    train_set_ratio: Float
        The ratio for the train set
    test_set: List
        The list for the test set

    **Returns**
    train_set: List
        List that holds the files in the training set
    validation_set: List
        List that holds the files in the validation set
    testSet: List
        List that holds the files in the test set"""

    track_list = get_dataset_track_list()
    train_set_length = np.round(len(track_list) * train_set_ratio)

    for track in test_set:
        try:
            track_list.remove(track)
        except:
            print('Ignoring file: {0}'.format(track))
            test_set.remove(track)

    artists = {}

    for track in track_list:
        artist_name = track.split('_')[0]
        track_name = track.split(artist_name)[1]
        if artist_name in artists.keys():
            artists[artist_name].append(track_name)
        else:
            artists[artist_name] = []
            artists[artist_name].append(track_name)

    artist_list = list(artists.keys())

    assigned_train = 0
    train_set = []
    assigned_artist_list = []
    if dataset_name == 'medleydb':
        # In medleydb, a special case with artist name 'MusicDelta' occurs because there are 35
        # songs recorded by them. That's why, they are assigned to training set deterministically!!
        artist_name = 'MusicDelta'
        number_of_tracks = len(artists[artist_name])
        for track_name in artists[artist_name]:
            track = '{0}_{1}'.format(artist_name, track_name)
            train_set.append(track)
        assigned_train += number_of_tracks
        artist_list.remove(artist_name)

    # Assigning rest of the train_set
    rnd_perm = np.random.permutation(len(artist_list))
    for i in rnd_perm:
        artist_name = artist_list[i]
        number_of_tracks = len(artists[artist_name])
        if (assigned_train + number_of_tracks) < train_set_length:
            assigned_artist_list.append(artist_name)
            for track_name in artists[artist_name]:
                track = '{0}_{1}'.format(artist_name, track_name)
                train_set.append(track)
            assigned_train += number_of_tracks

            if assigned_train == train_set_length:
                break

    # Remove the artists that are assigned validation_set
    for artist_name in assigned_artist_list:
        artist_list.remove(artist_name)

    # The rest of the list is assigned for validation
    validation_set = []
    for artist_name in artist_list:
        for track_name in artists[artist_name]:
            track = '{0}_{1}'.format(artist_name, track_name)
            validation_set.append(track)

    return train_set, validation_set, test_set


def generate_train_validation_test_set_lists(dataset_name, train_set_ratio, validation_set_ratio):
    track_list = get_dataset_track_list()
    train_set_length = np.round(len(track_list) * train_set_ratio)
    validation_set_length = np.round(len(track_list) * validation_set_ratio)

    artists = {}

    for track in track_list:
        artist_name = track.split('_')[0]
        track_name = track.split(artist_name + '_')[1]
        if artist_name in artists.keys():
            artists[artist_name].append(track_name)
        else:
            artists[artist_name] = []
            artists[artist_name].append(track_name)

    artist_list = list(artists.keys())

    assigned_train = 0
    assigned_validation = 0
    train_set = []
    validation_set = []
    test_set = []
    assigned_artist_list = []
    if dataset_name == 'medleydb':
        # In medleydb, a special case with artist name 'MusicDelta' occurs because there are 35
        # songs recorded by them. That's why, they are assigned to training set deterministically!!
        artist_name = 'MusicDelta'
        number_of_tracks = len(artists[artist_name])
        for track_name in artists[artist_name]:
            track = '{0}_{1}'.format(artist_name, track_name)
            train_set.append(track)
        assigned_train += number_of_tracks
        artist_list.remove(artist_name)

    # Assigning rest of the train_set
    rnd_perm = np.random.permutation(len(artist_list))
    for i in rnd_perm:
        artist_name = artist_list[i]
        number_of_tracks = len(artists[artist_name])
        if (assigned_train + number_of_tracks) < train_set_length:
            assigned_artist_list.append(artist_name)
            for track_name in artists[artist_name]:
                track = '{0}_{1}'.format(artist_name, track_name)
                train_set.append(track)
            assigned_train += number_of_tracks

            if assigned_train == train_set_length:
                break

    # Remove the artists that are assigned train_set
    for artist_name in assigned_artist_list:
        artist_list.remove(artist_name)

    # Assigning the validation_set
    rnd_perm = np.random.permutation(len(artist_list))
    for i in rnd_perm:
        artist_name = artist_list[i]
        number_of_tracks = len(artists[artist_name])
        if (assigned_validation + number_of_tracks) < validation_set_length:
            assigned_artist_list.append(artist_name)
            for track_name in artists[artist_name]:
                track = '{0}_{1}'.format(artist_name, track_name)
                validation_set.append(track)
            assigned_validation += number_of_tracks

            if assigned_validation == validation_set_length:
                break

    # Remove the artists that are assigned validation_set
    for artist_name in assigned_artist_list:
        if artist_name in artist_list:
            artist_list.remove(artist_name)

    # The rest of the list is assigned for test_set
    for artist_name in artist_list:
        for track_name in artists[artist_name]:
            track = '{0}_{1}'.format(artist_name, track_name)
            test_set.append(track)

    train_set.sort()
    validation_set.sort()
    test_set.sort()

    return train_set, validation_set, test_set


def get_feature_sequence(track_name, args):
    if args.feature_type == 'HF0':
        if args.corrected_pitch:
            feature_sequence_path = '{0}/{1}_corrected_pitch.h5'.format(get_hf0_path(), track_name)
            dict_key = 'HF0'
        else:
            feature_sequence_path = '{0}/{1}.h5'.format(get_hf0_path(), track_name)
            dict_key = 'HF0_firstestimation'
    elif args.feature_type == 'CQT':
        feature_sequence_path = '{0}/{1}.h5'.format(get_cqt_path(), track_name)
        dict_key = 'cqt'

    feats = h5py.File(feature_sequence_path, 'r')
    feature_sequence = np.array(feats[dict_key])

    return feature_sequence


def generate_set(set_choice, set_list, song_lengths, lb, args):
    song_lengths[set_choice] = {}
    patch_length = int(args.segment_length)
    idx = 0
    idx = int(idx)
    for ind, track_name in enumerate(set_list):
        labels = get_labels(track_name=track_name)
        feature_sequence = get_feature_sequence(track_name=track_name, args=args)

        if args.verbose:
            print('{0} {1}: {2} with length {3} frames or {4} secs'.format(set_choice, ind, track_name,
                                                                           feature_sequence.shape[1],
                                                                           feature_sequence.shape[1] * np.float(
                                                                               args.hop_size) / args.SR))
        length_track = np.min((feature_sequence.shape[1], len(labels)))

        if ind == 0:
            x_set = feature_sequence[:, :int(np.floor(length_track / patch_length) * patch_length)]
            x_set = np.append(x_set, feature_sequence[:, int(length_track - patch_length):length_track], axis=1)
            y_set = labels[:int(np.floor(length_track / patch_length) * patch_length)]
            y_set = np.append(y_set, labels[int(length_track - patch_length):length_track], axis=0)
        else:
            x_set = np.append(x_set, feature_sequence[:, :int(np.floor(length_track / patch_length) * patch_length)], axis=1)
            x_set = np.append(x_set, feature_sequence[:, int(length_track - patch_length):length_track], axis=1)
            y_set = np.append(y_set, labels[:int(np.floor(length_track / patch_length) * patch_length)], axis=0)
            y_set = np.append(y_set, labels[int(length_track - patch_length):length_track], axis=0)

        song_lengths[set_choice][track_name] = {}
        song_lengths[set_choice][track_name]['original'] = int(length_track)
        song_lengths[set_choice][track_name]['modified'] = int(
            np.floor(length_track / patch_length) * patch_length) + patch_length

        # !!Removed the 'idx' parameter from the song_lengths dictionary because it is int64 and not JSON serializable!!
        song_lengths[set_choice][track_name]['idx'] = int(idx)
        idx = idx + song_lengths[set_choice][track_name]['modified']

    if args.verbose:
        print('\nNormalizing the {0}..\n'.format(set_choice))

    x_set = normalize(x_set, norm='l1', axis=0)
    x_set = x_set.T

    y_set = lb.transform(y_set)

    return x_set, y_set


def save_set(set_choice, x_set, y_set, args):
    path = get_dataset_splits_save_path()
    file_name = '{0}/dataset-{1}_{2}.h5'.format(path, args.dataset_number, set_choice)
    out = h5py.File(file_name, 'w')

    out.create_dataset('x_{0}'.format(set_choice), x_set.shape, data=x_set)
    out.create_dataset('y_{0}'.format(set_choice), y_set.shape, data=y_set)
    out.close()


def save_split_lists(args, train_set, validation_set, test_set):
    splits = {'train': train_set, 'validation': validation_set, 'test': test_set}
    path = get_dataset_splits_save_path()
    with open('{0}/dataset-{1}_splits.json'.format(path, args.dataset_number), 'w') as f:
        json.dump(splits, f)


def save_song_lengths(args, song_lengths):
    path = get_dataset_splits_save_path()
    with open('{0}/dataset-{1}_song_lengths.json'.format(path, args.dataset_number), 'w') as f:
        json.dump(song_lengths, f)


def save_parameters(args):
    parameters = vars(args)
    path = get_dataset_splits_save_path()
    with open('{0}/dataset-{1}_parameters.json'.format(path, args.dataset_number), 'w') as f:
        json.dump(parameters, f)


def save_dataset_splits(x_train, y_train, x_validation, y_validation,
                        train_set, validation_set, test_set,
                        song_lengths, args):
    save_split_lists(args=args, train_set=train_set, validation_set=validation_set, test_set=test_set)
    save_set(set_choice='train', x_set=x_train, y_set=y_train, args=args)
    save_set(set_choice='validation', x_set=x_validation, y_set=y_validation, args=args)
    save_song_lengths(args=args, song_lengths=song_lengths)
    save_parameters(args=args)


def generate_dataset_splits(args):
    lb = LabelBinarizer()
    lb.fit(np.arange(-1, args.number_of_classes - 1))

    if args.shuffle_data:
        # Random assignment of datasets for training, validation and test sets
        train_set, validation_set, test_set = generate_train_validation_test_set_lists(dataset_name=args.dataset_name,
                                                                                       train_set_ratio=args.train_set_ratio,
                                                                                       validation_set_ratio=args.validation_set_ratio)
    else:  # read from a split source file
        with open(args.split_file_path) as f:
            dataset_split_lists = json.load(f)

        train_set = dataset_split_lists['train']
        validation_set = dataset_split_lists['validation']
        test_set = dataset_split_lists['test']

    # The time-frequency representation is split into (patch_length x feature_size) images
    # feature_size = number of F0s between semitones x Number of Octaves x Number of Semitones in octave + 1 (extra C6)
    feature_size = int(args.step_notes * 5 * 12 + 1)
    patch_length = int(args.segment_length)

    song_lengths = {}

    x_train, y_train = generate_set(set_choice='train_set',
                                    set_list=train_set,
                                    song_lengths=song_lengths,
                                    lb=lb,
                                    args=args)

    x_validation, y_validation = generate_set(set_choice='validation_set',
                                              set_list=validation_set,
                                              song_lengths=song_lengths,
                                              lb=lb,
                                              args=args)

    save_dataset_splits(x_train=x_train, x_validation=x_validation,
                        y_train=y_train, y_validation=y_validation,
                        train_set=train_set, validation_set=validation_set, test_set=test_set,
                        song_lengths=song_lengths, args=args)


if __name__ == '__main__':
    args = parse_input(sys.argv[1:])
    generate_dataset_splits(args=args)
