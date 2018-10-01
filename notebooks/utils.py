import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
from sklearn.metrics import confusion_matrix
import json

number_of_classes = 62
step_notes_class = 1

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
#####################################################
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


def class_to_freq(input_class, minF0=55, maxF0=1760, number_of_classes=62, step_notes=1):
    '''

    '''

    output_freq = np.zeros(input_class.shape)

    F0_list = minF0 * 2 ** (np.arange(number_of_classes - 1) / (12. * step_notes))
    F0_list = np.append(0, F0_list)

    output_freq = F0_list[input_class+1]

    return output_freq


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
#####################################################
def convert_frequency_to_class(pitch_frequencies):
    number_of_F0s = number_of_classes - 1

    F0_labels = 55. * (2 ** (np.arange(number_of_F0s, dtype=np.double) / (12 * step_notes_class)))

    # Non-linear quantization levels
    nlq_levels = F0_labels * (2 ** (1. / (12 * (step_notes_class + 1))))
    nlq_levels = nlq_levels[:-1]

    pitch_labels = []
    for i, freq in enumerate(pitch_frequencies):
        if freq <= 0:
            pitch_labels.append(-1)
        else:  # Apply quantization
            label_idx = np.sum(nlq_levels < freq)
            pitch_labels.append(label_idx)

    return np.array(pitch_labels)


def get_single_track_confusion_matrix(track_name):
    annotation, pitch_estimates = get_annotation_and_estimation(track_name=track_name)

    pitch_estimates_labels = convert_frequency_to_class(pitch_frequencies=pitch_estimates)
    annotation_labels = convert_frequency_to_class(pitch_frequencies=annotation)

    data_length = np.min((len(pitch_estimates_labels), len(annotation_labels)))
    conf_mat = confusion_matrix(annotation_labels[:data_length], pitch_estimates_labels[:data_length],
                                labels=np.arange(-1, number_of_classes - 1))

    return conf_mat


def get_testset_confusion_matrix():
    with open('dataset-ismir-splits.json', 'r') as f:
        dataset_splits = json.load(f)

    confusion_matrix_testset = np.zeros((number_of_classes, number_of_classes))
    for i, track_name in enumerate(dataset_splits['test']):
        confusion_matrix = get_single_track_confusion_matrix(track_name=track_name)
        confusion_matrix_testset += confusion_matrix

    return confusion_matrix_testset

#####################################################

def get_plt_title(track_name):
    data = pd.read_csv(
        '{0}/medleydb_evaluation_results/medleydb_CRNN_evaluation_results.csv'.format(get_path()),
        delimiter=',')
    data = np.array(data)
    track_list = list(data[:,0])
    track_index = track_list.index(track_name)
    oa = data[track_index, 1]
    rpa = data[track_index, 2]
    rca = data[track_index, 3]
    vr = data[track_index, 4]
    vfa = data[track_index, 5]

    title_str = '{0} - oa: {1:.3f}, rpa: {2:.3f}, rca: {3:.3f}, vr: {4:.3f}, vfa: {5:.3f}'.format(track_name,
                                                                                                  oa, rpa, rca, vr, vfa)

    return title_str


def get_track_list():
    data = pd.read_csv(
        '{0}/medleydb_evaluation_results/medleydb_CRNN_evaluation_results.csv'.format(get_path()),
        delimiter=',')
    data = np.array(data)
    track_list = list(data[:-1, 0])

    return track_list

def plot_annotation_vs_estimation(track_name, annotation, pitch_estimation, args):
    data_length = np.min((len(annotation), len(pitch_estimation)))
    time_index = np.arange(data_length) * 256./22050

    plt.plot(time_index, annotation[:data_length], 'o', color='black', markersize=6),
    plt.plot(time_index, pitch_estimation[:data_length], '.', color='red', markersize=2),
    plt.title(get_plt_title(track_name=track_name, args=args))
    plt.show()


def get_annotation_and_estimation(track_name):
    # Annotations
    labels = get_labels(track_name=track_name)
    annotation = class_to_freq(labels)
    # Pitch Estimation
    pitch_estimation = get_pitch_estimation_from_csv(track_name=track_name)

    # plot_annotation_vs_estimation(track_name=track_name, annotation=annotation, pitch_estimation=pitch_estimation,
    #                               args=args)

    return annotation, pitch_estimation

# This comparison works only for jazzomat dataset
if __name__ == '__main__':

    ####
    # ChrisPotter_Arjuna_Solo
    # ChrisPotter_PopTune#1_Solo
    # ColemanHawkins_Stompin'AtTheSavoy_Solo
    # JohnColtrane_GiantSteps-2_Solo
    #
    # KennyDorham_Doodlin'_Solo
    # SonnyRollins_I'llRememberApril_Solo
    #
    # KennyWheeler_PassItOn_Solo!!!!
    # SteveLacy_Let'sCoolOne_Solo!!!!
    #
    # ArtPepper_Anthropology_Solo
    # BennyCarter_SweetLorraine_Solo
    # BixBeiderbecke_RoyalGardenBlues_Solo
    # BuckClayton_DestinationK.C._Solo
    # DexterGordon_Montmartre_Solo
    # JohnColtrane_Impressions_1963_Solo

    # BenWebster_ByeByeBlackbird_Solo
    # BenWebster_DidYouCallHerToday_Solo
    # BenWebster_MyIdeal_Solo

    ####

    # track_name = 'JoeLovano_Work_Solo'
    track_name = "BenWebster_MyIdeal_Solo"

    args = read_arguments(['model1.py'])

    annotation = get_annotation(track_name=track_name)

    pitch_estimation = get_pitch_estimation_from_csv(dataset_name='jazzomat', track_name=track_name, args=args)

    plot_annotation_vs_estimation(track_name=track_name, annotation=annotation, pitch_estimation=pitch_estimation,
                                  args=args)