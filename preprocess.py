import glob
import os
import codecs
import numpy as np
import pyworld
import librosa
import tensorflow as tf
from tqdm import tqdm
import multiprocessing as mp
from hyperparams import hyperparams
from utils import write_file, read_file, match_qs, remove_sil, get_dur, get_syn, mmn, mvn
hp = hyperparams()

def check():
    label_paths = glob.glob(f'{hp.LABELS_DIR}/*.lab')
    wav_paths = glob.glob(f'{hp.WAVS_DIR}/*.wav')
    if len(label_paths) != len(wav_paths):
        raise Exception('Wav files are not equal to label files. Please check.')
    fname_dic = {}
    for i in label_paths:
        fname = os.path.basename(i)[:-4]
        fname_dic[fname] = 1
    for i in wav_paths:
        fname = os.path.basename(i)[:-4]
        if fname not in fname_dic.keys():
            raise Exception(f'{fname} is not in {hp.LABELS_DIR}, which results to mismatch. Please check.')
    if os.path.isdir(hp.WAVS_DIR) is False or os.path.isdir(hp.LABELS_DIR) is False:
        raise Exception(f'{hp.WAVS_DIR} is not created or not a directory. Please check.')
    if os.path.isdir(hp.DATA_DIR) is False:
        os.makedirs(hp.DATA_DIR)
    if os.path.isdir(hp.DUR_TF_DIR) is False:
        os.makedirs(hp.DUR_TF_DIR)
    if os.path.isdir(hp.SYN_TF_DIR) is False:
        os.makedirs(hp.SYN_TF_DIR)
    if os.path.isdir(hp.TEMP_DIR) is False:
        os.makedirs(hp.TEMP_DIR)
    return label_paths, wav_paths

def get_features(fpath: str):
    y, _ = librosa.load(fpath, sr=hp.SR, dtype=np.float64)
    f0, timeaxis = pyworld.harvest(y, hp.SR, f0_floor=71.0, f0_ceil=500.0)
    sp = pyworld.cheaptrick(y, f0, timeaxis, hp.SR, fft_size=hp.N_FFT)
    ap = pyworld.d4c(y, f0, timeaxis, hp.SR, fft_size=hp.N_FFT)
    coded_sp = pyworld.code_spectral_envelope(sp, hp.SR, number_of_dimensions=hp.CODED_SP_DIM)
    coded_ap = pyworld.code_aperiodicity(ap, hp.SR)
    return f0, coded_sp, coded_ap

def get_minmax_vector(files: list, dimension: int):
    '''
    :param files: List. Total files path.
    :param dimension: An integer. data dimension.
    :return: Min vector and Max vector. Shape: [1, D]
    '''
    length = len(files)
    min_matrix = np.zeros((length, dimension))
    max_matrix = np.zeros((length, dimension))
    for i in tqdm(range(length)):
        data = read_file(files[i], dimension)
        temp_min = np.amin(data, axis=0)
        temp_max = np.amax(data, axis=0)
        min_matrix[i,] = temp_min
        max_matrix[i,] = temp_max
    min_vector = np.amin(min_matrix, axis=0)
    max_vector = np.amax(max_matrix, axis=0)
    min_vector = min_vector.reshape((-1, dimension))
    max_vector = max_vector.reshape((-1, dimension))
    return min_vector, max_vector

def get_meanstd_vector(files, dimension):
    '''
    :param files: List. Total files path.
    :param dimension: An integer. data dimension.
    :return: Mean vector and Std vector. Shape: [1, D]
    '''
    length = len(files)
    mean_vector = np.zeros((1, dimension))
    std_vector = np.zeros((1, dimension))
    times = 0
    for i in tqdm(range(length)):
        features = read_file(files [i], dimension)
        time = features.shape[0]
        # duration : phone number
        # acoustic : frame number
        mean_vector += np.sum(features, axis=0)
        times += time
    mean_vector /= float(times)
    # print(mean_vector)
    for i in range(length):
        features = read_file(files[i], dimension)
        time = features.shape[0]
        mean_matrix = np.tile(mean_vector, (time, 1))
        std_vector += np.sum((features - mean_matrix) ** 2, axis=0)
    std_vector /= float(times)
    std_vector = std_vector ** 0.5
    std_vector[std_vector <= 0] = 1e-9
    mean_vector = np.reshape(mean_vector, (1, dimension))
    std_vector = np.reshape(std_vector, (1, dimension))
    return mean_vector, std_vector

def handle_text(label_paths):
    for i in tqdm(range(len(label_paths))):
        fname_noexc = os.path.basename(label_paths[i])[:-4]
        lab = codecs.open(label_paths[i], 'r').readlines()
        dur_lab = match_qs(lab, dimension=hp.DUR_LAB_DIM, frame_level=False)
        syn_lab = match_qs(lab, dimension=hp.SYN_LAB_DIM, frame_level=True)
        #nosil_dur_lab = remove_sil(dur_lab, labels=lab, frame_level=False)
        #nosil_syn_lab = remove_sil(syn_lab, labels=lab, frame_level=True)
        write_file(os.path.join(hp.TEMP_DIR, fname_noexc + '_in.dur'), dur_lab)
        write_file(os.path.join(hp.TEMP_DIR, fname_noexc + '_in.syn'), syn_lab)

def handle_feature(label_paths):
    for i in tqdm(range(len(label_paths))):
        fname_noexc = os.path.basename(label_paths[i])[:-4]
        wav_path = os.path.join(hp.WAVS_DIR, fname_noexc + '.wav')
        lab = codecs.open(label_paths[i], 'r').readlines()
        dur_feas = get_dur(lab, dimension=1)
        #nosil_dur_feas = remove_sil(dur_feas, labels=lab, frame_level=False)
        write_file(os.path.join(hp.TEMP_DIR, fname_noexc + '_out.dur'), dur_feas)
        f0, coded_sp, coded_ap = get_features(wav_path)
        syn_feas = get_syn(f0, coded_sp, coded_ap)
        #nosil_syn_feas = remove_sil(syn_feas, labels=lab, frame_level=True)
        write_file(os.path.join(hp.TEMP_DIR, fname_noexc + '_out.syn'), syn_feas)

def get_normalise_vector(label_paths):
    print('\t#---------------3.1 Get dur_min_max_vector---------------#\t')
    dur_in_files = [os.path.join(hp.TEMP_DIR, os.path.basename(i)[:-4] + '_in.dur') for i in label_paths]
    dur_min_vec, dur_max_vec = get_minmax_vector(dur_in_files, hp.DUR_LAB_DIM)
    dur_mm_vec = np.concatenate((dur_min_vec, dur_max_vec), axis=0)
    write_file(os.path.join(hp.DATA_DIR, 'dur_minmax_vec.npy'), dur_mm_vec)
    print('\t#---------------3.2 Get dur_mean_std_vector--------------#\t')
    dur_out_files = [os.path.join(hp.TEMP_DIR, os.path.basename(i)[:-4] + '_out.dur') for i in label_paths]
    dur_mean_vec, dur_std_vec = get_meanstd_vector(dur_out_files, hp.DURATION_DIM)
    dur_ms_vec = np.concatenate((dur_mean_vec, dur_std_vec), axis=0)
    write_file(os.path.join(hp.DATA_DIR, 'dur_meanstd_vec.npy'), dur_ms_vec)
    print('\t#---------------3.3 Get syn_min_max_vector---------------#\t')
    syn_in_files = [os.path.join(hp.TEMP_DIR, os.path.basename(i)[:-4] + '_in.syn') for i in label_paths]
    syn_min_vec, syn_max_vec = get_minmax_vector(syn_in_files, hp.SYN_LAB_DIM)
    syn_mm_vec = np.concatenate((syn_min_vec, syn_max_vec), axis=0)
    write_file(os.path.join(hp.DATA_DIR, 'syn_minmax_vec.npy'), syn_mm_vec)
    print('\t#---------------3.4 Get syn_mean_std_vector--------------#\t')
    syn_out_files = [os.path.join(hp.TEMP_DIR, os.path.basename(i)[:-4] + '_out.syn') for i in label_paths]
    syn_mean_vec, syn_std_vec = get_meanstd_vector(syn_out_files, hp.ACOUSTIC_DIM)
    syn_ms_vec = np.concatenate((syn_mean_vec, syn_std_vec), axis=0)
    write_file(os.path.join(hp.DATA_DIR, 'syn_meanstd_vec.npy'), syn_ms_vec)
    print('\t#---------------3.5 Done---------------------------------#\t')

def normalise(label_paths):
    # ------------Duration normalise vector------------------#
    dur_mm_vec = read_file(os.path.join(hp.DATA_DIR, 'dur_minmax_vec.npy'), dimension=hp.DUR_LAB_DIM)
    dur_min_vec = np.reshape(dur_mm_vec[0], (-1, hp.DUR_LAB_DIM))
    dur_max_vec = np.reshape(dur_mm_vec[1], (-1, hp.DUR_LAB_DIM))
    dur_ms_vec = read_file(os.path.join(hp.DATA_DIR, 'dur_meanstd_vec.npy'), dimension=hp.DURATION_DIM)
    dur_mean_vec = np.reshape(dur_ms_vec[0], (-1, hp.DURATION_DIM))
    dur_std_vec = np.reshape(dur_ms_vec[1], (-1, hp.DURATION_DIM))
    # ------------Acoustic normalise vector------------------#
    syn_mm_vec = read_file(os.path.join(hp.DATA_DIR, 'syn_minmax_vec.npy'), dimension=hp.SYN_LAB_DIM)
    syn_min_vec = np.reshape(syn_mm_vec[0], (-1, hp.SYN_LAB_DIM))
    syn_max_vec = np.reshape(syn_mm_vec[1], (-1, hp.SYN_LAB_DIM))
    syn_ms_vec = read_file(os.path.join(hp.DATA_DIR, 'syn_meanstd_vec.npy'), dimension=hp.ACOUSTIC_DIM)
    syn_mean_vec = np.reshape(syn_ms_vec[0], (-1, hp.ACOUSTIC_DIM))
    syn_std_vec = np.reshape(syn_ms_vec[1], (-1, hp.ACOUSTIC_DIM))
    for i in tqdm(range(len(label_paths))):
        fname_noexc = os.path.basename(label_paths[i])[:-4]
        # ------------------------Duration inputs-------------------------#
        durin_temp_file = os.path.join(hp.TEMP_DIR, fname_noexc + '_in.dur')
        durin_lab = read_file(durin_temp_file, dimension=hp.DUR_LAB_DIM)
        nor_durin_lab = mmn(durin_lab, dur_min_vec, dur_max_vec, dimension=hp.DUR_LAB_DIM)
        write_file(durin_temp_file, nor_durin_lab)
        # ------------------------Duration outputs------------------------#
        durout_temp_file = os.path.join(hp.TEMP_DIR, fname_noexc + '_out.dur')
        durout_feas = read_file(durout_temp_file, dimension=hp.DURATION_DIM)
        nor_durout_feas = mvn(durout_feas, dur_mean_vec, dur_std_vec, dimension=hp.DURATION_DIM)
        write_file(durout_temp_file, nor_durout_feas)
        # ------------------------Acoustic inputs-------------------------#
        synin_temp_file = os.path.join(hp.TEMP_DIR, fname_noexc + '_in.syn')
        synin_lab = read_file(synin_temp_file, dimension=hp.SYN_LAB_DIM)
        nor_synin_lab = mmn(synin_lab, syn_min_vec, syn_max_vec, dimension=hp.SYN_LAB_DIM)
        write_file(synin_temp_file, nor_synin_lab)
        # ------------------------Acoustic outputs------------------------#
        synout_temp_file = os.path.join(hp.TEMP_DIR, fname_noexc + '_out.syn')
        synout_feas = read_file(synout_temp_file, dimension=hp.ACOUSTIC_DIM)
        nor_synout_feas = mvn(synout_feas, syn_mean_vec, syn_std_vec, dimension=hp.ACOUSTIC_DIM)
        write_file(synout_temp_file, nor_synout_feas)

def write_tf(args):
    '''
    args:
    label_paths: File path list.
    id: Process id.
    '''
    (label_paths, id) = args
    global files_cnt
    dur_train_writer = tf.python_io.TFRecordWriter(os.path.join(hp.DUR_TF_DIR, f'{id}_dur_train.tfrecord'))
    dur_test_writer = tf.python_io.TFRecordWriter(os.path.join(hp.DUR_TF_DIR, f'{id}_dur_test.tfrecord'))
    syn_train_writer = tf.python_io.TFRecordWriter(os.path.join(hp.SYN_TF_DIR, f'{id}_syn_train.tfrecord'))
    syn_test_writer = tf.python_io.TFRecordWriter(os.path.join(hp.SYN_TF_DIR, f'{id}_syn_test.tfrecord'))
    for i in tqdm(range(len(label_paths))):
        fname_noexc = os.path.basename(label_paths[i])[:-4]
        durin = read_file(os.path.join(hp.TEMP_DIR, fname_noexc + '_in.dur'), dimension=hp.DUR_LAB_DIM)
        durout = read_file(os.path.join(hp.TEMP_DIR, fname_noexc + '_out.dur'), dimension=hp.DURATION_DIM)
        synin = read_file(os.path.join(hp.TEMP_DIR, fname_noexc + '_in.syn'), dimension=hp.SYN_LAB_DIM)
        synout = read_file(os.path.join(hp.TEMP_DIR, fname_noexc + '_out.syn'), dimension=hp.ACOUSTIC_DIM)
        if durin.shape[0] != durout.shape[0]:
            raise Exception('Duration data 1st dimension of inputs and outputs mismatched. Please check.')
        if synin.shape[0] != synout.shape[0]:
            diff = synout.shape[0] - synin.shape[0]
            synout = synout[diff:, :]
        if synin.shape[0] != synout.shape[0]:
            raise Exception('Acoustic data 1st dimension of inputs and outputs mismatched. Please check.')
        dur_features = {}
        syn_features = {}
        dur_features['x'] = tf.train.Feature(float_list=tf.train.FloatList(value=durin.reshape(-1)))
        dur_features['x_shape'] = tf.train.Feature(int64_list=tf.train.Int64List(value=durin.shape))
        dur_features['y'] = tf.train.Feature(float_list=tf.train.FloatList(value=durout.reshape(-1)))
        dur_features['y_shape'] = tf.train.Feature(int64_list=tf.train.Int64List(value=durout.shape))
        syn_features['x'] = tf.train.Feature(float_list=tf.train.FloatList(value=synin.reshape(-1)))
        syn_features['x_shape'] = tf.train.Feature(int64_list=tf.train.Int64List(value=synin.shape))
        syn_features['y'] = tf.train.Feature(float_list=tf.train.FloatList(value=synout.reshape(-1)))
        syn_features['y_shape'] = tf.train.Feature(int64_list=tf.train.Int64List(value=synout.shape))
        dur_tf_features = tf.train.Features(feature=dur_features)
        syn_tf_features = tf.train.Features(feature=syn_features)
        dur_tf_example = tf.train.Example(features=dur_tf_features)
        syn_tf_example = tf.train.Example(features=syn_tf_features)
        dur_tf_serialized = dur_tf_example.SerializeToString()
        syn_tf_serialized = syn_tf_example.SerializeToString()
        if files_cnt <= hp.TRAIN_SIZE:
            dur_train_writer.write(dur_tf_serialized)
            syn_train_writer.write(syn_tf_serialized)
        else:
            dur_test_writer.write(dur_tf_serialized)
            syn_test_writer.write(syn_tf_serialized)
        files_cnt += 1
    dur_train_writer.close()
    dur_test_writer.close()
    syn_train_writer.close()
    syn_test_writer.close()

def main():
    global files_cnt
    files_cnt = 0
    label_paths, _ = check()
    if hp.PRE_MULTI is False:
        print('#----------------------1. Handling text-------------------------#')
        handle_text(label_paths)
        print('#----------------------2. Handling feature----------------------#')
        handle_feature(label_paths)
        print('#----------------------3. Normalizing---------------------------#')
        get_normalise_vector(label_paths)
        normalise(label_paths)
        print('#----------------------4. Writing TFRecord----------------------#')
        write_tf((label_paths, 0))
        os.system(f'rm -rf {hp.TEMP_DIR}')
        print('#----------------------5. End Done------------------------------#')
    else:
        print('#----------------------1. Handling text-------------------------#')
        num_spilts = mp.cpu_count()
        num_spilts //= 2
        splits = [label_paths[i::num_spilts]
                  for i in range(num_spilts)]
        pool = mp.Pool(num_spilts)
        pool.map(handle_text, splits)
        pool.close()
        pool.join()
        print('#----------------------2. Handling feature----------------------#')
        pool = mp.Pool(num_spilts)
        pool.map(handle_feature, splits)
        pool.close()
        pool.join()
        print('#----------------------3. Normalizing---------------------------#')
        get_normalise_vector(label_paths)
        pool = mp.Pool(num_spilts)
        pool.map(normalise, splits)
        pool.close()
        pool.join()
        print('#----------------------4. Writing TFRecord----------------------#')
        pool = mp.Pool(num_spilts)
        splits = [(label_paths[i::num_spilts],
                   i)
                  for i in range(num_spilts)]
        pool.map(write_tf, splits)
        pool.close()
        pool.join()
        os.system(f'rm -rf {hp.TEMP_DIR}')
        print('#----------------------5. End Done------------------------------#')

if __name__ == '__main__':
    main()