import os
import numpy as np
import pyworld
import librosa
from duration_model import Duration_Graph
from acoustic_model import Acoustic_Graph
from MTTS.mandarin_frontend import txt2label
from utils import match_qs, mmn, read_file, demvn, ori_coarse_coding_features, extract_coarse_coding_and_postion_features
from hyperparams import hyperparams
hp = hyperparams()
import argparse

def only_chinese(sent):
    flag = True
    for ch in sent:
        if ch < '\u4e00' or ch > '\u9fff':
            flag = False
    return flag

def get_norm_vec(fpath: str, dimension: int):
    '''
    :param fpath: String. Normalise vector file path.
    :param dimension: An integer. Normalise vector dimension.
    :return: Pair. If min max vector then min_vec, max_vec, else return mean_vec, std_vec.
    '''
    norm_vec = read_file(fpath, dimension)
    f_vec = np.reshape(norm_vec[0], (-1, dimension))
    s_vec = np.reshape(norm_vec[1], (-1, dimension))
    return f_vec, s_vec

def extend_labels(inputs, duration):
    '''
    :param inputs: Numpy.array. [Unit_num, hp.DUR_IN_DIM]
    :param duration: Numpy.array. [Unit_num, ] or [Unit_num, 1]
    :return: [Phone_num, hp.SYN_IN_DIM]
    '''
    duration = np.reshape(duration, (-1, 1))
    if inputs.shape[0] != duration.shape[0]:
        raise Exception('This is a bug which results to labels unit_nums not equal to Duration unit_nums.')
    outputs = np.zeros((1, hp.SYN_IN_DIM))
    for i in range(inputs.shape[0]):
        frame_number = int(duration[i][0])
        # print('%d phone duration frame : %d'%(i, frame_number))
        coarse_coding_features_matrix = extract_coarse_coding_and_postion_features(ori_coarse_coding_features,
                                                                                   frame_number)
        # coarse_coding_features_matrix dimension is [frame_number, 4]
        label_frame_level_metrix = np.tile(inputs[i], (frame_number, 1))
        # label_frame_level_metrix dimension is [frame_number, 467]
        features_matrix = np.concatenate((label_frame_level_metrix, coarse_coding_features_matrix), axis=1)
        outputs = np.concatenate((outputs, features_matrix), axis=0)
    outputs = outputs[1:, :]
    return outputs

def handle(sent, fpath):
    labs = txt2label(sent)
    dur_in = match_qs(labs, hp.DUR_LAB_DIM, False)
    dur_min_vec, dur_max_vec = get_norm_vec(os.path.join(hp.DATA_DIR, 'dur_minmax_vec.npy'), hp.DUR_LAB_DIM)
    dur_in_norm = mmn(inputs=dur_in, min_vec=dur_min_vec, max_vec=dur_max_vec, dimension=hp.DUR_LAB_DIM)
    dur_net = Duration_Graph(mode='infer')
    duration = dur_net.infer(dur_in_norm)
    dur_mean_vec, dur_std_vec = get_norm_vec(os.path.join(hp.DATA_DIR, 'dur_meanstd_vec.npy'), hp.DURATION_DIM)
    duration_features = demvn(duration, dur_mean_vec, dur_std_vec, dimension=hp.DURATION_DIM)
    syn_in = extend_labels(dur_in, duration_features)
    syn_min_vec, syn_max_vec = get_norm_vec(os.path.join(hp.DATA_DIR, 'syn_minmax_vec.npy'), hp.SYN_LAB_DIM)
    syn_in_norm = mmn(inputs=syn_in, min_vec=syn_min_vec, max_vec=syn_max_vec, dimension=hp.SYN_LAB_DIM)
    syn_net = Acoustic_Graph(mode='infer')
    acoustic = syn_net.infer(syn_in_norm)
    syn_mean_vec, syn_std_vec = get_norm_vec(os.path.join(hp.DATA_DIR, 'syn_meanstd_vec.npy'), hp.ACOUSTIC_DIM)
    acoustic_features = demvn(acoustic, syn_mean_vec, syn_std_vec, dimension=hp.ACOUSTIC_DIM)
    index = 0
    f0_features = acoustic_features[:, index: index + hp.F0_DIM]
    index += hp.F0_DIM * 3
    coded_sp_features = acoustic_features[:, index: index + hp.CODED_SP_DIM]
    index += hp.CODED_SP_DIM * 3
    coded_ap_features = acoustic_features[:, index: index + hp.CODED_AP_DIM]
    decoded_sp_features = pyworld.decode_spectral_envelope(coded_sp_features, hp.SR, fft_size=hp.N_FFT)
    decoded_ap_features = pyworld.decode_aperiodicity(coded_ap_features, hp.SR, fft_size=hp.N_FFT)
    new_y = pyworld.synthesize(f0_features, decoded_sp_features, decoded_ap_features, hp.SR)
    librosa.output.write_wav(fpath, new_y, hp.SR)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sentence', '-s', type=str, help='Synthesize content. Only supported chinese.')
    parser.add_argument('--path', '-f', type=str, help='Synthesized file path.')
    parser.set_defaults(sentence=None)
    parser.set_defaults(path=None)
    args = parser.parse_args()
    sent = args.sentence
    path = args.path
    if sent is None or only_chinese(sent) is False:
        raise Exception('Input sentence is illegal. Only supported to chinese. Please check.')
    if path is None:
        raise Exception('Please input file path.')
    handle(sent, path)

if __name__ == '__main__':
    main()