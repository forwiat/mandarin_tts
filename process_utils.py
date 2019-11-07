import re
from hyperparams import hyperparams
import numpy as np
from scipy.stats import norm
hp = hyperparams()

def write_file(file, data):
    # write data to binary_file
    data = np.array(data, dtype=np.float32)
    fid = open(file, 'wb')
    data.tofile(fid)
    fid.close()

def read_file(file, dimension):
    # read data from binary_file and reshape to (-1, dimension)
    fid = open(file, 'rb')
    data = np.fromfile(fid, dtype=np.float32)
    #print(data.size)
    fid.close()
    data = data[:(dimension * data.size // dimension)]
    data = np.reshape(data, (-1, dimension))
    return data

def load_qs():
    def wildcards2regex(answer, convert_number_pattern=False):
        """
            Convert HTK-style question into regular expression for searching labels.
            If convert_number_pattern, keep the following sequences unescaped for
            extracting continuous values:
                (\d+)       -- handles digit without decimal point
                ([\d\.]+)   -- handles digits with and without decimal point
        """
        prefix = ''
        postfix = ''
        if '*' in answer:
            if not answer.startswith('*'):
                prefix = '\A'
            if not answer.endswith('*'):
                postfix = '\Z'
        answer = answer.strip('*')
        answer = re.escape(answer)
        ## convert remaining HTK wildcards * and ? to equivalent regex:
        answer = answer.replace('\\*', '.*')
        answer = answer.replace('\\?', '.')
        answer = prefix + answer + postfix
        if convert_number_pattern:
            answer = answer.replace('\\(\\\\d\\+\\)', '(\d+)')
            answer = answer.replace('\\(\\[\\\\d\\\\\\.\\]\\+\\)', '([\d\.]+)')
        return answer
    fid = open(hp.QS_PATH, 'r')
    continuous_dict = {}
    discrete_dict = {}
    continuous_index = 0
    discrete_index = 0
    LL = re.compile(re.escape('LL-'))
    for line in fid.readlines():
        line = line.replace('\n', '').replace('\t', ' ')
        temp_line = line.split('{')[1].split('}')[0].strip()
        answer_list = temp_line.split(',')
        question_type = line.split(' ')[0]
        question_key = line.split(' ')[1]
        if question_type == 'CQS':
            processed_answer = wildcards2regex(answer_list[0], convert_number_pattern=True)
            continuous_dict[str(continuous_index)] = re.compile(processed_answer)
            continuous_index += 1
        elif question_type == 'QS':
            discrete_list = []
            for temp_answer in answer_list:
                processed_answer = wildcards2regex(temp_answer)
                if LL.search(question_key):
                    processed_answer = '^' + processed_answer
                discrete_list.append(re.compile(processed_answer))
            discrete_dict[str(discrete_index)] = discrete_list
            discrete_index += 1
    return discrete_dict, continuous_dict
discrete_dict, continuous_dict = load_qs()

def compute_ori_coarse_coding_features():
    npoints = 600
    ori_coarse_coding_features = np.zeros((hp.COARSE_CODE_DIM, npoints))
    x = []
    for i in range(3):
        x.append(np.linspace(-1.5 + i * 0.5, 1.5 + i * 0.5, npoints))
    mul = [0.0, 0.5, 1.0]
    sigma = 0.4
    for i in range(3):
        ori_coarse_coding_features[i, :] = norm.pdf(x[i], mul[i], sigma)
    return ori_coarse_coding_features
ori_coarse_coding_features = compute_ori_coarse_coding_features()

# matching QS and return match result
def pattern_matching_binary(full_label, discrete_dict):
    lab_binary_vector = np.zeros((1, len(discrete_dict)))
    for i in range(len(discrete_dict)):
        answer_list = discrete_dict[str(i)]
        binary_flag = 0
        for compiled_answer in answer_list:
            matching_result = compiled_answer.search(full_label)
            if matching_result is not None:
                binary_flag = 1
                break
        lab_binary_vector[0, i] = binary_flag
    return lab_binary_vector

#matching CQS and return match result
def pattern_matching_continuous(full_label, continuous_dict):
    lab_continuous_vector = np.zeros((1, len(continuous_dict)))
    for i in range(len(continuous_dict)):
        continuous_value = -1.0
        compiled_answer = continuous_dict[str(i)]
        match_result = compiled_answer.search(full_label)
        if match_result is not None:
            continuous_value = match_result.group(1)
        lab_continuous_vector[0, i] = continuous_value
    return lab_continuous_vector

def extract_coarse_coding_and_postion_features(ori_coarse_coding_features, frame_number):
    # get frame postion features
    # be used in acoustic data
    frame = int(frame_number)
    features_matrix = np.zeros((frame, hp.COARSE_CODE_DIM + hp.FRAME_POSITION_DIM))
    # [frame, cc_dim + fp_dim] --> [frame, 4]
    for i in range(frame):
        rel_index = int((200 / float(frame)) * i)
        features_matrix[i, 0] = ori_coarse_coding_features[0, 300 + rel_index]
        features_matrix[i, 1] = ori_coarse_coding_features[1, 200 + rel_index]
        features_matrix[i, 2] = ori_coarse_coding_features[2, 100 + rel_index]
        features_matrix[i, 3] = float(frame)
    return features_matrix

def match_qs(labels, dimension: int, frame_level=False):
    '''
    :param labels: String list. MTTS handled labels.
    :param dimension: An integer. Duration 467. Acoustic 471.
    :param frame_level: Boolean. Duration False. Acoustic True.
    :return: Numpy.array. With shape [N, D].
    '''

    label_features_matrix = np.zeros((1, dimension))
    for line in labels:
        line = line.strip()
        temp_list = re.split('\s+', line)
        start_time = int(temp_list[0])
        end_time = int(temp_list[1])
        full_label = temp_list[2]
        frame_number = int(end_time / 50000) - int(start_time / 50000)
        # matching
        # be used in all duration and acoustic
        label_binary_vector = pattern_matching_binary(full_label, discrete_dict)
        label_continuous_vector = pattern_matching_continuous(full_label, continuous_dict)
        label_vector = np.concatenate((label_binary_vector, label_continuous_vector), axis=1)
        if frame_level:
            coarse_coding_features_matrix = extract_coarse_coding_and_postion_features(ori_coarse_coding_features,
                                                                                       frame_number)
            # coarse_coding_features_matrix dimension is [frame_number, 4]
            label_frame_level_metrix = np.tile(label_vector, (frame_number, 1))
            # label_frame_level_metrix dimension is [frame_number, 467]
            features_matrix = np.concatenate((label_frame_level_metrix, coarse_coding_features_matrix), axis=1)
            label_features_matrix = np.concatenate((label_features_matrix, features_matrix), axis=0)
        else:
            label_features_matrix = np.concatenate((label_features_matrix, label_vector), axis=0)
    label_features_matrix = label_features_matrix[1:, :]
    return label_features_matrix

def remove_sil(inputs, labels, frame_level=False):
    '''
    Explain: Like '-sil+' is a silence which will be removed.
    :param inputs: Numpy.array with shape [N, D].
    :param labels: String list. be Required in removing silence.
    :param frame_level: Boolean. Duration False. Acoustic True.
    :return: Numpy.array. With shape [N2, D].
    '''
    def _check_silence_pattern(full_label):
        silence_pattern = '*-sil+*'
        current_pattern = silence_pattern.strip('*')
        if current_pattern in full_label:
            return 1
        return 0
    no_silence_flag = []
    index = 0
    for line in labels:
        line = line.strip()
        temp_list = re.split('\s+', line)
        start_time = int(temp_list[0])
        end_time = int(temp_list[1])
        full_label = temp_list[2]
        frame_number = int(end_time / 50000) - int(start_time / 50000)
        binary_flag = _check_silence_pattern(full_label)
        if binary_flag == 0:
            if frame_level:
                for j in range(frame_number):
                    no_silence_flag.append(index + j)
                index += frame_number
            else:
                no_silence_flag.append(index)
                index += 1
    no_silence_flag = [ix for ix in no_silence_flag if ix < inputs.shape[0]]
    outputs = inputs[no_silence_flag]
    return outputs

def get_dur(labels, dimension):
    '''
    :param labels: String list.
    :param dimension: An integer. Duration feature dimension == 1
    :return: Numpy.array. [N, D]
    '''
    phone_num = len(labels)
    dur_feature_matrix = np.empty((phone_num, dimension))
    dur_feature_index = 0
    for line in labels:
        line = line.strip()
        temp_list = re.split('\s+', line)
        start_time = int(temp_list[0])
        end_time = int(temp_list[1])
        full_label = temp_list[2]
        frame_number = int(end_time / 50000) - int(start_time / 50000)
        phone_duration = frame_number
        current_phone_array = np.array([phone_duration])
        dur_feature_matrix[dur_feature_index:dur_feature_index + 1, ] = current_phone_array
        dur_feature_index += 1
    return dur_feature_index

def get_syn(f0, sp, ap):
    '''
    :param f0: Numpy.array. [T, 1] or [T, ]
    :param sp: Numpy.array. [T, D1]
    :param ap: Numpy.array. [T, D2]
    :return: Numpy.array. [T, D3]
    '''
    def _interpolate_f0(features):
        # args :
        # f0 or lf0 features
        # keep f0 or lf0 always > 0
        # do linear
        frame_number = features.size
        data = np.reshape(features, (frame_number, 1))
        vuv_vector = np.zeros((frame_number, 1))
        vuv_vector[data > 0.0] = 1.0
        vuv_vector[data <= 0.0] = 0.0
        ip_data = data
        last_value = 0.0
        for i in range(frame_number):
            if data[i] <= 0.0:
                j = i + 1
                for j in range(i + 1, frame_number):
                    if data[j] > 0.0:
                        break
                if j < frame_number - 1:
                    if last_value > 0.0:
                        step = (data[j] - data[i - 1]) / (j - i + 1)
                        for k in range(i, j):
                            ip_data[k] = data[i - 1] + step * (k - i + 1)
                    else:
                        for k in range(i, j):
                            ip_data[k] = data[j]
                else:
                    for k in range(i, frame_number):
                        ip_data[k] = last_value
            else:
                ip_data[i] = data[i]
                last_value = data[i]
        return ip_data, vuv_vector

    def _compute_dynamic_vector(vector, frame_number, win):
        #
        # args :
        # vector : features every col-dim
        # win : do delta or acc
        # return do delta or acc vector
        #
        vector = np.reshape(vector, (frame_number, 1))
        win_length = len(win)
        add_len = int(win_length / 2)
        cal_vector = np.zeros((frame_number + add_len * 2, 1))
        cal_vector[add_len: frame_number + add_len] = vector
        out_vector = np.zeros((frame_number, 1))
        # print('cal_vector shape\t'+ str(cal_vector.shape))
        # print('vector shape\t' + str(vector.shape))
        for i in range(add_len):
            cal_vector[i, 0] = vector[0, 0]
            cal_vector[frame_number + add_len + i, 0] = vector[frame_number - 1, 0]
        # for i in range(cal_vector.size - win_length + 1):
        for i in range(frame_number):
            for j in range(win_length):
                out_vector[i] += cal_vector[i + j, 0] * win[j]
        return out_vector

    def _compute_dynamic_matrix(features, frame_number, dimension, win):
        #
        # args :
        # features : that cal dynamic features
        # frame_number : frame of features
        # dimension : features dimension
        # win : do delta or acc
        dynamic_matrix = np.zeros((frame_number, dimension))
        for dim in range(dimension):
            dynamic_matrix[:, dim:dim + 1] = _compute_dynamic_vector(features[:, dim], frame_number, win)
        return dynamic_matrix

    f0 = np.reshape(f0, (-1, 1))
    frame_num = f0.shape[0]
    out_matrix = np.zeros((frame_num, hp.ACOUSTIC_DIM))
    #----------------------------F0 features---------------------------#
    f0_feas, vuv_vec = _interpolate_f0(f0)
    feature_index = 0
    out_matrix[:, feature_index: hp.F0_DIM] = f0_feas
    feature_index += hp.F0_DIM
    delta_f0_feas = _compute_dynamic_matrix(f0_feas, frame_num, hp.F0_DIM, hp.DELTA_WIN)
    out_matrix[:, feature_index: hp.F0_DIM] = delta_f0_feas
    feature_index += hp.F0_DIM
    acc_f0_feas = _compute_dynamic_matrix(f0_feas, frame_num, hp.F0_DIM, hp.ACC_WIN)
    out_matrix[:, feature_index: hp.F0_DIM] = acc_f0_feas
    feature_index += hp.F0_DIM
    #----------------------------SP features---------------------------#
    out_matrix[:, feature_index: hp.CODED_SP_DIM] = sp
    feature_index += hp.CODED_SP_DIM
    delta_sp_feas = _compute_dynamic_matrix(sp, frame_num, hp.CODED_SP_DIM, hp.DELTA_WIN)
    out_matrix[:, feature_index: hp.CODED_SP_DIM] = delta_sp_feas
    feature_index += hp.CODED_SP_DIM
    acc_sp_feas = _compute_dynamic_matrix(sp, frame_num, hp.CODED_SP_DIM, hp.ACC_WIN)
    out_matrix[:, feature_index: hp.CODED_SP_DIM] = acc_sp_feas
    feature_index += hp.CODED_SP_DIM
    #----------------------------AP features---------------------------#
    out_matrix[:, feature_index: hp.CODED_AP_DIM] = ap
    feature_index += hp.CODED_AP_DIM
    delta_ap_feas = _compute_dynamic_matrix(ap, frame_num, hp.CODED_AP_DIM, hp.DELTA_WIN)
    out_matrix[:, feature_index: hp.CODED_AP_DIM] = delta_ap_feas
    feature_index += hp.CODED_AP_DIM
    acc_ap_feas = _compute_dynamic_matrix(ap, frame_num, hp.CODED_AP_DIM, hp.ACC_WIN)
    out_matrix[:, feature_index: hp.CODED_AP_DIM] = acc_ap_feas
    feature_index += hp.CODED_AP_DIM
    #----------------------------VUV features--------------------------#
    out_matrix[:, feature_index: hp.VUV_DIM] = vuv_vec
    feature_index += hp.VUV_DIM
    return out_matrix

def mmn(inputs, min_vec, max_vec, dimension: int):
    '''
    :param inputs: Numpy.array. [T, D].
    :param min_vec: Numpy.array. [1, D].
    :param max_vec: Numpy.array. [1, D].
    :param dimension: An integer. Data dimension.
    :return: Numpy.array. [T, D].
    '''
    fea_diff_vector = max_vec - min_vec
    target_diff_value = hp.FMAX - hp.FMIN
    fea_diff_vector = np.reshape(fea_diff_vector, (1, dimension))
    target_diff_vector = np.zeros((1, dimension))
    target_diff_vector.fill(target_diff_value)
    target_diff_vector[fea_diff_vector <= 0.0] = 1.0
    fea_diff_vector[fea_diff_vector <= 0.0] = 1.0
    time = inputs.shape[0]  # duration : phone number
    # acoustic : frame number
    fea_diff_matrix = np.tile(fea_diff_vector, (time, 1))
    fea_min_matrix = np.tile(min_vec, (time, 1))
    target_diff_matrix = np.tile(target_diff_vector, (time, 1))
    target_min_matrix = np.zeros((time, dimension))
    target_min_matrix.fill(hp.FMIN)
    out_data = target_diff_matrix / fea_diff_matrix * (inputs - fea_min_matrix) + target_min_matrix
    return out_data

def mvn(inputs, mean_vec, std_vec, dimension: int):
    '''
    :param inputs: Numpy.array. [T, D].
    :param mean_vec: Numpy.array. [1, D].
    :param std_vec: Numpy.array. [1, D].
    :param dimension: An integer. Data dimension.
    :return: Numpy.array. [T, D].
    '''
    time = inputs.shape[0]
    mean_matrix = np.tile(mean_vec, (time, 1))
    std_matrix = np.tile(std_vec, (time, 1))
    out_data = (inputs - mean_matrix) / std_matrix
    return out_data

def demvn(inputs, mean_vec, std_vec, dimension: int):
    '''
    :param inputs: Numpy.array. [T, D].
    :param mean_vec: Numpy.array. [1, D].
    :param std_vec: Numpy.array. [1, D].
    :param dimension: An integer. Data dimension.
    :return: Numpy.array. [T, D].
    '''
    time = inputs.shape[0]
    mean_matrix = np.tile(mean_vec, (time, 1))
    std_mtrix = np.tile(std_vec, (time, 1))
    outputs = inputs * std_mtrix + mean_matrix
    return outputs