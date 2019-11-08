import os
import librosa
import pyworld
import glob
import numpy as np
class hyperparams:
    def __init__(self):
        #-----------------------Preprocess params--------------------------#
        ########Params you may need to change############
        self.WAVS_DIR = './data/wavs'
        self.LABELS_DIR = './data/labels'
        self.DATA_DIR = './data' # String. Save train and test data, min max vector and mean std vector.
        self.SR = 16000
        self.N_FFT = 512
        self.TRAIN_SIZE = 9000 # An integer. train data size. TEST_SIZE is the rest.
        self.PRE_MULTI = False # Boolean. Multiprocess can speed up preparing data in pre process. If true Defaults num_cpu//2.
        #################################################
        self.QS_PATH = './questions-mandarin.hed'
        self.DUR_TF_DIR = os.path.join(self.DATA_DIR, 'dur_tfrecord')
        self.SYN_TF_DIR = os.path.join(self.DATA_DIR, 'syn_tfrecord')
        self.TEMP_DIR = os.path.join(self.DATA_DIR, 'temp')
        self.COARSE_CODE_DIM = 3
        self.FRAME_POSITION_DIM = 1
        self.F0_DIM = 1
        self.CODED_SP_DIM = 60 # An integer. coded sp features is constriction of sp features.
        self.CODED_AP_DIM = self.get_codedap_dim() # An integer.
        self.VUV_DIM = 1
        self.DELTA_WIN = [-0.5, 0.0, 0.5] # Delta: First Order Difference
        self.ACC_WIN = [1.0, -2.0, 1.0] # Acc: Second Order Difference
        self.ACOUSTIC_DIM = self.get_acoustic_dim()
        self.DUR_LAB_DIM = 467
        self.SYN_LAB_DIM = 471
        self.DURATION_DIM = 1
        self.FMIN = 0.01
        self.FMAX = 0.99
        # -----------------------Train params-------------------------------#
        self.DUR_IN_DIM = self.DUR_LAB_DIM
        self.DUR_OUT_DIM = self.DURATION_DIM
        self.SYN_IN_DIM = self.SYN_LAB_DIM
        self.SYN_OUT_DIM = self.ACOUSTIC_DIM
        ########Params you may need to change############
        self.DUR_MODEL_DIR = './dur_model'
        self.DUR_LOG_DIR = './dur_logs'
        self.SYN_MODEL_DIR = './syn_model'
        self.SYN_LOG_DIR = './syn_logs'
        self.TRAIN_GRAPH = 'duration' # Options in ['duration', 'acoustic']. If training acoustic model then 'acoustic'.
        self.TEST_GRAPH = 'duration' # Options in ['duration', 'acoustic']. If testing acoustic model then 'acoustic'.
        #################################################
        ########Params you may need to change############
        self.DUR_BATCH = 64
        self.DUR_EPOCH = 30
        self.SYN_BATCH = 256
        self.SYN_EPOCH = 30
        self.DROPOUT_RATE = 0.5
        self.DUR_FC_NUM = 6
        self.DUR_LR = 0.001
        self.DUR_LR_DECAY_STEPS = 400
        self.DUR_LR_DECAY_RATE = 0.5
        self.DUR_PER_STEPS = 100
        self.SYN_LR = 0.001
        self.SYN_LR_DECAY_STEPS = 200
        self.SYN_LR_DECAY_RATE = 0.5
        self.SYN_K = 16
        self.SYN_HIAHWAY_BLOCK = 4
        self.SYN_PER_STEPS = 100
        #################################################

    def get_acoustic_dim(self):
        return self.F0_DIM * 3 + self.CODED_SP_DIM * 3 + self.CODED_AP_DIM * 3 + self.VUV_DIM

    def get_codedap_dim(self):
        fpath = glob.glob(f'{self.WAVS_DIR}/*.wav')[0]
        y, _ = librosa.load(fpath, sr=self.SR, dtype=np.float64)
        f0, timeaxis = pyworld.harvest(y, self.SR, f0_floor=71.0, f0_ceil=500.0)
        ap = pyworld.d4c(y, f0, timeaxis, self.SR, fft_size=self.N_FFT)
        coded_ap = pyworld.code_aperiodicity(ap, self.SR)
        codedap_dim = coded_ap.shape[1]
        return codedap_dim