import os
class hyperparams:
    def __init__(self):
        #-----------------------Preprocess params--------------------------#
        self.WAVS_DIR = './data/wavs'
        self.LABELS_DIR = './data/labels'
        self.DATA_DIR = './data'
        self.QS_PATH = './questions-mandarin.hed'
        self.SR = 16000
        self.N_FFT = 512
        self.TRAIN_SIZE = 9000 # Rest data size equals to TEST_SIZE
        self.PRE_MULTI = False
        self.DUR_TF_DIR = os.path.join(self.DATA_DIR, 'dur_tfrecord')
        self.SYN_TF_DIR = os.path.join(self.DATA_DIR, 'syn_tfrecord')
        self.TEMP_DIR = os.path.join(self.DATA_DIR, 'temp')
        self.COARSE_CODE_DIM = 3
        self.FRAME_POSITION_DIM = 1
        self.F0_DIM = 1
        self.CODED_SP_DIM = 60
        self.CODED_AP_DIM = 2
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
        self.DUR_BATCH = 500
        self.DUR_EPOCH = 30
        self.SYN_BATCH = 256
        self.SYN_EPOCH = 30
        self.DUR_IN_DIM = self.DUR_LAB_DIM
        self.DUR_OUT_DIM = self.DURATION_DIM
        self.SYN_IN_DIM = self.SYN_LAB_DIM
        self.SYN_OUT_DIM = self.ACOUSTIC_DIM
        self.DROPOUT_RATE = 0.5
        self.DUR_FC_NUM = 6
        self.DUR_LR = 0.001
        self.DUR_LR_DECAY_STEPS = 400
        self.DUR_LR_DECAY_RATE = 0.5
        self.DUR_MODEL_DIR = ''
        self.DUR_LOG_DIR = ''
        self.DUR_PER_STEPS = 100
        self.SYN_LR = 0.001
        self.SYN_LR_DECAY_STEPS = 400
        self.SYN_LR_DECAY_RATE = 0.5
        self.SYN_K = 16
        self.SYN_HIAHWAY_BLOCK = 4
        self.SYN_MODEL_DIR = ''
        self.SYN_LOG_DIR = ''
        self.SYN_PER_STEPS = 100
        self.TRAIN_GRAPH = 'duration'

    def get_acoustic_dim(self):
        return self.F0_DIM * 3 + self.CODED_SP_DIM * 3 + self.CODED_AP_DIM * 3 + self.VUV_DIM