from duration_model import Duration_Graph
from acoustic_model import Acoustic_Graph
from hyperparams import hyperparams
import os
import time
hp = hyperparams()
def check():
    if os.path.isdir(hp.DUR_MODEL_DIR) is False:
        os.makedirs(hp.DUR_MODEL_DIR)
    if os.path.isdir(hp.DUR_LOG_DIR) is False:
        os.makedirs(hp.DUR_LOG_DIR)
    if os.path.isdir(hp.SYN_MODEL_DIR) is False:
        os.makedirs(hp.SYN_MODEL_DIR)
    if os.path.isdir(hp.SYN_LOG_DIR) is False:
        os.makedirs(hp.SYN_LOG_DIR)

def main():
    check()
    mode = 'train'
    print('#-------------------------Start in ' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()) + '-------------------------#')
    if hp.TRAIN_GRAPH.lower() == 'duration':
        dur_train_graph = Duration_Graph(mode=mode)
        print(f'#-------------------------Duration {mode} Graph loaded done.-------------------------#')
        try:
            while 1:
                y_hat, loss, steps = dur_train_graph.train()
                print('#-------------------------Duration Graph train-------------------------#')
                print(f'#-------------------------steps : {steps} \t loss : {loss}-------------------------#')
        except:
            print('#-------------------------Duration Graph Train over.-------------------------#')
    elif hp.TRAIN_GRAPH.lower() == 'acoustic':
        syn_train_graph = Acoustic_Graph(mode=mode)
        print(f'#-------------------------Acoustic {mode} Graph loaded done.-------------------------#')
        try:
            while 1:
                y_hat, loss, steps = syn_train_graph.train()
                print('#-------------------------Acoustic Graph train-------------------------#')
                print(f'#-------------------------steps : {steps} \t loss : {loss}-------------------------#')
        except:
            print('#-------------------------Acoustic Graph Train over.-------------------------#')
    else:
        raise Exception(f'#-------------------------No supported TRAIN_GRAPH named {hp.TRAIN_GRAPH}. Please check.-------------------------#')

if __name__ == '__main__':
    main()