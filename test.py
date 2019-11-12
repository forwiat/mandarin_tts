from duration_model import Duration_Graph
from acoustic_model import Acoustic_Graph
from hyperparams import hyperparams
hp = hyperparams()
def main():
    mode = 'test'
    if hp.TEST_GRAPH.lower() is 'duration':
        dur_test_graph = Duration_Graph(mode=mode)
        print(f'#-------------------------{hp.TEST_GRAPH} {mode} Graph loaded done.-------------------------#')
        try:
            while 1:
                _, loss, steps = dur_test_graph.test()
                print('#-------------------------Duration Graph Test-------------------------#')
                print(f'#-------------------------steps : {steps} \t loss : {loss}-------------------------#')
        except:
            print('#-------------------------Duration Graph Test done.-------------------------#')
    elif hp.TEST_GRAPH.lower() is 'acoustic':
        syn_test_graph = Acoustic_Graph(mode=mode)
        print(f'#-------------------------{hp.TEST_GRAPH} {mode} Graph loaded done.-------------------------#')
        try:
            while 1:
                _, loss, steps = syn_test_graph.test()
                print('#-------------------------Acoustic Graph Test-------------------------#')
                print(f'#-------------------------steps : {steps} \t loss : {loss}-------------------------#')
        except:
            print('#-------------------------Acoustic Graph Test done.-------------------------#')
    else:
        raise Exception(f'#-------------------------No supported TEST_GRAPH named {hp.TEST_GRAPH}. Please check.-------------------------#')

if __name__ == '__main__':
    main()