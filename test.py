from duration_model import Duration_Graph
from acoustic_model import Acoustic_Graph
def main():
    mode = 'test'
    dur_train_graph = Duration_Graph(mode='train')
    print(f'Duration {mode} Graph loaded done.')
    try:
        while 1:
            y_hat, loss, steps = dur_train_graph.train()
            print('#-------------------------Duration Graph train-------------------------#')
            print(f'#-------------------------steps : {steps} \t loss : {loss}-------------------------#')
    except:
        print('Duration Graph Train over.')

if __name__ == '__main__':
    main()