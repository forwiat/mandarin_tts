# import time
# print(type(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())))
from MTTS.mandarin_frontend import txt2label
def only_chinese(sent):
    flag = True
    for ch in sent:
        if ch < '\u4e00' or ch > '\u9fff':
            flag = False
    return flag
import multiprocessing as mp
def mp_test(args):
    (a, pid) =  args
    num = 0
    for i in a:
        num += i
    print(pid)
    print(num)
# str = str(input())
# if only_chinese(str):
#     print(list(txt2label(str)))
if __name__ == '__main__':
    a = []
    for i in range(100):
        a.append(i)
    num_splits = mp.cpu_count()
    num_splits = int(num_splits/2)
    b = [(a[i::num_splits], i)
         for i in range(num_splits)]
    #1.st
    pool = mp.Pool(num_splits)
    pool.map(mp_test, b)
    pool.close()
    pool.join()
