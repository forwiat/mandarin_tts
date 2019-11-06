import re
import os
import unicodedata
import codecs
import librosa
from tqdm import tqdm

class design:
    def __init__(self, csv_path, skip=3, read_mode='ljspeech', sorted_mode=0, log_file=None):
        '''
        :param csv_path: String. lib csv file
        :param skip: An integer. skip phone num with default set skip=3
        :param read_mode: String. Different text format.
        :param sorted_mode: sorted mode by type or by total with default set sorted_mode=0
        :param log_file: None or String. Write result into log.
        '''
        self.csv_path = csv_path
        self.skip = skip
        self.sinphone_num_norepeat = 0
        self.sinphone_num_total = 0
        self.skiphone_num_norepeat = 0
        self.skiphone_num_total = 0
        self.sinphone = {}
        self.skiphone = {}
        self.index2num = {}
        self.sorted_mode = sorted_mode
        self.read_mode = read_mode
        self.log_file = log_file
        print('handling files ...')
        self.handle_files()
        print('sorting ...')
        self.sortby_sinskiphone(sorted_mode)
        print('done.')

    def handle_files(self):
        if os.path.isfile(self.csv_path):
            self.lines = codecs.open(self.csv_path, 'r', 'utf-8').readlines()
            index = 0
            for line in tqdm(self.lines):
                # print('handing {} file...'.format(index))
                if self.read_mode == 'ljspeech':
                    _, _, text = line.strip().split('|')
                else:
                    _, text = line.strip().split(' ')
                text = text.lower()
                text = self.remove_punc(text, {""})
                phone = self.convert_txt_using_pre_phone(text)
                ###############statistic single phone######################
                sinphone_num_norepeat = 0
                sinphone_num_total = len(phone)
                sinphone_vis = {}
                self.sinphone_num_total += len(phone)
                for i in range(len(phone)):
                    sinphone = phone[i]
                    if sinphone not in self.sinphone.keys():
                        self.sinphone[sinphone] = 1
                        self.sinphone_num_norepeat += 1
                    else:
                        self.sinphone[sinphone] += 1
                    if sinphone not in sinphone_vis.keys():
                        sinphone_num_norepeat += 1
                        sinphone_vis[sinphone] = 1
                ###############statistic single phone######################

                ###############statistic skip phone########################
                skiphone_num_norepeat = 0
                skiphone_num_total = max(0, len(phone) - self.skip + 1)
                skiphone_vis = {}
                self.skiphone_num_total += max(0, len(phone) - self.skip + 1)
                for i in range(len(phone) - self.skip + 1):
                    triphone = "".join(phone[i: i + self.skip])
                    if triphone not in self.skiphone.keys():
                        self.skiphone[triphone] = 1
                        self.skiphone_num_norepeat += 1
                    else:
                        self.skiphone[triphone] += 1
                    if triphone not in skiphone_vis.keys():
                        skiphone_vis[triphone] = 1
                        skiphone_num_norepeat += 1
                ###############statistic skip phone########################
                self.index2num[index] = [sinphone_num_norepeat, sinphone_num_total,
                                         skiphone_num_norepeat, skiphone_num_total]
                index += 1
        else:
            print('no such file! please check your csv file path ...')
            exit(0)

    def sortby_sinskiphone(self, sorted_mode=0):
        # sorted by sinphone firstly, then sorted by skiphone. decrease order.
        '''
        :param sorted_mode: sorted mode by no-repeat or by total with default set sorted_mode=0
        :return: None
        '''
        if sorted_mode == 0:
            self.res = sorted(self.index2num.items(), key=lambda x:(x[1][0], x[1][2]), reverse=True)
        else:
            self.res = sorted(self.index2num.items(), key=lambda x:(x[1][1], x[1][3]), reverse=True)

    def get_sent(self, num=None, wavs_dir=None, duration=None, sr=22050, get_mode='bynum'):
        '''
        :param num: An integer. get head num sentence.
        :param duration: An number. Required duration time. units: h
        :param wavs_dir: String. Total wav files directory.
        :param sr: An integer. If get_mode == 'bydur', then sr is sample rate.
        :param get_mode: String. 'bynum' or 'bydur'. Select by sentenses nums or by duration time.
        :return: head num sentence or total sentence(if num >= total sentence num).
                 no-repeat single phone convert rate
                 no-repeat skip phone convert rate
                 total single phone convert rate
                 total skip hone convert rate
        '''
        res_sent = []
        sinphone_num_total = 0
        sinphone_num_norepeat = 0
        skiphone_num_total = 0
        skiphone_num_norepeat = 0
        sinphone_vis = {}
        skiphone_vis = {}
        total_time = 0
        if get_mode == ' bynum':
            for i in tqdm(range(min(num, len(self.res)))):
                # print('getting {} sentence ...'.format(i))
                line = self.lines[self.res[i][0]]
                res_sent.append(line)
                if self.read_mode == 'ljspeech':
                    _, _, pos_sent = line.strip().split('|')
                else:
                    _, pos_sent = line.strip().split(' ')
                pos_sent = pos_sent.lower()
                phone = self.convert_txt_using_pre_phone(self.remove_punc(pos_sent, {""}))
                sinphone_num_total += len(phone)
                for j in range(len(phone)):
                    sinphone = phone[j]
                    if sinphone not in sinphone_vis.keys():
                        sinphone_vis[sinphone] = 1
                        sinphone_num_norepeat += 1
                skiphone_num_total += max(0, len(phone) - self.skip + 1)
                for j in range(len(phone) - self.skip + 1):
                    skiphone = "".join(phone[j: j + self.skip])
                    if skiphone not in skiphone_vis.keys():
                        skiphone_vis[skiphone] = 1
                        skiphone_num_norepeat += 1
        elif get_mode == 'bydur':
            for i in range(len(self.res)):
                print('Hanled duration time :  {} h at present, required time : {}'.format(total_time * 1.0 / 3600, duration))
                if total_time * 1.0/3600 > duration:
                    break
                line = self.lines[self.res[i][0]]
                res_sent.append(line)
                if self.read_mode == 'ljspeech':
                    fname_noexc, _, pos_sent = line.strip().split('|')
                else:
                    fname_noexc, pos_sent = line.strip().split(' ')
                fpath = os.path.join(wavs_dir, fname_noexc + '.wav')
                y, _ = librosa.load(fpath, sr=sr)
                S = librosa.stft(y)
                total_time += librosa.get_duration(S=S, sr=sr)
                pos_sent = pos_sent.lower()
                phone = self.convert_txt_using_pre_phone(self.remove_punc(pos_sent, {""}))
                sinphone_num_total += len(phone)
                for j in range(len(phone)):
                    sinphone = phone[j]
                    if sinphone not in sinphone_vis.keys():
                        sinphone_vis[sinphone] = 1
                        sinphone_num_norepeat += 1
                skiphone_num_total += max(0, len(phone) - self.skip + 1)
                for j in range(len(phone) - self.skip + 1):
                    skiphone = "".join(phone[j: j + self.skip])
                    if skiphone not in skiphone_vis.keys():
                        skiphone_vis[skiphone] = 1
                        skiphone_num_norepeat += 1
        # calculate no-repeat convert rate
        sinphone_norepeat_rate = sinphone_num_norepeat * 1.0 / (self.sinphone_num_norepeat + 1e-18)
        skiphone_norepeat_rate = skiphone_num_norepeat * 1.0 / (self.skiphone_num_norepeat + 1e-18)
        # calculate total convert rate
        sinphone_total_rate = sinphone_num_total * 1.0 / (self.sinphone_num_total + 1e-18)
        skiphone_total_rate = skiphone_num_total * 1.0 / (self.skiphone_num_total + 1e-18)
        if self.sorted_mode == 0:
            sorted_string = 'sorted by type'
        else:
            sorted_string = 'sorted by number'
        if self.log_file == None:
            print('##################################################################################')
            print('sorted_mode type : {} | sorted by {}'.format(self.sorted_mode, sorted_string))
            print('single phone type number : {} | total single phone type number : {}'.format(sinphone_num_norepeat, self.sinphone_num_norepeat))
            print('skip {} phone type number : {} | total skip {} phone type number : {}'.format(self.skip, skiphone_num_norepeat, self.skip, self.skiphone_num_norepeat))
            print('single phone total number : {} | total single phone total number : {}'.format(sinphone_num_total, self.sinphone_num_total))
            print('skip {} phone total number : {} | skip {} phone total number : {}'.format(self.skip, skiphone_num_total, self.skip, self.skiphone_num_total))
            print('##################################################################################')
            print('##################################################################################')
            print('handled sentence number : {}'.format(len(res_sent)))
            print('single phone type convert rate : {}'.format(sinphone_norepeat_rate))
            print('skip {} phone type convert rate : {}'.format(self.skip, skiphone_norepeat_rate))
            print('single phone total convert rate : {}'.format(sinphone_total_rate))
            print('skip {} phone total convert rate : {}'.format(self.skip, skiphone_total_rate))
            print('selected data duration time : {}'.format(total_time))
            print('##################################################################################')
        else:
            log_f = open(self.log_file, 'w')
            print('##################################################################################', file=log_f)
            print('sorted_mode type : {} | sorted by {}'.format(self.sorted_mode, sorted_string), file=log_f)
            print('single phone type number : {} | total single phone type number : {}'.format(sinphone_num_norepeat, self.sinphone_num_norepeat), file=log_f)
            print('skip {} phone type number : {} | total skip {} phone type number : {}'.format(self.skip, skiphone_num_norepeat, self.skip, self.skiphone_num_norepeat), file=log_f)
            print('single phone total number : {} | total single phone total number : {}'.format(sinphone_num_total, self.sinphone_num_total), file=log_f)
            print('skip {} phone total number : {} | skip {} phone total number : {}'.format(self.skip, skiphone_num_total, self.skip, self.skiphone_num_total), file=log_f)
            print('##################################################################################', file=log_f)
            print('##################################################################################', file=log_f)
            print('handled sentence number : {}'.format(len(res_sent)), file=log_f)
            print('single phone type convert rate : {}'.format(sinphone_norepeat_rate), file=log_f)
            print('skip {} phone type convert rate : {}'.format(self.skip, skiphone_norepeat_rate), file=log_f)
            print('single phone total convert rate : {}'.format(sinphone_total_rate), file=log_f)
            print('skip {} phone total convert rate : {}'.format(self.skip, skiphone_total_rate), file=log_f)
            print('selected data duration time : {}'.format(total_time), file=log_f)
            print('##################################################################################', file=log_f)
            print('write into log file {} done.'.format(self.log_file))
            log_f.close()
        print('select over.')
        return res_sent

    def res2file(self, fpath, sents):
        '''
        :param fpath: A String. txt or csv or others file path.
        :param sents: A List. handled sentence list.
        :return: None
        '''
        res_file = open(fpath, 'w')
        for i in sents:
            res_file.write(i)
        print('write into {} over.'.format(fpath))
        res_file.close()

    def convert_txt_using_pre_phone(self, text):
        word_list = []
        phones = os.popen("./get_phone \"" + text + "\"").read()
        for word_phone in phones.strip().split("\n"):
            word_phone = re.sub(r",|\+", "", word_phone)
            word_list.append(word_phone)
            word_list.append(" # ")
        del word_list[-1], word_list[0]
        result = []
        for word in "".join(word_list).split():
            if word != "#":
                # result.append(" ")
            # else:
                result.append(word)
        return result

    def remove_punc(self, text, exception:set={""}):
        # 输入为文本, 输出为标点符号,默认保留 ' , < , > ,
        out = [c if (unicodedata.category(c)[0] != 'P' or c in exception) else " " for c in text]
        return ''.join(out)

# if __name__ == '__main__':
#     handle_ljspeech = create(csv_path='/ssd3/exec/xyb/data/LJSpeech-1.1/head_200_metadata.csv', skip=3, sorted_mode=0)
#     total_sent = handle_ljspeech.get_sent(num=200)
#     handle_ljspeech.res2file(fpath='./handled_sent.csv', sents=total_sent)