from math import log
from lab_code import Part_1
from lab_code.Part_5_3 import HMM

Train_File = '../io_file/hmm/train.txt'  # 用于训练参数的文本文件
Word_Freq = {}  # 用于保存词典中的词和词频
Word_Num_Count = 0  # 记录总词数


class DicAction:

    # 构建离线词典并获得必要的数据结构，按照既定格式规约
    @staticmethod
    def gene_uni_dic(train_path=Train_File, dic_path='../io_file/dic/uni_dic.txt'):
        global Word_Freq  # 保存到全局变量中
        with open(train_path, 'r', encoding='utf-8') as seg_file:
            lines = seg_file.readlines()
        for line in lines:
            for word in line.split():
                word = word[1 if word[0] == '[' else 0:word.index('/')]
                Word_Freq[word] = Word_Freq.get(word, 0) + 1
        Word_Freq = {k: Word_Freq[k] for k in sorted(Word_Freq.keys())}  # 对词典排序
        with open(dic_path, 'w', encoding='utf-8') as dic_file:
            for word in Word_Freq.keys():
                dic_file.write(word + ' ' + str(Word_Freq[word]) + '\n')
        Word_Freq = {}
        DicAction.get_uni_dic(dic_path)

    # 从离线词典构建其数据结构，前提是离线词典已经按照既定格式组织好
    @staticmethod
    def get_uni_dic(dic_path='../io_file/dic/uni_dic.txt'):
        global Word_Num_Count
        with open(dic_path, encoding='utf-8') as dic_file:  # 读取离线词典
            lines = dic_file.readlines()
        for line in lines:
            word, freq = line.split()[0:2]  # 离线词典每行的属性通过空格分隔
            Word_Freq[word] = int(freq)  # 将该词存入到词典中
            Word_Num_Count += int(freq)
            for count in range(1, len(word)):  # 获取离线词典中每个词的前缀词
                prefix_word = word[:count]
                if prefix_word not in Word_Freq:  # 前缀不在word_freq中
                    Word_Freq[prefix_word] = 0  # 则存入并置词频为0

    # 通过构建的在线数据结构词典获得有向无环图DAG
    @staticmethod
    def get_dag(line):
        dag = {}  # 用于储存最终的DAG
        n = len(line)  # 句子长度
        for k in range(n):  # 遍历句子中的每一个字
            i = k
            dag[k] = []  # 开始保存处于第k个位置上的字的路径情况
            word_fragment = line[k]
            while i < n and word_fragment in Word_Freq:  # 以k位置开始的词的所在片段在词典中
                if Word_Freq[word_fragment] > 0:  # 若离线词典中存在该词
                    dag[k].append(i)  # 将该片段加入到临时的列表中
                i += 1
                word_fragment = line[k:i + 1]
            dag[k].append(k) if not dag[k] else dag[k]  # 未找到片段，则将单字加入
        return dag

    # 最大概率分词，用于概率最大路径计算
    @staticmethod
    def calc_line_dag(line, dag):
        n = len(line)
        route = {n: (0, 0)}
        log_total = log(Word_Num_Count)
        for idx in range(n - 1, -1, -1):  # 动态规划求最大路径
            route[idx] = max((log(Word_Freq.get(line[idx:x + 1], 0) or 1) - log_total +
                              route[x + 1][0], x) for x in dag[idx])
        return route

    # 对输入文本文件进行最大概率分词：maximum word frequency segmentation
    @staticmethod
    def mwf(txt_path=Part_1.Test_File, mwf_path='../io_file/seg/seg_mwf.txt'):
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        with open(mwf_path, 'w', encoding='utf-8') as mwf_file:
            for line in lines:
                line = line[:len(line) - 1]
                line_route = DicAction.calc_line_dag(line, DicAction.get_dag(line))
                old_start = 0
                seg_line = ''
                while old_start < len(line):
                    new_start = line_route[old_start][1] + 1
                    seg_line += line[old_start:new_start] + '/ '
                    old_start = new_start
                seg_line = HMM.oov_line(seg_line) if seg_line else ''  # 未登录词识别
                mwf_file.write(seg_line + '\n')
