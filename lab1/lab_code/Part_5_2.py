from math import log

from lab_code import Part_5_1, Part_1
from lab_code.Part_5_3 import HMM

Train_File = '../io_file/hmm/train.txt'  # 用于训练参数的文本文件


class DicAction:
    words_dic = {}  # 格式：{'玩':'北京':1,'玩':'BOS':'3'}表示'北京玩'和'玩的不错'

    @staticmethod  # 训练文本为hmm文件夹下的train.txt，用于生成二元文法的词典
    def gene_bi_dic(train_path=Train_File, dic_path='../io_file/dic/bigram_dic.txt'):
        with open(train_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            if line == '\n':
                continue
            words = line.split()  # 一行初步处理得到的词语列表
            words.append('EOS/ ')
            words.insert(0, 'BOS')
            for idx in range(1, len(words)):
                words[idx] = words[idx][1 if words[idx][0] == '[' else 0:words[idx].index('/')]
                if words[idx] not in DicAction.words_dic.keys():
                    DicAction.words_dic[words[idx]] = {}
                if words[idx - 1] not in DicAction.words_dic[words[idx]].keys():
                    DicAction.words_dic[words[idx]][words[idx - 1]] = 0
                DicAction.words_dic[words[idx]][words[idx - 1]] += 1  # 更新词频
        DicAction.words_dic = {k: DicAction.words_dic[k] for k in
                               sorted(DicAction.words_dic.keys())}
        with open(dic_path, 'w', encoding='utf-8') as f:
            for word in DicAction.words_dic:
                DicAction.words_dic[word] = {k: DicAction.words_dic[word][k] for k in
                                             sorted(DicAction.words_dic[word].keys())}
                for pre in DicAction.words_dic[word]:
                    f.write(word + ' ' + pre + ' ' + str(DicAction.words_dic[word][pre]) + '\n')

    @staticmethod  # 从离线词典构建其数据结构，前提是离线词典已经按照既定格式组织好
    def get_bi_dic(dic_path='../io_file/dic/bigram_dic.txt'):
        with open(dic_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            word, pre_word, freq = line.split()[0:3]
            if word not in DicAction.words_dic.keys():
                DicAction.words_dic[word] = {pre_word: int(freq)}
            else:
                DicAction.words_dic[word][pre_word] = int(freq)

    @staticmethod  # 用于计算一个已知上一个词的词log概率
    def get_log_pos(pre_word, word):
        pre_word_freq = Part_5_1.Word_Freq.get(pre_word, 0)  # 前词词频
        condition_word_freq = DicAction.words_dic.get(word, {}).get(pre_word, 0)  # 组合词频
        return log(condition_word_freq + 1) - log(pre_word_freq + Part_5_1.Word_Num_Count)

    # 最大概率分词，用于概率最大路径计算
    @staticmethod
    def calc_line_dag(line, dag):
        n = len(line) - 3  # 减去<EOS>的长度
        start = 3  # 跳过<BOS>从第一个字开始
        pre_graph = {'BOS': {}}  # 关键字为前词，值为对应的词和对数概率
        word_graph = {}  # 每个词节点存有上一个相连词的词图
        for x in dag[3]:  # 初始化前词为BOS的情况
            pre_graph['BOS'][(3, x + 1)] = DicAction.get_log_pos('BOS', line[3:x + 1])
        while start < n:  # 对每一个字可能的词生成下一个词的词典
            for idx in dag[start]:  # 遍历dag[start]中的每一个结束节点
                pre_word = line[start:idx + 1]  # 这个词是前一个词比如，'去北京'中的去
                temp = {}
                for next_end in dag[idx + 1]:
                    last_word = line[idx + 1:next_end + 1]
                    if line[idx + 1:next_end + 3] == 'EOS':  # 判断是否到达末尾
                        temp['EOS'] = DicAction.get_log_pos(pre_word, 'EOS')
                    else:
                        temp[(idx + 1, next_end + 1)] = DicAction.get_log_pos(pre_word, last_word)
                pre_graph[(start, idx + 1)] = temp  # 每一个以start开始的词都建立一个关于下一个词的词典
            start += 1
        pre_words = list(pre_graph.keys())  # 表示所有的前面的一个词
        for pre_word in pre_words:  # word_graph表示关键字对应的值为关键字的前词列表
            for word in pre_graph[pre_word].keys():  # 遍历pre_word词的后一个词word
                word_graph[word] = word_graph.get(word, list())
                word_graph[word].append(pre_word)
        pre_words.append('EOS')
        route = {}
        for word in pre_words:
            if word == 'BOS':
                route[word] = (0.0, 'BOS')
            else:
                pre_list = word_graph.get(word, list())  # 取得该词对应的前词列表
                route[word] = (-65507, 'BOS') if not pre_list else max(
                    (pre_graph[pre][word] + route[pre][0], pre) for pre in pre_list)
        return route

    @staticmethod
    def bigram(txt_path=Part_1.Test_File, bigram_path='../io_file/seg/seg_bigram.txt'):
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        with open(bigram_path, 'w', encoding='utf-8') as bigram_file:
            for line in lines:
                line = 'BOS' + line[:len(line) - 1] + 'EOS'
                dag = Part_5_1.DicAction.get_dag(line)
                line_route = DicAction.calc_line_dag(line, dag)
                seg_line = ''
                position = 'EOS'
                while True:
                    position = line_route[position][1]
                    if position == 'BOS':
                        break
                    seg_line = line[position[0]:position[1]] + '/ ' + seg_line
                seg_line = HMM.oov_line(seg_line) if seg_line else ''  # 未登录词处理
                bigram_file.write(seg_line + '\n')  # 写入分词文件中
