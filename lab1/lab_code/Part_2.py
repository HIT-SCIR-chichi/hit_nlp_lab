from lab_code import Part_1

Max_Len = 0  # 保存最大词长
Words = []  # 用于保存词典中的词，具有唯一性


def get_dic(dic_path='../io_file/dic/dic.txt'):  # 从文本中读取词典文件，构造Words和Max_Len的值
    global Max_Len
    with open(dic_path, 'r', encoding='utf-8') as dic_file:
        lines = dic_file.readlines()  # 读取词典中的词
    for line in lines:
        Words.append(line[0:len(line) - 1])  # 将该词加入词典列表中
        Max_Len = len(line) - 1 if len(line) - 1 > Max_Len else Max_Len  # 更新最大词长


class StrMatch:
    @staticmethod  # 前向最大匹配机械分词，采用最简单的、代码行数最少的列表储存所有词
    def fmm(txt_path=Part_1.Test_File, fmm_path='../io_file/seg/seg_fmm.txt'):
        with open(txt_path, 'r', encoding='utf-8') as txt_file:
            txt_lines = txt_file.readlines()
        with open(fmm_path, 'w', encoding='utf-8') as fmm_file:
            for line in txt_lines:
                seg_line, line = '', line[:len(line) - 1]  # 去掉最后一个换行符
                while len(line) > 0:
                    try_word = line[0:len(line) if len(line) < Max_Len else Max_Len]
                    while try_word not in Words:
                        if len(try_word) == 1:  # 字串长度为1，跳出循环
                            break
                        try_word = try_word[0:len(try_word) - 1]  # 继续减小词长
                    line = line[len(try_word):]  # 更新剩余的待分词行
                    seg_line += try_word + '/ '  # 得到一个分词结果
                fmm_file.write(Part_1.pre_line(seg_line) + '\n')  # 写入换行符

    @staticmethod  # 逆向最大匹配分词，采用最简单的列表存储所有的不相同的词
    def bmm(txt_path=Part_1.Test_File, bmm_path='../io_file/seg/seg_bmm.txt'):
        with open(txt_path, 'r', encoding='utf-8') as txt_file:
            txt_lines = txt_file.readlines()
        with open(bmm_path, 'w', encoding='utf-8') as bmm_file:
            for line in txt_lines:
                line = line[:len(line) - 1]  # 去掉最后一个换行符
                seg_list = []  # 保存后向匹配得到的一个词
                while len(line) > 0:
                    try_word = line if len(line) < Max_Len else line[len(line) - Max_Len:]
                    while try_word not in Words:
                        if len(try_word) == 1:  # 字串长度为1，跳出循环
                            break
                        try_word = try_word[1:]  # 截取头部的一个词
                    seg_list.insert(0, try_word + '/ ')  # 暂时将分词的结果保存
                    line = line[:len(line) - len(try_word)]  # 更新剩余的待分词行
                bmm_file.write(Part_1.pre_line(''.join(seg_list)) + '\n')  # 写入分词的一行
