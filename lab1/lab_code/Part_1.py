Train_File = '../io_file/hmm/train.txt'  # 生成的训练集文本文件路径
Test_File = '../io_file/hmm/test.txt'
Std_File = '../io_file/hmm/std.txt'
K = 10  # 表示将标准分词文件的9/10作为训练集


# 用于生成词典，用于实验第1、2、3、4部分
def gene_dic(train_path=Train_File, dic_path='../io_file/dic/dic.txt'):
    max_len, word_set = 0, set()  # 保存最大词长，保存所有的词，要求具有唯一性且可排序
    with open(train_path, 'r', encoding='utf-8') as seg_file:
        lines = seg_file.readlines()  # 读取训练文本
    with open(dic_path, 'w', encoding='utf-8') as dic_file:
        for line in lines:
            for word in line.split():
                if '/m' in word:  # 不考虑将将量词加入
                    continue
                word = word[1 if word[0] == '[' else 0:word.index('/')]  # 去掉两个空格之间的非词字符
                word_set.add(word)  # 将词加入词典
                max_len = len(word) if len(word) > max_len else max_len  # 更新最大词长
        word_list = list(word_set)
        word_list.sort()  # 对此列表进行排序
        dic_file.write('\n'.join(word_list))  # 一个词一行
    return word_list, max_len


# 最后处理一行文本，输入为1/ 9/ 9/ 8/ 年/ .
def pre_line(line):
    punctuation = '-./'
    buffer, result = '', ''
    word_list = line.split('/ ')
    word_list = word_list[:len(word_list) - 1]
    for idx, word in enumerate(word_list):
        if word.isascii() or word in punctuation:  # 若是字母、数字或者英文标点
            buffer += word
            if idx + 1 == len(word_list):
                result += buffer + '/ '
        else:
            if buffer:
                result += buffer + '/ '
                buffer = ''
            result += word + '/ '
    return result


# 将标准文本按行取作为训练集
def gene_train_txt(std_seg_path='../io_file/199801_seg&pos.txt', train_path=Train_File,
                   std_path=Std_File, k=K):
    with open(std_seg_path, 'r', encoding='gbk')as std_seg_file:
        std_seg_lines = std_seg_file.readlines()
    std_lines = []  # 用于输出标准分词答案
    with open(train_path, 'w', encoding='utf-8') as train_file:
        for idx, line in enumerate(std_seg_lines):
            if idx % k != 0:
                train_file.write(line)  # 按照行数模K将该行作为训练行
            else:
                std_lines.append(line)
    with open(std_path, 'w', encoding='utf-8') as std_file:
        std_file.write(''.join(std_lines))
    gene_test_txt()  # 相应的更改测试文本


# 将标准文本剩下的文本作为测试文本并生成标准对比文本，不需单独运行，已在训练文本改变的时候后默认修改测试文件
def gene_test_txt(std_test_path='../io_file/199801_sent.txt', test_path=Test_File, k=K):
    with open(std_test_path, 'r', encoding='gbk')as std_seg_file:
        std_seg_lines = std_seg_file.readlines()
    with open(test_path, 'w', encoding='utf-8') as train_file:
        for idx, line in enumerate(std_seg_lines):
            if idx % k == 0:
                train_file.write(line)  # 按照行数模K将该行作为训练行


# 对自己的分词结果和标准结果进行比对，并输出结果到文件中
def score(std_seg_encoding='utf-8', my_seg_encoding='utf-8', score_path='../io_file/score.txt',
          std_seg_path='../io_file/199801_seg&pos.txt', my_seg_path='../io_file/seg/seg_mwf.txt'):
    print('本次评测得分将输出在文本中:\t' + score_path)
    precision, recall, f_value = calc(std_seg_path, my_seg_path, std_seg_encoding, my_seg_encoding)
    score_result = '标准文件:\t' + std_seg_path + '\n对比文件:\t' + my_seg_path + '\n'
    score_result += '准确率:\t' + str(precision * 100) + '%\n召回率:\t' + str(recall * 100) + '%\n'
    score_result += 'F值:\t' + str(f_value * 100) + '%\n\n'
    open(score_path, 'a', encoding='UTF-8').write(score_result)


# 用于比较两个文本的差别
def compare_diff(seg_path1='../io_file/seg_fmm.txt', seg_path2='../io_file/seg_fmm_1.txt'):
    file = open(seg_path1, 'r', encoding='utf-8')
    count = 1
    for line1 in open(seg_path2, 'r', encoding='utf_8'):
        line2 = file.readline()
        if line1 != line2:
            print('行号:\t' + str(count))
            print('应该输出:' + line1)
            print('实际输出:' + line2)
        count += 1


def pre_process_seg(seg_path, encoding):
    file = open(seg_path, 'r', encoding=encoding)
    seg_list = []  # 保存最后结果
    for line in file:
        if line == '\n':
            continue
        new_line = ''  # 保存处理过后的一行
        for word in line.split():
            new_line += word[1 if word[0] == '[' else 0:word.index('/')] + '/ '
        seg_list.append(new_line)
    return seg_list


def calc(std_seg_path, my_seg_path, std_seg_encoding, my_seg_encoding, k=1):
    std_seg_words, right_seg_words, my_seg_words = 0, 0, 0
    standard_lines = pre_process_seg(std_seg_path, std_seg_encoding)
    my_lines = pre_process_seg(my_seg_path, my_seg_encoding)
    for idx, line in enumerate(standard_lines):
        line_words = line.split('/ ')  # 取出标准的分词文本中每行的词语
        my_line_words = my_lines[idx].split('/ ')  # 取对比文本中每行的词语
        size1 = len(line_words) - 1
        size2 = len(my_line_words) - 1
        std_seg_words += size1
        my_seg_words += size2
        i = j = 0
        num1, num2 = len(line_words[0]), len(my_line_words[0])
        while i < size1 and j < size2:
            if num1 == num2:
                right_seg_words += 1
                if i == size1 - 1:
                    break
                i += 1
                j += 1
                num1 += len(line_words[i])
                num2 += len(my_line_words[j])
            else:
                while True:
                    if num1 < num2:
                        i += 1
                        num1 += len(line_words[i])
                    elif num1 > num2:
                        j += 1
                        num2 += len(my_line_words[j])
                    else:
                        if i < size1 - 1:
                            num1 += len(line_words[i + 1])
                            num2 += len(my_line_words[j + 1])
                        i += 1
                        j += 1
                        break
    precision = right_seg_words / float(std_seg_words)
    recall = right_seg_words / float(my_seg_words)
    f_value = (k * k + 1) * precision * recall / (k * k * precision + recall)
    return precision, recall, f_value
