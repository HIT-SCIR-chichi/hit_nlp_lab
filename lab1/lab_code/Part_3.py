import time

from lab_code import Part_1, Part_2


def score(score_path='../io_file/score.txt', std_path='../io_file/199801_seg&pos.txt',
          fmm_path='../io_file/seg/seg_fmm_1.txt', bmm_path='../io_file/seg/seg_bmm_1.txt',
          std_encoding='gbk', my_encoding='utf-8'):
    score_result = '本次评测得分\n'
    precision, recall, f_value = Part_1.calc(std_path, fmm_path, std_encoding, my_encoding)  # FMM
    score_result += 'FMM准确率:\t' + str(precision * 100) + '%\nFMM召回率:\t' + str(recall * 100) + '%'
    score_result += "\nFMM的F值:\t" + str(f_value * 100) + '%\n\n'
    precision, recall, f_value = Part_1.calc(std_path, bmm_path, std_encoding, my_encoding)  # BMM
    score_result += 'BMM准确率:\t' + str(precision * 100) + '%\nBMM召回率:\t' + str(recall * 100) + '%'
    score_result += "\nBMM的F值:\t" + str(f_value * 100) + "%\n\n"
    open(score_path, 'a', encoding='UTF-8').write(score_result)


def time_optimize(time_cost_path='../io_file/time_cost.txt'):  # 用于评价实验第四部分做的优化时间对比
    print('说明1\n实验第二部分为了达到最少代码量的要求使用了最基本的list数据结构\n' +
          '因此运行时间很长，约5h，若跑完fmm和bmm要11h，但我可保证代码正确性，强烈建议运行第4部分')
    print('说明2\n测试的前提为训练文本没有改变；若改变，请运行相应模块的内容来初始化系统')
    choice = input('输入2运行第2部分，输入4运行第4部分\n')
    out_put_str = ''
    if choice == '2':
        Part_2.get_dic()  # 用于生成词典、确定最大词长
        print('正在进行第2部分前向最大匹配运行，请等待')
        start_time = time.time()
        Part_2.StrMatch.fmm()  # 前向最大匹配
        fmm_time = time.time()
        print('正在进行第2部分后向最大匹配运行，请等待')
        Part_2.StrMatch.bmm()  # 后向最大匹配
        bmm_time = time.time()
        out_put_str += '优化前：为满足最少代码量的要求，采用简单的list查找的方式进行词是否在词典中的判断\n'
    else:
        from lab_code.Part_4 import DicAction, StrMatch
        fmm_root = DicAction.get_fmm_dic()
        bmm_root = DicAction.get_bmm_dic()
        print('正在进行第4部分前向最大匹配运行，请等待')
        start_time = time.time()
        StrMatch.fmm(fmm_root)  # 前向最大匹配
        fmm_time = time.time()
        print('正在进行第4部分后向最大匹配运行，请等待')
        StrMatch.bmm(bmm_root)  # 后向最大匹配
        bmm_time = time.time()
        out_put_str += '优化后：采用hash查找提高查找速度；采用前缀匹配，减少查找次数\n'
    out_put_str += 'FMM耗时\t' + str(fmm_time - start_time) + 's\n'
    out_put_str += 'BMM耗时\t' + str(bmm_time - fmm_time) + 's\n\n'
    with open(time_cost_path, 'a', encoding='utf-8')as time_cost_file:
        time_cost_file.write(out_put_str)
