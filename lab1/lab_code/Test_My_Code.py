"""本文件是为了测试本次实验所有代码的准确性和优越性，可以通过此文件调用相应实验部分的接口，实现快速的检验成果"""
"""如果训练文本改变的话，请按提示输入重新构建词典和数据结构"""


def test_part_1():  # 测试程序运行生成词典
    print('正在产生离线词典')
    from lab_code import Part_1
    if input('训练文本改变输入：T；否则输入：F\n') == 'T':  # 若训练文本改变，需要运行此程序
        Part_1.gene_train_txt()  # 重新生成训练文件
    Part_1.gene_dic()  # 运行产生离线词典的程序


def test_part_2():  # 测试代码第二部分，即最少代码量实现机械匹配分词
    print('正在测试最少代码量实现机械匹配分词')
    from lab_code import Part_1
    from lab_code import Part_2
    if input('训练文本改变输入：T；否则输入：F\n') == 'T':  # 若训练文本改变，需要运行此程序
        Part_1.gene_train_txt()  # 重新生成训练文本
        Part_2.Words, Part_2.Max_Len = Part_1.gene_dic()  # 产生词典数据结构
    else:  # 若训练文本未改变，需要运行此行
        Part_2.get_dic()  # 用于生成词典、确定最大词长
    Part_2.StrMatch.fmm()  # 前向最大匹配
    Part_2.StrMatch.bmm()  # 后向最大匹配


# 测试代码第三部分，即正反向分词效果分析，主要是准确率计算，分词结果输出在了io_file/score.txt文件中
def test_part_3():
    from lab_code.Part_3 import score
    print('正在测试正反向分词效果')
    score(std_encoding='utf-8', std_path='../io_file/hmm/std.txt',
          fmm_path='../io_file/seg/seg_fmm_1.txt', bmm_path='../io_file/seg/seg_bmm_1.txt')


# 运行模块4的机械匹配分词算法
def run_part_4():
    my_choice = True
    from lab_code.Part_4 import DicAction, StrMatch
    if input('训练文本改变输入：T；否则输入：F\n') == 'T':  # 若训练文本改变，需要运行此程序
        from lab_code import Part_1
        Part_1.gene_train_txt()  # 重新生成训练文件
        DicAction.Words_List, max_len = Part_1.gene_dic()  # 初始化词列表
        my_choice = False  # 表示需要重新构建词典
    fmm_root = DicAction.get_fmm_dic(choice=my_choice, dic_path='../io_file/dic/dic.txt')
    StrMatch.fmm(fmm_root)  # 前向最大匹配
    bmm_root = DicAction.get_bmm_dic(choice=my_choice, dic_path='../io_file/dic/dic.txt')
    StrMatch.bmm(bmm_root)  # 后向最大匹配


# 测试代码第四部分，即首先运行模块4的机械匹配算法优化后的机械匹配分词实践对比
def test_part_4():
    from lab_code import Part_3
    Part_3.time_optimize()  # 分析两者的运行速度快慢，并记录在文件中io_file/time_cost.txt


# 测试一元文法+未登录词识别
def test_part_5_1():
    is_train_file_changed = input('训练文本改变输入：T；否则输入：F\n')
    from lab_code import Part_1
    from lab_code.Part_5_3 import TRAIN
    from lab_code.Part_5_1 import DicAction
    if is_train_file_changed == 'T':  # 若训练文本改变，需要运行此程序
        print('重新训练参数，重新构建词典')
        Part_1.gene_train_txt()
        TRAIN.tag_txt()  # 对训练文本重新进行训练，并将训练得到的参数写入文本文件中
        DicAction.gene_uni_dic()  # 产生离线词典并获得在线数据结构，在训练文本更新时，需要运行此行
    else:  # 若训练文本未改变，需要运行此行
        print('读取训练好的参数')
        TRAIN.get_para()  # 读离线参数，得到必要的HMM参数
        DicAction.get_uni_dic()  # 读离线词典，得到必要的数据结构
    DicAction.mwf()  # 对文本文件进行分词
    print('正在对一元文法分词结果进行评价，评价结果输出在io_file/score.txt中')
    Part_1.score(my_seg_encoding='utf-8', score_path='../io_file/score.txt',
                 std_seg_encoding='utf-8', std_seg_path='../io_file/hmm/std.txt',
                 my_seg_path='../io_file/seg/seg_mwf.txt')


# 测试二元文法+未登录词识别
def test_part_5_2():
    is_train_file_changed = input('训练文本改变输入：T；否则输入：F\n')
    from lab_code import Part_1
    from lab_code.Part_5_2 import DicAction
    from lab_code.Part_5_3 import TRAIN
    from lab_code import Part_5_1
    if is_train_file_changed == 'T':  # 若训练文本改变，需要运行此程序
        print('重新训练参数，重新构建词典')
        Part_1.gene_train_txt()  # 生成训练文件
        TRAIN.tag_txt()  # 对训练文本重新进行训练，并将训练得到的参数写入文本文件中
        Part_5_1.DicAction.gene_uni_dic()  # 产生离线词典并获得在线数据结构，在训练文本更新时，需运行此行
        DicAction.gene_bi_dic()  # 当二元文法词典改变时，需要运行此行
    else:  # 训练文本未改变
        print('读取训练好的参数')
        TRAIN.get_para()  # 从文本中读取HMM的训练参数
        Part_5_1.DicAction.get_uni_dic()  # 必要的初始化，为了初始化一元文法模块中的Word_Freqs
        DicAction.get_bi_dic()  # 必要的初始化，为了初始化Bigram中的words_dic
    DicAction.bigram()
    print('正在对二元文法分词结果评价，评价结果输出在io_file/score.txt中')
    Part_1.score(my_seg_encoding='utf-8', score_path='../io_file/score.txt',
                 std_seg_encoding='utf-8', std_seg_path='../io_file/hmm/std.txt',
                 my_seg_path='../io_file/seg/seg_bigram.txt')


# 测试纯HMM分词
def test_part_5_3():
    is_train_file_changed = input('训练文本改变输入：T；否则输入：F\n')
    from lab_code.Part_5_3 import TRAIN, Word_Dic, HMM
    from lab_code import Part_1
    if is_train_file_changed == 'T':  # 若训练文本改变，需要运行此程序
        print('重新训练参数，重新构建词典')
        Part_1.gene_train_txt()  # 生成训练文件
        TRAIN.tag_txt()  # 对训练文本重新进行训练，并将训练得到的参数写入文本文件中
    else:  # 训练文本未改变
        print('读取训练好的参数')
        TRAIN.get_para()  # 从文本中读取HMM的训练参数
    Word_Dic.clear()
    HMM.hmm()  # 仅使用HMM分词
    print('正在对HMM分词结果评价，评价结果输出在io_file/score.txt中')
    Part_1.score(my_seg_encoding='utf-8', score_path='../io_file/score.txt',
                 std_seg_encoding='utf-8', std_seg_path='../io_file/hmm/std.txt',
                 my_seg_path='../io_file/seg/seg_hmm.txt')


if __name__ == '__main__':
    # test_part_1()  # 测试程序运行生成词典，词典产生文件为io_file/dic/dic.txt
    # test_part_2()  # 测试代码第二部分，即最少代码量实现机械匹配分词
    # test_part_3()  # 测试代码第三部分，即正反向分词效果分析，分词结果输出在了io_file/score.txt文件中
    # test_part_4()  # 测试代码第四部分，即首先运行模块4的机械匹配算法优化后的机械匹配分词实践对比
    # run_part_4()  # 运行模块4的机械匹配分词算法
    # test_part_5_1()  # 测试一元文法+未登录词识别
    test_part_5_2()  # 测试二元文法+未登录词识别
    # test_part_5_3()  # 测试纯HMM分词
