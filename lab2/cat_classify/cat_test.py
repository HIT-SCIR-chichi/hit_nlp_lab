from cat_classify.cat_train import create_dic, Categories, Input_File, Answer_File
from gensim.models.word2vec import Word2Vec
from keras.models import load_model
import jieba
import csv


def get_data():
    idx_lst, asp_lst, opi_lst = [], [], []
    with open(Input_File, 'r', encoding='utf-8') as f:
        input_lines = csv.reader(f)
        for line in input_lines:
            idx_lst.append(int(line[0]))
            asp_lst.append(line[1])
            opi_lst.append(line[2])
    return idx_lst, asp_lst, opi_lst


def run_cat_model(asp_lst, opi_lst):
    embed_model = Word2Vec.load('../model/cate/word2vec.pkl')  # 加载词向量模型
    model = load_model('../model/cate/cate_classify.h5')  # 加载模型结构和权重
    concat_lst = []  # 保存属性和观点的组合词，作为测试数据
    for asp, opi in zip(asp_lst, opi_lst):
        words = []
        if asp != '_':
            words.extend(jieba.lcut(asp))
        if opi != '_':
            words.extend(jieba.lcut(opi))
        concat_lst.append(words)
    embed_model.train(concat_lst, total_examples=len(concat_lst), epochs=embed_model.epochs)
    test_feed = create_dic(embed_model, concat_lst)[2]
    result = model.predict_classes(test_feed)
    return [Categories[i] for i in result]  # 返回值为列表，形如['使用体验', '物流', '功效', '价格']


def write2file():
    idx_lst, asp_lst, opi_lst = get_data()
    cat_lst = run_cat_model(asp_lst, opi_lst)
    with open(Answer_File, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(zip(idx_lst, asp_lst, opi_lst, cat_lst))


if __name__ == '__main__':
    write2file()  # 测试模型并将结果写入文件
