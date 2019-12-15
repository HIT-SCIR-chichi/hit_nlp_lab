from config import Answer_2, Answer_3, Polarity
from opi_classify.opi_train import create_dic
from gensim.models.word2vec import Word2Vec
from keras.models import load_model
import jieba
import csv


def get_data():
    idx_lst, asp_lst, opi_lst, cat_lst = [], [], [], []
    with open(Answer_2, 'r', encoding='utf-8') as f:
        input_lines = csv.reader(f)
        for line in input_lines:
            idx_lst.append(int(line[0]))
            asp_lst.append(line[1])
            opi_lst.append(line[2])
            cat_lst.append(line[3])
    return idx_lst, asp_lst, opi_lst, cat_lst


def run_op_model(opi_lst, asp_lst):
    embed_model = Word2Vec.load('../model/opi/word2vec.pkl')  # 加载词向量模型
    model = load_model('../model/opi/opi_classify.h5')  # 加载模型结构和权重
    opi_lst = [jieba.lcut(line) for line in opi_lst]  # 将观点词分词
    # opi_lst = [jieba.lcut(opi if opi != '_' else asp) for opi, asp in zip(opi_lst, asp_lst)]
    test_vec = create_dic(embed_model, opi_lst)[2]
    result = model.predict_classes(test_vec)
    return [Polarity[i] for i in result]


def write2file():
    idx_lst, asp_lst, opi_lst, cat_lst = get_data()
    pol_lst = run_op_model(opi_lst, asp_lst)
    with open(Answer_3, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(zip(idx_lst, asp_lst, opi_lst, cat_lst, pol_lst))


if __name__ == '__main__':
    write2file()
