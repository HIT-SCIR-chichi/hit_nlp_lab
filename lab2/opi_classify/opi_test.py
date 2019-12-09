from opi_classify.opi_train import create_dic, Polarity
from gensim.models.word2vec import Word2Vec
from keras.models import load_model
import jieba
import csv

Input_File = '../answer/task2_answer.csv'  # 提取关键词得到的文件
Answer_File = '../answer/task3_answer.csv'  # 属性词分类得到的文件


def get_data():
    idx_lst, asp_lst, opi_lst, cat_lst = [], [], [], []
    with open(Input_File, 'r', encoding='utf-8') as f:
        input_lines = csv.reader(f)
        for line in input_lines:
            idx_lst.append(int(line[0]))
            asp_lst.append(line[1])
            opi_lst.append(line[2])
            cat_lst.append(line[3])
    return idx_lst, asp_lst, opi_lst, cat_lst


def run_op_model(opi_lst):
    embed_model = Word2Vec.load('../model/opi/word2vec.pkl')  # 加载词向量模型
    model = load_model('../model/opi/opi_classify.h5')  # 加载模型结构和权重
    opi_lst = [jieba.lcut(line) for line in opi_lst]  # 将观点词分词
    embed_model.train(opi_lst, total_examples=len(opi_lst), epochs=embed_model.epochs)  # 再次训练模型
    test_vec = create_dic(embed_model, opi_lst)[2]
    result = model.predict_classes(test_vec)
    return [Polarity[i] for i in result]


def write2file():
    idx_lst, asp_lst, opi_lst, cat_lst = get_data()
    pol_lst = run_op_model(opi_lst)
    with open(Answer_File, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(zip(idx_lst, asp_lst, opi_lst, cat_lst, pol_lst))


if __name__ == '__main__':
    write2file()
