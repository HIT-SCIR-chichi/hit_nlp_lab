from keras.layers.core import Dense, Dropout, Activation
from sklearn.model_selection import train_test_split
from gensim.corpora.dictionary import Dictionary
from keras.layers.embeddings import Embedding
from gensim.models.word2vec import Word2Vec
from keras.preprocessing import sequence
from keras.layers.recurrent import LSTM
from keras.utils import to_categorical
from keras.models import Sequential
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import jieba

Categories = ['整体', '使用体验', '功效', '价格', '物流', '气味', '包装', '真伪', '服务', '成分', '其他',
              '尺寸', '新鲜度']  # 所有的类别标签(下标越小，出现频率越高)
Input_File = '../answer/task1_answer.csv'  # 提取关键词得到的文件
Answer_File = '../answer/task2_answer.csv'  # 属性词分类得到文件
Train_Labels = '../source/train_labels.csv'
Max_Len = 10  # 特征词+，+观点词经分词后的分词数目
Embed_dim = 100  # 观点的词中的每一个输出的向量维度
Min_Count = 1  # Word2Vec训练中的最小词频阈值
Epochs = 6  # 训练模型次数
Batch = 16  # 一次送入的数据量


def load_file():  # 获得属性词+观点词的组合词、以及分类
    f = pd.read_csv(Train_Labels)  # 读取训练标签文本，获得观点及对应的极性词
    data, cat_lst = [], []
    for idx in range(len(f)):
        line = f.iloc[idx]  # 取一行标注
        concat = line['AspectTerms'] + '，' + line['OpinionTerms']  # 合并属性词和观点词，并以逗号隔开
        concat = concat.strip('_').strip('，') if '_' in concat else concat
        data.append(concat)
        cat_lst.append(Categories.index(line['Categories']))
    return data, cat_lst


def word2vec_train(data):  # 训练得到词向量
    embed_model = Word2Vec(data, min_count=Min_Count, size=Embed_dim)  # 训练词向量模型
    embed_model.save('../model/cate/word2vec.pkl')  # 保存词向量模型
    word2idx, word2vec, data = create_dic(embed_model, data)
    return word2idx, word2vec, data


def create_dic(embed_model, data):  # 建立词与id，词与向量的索引
    gensim_dict = Dictionary()
    gensim_dict.doc2bow(embed_model.wv.vocab.keys(), allow_update=True)
    word2idx = {v: k + 1 for k, v in gensim_dict.items()}
    word2vec = {word: embed_model[word] for word in word2idx.keys()}

    def parse_data(concat_lst):
        result = []
        for op in concat_lst:  # 所有的观点词
            new_txt = []
            for word in op:  # 一个观点词中的一个分词
                new_txt.append(word2idx[word] if word in word2idx else 0)
            result.append(new_txt)
        return result

    data = parse_data(data)
    data = sequence.pad_sequences(data, maxlen=Max_Len)
    return word2idx, word2vec, data


def get_data(word2idx, word2vec, data, label):
    n_symbols = len(word2idx) + 1  # 因为频数小于min_count的词语索引为0，且存在不在词典中的词，所以+1
    embed_weight = np.zeros((n_symbols, Embed_dim))  # n_symbols*100的0矩阵
    for word, index in word2idx.items():  # 从索引为1的词语开始，用词向量填充矩阵
        embed_weight[index, :] = word2vec[word]  # 词向量矩阵，第一行是0向量
    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2)
    y_train = to_categorical(y_train, num_classes=len(Categories))  # 将一维n元素列表转化为n维13元素（独热表示）列表
    y_test = to_categorical(y_test, num_classes=len(Categories))
    return n_symbols, embed_weight, x_train, y_train, x_test, y_test


def main():
    data, cat_lst = load_file()  # 一维列表
    data = [jieba.lcut(line) for line in data]  # 分词
    word2idx, word2vec, data = word2vec_train(data)  # 训练embedding向量
    n, embed, x_train, y_train, x_test, y_test = get_data(word2idx, word2vec, data, cat_lst)

    model = Sequential()
    model.add(Embedding(n, Embed_dim, mask_zero=True, weights=[embed], input_length=Max_Len))
    model.add(LSTM(units=50, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(len(Categories)))
    model.add(Activation('softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    hist = model.fit(x_train, y_train, Batch, Epochs, validation_data=[x_test, y_test])
    model.save('../model/cate/cate_classify.h5')

    plt.plot(range(Epochs), hist.history['val_acc'], label='val_acc')
    plt.show()


if __name__ == '__main__':
    main()
