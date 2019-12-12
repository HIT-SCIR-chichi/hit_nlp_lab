from keras.layers import Dense, Dropout, Activation, Embedding, LSTM
from keras.preprocessing.sequence import pad_sequences
from gensim.corpora.dictionary import Dictionary
from gensim.models.word2vec import Word2Vec
from keras.utils import to_categorical
from keras.models import Sequential
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import jieba

Input_File = '../answer/task2_answer.csv'  # 提取关键词得到的文件
Answer_File = '../answer/task3_answer.csv'  # 属性词分类得到文件
Train_Labels = '../source/train_labels.csv'
Polarity = ['中性', '正面', '负面']
max_len = 10  # 观点词经分词后的分词数目
Embed_dim = 100  # 观点的词中的每一个输出的向量维度
Min_Count = 1  # Word2Vec训练中的最小词频阈值
Epochs = 6
Batch = 16


def load_file():  # 获得观点词及其对应的极性
    train_csv_label = pd.read_csv(Train_Labels)  # 读取训练标签文本，获得观点及对应的极性词
    data, label = [], []
    for idx in range(len(train_csv_label)):
        line = train_csv_label.iloc[idx]  # 取一行标注
        if line['OpinionTerms'] == '_':  # 不考虑观点词为空的情况
            continue
        data.append(line['OpinionTerms'])
        label.append(Polarity.index(line['Polarities']))
    return data, label


def word2vec_train(data):  # 训练得到词向量
    embed_model = Word2Vec(data, min_count=Min_Count, size=Embed_dim)  # 训练词向量模型
    embed_model.save('../model/opi/word2vec.pkl')  # 保存词向量模型
    word2idx, word2vec, data = create_dic(embed_model, data)
    return word2idx, word2vec, data


def create_dic(embed_model, data):  # 建立词与id，词与向量的索引
    gensim_dict = Dictionary()
    gensim_dict.doc2bow(embed_model.wv.vocab.keys(), allow_update=True)
    word2idx = {v: k + 1 for k, v in gensim_dict.items()}  # 所有频数超过10的词语的索引,(k->v)=>(v->k)
    word2vec = {word: embed_model[word] for word in word2idx.keys()}  # 频数超10词语词向量word->model(word)

    def parse_data(op_lst):
        result = []
        for op in op_lst:  # 所有的观点词
            new_txt = []
            for word in op:  # 一个观点词中的一个分词
                new_txt.append(word2idx[word] if word in word2idx else 0)
            result.append(new_txt)
        return result

    data = parse_data(data)
    data = pad_sequences(data, maxlen=max_len)
    return word2idx, word2vec, data


def get_data(word2idx, word2vec, label):
    n_symbols = len(word2idx) + 1  # 因为频数小于min_count的词语索引为0，且存在不在词典中的词，所以+1
    embed_weight = np.zeros((n_symbols, Embed_dim))  # n_symbols*100的0矩阵
    for word, index in word2idx.items():  # 从索引为1的词语开始，用词向量填充矩阵
        embed_weight[index, :] = word2vec[word]  # 词向量矩阵，第一行是0向量
    label = to_categorical(label, num_classes=3)  # 将一维n元素列表转化为n维3元素（独热表示）列表
    return n_symbols, embed_weight, label


def main():
    data, label = load_file()  # 一维列表
    data = [jieba.lcut(line) for line in data]  # 分词
    word2idx, word2vec, data = word2vec_train(data)  # 训练embedding向量
    n, embed, label = get_data(word2idx, word2vec, label)
    model = Sequential()
    model.add(Embedding(n, Embed_dim, mask_zero=True, weights=[embed], input_length=max_len))
    model.add(LSTM(units=50, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(len(Polarity)))
    model.add(Activation('softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    hist = model.fit(data, label, Batch, Epochs, validation_split=0.2)
    model.save('../model/opi/opi_classify.h5')

    plt.plot(range(Epochs), hist.history['val_acc'], label='val_acc')
    plt.show()


if __name__ == '__main__':
    main()
