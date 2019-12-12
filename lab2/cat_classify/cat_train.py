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

Categories = ['整体', '使用体验', '功效', '价格', '物流', '气味', '包装', '真伪', '服务', '成分', '其他',
              '尺寸', '新鲜度']  # 所有的类别标签(下标越小，出现频率越高)
Input_File = '../answer/task1_answer.csv'  # 提取关键词得到的文件
Answer_File = '../answer/task2_answer.csv'  # 属性词分类得到文件
Train_Labels = '../source/train_labels.csv'
Max_Len = 10  # 特征词+观点词分词后的分词数目最大值
Embed_dim = 100  # 观点的词中的每一个输出的向量维度
Min_Count = 1  # Word2Vec训练中的最小词频阈值
Epochs = 6
Batch = 16


def load_file():  # 获得属性词+观点词的组合词、以及分类
    f = pd.read_csv(Train_Labels)  # 读取训练标签文本，获得观点及对应的极性词
    data, cat_lst = [], []  # 第一维表示组合词的数目，第二维表示组合词经分词后的数目
    for idx in range(len(f)):
        line = f.iloc[idx]  # 取一行标注
        words = []
        if line['AspectTerms'] != '_':
            words.extend(jieba.lcut(line['AspectTerms']))
        if line['OpinionTerms'] != '_':
            words.extend(jieba.lcut(line['OpinionTerms']))
        data.append(words)
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
    data = pad_sequences(data, maxlen=Max_Len)
    return word2idx, word2vec, data


def get_data(word2idx, word2vec, label):
    n_symbols = len(word2idx) + 1  # 因为频数小于min_count的词语索引为0，且存在不在词典中的词，所以+1
    embed = np.zeros((n_symbols, Embed_dim))  # n_symbols*100的0矩阵
    for word, index in word2idx.items():  # 从索引为1的词语开始，用词向量填充矩阵
        embed[index, :] = word2vec[word]  # 词向量矩阵，第一行是0向量
    label = to_categorical(label, num_classes=len(Categories))  # 将一维n元素列表转化为n维13元素（独热表示）列表
    return n_symbols, embed, label


def main():
    data, cat_lst = load_file()  # data为二维列表，cat_lst为一维列表
    word2idx, word2vec, data = word2vec_train(data)  # 训练embedding向量
    n, embed, label = get_data(word2idx, word2vec, cat_lst)
    model = Sequential()
    model.add(Embedding(n, Embed_dim, mask_zero=True, weights=[embed], input_length=Max_Len))
    model.add(LSTM(units=50, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(len(Categories)))
    model.add(Activation('softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    hist = model.fit(data, label, Batch, Epochs, validation_split=0.1)
    model.save('../model/cate/cate_classify.h5')

    plt.plot(range(Epochs), hist.history['val_acc'], label='val_acc')
    plt.show()


if __name__ == '__main__':
    main()
