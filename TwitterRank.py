#!/usr/bin/python2
# -*-coding:utf-8-*-

# Copyright (c) 2014 lufo <lufo816@gmail.com>
import lda
import numpy as np
import re
import StopWords

stop_word_list = StopWords.stop_word_list


def text_parse(big_string):
    """
    从字符串中提取处它的所有单词
    :param big_string:字符串
    :return:列表，所有的出现过的单词，可重复
    """
    list_of_tokens = re.split(r'\W*', big_string)
    return [tok.lower() for tok in list_of_tokens if len(tok) > 2]


def create_vocab_list(data_set):
    """
    提取出一系列文章出现过的所有词汇
    :param data_set:列表，每个元素也是列表，表示一篇文章，文章列表由单词组成
    :return:列表，表示这些文章出现过的所有词汇，每个元素是一个词汇
    """
    vocab_set = set([])
    for document in data_set:
        vocab_set = vocab_set | set(document)
    return list(vocab_set)


def normalize(mat):
    '''
    将矩阵每一行归一化(一范数为1)
    :param mat: 矩阵
    :return: list,归一化的矩阵
    '''
    row_normalized_mat = []
    for row_mat in mat:
        normalized_row = []
        row = np.array(row_mat).reshape(-1, ).tolist()
        row_sum = sum(row)
        for item in row:
            if row_sum != 0:
                normalized_row.append(float(item) / float(row_sum))
            else:
                normalized_row.append(0)
        row_normalized_mat.append(normalized_row)
    return row_normalized_mat


def get_sim(t, i, j, row_normalized_dt):
    '''
    获得sim(i,j)
    '''
    sim = 1.0 - abs(row_normalized_dt[i][t] - row_normalized_dt[j][t])
    return sim


def get_Pt(t, samples, tweets_list, friends_tweets_list, row_normalized_dt, relationship):
    '''
    获得Pt,Pt(i,j)表示i关注j，在主题t下i受到j影响的概率
    '''
    Pt = []
    for i in range(samples):
        friends_tweets = friends_tweets_list[i]
        temp = []
        for j in range(samples):
            if relationship[j][i] == 1:
                if friends_tweets != 0:
                    temp.append(float(tweets_list[j]) / float(friends_tweets) * get_sim(t, i, j, row_normalized_dt))
                else:
                    temp.append(0.0)
            else:
                temp.append(0.0)
        Pt.append(temp)
    return Pt


def get_TRt(gamma, Pt, Et):
    '''
    获得TRt，在t topic下每个用户的影响力矩阵
    '''
    TRt = np.mat(Et).transpose()
    iter = 0
    # np.linalg.norm(old_TRt,new_TRt)
    while iter < 100:
        TRt = gamma * (np.dot(np.mat(Pt), TRt)) + (1 - gamma) * np.mat(Et).transpose()
        iter += 1
    return TRt


def twitter_rank():
    doc_list = []
    samples = 100
    for i in range(1, samples + 1):
        temp = text_parse(open('tweet_cont/tweet_cont_%d.txt' % i).read())
        word_list = [word.lower() for word in temp if (word + ' ' not in stop_word_list and not word.isspace())]
        doc_list.append(word_list)
    vocab_list = create_vocab_list(doc_list)
    # x为i行j列list，i为样本数，j为特征数，Xij表示第i个样本中特征j出现的次数
    x = []
    for doc in doc_list:
        temp = []
        for vocab in vocab_list:
            temp.append(doc.count(vocab))
        x.append(temp)
    topics = 5
    model = lda.LDA(n_topics=topics, n_iter=1000, random_state=1)
    model.fit(np.array(x))
    # topic为i行j列array，i为主题数，j为特征数，Xij表示第i个主题中特征j出现的次数
    topic_word = model.topic_word_
    n_top_words = 5
    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(vocab_list)[np.argsort(topic_dist)][:-n_top_words:-1]
        print('Topic {}: {}'.format(i + 1, ' '.join(topic_words)))
    dt = np.mat(model.ndz_)
    print dt.shape
    row_normalized_dt = normalize(dt)
    # col_normalized_dt为dt每列归一化的转置，之所以取转置是为了取dt的归一化矩阵的每一行更方便
    col_normalized_dt_array = np.array(normalize(dt.transpose()))
    col_normalized_dt = col_normalized_dt_array.reshape(col_normalized_dt_array.shape).tolist()
    tweets_list = []
    fr = open('number_of_tweets.txt')
    for line in fr.readlines():
        tweets_list.append(int(line))
    fr.close()
    # relationship i行j列,relationship[i][j]=1表示j关注i
    relationship = []
    for i in range(1, samples + 1):
        fr = open('follower/follower_%d.txt' % i)
        temp = []
        for line in fr.readlines():
            temp.append(int(line))
        fr.close()
        relationship.append(temp)
    friends_tweets_list = [0 for i in range(samples)]
    for j in range(samples):
        for i in range(samples):
            if relationship[i][j] == 1:
                friends_tweets_list[j] += tweets_list[i]
    print friends_tweets_list
    user = []
    fr = open('result.txt')
    for line in fr.readlines():
        user.append(line)
    TR = []
    for i in range(topics):
        Pt = get_Pt(i, samples, tweets_list, friends_tweets_list, row_normalized_dt, relationship)
        Et = col_normalized_dt[i]
        TR.append(np.array(get_TRt(0.5, Pt, Et)).reshape(-1, ).tolist())
        print user[TR[i].index(max(TR[i]))]
    TR_sum = [0 for i in range(samples)]
    for i in range(topics):
        for j in range(samples):
            TR_sum[j] += TR[i][j]
    TR_sum.sort()
    for i in TR_sum:
        print user[TR_sum.index(i)]


def main():
    twitter_rank()


if __name__ == '__main__':
    main()