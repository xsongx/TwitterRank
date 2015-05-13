#!/usr/bin/python2
# -*-coding:utf-8-*-

# Copyright (c) 2014 lufo <lufo816@gmail.com>
import lda
import numpy as np
import re
import StopWords
import scipy.stats

stop_word_list = StopWords.stop_word_list


def text_parse(big_string):
    """
    从字符串中提取处它的所有单词
    :param big_string:字符串
    :return:列表，所有的出现过的单词，可重复
    """
    list_of_tokens = re.split(r'\W*', big_string)
    return [tok.lower() for tok in list_of_tokens if len(tok) > 2]


def create_vocab_list():
    """
    获得词汇表
    :return:列表，每个元素是一个词汇
    """
    vocab_list = []
    with open('dict.txt') as dict:
        vocab_list = [word.lower().strip() for word in dict if (word.lower().strip() + ' ' not in stop_word_list)]
    return vocab_list


def normalize(mat):
    '''
    将矩阵每一行归一化(一范数为1)
    :param mat: 矩阵
    :return: list,行归一化的矩阵
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
    # 下列三行代码为使用 KL 散度衡量相似度
    # pk = [row_normalized_dt[i][t]]
    # qk = [row_normalized_dt[j][t]]
    # sim = 1 - (scipy.stats.entropy(pk, qk) + scipy.stats.entropy(qk, pk)) / 2
    return sim


def get_Pt(t, samples, tweets_list, friends_tweets_list, row_normalized_dt, relationship):
    '''
    获得Pt,Pt[i][j]表示i关注j，在主题t下i受到j影响的概率
    '''
    Pt = []
    for i in xrange(samples):
        friends_tweets = friends_tweets_list[i]
        temp = []
        for j in xrange(samples):
            if relationship[j][i] == 1:
                if friends_tweets != 0:
                    temp.append(float(tweets_list[j]) / float(friends_tweets) * get_sim(t, i, j, row_normalized_dt))
                else:
                    temp.append(0.0)
            else:
                temp.append(0.0)
        Pt.append(temp)
    return Pt


def get_TRt(gamma, Pt, Et, iter=1000, tolerance=1e-16):
    '''
    获得TRt，在t topic下每个用户的影响力矩阵
    :param gamma: 获得 TRt 的公式中的调节参数
    :param Pt: Pt 矩阵,Pt[i][j]表示i关注j，在主题t下i受到j影响的概率
    :param Et: Et 矩阵,Et[i]代表用户 i 对主题 t 的关注度,已经归一化,所有元素相加为1
    :param iter: 最大迭代数
    :param tolerance: TRt迭代后 与迭代前欧氏距离小于tolerance时停止迭代
    :return: TRt,TRt[i]代表在主题 t 下用户 i 的影响力
    '''
    TRt = np.mat(Et).transpose()
    old_TRt = TRt
    i = 0
    # np.linalg.norm(old_TRt,new_TRt)
    while i < iter:
        TRt = gamma * (np.dot(np.mat(Pt), TRt)) + (1 - gamma) * np.mat(Et).transpose()
        euclidean_dis = np.linalg.norm(TRt - old_TRt)
        # print 'dis', dis
        if euclidean_dis < tolerance:
            break
        old_TRt = TRt
        i += 1
    return TRt


def get_doc_list(samples):
    """
    得到一个列表,每个元素为一片文档
    :param samples: 文档的个数
    :return: list,每个元素为一篇文档
    """
    doc_list = []
    for i in xrange(1, samples + 1):
        with open('tweet_cont/tweet_cont_%d.txt' % i) as fr:
            temp = text_parse(fr.read())
        word_list = [word.lower() for word in temp if (word + ' ' not in stop_word_list and not word.isspace())]
        doc_list.append(word_list)
    return doc_list


def get_feature_matrix(doc_list, vocab_list):
    """
    获得每篇文档的特征矩阵,每个词作为一个特征
    :param doc_list: list,每个元素为一篇文档
    :param vocab_list: list，词汇表，每个元素是一个词汇
    :return: i行j列list，i为样本数，j为特征数，feature_matrix_ij表示第i个样本中特征j出现的次数
    """
    feature_matrix = []
    # word_index 为字典,每个 key 为单词,value 为该单词在 vocab_list 中的下标
    word_index = {}
    for i in xrange(len(vocab_list)):
        word_index[vocab_list[i]] = i
    for doc in doc_list:
        temp = [0 for i in xrange(len(vocab_list))]
        for word in doc:
            if word in word_index:
                temp[word_index[word]] += 1
        feature_matrix.append(temp)
    return feature_matrix


def get_tweets_list():
    """
    获取每个用户发过的 tweet 数量
    :return: list,第 i 个元素为第 i 个用户发过的 tweet 数
    """
    tweets_list = []
    with open('number_of_tweets.txt') as fr:
        for line in fr.readlines():
            tweets_list.append(int(line))
    return tweets_list


def get_relationship(samples):
    """
    得到用户关系矩阵
    :param samples: 用户的个数
    :return: i行j列,relationship[i][j]=1表示j关注i
    """
    relationship = []
    for i in xrange(1, samples + 1):
        with open('follower/follower_%d.txt' % i) as fr:
            temp = []
            for line in fr.readlines():
                temp.append(int(line))
        relationship.append(temp)
    return relationship


def get_friends_tweets_list(samples, relationship, tweets_list):
    """
    得到每个用户关注的所以用户发过的 tweet 数量之和
    :param samples: 用户的个数
    :param relationship: 用户关系矩阵,i行j列,relationship[i][j]=1表示j关注i
    :param tweets_list: list,第 i 个元素为第 i 个用户发过的 tweet 数
    :return: list,第 i 个元素为第 i 个用户关注的所有人发过的 tweet 数之和
    """
    friends_tweets_list = [0 for i in xrange(samples)]
    for j in xrange(samples):
        for i in xrange(samples):
            if relationship[i][j] == 1:
                friends_tweets_list[j] += tweets_list[i]
    return friends_tweets_list


def get_user_list():
    """
    获取用户 id 列表
    :return: list,第 i 个元素为用户 i 的 id
    """
    user = []
    with open('user_id.txt') as fr:
        for line in fr.readlines():
            user.append(line)
    return user


def get_TR(topics, samples, tweets_list, friends_tweets_list, row_normalized_dt, col_normalized_dt, relationship,
           gamma=0.2, tolerance=1e-16):
    """
    获取 TR 矩阵,代表每个主题下每个用户的影响力
    :param topics: 主题数
    :param samples: 用户数
    :param tweets_list: list,第 i 个元素为第 i 个用户发过的 tweet 数
    :param friends_tweets_list: list,第 i 个元素为第 i 个用户关注的所有人发过的 tweet 数之和
    :param row_normalized_dt: dt 的行归一化矩阵
    :param col_normalized_dt: dt 的列归一化矩阵
    :param relationship: i行j列,relationship[i][j]=1表示j关注i
    :param gamma: 获得 TRt 的公式中调节参数
    :param tolerance: TRt迭代后 与迭代前欧氏距离小于tolerance时停止迭代
    :return: list,TR[i][j]为第 i 个主题下用户 j 的影响力
    """
    TR = []
    for i in xrange(topics):
        Pt = get_Pt(i, samples, tweets_list, friends_tweets_list, row_normalized_dt, relationship)
        Et = col_normalized_dt[i]
        TR.append(np.array(get_TRt(gamma, Pt, Et, tolerance)).reshape(-1, ).tolist())
    return TR


def get_TR_sum(TR, samples, topics):
    """
    获取总的 TR 矩阵,有 i 个元素,TR_sum[i]为用户 i 在所有主题下影响力之和
    :param TR: list,TR[i][j]为第 i 个主题下用户 j 的影响力
    :param samples: 用户数
    :param topics: 主题数
    :return: list,有 i 个元素,TR_sum[i]为用户 i 在所有主题下影响力之和
    """
    TR_sum = [0 for i in xrange(samples)]
    for i in xrange(topics):
        for j in xrange(samples):
            TR_sum[j] += TR[i][j]
    TR_sum.sort()
    return TR_sum


def get_lda_model(samples, topics, n_iter):
    """
    获得训练后的 LDA 模型
    :param samples: 文档数
    :param topics: 主题数
    :param n_iter: 迭代数
    :return: model,训练后的 LDA 模型
             vocab_list,列表，表示这些文档出现过的所有词汇，每个元素是一个词汇
    """
    doc_list = get_doc_list(samples)
    vocab_list = create_vocab_list()
    feature_matrix = get_feature_matrix(doc_list, vocab_list)
    model = lda.LDA(n_topics=topics, n_iter=n_iter)
    model.fit(np.array(feature_matrix))
    return model, vocab_list


def print_topics(model, vocab_list, n_top_words=5):
    """
    输出模型中每个 topic 对应的前 n 个单词
    :param model:  lda 模型
    :param vocab_list: 列表，表示这些文档出现过的所有词汇，每个元素是一个词汇
    """
    topic_word = model.topic_word_
    # print 'topic_word',topic_word
    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(vocab_list)[np.argsort(topic_dist)][:-n_top_words:-1]
        print('Topic {}: {}'.format(i + 1, ' '.join(topic_words)))


def get_TR_using_DT(dt, samples, topics=5, gamma=0.2, tolerance=1e-16):
    """
    已知 DT 矩阵得到 TR 矩阵
    :param dt: dt 矩阵代表文档的主题分布,dt[i][j]代表文档 i 中属于主题 j 的比重
    :param samples: 文档数
    :param topics:  主题数
    :param gamma: 获得 TRt 的公式中调节参数
    :param tolerance: TRt迭代后 与迭代前欧氏距离小于tolerance时停止迭代
    :return TR: list,TR[i][j]为第 i 个主题下用户 j 的影响力
    :return TR_sum: list,有 i 个元素,TR_sum[i]为用户 i 在所有主题下影响力之和
    """
    row_normalized_dt = normalize(dt)
    # col_normalized_dt为dt每列归一化的转置，之所以取转置是为了取dt的归一化矩阵的每一行更方便
    col_normalized_dt_array = np.array(normalize(dt.transpose()))
    col_normalized_dt = col_normalized_dt_array.reshape(col_normalized_dt_array.shape).tolist()
    tweets_list = get_tweets_list()
    relationship = get_relationship(samples)
    friends_tweets_list = get_friends_tweets_list(samples, relationship, tweets_list)
    user = get_user_list()
    TR = get_TR(topics, samples, tweets_list, friends_tweets_list, row_normalized_dt, col_normalized_dt, relationship,
                gamma, tolerance)
    for i in xrange(topics):
        print TR[i]
        print user[TR[i].index(max(TR[i]))]
    TR_sum = get_TR_sum(TR, samples, topics)
    return TR, TR_sum


def get_doc_topic_distribution_using_lda_model(model, feature_matrix):
    """
    使用训练好的 LDA 模型得到新文档的主题分布
    :param model: lda 模型
    :param feature_matrix: i行j列list，i为样本数，j为特征数，feature_matrix[i][j]表示第i个样本中特征j出现的次数
    :return:
    """
    return model.transform(np.array(feature_matrix), max_iter=100, tol=0)


def using_lda_model_test_other_data(topics=5, n_iter=100, num_of_train_data=10, num_of_test_data=5, gamma=0.2,
                                    tolerance=1e-16):
    """
    训练 LDA 模型然后用训练好的 LDA 模型得到新文档的主题然后找到在该文档所对应的主题中最有影响力的用户
    :param topics:  LDA 主题数
    :param n_iter:  LDA 模型训练迭代数
    :param num_of_train_data: 训练集数据量
    :param num_of_test_data: 测试集数据量
    :param gamma: 获得 TRt 的公式中调节参数
    :param tolerance: TRt迭代后 与迭代前欧氏距离小于tolerance时停止迭代
    """
    model, vocab_list = get_lda_model(samples=num_of_train_data, topics=topics, n_iter=n_iter)
    dt = model.doc_topic_
    print_topics(model, vocab_list, n_top_words=5)
    TR, TR_sum = get_TR_using_DT(dt, samples=num_of_train_data, topics=topics, gamma=gamma, tolerance=tolerance)
    doc_list = get_doc_list(samples=num_of_test_data)
    feature_matrix = get_feature_matrix(doc_list, vocab_list)
    dt = get_doc_topic_distribution_using_lda_model(model, feature_matrix)
    # doc_user[i][j]表示第 i 个文本与第 j 个用户的相似度
    doc_user = np.dot(dt, TR)
    user = get_user_list()
    for i, doc in enumerate(doc_user):
        print user[i], user[list(doc).index(max(doc))]


def twitter_rank(topics=5, n_iter=100, samples=30, gamma=0.2, tolerance=1e-16):
    """
    对文档做twitter rank
    :param topics: 主题数
    :param n_iter: 迭代数
    :param samples: 文档数
    :param gamma: 获得 TRt 的公式中调节参数
    :param tolerance: TRt迭代后 与迭代前欧氏距离小于tolerance时停止迭代
    :return:
    """
    model, vocab_list = get_lda_model(samples, topics, n_iter)
    # topic_word为i行j列array，i为主题数，j为特征数，topic_word_ij表示第i个主题中特征j出现的比例
    print_topics(model, vocab_list, n_top_words=5)
    # dt 矩阵代表文档的主题分布,dt[i][j]代表文档 i 中属于主题 j 的比重
    dt = np.mat(model.doc_topic_)
    TR, TR_sum = get_TR_using_DT(dt, samples, topics, gamma, tolerance)


def main():
    twitter_rank()
    # using_lda_model_test_other_data()


if __name__ == '__main__':
    main()
