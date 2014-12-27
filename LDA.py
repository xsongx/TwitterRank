#!/usr/bin/python2
# -*-coding:utf-8-*-

# Copyright (c) 2014 lufo <lufo816@gmail.com>
import lda
import numpy as np
import re

stop_word_list = ['a ', 'able ', 'about ', 'above ', 'abroad ', 'according ', 'accordingly ', 'across ',
                  'actually ',
                  'adj ', 'after ', 'afterwards ', 'again ', 'against ', 'ago ', 'ahead ', "ain't ", 'all ',
                  'allow ',
                  'allows ', 'almost ', 'alone ', 'along ', 'alongside ', 'already ', 'also ', 'although ',
                  'always ',
                  'am ', 'amid ', 'amidst ', 'among ', 'amongst ', 'an ', 'and ', 'another ', 'any ', 'anybody ',
                  'anyhow ', 'anyone ', 'anything ', 'anyway ', 'anyways ', 'anywhere ', 'apart ', 'appear ',
                  'appreciate ', 'appropriate ', 'are ', "aren't ", 'around ', 'as ', "a's ", 'aside ', 'ask ',
                  'asking ', 'associated ', 'at ', 'available ', 'away ', 'awfully ', 'b ', 'back ', 'backward ',
                  'backwards ', 'be ', 'became ', 'because ', 'become ', 'becomes ', 'becoming ', 'been ',
                  'before ',
                  'beforehand ', 'begin ', 'behind ', 'being ', 'believe ', 'below ', 'beside ', 'besides ',
                  'best ',
                  'better ', 'between ', 'beyond ', 'both ', 'brief ', 'but ', 'by ', 'c ', 'came ', 'can ',
                  'cannot ',
                  'cant ', "can't ", 'caption ', 'cause ', 'causes ', 'certain ', 'certainly ', 'changes ',
                  'clearly ',
                  "c'mon ", 'co ', 'co. ', 'com ', 'come ', 'comes ', 'concerning ', 'consequently ', 'consider ',
                  'considering ', 'contain ', 'containing ', 'contains ', 'corresponding ', 'could ', "couldn't ",
                  'course ', "c's ", 'currently ', 'd ', 'dare ', "daren't ", 'definitely ', 'described ',
                  'despite ',
                  'did ', "didn't ", 'different ', 'directly ', 'do ', 'does ', "doesn't ", 'doing ', 'done ',
                  "don't ",
                  'down ', 'downwards ', 'during ', 'e ', 'each ', 'edu ', 'eg ', 'eight ', 'eighty ', 'either ',
                  'else ', 'elsewhere ', 'end ', 'ending ', 'enough ', 'entirely ', 'especially ', 'et ', 'etc ',
                  'even ', 'ever ', 'evermore ', 'every ', 'everybody ', 'everyone ', 'everything ', 'everywhere ',
                  'ex ', 'exactly ', 'example ', 'except ', 'f ', 'fairly ', 'far ', 'farther ', 'few ', 'fewer ',
                  'fifth ', 'first ', 'five ', 'followed ', 'following ', 'follows ', 'for ', 'forever ', 'former ',
                  'formerly ', 'forth ', 'forward ', 'found ', 'four ', 'from ', 'further ', 'furthermore ', 'g ',
                  'get ', 'gets ', 'getting ', 'given ', 'gives ', 'go ', 'goes ', 'going ', 'gone ', 'got ',
                  'gotten ',
                  'greetings ', 'h ', 'had ', "hadn't ", 'half ', 'happens ', 'hardly ', 'has ', "hasn't ", 'have ',
                  "haven't ", 'having ', 'he ', "he'd ", "he'll ", 'hello ', 'help ', 'hence ', 'her ', 'here ',
                  'hereafter ', 'hereby ', 'herein ', "here's ", 'hereupon ', 'hers ', 'herself ', "he's ", 'hi ',
                  'him ', 'himself ', 'his ', 'hither ', 'hopefully ', 'how ', 'howbeit ', 'however ', 'hundred ',
                  'i ',
                  "i'd ", 'ie ', 'if ', 'ignored ', "i'll ", "i'm ", 'immediate ', 'in ', 'inasmuch ', 'inc ',
                  'inc. ',
                  'indeed ', 'indicate ', 'indicated ', 'indicates ', 'inner ', 'inside ', 'insofar ', 'instead ',
                  'into ', 'inward ', 'is ', "isn't ", 'it ', "it'd ", "it'll ", 'its ', "it's ", 'itself ',
                  "i've ",
                  'j ', 'just ', 'k ', 'keep ', 'keeps ', 'kept ', 'know ', 'known ', 'knows ', 'l ', 'last ',
                  'lately ', 'later ', 'latter ', 'latterly ', 'least ', 'less ', 'lest ', 'let ', "let's ",
                  'like ',
                  'liked ', 'likely ', 'likewise ', 'little ', 'look ', 'looking ', 'looks ', 'low ', 'lower ',
                  'ltd ',
                  'm ', 'made ', 'mainly ', 'make ', 'makes ', 'many ', 'may ', 'maybe ', "mayn't ", 'me ', 'mean ',
                  'meantime ', 'meanwhile ', 'merely ', 'might ', "mightn't ", 'mine ', 'minus ', 'miss ', 'more ',
                  'moreover ', 'most ', 'mostly ', 'mr ', 'mrs ', 'much ', 'must ', "mustn't ", 'my ', 'myself ',
                  'n ',
                  'name ', 'namely ', 'nd ', 'near ', 'nearly ', 'necessary ', 'need ', "needn't ", 'needs ',
                  'neither ', 'never ', 'neverf ', 'neverless ', 'nevertheless ', 'new ', 'next ', 'nine ',
                  'ninety ',
                  'no ', 'nobody ', 'non ', 'none ', 'nonetheless ', 'noone ', 'no-one ', 'nor ', 'normally ',
                  'not ',
                  'nothing ', 'notwithstanding ', 'novel ', 'now ', 'nowhere ', 'o ', 'obviously ', 'of ', 'off ',
                  'often ', 'oh ', 'ok ', 'okay ', 'old ', 'on ', 'once ', 'one ', 'ones ', "one's ", 'only ',
                  'onto ',
                  'opposite ', 'or ', 'other ', 'others ', 'otherwise ', 'ought ', "oughtn't ", 'our ', 'ours ',
                  'ourselves ', 'out ', 'outside ', 'over ', 'overall ', 'own ', 'p ', 'particular ',
                  'particularly ',
                  'past ', 'per ', 'perhaps ', 'placed ', 'please ', 'plus ', 'possible ', 'presumably ',
                  'probably ',
                  'provided ', 'provides ', 'q ', 'que ', 'quite ', 'qv ', 'r ', 'rather ', 'rd ', 're ', 'really ',
                  'reasonably ', 'recent ', 'recently ', 'regarding ', 'regardless ', 'regards ', 'relatively ',
                  'respectively ', 'right ', 'round ', 's ', 'said ', 'same ', 'saw ', 'say ', 'saying ', 'says ',
                  'second ', 'secondly ', 'see ', 'seeing ', 'seem ', 'seemed ', 'seeming ', 'seems ', 'seen ',
                  'self ',
                  'selves ', 'sensible ', 'sent ', 'serious ', 'seriously ', 'seven ', 'several ', 'shall ',
                  "shan't ",
                  'she ', "she'd ", "she'll ", "she's ", 'should ', "shouldn't ", 'since ', 'six ', 'so ', 'some ',
                  'somebody ', 'someday ', 'somehow ', 'someone ', 'something ', 'sometime ', 'sometimes ',
                  'somewhat ',
                  'somewhere ', 'soon ', 'sorry ', 'specified ', 'specify ', 'specifying ', 'still ', 'sub ',
                  'such ',
                  'sup ', 'sure ', 't ', 'take ', 'taken ', 'taking ', 'tell ', 'tends ', 'th ', 'than ', 'thank ',
                  'thanks ', 'thanx ', 'that ', "that'll ", 'thats ', "that's ", "that've ", 'the ', 'their ',
                  'theirs ', 'them ', 'themselves ', 'then ', 'thence ', 'there ', 'thereafter ', 'thereby ',
                  "there'd ", 'therefore ', 'therein ', "there'll ", "there're ", 'theres ', "there's ",
                  'thereupon ',
                  "there've ", 'these ', 'they ', "they'd ", "they'll ", "they're ", "they've ", 'thing ',
                  'things ',
                  'think ', 'third ', 'thirty ', 'this ', 'thorough ', 'thoroughly ', 'those ', 'though ', 'three ',
                  'through ', 'throughout ', 'thru ', 'thus ', 'till ', 'to ', 'together ', 'too ', 'took ',
                  'toward ',
                  'towards ', 'tried ', 'tries ', 'truly ', 'try ', 'trying ', "t's ", 'twice ', 'two ', 'u ',
                  'un ',
                  'under ', 'underneath ', 'undoing ', 'unfortunately ', 'unless ', 'unlike ', 'unlikely ',
                  'until ',
                  'unto ', 'up ', 'upon ', 'upwards ', 'us ', 'use ', 'used ', 'useful ', 'uses ', 'using ',
                  'usually ',
                  'v ', 'value ', 'various ', 'versus ', 'very ', 'via ', 'viz ', 'vs ', 'w ', 'want ', 'wants ',
                  'was ', "wasn't ", 'way ', 'we ', "we'd ", 'welcome ', 'well ', "we'll ", 'went ', 'were ',
                  "we're ",
                  "weren't ", "we've ", 'what ', 'whatever ', "what'll ", "what's ", "what've ", 'when ', 'whence ',
                  'whenever ', 'where ', 'whereafter ', 'whereas ', 'whereby ', 'wherein ', "where's ",
                  'whereupon ',
                  'wherever ', 'whether ', 'which ', 'whichever ', 'while ', 'whilst ', 'whither ', 'who ',
                  "who'd ",
                  'whoever ', 'whole ', "who'll ", 'whom ', 'whomever ', "who's ", 'whose ', 'why ', 'will ',
                  'willing ', 'wish ', 'with ', 'within ', 'without ', 'wonder ', "won't ", 'would ', "wouldn't ",
                  'x ',
                  'y ', 'yes ', 'yet ', 'you ', "you'd ", "you'll ", 'your ', "you're ", 'yours ', 'yourself ',
                  'yourselves ', "you've ", 'z ', 'zero ']


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
    new_TRt = np.mat(Et).transpose()
    iter = 0
    # np.linalg.norm(old_TRt,new_TRt)
    while iter < 100:
        old_TRt = new_TRt
        new_TRt = gamma * (np.dot(np.mat(Pt), old_TRt)) + (1 - gamma) * old_TRt
        iter += 1
    return new_TRt


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
    topics = 10
    model = lda.LDA(n_topics=topics, n_iter=1000, random_state=1)
    model.fit(np.array(x))
    # topic为i行j列array，i为主题数，j为特征数，Xij表示第i个主题中特征j出现的次数
    topic_word = model.topic_word_
    n_top_words = 5
    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(vocab_list)[np.argsort(topic_dist)][:-n_top_words:-1]
        print('Topic {}: {}'.format(i, ' '.join(topic_words)))
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
        TR.append(np.array(get_TRt(0.02, Pt, Et)).reshape(-1, ).tolist())
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