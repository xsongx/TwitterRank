#!/usr/bin/python2
# -*-coding:utf-8-*-

# Copyright (c) 2014 lufo <lufo816@gmail.com>
import tweepy
import string
import time
import random
import FriendList


def get_tweet(api, user_list):
    '''
    获取用户发过的所有tweet内容
    :param api: tweepy类，用于获取twitter信息
    :param user_list: 前100用户列表
    '''
    i = 0
    for user in user_list:
        i += 1
        fw = open('tweet_cont/tweet_cont_%d.txt' % i, 'w+')
        public_tweets = api.user_timeline(id=user, count=200)
        for tweet in public_tweets:
            for char in tweet.text:
                if char.isalnum() and char.encode("utf-8") in string.printable:
                    print char
                    fw.write(char)
                elif char.isspace():
                    fw.write('\n')
        fw.close()


def get_tweets_count(api, user_list):
    '''
    获取这些用户发过的tweet数
    :param api: tweepy类，用于获取twitter信息
    :param user_list: 前100用户列表
    '''
    fw = open('number_of_tweets.txt', 'w+')
    for user in user_list:
        user_information = api.get_user(id=user)
        fw.write(str(user_information._json['statuses_count']) + '\n')
    fw.close()


def get_friends(api, user_list):
    '''
    获得每个用户关注的人
    :param api: tweepy类，用于获取twitter信息
    :param user_list: 前100用户列表
    :return: 列表，每个元素为该用户关注的人的列表
    '''
    friend_list = []
    fw_error = open('error.txt', 'a')
    for user in user_list:
        try:
            friend_list.append(api.friends_ids(id=user))
        except Exception, e:
            fw_error.write(str(e))
        time.sleep(61)
    fw_error.close()
    fw_friend_list = open('friend_list.py', 'w+')
    fw_friend_list.write(str(friend_list))
    fw_friend_list.close()
    return friend_list


def get_relationship(api, user_list):
    '''
    获取这些用户之间的关注关系，如果这个user被第i个人关注，它对应的文件里第i行为1
    :param api: tweepy类，用于获取twitter信息
    :param user_list: 前100用户列表
    '''
    # friend_list = get_friends(api, user_list)
    friend_list = FriendList.friend_list
    fw_error = open('error.txt', 'a')
    i = 0
    for user in user_list:
        i += 1
        fw = open('follower/follower_%d.txt' % i, 'w+')
        try:
            user_id = api.get_user(id=user)._json['id']
        except Exception, e:
            fw_error.write(str(e))
        time.sleep(7)
        for friend in friend_list:
            if user_id in friend:
                fw.write('1\n')
            else:
                fw.write('0\n')
        fw.close()
    fw_error.close()


def main():
    consumer_key = '3NysYEyPiQmMw3NswOAN03wX9'
    consumer_secret = 'ZL4ZgW3BsvnVISDq3kMUUrz6ExnaIRZrfY1dObIEUE9wxE89UI'
    access_token = '606101675-ym9bEgtnnOZat3VjAWDx4AWDewC4YCzBleQQlGkP'
    access_token_secret = 'jGp3dXodcBtwld0FzEHAZtH9ZanF93aw6SKgFz7k4thVU'
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)
    fr = open('result.txt')
    user_list = []
    for name in fr.readlines():
        user_list.append(name)
    get_tweet(api, user_list)
    # get_tweets_count(api, user_list)
    # get_friends(api, user_list)
    # get_relationship(api, user_list)


if __name__ == '__main__':
    main()