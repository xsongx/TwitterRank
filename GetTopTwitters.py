#!/usr/bin/python2
# -*-coding:utf-8-*-

# Copyright (c) 2014 lufo <lufo816@gmail.com>
import re
import urllib2

url = 'http://twitaholic.com/'
id = '@(.*)<br />'  # 定义正则表达式的句法规则
id_re = re.compile(id)  # 通过compile函数“编译”正则表达式

res = urllib2.urlopen(url)
data = res.read()
id_info = id_re.findall(data)

f = open('result.txt', 'w+')

for item in id_info:
    print item
    f.write(item + '\n')