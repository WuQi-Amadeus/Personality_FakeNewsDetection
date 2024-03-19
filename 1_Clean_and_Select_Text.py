import os
import shutil
import json
from jsonpath import jsonpath
import re


def copyjson(srcfile, dstpath):
    if os.path.isfile(srcfile):
        fpath, fname = os.path.split(srcfile)
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)
        shutil.copy(srcfile, dstpath + fname)
        print('copy {}->{}'.format(srcfile, dstpath + fname))


# srcfile 需要复制、移动的文件
# dstpath 目的地址
src_dir = 'rumdect/Weibo/'
dst_label0 = 'rumdect/label0/'
dst_label1 = 'rumdect/label1/'
fileHandler = open('rumdect/Weibo.txt', 'r', encoding='utf-8')

# 将微博新闻分为真新闻（label0）与假新闻（label1）
while True:
    line = fileHandler.readline().strip()
    if not line:
        break
    l = line.split('\t')
    l2 = l[2].split(' ')
    if l[1] == 'label:0':
        for f in l2:
            src_file = src_dir + f + '.json'
            copyjson(src_file, dst_label0)
    elif l[1] == 'label:1':
        for f in l2:
            src_file = src_dir + f + '.json'
            copyjson(src_file, dst_label1)
fileHandler.close()


# 文本清洗
def clean(text):
    text = re.sub(r"(回复)?(//)?\s*@\S*?\s*(:| |$)", " ", text)  # 去除正文中的@和回复/转发中的用户名
    text = re.sub(r"\[\S+\]", "", text)      # 去除表情符号
    # text = re.sub(r"#\S+#", "", text)      # 保留话题内容
    URL_REGEX = re.compile(
        r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))',
        re.IGNORECASE)
    text = re.sub(URL_REGEX, "", text)       # 去除网址
    text = text.replace("转发微博", "")       # 去除无意义的词语
    text = text.replace("轉發微博", "")
    text = text.replace("http://", "")
    text = re.sub(r"\s+", " ", text)  # 合并正文中过多的空格
    return text.strip()


path0 = 'rumdect/label0/'
path1 = 'rumdect/label1/'
wpath0 = 'rumdect/data/新闻与评论文本/真新闻/'
wpath1 = 'rumdect/data/新闻与评论文本/假新闻/'

for file_name in os.listdir(path0):
    print(file_name)
    readpath = path0 + file_name
    writepath = wpath0 + file_name[:-5] + '.txt'
    texts = ['0']
    with open(readpath, 'r', encoding='utf-8') as f:
        res = json.load(f)
        count = 0
        for d in res:
            ru = clean(jsonpath(d, '$.text')[0])
            if ru not in ['', '。']:
                texts.append(ru)
                count += 1
            if count == 81:
                break
    f.close()
    with open(writepath, 'w', newline='', encoding='utf-8') as f:
        for text in texts:
            f.write(text + '\n')
    f.close()

for file_name in os.listdir(path1):
    print(file_name)
    readpath = path1 + file_name
    writepath = wpath1 + file_name[:-5] + '.txt'
    texts = ['1']
    with open(readpath, 'r', encoding='utf-8') as f:
        res = json.load(f)
        count = 0
        for d in res:
            ru = clean(jsonpath(d, '$.text')[0])
            if ru not in ['', '。']:
                texts.append(ru)
                count += 1
            if count == 81:
                break
    f.close()
    with open(writepath, 'w', newline='', encoding='utf-8') as f:
        for text in texts:
            f.write(text + '\n')
    f.close()

