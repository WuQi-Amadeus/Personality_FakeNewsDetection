import os
import time
import random


def read_data(rpath):
    data = []
    for txt_name in os.listdir(rpath):
        read_path = rpath + txt_name
        with open(read_path, 'r', encoding='utf-8') as f:
            tmp = f.read().splitlines()
            if len(tmp) >= 82:
                data.append(tmp[:82])
    return data


def get_data():
    s = time.time()
    rpath0 = 'rumdect/data/新闻与评论文本/真新闻/'
    rpath1 = 'rumdect/data/新闻与评论文本/假新闻/'
    data0 = read_data(rpath0)
    data1 = read_data(rpath1)
    random.seed(6666)
    random.shuffle(data0)
    random.shuffle(data1)
    data0 = data0[:1280]  # 1351
    data1 = data1[:1280]  # 1286
    data = data0 + data1
    random.shuffle(data)
    slic = int(len(data) * 0.7)
    train_data = data[:slic]
    test_data = data[slic:]
    e = time.time()
    print('获取数据用时', e - s, '秒')
    return train_data, test_data


if __name__ == '__main__':
    train_data, test_data = get_data()

    print(len(train_data))
    print(len(train_data[0]))
    count0, count1 = 0, 0
    for d in train_data:
        clas = int(d[0])
        if clas == 0:
            count0 += 1
        else:
            count1 += 1
    print(count0, '; ', count1)

    print(len(test_data))
    print(len(test_data[0]))
    count0, count1 = 0, 0
    for d in test_data:
        clas = int(d[0])
        if clas == 0:
            count0 += 1
        else:
            count1 += 1
    print(count0, '; ', count1)

    with open('rumdect/data/train_text.txt', 'w', newline='', encoding='utf-8') as f:
        for d in train_data:
            f.write(str(d) + '\n')
    f.close()
    with open('rumdect/data/test_text.txt', 'w', newline='', encoding='utf-8') as f:
        for d in test_data:
            f.write(str(d) + '\n')
    f.close()
