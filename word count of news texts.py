import os
import numpy as np
import matplotlib.pyplot as plt
import random

rpath0 = 'rumdect/data/新闻与评论文本/真新闻/'
rpath1 = 'rumdect/data/新闻与评论文本/假新闻/'
random.seed(6666)


# 统计新闻字数
def summarize_news(rpath):
    nums = []
    for file_name in os.listdir(rpath):
        file_path = rpath + file_name
        with open(file_path, 'r', encoding='utf-8') as f:
            texts = f.read().splitlines()
            if len(texts) >= 82:
                nums.append(len(texts[1]))
        f.close()
    random.shuffle(nums)
    nums = nums[:1280]
    max_nums = np.max(nums)
    min_nums = np.min(nums)
    mean_nums = np.mean(nums)
    print('新闻数量：', len(nums))
    print('新闻最大字数：', max_nums)
    print('新闻最小字数：', min_nums)
    print('新闻平均字数：', mean_nums)
    return nums


if __name__ == '__main__':
    nums0 = summarize_news(rpath0)
    nums1 = summarize_news(rpath1)
    plt.hist(nums0, bins=175, alpha=0.5, label='real')
    plt.hist(nums1, bins=175, alpha=0.5, label='fake')
    plt.xlabel('number of words')
    plt.ylabel('frequency')
    plt.title('Word count of This experiment news texts')
    plt.legend()
    plt.show()

