from gensim.models import Word2Vec
import jieba
import itertools
import numpy as np
from Fix_Random_Seed import setup_seed

setup_seed(6666)
all_comments = []
with open('rumdect/data/train_text.txt', 'r', encoding='utf-8') as f:
    train_texts = f.read().splitlines()
    train_texts = [eval(t) for t in train_texts]
    all_comments += [t[2:] for t in train_texts]
f.close()
with open('rumdect/data/test_text.txt', 'r', encoding='utf-8') as f:
    test_texts = f.read().splitlines()
    test_texts = [eval(t) for t in test_texts]
    all_comments += [t[2:] for t in test_texts]
f.close()
corpus = list(itertools.chain(*all_comments))
tokenized_corpus = [list(jieba.cut(sentence)) for sentence in corpus]

# 训练Word2Vec模型
model = Word2Vec(tokenized_corpus, vector_size=100, window=2, min_count=1, sg=0)  # 使用CBOW模型


def sentence_vector(sentence, model):
    words = list(jieba.cut(sentence))
    word_vectors = [model.wv[word] for word in words if word in model.wv.key_to_index]
    if len(word_vectors) > 0:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(model.vector_size)


train_word2vec = []
for texts in train_texts:
    line = [texts[0], texts[1]]
    vector = np.zeros(model.vector_size)
    for comment in texts[2:]:
        vector += sentence_vector(comment, model)
    vector /= len(texts[2:])
    line.append(list(vector))
    print(line[0])
    train_word2vec.append(line)

test_word2vec = []
for texts in test_texts:
    line = [texts[0], texts[1]]
    vector = np.zeros(model.vector_size)
    for comment in texts[2:]:
        vector += sentence_vector(comment, model)
    vector /= len(texts[2:])
    line.append(list(vector))
    print(line[0])
    test_word2vec.append(line)

with open('rumdect/data/train_word2vec.txt', 'w', encoding='utf-8') as f:
    for x in train_word2vec:
        f.write(str(x) + '\n')
f.close()
with open('rumdect/data/test_word2vec.txt', 'w', encoding='utf-8') as f:
    for x in test_word2vec:
        f.write(str(x) + '\n')
f.close()
