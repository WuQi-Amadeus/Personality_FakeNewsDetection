import time
from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from Fix_Random_Seed import setup_seed

setup_seed(6666)


def get_data(path='rumdect/data/test_data.txt'):
    s = time.time()
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        tmp = f.read().splitlines()
        for d in tmp:
            data.append(eval(d))
    f.close()
    input1, input2, label = [], [], []
    for d in data:
        input1.append(d[1])
        label.append(eval(d[0]))
        tmp = []
        for i in range(3, len(d)):
            rev_pers = eval(d[i])
            rev_pers = map(lambda x: (x - 0.5) * 2, rev_pers)
            tmp += rev_pers
        input2.append(tmp)
    e = time.time()
    print('获取数据用时：{:.2f}秒'.format(e - s))
    return input1, input2, label


def get_bert(path='rumdect/data/test_bert.txt'):
    s = time.time()
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        tmp = f.read().splitlines()
        for d in tmp:
            data.append(eval(d))
    f.close()
    input1, input2, label = [], [], []
    for d in data:
        label.append(eval(d[0]))
        input1.append(d[1])
        input2.append(list(d[2]))
    e = time.time()
    print('获取数据用时：{:.2f}秒'.format(e - s))
    return input1, input2, label


def get_word(path='rumdect/data/test_word2vec.txt'):
    s = time.time()
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        tmp = f.read().splitlines()
        for d in tmp:
            data.append(eval(d))
    f.close()
    input1, input2, label = [], [], []
    for d in data:
        label.append(eval(d[0]))
        input1.append(d[1])
        input2.append(list(d[2]))
    e = time.time()
    print('获取数据用时：{:.2f}秒'.format(e - s))
    return input1, input2, label


def get_result(TP, TN, FP, FN):
    accuracy = float((TP + TN) / total)
    precision = float(TP / (TP + FP))
    recall = float(TP / (TP + FN))
    print()
    print('----------假新闻---真新闻--------')
    print('-预测正确--{}-----{}----'.format(TP, TN))
    print('-预测错误--{}-----{}----'.format(FP, FN))
    print('模型准确率：{:.2%}'.format(accuracy))
    print('模型查准率：{:.2%}'.format(precision))
    print('模型查全率：{:.2%}'.format(recall))
    print('模型F1得分：{:.2%}'.format(2 * precision * recall / (precision + recall)))


# BERT
class BERTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model1_name = './used_model/chinese-bert-wwm'
        self.tokenizer = BertTokenizer.from_pretrained(self.model1_name)
        self.model1 = BertModel.from_pretrained(self.model1_name)
        self.fc1 = nn.Linear(768, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, input1):
        # 前向传播流程
        batch_tokenized = self.tokenizer.batch_encode_plus(input1,
                                                           add_special_tokens=True,
                                                           truncation=True,
                                                           max_length=200,
                                                           padding='max_length',
                                                           return_tensors='pt')
        hidden_outputs = self.model1(input_ids=batch_tokenized['input_ids'],
                                     attention_mask=batch_tokenized['attention_mask'])
        out_puts = hidden_outputs[0][:, 0, :]
        x1 = F.relu(self.fc1(out_puts))
        predict_results = F.softmax(self.fc2(x1), dim=1)
        return predict_results


# BERT+Personality
class PersClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model1_name = './used_model/chinese-bert-wwm'
        self.tokenizer = BertTokenizer.from_pretrained(self.model1_name)
        self.model1 = BertModel.from_pretrained(self.model1_name)

        self.fc1 = nn.Linear(768 + 400, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, input1, input2):
        # 前向传播流程
        batch_tokenized = self.tokenizer.batch_encode_plus(input1,
                                                           add_special_tokens=True,
                                                           truncation=True,
                                                           max_length=200,
                                                           padding='max_length',
                                                           return_tensors='pt')
        hidden_outputs = self.model1(input_ids=batch_tokenized['input_ids'],
                                     attention_mask=batch_tokenized['attention_mask'])
        out_puts = hidden_outputs[0][:, 0, :]
        tensor_input2 = torch.tensor(input2)
        combined = torch.cat((out_puts, tensor_input2), dim=1)
        x1 = F.relu(self.fc1(combined))
        predict_results = F.softmax(self.fc2(x1), dim=1)
        return predict_results


# BERT+BERT
class AggrClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model1_name = './used_model/chinese-bert-wwm'
        self.tokenizer = BertTokenizer.from_pretrained(self.model1_name)
        self.model1 = BertModel.from_pretrained(self.model1_name)

        self.fc1 = nn.Linear(768*2, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, input1, input2):
        # 前向传播流程
        news_tokenized = self.tokenizer.batch_encode_plus(input1,
                                                          add_special_tokens=True,
                                                          truncation=True,
                                                          max_length=200,
                                                          padding='max_length',
                                                          return_tensors='pt')
        news_hidden_outputs = self.model1(input_ids=news_tokenized['input_ids'],
                                          attention_mask=news_tokenized['attention_mask'])
        news_outputs = news_hidden_outputs[0][:, 0, :]

        comments_outputs = torch.tensor(input2)
        inputs = torch.cat((news_outputs, comments_outputs), dim=1)

        x1 = F.relu(self.fc1(inputs))
        predict_results = F.softmax(self.fc2(x1), dim=1)
        return predict_results


# BERT+Word2Vec
class WordClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model1_name = './used_model/chinese-bert-wwm'
        self.tokenizer = BertTokenizer.from_pretrained(self.model1_name)
        self.model1 = BertModel.from_pretrained(self.model1_name)

        self.fc1 = nn.Linear(768+100, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, input1, input2):
        # 前向传播流程
        news_tokenized = self.tokenizer.batch_encode_plus(input1,
                                                          add_special_tokens=True,
                                                          truncation=True,
                                                          max_length=200,
                                                          padding='max_length',
                                                          return_tensors='pt')
        news_hidden_outputs = self.model1(input_ids=news_tokenized['input_ids'],
                                          attention_mask=news_tokenized['attention_mask'])
        news_outputs = news_hidden_outputs[0][:, 0, :]

        comments_outputs = torch.tensor(input2)
        inputs = torch.cat((news_outputs, comments_outputs), dim=1)

        x1 = F.relu(self.fc1(inputs))
        predict_results = F.softmax(self.fc2(x1), dim=1)
        return predict_results


if __name__ == '__main__':
    test_input1, test_input2, test_label = get_data('rumdect/data/test_data.txt')
# --------BERT model--------------------------------------
    BERT_model = BERTClassifier()
    for e in range(5):
        saved_model_path = 'saved-model/BERT_h128_e{}.pkl'.format(e)
        checkpoint = torch.load(saved_model_path)
        BERT_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        TP, TN, FP, FN = 0, 0, 0, 0
        total = len(test_label)
        with torch.no_grad():
            print(saved_model_path)
            for i in range(total):
                print('\r测试集第', i+1, '篇新闻', end='')
                output = BERT_model([test_input1[i]])
                _, predict = torch.max(output, 1)
                TP += 1 if predict == test_label[i] and test_label[i] == 1 else 0
                TN += 1 if predict == test_label[i] and test_label[i] == 0 else 0
                FP += 1 if predict != test_label[i] and test_label[i] == 0 else 0
                FN += 1 if predict != test_label[i] and test_label[i] == 1 else 0
        get_result(TP, TN, FP, FN)

# -----------BERT+Personality model-------------------------------
    Pers_model = PersClassifier()
    for e in range(5):
        saved_model_path = 'saved-model/Pers_h128_e{}.pkl'.format(e)
        checkpoint = torch.load(saved_model_path)
        Pers_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        TP, TN, FP, FN = 0, 0, 0, 0
        total = len(test_label)
        with torch.no_grad():
            print(saved_model_path)
            for i in range(total):
                print('\r测试集第', i+1, '篇新闻', end='')
                output = Pers_model([test_input1[i]], [test_input2[i]])
                _, predict = torch.max(output, 1)
                TP += 1 if predict == test_label[i] and test_label[i] == 1 else 0
                TN += 1 if predict == test_label[i] and test_label[i] == 0 else 0
                FP += 1 if predict != test_label[i] and test_label[i] == 0 else 0
                FN += 1 if predict != test_label[i] and test_label[i] == 1 else 0
        get_result(TP, TN, FP, FN)

# --------BERT+BERT model----------------------------------------------------
    test_input1, test_input2, test_label = get_bert('rumdect/data/test_bert.txt')
    Aggr_model = AggrClassifier()
    for e in range(5):
        saved_model_path = 'saved-model/aggr_h128_e{}.pkl'.format(e)
        checkpoint = torch.load(saved_model_path)
        Aggr_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        TP, TN, FP, FN = 0, 0, 0, 0
        total = len(test_label)
        with torch.no_grad():
            print(saved_model_path)
            for i in range(total):
                print('\r测试集第', i+1, '篇新闻', end='')
                output = Aggr_model([test_input1[i]], [test_input2[i]])
                _, predict = torch.max(output, 1)
                TP += 1 if predict == test_label[i] and test_label[i] == 1 else 0
                TN += 1 if predict == test_label[i] and test_label[i] == 0 else 0
                FP += 1 if predict != test_label[i] and test_label[i] == 0 else 0
                FN += 1 if predict != test_label[i] and test_label[i] == 1 else 0
        get_result(TP, TN, FP, FN)

# --------BERT+Word2Vec model----------------------------------------------------
    test_input1, test_input2, test_label = get_word('rumdect/data/test_word2vec.txt')
    Word_model = WordClassifier()
    for e in range(5):
        saved_model_path = 'saved-model/word2vec_h128_e{}.pkl'.format(e)
        checkpoint = torch.load(saved_model_path)
        Word_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        TP, TN, FP, FN = 0, 0, 0, 0
        total = len(test_label)
        with torch.no_grad():
            print(saved_model_path)
            for i in range(total):
                print('\r测试集第', i+1, '篇新闻', end='')
                output = Word_model([test_input1[i]], [test_input2[i]])
                _, predict = torch.max(output, 1)
                TP += 1 if predict == test_label[i] and test_label[i] == 1 else 0
                TN += 1 if predict == test_label[i] and test_label[i] == 0 else 0
                FP += 1 if predict != test_label[i] and test_label[i] == 0 else 0
                FN += 1 if predict != test_label[i] and test_label[i] == 1 else 0
        get_result(TP, TN, FP, FN)
