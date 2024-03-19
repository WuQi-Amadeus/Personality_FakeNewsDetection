import os.path
import time
from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from Fix_Random_Seed import setup_seed
import os

setup_seed(6666)
start_time = time.time()
neuron = 128
saved_loss_path = 'saved-loss/BERT_h{}_loss.txt'.format(neuron)
saved_predict_path = 'saved-loss/BERT_h{}_predict.txt'.format(neuron)
folders = ['saved-loss', 'saved-model']
for folder in folders:
    if not os.path.exists(folder):
        os.makedirs(folder)


def get_data(path):
    s = time.time()
    data = []
    with open(path, 'r', newline='', encoding='utf-8') as f:
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
            tmp.append(eval(d[i]))
        input2.append(tmp)

    e = time.time()
    print('获取数据用时：{:.2f}秒'.format(e - s))
    return input1, input2, label


class BERTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model1_name = './used_model/chinese-bert-wwm'
        self.tokenizer = BertTokenizer.from_pretrained(self.model1_name)
        self.model1 = BertModel.from_pretrained(self.model1_name)
        self.fc1 = nn.Linear(768, neuron)
        self.fc2 = nn.Linear(neuron, 2)

    def forward(self, input1):
        # 前向传播流程
        batch_tokenized = self.tokenizer.batch_encode_plus(input1,
                                                           add_special_tokens=True,
                                                           truncation=True,
                                                           max_length=200,
                                                           padding='max_length',
                                                           return_tensors='pt')
        input_ids = batch_tokenized['input_ids']
        attention_mask = batch_tokenized['attention_mask']
        hidden_outputs = self.model1(input_ids, attention_mask=attention_mask)
        out_puts = hidden_outputs[0][:, 0, :]

        x1 = F.relu(self.fc1(out_puts))
        predict_results = F.softmax(self.fc2(x1), dim=1)
        return predict_results


if __name__ == '__main__':
    # 获取数据
    train_input1, train_input2, train_label = get_data('rumdect/data/train_data.txt')
    test_input1, test_input2, test_label = get_data('rumdect/data/test_data.txt')

    # 训练
    model = BERTClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    start_epoch = 0
    num_epochs = 5
    batch_size = 32
    batch_count = int(len(train_label) / batch_size)
    saved_loss = []

    '''
    # 加载参数
    saved_model_path = 'saved-model/BERT_h128_e4.pkl'
    checkpoint = torch.load(saved_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = num_epochs
    '''

    for epoch in range(start_epoch, num_epochs):
        running_loss = 0.0
        tmp_loss = []
        for i in range(0, batch_size * batch_count, batch_size):
            input1 = train_input1[i: i + batch_size]
            input2 = train_input2[i: i + batch_size]
            label = train_label[i: i + batch_size]
            # 模型计算结果
            output = model(input1)
            # 计算损失和梯度，更新模型参数
            loss = criterion(output, torch.tensor(label))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 记录损失
            running_loss += loss.item()
            tmp_loss.append(loss.item())
            print("----Epoch:%d:  Batch:%d, 累计Loss %.4f" % (epoch, i / batch_size + 1, running_loss))
        epoch_loss = running_loss / batch_count
        saved_loss.append(tmp_loss)
        print("Epoch:%d: Loss %.4f" % (epoch, epoch_loss))
        # 保存训练的模型
        checkpoint = {'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(),
                      'epoch': epoch}
        saved_model_path = 'saved-model/BERT_h{}_e{}.pkl'.format(neuron, epoch)
        torch.save(checkpoint, saved_model_path)

    # 保存每轮的损失值
    with open(saved_loss_path, 'w', newline='', encoding='utf-8') as f:
        for d in saved_loss:
            f.write(str(d) + '\n')
    f.close()

    # 预测
    TP, TN, FP, FN = 0, 0, 0, 0
    total = len(test_label)
    with torch.no_grad():
        for i in range(total):
            print('测试集第', i, '篇新闻')
            output = model([test_input1[i]])
            _, predict = torch.max(output, 1)
            TP += 1 if predict == test_label[i] and test_label[i] == 1 else 0
            TN += 1 if predict == test_label[i] and test_label[i] == 0 else 0
            FP += 1 if predict != test_label[i] and test_label[i] == 0 else 0
            FN += 1 if predict != test_label[i] and test_label[i] == 1 else 0
    accuracy = float((TP + TN) / total)
    precision = float(TP / (TP + FP))
    recall = float(TP / (TP + FN))
    print('---------假新闻---真新闻----')
    print('-预测正确--{}-----{}----'.format(TP, TN))
    print('-预测错误--{}-----{}----'.format(FP, FN))
    print('模型准确率：', accuracy)
    print('模型查准率：', precision)
    print('模型查全率：', recall)

    end_time = time.time()
    print('训练用时：{:.2f}小时'.format((end_time - start_time)/3600))

    with open(saved_predict_path, 'w', newline='', encoding='utf-8') as f:
        f.write('---------假新闻---真新闻----' + '\n')
        f.write('-预测正确--{}-----{}----'.format(TP, TN) + '\n')
        f.write('-预测错误--{}-----{}----'.format(FP, FN) + '\n')
        f.write('模型准确率：' + str(accuracy) + '\n')
        f.write('模型查准率：' + str(precision) + '\n')
        f.write('模型查全率：' + str(recall))
    f.close()
