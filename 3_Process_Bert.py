import torch
from transformers import BertTokenizer, BertModel
from Fix_Random_Seed import setup_seed

setup_seed(6666)
bert_name = './used_model/chinese-bert-wwm'
tokenizer = BertTokenizer.from_pretrained(bert_name)
model = BertModel.from_pretrained(bert_name)


def comments_to_bert(rpath, wpath):
    with open(rpath, 'r', encoding='utf-8') as f:
        texts = f.read().splitlines()
        texts = [eval(t) for t in texts]
    f.close()
    all_comments = [t[2:] for t in texts]

    text_bert = []
    with torch.no_grad():
        for index, comments in enumerate(all_comments):
            encoded_comments = tokenizer.batch_encode_plus(comments,
                                                           add_special_tokens=True,
                                                           truncation=True,
                                                           max_length=50,
                                                           padding='max_length',
                                                           return_tensors='pt'
                                                           )
            bert_comments = model(**encoded_comments)
            features = bert_comments.last_hidden_state[:, 0, :]
            average_features = torch.mean(features, dim=0)
            list_average = list(average_features)
            list_average = [float(x) for x in list_average]
            text_bert.append([texts[index][0], texts[index][1], list_average])
            print('第{}篇新闻的评论已经转换为Bert特征向量'.format(index))
    with open(wpath, 'w', encoding='utf-8') as f:
        for x in text_bert:
            f.write(str(x) + '\n')
    f.close()


if __name__ == '__main__':
    rpath_train = 'rumdect/data/train_text.txt'
    wpath_train = 'rumdect/data/train_bert.txt'
    comments_to_bert(rpath_train, wpath_train)
    rpath_test = 'rumdect/data/test_text.txt'
    wpath_test = 'rumdect/data/test_bert.txt'
    comments_to_bert(rpath_test, wpath_test)
