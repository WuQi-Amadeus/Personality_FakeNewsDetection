import os
from transformers import AutoModelWithLMHead, AutoTokenizer, pipeline
from transformers import BertForSequenceClassification, BertTokenizer
import torch
import time

# 中译英模型，读取英文方式：translation(str, max_length=400)[0]['translation_text']
mode_name = "./used_model/trans_model"
model = AutoModelWithLMHead.from_pretrained(mode_name)
tokenizer = AutoTokenizer.from_pretrained(mode_name)
translation = pipeline("translation_zh_to_en", model=model, tokenizer=tokenizer)

# 人格预测模型
model = BertForSequenceClassification.from_pretrained("./used_model/personality_model/", num_labels=5)
tokenizer = BertTokenizer.from_pretrained('./used_model/personality_model/', do_lower_case=True)
model.config.label2id = {
    "Extroversion": 0,
    "Neuroticism": 1,
    "Agreeableness": 2,
    "Conscientiousness": 3,
    "Openness": 4,
}
model.config.id2label = {
    "0": "Extroversion",
    "1": "Neuroticism",
    "2": "Agreeableness",
    "3": "Conscientiousness",
    "4": "Openness",
}


def personality_detection(model_input: str) -> list:
    """
    Performs personality prediction on the given input text

    Args:
        model_input (str): The text conversation

    Returns:
        dict: A dictionary where keys are speaker labels and values are their personality predictions
    """

    if len(model_input) == 0:
        ret = [0, 0, 0, 0, 0]
        return ret
    else:
        dict_custom = {}
        preprocess_part1 = model_input[:len(model_input)]
        dict1 = tokenizer.encode_plus(preprocess_part1, max_length=1024, padding=True, truncation=True)
        dict_custom['input_ids'] = [dict1['input_ids'], dict1['input_ids']]
        dict_custom['token_type_ids'] = [dict1['token_type_ids'], dict1['token_type_ids']]
        dict_custom['attention_mask'] = [dict1['attention_mask'], dict1['attention_mask']]
        outs = model(torch.tensor(dict_custom['input_ids']), token_type_ids=None, attention_mask=torch.tensor(dict_custom['attention_mask']))
        b_logit_pred = outs[0]
        pred_label = torch.sigmoid(b_logit_pred)
        ret = [float(pred_label[0][0]),
               float(pred_label[0][1]),
               float(pred_label[0][2]),
               float(pred_label[0][3]),
               float(pred_label[0][4])]
        return ret


rpath0 = 'rumdect/data/新闻与评论文本/真新闻/'
rpath1 = 'rumdect/data/新闻与评论文本/假新闻/'
wpath0 = 'rumdect/data/新闻与评论人格/真新闻/'
wpath1 = 'rumdect/data/新闻与评论人格/假新闻/'


# 真新闻+假新闻=2351+2313=4664
def process(rpath: str, wpath: str):
    count = 0
    for file_name in os.listdir(rpath):
        if file_name not in os.listdir(wpath):
            start_time = time.time()
            rfile = rpath + file_name
            with open(rfile,  'r', encoding='utf-8') as f:
                texts = f.read().splitlines()
                personalities = []
                for i in range(1, len(texts)):
                    try:
                        en_text = translation(texts[i], max_length=400)[0]['translation_text']
                        personality = personality_detection(en_text)
                        personalities.append(personality)
                    except Exception as e:
                        print(e)

                wfile = wpath + file_name
                with open(wfile, 'w', newline='', encoding='utf-8') as f2:
                    f2.write(texts[0] + '\n')
                    f2.write(texts[1] + '\n')
                    for p in personalities:
                        f2.write(str(p) + '\n')
                f2.close()
            f.close()
            end_time = time.time()
            consume_time = end_time - start_time
            count += 1
            print(wpath[-4:-1], ':第', count, '篇; 用时：', consume_time, '秒')
        else:
            count += 1
            print(wpath[-4:-1], ':第', count, '篇')


if __name__ == '__main__':
    process(rpath0, wpath0)
    process(rpath1, wpath1)
