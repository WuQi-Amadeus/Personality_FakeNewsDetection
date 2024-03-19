# Personality_FakeNewsDetection
 
本实验借用的3个预训练模型需要你自行下载到本地。

./used_model/chinese-bert-wwm/: https://huggingface.co/hfl/chinese-bert-wwm

./used_model/personality_model/: https://huggingface.co/Minej/bert-base-personality

./used_model/trans_model/: https://huggingface.co/liam168/trans-opus-mt-zh-en

此外，./saved-model/文件夹用于保存训练过的模型参数，这是我训练的模型参数：

https://www.123pan.com/s/oW3Qjv-YLRQ.html

4个模型训练5轮的结果都保存在上面的网盘中，20GB

我在模型中固定了随机种子，在实践中发现一段时间内模型的训练结果不会发生改变。
可是经过一段时间后训练结果会发生改变，本人水平有限不知道问题出在哪里。
