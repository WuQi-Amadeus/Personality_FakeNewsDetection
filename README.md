# Personality_FakeNewsDetection
 
本实验借用的3个预训练模型需要你自行下载到本地。

./used_model/chinese-bert-wwm/: https://huggingface.co/hfl/chinese-bert-wwm

./used_model/personality_model/: https://huggingface.co/Minej/bert-base-personality

./used_model/trans_model/: https://huggingface.co/liam168/trans-opus-mt-zh-en

我也提供了上述三个模型的网盘链接：
https://www.123pan.com/s/oW3Qjv-AqRQ.html

rumdect 文件夹下保存了原始的微博虚假新闻数据集（Weibo 文件夹与Weibo.txt 文本）与处理后本实验使用的数据集（data、label0 与label1 文件夹）。原始微博数据集为Jing Ma的Weibo数据集。本实验使用的是Weibo数据集的子集，包括1280篇真新闻与1280篇假新闻，每篇新闻还有80条评论。

saved-loss 文件夹与saved-model 文件夹用于保存实验过程中生成的结果和模型参数。因为模型参数过大，本数据集并没有保存，可以通过实验代码运行后获得。
