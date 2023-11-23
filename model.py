import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer


class Config(object):

    """配置参数"""
    def __init__(self):
        self.model_name = 'bert'
        self.save_path = 'checkpoint/' + self.model_name + '.ckpt'        # 模型训练结果
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')   # 设备
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = 2                                            # 标签类别数
        self.max_length = 128
        self.batch_size = 16
        self.num_epochs = 3                                            # epoch数
        self.learning_rate = 5e-5                                       # 学习率
        self.bert_path = './pretrain_model/bert-base-chinese'
        self.hidden_size = 768


class MyModel(nn.Module):

    def __init__(self, config):
        super(MyModel, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        input_ids = x['input_ids']  # 输入的句子每个词对应的id
        attn_mask = x['attention_mask']  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        _, pooled = self.bert(input_ids=input_ids, attention_mask=attn_mask, inputs_embeds=None,
                              return_dict=False)
        out = self.fc(pooled)
        return out