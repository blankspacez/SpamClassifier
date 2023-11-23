import codecs

import torch
from transformers import BertTokenizer

from data_process import clean_str
from model import Config, MyModel
import torch.nn.functional as F

def predict(model, input_file, checkpoint_file):

    # 处理数据
    email = ''
    # 注意要用 'ignore'，不然会报错
    f = codecs.open(input_file, 'r', 'gbk', errors='ignore')
    for line in f:
        line = clean_str(line)
        email += line
    f.close()

    input_data = [email.strip()]

    # 将模型设置为评估模式
    model.eval()

    # 加载预训练的权重
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint)

    # 关闭梯度计算
    with torch.no_grad():
        # 将输入数据转换为 PyTorch 的 Tensor
        tokenizer = BertTokenizer.from_pretrained(config.bert_path)
        input_encoding = tokenizer(input_data, truncation=True, padding=True, max_length=config.max_length, return_tensors='pt')
        print(input_encoding)

        # 模型推断
        out = model(input_encoding)
        print(out)

        softmax_output = F.softmax(out, dim=1)

        # 获取概率最高的类别
        _, predicted_class = torch.max(softmax_output, 1)

        class_mapping = {0: "spam", 1: "ham"}
        # 获取预测结果
        prediction = class_mapping[predicted_class.item()]

    return prediction


if __name__ == '__main__':
    config = Config()
    model = MyModel(config)

    prediction = predict(model, 'email_test.txt', 'checkpoint/bert.ckpt')
    print(prediction)

