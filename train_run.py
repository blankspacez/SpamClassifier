import torch
from transformers import BertTokenizer, BertModel
import numpy as np
from torch.utils.data import Dataset, DataLoader
import codecs
from sklearn.model_selection import train_test_split
from model import Config, MyModel
from utils import train, test


# 构造数据集类，转换为Dataloader可以读取的形式
class NewDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    # 读取单个样本
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}  # encoding
        item['labels'] = torch.tensor(int(self.labels[idx]))    # label
        return item   # item字典包含encoding和label

    def __len__(self):
        return len(self.labels)


if __name__ == '__main__':
    config = Config()
    # dataProcess()    # 处理原始数据集trec06c.tgz，得到文件text.txt和labels.txt
    print(f"{config.device}\n")
    # 文本
    # new_texts = [x[-300:].strip() for x in codecs.open('text.txt',encoding='utf8')]
    new_texts = [x.strip() for x in codecs.open('text.txt', encoding='utf8')]
    # 标签
    new_labels = [x.strip() for x in codecs.open('label.txt')]

    # 划分训练集、测试集、验证集
    X_train, X_temp, y_train, y_temp = train_test_split(new_texts[:], new_labels[:], test_size=0.2)
    X_test, X_dev, y_test, y_dev = train_test_split(X_temp, y_temp, test_size=0.5)

    # 将文本变成词向量
    tokenizer = BertTokenizer.from_pretrained(config.bert_path)
    train_encoding = tokenizer(X_train, truncation=True, padding=True, max_length=config.max_length, return_tensors='pt')
    test_encoding = tokenizer(X_test, truncation=True, padding=True, max_length=config.max_length, return_tensors='pt')
    dev_encoding = tokenizer(X_dev, truncation=True, padding=True, max_length=config.max_length, return_tensors='pt')

    print("Tokenization has been completed.\n")

    train_dataset = NewDataset(train_encoding, y_train)
    test_dataset = NewDataset(test_encoding, y_test)
    dev_dataset = NewDataset(dev_encoding, y_dev)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size, shuffle=True)

    model = MyModel(config).to(config.device)
    train(config, model, train_loader, dev_loader)

    test(config, model, test_loader)