import re
import os
import codecs

'''解压数据集'''


# 去掉非中文字符
def clean_str(string):
    string = re.sub(r"[^\u4e00-\u9fff]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()


# 提取邮件内容
def get_data_in_a_file(original_path, save_path='text.txt'):
    files = os.listdir(original_path)
    for file in files:
        if os.path.isdir(original_path + '/' + file):
            get_data_in_a_file(original_path + '/' + file, save_path=save_path)
        else:
            email = ''
            # 注意要用 'ignore'，不然会报错
            f = codecs.open(original_path + '/' + file, 'r', 'gbk', errors='ignore')
            # lines = f.readlines()
            for line in f:
                line = clean_str(line)
                email += line
            f.close()

            f = open(save_path, 'a', encoding='utf8')
            f.write(email + '\n')
            f.close()


# 提取邮件标签
def get_label_in_a_file(original_path, save_path='label.txt'):
    f = open(original_path, 'r')
    label_list = []
    for line in f:
        label = line.split(" ")[0]
        # spam
        if label == 'spam':
            label_list.append('0')
        # ham
        elif label == 'ham':
            label_list.append('1')

    f = open(save_path, 'w', encoding='utf8')
    f.write('\n'.join(label_list))
    f.close()


def dataProcess():
    print('Storing emails in a file ...')
    get_data_in_a_file('trec06c/data', save_path='text.txt')
    print('Store emails finished !')

    print('Storing labels in a file ...')
    get_label_in_a_file('trec06c/full/index', save_path='label.txt')
    print('Store labels finished !')