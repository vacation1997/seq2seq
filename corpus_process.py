from keras_preprocessing.text import Tokenizer
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import logging as log


def read_corpus(corpus_file):
    """
    读取语料文件
    参数：corpus_file 文件路径
    返回值：np数组 encode_data 例：['123',...]
           np数组 decode_data 例：['^壹佰贰拾叁元整$',...]
    """
    with open(corpus_file, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')

    encode_data, decode_data = [], []
    for line in lines:
        if not line.strip():
            continue
        try:
            input_text, target_text = line.split('\t')
            encode_data.append(input_text)
            decode_data.append('^' + target_text + '$')
        except ValueError:
            log.error(f'Error line:{line}')
            input_text = ''
            target_text = ''
    return np.asarray(encode_data), np.asarray(decode_data)


def data_codec(data, tokenizer, one_hot=False):
    """
    数据集编码
    参数：data 数据集
         tokenizer keras中的字符转化工具
    返回值：np数组 二维或是三维（区别于encode或是decode）
    """
    # 函数有两个用处：encode字符处理 和 decode字符处理--decode需转化成one-hot矩阵
    dmaxlen = max([len(it) for it in data])
    fill_value = 0
    codes = []
    for it in data:
        code = tokenizer.texts_to_sequences(it)
        code = [c[0] for c in code]  # 降维
        if one_hot:
            code = to_categorical(
                code[1:], num_classes=len(tokenizer.word_index) + 1)
        codes.append(code)
    if one_hot and codes:
        fill_value = np.zeros_like(codes[0][0])
    codes = pad_sequences(codes, maxlen=dmaxlen,
                          padding='post', value=fill_value)
    return codes
