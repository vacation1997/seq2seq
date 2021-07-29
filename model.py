import os
import numpy as np
import keras as K
from keras.preprocessing.text import Tokenizer
import logging as log

from corpus_process import read_corpus, data_codec


def build_seq2seq_train_model(num_encoder_tokens, num_decoder_tokens, encoder_embedding_dim, decoder_embedding_dim, latent_dim):
    """
    构建seq2seq的训练模型
    参数：num_encoder_tokens encoder的句长
         num_decoder_tokens decoder的句长
         encoder_embedding_dim encoder的embedding的扩展维度
         decoder_embedding_dim decoder的embedding的扩展维度
    返回值：seq2seq训练模型
    """
    # encoder
    encoder_inputs = K.layers.Input(shape=(None,), name='encoder_inputs')
    encoder_embedding = K.layers.Embedding(
        num_encoder_tokens, encoder_embedding_dim, mask_zero=True, name='encoder_embedding')(encoder_inputs)
    encoder_lstm = K.layers.LSTM(latent_dim, return_state=True, return_sequences=False,
                                 dropout=0.2, recurrent_dropout=0.5, name="encoder_lstm")
    _, encoder_state_h, encoder_state_c = encoder_lstm(encoder_embedding)
    encoder_states = [encoder_state_h, encoder_state_c]

    # decoder
    decoder_inputs = K.layers.Input(shape=(None,), name='decoder_inputs')
    decoder_embedding = K.layers.Embedding(num_decoder_tokens, decoder_embedding_dim, mask_zero=True,
                                           name='decoder_embedding')(decoder_inputs)
    decoder_lstm = K.layers.LSTM(latent_dim, return_state=True, return_sequences=True,
                                 dropout=0.2, recurrent_dropout=0.5, name="decoder_lstm")
    rnn_outputs, *_ = decoder_lstm(decoder_embedding,
                                   initial_state=encoder_states)
    decoder_dense = K.layers.Dense(
        num_decoder_tokens, activation='softmax', name='decoder_dense')
    decoder_outputs = decoder_dense(rnn_outputs)

    # encoder和decoder组装
    seq2seq_train_model = K.models.Model(
        inputs=[encoder_inputs, decoder_inputs], outputs=[decoder_outputs])
    seq2seq_train_model.compile(
        optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return seq2seq_train_model


def fit_seq2seq_train_model(model, train_data, batch_size, epochs, callbacks):
    """
    训练seq2seq训练模型
    """
    X, y = train_data
    model.fit(X, y, batch_size=batch_size, epochs=epochs,
              validation_split=0.2, callbacks=callbacks)


def save_seq2seq_train_model(model, path):
    """
    保存seq2seq训练模型
    """
    model.save(path)


def build_seq2seq_model(model_path, latent_dim):
    """
    构建seq2seq的预测模型
    """
    # encoder
    model = K.models.load_model(model_path)
    encoder_inputs = K.layers.Input(shape=(None,))
    encoder_embedding = model.get_layer('encoder_embedding')(encoder_inputs)
    encoder_lstm = model.get_layer('encoder_lstm')
    _, *encoder_states = encoder_lstm(encoder_embedding)
    encoder_model = K.models.Model(encoder_inputs, encoder_states)

    # decoder
    decoder_inputs = K.layers.Input(shape=(None,))
    decoder_embedding = model.get_layer('decoder_embedding')(decoder_inputs)
    decoder_state_h = K.layers.Input(shape=(latent_dim,))
    decoder_state_c = K.layers.Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_h, decoder_state_c]
    decoder_lstm = model.get_layer('decoder_lstm')
    rnn_outputs, *decoder_states = decoder_lstm(
        decoder_embedding, initial_state=decoder_states_inputs)

    # encoder和decoder组装
    decoder_outputs = model.get_layer('decoder_dense')(rnn_outputs)
    decoder_model = K.models.Model(
        [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    seq2seq_model = {}
    seq2seq_model['encoder'], seq2seq_model['decoder'] = encoder_model, decoder_model
    return seq2seq_model


def predict(inp_txt, seq2seq_model, enc_token, dec_token):
    """
    模型推理
    """
    reverse_target_word_index = dec_token.index_word
    max_decoder_seq_length = len(dec_token.word_index)
    input_seq = np.asarray([[enc_token.word_index[c] for c in inp_txt], ])
    states_value_h, states_value_c = seq2seq_model['encoder'].predict([
                                                                      input_seq])
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = dec_token.word_index['^']

    stop_condition = False
    decoded_sentence = ''

    while not stop_condition:
        # 通过encoder层最后输出的states，加上起始字符，进行预测
        # 1个样本(批次)、1次循环、19个结果(19个字符索引的概率)
        output, decoder_state_h, decoder_states_c = seq2seq_model['decoder'].predict(
            [target_seq] + [states_value_h, states_value_c])

        # 概率判断预测字符的索引
        sampled_token_index = np.argmax(output[0, 0, :])
        sampled_word = reverse_target_word_index[sampled_token_index]

        # print(output.shape)
        # print('预测的目标字符索引:',sampled_token_index)
        # print('预测的目标字符:', sampled_word)

        # 如果预测到了结束字符，或预测的字符长度超过了目标最大字符长度，则设置循环终止
        if sampled_word == '$' or len(decoded_sentence) > max_decoder_seq_length:
            stop_condition = True
            continue
        # 拼接预测字符
        decoded_sentence += sampled_word
        # 更新预测起始字符
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        # 更新预测输入
        states_value_h = decoder_state_h
        states_value_c = decoder_states_c

    return decoded_sentence


class advanced_interface:
    def __init__(self) -> None:
        # 输入、输出向量的维度
        self.encoder_embedding_dim, self.decoder_embedding_dim = 10, 20
        # 隐藏层维度、训练批次大小、循环轮数
        self.latent_dim, self.batch_size, self.epochs = 128, 64, 10
        # 语料文件
        self.corpus_file = 'dataset.txt'

        self.encode_data, self.decode_data = read_corpus(self.corpus_file)
        self.enc_token = Tokenizer(filters='', char_level=True)
        self.enc_token.fit_on_texts(self.encode_data)
        self.dec_token = Tokenizer(filters='', char_level=True)
        self.dec_token.fit_on_texts(self.decode_data)
        self.num_enc_tokens = len(self.enc_token.word_index)
        self.num_dec_tokens = len(self.dec_token.word_index)

    def create_save_model(self):
        # 创建模型
        model = build_seq2seq_train_model(
            self.num_enc_tokens+1, self.num_dec_tokens+1, self.encoder_embedding_dim, self.decoder_embedding_dim, self.latent_dim)

        # 模型训练用数据
        indices = np.arange(self.encode_data.shape[0])
        np.random.shuffle(indices)
        encode_data = self.encode_data[indices]
        decode_data = self.decode_data[indices]

        ecodes = data_codec(encode_data, self.enc_token)
        dcodes = data_codec(decode_data, self.dec_token)
        dcode_labels = data_codec(decode_data, self.dec_token, one_hot=True)
        # 训练模型
        fit_seq2seq_train_model(
            model, ([ecodes, dcodes], dcode_labels), self.batch_size, self.epochs, [K.callbacks.EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')])
        save_seq2seq_train_model(model, 'seq2seq_train_model.h5')

    def predict(self, data):
        seq2seq_model = build_seq2seq_model(
            'seq2seq_train_model.h5', self.latent_dim)
        # inp_text = input('请输入一段数字:')
        decoded_sentence = predict(
            data, seq2seq_model, self.enc_token, self.dec_token)

        print('预测字符序列:', decoded_sentence)
        return decoded_sentence


if __name__ == '__main__':
    log.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=log.DEBUG,
                    filename='test.log',
                    filemode='a')

    ai = advanced_interface()
    # ai.create_save_model()
    ai.predict()
