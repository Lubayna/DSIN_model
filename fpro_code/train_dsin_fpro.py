# coding: utf-8
import os

import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score
from keras import backend as K

from config import DSIN_SESS_COUNT, DSIN_SESS_MAX_LEN, FRAC
from deepctr.models import DSIN

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tfconfig = tf.compat.v1.ConfigProto()
tfconfig.gpu_options.allow_growth = True
K.set_session(tf.compat.v1.Session(config=tfconfig))


def nll1(y_true, y_pred): #loss function
    """ Negative log likelihood. """

    # keras.losses.binary_crossentropy give the mean
    # over the last axis. we require the sum
    return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)

if tf.__version__ >='2.0.0':
    tf.compat.v1.disable_eager_execution()

if __name__ == "__main__":
    SESS_COUNT = DSIN_SESS_COUNT  # 会话个数
    SESS_MAX_LEN = DSIN_SESS_MAX_LEN  # 会话长度

    # VarLenSparseFeat：处理类似文本序列的可变长度类型特征。
    # SparseFeat：用于处理类别特征，如性别、国籍等类别特征，将类别特征转为固定维度的稠密特征。
    # maxlen：所有样本中该特征列的长度最大值
    dnn_feature_columns = pd.read_pickle('/Users/yuxuanyang/Downloads/DSIN-master/model_input/new_dsin_fd_' +
                                         str(FRAC) + '_' + str(SESS_COUNT) + '.pkl')
    model_input = pd.read_pickle(
        '/Users/yuxuanyang/Downloads/DSIN-master/model_input/new_dsin_input_' + str(FRAC) + '_' + str(SESS_COUNT) + '.pkl')
    label = pd.read_pickle('/Users/yuxuanyang/Downloads/DSIN-master/model_input/new_dsin_label_'+
                           str(FRAC) + '_' + str(SESS_COUNT) + '.pkl')

    sample_sub = pd.read_pickle(
        '/Users/yuxuanyang/Downloads/DSIN-master/sampled_data/raw_sample_' + str(FRAC) + '.pkl')

    sample_sub['idx'] = list(range(sample_sub.shape[0]))
    train_idx = sample_sub.loc[sample_sub.time_stamp <
                               1494633600, 'idx'].values
    test_idx = sample_sub.loc[sample_sub.time_stamp >=
                              1494633600, 'idx'].values

    train_input = {k: v[train_idx] for k, v in model_input.items()}
    test_input = {k: v[test_idx] for k, v in model_input.items()}
   # print(train_input)
    # print(len(test_input))
    # for k, v in model_input.items():
    #     print('name: {}, length: {}'.format(k, len(v)))
    #     print(v)
   # print(type(test_input))
    train_label = label[train_idx]
    test_label = label[test_idx]
    # print(len(test_label))
    # print(len(test_label[test_label == 1]))
    sess_count = SESS_COUNT
    sess_len_max = SESS_MAX_LEN
    BATCH_SIZE = 4096

    sess_feature_list = ['cate_id', 'brand']
    TEST_BATCH_SIZE = 2 ** 14
#    # print('lalala{}'.format(dnn_feature_columns))
#     for i in dnn_feature_columns:
#         print(i)
#         print(len(i))
#         print('%%%%%%%%%%%%%%%%%%%%%%%%%')
#    # print('%%%%%%%%%%%%%%%%%%%%%%%%%')
#     print(sess_feature_list)
    model = DSIN(dnn_feature_columns, sess_feature_list, sess_max_count=sess_count, bias_encoding=True,
                 att_embedding_size=1, att_head_num=8, dnn_hidden_units=(200, 80), dnn_activation='relu',)
    # 编译模型
    model.compile('adam', K.binary_crossentropy, metrics=[K.binary_crossentropy, ])
    # 训练模型
    hist_ = model.fit(train_input, train_label, batch_size = BATCH_SIZE, epochs=1, initial_epoch=0, verbose=1,)
    # # 得到预测结果
    pred_ans = model.predict(test_input, TEST_BATCH_SIZE)
    print(pred_ans)
    np.savetxt('pred_ans', pred_ans, delimiter = ',')
    print("test LogLoss", round(log_loss(test_label, pred_ans), 4), "test AUC",
          round(roc_auc_score(test_label, pred_ans), 4))
