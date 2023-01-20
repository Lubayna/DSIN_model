
# coding: utf-8
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import log_loss, roc_auc_score, roc_curve, f1_score, classification_report
# from tensorflow.python.keras import backend as K
from keras import backend as K
from config import DSIN_SESS_COUNT, DSIN_SESS_MAX_LEN, FRAC
from deepctr.models.sequence.dsin import DSIN
#from _tkinter import _flatten
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tfconfig = tf.compat.v1.ConfigProto()
tfconfig.gpu_options.allow_growth = True
K.set_session(tf.compat.v1.Session(config=tfconfig))

def nll1(y_true, y_pred):
    """ Negative log likelihood. """

    # keras.losses.binary_crossentropy give the mean
    # over the last axis. we require the sum
    return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)


if tf.__version__ >='2.0.0':
    tf.compat.v1.disable_eager_execution()

if __name__ == "__main__":
    SESS_COUNT = DSIN_SESS_COUNT
    SESS_MAX_LEN = DSIN_SESS_MAX_LEN

    dnn_feature_columns = pd.read_pickle('/Users/yuxuanyang/Downloads/DSIN-master/model_input/new_dsin_fd_' +
                                         str(FRAC) + '_' + str(SESS_COUNT) + '.pkl')
    print('dsin_fd_0.25_5')
    print(dnn_feature_columns)
    model_input = pd.read_pickle(
        '/Users/yuxuanyang/Downloads/DSIN-master/model_input/new_dsin_input_' + str(FRAC) + '_' + str(SESS_COUNT) + '.pkl')
    label = pd.read_pickle('/Users/yuxuanyang/Downloads/DSIN-master/model_input/new_dsin_label_' +
                           str(FRAC) + '_' + str(SESS_COUNT) + '.pkl')
    # print('dsin_input_0.25_5')
    # print(len(model_input))
    # print(model_input)
    # print('dsin_input_0.25_5_key')
    # print(model_input.keys())

   # print(label)
    sample_sub = pd.read_pickle(
        '/Users/yuxuanyang/Downloads/DSIN-master/sampled_data/raw_sample_' + str(FRAC) + '.pkl')
    # print('raw_sample')
    # print(len(sample_sub))
    # print(sample_sub)
    sample_sub['idx'] = list(range(sample_sub.shape[0]))
    train_idx = sample_sub.loc[sample_sub.time_stamp <
                               1494633600, 'idx'].values
    test_idx = sample_sub.loc[sample_sub.time_stamp >=
                              1494633600, 'idx'].values

    train_input = {k: v[train_idx] for k, v in model_input.items()}
    test_input = {k: v[test_idx] for k, v in model_input.items()}
    print('train_input')
    print(len(train_input))
    print('test_input')
    print(len(test_input))
    train_label = label[train_idx]
    test_label = label[test_idx]

    print('train_user_id')
    print(len(train_input['userid']))
    print('test_user_id')
    print(len(test_input['userid']))


    sess_count = SESS_COUNT
    sess_len_max = SESS_MAX_LEN
    BATCH_SIZE = 4096

    sess_feature_list = ['cate_id', 'brand']
    TEST_BATCH_SIZE = 2 ** 14

    model = DSIN(dnn_feature_columns, sess_feature_list, sess_max_count=sess_count, bias_encoding=True,
                 att_embedding_size=1, att_head_num=8, dnn_hidden_units=(256, 128), dnn_activation='relu',
                 dnn_dropout=0.2, dnn_use_bn=False, l2_reg_dnn=0, l2_reg_embedding=1e-06, seed=2022, task='binary')
    model.compile('adam', loss=nll1,
                  metrics=[nll1, ])

    hist_ = model.fit(train_input, train_label, batch_size=BATCH_SIZE,
                      epochs=1, initial_epoch=0, verbose=1, )

    pred_ans = model.predict(test_input, TEST_BATCH_SIZE)

    print('pred_ans')
    print(type(pred_ans))
    print(len(pred_ans))
    print(pred_ans)
    print(np.sum(pred_ans))

    print('test_label')
    print(len(test_label))
    print(test_label)
    print(np.sum(test_label))
    pd.DataFrame(pred_ans).to_csv('/Users/yuxuanyang/Downloads/DSIN-master/pred_ans.csv')
    pd.DataFrame(test_label).to_csv('/Users/yuxuanyang/Downloads/DSIN-master/test_label.csv')
    #print(classification_report(test_label, pred_ans))
    print("test LogLoss", round(log_loss(test_label, pred_ans), 4), "test AUC",
          round(roc_auc_score(test_label, pred_ans), 4))

    #"F1 score", round(f1_score(test_label, pred_ans), 4))
   #rec = precision_recall_fscore_k(test_label, pred_ans, k=5, digs=2)
    #print("recall@5", round(rec, 4))

    # fpr, tpr, thresholds = roc_curve(test_label, pred_ans)
    # print(fpr)
    # print(tpr)
    # print(thresholds)
    # plt.plot(fpr, tpr, marker='o')
    # plt.xlabel('FPR: False positive rate')
    # plt.ylabel('TPR: True positive rate')
    # plt.grid()
    # plt.show()