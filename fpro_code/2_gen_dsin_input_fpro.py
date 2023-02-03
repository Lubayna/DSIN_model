# coding: utf-8

import os

import numpy as np
import pandas as pd
from config import DSIN_SESS_COUNT, DSIN_SESS_MAX_LEN, FRAC, ID_OFFSET
from deepctr.feature_column import SparseFeat, DenseFeat, VarLenSparseFeat
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm
from keras_preprocessing.sequence import pad_sequences

FRAC = FRAC
SESS_COUNT = DSIN_SESS_COUNT

# 用抽样的广告点击数据的行的用户和时间来生成session特征数据
def gen_sess_feature_dsin(row):
    sess_count = DSIN_SESS_COUNT
    sess_max_len = DSIN_SESS_MAX_LEN
    sess_input_dict = {}            # session输入
    sess_input_length_dict = {}
    for i in range(sess_count):  # session的个数 5
        sess_input_dict['sess_' + str(i)] = {'cate_id': [], 'brand': []}
        sess_input_length_dict['sess_' + str(i)] = 0
    sess_length = 0   # session长度
    # 得到本次测试的用户和时间
    user, time_stamp = row[1]['user'], row[1]['time_stamp']
    # sample_time = pd.to_datetime(timestamp_datetime(time_stamp ))
    if user not in user_hist_session: 
        for i in range(sess_count):
            sess_input_dict['sess_' + str(i)]['cate_id'] = [0]
            sess_input_dict['sess_' + str(i)]['brand'] = [0]
            sess_input_length_dict['sess_' + str(i)] = 0
        sess_length = 0
    else:
        valid_sess_count = 0
        # 得到本用户的session个数
        last_sess_idx = len(user_hist_session[user]) - 1
        for i in reversed(range(len(user_hist_session[user]))):
            cur_sess = user_hist_session[user][i]
            # 如果session中的时间比广告测试时间更早，
            if cur_sess[0][2] < time_stamp:
                in_sess_count = 1
                # 检测每个session中的时间，如果在广告测试之前，session内clk次数加一
                for j in range(1, len(cur_sess)):
                    if cur_sess[j][2] < time_stamp:
                        in_sess_count += 1
                # 当session中的clk数多于2
                if in_sess_count > 2:
                     # 得到每个session中的cate和brand的序列
                     # 短的时候取全部，长的时候取后面一部分
                    sess_input_dict['sess_0']['cate_id'] = [e[0] for e in \
                    cur_sess[max(0, in_sess_count - sess_max_len):in_sess_count]]
                    sess_input_dict['sess_0']['brand'] = [e[1] for e in \
                    cur_sess[max(0, in_sess_count - sess_max_len):in_sess_count]]
                    # 得到session的长度
                    sess_input_length_dict['sess_0'] = min(sess_max_len, in_sess_count)
                    last_sess_idx = i
                    valid_sess_count += 1
                    break
        for i in range(1, sess_count):
            if last_sess_idx - i >= 0:
                cur_sess = user_hist_session[user][last_sess_idx - i]
                sess_input_dict['sess_' + str(i)]['cate_id'] = [e[0]
                                                                for e in cur_sess[-sess_max_len:]]
                sess_input_dict['sess_' + str(i)]['brand'] = [e[1]
                                                              for e in cur_sess[-sess_max_len:]]
                sess_input_length_dict['sess_' +
                                       str(i)] = min(sess_max_len, len(cur_sess))
                valid_sess_count += 1
            else:
                sess_input_dict['sess_' + str(i)]['cate_id'] = [0]
                sess_input_dict['sess_' + str(i)]['brand'] = [0]
                sess_input_length_dict['sess_' + str(i)] = 0

        sess_length = valid_sess_count
    return sess_input_dict, sess_input_length_dict, sess_length
    # 对于每个广告测试，返回对应这之前的session输入，每条session长度的长度，session的个数


if __name__ == "__main__":

    user_hist_session = {}
    FILE_NUM = len(
        list(filter(lambda x: x.startswith('user_hist_session_' + str(FRAC) + '_dsin_'),
                    os.listdir('./DSIN-master/fpro_data/'))))

    print('total', FILE_NUM, 'files')

    for i in range(FILE_NUM):  # 结合两个user_session文件
        # 每个用户的所有session，每个session中有多个点击数据
        # 点击数据 = （cate，brand，datetime）
        user_hist_session_ = pd.read_pickle(
            './DSIN-master/fpro_data/new_user_hist_session_' + str(FRAC) + '_dsin_' + str(i) + '.pkl')  # 19,34
        user_hist_session.update(user_hist_session_)
        del user_hist_session_
    # 抽样的广告点击数据
    sample_sub = pd.read_pickle(
        './DSIN-master/fpro_data/raw_sample_' + str(FRAC) + '.pkl')
    print(sample_sub) 
    # index_list = []
    sess_input_dict = {} # session的输入字典 字典key是session index
    sess_input_length_dict = {}   # value是cate，brand
    for i in range(SESS_COUNT):
        sess_input_dict['sess_' + str(i)] = {'cate_id': [], 'brand': []}
        sess_input_length_dict['sess_' + str(i)] = []
    # print(sess_input_dict)
    print(sess_input_length_dict)

    sess_length_list = []
    for row in tqdm(sample_sub[['user', 'time_stamp']].iloc[:10,:].iterrows()):
        sess_input_dict_, sess_input_length_dict_, sess_length = gen_sess_feature_dsin(row)
        # index_list.append(index)
        for i in range(SESS_COUNT):
            sess_name = 'sess_' + str(i)
            sess_input_dict[sess_name]['cate_id'].append(
                sess_input_dict_[sess_name]['cate_id'])
            sess_input_dict[sess_name]['brand'].append(
                sess_input_dict_[sess_name]['brand'])
            sess_input_length_dict[sess_name].append(
                sess_input_length_dict_[sess_name])
        sess_length_list.append(sess_length)
    print('生成了每条session的cate序列和brand序列，每条session长度的长度，session的个数')

    user = pd.read_pickle('./DSIN-master/fpro_data/user_profile_' + str(FRAC) + '.pkl')
    ad = pd.read_pickle('./DSIN-master/fpro_data/ad_feature_enc_' + str(FRAC) + '.pkl')
    user = user.fillna(-1)
    user.rename(columns={'new_user_class_level ': 'new_user_class_level'}, inplace=True)
    sample_sub = pd.read_pickle('./DSIN-master/fpro_data/raw_sample_' + str(FRAC) + '.pkl')
    sample_sub.rename(columns={'user': 'userid'}, inplace=True)
    data = pd.merge(sample_sub, user, how='left', on='userid', )
    data = pd.merge(data, ad, how='left', on='adgroup_id')

    sparse_features = ['userid', 'adgroup_id', 'pid', 'cms_segid', 'cms_group_id', 'final_gender_code', 'age_level',
                       'pvalue_level', 'shopping_level', 'occupation', 'new_user_class_level', 'campaign_id',
                       'customer']
    dense_features = ['price']

    for feat in tqdm(sparse_features):
        lbe = LabelEncoder()  # or Hash 对于每个类别特征进行标签编译，之后更新
        data[feat] = lbe.fit_transform(data[feat])
    mms = StandardScaler() # 对稠密特征进行标准化编译，之后更新
    data[dense_features] = mms.fit_transform(data[dense_features])
    # 将类别特征转为稠密特征
    # ID_OFFSET = 1000
    sparse_feature_list = [SparseFeat(feat, vocabulary_size=data[feat].max(
    ) + ID_OFFSET) for feat in sparse_features + ['cate_id', 'brand']]
    for s in sparse_feature_list:
        print(s)
    dense_feature_list = [DenseFeat(feat, dimension=1) for feat in dense_features]
    for s in dense_feature_list:
        print(s)
    # 会话特征
    sess_feature = ['cate_id', 'brand']
    feature_dict = {}
    # 添加原有的类别和稠密特征
    for feat in sparse_feature_list + dense_feature_list:
        feature_dict[feat.name] = data[feat.name].values
    for i in tqdm(range(SESS_COUNT)):
        sess_name = 'sess_' + str(i)
        for feat in sess_feature:
            # 添加广告测试之前的session的cate序列和brand序列
            feature_dict[sess_name + '_' + feat] = pad_sequences(
            sess_input_dict[sess_name][feat], maxlen=DSIN_SESS_MAX_LEN, padding='post')
            # 
            sparse_feature_list.append(VarLenSparseFeat(SparseFeat(sess_name + '_' + feat, 
          vocabulary_size=data[feat].max()+ID_OFFSET, embedding_name='feat'), maxlen=DSIN_SESS_MAX_LEN))
    feature_dict['sess_length'] = np.array(sess_length_list)
    print(feature_dict)
    feature_columns = sparse_feature_list + dense_feature_list
    model_input = feature_dict

    # if not os.path.exists('./DSIN-master/model_input/'):
    #     os.mkdir('./DSIN-master/model_input/')

    # pd.to_pickle(model_input, './DSIN-master/model_input/brand_dsin_input_' +
    #              str(FRAC) + '_' + str(SESS_COUNT) + '.pkl')
    # pd.to_pickle(data['clk'].v    alues, './DSIN-master/model_input/brand_dsin_label_' +
    #              str(FRAC) + '_' + str(SESS_COUNT) + '.pkl')
    # pd.to_pickle(feature_columns,
    #              './DSIN-master/model_input/brand_dsin_fd_' + str(FRAC) + '_' + str(SESS_COUNT) + '.pkl')
    # print("gen dsin input done")
