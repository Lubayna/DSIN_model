# coding: utf-8
import gc

import pandas as pd
from joblib import Parallel, delayed

from config import FRAC
#import random
#cate_intervals = pd.read_pickle('/Users/yuxuanyang/Downloads/DSIN-master/sampled_data/cate_intervals')
brand_intervals = pd.read_pickle('/Users/yuxuanyang/Downloads/DSIN-master/sampled_data/brand_intervals')
#print(cate_intervals)
def gen_session_list_dsin(uid, t):
  #  cate_intervals = pd.read_pickle('/Users/yuxuanyang/Downloads/DSIN-master/sampled_data/cate_intervals')
    t.sort_values('time_stamp', inplace=True, ascending=True)
    last_time = 1483574401  # pd.to_datetime("2017-01-05 00:00:01")
    session_list = []
    session = []
    for row in t.iterrows():
        time_stamp = row[1]['time_stamp']
        # pd_time = pd.to_datetime(timestamp_datetime(time_stamp))
        delta = time_stamp - last_time
        cate_id = row[1]['cate']
        brand_id = row[1]['brand']
        # delta.total_seconds()
        # if cate_id in cate_intervals.index:
        #     cate_threshold = cate_intervals.at[cate_id, 'interval']
        if brand_id in brand_intervals.index:
            brand_threshold = brand_intervals.at[brand_id, 'brand_threshold']
        if brand_threshold and delta > brand_threshold:  # Session begin when current behavior and the last behavior are separated by more than 30 minutes.
            if len(session) > 2:  # Only use sessions that have >2 behaviors
                session_list.append(session[:])
            session = []

        session.append((cate_id, brand_id, time_stamp))
        last_time = time_stamp
    if len(session) > 2:
        session_list.append(session[:])
    return uid, session_list


def applyParallel(df_grouped, func, n_jobs, backend='multiprocessing'):
    """Use Parallel and delayed """  # backend='threading'
    results = Parallel(n_jobs=n_jobs, verbose=4, backend=backend)(
        delayed(func)(name, group) for name, group in df_grouped)

    return {k: v for k, v in results}


def gen_user_hist_sessions(model, FRAC=0.25):
    if model not in ['dsin']:
        raise ValueError('model must be din or dmsn')

    print("gen " + model + " hist sess", FRAC)
    name = '/Users/yuxuanyang/Downloads/DSIN-master/sampled_data/behavior_log_pv_user_filter_enc_' + str(FRAC) + '.pkl'
    data = pd.read_pickle(name)
    data = data.loc[data.time_stamp >= 1493769600]  # 0503-0513
    # print('behavior_log_pv_user_filter_enc_')
    # print(data.head(10))
    # 0504~1493856000
    # 0503 1493769600

    user = pd.read_pickle('/Users/yuxuanyang/Downloads/DSIN-master/sampled_data/user_profile_' + str(FRAC) + '.pkl')
    n_samples = user.shape[0]
    print(n_samples)
    batch_size = 150000
    iters = (n_samples - 1) // batch_size + 1

    print("total", iters, "iters", "batch_size", batch_size)
    for i in range(0, iters):
        target_user = user['userid'].values[i * batch_size:(i + 1) * batch_size]
        sub_data = data.loc[data.user.isin(target_user)]
        print(i, 'iter start')
        df_grouped = sub_data.groupby('user')
        user_hist_session = applyParallel(
            df_grouped, gen_session_list_dsin, n_jobs=-1, backend='multiprocessing')
        print('this one')
        print(len(user_hist_session))
        # u1, x1 = random.choice(list(user_hist_session.items()))
        # print(u1, x1)
        # u2, x2 = random.choice(list(user_hist_session.items()))
        # print(u2, x2)
       # pd.to_pickle(user_hist_session, '/Users/yuxuanyang/Downloads/DSIN-master/sampled_data/new_user_hist_session.pkl')
        pd.to_pickle(user_hist_session, '/Users/yuxuanyang/Downloads/DSIN-master/sampled_data/brand_user_hist_session_' +
                     str(FRAC) + '_' + model + '_' + str(i) + '.pkl')
        print(i, 'pickled')
        del user_hist_session
        gc.collect()
        print(i, 'del')

    print("1_gen " + model + " hist sess done")

if __name__ == "__main__":

   # gen_user_hist_sessions('din', FRAC)
    gen_user_hist_sessions('dsin', FRAC)

