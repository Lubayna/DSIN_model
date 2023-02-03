# coding: utf-8
import gc

import pandas as pd
from joblib import Parallel, delayed

from config import FRAC
cate_intervals = pd.read_pickle('/Users/yuxuanyang/Downloads/DSIN-master/sampled_data/cate_intervals')
def gen_session_list_dsin(uid, t):
    t.sort_values('time_stamp', inplace=True, ascending=True)
    last_time = 1559314800
  # pd.to_datetime("2019-06-01 00:00:01") 为了设定一个比较早远的时间，来切断使用
    session_list = []
    session = []
    for row in t.iterrows():
        time_stamp = row[1]['time_stamp']
        # pd_time = pd.to_datetime(timestamp_datetime(time_stamp))
        delta = time_stamp - last_time
        app_id = row[1]['app_id']
        cate_id = row[1]['cate']
        store = row[1]['store']
        if cate_id in cate_intervals.index:
            threshold = cate_intervals.at[cate_id, 'interval']
        if threshold and delta > threshold:  # Session begin when current behavior and the last behavior are separated by more than 30 minutes.
            if len(session) > 2:  # Only use sessions that have >2 behaviors
                session_list.append(session[:])
            session = []
            
        # delta.total_seconds()
        if delta > 30 * 60:  # Session begin when current behavior and the last behavior are separated by more than 30 minutes.
            if len(session) > 2:  # Only use sessions that have >2 behaviors
                session_list.append(session[:])
            session = []

        session.append((app_id, cate_id, store, time_stamp))
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
    if model not in [ 'dsin']:
        raise ValueError('model must be din or dmsn')

    print("gen " + model + " hist sess", FRAC)
    name = './fpro_data/sample_user_logs.pkl'
    data = pd.read_pickle(name)
    user_list = list(set(data['user_id'].to_list()))
    user_num = len(user_list)
    batch_size = 10000
    iters = (user_num - 1) // batch_size + 1
    print(iters)
    print("total", iters, "iters", "batch_size", batch_size)
    for i in range(0, iters):
        print('{0}iter starts'.format(i))
        target_user = user_list[i * batch_size:(i + 1) * batch_size]
        sub_data = data.loc[data.user_id.isin(target_user)]
        print(i, 'iter start')
        df_grouped = sub_data.groupby('user_id')
        user_hist_session = applyParallel(
            df_grouped, gen_session_list_dsin, n_jobs=10, backend='multiprocessing')
        pd.to_pickle(user_hist_session, './fpro_data/user_hist_session_DSIN_{}.pkl'.format(i))
        print(i, 'pickled')
        print(user_hist_session)
        del user_hist_session
        gc.collect()
        print(i, 'del')

    print("1_gen " + model + " hist sess done")

if __name__ == "__main__":
    gen_user_hist_sessions('dsin', FRAC)
