# coding: utf-8
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.ensemble import IsolationForest
from config import FRAC

log = pd.read_pickle('/Users/yuxuanyang/Downloads/DSIN-master/fpro_data/sample_user_logs.pkl')
def gen_intervals(user, user_log, groupby_sdt):
    sorted_log = user_log.sort_values(by = ['datetime'])
    if len(sorted_log) > 1:
        intervals = {}
        time1 = sorted_log.time_stamp.to_numpy()[:-1]
        time2= sorted_log.time_stamp.to_numpy()[1:]
        name = sorted_log[groupby_sdt].to_numpy()[:-1]
        diffs = time2 - time1 
        return pd.DataFrame({groupby_sdt: name, 'interval': diffs})

def applyParallel(dfGrouped, func, groupby_sdt):
    res = Parallel(n_jobs= -1)(delayed(func)(name, group, groupby_sdt) for name, group in tqdm(dfGrouped))
    return pd.concat(res)
#删除异常值 用异常分数的0.5作为阀值
def iforest_remove_outliner(name, intervals_groupby, groupby_sdt):
    sll = intervals_groupby.interval.tolist()

    X = np.array(sll).reshape(len(sll),1)
    clf = IsolationForest(random_state=0).fit(X)
    predict_result=clf.predict(X)
    tl=[]
    for i in range(len(sll)):
        if predict_result[i] == 1:
            tl.append(sll[i])
    if len(tl) > 0:
        thresholds = pd.DataFrame({groupby_sdt : [name],'{}_threshold'.format(groupby_sdt):[max(tl)]})
        return thresholds

if __name__ == "__main__":
    cate_intervals = applyParallel(log.groupby('user'), gen_intervals, 'cate')
    cate_thresholds = applyParallel(cate_intervals.groupby('cate'), iforest_remove_outliner, 'cate')
    pd.to_pickle(cate_thresholds.set_index('cate'), '/Users/yuxuanyang/Downloads/DSIN-master/sampled_data/cate_intervals')
    brand_intervals = applyParallel(log.groupby('user'), gen_intervals, 'brand')
    brand_thresholds = applyParallel(brand_intervals.groupby('brand'), iforest_remove_outliner, 'brand')
    pd.to_pickle(brand_thresholds.set_index('brand'), '/Users/yuxuanyang/Downloads/DSIN-master/sampled_data/brand_intervals')