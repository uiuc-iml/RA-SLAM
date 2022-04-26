import os
import glob
import numpy as np

file_list = glob.glob('./*.npy')

conf_mat = np.zeros((21, 21))
for fn in file_list:
    cur_conf_mat = np.load(fn)
    conf_mat += cur_conf_mat

correct_cnt = np.trace(conf_mat)
all_cnt = np.sum(conf_mat)
print(f"Accuracy: {correct_cnt}/{all_cnt} = {correct_cnt / all_cnt}")