#
# this module cal the test acc rate.
# now it can just cal the full data test acc rate.
#
# attention: this module can work will just when the test set is fully released.
#

import numpy as np
import pandas as pd
import os
import struct

from data_set import data_config as cfg


class Eval:
    def __init__(self):
        self.gt_path = '/media/super/Dev Data/ml_data_set/Kaggle_MNIST/data/full_data/t10k-labels.idx1-ubyte'
        self.pred_path = '/media/super/Dev Data/ml_data_set/Kaggle_MNIST/result/pred.csv'
        self.result = 0.0

    def load_data(self):
        # gt label
        with open(self.gt_path, 'rb') as lbpath:
            magic, n = struct.unpack('>II', lbpath.read(8))
            gt_list = np.fromfile(lbpath, dtype=np.uint8).tolist()
        # predict label
        pred_df = pd.read_csv(self.pred_path, usecols=['Label'])
        pred_list = np.array(pred_df).tolist()

        return gt_list, pred_list

    def cal(self, gt_list, pred_list):
        count = 0
        gt_num = len(gt_list)
        for i in range(len(gt_list)):
            if gt_list[i] != pred_list[i]:
                count += 1
        self.result = count / gt_num

        return self.result


def main():
    eval = Eval()
    gt_list, pred_list = eval.load_data()
    print(eval.cal(gt_list, pred_list))


if __name__ == '__main__':
    main()

