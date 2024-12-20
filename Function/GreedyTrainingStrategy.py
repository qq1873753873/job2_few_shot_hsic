import torch
import numpy as np
from Function import generate


class GTS(object):
    def __init__(self, score, train_data_index, valid_data_index, test_data_index):
        super().__init__()
        self.score = score  # self.score.shape ==> (21025, 16)
        self.train_data_index = train_data_index
        self.valid_data_index = valid_data_index
        self.test_data_index = test_data_index
        self.train_sample_temp = []
        self.valid_sample_temp = []

    def balanced_sample(self, dataset):

        valid_query = []
        valid_pool = self.score[self.valid_data_index, :]  # pool.shape ==> (512, 16)
        for i in range(valid_pool.shape[0]):
            index = max(np.array(valid_pool[i, :].cpu()))
            valid_query.append(index)

        if dataset == 'UP':
            valid_query_index = np.argsort(valid_query)[:8]  # 返回得分最低的8个样本的索引（从验证集中切片）  up
        elif dataset == 'IP':
            valid_query_index = np.argsort(valid_query)[:20]  # 返回得分最低的20个样本的索引（从验证集中切片） in
        elif dataset == 'LK':
            valid_query_index = np.argsort(valid_query)[:10]  # 返回得分最低的10个样本的索引（从验证集中切片） lk
        elif dataset == 'HT':
            valid_query_index = np.argsort(valid_query)[:15]  # 返回得分最低的15个样本的索引（从验证集中切片） ht
        else:
            valid_query_index = np.argsort(valid_query)[:5]  # 返回得分最低的10个样本的索引（从验证集中切片）

        for i in range(len(valid_query_index)):
            self.valid_sample_temp.append(self.valid_data_index[valid_query_index[i]])

        for i in range(len(valid_query_index)):
            self.valid_data_index.remove(self.valid_sample_temp[i])
            self.train_data_index.append(self.valid_sample_temp[i])

        TRAIN_SIZE = len(self.train_data_index)
        VAL_SIZE = len(self.valid_data_index)
        print('New training size: ', TRAIN_SIZE, 'New validation size: ', VAL_SIZE)

        return self.train_data_index, self.valid_data_index, self.test_data_index

    def sampling(self, height, width, class_count, test_data_index, train_data_index,
                                                    val_data_index, gt, device):
        Test_GT, train_samples_gt, test_samples_gt, val_samples_gt, train_samples_gt_onehot, \
            test_samples_gt_onehot, val_samples_gt_onehot, train_label_mask, test_label_mask, \
            val_label_mask = generate.index_samples(height, width, class_count, test_data_index, train_data_index,
                                                    val_data_index, gt)
        train_samples_gt = torch.from_numpy(train_samples_gt.astype(np.float32)).to(device)
        test_samples_gt = torch.from_numpy(test_samples_gt.astype(np.float32)).to(device)
        val_samples_gt = torch.from_numpy(val_samples_gt.astype(np.float32)).to(device)
        # 转到GPU
        train_samples_gt_onehot = torch.from_numpy(train_samples_gt_onehot.astype(np.float32)).to(device)
        test_samples_gt_onehot = torch.from_numpy(test_samples_gt_onehot.astype(np.float32)).to(device)
        val_samples_gt_onehot = torch.from_numpy(val_samples_gt_onehot.astype(np.float32)).to(device)
        # 转到GPU
        train_label_mask = torch.from_numpy(train_label_mask.astype(np.float32)).to(device)
        test_label_mask = torch.from_numpy(test_label_mask.astype(np.float32)).to(device)
        val_label_mask = torch.from_numpy(val_label_mask.astype(np.float32)).to(device)

        return train_samples_gt, test_samples_gt, val_samples_gt,\
        train_samples_gt_onehot, test_samples_gt_onehot, val_samples_gt_onehot,\
        train_label_mask, test_label_mask, val_label_mask
