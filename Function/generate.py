import random
import numpy as np
from Function import One_hot


def samples(samples_type, class_count, train_ratio, val_ratio, train_samples_per_class, val_samples, gt):
    gt_reshape = np.reshape(gt, [-1])
    train_rand_idx = []
    val_rand_idx = []
    if samples_type == 'ratio':
        # i=(0,15]
        for i in range(class_count):
            idx = np.where(gt_reshape == i + 1)[-1]  # 寻找每个标签的索引
            samplesCount = len(idx)  # 每个标签的数目
            rand_list = [i for i in range(samplesCount)]  # 用于随机的列表
            rand_idx = random.sample(rand_list, np.ceil(samplesCount * train_ratio).astype('int32'))  # 随机数数量 四舍五入(改为上取整)
            rand_real_idx_per_class = idx[rand_idx]  # 训练样本的索引
            train_rand_idx.append(rand_real_idx_per_class)
        train_rand_idx = np.array(train_rand_idx, dtype=object)
        train_data_index = []
        for c in range(train_rand_idx.shape[0]):
            a = train_rand_idx[c]
            for j in range(a.shape[0]):
                train_data_index.append(a[j])
        train_data_index = np.array(train_data_index)

        # 将测试集（所有样本，包括训练样本）也转化为特定形式
        train_data_index = set(train_data_index)
        all_data_index = [i for i in range(len(gt_reshape))]
        all_data_index = set(all_data_index)

        # 背景像元的标签
        background_idx = np.where(gt_reshape == 0)[-1]
        background_idx = set(background_idx)
        test_data_index = all_data_index - train_data_index - background_idx

        # 从测试集中随机选取部分样本作为验证集
        val_data_count = int(val_ratio * (len(test_data_index) + len(train_data_index)))  # 验证集数量
        val_data_index = random.sample(test_data_index, val_data_count)
        val_data_index = set(val_data_index)
        test_data_index = test_data_index - val_data_index  # 由于验证集为从测试集分裂出，所以测试集应减去验证集

        # 将训练集 验证集 测试集 整理
        test_data_index = list(test_data_index)
        train_data_index = list(train_data_index)
        val_data_index = list(val_data_index)

        print('train', len(train_data_index))
        print('test', len(test_data_index))
        print('val', len(val_data_index))

    if samples_type == 'same_num':
        for i in range(class_count):
            idx = np.where(gt_reshape == i + 1)[-1]
            samplesCount = len(idx)
            real_train_samples_per_class = train_samples_per_class
            rand_list = [i for i in range(samplesCount)]  # 用于随机的列表
            if real_train_samples_per_class > samplesCount:
                real_train_samples_per_class = samplesCount
            rand_idx = random.sample(rand_list,
                                     real_train_samples_per_class)  # 随机数数量 四舍五入(改为上取整)
            rand_real_idx_per_class_train = idx[rand_idx[0:real_train_samples_per_class]]
            train_rand_idx.append(rand_real_idx_per_class_train)
        train_rand_idx = np.array(train_rand_idx)
        val_rand_idx = np.array(val_rand_idx)
        train_data_index = []
        for c in range(train_rand_idx.shape[0]):
            a = train_rand_idx[c]
            for j in range(a.shape[0]):
                train_data_index.append(a[j])
        train_data_index = np.array(train_data_index)

        train_data_index = set(train_data_index)
        all_data_index = [i for i in range(len(gt_reshape))]
        all_data_index = set(all_data_index)

        # 背景像元的标签
        background_idx = np.where(gt_reshape == 0)[-1]
        background_idx = set(background_idx)
        test_data_index = all_data_index - train_data_index - background_idx

        # 从测试集中随机选取部分样本作为验证集
        val_data_count = int(val_samples)  # 验证集数量
        val_data_index = random.sample(test_data_index, val_data_count)
        val_data_index = set(val_data_index)

        test_data_index = test_data_index - val_data_index
        # 将训练集 验证集 测试集 整理
        test_data_index = list(test_data_index)
        train_data_index = list(train_data_index)
        val_data_index = list(val_data_index)

    return test_data_index, train_data_index, val_data_index


def index_samples(height, width, class_count, test_data_index, train_data_index, val_data_index, gt):
    gt_reshape = np.reshape(gt, [-1])
    train_samples_gt = np.zeros(gt_reshape.shape)  # (20215,)
    # train_data_index训练索引
    for i in range(len(train_data_index)):
        # print(gt_reshape[train_data_index[i]])
        train_samples_gt[train_data_index[i]] = gt_reshape[train_data_index[i]]
        pass

    # 获取测试样本的标签图
    test_samples_gt = np.zeros(gt_reshape.shape)
    for i in range(len(test_data_index)):
        test_samples_gt[test_data_index[i]] = gt_reshape[test_data_index[i]]
        pass

    Test_GT = np.reshape(test_samples_gt, [height, width])  # 测试样本图

    # 获取验证集样本的标签图
    val_samples_gt = np.zeros(gt_reshape.shape)
    for i in range(len(val_data_index)):
        val_samples_gt[val_data_index[i]] = gt_reshape[val_data_index[i]]
        pass

    train_samples_gt = np.reshape(train_samples_gt, [height, width])
    test_samples_gt = np.reshape(test_samples_gt, [height, width])
    val_samples_gt = np.reshape(val_samples_gt, [height, width])

    train_samples_gt_onehot = One_hot.GT_To_One_Hot(train_samples_gt, class_count, height, width)
    test_samples_gt_onehot = One_hot.GT_To_One_Hot(test_samples_gt, class_count, height, width)
    val_samples_gt_onehot = One_hot.GT_To_One_Hot(val_samples_gt, class_count, height, width)

    train_samples_gt_onehot = np.reshape(train_samples_gt_onehot, [-1, class_count]).astype(int)
    test_samples_gt_onehot = np.reshape(test_samples_gt_onehot, [-1, class_count]).astype(int)
    val_samples_gt_onehot = np.reshape(val_samples_gt_onehot, [-1, class_count]).astype(int)

    ############制作训练数据和测试数据的gt掩膜.根据GT将带有标签的像元设置为全1向量##############
    # 训练集
    train_label_mask = np.zeros([height*width, class_count])
    temp_ones = np.ones([class_count])
    train_samples_gt = np.reshape(train_samples_gt, [height*width])
    for i in range(height*width):
        if train_samples_gt[i] != 0:
            train_label_mask[i] = temp_ones
    train_label_mask = np.reshape(train_label_mask, [height*width, class_count])

    # 测试集
    test_label_mask = np.zeros([height*width, class_count])
    temp_ones = np.ones([class_count])
    test_samples_gt = np.reshape(test_samples_gt, [height*width])
    for i in range(height*width):
        if test_samples_gt[i] != 0:
            test_label_mask[i] = temp_ones
    test_label_mask = np.reshape(test_label_mask, [height*width, class_count])

    # 验证集
    val_label_mask = np.zeros([height*width, class_count])
    temp_ones = np.ones([class_count])
    val_samples_gt = np.reshape(val_samples_gt, [height*width])
    for i in range(height*width):
        if val_samples_gt[i] != 0:
            val_label_mask[i] = temp_ones
    val_label_mask = np.reshape(val_label_mask, [height*width, class_count])

    return Test_GT, train_samples_gt, test_samples_gt, val_samples_gt,\
        train_samples_gt_onehot, test_samples_gt_onehot, val_samples_gt_onehot,\
        train_label_mask, test_label_mask, val_label_mask
