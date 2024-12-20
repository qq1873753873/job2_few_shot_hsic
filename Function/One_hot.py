import numpy as np


def GT_To_One_Hot(gt, class_count, height, width):
    '''
    将gt的每个位置的标签编码成16位的编码。例如标签为1=>100000000000,标签为2=>010000000000
    Convet Gt to one-hot labels
    :param gt:
    :param class_count:
    :return:
    '''
    GT_One_Hot = []  # 转化为one-hot形式的标签
    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):
            temp = np.zeros(class_count, dtype=np.float32)  # (class num,1)
            if gt[i, j] != 0:
                temp[int(gt[i, j]) - 1] = 1
            GT_One_Hot.append(temp)
    GT_One_Hot = np.reshape(GT_One_Hot, [height, width, class_count])
    return GT_One_Hot
