import scipy.io as sio


def load_dataset(Dataset):
    if Dataset == 'IP':
        mat_data = sio.loadmat('Data/Indian_pines_corrected.mat')  # 图像数据
        mat_gt = sio.loadmat('Data/Indian_pines_gt.mat')  # 标签数据
        data_hsi = mat_data['indian_pines_corrected']
        gt_hsi = mat_gt['indian_pines_gt']

    if Dataset == 'UP':
        uPavia = sio.loadmat('Data/PaviaU.mat')
        gt_uPavia = sio.loadmat('Data/PaviaU_gt.mat')
        data_hsi = uPavia['paviaU']
        gt_hsi = gt_uPavia['paviaU_gt']

    return data_hsi, gt_hsi
