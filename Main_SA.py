import os
import time
import torch
import random
import datetime
import numpy as np
import matplotlib.pyplot as plt
import spectral as spy
from sklearn import metrics
from sklearn import preprocessing
from Function import load_data, generate, LDA_SLIC, GreedyTrainingStrategy
from Network import GS_GraphSAT

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
samples_type = ['ratio', 'same_num'][1]

dataset = 'SA'
Dataset = dataset.upper()
data, gt = load_data.load_dataset(Dataset)

Seed_List = [1, 2, 3, 4, 5]
#Seed_List = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
gt_hsi = gt.reshape(np.prod(gt.shape[:2]),)
class_count = max(gt_hsi)
print('The class numbers of the HSI data is:', class_count)

print('-----Importing Setting Parameters----')
learning_rate, max_epoch, curr_train_ratio, val_ratio, Scale = 0.0005, 100, 0.003, 0.007, 100  # up 0.5%
# learning_rate, max_epoch, curr_train_ratio, val_ratio, Scale = 0.0005, 500, 0.005, 0.005, 500  # up 0.5%

dataset_name = Dataset
torch.cuda.empty_cache()
OA_ALL = []
AA_ALL = []
KPP_ALL = []
AVG_ALL = []
Train_Time_ALL = []
Test_Time_ALL = []

superpixel_scale = Scale
train_samples_per_class = 10
val_samples = class_count
train_ratio = curr_train_ratio
cmap = plt.get_cmap('jet', class_count + 1)
plt.set_cmap(cmap)
m, n, d = data.shape

orig_data = data

height, width, bands = data.shape
data = np.reshape(data, [height * width, bands])
minMax = preprocessing.StandardScaler()
data = minMax.fit_transform(data)
data = np.reshape(data, [height, width, bands])

def Draw_Classification_Map(label, name: str, scale: float = 4.0, dpi: int = 400):
        '''
        get classification map , then save to given path
        :param label: classification label, 2D
        :param name: saving path and file's name
        :param scale: scale of image. If equals to 1, then saving-size is just the label-size
        :param dpi: default is OK
        :return: null
        '''
        bg_idx = np.where(gt_reshape == 0, 0, 1)
        bg_idx = bg_idx.reshape((height, width))
        fig, ax = plt.subplots()
        numlabel = np.array(label)
        numlabel = numlabel * bg_idx
        IP_color = np.array([
            [0,0,0],
            [0,0,255],
            [255,0,0],
            [0,255,0],
            [255,255,0],
            [135,206,250],
            [255,0,255],
            [6,128,67],
            [147,75,67],
            [241,215,126],
            [148,148,231],
            [255,165,0],
            [30,144,255],
            [127,255,170],
            [138,43,226],
            [255,105,180],
            [220,220,220]
        ])
        v = spy.imshow(classes=numlabel.astype(np.int16), fignum=fig.number, colors=IP_color)
        ax.set_axis_off()
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        fig.set_size_inches(label.shape[1] * scale / dpi, label.shape[0] * scale / dpi)
        foo_fig = plt.gcf()  # 'get current figure'
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        foo_fig.savefig(name + '.png', format='png',transparent=True, dpi=dpi, pad_inches=0)
        pass

for curr_seed in Seed_List:

    print('train_inter:', curr_seed)
    random.seed(curr_seed)
    gt_reshape = np.reshape(gt, [-1])  # (21025,)
    train_rand_idx = []
    val_rand_idx = []
    test_data_index, train_data_index, val_data_index = generate.samples(samples_type, class_count, train_ratio,
                                                                    val_ratio, train_samples_per_class, val_samples, gt)

    Test_GT, train_samples_gt, test_samples_gt, val_samples_gt, train_samples_gt_onehot,\
    test_samples_gt_onehot, val_samples_gt_onehot, train_label_mask, test_label_mask, \
    val_label_mask = generate.index_samples(height, width, class_count, test_data_index, train_data_index, val_data_index, gt)

    ls = LDA_SLIC.LDA_SLIC(data, np.reshape(train_samples_gt, [height, width]), class_count - 1)
    tic0 = time.time()
    Q, S, A, Seg = ls.simple_superpixel(scale=superpixel_scale)

    toc0 = time.time()
    LDA_SLIC_Time = toc0 - tic0

    print("LDA-SLIC costs time: {}".format(LDA_SLIC_Time))
    print('Q', Q.shape)
    Q = torch.from_numpy(Q).to(device)
    A = torch.from_numpy(A).to(device)

    train_samples_gt = torch.from_numpy(train_samples_gt.astype(np.float32)).to(device)
    test_samples_gt = torch.from_numpy(test_samples_gt.astype(np.float32)).to(device)
    val_samples_gt = torch.from_numpy(val_samples_gt.astype(np.float32)).to(device)

    train_samples_gt_onehot = torch.from_numpy(train_samples_gt_onehot.astype(np.float32)).to(device)
    test_samples_gt_onehot = torch.from_numpy(test_samples_gt_onehot.astype(np.float32)).to(device)
    val_samples_gt_onehot = torch.from_numpy(val_samples_gt_onehot.astype(np.float32)).to(device)

    train_label_mask = torch.from_numpy(train_label_mask.astype(np.float32)).to(device)
    test_label_mask = torch.from_numpy(test_label_mask.astype(np.float32)).to(device)
    val_label_mask = torch.from_numpy(val_label_mask.astype(np.float32)).to(device)

    net_input = np.array(data, np.float32)
    net_input = torch.from_numpy(net_input.astype(np.float32)).to(device)

    net = GS_GraphSAT(height, width, net_input.shape[2], class_count, Q, A).to(device)

    # 参数量
    # print("parameters", net.parameters(), len(list(net.parameters())))
    # total = sum([param.nelement() for param in net.parameters()])
    # print('parameters:', total)
    # from thop import profile
    # flops, params = profile(net, inputs=(net_input, ))
    # print('flops:', flops)

    def compute_loss(predict: torch.Tensor, reallabel_onehot: torch.Tensor, reallabel_mask: torch.Tensor):
        real_labels = reallabel_onehot
        we = -torch.mul(real_labels, torch.log(predict))
        we = torch.mul(we, reallabel_mask)
        pool_cross_entropy = torch.sum(we)
        return pool_cross_entropy

    zeros = torch.zeros([m * n]).to(device).float()

    def evaluate_performance(network_output, train_samples_gt, train_samples_gt_onehot, require_AA_KPP=False,
                                printFlag=True):
        if False == require_AA_KPP:
            with torch.no_grad():
                available_label_idx = (train_samples_gt != 0).float()
                available_label_count = available_label_idx.sum()
                correct_prediction = torch.where(
                    torch.argmax(network_output, 1) == torch.argmax(train_samples_gt_onehot, 1), available_label_idx,
                    zeros).sum()
                OA = correct_prediction.cpu() / available_label_count

                return OA
        else:
            with torch.no_grad():

                available_label_idx = (train_samples_gt != 0).float()
                available_label_count = available_label_idx.sum()
                correct_prediction = torch.where(
                    torch.argmax(network_output, 1) == torch.argmax(train_samples_gt_onehot, 1), available_label_idx,
                    zeros).sum()
                OA = correct_prediction.cpu() / available_label_count
                OA = OA.cpu().numpy()

                # 计算AA
                zero_vector = np.zeros([class_count])
                output_data = network_output.cpu().numpy()
                train_samples_gt = train_samples_gt.cpu().numpy()
                train_samples_gt_onehot = train_samples_gt_onehot.cpu().numpy()

                output_data = np.reshape(output_data, [m * n, class_count])
                idx = np.argmax(output_data, axis=-1)
                for z in range(output_data.shape[0]):
                    if ~(zero_vector == output_data[z]).all():
                        idx[z] += 1
                # idx = idx + train_samples_gt
                count_perclass = np.zeros([class_count])
                correct_perclass = np.zeros([class_count])
                for x in range(len(train_samples_gt)):
                    if train_samples_gt[x] != 0:
                        count_perclass[int(train_samples_gt[x] - 1)] += 1
                        if train_samples_gt[x] == idx[x]:
                            correct_perclass[int(train_samples_gt[x] - 1)] += 1
                test_AC_list = correct_perclass / count_perclass
                test_AA = np.average(test_AC_list)

                # 计算KPP
                test_pre_label_list = []
                test_real_label_list = []
                output_data = np.reshape(output_data, [m * n, class_count])
                idx = np.argmax(output_data, axis=-1)
                idx = np.reshape(idx, [m, n])
                for ii in range(m):
                    for jj in range(n):
                        if Test_GT[ii][jj] != 0:
                            test_pre_label_list.append(idx[ii][jj] + 1)
                            test_real_label_list.append(Test_GT[ii][jj])
                test_pre_label_list = np.array(test_pre_label_list)
                test_real_label_list = np.array(test_real_label_list)
                kappa = metrics.cohen_kappa_score(test_pre_label_list.astype(np.int16),
                                                test_real_label_list.astype(np.int16))
                test_kpp = kappa

                if printFlag:
                    print("test OA=", OA, "AA=", test_AA, 'kpp=', test_kpp)
                    print('acc per class:')
                    print(test_AC_list)

                OA_ALL.append(OA)
                AA_ALL.append(test_AA)
                KPP_ALL.append(test_kpp)
                AVG_ALL.append(test_AC_list)

                # 保存数据信息
                f = open('results/' + dataset_name + '_results.txt', 'a+')
                str_results = '\n======================' \
                                + " iteration=" + str(curr_seed) \
                                + " learning rate=" + str(learning_rate) \
                                + " epochs=" + str(max_epoch) \
                                + "\nsamples_type={}".format(samples_type) \
                                + "\ntrain_samples_per_class={}".format(train_samples_per_class) \
                                + " train ratio=" + str(train_ratio) \
                                + " val ratio=" + str(val_ratio) \
                                + " ======================" \
                                + "\nOA=" + str(OA) \
                                + "\nAA=" + str(test_AA) \
                                + '\nkpp=' + str(test_kpp) \
                                + '\nacc per class:' + str(test_AC_list) + "\n"
                # + '\ntrain time:' + str(time_train_end - time_train_start) \
                # + '\ntest time:' + str(time_test_end - time_test_start) \
                f.write(str_results)
                f.close()
                return OA


    # 贪心训练策略参数
    LOSS_LIST = []
    # early_stop_nums = 9
    # early_stop_nums = 7
    early_stop_nums = 5
    # early_stop_nums = 3
    early_epoch = 0
    early_flags_nums = 10
    early_flags = 0
    # 训练
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    best_loss = 99999
    best_OA = 0.5
    net.train()
    tic1 = time.perf_counter()
    for i in range(max_epoch + 1):
        query = []
        optimizer.zero_grad()  # zero the gradient buffers
        output = net(net_input)
        loss = compute_loss(output, train_samples_gt_onehot, train_label_mask)
        loss.backward(retain_graph=False)
        optimizer.step()  # Does the update
        with torch.no_grad():
            net.eval()
            output = net(net_input)

            al = GreedyTrainingStrategy.GTS(output, train_data_index, val_data_index, test_data_index)

            trainloss = compute_loss(output, train_samples_gt_onehot, train_label_mask)
            trainOA = evaluate_performance(output, train_samples_gt, train_samples_gt_onehot)
            valloss = compute_loss(output, val_samples_gt_onehot, val_label_mask)
            valOA = evaluate_performance(output, val_samples_gt, val_samples_gt_onehot)
            # print("{}\ttrain loss={}\t train OA={} val loss={}\t val OA={}".format(str(i + 1), trainloss, trainOA,
            #                                                                         valloss, valOA))
            print("Epoch={} \t train loss={}\t  train OA={}\t  val loss={}\t  val OA={}".format(str(i + 1),
                    "%.4f" % trainloss, "%.2f%%" % (trainOA * 100), "%.4f" % valloss, "%.2f%%" % (valOA * 100)))
            if valloss < best_loss:
                best_loss = valloss
                torch.save(net.state_dict(), "model/best_model.pt")
                # print('save model...')
            LOSS_LIST.append(valloss)

        torch.cuda.empty_cache()
        net.train()

        if early_flags != early_flags_nums and i >= 80:
            if LOSS_LIST[-2] < LOSS_LIST[-1]:
                print('curr hop:', early_epoch)
                early_epoch += 1
                LOSS_LIST[-1] = LOSS_LIST[-2]
                print('curr hop+1:', early_epoch)
                if early_epoch == early_stop_nums:
                    print('trigger hop:', early_epoch)
                    print('LOSS:', LOSS_LIST[-2])
                    train_data_index, val_data_index, test_data_index = al.balanced_sample(dataset_name)

                    train_samples_gt, test_samples_gt, val_samples_gt, \
                        train_samples_gt_onehot, test_samples_gt_onehot, val_samples_gt_onehot, \
                        train_label_mask, test_label_mask, val_label_mask \
                        = al.sampling(height, width, class_count, test_data_index, train_data_index, val_data_index, gt, device)
                    early_flags += 1
                    early_epoch = 0
            else:
                early_epoch = 0

    toc1 = time.perf_counter()
    print("\n\n====================training done. starting evaluation...========================\n")
    training_time = toc1 - tic1 + LDA_SLIC_Time
    Train_Time_ALL.append(training_time)
    day = datetime.datetime.now()
    day_str = day.strftime('%m_%d_%H_%M')
    torch.cuda.empty_cache()
    with torch.no_grad():
        net.load_state_dict(torch.load("model/best_model.pt"))
        net.eval()
        tic2 = time.perf_counter()
        output = net(net_input)
        toc2 = time.perf_counter()

        testloss = compute_loss(output, test_samples_gt_onehot, test_label_mask)
        testOA = evaluate_performance(output, test_samples_gt, test_samples_gt_onehot, require_AA_KPP=True,
                                        printFlag=False)
        # print("{}\ttest loss={}\t test OA={}".format(str(i + 1), testloss, testOA))
        print("\ttest loss={}\t test OA={}".format(testloss, testOA))

        # 计算
        testing_time = toc2 - tic2 + LDA_SLIC_Time
        Test_Time_ALL.append(testing_time)
        # 画图
        classification_map = torch.argmax(
            output, 1).reshape([height, width]).cpu() + 1
        difference_map = np.zeros([height, width])
        for i in range(difference_map.shape[0]):
            for j in range(difference_map.shape[1]):
                if gt[i][j] != 0:
                    if classification_map[i][j] == gt[i][j]:
                        difference_map[i][j] = 1
                    else:
                        difference_map[i][j] = 2
        create_folder_date = datetime.datetime.now().strftime('%m_%d_%H_%M_%S')
        os.makedirs('./results/' + dataset_name + '/' + create_folder_date + '/')
        Draw_Classification_Map(
            classification_map, './results/'+dataset_name+'/'+ create_folder_date + '/clas_'+ str(testOA))

    torch.cuda.empty_cache()
    del net

OA_ALL = np.array(OA_ALL)
AA_ALL = np.array(AA_ALL)
KPP_ALL = np.array(KPP_ALL)
AVG_ALL = np.array(AVG_ALL)
Train_Time_ALL = np.array(Train_Time_ALL)
Test_Time_ALL = np.array(Test_Time_ALL)

print("\nsamples_type={}".format(samples_type),"  train_ratio={}".format(curr_train_ratio),"  train_samples_per_class={}".format(train_samples_per_class),
        "\n==============================================================================")
print('OA=', np.mean(OA_ALL), '+-', np.std(OA_ALL))
print('AA=', np.mean(AA_ALL), '+-', np.std(AA_ALL))
print('Kpp=', np.mean(KPP_ALL), '+-', np.std(KPP_ALL))
print('AVG=', np.mean(AVG_ALL, 0), '+-', np.std(AVG_ALL, 0))
print("Average training time:{}".format(np.mean(Train_Time_ALL)))
print("Average testing time:{}".format(np.mean(Test_Time_ALL)))
day = datetime.datetime.now()
day_str = day.strftime('%m_%d_%H_%M')

f = open('results/' + dataset_name + '_results.txt', 'a+')
str_results = '\n\n************************************************'+day_str +'************************************************'\
                + "\nsamples_type={}".format(samples_type) \
                + "\ntrain_samples_per_class={}".format(train_samples_per_class) \
                + "\ntrain_ratio={}".format(curr_train_ratio) \
                + "\nval_ratio={}".format(val_ratio)\
                + '\nOA=' + str(np.mean(OA_ALL)) + '+-' + str(np.std(OA_ALL)) \
                + '\nAA=' + str(np.mean(AA_ALL)) + '+-' + str(np.std(AA_ALL)) \
                + '\nKpp=' + str(np.mean(KPP_ALL)) + '+-' + str(np.std(KPP_ALL)) \
                + '\nAVG=' + str(np.mean(AVG_ALL, 0)) + '+-' + str(np.std(AVG_ALL, 0)) \
                + "\nAverage training time:{}".format(np.mean(Train_Time_ALL)) \
                + "\nAverage testing time:{}".format(np.mean(Test_Time_ALL))
f.write(str_results)
f.close()
