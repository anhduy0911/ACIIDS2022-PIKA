import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import config as CFG
import json

def plot_correlation():
    df = pd.read_csv(CFG.log_dir_data + 'test_n_class')
    labs = df.iloc[:, 0].tolist()
    corr_matrix = np.load(CFG.log_dir_run + 'KG_v2_test.npy')
    print(corr_matrix.shape)
    corr_matrix = corr_matrix[:, labs]
    corr_matrix = corr_matrix[labs, :]
    corr_matrix = corr_matrix / np.max(corr_matrix)
    print(corr_matrix.shape)
    sns.heatmap(corr_matrix, linewidths=0.5)
    plt.savefig(CFG.log_dir_run + 'condense_KG_v2' + '.png')

def plot_dis_train_test():
    train_dts = json.load(open(CFG.log_dir_run + 'baseline_train'))
    train_arr = np.zeros(CFG.n_class)
    
    keys = [str(i) for i in range(CFG.n_class)]

    novel_classes_5 = []
    novel_classes_10 = []

    for k, v in train_dts.items():
        if k in keys:
            train_arr[int(k)] = int(v['support'])
            if int(v['support']) == 5:
                novel_classes_5.append(int(k))
            elif int(v['support']) == 10:
                novel_classes_10.append(int(k))
    # test_dts = json.load(open(CFG.log_dir_run + 'KG_v1_test'))
    # test_arr = np.zeros(CFG.n_class)
    
    # for k, v in test_dts.items():
    #     if k in keys:
    #         test_arr[int(k)] = int(v['support'])
    test_arr = [20 for i in range(CFG.n_class)]
    print(len(train_arr))
    print(f'novel_classes_5: {novel_classes_5}')
    print(f'novel_classes_10: {novel_classes_10}')

    f, ax = plt.subplots(figsize=(8, 6))

    current_palette = sns.color_palette('colorblind')
    # plt.xticks(fontsize=5)
    sns.barplot(x=list(range(CFG.n_class)),y=train_arr.tolist(), color=current_palette[2], label='train')
    # sns.barplot(x=list(range(CFG.n_class)),y=test_arr, color=current_palette[4], label='test')
    plt.xticks(np.arange(0,CFG.n_class,1), fontsize = 5)
    # plt.yticks(np.arange(0,10,1), fontsize = 20)
    plt.legend(frameon = False, bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(CFG.log_dir_data + 'dts' + '.png')

def plot_pred():
    train_dts = json.load(open(CFG.log_dir_run + 'KG_v2_test'))
    train_arr = np.zeros(CFG.n_class)
    train_pred = np.zeros(CFG.n_class)
    
    keys = [str(i) for i in range(CFG.n_class)]

    for k, v in train_dts.items():
        if k in keys:
            train_arr[int(k)] = int(v['support'])
            train_pred[int(k)] = int(v['support'] * v['recall'])
    # sns.set()
    # plt.savefig(CFG.log_dir_data + 'test_dts' + '.png')
    f, ax = plt.subplots(figsize=(8, 6))

    current_palette = sns.color_palette('colorblind')
    # plt.xticks(fontsize=5)
    sns.barplot(x=list(range(CFG.n_class)),y=train_arr.tolist(), color=current_palette[2], label='real_sample')
    sns.barplot(x=list(range(CFG.n_class)),y=train_pred.tolist(), color=current_palette[4], label='pred_sample')
    plt.xticks(np.arange(0,CFG.n_class,1), fontsize = 10)
    # plt.yticks(np.arange(0,10,1), fontsize = 20)
    plt.legend(frameon = False)
    plt.tight_layout()
    plt.savefig(CFG.log_dir_data + 'pred' + '.png')
    
if __name__ == '__main__':
    # plot_correlation()
    # plot_pred()
    plot_dis_train_test()