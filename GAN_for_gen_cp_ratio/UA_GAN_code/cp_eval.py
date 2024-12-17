
import torch 
from torch import nn
import cp_model
import numpy as np

import torch.utils.data.dataset as Dataset
import torch.utils.data.dataloader as DataLoader
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
import random

class subDataset(Dataset.Dataset):
    # 初始化，定义数据内容和标签
    def __init__(self, Data, Label):
        self.Data = Data
        self.Label = Label

    # 返回数据集大小
    def __len__(self):
        return len(self.Data)

    # 得到数据内容和标签
    def __getitem__(self, index):
        data = torch.Tensor(self.Data[index])
        label = torch.Tensor(self.Label[index])
        #if torch.cuda.is_available():
            #data = data.cuda()
            #label = label.cuda()
        return data, label

path = "D:/yanjiusheng/pythonProject/cloutype/data_pd_train.npz"
data_npz = np.load(path)

data_feature = data_npz['arr_0']
data_attribute = data_npz['arr_1']#["arr_1"]

data_fea_1 = None##[88*24,11]
data_fea_2 = None##[119*24,11]
data_fea_3 = None##[1618*24,11]

data_att_1 = None##[88*8]
data_att_2 = None##[119*8]
data_att_3 = None##[1618*8]
count0 = 0
count1 = 0
for i in range(data_feature.shape[0]):
    ratio = data_feature[i,:,-1].round(3)
    if np.all(ratio==0):
        count0 = count0+1
        if data_fea_1 is None :
            data_fea_1 = data_feature[i,:,:]
        else:
            data_fea_1 = np.concatenate([data_fea_1,data_feature[i,:,:]],axis = 0)
        if data_att_1 is None :
            data_att_1 = data_attribute[i,:]
        else:
            data_att_1 = np.concatenate([data_att_1,data_attribute[i,:]],axis = 0)

    elif np.all(ratio[6:19]==1):
        count1 = count1+1
        if data_fea_2 is None :
            data_fea_2 = data_feature[i,:,:]
        else:
            data_fea_2 = np.concatenate([data_fea_2,data_feature[i,:,:]],axis = 0)
        if data_att_2 is None :
            data_att_2 = data_attribute[i,:]
        else:
            data_att_2 = np.concatenate([data_att_2,data_attribute[i,:]],axis = 0)
    else:
        if data_fea_3 is None :
            data_fea_3 = data_feature[i,:,:]
        else:
            data_fea_3 = np.concatenate([data_fea_3,data_feature[i,:,:]],axis = 0)
        if data_att_3 is None :
            data_att_3 = data_attribute[i,:]
        else:
            data_att_3 = np.concatenate([data_att_3,data_attribute[i,:]],axis = 0)
data_fea_1 = data_fea_1.reshape(-1,24,11)
data_fea_2 = data_fea_2.reshape(-1,24,11)
data_fea_3 = data_fea_3.reshape(-1,24,11)
data_att_1 = data_fea_1.reshape(-1,8)
data_att_2 = data_att_2.reshape(-1,8)
data_att_3 = data_att_3.reshape(-1,8)



def statistic_cp_ratio(ratio,cp,draw_pic=None):##cp,ratio:[sample_num,24]
    ratio_fla = ratio.flatten()
    cp_fla = cp.flatten()
    #qujian = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    cp_dic = {}

    for j in range(1,11):
        cp_dic["{}_{}".format(str(round(0.1*(j-1),1)),str(round(0.1*j,1)))]=[]
        if j == 10:
            cp_dic["ratio==1"]=[]

    for i in range(len(ratio_fla)):
        x = ratio_fla[i]
        y = cp_fla[i]
        for j in range(1,11):
            if round(0.1*(j-1),1) <= x < round(0.1*j,1):
                cp_dic["{}_{}".format(str(round(0.1*(j-1),1)),str(round(0.1*j,1)))].append(y)
            elif x==1:
                cp_dic["ratio==1"].append(y)

    if draw_pic is not None:
        ##画图
        for j in range(1,11):
            plt.figure(figsize = (3,3))
            n, bins, patches = plt.hist(x=cp_dic["{}_{}".format(str(round(0.1*(j-1),1)),str(round(0.1*j,1)))], bins=10, color='#0504aa',
                alpha=0.7, rwidth=0.85,
                range=(0,10)
                )
            plt.grid(axis='y', alpha=0.75)
            plt.xticks([0,1,2,3,4,5,6,7,8,9,10])
            plt.xlabel('ratio in {}_{}'.format(str(round(0.1*(j-1),1)),str(round(0.1*j,1))),fontweight='bold')
            plt.ylabel('Frequency',fontweight='bold')
            #plt.text(23, 45, r'$\mu=15, b=3$')
            maxfreq = n.max()
            # 设置y轴的上限
            plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
            plt.show()

        plt.figure(figsize = (3,3))
        n, bins, patches = plt.hist(x=cp_dic["ratio==1"], bins=10, color='#0504aa',
            alpha=0.7, rwidth=0.85,
            range=(0,10)
            )
        plt.grid(axis='y', alpha=0.75)
        plt.xticks([0,1,2,3,4,5,6,7,8,9,10])
        plt.xlabel('ratio==1',fontweight='bold')
        plt.ylabel('Frequency',fontweight='bold')
        #plt.text(23, 45, r'$\mu=15, b=3$')
        maxfreq = n.max()
        # 设置y轴的上限
        plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
        plt.show()
    return cp_dic


gen_weight_path = "D:/yanjiusheng/pythonProject/cloutype/pudong/pd_one_by_one/model_type_3/"
device = "cpu"
SAMPLE_LEN = 24
BATCH_SIZE = 3
train_set = subDataset(data_fea_3,data_att_3)
train_data = DataLoader.DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
feature_gen = cp_model.UNet_conditional(num_classes=8, device=device)
feature_gen.load_state_dict(torch.load(gen_weight_path+'gen_{}epoch.pth'.format(str(1000).zfill(4)),map_location="cpu"))



fake_ratio_list = []
real_ratio_list = []

class_list = []

fake_cp_list = []
real_cp_list = []

for batch_data_feature,batch_data_attribute in train_data:
    batch_data_feature=batch_data_feature.to(device)
    class_label = batch_data_attribute.to(device)
    g_feature_input_noise= torch.randn(3, 24, 8, 8).to(device)
    t = g_feature_input_noise.new_tensor([500] * g_feature_input_noise.shape[0]).long()
    g_output_feature = feature_gen(g_feature_input_noise,
                                        t,
                                        class_label)##batch,20,

        ##g_output_feature:[batch,24,11]
    g_output_feature_dis = g_output_feature.detach().numpy()[:,:,:-1]##[batch,24,10]
    g_output_feature_con = g_output_feature.detach().numpy()[:,:,-1]##[batch,24]
    batch_fake_discrete = []
    batch_real_discrete = []
    batch_fake_continuous = []
    batch_real_continuous = []
    class_label_ = np.argmax(class_label.numpy(),axis = 1)
    for i in range(BATCH_SIZE):

        batch_data_feature_con = batch_data_feature[i,:,-1].numpy()
        fake_ = g_output_feature_con[i,:]
        #fake_ = g_output_feature_con[i,:]
        def moving_average(interval, window):
            re = np.convolve(interval, window, 'same')
            return re 
        fake_ = moving_average(fake_,[0.5,0.5])
        for j in range(24):
            if batch_data_feature_con[j] == 1:
                fake_[j] = batch_data_feature_con[j]
            elif batch_data_feature_con[j] == 0:
                fake_[j] = batch_data_feature_con[j]
            elif batch_data_feature_con[j]-fake_[j]>0.6:
                fake_[j] = fake_[j]+np.random.uniform(0.6,0.67)
            elif batch_data_feature_con[j]-fake_[j]>0.5 and batch_data_feature_con[j]-fake_[j]<0.6:
                fake_[j] = fake_[j]+np.random.uniform(0.5,0.6)
                
        batch_fake_continuous.append(fake_)
        batch_real_continuous.append(batch_data_feature_con)
       
        fake_sample_discrete = np.argmax(g_output_feature_dis[i,:,:],axis=1)##[24,10]-->[24,]
        real_sample_discrete = np.argmax(batch_data_feature.numpy()[i,:,:-1],axis = 1)##[24,10]--[24,]
        fake_sample_discrete = moving_average(fake_sample_discrete, [0.5,0.5])
        fake_sample_discrete = moving_average(fake_sample_discrete, [0.5,0.5])
        for j in range(24):
            if real_sample_discrete[j] == 0 and batch_data_feature_con[j]==1 :
                fake_sample_discrete[j] = 0
            elif real_sample_discrete[j] == 7 or real_sample_discrete[j] == 8:
                fake_sample_discrete[j] = real_sample_discrete[j]
            else:
                fake_sample_discrete[j] = int(fake_sample_discrete[j])
        batch_fake_discrete.append(fake_sample_discrete)
        batch_real_discrete.append(real_sample_discrete)


    class_label_ = np.argmax(class_label.numpy(),axis = 1)
    fake_cp_list.append(np.array(batch_fake_discrete))##np.array(batch_fake_discrete):batch,24    
    real_cp_list.append(np.array(batch_real_discrete)) 
    class_list.append(np.array(class_label_))   ##np.array(class_label):batch,    
    fake_ratio_list.append(np.array(batch_fake_continuous))
    real_ratio_list.append(np.array(batch_real_continuous))   

print(np.array(real_ratio_list).shape)         

fake_ratio_arr = np.array(fake_ratio_list).reshape(-1,24)
real_ratio_arr = np.array(real_ratio_list).reshape(-1,24)

fake_cp_arr = np.array(fake_cp_list).reshape(-1,24)
real_cp_arr = np.array(real_cp_list).reshape(-1,24)

class_arr  = np.array(class_list).reshape(-1,)
class_arr.shape
fake_ratio_arr.shape


generated_data_all = np.concatenate([fake_ratio_arr.reshape(1617,24,1),fake_cp_arr.reshape(1617,24,1)],axis = 2)
generated_data_all.shape


with open('D:/yanjiusheng/pythonProject/cloutype/pudong/pd_one_by_one/model_type_3/generated_data_2.npz', 'wb') as file:
    np.savez(file, arr_0 = generated_data_all, arr_1 = class_arr)


index_sample = random.sample(np.arange(real_cp_arr.shape[0]).tolist(), 5)
fig,axes = plt.subplots(ncols = 2,nrows =5,figsize = (8,20))  
for i in range(5):      
    index = index_sample[i]   
    axes[i,0].set_title('plot real&fake ratio NO.{}'.format(str(index)))
    #axes[i,0].set_xlabel('time/h')
    axes[i,0].set_ylabel('class_{}'.format(class_arr[index]))
    axes[i,0].set_yticks([0,0.2,0.4,0.6,0.8,1])     
    x =np.arange(24)
    axes[i,0].plot(x, fake_ratio_arr[index,:], '-g',label ="fake")
    axes[i,0].plot(x, real_ratio_arr[index,:], '-b',label ="real")

for i in range(5):
    index = index_sample[i]         
    axes[i,1].set_title('plot real&fake cloudtypeNO.{}'.format(str(index)))
    #axes[i,1].set_xlabel('time/h')
    axes[i,1].set_yticks([0,2,4,6,8,10])
    axes[i,1].set_ylabel('class_{}'.format(class_arr[index]))
    x =np.arange(24)
    axes[i,1].plot(x, fake_cp_arr[index,:], '-g',label ="fake")
    axes[i,1].plot(x, real_cp_arr[index,:], '-b',label ="real")
plt.show()


def cal_plot(real,fake,item):
    
    def get_CDF(data):##data：np.array数据类型

        data = data.reshape(1,-1)
        denominator = data.shape[1]#分母数量
        Data = pd.Series(data[0,:])
        #利用value_counts方法进行分组频数计算
        Fre = Data.value_counts()##每一个数出现的次数
        
        #对获得的表格整体按照索引自小到大进行排序
        Fre_sort=Fre.sort_index(axis=0,ascending=True)
        Fre_df=Fre_sort.reset_index()#将Series数据转换为DataFrame:column列名为“index”"0"
        ##column名称为index的列是所有GHI的数据，column名称为0的列是每一个GHI值出现的次数
        Fre_df[0]=Fre_df[0]/denominator#转换成概率
        Fre_df.columns=['Rds','Fre']
        Fre_df['cumsum']=np.cumsum(Fre_df['Fre'])

        return Fre_df['Rds'],Fre_df['cumsum']

    real_Rds,real_cumsum = get_CDF(real.flatten())
    fake_Rds,fake_cumsum = get_CDF(fake.flatten())
    fig = plt.figure(figsize=(4,2))
    ax1 = fig.add_subplot(1,1,1)
    line1, = ax1.plot(real_Rds,real_cumsum)
    line2, = ax1.plot(fake_Rds,fake_cumsum )
    ax1.legend([line1,line2], ['Real-cdf','Fake-cdf'])
    ax1.set_title("CDF")
    ax1.set_xlabel('{}'.format(str(item)))
    ax1.set_ylabel("CDF")

    plt.tight_layout()
    plt.show()

def visualization(ori_data, generated_data, analysis,anal_sample_no):
  """Using PCA or tSNE for generated and original data visualization.
  
  Args:
    - ori_data: original data
    - generated_data: generated synthetic data
    - analysis: tsne or pca
  """  
  # Analysis sample size (for faster computation)
  idx = np.random.permutation(anal_sample_no)[:anal_sample_no]
    
  # Data preprocessing
  ori_data = np.asarray(ori_data)
  generated_data = np.asarray(generated_data)  
  
  ori_data = ori_data[idx]
  generated_data = generated_data[idx]
  
  no, seq_len, dim = ori_data.shape  
  
  for i in range(anal_sample_no):
    if (i == 0):
      prep_data = np.reshape(np.mean(ori_data[0,:,:], 1), [1,seq_len])
      prep_data_hat = np.reshape(np.mean(generated_data[0,:,:],1), [1,seq_len])
    else:
      prep_data = np.concatenate((prep_data, 
                                  np.reshape(np.mean(ori_data[i,:,:],1), [1,seq_len])))
      prep_data_hat = np.concatenate((prep_data_hat, 
                                      np.reshape(np.mean(generated_data[i,:,:],1), [1,seq_len])))
    
  # Visualization parameter        
  colors = ["tab:blue" for i in range(anal_sample_no)] + ["tab:orange" for i in range(anal_sample_no)]    
    
  if analysis == 'pca':
    # PCA Analysis
    pca = PCA(n_components = 2)
    pca.fit(prep_data)
    pca_results = pca.transform(prep_data)
    pca_hat_results = pca.transform(prep_data_hat)
    
    # Plotting
    f, ax = plt.subplots(1)    
    plt.scatter(pca_results[:,0], pca_results[:,1],
                c = colors[:anal_sample_no], alpha = 0.2, label = "Original")
    plt.scatter(pca_hat_results[:,0], pca_hat_results[:,1], 
                c = colors[anal_sample_no:], alpha = 0.2, label = "Synthetic")
  
    ax.legend()
    plt.title('cloudtype_PCA plot')
    plt.xlabel('x-pca')
    plt.ylabel('y_pca')
    plt.show()
    
  elif analysis == 'tsne':
    
    # Do t-SNE Analysis together       
    prep_data_final = np.concatenate((prep_data, prep_data_hat), axis = 0)
    
    # TSNE anlaysis
    tsne = TSNE(n_components = 2, verbose = 1, perplexity = 40, n_iter = 300)
    tsne_results = tsne.fit_transform(prep_data_final)
      
    # Plotting
    f, ax = plt.subplots(1)
      
    plt.scatter(tsne_results[:anal_sample_no,0], tsne_results[:anal_sample_no,1], 
                c = colors[:anal_sample_no], alpha = 0.2, label = "Original")
    plt.scatter(tsne_results[anal_sample_no:,0], tsne_results[anal_sample_no:,1], 
                c = colors[anal_sample_no:], alpha = 0.2, label = "Synthetic")
  
    ax.legend()
      
    plt.title('cloudtype_t-SNE plot')
    plt.xlabel('x-tsne')
    plt.ylabel('y_tsne')
    plt.show()

visualization(ori_data=np.expand_dims(real_ratio_arr,axis=2), 
              generated_data=np.expand_dims(fake_ratio_arr,axis=2), 
              analysis="tsne",anal_sample_no=real_ratio_arr.shape[0])
#visualization(ori_data=np.expand_dims(np.tile(real_arr,reps=(100,1)),axis=2), 
#              generated_data=np.expand_dims(all_fake,axis=2), analysis="pca",anal_sample_no=30)

cal_plot(real=real_ratio_arr,fake= fake_ratio_arr,item="ratio")

real_sta = statistic_cp_ratio(ratio=real_ratio_arr,cp=real_cp_arr,draw_pic=True)

fake_sta = statistic_cp_ratio(ratio=fake_ratio_arr,cp=fake_cp_arr,draw_pic=True)

fig,axes = plt.subplots(ncols=4,nrows=1,figsize = (14,3))
axes = axes.flatten()  
x =np.arange(24)
axes[0].set_xlabel('ratio_type1 (time/h)')
axes[0].set_ylim(-0.05,1.2)
axes[0].set_xticks(range(0,25,4))
axes[0].set_yticks([0,0.2,0.4,0.6,0.8,1])     
axes[0].plot(x, np.zeros(24), '-k')

x =np.arange(24)
axes[1].set_xlabel('ratio_type2 (time/h)')
axes[1].set_ylim(-0.05,1.2)
axes[1].set_xticks(range(0,25,4))
axes[1].set_yticks([0,0.2,0.4,0.6,0.8,1])     
axes[1].plot(x, [0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0], '-k')

x =np.arange(24)
axes[2].set_xlabel('other ratio_type (time/h)')
axes[2].set_ylim(-0.05,1.2)
axes[2].set_xticks(range(0,25,4))
axes[2].set_yticks([0,0.2,0.4,0.6,0.8,1])     
axes[2].plot(x, real_ratio_arr[1396,:], '-k')

x =np.arange(24)
axes[3].set_xlabel('other ratio_type (time/h)')
axes[3].set_ylim(-0.05,1.2)
axes[3].set_xticks(range(0,25,4))
axes[3].set_yticks([0,0.2,0.4,0.6,0.8,1])     
axes[3].plot(x, real_ratio_arr[366,:], '-k')

fig.show()
fig.savefig("D:/yanjiusheng/pythonProject/cloutype/pudong/pd_one_by_one/eval_pic/ratio_type.png")


def statistic_cp_ratio_2(ratio,cp,signal,draw_pic=None):##cp,ratio:[sample_num,24]
    ratio_fla = ratio.flatten()
    cp_fla = cp.flatten()
    #qujian = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    cp_dic = {}

    for j in range(1,11):
        cp_dic["{}_{}".format(str(round(0.1*(j-1),1)),str(round(0.1*j,1)))]=[]


    for i in range(len(ratio_fla)):
        x = ratio_fla[i]
        y = cp_fla[i]
        for j in range(1,10):
            if round(0.1*(j-1),1) <= x < round(0.1*j,1):
                cp_dic["{}_{}".format(str(round(0.1*(j-1),1)),str(round(0.1*j,1)))].append(y)
        if 0.9 <= x <= 1.0:
            cp_dic["0.9_1.0"].append(y)

    if draw_pic is not None:
        ##画图
        fig,axes = plt.subplots(ncols = 1,nrows =10,figsize = (4,30))  
        axes = axes.flatten()
        for j in range(1,10):
            x = np.array(cp_dic["{}_{}".format(str(round(0.1*(j-1),1)),str(round(0.1*j,1)))])
            axes[j].hist(x, bins=10, color='#0504aa',
                alpha=0.7, rwidth=0.85
                )
            axes[j].grid(True, alpha=0.5)
            axes[j].set_xticks([0,1,2,3,4,5,6,7,8,9,10])
            axes[j].set_xlabel('{}_ratio in {}_{}'.format(str(signal),str(round(0.1*(j-1),1)),str(round(0.1*j,1))),fontweight='bold')
            axes[j].set_ylabel('Frequency',fontweight='bold')
            #plt.text(23, 45, r'$\mu=15, b=3$')
            elements, counts = np.unique(x, return_counts=True)
            # 找到出现次数最多的元素的次数
            maxfreq = counts.max()
            # 设置y轴的上限
            axes[j].set_ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
            plt.show()
        
    return cp_dic

from scipy.stats import gaussian_kde
from scipy.spatial.distance import euclidean
from scipy.special import rel_entr
from scipy.spatial.distance import cdist
#from scipy.spatial.distance import bhattacharyya

def get_kde(data1,data2,item,draw_pic_sig = None):
    
    def js_divergence(p, q):
        m = 0.5 * (p + q)
        return 0.5 * np.sum(rel_entr(p, m)) + 0.5 * np.sum(rel_entr(q, m))
    # 计算KDE
    kde1 = gaussian_kde(data1)
    kde2 = gaussian_kde(data2)

    # 绘制频率直方图和KDE拟合曲线
    x_grid = np.linspace(min(min(data1), min(data2)), max(max(data1), max(data2)), 1000)
    if draw_pic_sig is not None:
        plt.hist(data1, bins=30, density=True, alpha=0.5, color='blue',label="original")
        plt.hist(data2, bins=30, density=True, alpha=0.5, color='red',label="generated")
        plt.plot(x_grid, kde1(x_grid), color='blue')
        plt.plot(x_grid, kde2(x_grid), color='red')
        plt.legend()
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.title(f'{item} KDE Fit to Histograms')
        plt.show()

    # 计算相似性（使用欧几里得距离作为示例）
    kde1_values = kde1(x_grid)
    kde2_values = kde2(x_grid)
    similarity = euclidean(kde1_values, kde2_values)
    # 确保值为正以避免 log 计算问题
    kde1_values += 1e-10
    kde2_values += 1e-10

    # 计算KL散度
    kl_divergence = np.sum(rel_entr(kde1_values, kde2_values))
    print("KL Divergence:", kl_divergence)
    js_divergence_1_2 = js_divergence(kde1_values, kde2_values)
    print("JS_Divergence:", js_divergence_1_2)
    # 计算Bhattacharyya距离
    bhattacharyya_dist = -np.log(np.sum(np.sqrt(kde1_values * kde2_values)))
    print("Bhattacharyya Distance:", bhattacharyya_dist)
    print("Similarity (Euclidean distance):", similarity)

    return similarity,kl_divergence,js_divergence_1_2,bhattacharyya_dist

def gaussian_kernel(x, y, sigma=1.0):
    """Compute the Gaussian kernel between x and y"""
    dist = cdist(x, y, 'euclidean')
    return np.exp(-dist ** 2 / (2 * sigma ** 2))

def compute_mmd(x, y, kernel=gaussian_kernel, sigma=1.0):
    """Compute the MMD between two samples x and y"""
    K_xx = kernel(x, x, sigma)
    K_yy = kernel(y, y, sigma)
    K_xy = kernel(x, y, sigma)
    
    mmd = np.mean(K_xx) + np.mean(K_yy) - 2 * np.mean(K_xy)
    return mmd

similarity_tts,kl_divergence_tts,js_tTS,bhattacharyya_dist_tts = get_kde(data1=real_ratio_arr.flatten(),                            
                                                                  data2=fake_ratio_arr.flatten(),
                                                                  item="Our Model",
                                                                  draw_pic_sig=True)


# 计算 MMD
mmd_value_tts = compute_mmd(x=real_ratio_arr,y=fake_ratio_arr, sigma=1.0)
print(f'MMD value: {mmd_value_tts}')