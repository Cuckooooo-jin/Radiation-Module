import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
import random
from numpy import sin,cos,tan,pi,arccos

import torch
from torch import nn

from torch.utils.data import DataLoader,Dataset
import os

'''
with open('D:/yanjiusheng/pythonProject/cloutype/pudong/pd_one_by_one/model_type_3/generated_data_2.npz', 'wb') as file:
    np.savez(file, arr_0 = generated_data_all, arr_1 = class_arr)
'''
def get_generated_data(data_path):##"/'D:/yanjiusheng/pythonProject/cloutype/pudong/pd_one_by_one/model_type_3/generated_data_2.npz"
    data_npz_gen = np.load(
        data_path
        )    
    generated_data_all = data_npz_gen["arr_0"]
    #data_ratio = generated_data_all[:,:,0]
    #data_cp = generated_data_all[:,:,1]
    return generated_data_all

def get_ratio(ratio_1_num,ratio_2_num,ratio_3_num,ratio_gen):
    ratio_type_1 = np.zeros(24)
    ratio_type_2 = [0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0]
    ratio_1 = np.tile(ratio_type_1,ratio_1_num).reshape(ratio_1_num,24)
    ratio_2 = np.tile(ratio_type_2,ratio_2_num).reshape(ratio_2_num,24)
    index_= random.sample(np.arange(ratio_gen.shape[0]).tolist(), ratio_3_num)
    ratio_3 = None
    for xx in index_:
        if ratio_3 is None:
            ratio_3 =  ratio_gen[xx,:].reshape(1,24)
        else:
            ratio_3 = np.concatenate([ratio_3,ratio_gen[xx,:].reshape(1,24)],axis= 0)
    
    ratio_ = np.concatenate([ratio_1,ratio_2,ratio_3],axis = 0)
    return ratio_

def get_dni_gen(day_index,cl_dni,day_type,generated_ratio,gen_nums):
    
    cl_dni_daily = cl_dni[day_index,:]##[24,]
    cl_dni_daily_ = np.tile(cl_dni_daily,gen_nums).reshape(gen_nums,24)
    
    if day_type=="sunny":
        ratio_ = get_ratio(0,108,349,generated_ratio)
        random_index = random.sample(np.arange(ratio_.shape[0]).tolist(),gen_nums)
        ratio_sample = None
        for xx in random_index:
            if ratio_sample is None:
                ratio_sample = ratio_[xx].reshape(1,24)
            else:
                ratio_sample = np.concatenate([ratio_sample,ratio_[xx].reshape(1,24)],axis = 0)

    elif day_type=="cloudy":
        ratio_ = get_ratio(55,11,847,generated_ratio)
        random_index = random.sample(np.arange(ratio_.shape[0]).tolist(),gen_nums)
        ratio_sample = None
        for xx in random_index:
            if ratio_sample is None:
                ratio_sample = ratio_[xx].reshape(1,24)
            else:
                ratio_sample = np.concatenate([ratio_sample,ratio_[xx].reshape(1,24)],axis = 0)

    elif day_type=="overcast":
        ratio_ = get_ratio(33,0,422,generated_ratio)
        random_index = random.sample(np.arange(ratio_.shape[0]).tolist(),gen_nums)
        ratio_sample = None
        for xx in random_index:
            if ratio_sample is None:
                ratio_sample = ratio_[xx].reshape(1,24)
            else:
                ratio_sample = np.concatenate([ratio_sample,ratio_[xx].reshape(1,24)],axis = 0)

    gen_dni = cl_dni_daily_*ratio_sample##[gen_nums,24]
    print("sample_ratio_index:",random_index)
    return gen_dni

class Mydataset(Dataset):
    def __init__(self, xx, yy, transform=None):
        self.x = xx
        self.y = yy
        self.tranform = transform

    def __getitem__(self, index):
        x1 = self.x[index]
        y1 = self.y[index]
        x1 = torch.from_numpy(x1).float()
        y1 = torch.tensor(y1,dtype=torch.float32)
        if self.tranform != None:
            return self.tranform(x1), y1
        return x1, y1

    def __len__(self):
        return len(self.x)

def weight_init(model):
    ##according to the DCGAN paper
    with torch.no_grad():
        
        for m in model.modules():
            if isinstance(m,(nn.Conv2d,nn.Conv1d,nn.ConvTranspose2d,
                             nn.BatchNorm1d,nn.BatchNorm2d,nn.InstanceNorm1d)):
                nn.init.normal_(m.weight.data,0,0.02)

            if isinstance(m,(nn.Linear)):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        torch.nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        param.data.fill_(0)
                        
            if isinstance(m,(nn.RNN,nn.LSTM,nn.GRU)):  
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        torch.nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        torch.nn.init.xavier_uniform_(param.data)
                    elif 'bias_ih' in name:
                        param.data.fill_(1)
                    elif 'bias_hh' in name:
                        param.data.fill_(0)

class multi_SelfAttention(nn.Module):
    """ Self-Attention """

    def __init__(self, n_head, embed_dim, d_k, d_v, batch_size):
        super(multi_SelfAttention,self).__init__()
        
        self.batch = batch_size
        self.n_head = n_head
        self.d_q = d_k
        self.d_k = d_k
        self.d_v = d_v
        self.d_x = embed_dim
        self.d_o = embed_dim

        self.softmax = nn.Softmax(dim=2)

        self.scale = np.power(self.d_k, 0.5)
        self.wq = nn.Parameter(torch.Tensor(self.d_x, self.d_k))
        self.wk = nn.Parameter(torch.Tensor(self.d_x, self.d_k))
        self.wv = nn.Parameter(torch.Tensor(self.d_x, self.d_v))

        self.fc_q = nn.Linear(self.d_k, self.n_head * self.d_k)
        self.fc_k = nn.Linear(self.d_k, self.n_head * self.d_k)
        self.fc_v = nn.Linear(self.d_v, self.n_head * self.d_v)

        self.fc_o = nn.Linear(self.n_head * self.d_v, self.d_o)

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / np.power(param.size(-1), 0.5)
            param.data.uniform_(-stdv, stdv)

    def forward(self, z):

        q = torch.matmul(z, self.wq)   
        k = torch.matmul(z, self.wk)
        v = torch.matmul(z, self.wv)

        q = self.fc_q(q)##单头变为多头
        k = self.fc_k(k)
        v = self.fc_v(v)
        n_x = q.size(1)
        q = q.view(self.batch, n_x, self.n_head, self.d_q).permute(2, 0, 1, 3).contiguous().view(-1, n_x, self.d_q)##[n_head*batch,n_q,d_q]
        k = k.view(self.batch, n_x, self.n_head, self.d_k).permute(2, 0, 1, 3).contiguous().view(-1, n_x, self.d_k)
        v = v.view(self.batch, n_x, self.n_head, self.d_v).permute(2, 0, 1, 3).contiguous().view(-1, n_x, self.d_v)

        u = torch.bmm(q, k.transpose(1, 2)) # 1.Matmul
        u = u / self.scale # 2.Scale

        attn = self.softmax(u) # 3.Softmax
        output = torch.bmm(attn, v) 
        output = output.view(self.n_head, self.batch, n_x, self.d_v).permute(1, 2, 0, 3).contiguous().view(self.batch, n_x, -1) # 4.Concat
        output = self.fc_o(output) # 5.仿射变换得到最终输出

        return output##output:batch_size,seq_len,embed_dim
#CNN
class Generator(nn.Module):

    def __init__(self,in_dim, embed_dim,seq_len,
                 n_head, d_k, d_v, batch_size,
                 depth,attn_drop_rate,forward_drop_rate,forward_expansion,cnn1):
        #channels_input,feature_g,channels_output,num_layers,dropout,seq_len,batch_size,n_head):
        super(Generator,self).__init__()
        self.in_dim = in_dim
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.depth = depth
        self.attn_drop_rate = attn_drop_rate
        self.forward_drop_rate = forward_drop_rate
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.batch_size = batch_size
        self.forward_expansion = forward_expansion
        self.cnn1 = cnn1
        self.num_layers = 1

        self.lstm1 = nn.LSTM(input_size = self.in_dim,hidden_size = self.embed_dim,
                             num_layers = 1, batch_first = True)
        
        self.residual_att = nn.Sequential(
                nn.LayerNorm(self.embed_dim),
                multi_SelfAttention(self.n_head, self.embed_dim, self.d_k, self.d_v, self.batch_size),
                nn.Dropout(self.attn_drop_rate)
                )
        
        self.residual_feedforward = nn.Sequential(
                nn.LayerNorm(self.embed_dim),
                nn.Linear(self.embed_dim,self.forward_expansion * self.embed_dim),
                nn.GELU(),
                nn.Dropout(self.forward_drop_rate),
                nn.Linear(self.forward_expansion * self.embed_dim, self.embed_dim),
                nn.Dropout(self.forward_drop_rate))
        
        self.gen=nn.Sequential(
                nn.Conv1d(in_channels=self.embed_dim,out_channels=self.cnn1,kernel_size=3,stride=1,padding=1),
                nn.BatchNorm1d(num_features=self.cnn1,affine=True)
                            )
        self.fc = nn.Sequential(
                nn.Flatten(start_dim=1,end_dim=2),
                nn.Linear(self.cnn1*self.seq_len,1),
                nn.ReLU()
                )



    def forward(self, x):

        h_0_1 = torch.randn(self.num_layers,self.batch_size,self.embed_dim).to(device="cuda" if torch.cuda.is_available() else "cpu")
        c_0_1 = torch.randn(self.num_layers,self.batch_size,self.embed_dim).to(device="cuda" if torch.cuda.is_available() else "cpu")
        x_lstm,_ = self.lstm1(x,(h_0_1,c_0_1))

        for _ in range(self.depth):
            res_att = x_lstm
            x_lstm = self.residual_att(x_lstm)
            x_lstm = x_lstm+res_att

            res_forward = x_lstm
            x_lstm = self.residual_feedforward(x_lstm)
            x_lstm = x_lstm+res_forward
        x_cnn = x_lstm.permute(0,2,1)
        x_ = self.gen(x_cnn)
        output = self.fc(x_)
        return output

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def get_ori_data(sequence,input_dni,day_index,ori_data_path):
    
    def normalization(data):
        _range = np.max(data) - np.min(data)
        return (data - np.min(data)) / _range
    
    ori_fake_dni = input_dni.reshape(input_dni.shape[0]*input_dni.shape[1],)
    norm_fake_dni =  normalization(ori_fake_dni)
    max_dni= np.max(ori_fake_dni)

    ori_dhi = pd.read_csv(ori_data_path,usecols=['DHI']).values##[5*8760,1]
    ori_dhi = ori_dhi.reshape(-1,24)##[365*5,24]
    ori_dhi_list = []
    max_dhi = np.amax(ori_dhi,axis=1)
    max_dhi_list = []
    for i in range(5):
        ori_dhi_list.append(ori_dhi[(i+1)*day_index,:])
        max_dhi_list.append(max_dhi[(i+1)*day_index])
    max_dhi_list.append(800)

    #pd_2324_dhi = pd.read_csv("D:/yanjiusheng/pythonProject/ori_pv_start_from_0101.csv",usecols=['dhi'])
    #pd_2324_dhi_ = pd_2324_dhi.iloc[:8616,:].values.reshape(-1,24)
    #max_dhi = np.amax(pd_2324_dhi_,axis=1)

    X = []
    for i in range(norm_fake_dni.shape[0] - sequence):
        X.append(norm_fake_dni[i:(i + sequence)])
    
    test_set = Mydataset(np.array(X),np.array(X))
    test_data = DataLoader(dataset=test_set, batch_size=32, shuffle=False, num_workers=0, drop_last=True)
    print(np.array(ori_dhi_list).shape)
    return test_data, max_dhi_list,\
         np.array(ori_dhi_list)
        #pd_2324_dhi_[day_index,:]

def get_dhi(fake_dni,day_index,ori_data_path,model_data_path):

    device = "cuda" 
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'

    gen = Generator(in_dim=1, embed_dim=8,seq_len=4,
                 n_head=4, d_k=16, d_v=32, batch_size=32,
                 depth=2,attn_drop_rate=0.1,forward_drop_rate=0.1,forward_expansion=4,cnn1=16)

    gen = gen.to(device)

    test_data,max_dhi_list,ori_dhi_daily= get_ori_data(sequence=4,input_dni=fake_dni,day_index=day_index,
                                                       ori_data_path=ori_data_path)
    ori_dni = pd.read_csv(ori_data_path,usecols=['DNI']).values[-2*8760:-8760,:]##[8760,1]
    ori_dni_ = ori_dni.reshape(-1,24)
    #train_set = Mydataset(trainx,trainy)#,transform=transforms.ToTensor())
    #test_set = Mydataset(dni,dni)
    #test_data = DataLoader(dataset=test_set, batch_size=32, shuffle=False, num_workers=0, drop_last=True)
    #D:/yanjiusheng/pythonProject/cloutype/pudong/pd_one_by_one/model_type_3/dni_to_dhi_0110epoch.pth
    gen.load_state_dict(torch.load(model_data_path
                                   ))
    
    preds_110 = []
    for idx, (x, label) in enumerate(test_data):
        x = x.unsqueeze(2).to(device)#[b,4,1]
        pred = gen(x)#[b,1]
        pred = pred.squeeze(1).detach().cpu()#[b,]
        list = pred.tolist()
        preds_110.extend(list)
    
    return preds_110,max_dhi_list,ori_dhi_daily,ori_dni_[day_index,:]##max_dhi_list长度为5的列表，存储了历史五年day_index那天的ghi最大值,
    ##ori_dhi_daily np.array (5,24)，存储了历史五年day_index那天的ghi逐时序列      

def cal_daily_fake_pv(fake_dni,fake_dhi,day_index,fake_nums,eff_arr):
    #fake_data: 8*24 so,fake_nums = 24,eff_arr.shape 321,24
    
    lon_lat_index = [121.80,31.14]
    s = 5##倾角为5°
    Total_s = 1.046*1.050*765
    latitude = lon_lat_index[1]
    longtitude = lon_lat_index[0]

    hours = np.tile(np.arange(24),fake_nums).reshape(fake_nums,24)##[0,1,2...,23,0,1,2...,23]重复fake_num次数
    day = np.array([day_index]*24*fake_nums).reshape(fake_nums,24)
    theta0 = 360*day/365##[fake_nums,24]
    sigma = 0.006918 - 0.399912*cos(theta0*pi/180)+ 0.070257*sin(theta0*pi/180)-0.006758*cos(2*theta0*pi/180)\
        + 0.000907*sin(2*theta0*pi/180)-0.002697*cos(3*theta0*pi/180)+ 0.001480 *sin(3*theta0*pi/180)##rad,[fake_nums,24]
    
    tt = 15*hours+latitude-300##[fake_nums,24]
    sinh0 = sin(longtitude*pi/180)*sin(sigma)+cos(longtitude*pi/180)*cos(sigma)*cos(tt*pi/180)##[fake_nums,24]
    
    ##cal rb
    lc = (latitude-120)/60*4
    #omega= (hours+lc-12)*15*pi/180
    ap = 2*pi*(284+day)/365##rad,##[fake_nums,24]
    chiwei = pi/180*(23.45*sin(ap))##rad ,##[fake_nums,24]
    omegas = arccos(-tan(pi*longtitude/180)*tan(chiwei))##日落时间的时角,rad,##[fake_nums,24]
    omegas_dot = min(omegas[0,0],arccos(-1*tan(pi/180*(longtitude-s))*tan(chiwei))[0,0])##rad


    Rb_fenmu = cos(pi/180*(longtitude-s))*cos(chiwei)*sin(omegas_dot)+omegas_dot*sin(pi/180*(longtitude-s))*sin(chiwei)
    Rb_fenzi = cos(pi/180*longtitude)*cos(chiwei)+omegas*sin(pi/180*longtitude)*sin(chiwei)##[fake_nums,24]
    Rb = Rb_fenmu/Rb_fenzi  

    EDNI = 1366.1*(1+0.033*cos(2*pi*day/365))##w/m^2,##[fake_nums,24]
    omega = (hours+lc-12)*15*pi/180##rad, ##[fake_nums,24]
    omega_ = omega[0,:]
    hourly_sin_omega_list = []
    hourly_omega_list = []
    for i in range(hours.shape[1]-1):
        hourly_sin_omega_list.append(sin(omega_[i+1])-sin(omega_[i]))
        hourly_omega_list.append(omega_[i+1]-omega_[i])
    hourly_sin_omega_list.append(sin(omega_[0])-sin(omega_[-1]))
    hourly_omega_list.append(omega_[0]-omega_[-1])

    hourly_sin_omega_arr = np.tile(np.array(hourly_sin_omega_list),fake_nums).reshape(fake_nums,-1)##[fake_nums,24]
    hourly_omega_arr = np.tile(np.array(hourly_omega_list),fake_nums).reshape(fake_nums,-1)##[fake_nums,24]

    EHR_h = 24/pi*1366.1*(1+0.033*cos(pi*360/(180*365)))*(cos(pi*longtitude/180)*\
                             cos(chiwei)*np.array(hourly_sin_omega_arr)+np.array(hourly_omega_arr)*sin(longtitude*pi/180)*sin(chiwei))
    ##kwh/m2

    dni_h = fake_dni/1000
    dhi_h = fake_dhi/1000
    bizhi = dni_h/EHR_h
    G_bt=Rb*dni_h##
    G_dt=dhi_h*(bizhi*Rb+0.5*(1-bizhi)*(1+cos(pi*s/180)))
    rou = 0.8
    G_rt=0.5*rou*(dni_h*sinh0+dhi_h)*(1-cos(pi*s/180))
    G_all=G_bt+G_rt+G_dt

    eff_sample_index = random.sample(np.arange(eff_arr.shape[0]).tolist(), fake_nums)
    eff_sample  = None
    for xx in eff_sample_index:
        if eff_sample is None:
            eff_sample = eff_arr[xx,:].reshape(1,24)
        else:
            eff_sample = np.concatenate([eff_sample,eff_arr[xx,:].reshape(1,24)],axis = 0)
    
    pv = G_all*Total_s*eff_sample
    return pv##kwh(fake_nums,24)

def main(day_index,gen_ratio_folder_path,ori_data_path,model_data_path,eff_arr_path):

    ratio_gen = None
    cp_gen = None

    for i in range(5):
        datapath = gen_ratio_folder_path+ "generated_data_{}.npz".format(str(i+1))
        if ratio_gen is None:
            ratio_gen= get_generated_data(datapath)[:,:,0]
            cp_gen= get_generated_data(datapath)[:,:,1]
            #print(ratio_gen.shape)
        else:
            ratio_gen = np.concatenate([ratio_gen,get_generated_data(datapath)[:,:,0]],axis= 0)
            cp_gen = np.concatenate([cp_gen,get_generated_data(datapath)[:,:,1]],axis= 0)

    ori_df = pd.read_csv(ori_data_path)
    '''
    zenith_angle = ori_df.loc[:,["Solar Zenith Angle"]].values[-2*8760:-8760,:]#(8760, 1)
    h0_pd = 90-zenith_angle##单位是°
    sinh0_pd_daily = sin(h0_pd*pi/180).reshape(365,24)
    '''
    cldni = ori_df.loc[:,["Clearsky DNI"]].values[-2*8760:-8760,:].reshape(365,24)

    day_index = 67
    dni_gen_1 = get_dni_gen(day_index=day_index,cl_dni=cldni,day_type="cloudy",generated_ratio=ratio_gen,gen_nums=10)
    dni_use = dni_gen_1[1:9,:]

    preds_1,max_dhi,ori_dhi_daily,ori_dni_daily = get_dhi(dni_gen_1,day_index=day_index,
                                                    ori_data_path=ori_data_path,
                                                    model_data_path=model_data_path)
    preds_1_use = np.array(preds_1)[21:213,].reshape(-1,24)
    dhi_use = [] 

    for i in range(8):  
        max_sample = random.sample(np.arange(np.array(max_dhi).shape[0]).tolist(), 1)
        max_sample = max_sample[0]
        for j in range(24):
            if ori_dhi_daily[0][j]==0 and preds_1_use[i,j]!= 0 :
                preds_1_use[i,j] = 0
        dhi_use.append(max_dhi[max_sample]*preds_1_use[i,:])

    eff_npz = np.load(eff_arr_path)    
    eff_data =eff_npz["arr_0"]
    pv_daily = cal_daily_fake_pv(fake_dni=dni_use,fake_dhi=np.array(dhi_use),day_index=day_index,fake_nums=8,eff_arr=eff_data)
    return pv_daily##[8,24]

for i in range (10):
    day_list = np.arange(1,366).tolist()
    for value in [52,117,189,197,308,317]:  # Values to remove 2/21, 2/28, 7/11, 7/19, 11/8, 11/18
        while value in day_list:
            day_list.remove(value)
    #print(len(day_list)) #359
    pv_gen_list = []
    for item in day_list:
        pv = main(day_index=item,
        gen_ratio_folder_path = "/remote-home/21310019/2024/pv_TSTR/TS/fake_ratio/",
        ori_data_path = "/remote-home/21310019/2024/pv_TSTR/TS/data_folder/pudongjichang.csv",
        model_data_path = '/remote-home/21310019/CNN_LSTM/dni_to_dhi_pred/gen/gen_0110epoch.pth',
        eff_arr_path = "/remote-home/21310019/2024/pv_TSTR/TS/data_folder/eff_arr.npz")
        pv_gen_list.append(pv)

    print(np.array(pv_gen_list).shape)
    with open('/remote-home/21310019/2024/pv_TSTR/TS/data_folder/fake_pv_arr_{}.npz'.format(i+1), 'wb') as file:
        np.savez(file, arr_0 = np.array(pv_gen_list))