import numpy as np
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
import os
from math import sqrt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import datetime
from tqdm import trange
import pandas as pd

def get_test_real_data(sequence):

    data_real = pd.read_csv("/remote-home/21310019/2024/pv_TSTR/TS/data_folder/ori_pv_start_from_0101.csv",usecols=["PV/kwh"]).values
    data_real_ = data_real[9408:,:].reshape(-1,24)
    def normalization(data):
        _range = np.max(data) - np.min(data)
        return (data - np.min(data)) / _range
    #data_real_ = normalization(data_real)
    #data_real_ = data_real_[9408:,:].reshape(-1,24)

    #class_label = data_npz["arr_1"]
    daily_slice = []
    daily_length_list = []
    daily_length_count = []

    for i in range(data_real_.shape[0]):
        daily = data_real_[i,:]
        sub_slice = []
        for j in range(24):
            if daily[j] != 0:
                daily_length_count.append(j)
                sub_slice.append(daily[j])
        daily_length_list.append(len(sub_slice)) 
        daily_slice.append(sub_slice)  

    #print(len(daily_slice))####416  
    pv_drop0 = None
    for i in range(len(daily_slice)):
        fea = np.array(daily_slice[i])
        if pv_drop0 is None:
            pv_drop0 = fea
        else:
            pv_drop0 = np.concatenate([pv_drop0,fea],axis=0)

    X = []
    Y = []
    for i in range(pv_drop0.shape[0] - sequence):
        X.append(pv_drop0[i:(i + sequence)])##[4,1]
        Y.append(pv_drop0[i + sequence])##1
    print(np.array(X).shape)

    max_value = {}
    min_value = {}

    # 构建batch
    testx_pre = np.array(X)
    testx = normalization(testx_pre)
    
    max_value["testx"] = np.max(testx_pre)
    min_value["testx"] = np.min(testx_pre)

    testy_pre = np.array(Y)
    testy = normalization(testy_pre)
    max_value["testy"] = np.max(testy_pre)
    min_value["testy"] = np.min(testy_pre)

    #return testx_pre,testy_pre
    return testx,testy

def get_train_real_data(sequence):

    data_real = pd.read_csv("/remote-home/21310019/2024/pv_TSTR/TS/data_folder/ori_pv_start_from_0101.csv",usecols=["PV/kwh"]).values
    ori_data_ = data_real[:8616,:].reshape(-1,24)
    #class_label = data_npz["arr_1"]
    daily_slice = []
    daily_length_list = []
    daily_length_count = []

    for i in range(ori_data_.shape[0]):
        daily = ori_data_[i,:]
        sub_slice = []
        for j in range(24):
            if daily[j] != 0:
                daily_length_count.append(j)
                sub_slice.append(daily[j])
        daily_length_list.append(len(sub_slice)) 
        daily_slice.append(sub_slice)  

    #print(len(daily_slice))####416  
    pv_drop0 = None
    for i in range(len(daily_slice)):
        fea = np.array(daily_slice[i])
        if pv_drop0 is None:
            pv_drop0 = fea
        else:
            pv_drop0 = np.concatenate([pv_drop0,fea],axis=0)

    X = []
    Y = []
    for i in range(pv_drop0.shape[0] - sequence):
        X.append(pv_drop0[i:(i + sequence)])##[4,1]
        Y.append(pv_drop0[i + sequence])##1
    print(np.array(X).shape)

    def normalization(data):
        _range = np.max(data) - np.min(data)
        return (data - np.min(data)) / _range

    max_value = {}
    min_value = {}

    # 构建batch
    total_len = len(Y)
    train_len = int(0.9*total_len)
    test_len = total_len-train_len
    
    trainx_pre = np.array(X)
    trainx = normalization(trainx_pre)
    
    max_value["trainx"] = np.max(trainx_pre)
    min_value["trainx"] = np.min(trainx_pre)
    
    trainy_pre = np.array(Y)
    trainy = normalization(trainy_pre)

    max_value["trainy"] = np.max(trainy_pre)
    min_value["trainy"] = np.min(trainy_pre)

    return trainx, trainy

def get_fake_data(data_path,sequence):
    
    data_npz = np.load(
            data_path)
    data_real = pd.read_csv("/remote-home/21310019/2024/pv_TSTR/TS/data_folder/ori_pv_start_from_0101.csv",usecols=["PV/kwh"]).values
    
    ori_data = data_npz["arr_0"].reshape(-1,24)
    data_real_ = np.concatenate([data_real[:8616,:],data_real[9408:,:]],axis = 0).reshape(-1,24)
    ori_data_ = np.concatenate([data_real_,ori_data],axis = 0)
    #class_label = data_npz["arr_1"]
    daily_slice = []
    daily_length_list = []
    daily_length_count = []

    for i in range(ori_data_.shape[0]):
        daily = ori_data_[i,:]
        sub_slice = []
        for j in range(24):
            if daily[j] != 0:
                daily_length_count.append(j)
                sub_slice.append(daily[j])
        daily_length_list.append(len(sub_slice)) 
        daily_slice.append(sub_slice)  

    #print(len(daily_slice))####416  
    pv_drop0 = None
    for i in range(len(daily_slice)):
        fea = np.array(daily_slice[i])
        if pv_drop0 is None:
            pv_drop0 = fea
        else:
            pv_drop0 = np.concatenate([pv_drop0,fea],axis=0)

    X = []
    Y = []
    for i in range(pv_drop0.shape[0] - sequence):
        X.append(pv_drop0[i:(i + sequence)])##[4,1]
        Y.append(pv_drop0[i + sequence])##1
    print(np.array(X).shape)

    def normalization(data):
        _range = np.max(data) - np.min(data)
        return (data - np.min(data)) / _range

    max_value = {}
    min_value = {}

    # 构建batch
    total_len = len(Y)
    train_len = int(0.9*total_len)
    test_len = total_len-train_len
    
    trainx_pre = np.array(X[:int(0.9 * total_len)])
    trainx = normalization(trainx_pre)
    
    max_value["trainx"] = np.max(trainx_pre)
    min_value["trainx"] = np.min(trainx_pre)
    
    trainy_pre = np.array(Y[:int(0.9 * total_len)])
    trainy = normalization(trainy_pre)

    max_value["trainy"] = np.max(trainy_pre)
    min_value["trainy"] = np.min(trainy_pre)
     
    testx_pre = np.array(X[int(0.9 * total_len):])
    testx = normalization(testx_pre)
    
    max_value["testx"] = np.max(testx_pre)
    min_value["testx"] = np.min(testx_pre)

    testy_pre = np.array(Y[int(0.9 * total_len):])
    testy = normalization(testy_pre)
    max_value["testy"] = np.max(testy_pre)
    min_value["testy"] = np.min(testy_pre)


    return trainx, trainy,testx,testy


class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels,4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )
    
    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)

'''    
##test dim of sa
    #if __name__ == '__main__':
    #    net =SelfAttention(channels=4,size = 1)
    #    #print(sum([p.numel() for p in net.parameters()]))
    #    x = torch.randn(32, 4)
    #    x_ = x.reshape(32,4,1,1)
    #    print(x_.shape)
    #    print(net(x_).shape)   
'''

class ps_pred(nn.Module):

    def __init__(self,in_dim,seq_len,
                 ln_hiddensize,
                 batch_size,
                 cnn1):
        super(ps_pred,self).__init__()
        self.in_dim = in_dim
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.cnn1 = cnn1
        self.num_layers = 2
        self.hidden_size = ln_hiddensize

        self.sa1 = SelfAttention(channels=self.seq_len, size = self.in_dim)
        
        self.cnn=nn.Sequential(
                nn.Conv1d(in_channels=self.in_dim,out_channels=self.cnn1,kernel_size=3,stride=1,padding=1),
                nn.BatchNorm1d(num_features=self.cnn1,affine=True)
                            )
        self.fc = nn.Sequential(
                nn.Flatten(start_dim=1,end_dim=2),
                nn.Linear(self.cnn1*self.seq_len,self.hidden_size),
                nn.LeakyReLU(),
                nn.Linear(self.hidden_size,1),
                nn.LeakyReLU()
                )
    def forward(self, x):
        x_ = x.reshape(self.batch_size,self.seq_len,1,1)
        x_sa = self.sa1(x_)
        x_sa = x_sa.squeeze(3)
        x_cnn_input = x_sa.permute(0,2,1)
        x_ = self.cnn(x_cnn_input)
        output = self.fc(x_)
        return output

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

def train(gen, dataloader, device, opt_gen):

    for real_seq,real_label in dataloader:
        
        real_seq = real_seq.unsqueeze(2).to(device)#[b,4,1]
        real_label = real_label.unsqueeze(1).to(device)#[b,1]
        prediction = gen(real_seq)
        mse_loss =  torch.nn.MSELoss(reduction='none')
        G_loss_1 = torch.mean(mse_loss(prediction,real_label))
        G_loss = G_loss_1
        opt_gen.zero_grad()
        G_loss.backward(retain_graph = True)
        opt_gen.step()

    return G_loss.item()

def eval(eval_epoch,pred_num,fig_path,test_data,gen,gen_weight_path,signal):

    gen.load_state_dict(torch.load(gen_weight_path+'gen_{}epoch.pth'.format(str(eval_epoch).zfill(4))))
    
    preds_list = []
    labels_list = []
    mse_list =[]
    for idx, (x, labels) in enumerate(test_data):
        x = x.unsqueeze(2).to(device)#[b,4,1]
        pred = gen(x)#[b,1]
        pred = pred.squeeze(1).detach().cpu().tolist()#[b,]
        preds_list.extend(pred)
        label = labels.tolist()
        labels_list.extend(label)
        for i in range(len(pred)):
            mse_list.append(np.sqrt(abs(pred[i]**2-label[i]**2)))
    #print(len(preds_list))    
    #print("mean_absolute_error:", mean_absolute_error(labels_list, preds_list))
    #print("mean_squared_error:", mean_squared_error(labels_list, preds_list))
    print("{}_rmse:".format(signal), sqrt(mean_squared_error(labels_list, preds_list)))
    
    plt.figure()
    plt.title('pred & real {}'.format(signal))
    plt.xlabel('timesteps')
    plt.ylabel('values')
    plt.ylim(0,1)
    plt.plot(np.arange(pred_num), preds_list[:pred_num], '-g')
    plt.plot(np.arange(pred_num),labels_list[:pred_num],'-b')
    plt.legend(['pred', 'real'])
    plt.savefig(fig_path + 'eval_{}.png'.format(signal), bbox_inches='tight')
    plt.close()
    
    return mse_list#,(mean_squared_error(labels_list, preds_list),)#,

if __name__ == '__main__':

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    print(f"Using {device} device")

    gen_fake_weight_path = "/remote-home/21310019/2024/pv_TSTR/TR/model/gen/"
    #gen_real_weight_path = "/remote-home/21310019/2024/passenger_flow/TSTR/TR/model_withoutga/gen/"
    loss_path = "/remote-home/21310019/2024/pv_TSTR/TR/model/loss/"
    fig_path =  "/remote-home/21310019/2024/pv_TSTR/TR/model/eval/"
    if not os.path.exists(gen_fake_weight_path):
        os.makedirs(gen_fake_weight_path)
    if not os.path.exists(loss_path):
        os.makedirs(loss_path)
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    num_filters=128
    hidden_size=128
    learning_rate=0.001
    BATCH_SIZE =32
    gen = ps_pred(in_dim=1,seq_len=4,
                 ln_hiddensize=hidden_size,
                 batch_size=BATCH_SIZE,
                 cnn1=num_filters)
    
    beta_1 = 0.5
    beta_2 = 0.999
    gen_optimizer = torch.optim.Adam(gen.parameters(), lr=learning_rate, betas=(beta_1, beta_2))#优化控制学习率
    gen = gen.apply(weight_init).to(device)
    trainx,trainy = get_train_real_data(sequence=4 )
    testx_real,testy_real = get_test_real_data(sequence=4)
    train_set = Mydataset(trainx,trainy)#,transform=transforms.ToTensor())
    #test_set_fake = Mydataset(testx,testy)
    test_set_real = Mydataset(testx_real,testy_real)

    test_data_real = DataLoader(dataset=test_set_real, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, drop_last=True)
    #test_data_fake = DataLoader(dataset=test_set_fake, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, drop_last=True)

    train_data = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, drop_last=True)
    # net.load_state_dict(torch.load(net_weight_path + 'net_0036epoch.pth'))
    epochs =2500
    epoch_index = []
    gen_loss_index = []

    start_time = datetime.datetime.now()

    #logger = trange(epochs, desc=f"gen_loss")
    '''
    for epoch in range(epochs):

        epoch_ = epoch + 1
        #print(f'Epoch {epoch_}\n-----------------------------------------------')
        #print('training set start')
        #print('-----------------------------------------------')
        gen_loss= train(dataloader=train_data, gen=gen, 
                                    opt_gen=gen_optimizer, device=device)

        torch.cuda.empty_cache()#清空显存缓冲区
        epoch_index.append(epoch_)
        gen_loss_index.append(gen_loss)
        if epoch_ % 500 == 0:
            fig,(ax1) = plt.subplots(ncols=1,nrows=1,figsize = (4,5))
            ax1.set_title('Training Net at Epoch = {}'.format(str(epoch_).zfill(4)))
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('gen Loss')
            ax1.plot(epoch_index, gen_loss_index, '-k')

            plt.savefig(loss_path + '{}.png'.format(str(epoch_).zfill(4)), bbox_inches='tight')
            plt.close()
            torch.save(gen.state_dict(), gen_fake_weight_path + 'gen_{}epoch.pth'.format(str(epoch_).zfill(4)))
        #if epoch_ % 500 == 0:
        #    loss_list = eval(eval_epoch=epoch_,pred_num=BATCH_SIZE,fig_path=fig_path,test_data=test_data)
        #logger.set_description(f"Epoch: {epoch_},  G: {gen_loss:.4f}")
    '''
    loss_fake = eval(eval_epoch=epochs,pred_num=BATCH_SIZE,fig_path=fig_path,test_data=test_data_real,gen = gen,
                     gen_weight_path=gen_fake_weight_path,signal="real")##fake_rmse: 0.38362002964643344
    

