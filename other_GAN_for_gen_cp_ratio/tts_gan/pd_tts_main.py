import tts_model_self
import tts_train

import torch
from torch import nn

import torch.utils.data.dataset as Dataset
import torch.utils.data.dataloader as DataLoader

import datetime
from tqdm import trange
import matplotlib.pyplot as plt
import os
import numpy as np

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

def get_data(path):

    data_npz = np.load(
            os.path.join(path, "data_pd_train.npz")
            )
    '''
    with open(os.path.join(path, "data_feature_output.pkl"), "rb") as f:
        data_feature_outputs = pickle.load(f)
    with open(os.path.join(path,"data_attribute_output.pkl"), "rb") as f:
        data_attribute_outputs = pickle.load(f)
    '''   
    data_feature = data_npz['arr_0']
    data_attribute = data_npz['arr_1']#["arr_1"]
    
    data_fea_1 = None##[88*24,11]
    data_fea_2 = None##
    data_fea_3 = None##

    data_att_1 = None##[88*8]
    data_att_2 = None##
    data_att_3 = None##
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

        elif np.all(ratio[6:19]==1) or np.all(ratio[5:19]==1) or \
            np.all(ratio[6:18]==1) or np.all(ratio[5:18]==1):

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
    return data_fea_3,data_att_3

if __name__ == '__main__':

    torch.manual_seed(10)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    print(f"Using {device} device")

    
    weight_path = "/remote-home/21310019/2024/cloudtype_pd_other_GANs/tts_gan/model/weights/"
    if not os.path.exists(weight_path):
        os.makedirs(weight_path)

   
    loss_path = "/remote-home/21310019/2024/cloudtype_pd_other_GANs/tts_gan/model/loss/"
    if not os.path.exists(loss_path):
        os.makedirs(loss_path)

    fig_path = "/remote-home/21310019/2024/cloudtype_pd_other_GANs/tts_gan/model/fig/"
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)


    SAMPLE_LEN = 24
    BATCH_SIZE = 3

    data_fea_3,data_att_3 = get_data(path = "/remote-home/21310019/2024/cloudtype_pd_other_GANs")
    train_set = subDataset(data_fea_3,data_att_3)
    train_data = DataLoader.DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)

    generator = tts_model_self.Generator(noise_dim=19, embed_dim=256,channels=256,seq_len=24,
                 n_head=4, d_k=16, d_v=32, batch_size=BATCH_SIZE,
                 depth=4,attn_drop_rate=0.1,forward_drop_rate=0.1,forward_expansion=4)
    
    discriminator = tts_model_self.Discriminator(in_channels=19,patch_size=4,emb_size=256, 
                 forward_expansion=4,seq_len = 24,depth=4, n_classes=1,n_head=4, d_k=16, d_v=32, batch_size=3,
                 attn_drop_rate=0.1,forward_drop_rate=0.1)
    lr = 2e-4
    beta_1 = 0.5
    beta_2 = 0.999

    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=lr*2, betas=(beta_1, beta_2))#优化控制学习率
    gen = generator.apply(weight_init).to(device)

    dis_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr/5, betas=(beta_1, beta_2))
    dis = discriminator.apply(weight_init).to(device)

    start_time = datetime.datetime.now()

    epochs = 20000
    gen_loss_index = []
    dis_loss_index = []
    epoch_index = []
    logger = trange(epochs, desc=f"Epoch: 0,  G_loss: 0, D_loss: 0")
    for epoch in logger:
        epoch_ = epoch + 1
        print(f'Epoch {epoch_}\n-----------------------------------------------')
        print('-----------------------------------------------')
        G_loss,D_loss,class_label,batch_fake_discrete,batch_fake_continuous,\
        batch_real_discrete,batch_real_continuous= tts_train.train_gp(gen=gen, dis=dis, dataloader=train_data, 
                                                  device=device,opt_gen=gen_optimizer, opt_dis=dis_optimizer)

        epoch_index.append(epoch_)
        gen_loss_index.append(G_loss)
        dis_loss_index.append(D_loss)
        fig,(ax1,ax2,ax3) = plt.subplots(ncols=1,nrows=3,figsize = (6,12))

        if epoch_ %  100 == 0:

            ax1.set_title('Training Net at Epoch = {}'.format(str(epoch_).zfill(4)))
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('gen Loss')
            ax1.plot(epoch_index, gen_loss_index, '-b')

            ax2.set_title('Training Net at Epoch = {}'.format(str(epoch_).zfill(4)))
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('dis loss ')
            ax2.plot(epoch_index, dis_loss_index, '-k')

            ax3.set_title('Training Net at Epoch = {}'.format(str(epoch_).zfill(4)))
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('dis loss & gen loss')
            ax3.plot(epoch_index, gen_loss_index, '-b')
            ax3.plot(epoch_index, dis_loss_index, '-k')
            
            plt.savefig(loss_path + '{}epoch.png'.format(str(epoch_).zfill(4)), bbox_inches='tight')
            plt.close()
            torch.save(gen.state_dict(), weight_path + 'gen_{}epoch.pth'.format(str(epoch_).zfill(4)))

        if epoch_% 100 == 0:
            fig,axes = plt.subplots(ncols = 2,nrows = 3,figsize = (9,15))  
            #axes.flatten()
            for i in range(3):         
                axes[i,0].set_title('plot real&fake ratio at Epoch = {}'.format(str(epoch_).zfill(4)))
                axes[i,0].set_xlabel('time/h')
                axes[i,0].set_ylabel('class_{}'.format(class_label[i]))
                axes[i,0].set_yticks([0,0.2,0.4,0.6,0.8,1])
    
                fake_ = np.zeros_like(batch_fake_continuous[i,:])
                for j in range(24):
                    if batch_real_continuous[i,j] == 1:
                        fake_[j] = batch_real_continuous[i,j]
                    elif batch_real_continuous[i,j] != 0:
                        fake_[j] = batch_fake_continuous[i,j]
                    
                x =np.arange(24)
                axes[i,0].plot(x, fake_, '-g',label ="fake_{}".format(i))
                axes[i,0].plot(x, batch_real_continuous[i,:], '-b',label ="real_{}".format(i))
            
            for i in range(3):         
                axes[i,1].set_title('plot real&fake cloudtype at Epoch = {}'.format(str(epoch_).zfill(4)))
                axes[i,1].set_xlabel('Epoch')
                axes[i,1].set_yticks([0,2,4,6,8,10])
                axes[i,1].set_ylabel('class_{}'.format(class_label[i]))
                x =np.arange(24)
                axes[i,1].plot(x, batch_fake_discrete[i], '-g',label ="fake_{}".format(i))
                axes[i,1].plot(x, batch_real_discrete[i], '-b',label ="real_{}".format(i))
            plt.savefig(fig_path + 'fake&real_{}epoch.png'.format(str(epoch_).zfill(4)), bbox_inches='tight')
            plt.close()

        elapsed_time = datetime.datetime.now() - start_time
        logger.set_description(f"Epoch: {epoch_},  G: {G_loss:.4f}, D: {D_loss:.4f}")
        print(f"Time: {elapsed_time}\n")

    torch.cuda.empty_cache()

