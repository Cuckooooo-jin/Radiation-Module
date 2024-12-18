import os
import torch
from torch import nn
import vae_model
import numpy as np
import torch.utils.data.dataset as Dataset
import torch.utils.data.dataloader as DataLoader
import datetime
from tqdm import trange
import matplotlib.pyplot as plt

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
            if isinstance(m,(nn.Conv2d,nn.ConvTranspose2d,nn.BatchNorm2d)):
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

def train(enc, m_s, dec,dataloader, device, opt_enc, opt_m_s, opt_dec):    
    
    for data,labels in dataloader:
        labels =labels.to(device)
        data = data.to(device)##(batch,24,11)
        batch_size = data.shape[0]
        seq_len = data.shape[1]
        label_repeat = torch.repeat_interleave(labels,seq_len,0).reshape(batch_size,seq_len,8)##[b,24,8]
        enc_input= torch.cat([data,label_repeat],dim = -1).to(device)##[batch,24,19]

        h_end = enc(enc_input,device)
        latent_mean,latent_logvar = m_s(h_end)

        std = torch.exp(0.5*latent_logvar)
        eps = torch.randn_like(std).to(device)
        latent = eps.mul(std).add(latent_mean)
        ##x_decoded = dec(latent,x)##model 3
        x_decoded = dec(latent,h_end.detach(),device)##2.0

        #comput loss
        loss_fn = nn.MSELoss()
        reconstruction_loss = loss_fn(x_decoded,data)
        #kl_div = -torch.sum(1 * torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))
        kl_loss = -0.5*torch.mean(1+latent_logvar-latent_mean.pow(2)-latent_logvar.exp())
        ##backprop
        loss = reconstruction_loss+kl_loss
        
        opt_enc.zero_grad()
        opt_dec.zero_grad()
        opt_m_s.zero_grad()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=dec.parameters(), max_norm=10, norm_type=2)

        opt_enc.step()
        opt_m_s.step()
        opt_dec.step()

        g_output_feature_dis = x_decoded.detach().cpu().numpy()[:,:,:-1]##[batch,24,10]
        g_output_feature_con = x_decoded.detach().cpu().numpy()[:,:,-1]##[batch,24]
        batch_fake_discrete = []
        batch_real_discrete = []
        for i in range(batch_size):
            fake_sample_discrete = g_output_feature_dis[i,:,:]
            batch_fake_discrete.append(np.argmax(fake_sample_discrete,axis = 1))
            batch_real_discrete.append(np.argmax(data.cpu().numpy()[i,:,:-1],axis = 1))
        class_label_ = np.argmax(labels.cpu().numpy(),axis = 1)

    return reconstruction_loss.item(),kl_loss.item(),loss.item(),class_label_,batch_fake_discrete,g_output_feature_con,\
        batch_real_discrete,data.cpu().numpy()[:,:,-1]


if __name__ == '__main__':

    #device = "cuda" if torch.cuda.is_available() else "cpu"
    device =  "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    print(f"Using {device} device")

    enc_weight_path = "/remote-home/21310019/2024/cloudtype_pd_other_GANs/vae/model/enc/"
    m_s_weight_path = "/remote-home/21310019/2024/cloudtype_pd_other_GANs/vae/model/m_s/"
    dec_weight_path = "/remote-home/21310019/2024/cloudtype_pd_other_GANs/vae/model/dec/"
    loss_path = "/remote-home/21310019/2024/cloudtype_pd_other_GANs/vae/model/loss/"
    fig_path = "/remote-home/21310019/2024/cloudtype_pd_other_GANs/vae/model/fig/"
    
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    if not os.path.exists(enc_weight_path):
        os.makedirs(enc_weight_path)
    
    if not os.path.exists(m_s_weight_path):
        os.makedirs(m_s_weight_path)
    
    if not os.path.exists(dec_weight_path):
        os.makedirs(dec_weight_path)

    if not os.path.exists(loss_path):
        os.makedirs(loss_path)
    
    FEATURE_DIM = 11
    HIDDEN_SIZE = 256
    LATENT_LEN = 8
    OUTPUT_SIZE = 11
    NUM_LAYRS = 3
    BATCH_SIZE = 3
    SEQ_LEN = 24
    
    enc = vae_model.Encoder(feature_num=FEATURE_DIM+8,hidden_size=HIDDEN_SIZE,num_layers=NUM_LAYRS,batch_size=BATCH_SIZE,seq_length=SEQ_LEN)
    m_s = vae_model.mean_std(hidden_size=HIDDEN_SIZE,latent_length=LATENT_LEN)
    dec = vae_model.Decoder(hidden_size=HIDDEN_SIZE,latent_length=LATENT_LEN,
                  num_layers=NUM_LAYRS,batch_size=BATCH_SIZE,seq_length=SEQ_LEN,feature_dim=FEATURE_DIM)
    
    enc.apply(weight_init).to(device)
    m_s.apply(weight_init).to(device)
    dec.apply(weight_init).to(device)
  
    lr = 2e-4
    beta_1 = 0.5
    beta_2 = 0.99

    enc_optimizer = torch.optim.Adam(enc.parameters(), lr=lr, betas=(beta_1, beta_2))#优化控制学习率
    m_s_optimizer = torch.optim.Adam(m_s.parameters(), lr=lr, betas=(beta_1, beta_2))#优化控制学习率
    dec_optimizer = torch.optim.Adam(dec.parameters(), lr=lr, betas=(beta_1, beta_2))#优化控制学习率
    
    # net.load_state_dict(torch.load(net_weight_path + 'net_0036epoch.pth'))
    epochs =20000
    epoch_index = []
    overall_loss_index = []
    kl_loss_index = []
    recon_loss_index = []

    data_fea_3,data_att_3 = get_data(path = "/remote-home/21310019/2024/cloudtype_pd_other_GANs")
    train_set = subDataset(data_fea_3,data_att_3)
    train_data = DataLoader.DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)


    start_time = datetime.datetime.now()

    logger = trange(epochs, desc=f"gen_loss: 0, dis_loss: 0")
    
    for epoch in logger:

        epoch_ = epoch + 1
        print(f'Epoch {epoch_}\n-----------------------------------------------')
        print('training set start')
        print('-----------------------------------------------')
        
        reconstruction_loss,kl_loss,loss,class_label,batch_fake_discrete,batch_fake_continuous,\
        batch_real_discrete,batch_real_continuous= train(dataloader=train_data,enc=enc, m_s=m_s, dec=dec,
                                                device=device, opt_enc=enc_optimizer
                                                , opt_m_s=m_s_optimizer, opt_dec=dec_optimizer)
  

        torch.cuda.empty_cache()#清空显存缓冲区
        epoch_index.append(epoch_)
        overall_loss_index.append(loss)
        kl_loss_index.append(kl_loss)
        recon_loss_index.append(reconstruction_loss)

        logger.set_description(f"Epoch: {epoch}, Loss: {loss:.4f}")
        

        if epoch_ % 1000 == 0:
            fig,(ax1,ax2,ax3) = plt.subplots(ncols=1,nrows=3,figsize = (4,12))
            ax1.set_title('Training Net at Epoch = {}'.format(str(epoch_).zfill(4)))
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('reconstruction loss + kl Loss')
            ax1.plot(epoch_index, overall_loss_index, '-k')
            
            ax2.set_title('Training Net at Epoch = {}'.format(str(epoch_).zfill(4)))
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('reconstruction loss ')
            ax2.plot(epoch_index, recon_loss_index, '-k')

            ax3.set_title('Training Net at Epoch = {}'.format(str(epoch_).zfill(4)))
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel(' kl Loss')
            ax3.plot(epoch_index, kl_loss_index, '-k')

            plt.savefig(loss_path + '{}.png'.format(str(epoch_).zfill(4)), bbox_inches='tight')
            plt.close()

            torch.save(enc.state_dict(), enc_weight_path + 'enc_{}epoch.pth'.format(str(epoch_).zfill(5)))
            torch.save(m_s.state_dict(), m_s_weight_path + 'm_s_{}epoch.pth'.format(str(epoch_).zfill(5)))
            torch.save(dec.state_dict(), dec_weight_path + 'dec_{}epoch.pth'.format(str(epoch_).zfill(5)))
        
        if epoch_% 1000 == 0:
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
        print(f"Time: {elapsed_time}\n")

    torch.cuda.empty_cache()