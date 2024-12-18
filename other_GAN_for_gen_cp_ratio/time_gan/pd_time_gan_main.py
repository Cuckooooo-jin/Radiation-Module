
# -*- coding: UTF-8 -*-
# Local modules
import timegan_model
import timegan_train
##6s/7s/1min each epoch
import torch
from torch import nn

import torch.utils.data.dataset as Dataset
import torch.utils.data.dataloader as DataLoader
#from torch.utils.tensorboard import SummaryWriter

import datetime
from tqdm import trange
import matplotlib.pyplot as plt
import os
import numpy as np
import pickle

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
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    print(f"Using {device} device")

    weight_path_list = []
    loss_path_list = []
    
    for i in ['emb','rec','sup','joint']:
        weight_path = "/remote-home/21310019/2024/cloudtype_pd_other_GANs/time_gan/model/{}_weight/".format(str(i))
        if not os.path.exists(weight_path):
            os.makedirs(weight_path)
        weight_path_list.append(weight_path)
    
    for i in ['emb_only','sup_only','joint']:     
        loss_path = "/remote-home/21310019/2024/cloudtype_pd_other_GANs/time_gan/model/{}_loss/".format(str(i))
        if not os.path.exists(loss_path):
            os.makedirs(loss_path)
        loss_path_list.append(loss_path)


    fig_path = "/remote-home/21310019/2024/cloudtype_pd_other_GANs/time_gan/model/fig/"
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)


    SAMPLE_LEN = 24
    BATCH_SIZE = 3
    '''
    index_sample = random.sample(np.arange(data_feature.shape[0]).tolist(),120)
    minibatch_data_feature = []
    minibatch_data_attribute = []
    print(index_sample)
    for zz in index_sample:
        minibatch_data_feature.append(data_feature[zz,:,:])
        minibatch_data_attribute.append(data_attribute[zz,:])
    '''

    data_fea_3,data_att_3 = get_data(path = "/remote-home/21310019/2024/cloudtype_pd_other_GANs")
    train_set = subDataset(data_fea_3,data_att_3)
    train_data = DataLoader.DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)

    FEATURE_DIM =11
    ATT_DIM =8
    BATCH_SIZE = 3
    SEQ_LEN = 24
    HIDDEN_DIM = 256
    EMB_DIM = 11
    NUM_LAYERS = 3
    NOISE_DIM = FEATURE_DIM

    embedder = timegan_model.EmbeddingNetwork(feature_dim=FEATURE_DIM+ATT_DIM,
                                              hidden_dim=HIDDEN_DIM,
                                              num_layers=NUM_LAYERS,
                                              emb_dim=EMB_DIM)
    
    recovery = timegan_model.RecoveryNetwork(feature_dim=FEATURE_DIM,
                                             hidden_dim=HIDDEN_DIM,
                                             num_layers=NUM_LAYERS,emb_dim=EMB_DIM)
    
    generator = timegan_model.GeneratorNetwork(noise_dim=NOISE_DIM+ATT_DIM,hidden_dim=HIDDEN_DIM,num_layers=NUM_LAYERS,emb_dim=EMB_DIM)
    supervisor = timegan_model.SupervisorNetwork(hidden_dim=HIDDEN_DIM,emb_dim=EMB_DIM,num_layers=NUM_LAYERS)
    discriminator = timegan_model.DiscriminatorNetwork(hidden_dim=HIDDEN_DIM,num_layers=NUM_LAYERS,emb_dim=EMB_DIM)
    

    lr = 2e-4
    beta_1 = 0.5
    beta_2 = 0.999

    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=lr*2, betas=(beta_1, beta_2))#优化控制学习率
    gen = generator.apply(weight_init).to(device)

    dis_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr/5, betas=(beta_1, beta_2))
    dis = discriminator.apply(weight_init).to(device)

    emb_optimizer = torch.optim.Adam(embedder.parameters(), lr=lr, betas=(beta_1, beta_2))
    emb = embedder.apply(weight_init).to(device)

    rec_optimizer = torch.optim.Adam(recovery.parameters(), lr=lr, betas=(beta_1, beta_2))
    rec = recovery.apply(weight_init).to(device)

    sup_optimizer = torch.optim.Adam(supervisor.parameters(), lr=lr, betas=(beta_1, beta_2))
    sup = supervisor.apply(weight_init).to(device)
    
    g_lr=0.001   
    d_lr=0.0005
    beta_1 = 0.5
    beta_2 = 0.999

    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=lr*2, betas=(beta_1, beta_2))#优化控制学习率
    gen = generator.apply(weight_init).to(device)

    dis_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr/5, betas=(beta_1, beta_2))
    dis = discriminator.apply(weight_init).to(device)

    emb_optimizer = torch.optim.Adam(embedder.parameters(), lr=lr, betas=(beta_1, beta_2))
    emb = embedder.apply(weight_init).to(device)

    rec_optimizer = torch.optim.Adam(recovery.parameters(), lr=lr, betas=(beta_1, beta_2))
    rec = recovery.apply(weight_init).to(device)

    sup_optimizer = torch.optim.Adam(supervisor.parameters(), lr=lr, betas=(beta_1, beta_2))
    sup = supervisor.apply(weight_init).to(device)

    start_time = datetime.datetime.now()


##train embedding:
    emb_epochs = 2000
    emb_only_index = []
    emb_epoch_index = []
    for emb_epoch in range(emb_epochs):
        epoch_ = emb_epoch + 1
        emb_loss = timegan_train.embedding_train(embedder=emb,recovery=rec,dataloader=train_data,
                                               emb_opt=emb_optimizer,rec_opt=rec_optimizer,device=device)

        '''
        for name, parms in gen.named_parameters():
            print('-->gen_name:', name, '-->grad_requirs:', parms.requires_grad, 
                  '--gen_weight', torch.mean(torch.FloatTensor(parms.data)), 
                  ' -->gen_grad_value:', torch.mean(torch.FloatTensor(parms.grad)))
        '''
        emb_epoch_index.append(epoch_)
        emb_only_index.append(emb_loss)
        fig,ax1 = plt.subplots(ncols = 1,nrows = 1,figsize = (4,4))
        if epoch_ %  1000== 0:

            ax1.set_title('Training autoencoder at Epoch = {}'.format(str(epoch_).zfill(4)))
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('EMB Loss')
            ax1.plot(emb_epoch_index, emb_only_index, '-b')

            plt.savefig(loss_path_list[0]+'emb_{}epoch.png'.format(str(epoch_).zfill(4)), bbox_inches='tight')
            plt.close()
            torch.save(emb.state_dict(), weight_path_list[0]+'emb_{}epoch.pth'.format(str(epoch_).zfill(4)))
            torch.save(rec.state_dict(), weight_path_list[1]+'rec_{}epoch.pth'.format(str(epoch_).zfill(4)))
        
        emb_elapsed_time = datetime.datetime.now() - start_time
        print(f"emb_Time: {emb_elapsed_time}\n")

    ##train supervisor:
    sup_epochs = 2000
    sup_only_index = []
    sup_epoch_index = []
    
    for sup_epoch in range(sup_epochs):
        sup_epoch_ = sup_epoch + 1
        sup_loss = timegan_train.supervisor_trainer(embedder=emb,supervisor=sup,dataloader=train_data,sup_opt=sup_optimizer
                                                    ,device=device)
        sup_epoch_index.append(sup_epoch_)
        sup_only_index.append(sup_loss)
        fig,ax1 = plt.subplots(ncols = 1,nrows = 1,figsize = (4,4))
        
        if sup_epoch_ %  1000== 0:
            
            ax1.set_title('Training supervisor at Epoch = {}'.format(str(sup_epoch_).zfill(4)))
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('SUP Loss')
            ax1.plot(sup_epoch_index, sup_only_index, '-b')

            plt.savefig(loss_path_list[1]+'sup_{}epoch.png'.format(str(sup_epoch_).zfill(4)), bbox_inches='tight')
            plt.close()
            torch.save(sup.state_dict(), weight_path_list[2]+'sup_{}epoch.pth'.format(str(sup_epoch_).zfill(4)))

        #writter_sup.add_scalar("Supervisor/Loss:", sup_loss, sup_epoch_)
        #writter_sup.flush()
        sup_elapsed_time = datetime.datetime.now() - emb_elapsed_time
        print(f"sup_Time: {sup_elapsed_time}\n")

    joint_epochs = 20000
    joint_index = {}
    joint_index['emb'] = []
    joint_index["gen"] = []
    joint_index['dis'] =[]
    joint_epoch_index = []
    logger_joint = trange(joint_epochs, desc=f"Epoch: 0, E_loss: 0, G_loss: 0, D_loss: 0")

    for joint_epoch in logger_joint:
        joint_epoch_ = joint_epoch + 1
        print(f'joint Epoch {joint_epoch_}\n-----------------------------------------------')
        print('-----------------------------------------------')
        G_loss,E_loss,D_loss,class_label,batch_fake_discrete,batch_fake_continuous,\
        batch_real_discrete,batch_real_continuous= timegan_train.joint_trainer(embedder=emb,recovery=rec,supervisor=sup,generator=gen,discriminator=dis,
                                dataloader=train_data,emb_opt=emb_optimizer,rec_opt=rec_optimizer,sup_opt=sup_optimizer
                                ,gen_opt=gen_optimizer,dis_opt=dis_optimizer,device=device,
                                batch_size=BATCH_SIZE,seq_len=SEQ_LEN,Z_dim=NOISE_DIM,gamma=1,dis_thresh=0.15)
        
        logger_joint.set_description(f"Epoch: {joint_epoch_}, G_Loss: {G_loss:.4f} , D_Loss:{D_loss:.4f}")
        joint_epoch_index.append(joint_epoch_)
        joint_index['emb'].append(E_loss)
        joint_index['gen'].append(G_loss)
        joint_index['dis'].append(D_loss)
        fig,[ax1,ax2,ax3,ax4] = plt.subplots(ncols = 1,nrows = 4,figsize = (4,20))

        if joint_epoch_ % 100 == 0:

            ax1.set_title('Joint Training _Autoencoder at Epoch = {}'.format(str(joint_epoch_).zfill(4)))
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Embedding Loss')
            ax1.plot(joint_epoch_index, joint_index['emb'], '-b')

            ax2.set_title('Joint Training _Generator at Epoch = {}'.format(str(joint_epoch_).zfill(4)))
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Generator Loss')
            ax2.plot(joint_epoch_index, joint_index['gen'], '-b')

            ax3.set_title('Joint Training _Discriminator at Epoch = {}'.format(str(joint_epoch_).zfill(4)))
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Discriminator Loss')
            ax3.plot(joint_epoch_index, joint_index['dis'], '-b')

            ax4.set_title('Joint Training at Epoch = {}'.format(str(joint_epoch_).zfill(4)))
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Geneator&Discriminator Loss')
            ax4.plot(joint_epoch_index, joint_index['gen'], '-b')
            ax4.plot(joint_epoch_index, joint_index['dis'], '-g')

            plt.savefig(loss_path_list[2] + 'joint_{}epoch.png'.format(str(joint_epoch_).zfill(4)), bbox_inches='tight')
            plt.close()
            torch.save(sup.state_dict(), weight_path_list[-1]+'sup_{}epoch.pth'.format(str(joint_epoch_).zfill(4)))
            torch.save(gen.state_dict(), weight_path_list[-1]+'gen_{}epoch.pth'.format(str(joint_epoch_).zfill(4)))
            torch.save(rec.state_dict(), weight_path_list[-1]+'rec_{}epoch.pth'.format(str(joint_epoch_).zfill(4)))

        if joint_epoch_% 100 == 0:
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

        joint_elapsed_time = datetime.datetime.now() - sup_elapsed_time
        print(f"joint_Time: {joint_elapsed_time}\n")
    torch.cuda.empty_cache()
    
    
