import torch
from torch import nn
import numpy as np
import cp_model


def cal_g_loss_gp(d_fake_train
              ):
        ##calculte loss
        g_loss = -torch.mean(d_fake_train)
        return g_loss
def cal_g_loss_bce(d_fake_train,
               device
              ):
        ##calculte loss
        adversarial_loss = nn.BCELoss()
        real_labels_fea = torch.ones_like(d_fake_train, device=device)
        G_loss_fea = adversarial_loss(d_fake_train, real_labels_fea)
        return G_loss_fea

def cal_d_loss_bce(d_fake_train,
                d_real_train,
                #attr_d_gp_coe,g_output_feature_train,real_feature_pl,g_output_attribute_train,real_attribute_pl,
                #discriminator,d_gp_coe,attr_discriminator,
                device): 
    adversarial_loss = nn.BCELoss()
    real_labels_fea = torch.ones_like(d_real_train, device=device)
    fake_labels_fea = torch.zeros_like(d_fake_train, device=device)

    d_loss_real_fea = adversarial_loss(d_real_train, real_labels_fea)
    d_loss_fake_fea = adversarial_loss(d_fake_train, fake_labels_fea)
    D_loss_fea = 0.5 * (d_loss_real_fea + d_loss_fake_fea)

    return D_loss_fea
def cal_d_loss_gp(batch_size,d_fake_train,#attr_d_fake_train
                d_real_train,#attr_d_real_train,
                g_output_feature_train,real_feature_pl,#g_output_attribute_train,
                real_attribute_pl,
                discriminator,d_gp_coe,#attr_discriminator,
                device):   
        
    d_loss_fake = torch.mean(d_fake_train)       
    d_loss_real = -torch.mean(d_real_train)

    alpha_dim2 = torch.rand(batch_size,24).to(device)
    alpha_dim3 = torch.unsqueeze(alpha_dim2, 2).to(device)

    differences_input_feature = (g_output_feature_train -real_feature_pl).to(device)
    interpolates_input_feature = (real_feature_pl +alpha_dim3 * differences_input_feature).to(device)
   
    mix_scored = discriminator(interpolates_input_feature,real_attribute_pl)

    gradient_fea= torch.autograd.grad(inputs=interpolates_input_feature,
                            outputs = mix_scored,
                            grad_outputs= torch.ones_like(mix_scored),
                            create_graph=True,
                            retain_graph= True )[0]
    gradient_fea = gradient_fea.view(gradient_fea.shape[0],-1)
    gradient_norm_fea = gradient_fea.norm(2,dim=1)
    gradient_penalty_fea = torch.mean((gradient_norm_fea-1)**2)
    
    d_loss_gp =gradient_penalty_fea

    d_loss = (d_loss_fake +d_loss_real +d_gp_coe * d_loss_gp)
    
    return d_loss

def train(dataloader,d_rounds,batch_size,device,
          gen_feature,discriminator,opt_dis,
          opt_gen,g_rounds,d_gp_coe):
    ema = cp_model.EMA(beta = 0.999)
    for batch_data_feature,batch_data_attribute in dataloader:

        batch_data_feature=batch_data_feature.to(device)
        class_label = batch_data_attribute.to(device)

        for _ in range(d_rounds):

            ##prepare_data:
            g_feature_input_noise= torch.randn(3, 24, 8, 8).to(device)
            t = g_feature_input_noise.new_tensor([500] * g_feature_input_noise.shape[0]).long()
            g_output_feature = gen_feature(g_feature_input_noise,
                                           t,
                                           class_label)##batch,20,
            
            d_fake_train = discriminator(g_output_feature.detach(),class_label)            
            d_real_train = discriminator(batch_data_feature,class_label)

            d_loss = cal_d_loss_bce(d_fake_train=d_fake_train,
                d_real_train=d_real_train,
                #attr_d_gp_coe,g_output_feature_train,real_feature_pl,g_output_attribute_train,real_attribute_pl,
                #discriminator,d_gp_coe,attr_discriminator,
                device=device)
            
            '''
            d_loss= cal_d_loss_gp(batch_size=batch_size,d_fake_train=d_fake_train,
                                            d_real_train=d_real_train,    
                                            #attr_d_real_train=attr_d_real_train,attr_d_gp_coe=attr_d_gp_coe,
                                            g_output_feature_train=g_output_feature,
                                            real_feature_pl=batch_data_feature,
                                            #g_output_attribute_train=g_output_attribute,
                                            real_attribute_pl=class_label,
                                            discriminator=discriminator,
                                            d_gp_coe=d_gp_coe,
                                            #attr_discriminator=attr_discriminator,
                                            device = device)
            '''
            
            opt_dis.zero_grad()

            d_loss.backward(retain_graph = True)
            '''
            if d_loss >= 0.4:
                opt_dis.step()
            else:
                pass
            '''

            for _ in range(g_rounds):

                g_feature_input_noise= torch.randn(3, 24, 8, 8).to(device)
                g_output_feature = gen_feature(g_feature_input_noise,t,
                                        class_label)

                d_fake_train = discriminator(g_output_feature.detach(),class_label)            
                d_real_train = discriminator(batch_data_feature,class_label)

                #g_loss = cal_g_loss(d_fake_train = d_fake_train,attr_d_fake_train = attr_d_fake_train,g_attr_d_coe=g_attr_d_coe)
                g_loss = cal_g_loss_bce(d_fake_train = d_fake_train,#attr_d_fake_train = attr_d_fake_train,
                                        #g_attr_d_coe=g_attr_d_coe,
                                        device = device)
                opt_gen.zero_grad()
                g_loss.backward()
                opt_gen.step()
            
        ema.step_ema(ema_model = gen_feature, model=gen_feature, step_start_ema=2000)

        ##g_output_feature:[batch,24,11]
        g_output_feature_dis = g_output_feature.detach().cpu().numpy()[:,:,:-1]##[batch,24,10]
        g_output_feature_con = g_output_feature.detach().cpu().numpy()[:,:,-1]##[batch,24]
        batch_fake_discrete = []
        batch_real_discrete = []
        for i in range(batch_size):
            fake_sample_discrete = g_output_feature_dis[i,:,:]
            batch_fake_discrete.append(np.argmax(fake_sample_discrete,axis = 1))
            batch_real_discrete.append(np.argmax(batch_data_feature.cpu().numpy()[i,:,:-1],axis = 1))
        class_label_ = np.argmax(class_label.cpu().numpy(),axis = 1)
    return d_loss.item(),g_loss.item(),\
       class_label_,batch_fake_discrete,g_output_feature_con,\
        batch_real_discrete,batch_data_feature.cpu().numpy()[:,:,-1]