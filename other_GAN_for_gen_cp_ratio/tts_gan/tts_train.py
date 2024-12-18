import torch 
from torch import nn
import numpy as np

def gradient_penalty(dis,real,fake,device):
    batch_size = real.shape[0]
    seq_len= real.shape[-1]
    real = real.to(device)
    fake = fake.to(device)
    dis = dis.to(device)
    
    alpha = torch.rand(batch_size,1,1,1).to(device)
    interpolated = real*alpha + fake*(1-alpha)

    mix_scored = dis(interpolated)

    gradient = torch.autograd.grad(inputs=interpolated,
                                   outputs = mix_scored,
                                    grad_outputs= torch.ones_like(mix_scored),
                                    create_graph=True,
                                    retain_graph= True )[0]
    gradient = gradient.reshape(gradient.shape[0],-1)
    gradient_norm = gradient.norm(2,dim=1)
    gradient_penalty = torch.mean((gradient_norm-1)**2)
    return gradient_penalty

def train_gp(gen, dis, dataloader, device, opt_gen, opt_dis):

    for real,real_label in dataloader:

        real = real.to(device)#[b,24,11]
        real_label = real_label.to(device)
        for _ in range(1):
        #train discriminator
            opt_dis.zero_grad()
            batch_size = real.shape[0]
            seq_len =real.shape[1]

            Z= torch.randn(batch_size, 24, 11).to(device)
            t = Z.new_tensor([500] * Z.shape[0]).long()

            batch_size = real.shape[0]
            seq_len =real.shape[1]
            label_repeat = torch.repeat_interleave(real_label,seq_len,0).reshape(batch_size,seq_len,8)##[b,24,8]
            gen_input= torch.cat([Z,label_repeat],dim = -1).to(device)##[batch,24,19]
            fake_seq = gen(gen_input)##batch,24,11

            dis_input_fake = torch.cat([fake_seq,label_repeat],dim = 2).to(device)##[batch,24,19]
            dis_input_fake = dis_input_fake.unsqueeze(2)
            dis_input_fake = dis_input_fake.permute(0, 3, 2, 1)

            dis_input_real = torch.cat([real,label_repeat],dim = 2).to(device)##[batch,24,19]
            dis_input_real = dis_input_real.unsqueeze(2)
            dis_input_real = dis_input_real.permute(0, 3, 2, 1)

            D_logit_real =  dis(dis_input_real)
            D_logit_fake = dis(dis_input_fake.detach())

            gp = gradient_penalty(dis,dis_input_real,dis_input_fake,device)
            D_loss = torch.mean(D_logit_fake) - torch.mean(D_logit_real) + gp * 5
            opt_dis.zero_grad()
            D_loss.backward(retain_graph = True)
            torch.nn.utils.clip_grad_norm_(dis.parameters(), 5.)
            opt_dis.step()
            

            ## Generator Training
            for _ in range(10):
                opt_gen.zero_grad()

                batch_size = real.shape[0]
                seq_len =real.shape[1]

                Z= torch.randn(batch_size, 24, 11).to(device)
                t = Z.new_tensor([500] * Z.shape[0]).long()

                label_repeat = torch.repeat_interleave(real_label,seq_len,0).reshape(batch_size,seq_len,8)##[b,24,8]
                gen_input= torch.cat([Z,label_repeat],dim = -1).to(device)##[batch,24,19]
                fake_seq = gen(gen_input)##batch,24,11

                dis_input_fake = torch.cat([fake_seq,label_repeat],dim = 2).to(device)##[batch,24,19]
                dis_input_fake = dis_input_fake.unsqueeze(2)
                dis_input_fake = dis_input_fake.permute(0, 3, 2, 1)

                dis_input_real = torch.cat([real,label_repeat],dim = 2).to(device)##[batch,24,19]
                dis_input_real = dis_input_real.unsqueeze(2)
                dis_input_real = dis_input_real.permute(0, 3, 2, 1)

                D_logit_real =  dis(dis_input_real)
                D_logit_fake = dis(dis_input_fake.detach())
                G_loss = -torch.mean(D_logit_fake)
                
                G_loss.backward()
                torch.nn.utils.clip_grad_norm_(gen.parameters(), 5.)
                opt_gen.step()


        g_output_feature_dis = fake_seq.detach().cpu().numpy()[:,:,:-1]##[batch,24,10]
        g_output_feature_con = fake_seq.detach().cpu().numpy()[:,:,-1]##[batch,24]
        batch_fake_discrete = []
        batch_real_discrete = []
        for i in range(batch_size):
            fake_sample_discrete = g_output_feature_dis[i,:,:]
            batch_fake_discrete.append(np.argmax(fake_sample_discrete,axis = 1))
            batch_real_discrete.append(np.argmax(real.cpu().numpy()[i,:,:-1],axis = 1))
        class_label_ = np.argmax(real_label.cpu().numpy(),axis = 1)

    return G_loss.item(), D_loss.item(),class_label_,batch_fake_discrete,g_output_feature_con,\
        batch_real_discrete,real.cpu().numpy()[:,:,-1]