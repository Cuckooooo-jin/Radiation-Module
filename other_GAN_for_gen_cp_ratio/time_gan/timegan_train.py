import torch
import numpy as np
#from torch.utils.tensorboard import SummaryWriter

def embedding_train(embedder,recovery,dataloader,emb_opt,rec_opt,device):

    for seq,label in dataloader:

        seq = seq.to(device)#[b,24,11]
        label = label.to(device)##[b,8]

        batch_size = seq.shape[0]
        seq_len = seq.shape[1]

        label_repeat = torch.repeat_interleave(label,seq_len,0).reshape(batch_size,seq_len,8)##[b,24,8]
        emb_input= torch.cat([seq,label_repeat],dim = -1).to(device)##[batch,24,19]

        emb_opt.zero_grad()
        rec_opt.zero_grad()

        # Forward Pass
        H = embedder(emb_input)
        X_tilde = recovery(H)##[b,24,11]

        # Reconstruction Loss
        E_loss_T0 = torch.nn.functional.mse_loss(X_tilde, seq)
        E_loss0 = 10 * torch.sqrt(E_loss_T0)
        
        loss = np.sqrt(E_loss_T0.item())
        E_loss0.backward()

        emb_opt.step()
        rec_opt.step()

        return loss##np.sqrt(E_loss_T0.item())

def supervisor_trainer(embedder,supervisor,dataloader,sup_opt,device):

    for seq, label in dataloader:

        seq = seq.to(device)
        label = label.to(device)##[b,8]

        batch_size = seq.shape[0]
        seq_len = seq.shape[1]

        label_repeat = torch.repeat_interleave(label,seq_len,0).reshape(batch_size,seq_len,8)##[b,24,8]
        emb_input= torch.cat([seq,label_repeat],dim = -1).to(device)##[batch,24,19]

        # Reset gradients
        sup_opt.zero_grad()

        # Supervision Forward Pass
        H = embedder(emb_input)
        H_hat_supervise = supervisor(H)

        # Supervised loss
        S_loss = torch.nn.functional.mse_loss(H_hat_supervise[:,:-1,:], H[:,1:,:])        # Teacher forcing next output

        # Backward Pass
        S_loss.backward()
        loss = np.sqrt(S_loss.item())

        # Update model parameters
        sup_opt.step()

    return loss

#joint trainer dis在先
def joint_trainer(embedder,recovery,supervisor,generator,discriminator,
                dataloader,emb_opt,rec_opt,sup_opt,gen_opt,dis_opt,device,
                batch_size,seq_len,Z_dim,gamma,dis_thresh):

    for seq , label in dataloader:
        seq = seq.to(device)
        label = label.to(device)
        ##dis trainning
        for _ in range(1):
            # Random Generator
            Z = torch.rand(batch_size, seq_len, Z_dim)
            #Z = torch.tensor(0.5*np.ones_like(z)+np.random.uniform(low=-0.5,high=0.5,size =[batch_size,seq,Z_dim]),dtype=torch.float32)
            Z = Z.to(device)

            batch_size = seq.shape[0]
            seq_len = seq.shape[1]
            label_repeat = torch.repeat_interleave(label,seq_len,0).reshape(batch_size,seq_len,8)##[b,24,8]
            emb_input= torch.cat([seq,label_repeat],dim = -1).to(device)##[batch,24,19]
            gen_input= torch.cat([Z,label_repeat],dim = -1).to(device)##[batch,24,19]

            ## Discriminator Training
            dis_opt.zero_grad()
            # Forward Pass
            # Real
            H = embedder(emb_input).detach()
        
            # Generator
            E_hat = generator(gen_input).detach()
            H_hat = supervisor(E_hat).detach()

            # Forward Pass
            Y_real = discriminator(H)            # Encoded original data
            Y_fake = discriminator(H_hat)        # Output of generator + supervisor
            Y_fake_e =discriminator(E_hat)      # Output of generator

            D_loss_real = torch.nn.functional.binary_cross_entropy_with_logits(Y_real, torch.ones_like(Y_real))
            D_loss_fake = torch.nn.functional.binary_cross_entropy_with_logits(Y_fake, torch.zeros_like(Y_fake))
            D_loss_fake_e = torch.nn.functional.binary_cross_entropy_with_logits(Y_fake_e, torch.zeros_like(Y_fake_e))

            D_loss = D_loss_real + D_loss_fake + gamma * D_loss_fake_e

            # Check Discriminator loss
            if D_loss > dis_thresh:
                # Backward Pass
                D_loss.backward()
                # Update model parameters
                dis_opt.step()
            
            D_loss = D_loss.item()

            ## Generator Training
            for _ in range(10):
                # Random Generator
                Z = torch.rand(batch_size, seq_len, Z_dim)
                #Z = torch.tensor(0.5*np.ones_like(z)+np.random.uniform(low=-0.5,high=0.5,size = [batch_size,seq,Z_dim]),dtype=torch.float32)
                Z = Z.to(device)
                
                batch_size = seq.shape[0]
                seq_len = seq.shape[1]
                label_repeat = torch.repeat_interleave(label,seq_len,0).reshape(batch_size,seq_len,8)##[b,24,8]
                emb_input= torch.cat([seq,label_repeat],dim = -1).to(device)##[batch,24,19]
                gen_input= torch.cat([Z,label_repeat],dim = -1).to(device)##[batch,24,19]

                # Forward Pass (Generator)
                embedder.zero_grad()
                supervisor.zero_grad()
                generator.zero_grad()

                # Supervisor Forward Pass
                H = embedder(emb_input)
                H_hat_supervise = supervisor(H)

                # Generator Forward Pass
                E_hat = generator(gen_input)
                H_hat = supervisor(E_hat)

                # Synthetic data generated
                X_hat = recovery(H_hat)

                # Generator Loss
                # 1. Adversarial loss
                Y_fake = discriminator(H_hat)        # Output of supervisor
                Y_fake_e = discriminator(E_hat)      # Output of generator

                G_loss_U = torch.nn.functional.binary_cross_entropy_with_logits(Y_fake, torch.ones_like(Y_fake))
                G_loss_U_e = torch.nn.functional.binary_cross_entropy_with_logits(Y_fake_e, torch.ones_like(Y_fake_e))

                # 2. Supervised loss
                G_loss_S = torch.nn.functional.mse_loss(H_hat_supervise[:,:-1,:], H[:,1:,:])        # Teacher forcing next output

                # 3. Two Momments
                G_loss_V1 = torch.mean(torch.abs(torch.sqrt(X_hat.var(dim=0, unbiased=False) + 1e-6) - torch.sqrt(seq.var(dim=0, unbiased=False) + 1e-6)))
                G_loss_V2 = torch.mean(torch.abs((X_hat.mean(dim=0)) - (seq.mean(dim=0))))

                G_loss_V = G_loss_V1 + G_loss_V2

                # 4. Summation
                G_loss = G_loss_U + gamma * G_loss_U_e + 100 * torch.sqrt(G_loss_S) + 100 * G_loss_V

                G_loss.backward()
                G_loss = np.sqrt(G_loss.item())

                # Update model parameters
                #gen_opt.step()
                #sup_opt.step()

                #Embedding Forward Pass 
                recovery.zero_grad()
                # Forward Pass

                H = embedder(emb_input)
                H_hat_supervise = supervisor(H)
                X_tilde = recovery(H)
                # For Joint training
                G_loss_S = torch.nn.functional.mse_loss(H_hat_supervise[:,:-1,:], H[:,1:,:]) # Teacher forcing next output

                # Reconstruction Loss
                E_loss_T0 = torch.nn.functional.mse_loss(X_tilde, seq)
                E_loss0 = 10 * torch.sqrt(E_loss_T0)
                E_loss = E_loss0 + 0.1 * G_loss_S

                E_loss.backward()
                # Update model parameters
                emb_opt.step()
                rec_opt.step()           
                gen_opt.step()
                sup_opt.step()
        
        g_output_feature_dis = X_hat.detach().cpu().numpy()[:,:,:-1]##[batch,24,10]
        g_output_feature_con = X_hat.detach().cpu().numpy()[:,:,-1]##[batch,24]
        batch_fake_discrete = []
        batch_real_discrete = []
        for i in range(batch_size):
            fake_sample_discrete = g_output_feature_dis[i,:,:]
            batch_fake_discrete.append(np.argmax(fake_sample_discrete,axis = 1))
            batch_real_discrete.append(np.argmax(seq.cpu().numpy()[i,:,:-1],axis = 1))
        class_label_ = np.argmax(label.cpu().numpy(),axis = 1)

    return G_loss,E_loss.item(),D_loss,class_label_,batch_fake_discrete,g_output_feature_con,\
        batch_real_discrete,seq.cpu().numpy()[:,:,-1]

       