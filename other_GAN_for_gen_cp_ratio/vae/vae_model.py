import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self,feature_num,hidden_size,num_layers,batch_size,seq_length) :
        super(Encoder,self).__init__()

        self.feature_num = feature_num##19   
        self.hidden_size=hidden_size##128
        self.num_layers=num_layers##3
        self.batch_size=batch_size##51
        self.seq_length=seq_length##24
        self.num_directions=1 # 单向LSTM
        
        self.model = nn.LSTM(input_size=self.feature_num,hidden_size=self.hidden_size,num_layers=self.num_layers,batch_first=True)
        self.fc = nn.Sequential(
                    nn.Linear(self.num_layers*self.hidden_size,512),
                    nn.LeakyReLU(),
                    nn.Linear(512,self.hidden_size),
                    nn.LeakyReLU())

    def forward(self,x,device):

        h_0 = torch.randn(self.num_layers,self.batch_size,self.hidden_size).to(device)
        c_0 = torch.randn(self.num_layers,self.batch_size,self.hidden_size).to(device)
        
        output, (h_n,c_n) = self.model(x,(h_0,c_0))
        h_n = h_n.permute(0,1,2).view(self.batch_size,-1)
        return self.fc(h_n)##[batch_size,hidden_size]
    
class mean_std(nn.Module):
    def __init__(self,hidden_size,latent_length) :
        super(mean_std,self).__init__()

        self.hidden_size=hidden_size##256
        self.latent_length=latent_length##8

        self.hid_mu = nn.Sequential(
            nn.Linear(self.hidden_size,128),
            nn.Linear(128,32),
            nn.Linear(32,self.latent_length))
        self.hid_sigma = nn.Sequential(
            nn.Linear(self.hidden_size,128),
            nn.Linear(128,32),
            nn.Linear(32,self.latent_length))
    def forward(self,h_n):
        latent_mean = self.hid_mu(h_n)
        latent_logvar = self.hid_sigma(h_n)

        return latent_mean,latent_logvar##[batch_size,latent_length]

class Decoder(nn.Module):
    def __init__(self,hidden_size,latent_length,num_layers,batch_size,seq_length,feature_dim) :
        super(Decoder,self).__init__()

        self.latent_length = latent_length##8      
        self.hidden_size=hidden_size##128
        self.num_layers=num_layers##3
        self.batch_size=batch_size##
        self.seq_length=seq_length##24
        self.num_directions=1 # 单向LSTM
        self.feature_dim = feature_dim
        ##model_3
        '''
        self.lat_hid = nn.Sequential(nn.Flatten(start_dim=1,end_dim=-1),                                    
                                     nn.Linear(self.seq_length,self.latent_length),
                                     nn.LeakyReLU())
        '''
        ##model_2
        self.lat_hid = nn.Sequential(                                 
                                     nn.Linear(self.hidden_size,self.latent_length),
                                     nn.LeakyReLU())
        ##mdel_3
        
        self.model = nn.LSTM(input_size=self.latent_length,hidden_size=self.hidden_size,num_layers=self.num_layers,batch_first=True)
        self.hid_out = nn.Sequential(  
                                    nn.Flatten(start_dim=1,end_dim=-1),
                                    nn.Linear(in_features=self.hidden_size, out_features=240)
                                    )
        self.outln_continuous = torch.nn.Sequential(
            nn.Flatten(start_dim=1,end_dim=-1),
            nn.Linear(in_features=self.hidden_size, out_features=64),
            nn.LayerNorm(normalized_shape=[self.batch_size,64]),   
            nn.LeakyReLU(),
            nn.Linear(64,self.seq_length),
            nn.ReLU()
            )
        self.out_discrete =torch.nn.Softmax(dim = 1)
        '''
        ##model_2
        self.model = nn.LSTM(input_size=self.latent_length,hidden_size=2*self.hidden_size,num_layers=self.num_layers,batch_first=True)
        self.hid_out = nn.Sequential(  
                                    nn.Flatten(start_dim=1,end_dim=-1),
                                    nn.Linear(in_features=(self.hidden_size)*4, out_features=128),
                                    nn.LayerNorm(normalized_shape=[self.batch_size,128]),   
                                    nn.LeakyReLU(),
                                    nn.Linear(in_features=128, out_features=64),
                                    nn.LayerNorm(normalized_shape=[self.batch_size,64]),   
                                    nn.LeakyReLU(),
                                    nn.Linear(in_features=64, out_features=self.seq_length) )
        '''
    ##model_3
    ''' 
    def forward(self,latent,x):
        h_0 = torch.randn(self.num_layers,self.batch_size,self.hidden_size).to(device="cuda" if torch.cuda.is_available() else "cpu")
        c_0 = torch.randn(self.num_layers,self.batch_size,self.hidden_size).to(device="cuda" if torch.cuda.is_available() else "cpu")
        x = self.lat_hid(x)
        latent_ = latent.unsqueeze(1)
        noise = torch.rand_like(latent_).to(device="cuda" if torch.cuda.is_available() else "cpu")
        h = x.unsqueeze(1)+noise
        decoder_input = 0.3*latent_+0.4*noise+0.3*h
        decoder_output, (h_n,c_n) = self.model(decoder_input,(h_0,c_0))##[]
        output = self.hid_out(decoder_output)

        return self.sigmoid(output).view(self.batch_size,self.seq_length,self.output_size) ##[batch_size,seq_len,output_size]
    '''
    ##model_2
    def forward(self,latent,x,device):
        h_0 = torch.randn(self.num_layers,self.batch_size,self.hidden_size).to(device)
        c_0 = torch.randn(self.num_layers,self.batch_size,self.hidden_size).to(device)

        x = self.lat_hid(x)

        latent_ = latent.unsqueeze(1)##[b,1,8]
        noise = torch.rand_like(latent_).to(device)
        h = x.unsqueeze(1)+noise##[batch,1,8]
        decoder_input = 0.4*latent_+0.6*h
        
        decoder_output, (h_n,c_n) = self.model(decoder_input,(h_0,c_0))##[b,1,256]

        output_con = self.outln_continuous(decoder_output)
        output_con = output_con.unsqueeze(2)#[b,24,1]

        output_dis1 = self.hid_out(decoder_output)
        output_dis = output_dis1.view(self.batch_size,self.seq_length,10)
        output_dis = self.out_discrete(output_dis)

        return torch.cat([output_dis,output_con],dim = 2)##[b,24,11] ##[batch_size,seq_len,output_size]
'''
if __name__ == '__main__':
    # net = UNet(device="cpu")
    device =  "cpu"
    FEATURE_DIM = 11
    HIDDEN_SIZE = 256
    LATENT_LEN = 8
    OUTPUT_SIZE = 11
    NUM_LAYRS = 3
    BATCH_SIZE = 3
    SEQ_LEN = 24
    
    enc = Encoder(feature_num=FEATURE_DIM+8,hidden_size=HIDDEN_SIZE,num_layers=NUM_LAYRS,
                  batch_size=BATCH_SIZE,seq_length=SEQ_LEN).to(device)
    m_s = mean_std(hidden_size=HIDDEN_SIZE,latent_length=LATENT_LEN).to(device)
    dec = Decoder(hidden_size=HIDDEN_SIZE,latent_length=LATENT_LEN,
                  num_layers=NUM_LAYRS,batch_size=BATCH_SIZE,seq_length=SEQ_LEN,feature_dim=FEATURE_DIM).to(device)
    
    enc_input = torch.randn(3, 24,19).to(device)

    h_end = enc(enc_input,device)
    latent_mean,latent_logvar = m_s(h_end)

    std = torch.exp(0.5*latent_logvar)
    eps = torch.randn_like(std)
    latent = eps.mul(std).add(latent_mean)
    ##x_decoded = dec(latent,x)##model 3
    print(latent.shape,h_end.shape)
    x_decoded = dec(latent,h_end.detach(),device)##2.0
    print(x_decoded.shape)
'''