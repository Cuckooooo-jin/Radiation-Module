import torch
from torch import nn
import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torch import Tensor


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

    def __init__(self,noise_dim, embed_dim,channels,seq_len,
                 n_head, d_k, d_v, batch_size,
                 depth,attn_drop_rate,forward_drop_rate,forward_expansion):
        #channels_input,feature_g,channels_output,num_layers,dropout,seq_len,batch_size,n_head):
        super(Generator,self).__init__()
        self.noise_dim = noise_dim
        self.channels = channels
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

        self.l1 = nn.Linear(self.noise_dim, self.embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len, self.embed_dim))
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
            
        self.deconv = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.channels, 1, 1, 0),
            nn.BatchNorm2d(self.channels),
            nn.ReLU()
        )
        self.outln_continuous = nn.Sequential(
            nn.Linear(self.channels,1),
            nn.ReLU()
            )
        self.outln_discrete = nn.Sequential(
            nn.Linear(self.channels,10),
            nn.Softmax(dim = 1))

    def forward(self, z):

        x = self.l1(z)
        x = x + self.pos_embed
        H = 1
        W = self.seq_len

        for _ in range(self.depth):
            res_att = x
            x = self.residual_att(x)
            x = x+res_att

            res_forward = x
            x = self.residual_feedforward(x)
            x = x+res_forward

        x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
        output = self.deconv(x.permute(0, 3, 1, 2))
        output = output.view(-1, self.channels, H*W)##[b,channels,1*seq_len]
        output_1 = output.permute(0,2,1)
        #print(output_.shape)
        output_continuous = self.outln_continuous(output_1)
        output_discrete = self.outln_discrete(output_1)
        
        return torch.cat([output_discrete,output_continuous],dim = 2)
    

class Discriminator(nn.Sequential):
    def __init__(self, in_channels,patch_size,emb_size, attn_drop_rate,forward_drop_rate,
                 forward_expansion,seq_len,depth, n_classes,n_head,d_k, d_v, batch_size):
        super(Discriminator,self).__init__()
        self.depth = depth
        self.n_classes = n_classes
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.emb_size = emb_size
        self.seq_len = seq_len
        self.n_head =n_head
        self.d_k = d_k
        self.d_v = d_v
        self.batch_size = batch_size
        self.attn_drop_rate =attn_drop_rate
        self.forward_drop_rate = forward_drop_rate
        self.forward_expansion = forward_expansion


        self.projection = nn.Sequential(
            Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)',s1 = 1, s2 = self.patch_size),
            nn.Linear(self.patch_size*self.in_channels, self.emb_size))
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.emb_size))
        self.positions = nn.Parameter(torch.randn((self.seq_len // self.patch_size) + 1, self.emb_size))

        self.residual_att = nn.Sequential(
                nn.LayerNorm(self.emb_size),
                multi_SelfAttention(self.n_head, self.emb_size, self.d_k, self.d_v, self.batch_size),
                nn.Dropout(self.attn_drop_rate)
                )
        
        self.residual_feedforward = nn.Sequential(
                nn.LayerNorm(self.emb_size),
                nn.Linear(self.emb_size,self.forward_expansion * self.emb_size),
                nn.GELU(),
                nn.Dropout(self.forward_drop_rate),
                nn.Linear(self.forward_expansion * self.emb_size, self.emb_size),
                nn.Dropout(self.forward_drop_rate))
        
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(self.emb_size),
            nn.Linear(self.emb_size,self.n_classes))

        
    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        #prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)
        # position
        x += self.positions
        for _ in range(self.depth):
            res_att = x
            x = self.residual_att(x)
            x = x+res_att

            res_forward = x
            x = self.residual_feedforward(x)
            x = x+res_forward
        out_put = self.clshead(x)##(3, 256, 1, 24)
        return out_put

'''
if __name__ == '__main__':
    # net = UNet(device="cpu")
    net = Generator(noise_dim=19, embed_dim=256,channels=256,seq_len=24,
                 n_head=4, d_k=16, d_v=32, batch_size=3,
                 depth=4,attn_drop_rate=0.5,forward_drop_rate=0.5,forward_expansion=4)
    dis_net = Discriminator(in_channels=19,patch_size=4,emb_size=256, 
                 forward_expansion=4,seq_len = 24,depth=4, n_classes=1,n_head=4, d_k=16, d_v=32, batch_size=3,
                 attn_drop_rate=0.1,forward_drop_rate=0.1)
    print(sum([p.numel() for p in net.parameters()]))
    x = torch.randn(3, 24,19)
   
    gen_output = net(x)
    g_output_feature_dis = gen_output.detach().numpy()
    print("g_output",g_output_feature_dis.shape)

    x1 = torch.randn(3, 24,8)
    dis_input = torch.cat([gen_output,x1],dim = 2)
    dis_input = dis_input.unsqueeze(2)
    dis_input = dis_input.permute(0, 3, 2, 1)

    dis_out = dis_net(dis_input)
   
    dis_out = dis_out.detach().numpy()
    print("dis_output",dis_out.shape)

    #print("dis_output",dis(net(x, t, y),y).shape)
    #print("gen_output",net(x, t, y).shape)
'''