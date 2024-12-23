
import torch
import torch.nn as nn
import torch.nn.functional as F

from enum import Enum

class OutputType(Enum):
    CONTINUOUS = "CONTINUOUS"
    DISCRETE = "DISCRETE"

class Normalization(Enum):
    ZERO_ONE = "ZERO_ONE"
    MINUSONE_ONE = "MINUSONE_ONE"

class Output(object):
    def __init__(self, type_, dim, normalization=None, is_gen_flag=False):
        self.type_ = type_
        self.dim = dim
        self.normalization = normalization
        self.is_gen_flag = is_gen_flag

        if type_ == OutputType.CONTINUOUS and normalization is None:
            raise Exception("normalization must be set for continuous output")


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())

class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
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

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(1),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=1, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb

class UNet_conditional(nn.Module):
    def __init__(self, c_in=24, c_out=24, time_dim=256, num_classes=None, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128, 8)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, 8)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, 8)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128, 8)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, 8)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, 8)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)
        self.outln_continuous = nn.Sequential(
            nn.Linear(64,1),
            nn.ReLU()
            )
        self.outln_discrete = nn.Sequential(
            nn.Linear(64,10),
            nn.Softmax(dim = 1))
        if num_classes is not None:
            self.label_emb =nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                num_classes,
              self.time_dim
              ))
       #nn.Embedding(num_classes, time_dim)     
            

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, y):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)
        #print("t",t.shape)
        if y is not None:
            t += self.label_emb(y)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        #print(x2.shape)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        #print("x3",x3.shape)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)
        #print("x4",x4.shape)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)
        #print("x4",x4.shape)

        x = self.up1(x4, x3, t)
        #print("x",x.shape)

        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output_1 = self.outc(x)
        #print(output_1.shape)
        output_ = output_1.view(output_1.shape[0],output_1.shape[1],-1)
        #print(output_.shape)
        output_continuous = self.outln_continuous(output_)
        output_discrete = self.outln_discrete(output_)
        return torch.cat([output_discrete,output_continuous],dim = 2)

class Discriminator(nn.Module):

    def __init__(self,attr_dim,fea_len,fea_dim,num_units=256):
        super(Discriminator, self).__init__()
        self.attr_dim = attr_dim##8
        self.fea_len = fea_len##24
        self.fea_dim = fea_dim##11
        self.num_units = num_units
        self.flatten = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=- 1),
            nn.Linear(self.fea_len*self.fea_dim,self.num_units)
        )
        
        self.label_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(attr_dim,self.num_units)
        )
        self.model = nn.Sequential(
            nn.Linear(self.num_units*2,self.num_units),
            nn.LeakyReLU(),
            nn.Linear(self.num_units,1),
            #nn.LeakyReLU(),
            #nn.Linear(self.num_units,1),
            nn.Sigmoid()
        )

    def forward(self,feature,attribute):
        fea_fla = self.flatten(feature)
        attr_emb = self.label_emb(attribute)
        input_ = torch.cat([fea_fla,attr_emb],dim = -1)##[batch,256*2]
        output = self.model(input_)
        out = torch.squeeze(output,1)
        return out##[batch,]

'''
if __name__ == '__main__':
    # net = UNet(device="cpu")
    net = UNet_conditional(num_classes=8, device="cpu")
    print(sum([p.numel() for p in net.parameters()]))
    x = torch.randn(3, 24, 8, 8)
    t = x.new_tensor([500] * x.shape[0]).long()##[500,500,500]
    y = torch.randn(3,8)##[1,1,1]
    dis = Discriminator(attr_dim=8,fea_len=24,fea_dim=11,num_units=256)
    gen_output = net(x, t, y)
    g_output_feature_dis = gen_output.detach().numpy()[:,:,-1]
    print("g_output_feature_dis",g_output_feature_dis.shape)
    #print("dis_output",dis(net(x, t, y),y).shape)
    #print("gen_output",net(x, t, y).shape)
'''