import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding import get_encoder
from activation import trunc_exp
from .renderer import NeRFRenderer

# Audio feature extractor
class AudioAttNet(nn.Module):
    def __init__(self, dim_aud=64, seq_len=8):
        super(AudioAttNet, self).__init__()
        self.seq_len = seq_len # 8
        self.dim_aud = dim_aud # 64
        self.attentionConvNet = nn.Sequential(  # b x subspace_dim x seq_len
            nn.Conv1d(self.dim_aud, 16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(16, 8, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(8, 4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(4, 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(2, 1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True)
        )
        self.attentionNet = nn.Sequential(
            nn.Linear(in_features=self.seq_len, out_features=self.seq_len, bias=True), # self.seq_len = 8
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # x: [1, seq_len, dim_aud]; x.shape - torch.Size([1, 8, 64])
        y = x.permute(0, 2, 1)  # [1, dim_aud, seq_len] # torch.Size([1, 64, 8])
        y = self.attentionConvNet(y)  # torch.Size([1, 1, 8])
        y = self.attentionNet(y.view(1, self.seq_len)).view(1, self.seq_len, 1) # torch.Size([1, 8, 1]), y.view(1, self.seq_len).shape - torch.Size([1, 8]), self.attentionNet(y.view(1, self.seq_len)).shape - torch.Size([1, 8])
        return torch.sum(y * x, dim=1) # [1, dim_aud] # torch.Size([1, 64])


# Audio feature extractor
class AudioNet(nn.Module):
    def __init__(self, dim_in=29, dim_aud=64, win_size=16):
        super(AudioNet, self).__init__()
        self.win_size = win_size # 16
        self.dim_aud = dim_aud # dim_aud = 64 # dim_in = 44
        self.encoder_conv = nn.Sequential(  # n x 29 x 16
            nn.Conv1d(dim_in, 32, kernel_size=3, stride=2, padding=1, bias=True),  # n x 32 x 8
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(32, 32, kernel_size=3, stride=2, padding=1, bias=True),  # n x 32 x 4
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1, bias=True),  # n x 64 x 2
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1, bias=True),  # n x 64 x 1
            nn.LeakyReLU(0.02, True),
        )
        self.encoder_fc1 = nn.Sequential(
            nn.Linear(64, 64),
            nn.LeakyReLU(0.02, True),
            nn.Linear(64, dim_aud),
        )

    def forward(self, x):
        half_w = int(self.win_size/2) # 8 # x.shape - torch.Size([8, 44, 16])
        x = x[:, :, 8-half_w:8+half_w] # x.shape - torch.Size([8, 44, 16])
        x = self.encoder_conv(x).squeeze(-1) # torch.Size([8, 64, 1]).squeeze(-1) = torch.Size([8, 64])
        x = self.encoder_fc1(x) # torch.Size([8, 64])
        return x

class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers):
        super().__init__()
        self.dim_in = dim_in # 96 # 65
        self.dim_out = dim_out # 2 # 65
        self.dim_hidden = dim_hidden # 64 # 64
        self.num_layers = num_layers # 3 # 3

        net = []
        for l in range(num_layers):
            net.append(nn.Linear(self.dim_in if l == 0 else self.dim_hidden, self.dim_out if l == num_layers - 1 else self.dim_hidden, bias=False))

        self.net = nn.ModuleList(net)
    
    def forward(self, x):
        for l in range(self.num_layers):
            x = self.net[l](x)
            if l != self.num_layers - 1:
                x = F.relu(x, inplace=True)
        return x


class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                 opt,
                 # main network
                 num_layers=3,
                 hidden_dim=64,
                 geo_feat_dim=64,
                 num_layers_color=2,
                 hidden_dim_color=64,
                 # audio pre-encoder
                 audio_dim=64,
                 # deform_ambient net
                 num_layers_ambient=3,
                 hidden_dim_ambient=64,
                 # ambient net
                 ambient_dim=2,
                 # torso net (hard coded for now)
                 ):
        super().__init__(opt)

        # audio embedding
        self.emb = self.opt.emb # False

        if 'esperanto' in self.opt.asr_model:
            self.audio_in_dim = 44 # 44
        elif 'deepspeech' in self.opt.asr_model:
            self.audio_in_dim = 29
        else:
            self.audio_in_dim = 32
            
        if self.emb: # False
            self.embedding = nn.Embedding(self.audio_in_dim, self.audio_in_dim) # 44, 44

        # audio network
        self.audio_dim = audio_dim # 64
        self.audio_net = AudioNet(self.audio_in_dim, self.audio_dim) # 44, 64 # CNN -> reduce spatial to 1, followed by FC layer

        self.att = self.opt.att # 2
        if self.att > 0: # 2
            self.audio_att_net = AudioAttNet(self.audio_dim) # 64

        # ambient network
        self.encoder, self.in_dim = get_encoder('tiledgrid', input_dim=3, num_levels=16, level_dim=2, base_resolution=16, log2_hashmap_size=16, desired_resolution=2048 * self.bound, interpolation='linear') # self.bound = 1 # GridEncoder, 32
        self.encoder_ambient, self.in_dim_ambient = get_encoder('tiledgrid', input_dim=ambient_dim, num_levels=16, level_dim=2, base_resolution=16, log2_hashmap_size=16, desired_resolution=2048, interpolation='linear') # GridEncoder, 32 # ambient_dim = 2

        self.num_layers_ambient = num_layers_ambient # 3
        self.hidden_dim_ambient = hidden_dim_ambient # 64
        self.ambient_dim = ambient_dim # 2

        self.ambient_net = MLP(self.in_dim + self.audio_dim, self.ambient_dim, self.hidden_dim_ambient, self.num_layers_ambient) # 32 + 64 = 96, 2, 64, 3

        # sigma network
        self.num_layers = num_layers # 3
        self.hidden_dim = hidden_dim # 64
        self.geo_feat_dim = geo_feat_dim # 64

        self.eye_dim = 1 if self.exp_eye else 0 # self.exp_eye = True # self.eye_dim = 1

        self.sigma_net = MLP(self.in_dim + self.in_dim_ambient + self.eye_dim, 1 + self.geo_feat_dim, self.hidden_dim, self.num_layers) # 65 (32 + 32 + 1), 65, 64, 3

        # color network
        self.num_layers_color = num_layers_color # 2
        self.hidden_dim_color = hidden_dim_color # 64
        self.encoder_dir, self.in_dim_dir = get_encoder('spherical_harmonics') # SHEncoder: input_dim=3 degree=4, 16
        
        self.color_net = MLP(self.in_dim_dir + self.geo_feat_dim + self.individual_dim, 3, self.hidden_dim_color, self.num_layers_color) # (16 + 64 + 4) 84, 3, 64, 2

        if self.torso: # True
            # torso deform network
            self.torso_deform_encoder, self.torso_deform_in_dim = get_encoder('frequency', input_dim=2, multires=10) # FreqEncoder: input_dim=2 degree=10 output_dim=42, 42
            self.pose_encoder, self.pose_in_dim = get_encoder('frequency', input_dim=6, multires=4) # FreqEncoder: input_dim=6 degree=4 output_dim=54, 54
            self.torso_deform_net = MLP(self.torso_deform_in_dim + self.pose_in_dim + self.individual_dim_torso, 2, 64, 3) # 104, 2, 64, 3

            # torso color network
            self.torso_encoder, self.torso_in_dim = get_encoder('tiledgrid', input_dim=2, num_levels=16, level_dim=2, base_resolution=16, log2_hashmap_size=16, desired_resolution=2048, interpolation='linear') # GridEncoder, 32
            # self.torso_net = MLP(self.torso_in_dim + self.torso_deform_in_dim + self.pose_in_dim + self.individual_dim_torso + self.audio_dim, 4, 64, 3)
            self.torso_net = MLP(self.torso_in_dim + self.torso_deform_in_dim + self.pose_in_dim + self.individual_dim_torso, 4, 32, 3) # 136, 4, 32, 3

       
    def encode_audio(self, a):
        # a: [1, 29, 16] or [8, 29, 16], audio features from deepspeech
        # if emb, a should be: [1, 16] or [8, 16]

        # fix audio traininig
        if a is None: return None # torch.Size([8, 44, 16]) # a.shape = torch.Size([8, 44, 16])

        if self.emb:
            a = self.embedding(a).transpose(-1, -2).contiguous() # [1/8, 29, 16]

        enc_a = self.audio_net(a) # [1/8, 64] # torch.Size([8, 64])

        if self.att > 0:
            enc_a = self.audio_att_net(enc_a.unsqueeze(0)) # torch.Size([1, 64]) # [1, 64] # CNN - stride 1, pad 1 -> reduce 8 channels to 1 channel while maintaining spatial dim
            
        return enc_a


    def forward_torso(self, x, poses, enc_a, c=None):
        # x: [N, 2] in [-1, 1]
        # head poses: [1, 6]
        # c: [1, ind_dim], individual code

        # test: shrink x
        x = x * self.opt.torso_shrink # self.opt.torso_shrink = 0.8, x.shape = torch.Size([65793, 2])

        # deformation-based 
        enc_pose = self.pose_encoder(poses) # torch.Size([1, 54])
        enc_x = self.torso_deform_encoder(x) # torch.Size([65793, 42])

        if c is not None:
            h = torch.cat([enc_x, enc_pose.repeat(x.shape[0], 1), c.repeat(x.shape[0], 1)], dim=-1) # torch.Size([65793, 104])
        else:
            h = torch.cat([enc_x, enc_pose.repeat(x.shape[0], 1)], dim=-1)

        dx = self.torso_deform_net(h) # dx.shape = torch.Size([65793, 2])

        x = (x + dx).clamp(-1, 1) # torch.Size([65793, 2])

        x = self.torso_encoder(x, bound=1) # torch.Size([65793, 32])

        # h = torch.cat([x, h, enc_a.repeat(x.shape[0], 1)], dim=-1)
        h = torch.cat([x, h], dim=-1) # torch.Size([65793, 136])

        h = self.torso_net(h) # torch.Size([65793, 4])

        alpha = torch.sigmoid(h[..., :1]) # torch.Size([65793, 1])
        color = torch.sigmoid(h[..., 1:]) # torch.Size([65793, 3])

        return alpha, color, dx


    def forward(self, x, d, enc_a, c, e=None):
        # torch.Size([202624, 3]), torch.Size([202624, 3]), torch.Size([1, 64]), torch.Size([4]), torch.Size([1, 1])
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]
        # enc_a: [1, aud_dim]
        # c: [1, ind_dim], individual code
        # e: [1, 1], eye feature

        # starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        # starter.record()

        if enc_a is None:
            ambient = torch.zeros_like(x[:, :self.ambient_dim])
            enc_x = self.encoder(x, bound=self.bound)
            enc_w = self.encoder_ambient(ambient, bound=1)
        else:
            
            enc_a = enc_a.repeat(x.shape[0], 1) # torch.Size([202624, 64]) # audio_encoder # x.shape[0] = 202624, # iter 2 - enc_a.shape = torch.Size([189696, 64])
            enc_x = self.encoder(x, bound=self.bound) # torch.Size([202624, 32]) # self.bound = 1

            # ender.record(); torch.cuda.synchronize(); curr_time = starter.elapsed_time(ender); print(f"enocoder_deform = {curr_time}"); starter.record()

            # ambient
            ambient = torch.cat([enc_x, enc_a], dim=1) # torch.Size([202624, 96])
            ambient = self.ambient_net(ambient).float() # torch.Size([202624, 2])
            ambient = torch.tanh(ambient) # map to [-1, 1] # torch.Size([202624, 2])

            # ender.record(); torch.cuda.synchronize(); curr_time = starter.elapsed_time(ender); print(f"de-an net = {curr_time}"); starter.record()
    
            # sigma
            enc_w = self.encoder_ambient(ambient, bound=1) # torch.Size([202624, 32])

        # ender.record(); torch.cuda.synchronize(); curr_time = starter.elapsed_time(ender); print(f"encoder = {curr_time}"); starter.record()

        if e is not None:
            h = torch.cat([enc_x, enc_w, e.repeat(x.shape[0], 1)], dim=-1) # torch.Size([202624, 65])
        else:
            h = torch.cat([enc_x, enc_w], dim=-1)

        h = self.sigma_net(h) # torch.Size([202624, 65]) # Linear + ReLu

        # ender.record(); torch.cuda.synchronize(); curr_time = starter.elapsed_time(ender); print(f"sigma_net = {curr_time}"); starter.record()
        sigma = trunc_exp(h[..., 0]) # torch.Size([202624])
        geo_feat = h[..., 1:] # torch.Size([202624, 64])

        # color
        enc_d = self.encoder_dir(d) # torch.Size([202624, 16])

        # ender.record(); torch.cuda.synchronize(); curr_time = starter.elapsed_time(ender); print(f"encoder_dir = {curr_time}"); starter.record()

        if c is not None:
            h = torch.cat([enc_d, geo_feat, c.repeat(x.shape[0], 1)], dim=-1) # torch.Size([202624, 84])
        else:
            h = torch.cat([enc_d, geo_feat], dim=-1)
        
        h = self.color_net(h) # torch.Size([202624, 3])
        # ender.record(); torch.cuda.synchronize(); curr_time = starter.elapsed_time(ender); print(f"color_net = {curr_time}"); starter.record()
        
        # sigmoid activation for rgb
        color = torch.sigmoid(h) # torch.Size([202624, 3])

        return sigma, color, ambient # ambient - torch.Size([202624, 2])


    def density(self, x, enc_a, e=None):
        # x: [N, 3], in [-bound, bound]

        if enc_a is None:
            ambient = torch.zeros_like(x[:, :self.ambient_dim])
            enc_x = self.encoder(x, bound=self.bound)
            enc_w = self.encoder_ambient(ambient, bound=1)
        else:

            enc_a = enc_a.repeat(x.shape[0], 1) 
            enc_x = self.encoder(x, bound=self.bound)

            # ender.record(); torch.cuda.synchronize(); curr_time = starter.elapsed_time(ender); print(f"enocoder_deform = {curr_time}"); starter.record()

            # ambient
            ambient = torch.cat([enc_x, enc_a], dim=1)
            ambient = self.ambient_net(ambient).float()
            ambient = torch.tanh(ambient) # map to [-1, 1]

            # ender.record(); torch.cuda.synchronize(); curr_time = starter.elapsed_time(ender); print(f"de-an net = {curr_time}"); starter.record()

            # sigma
            enc_w = self.encoder_ambient(ambient, bound=1)

        # ender.record(); torch.cuda.synchronize(); curr_time = starter.elapsed_time(ender); print(f"encoder = {curr_time}"); starter.record()

        if e is not None:
            h = torch.cat([enc_x, enc_w, e.repeat(x.shape[0], 1)], dim=-1)
        else:
            h = torch.cat([enc_x, enc_w], dim=-1)

        h = self.sigma_net(h)

        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        return {
            'sigma': sigma,
            'geo_feat': geo_feat,
        }


    # optimizer utils
    def get_params(self, lr, lr_net, wd=0):

        # ONLY train torso
        if self.torso:
            params = [
                {'params': self.torso_encoder.parameters(), 'lr': lr},
                {'params': self.torso_net.parameters(), 'lr': lr_net, 'weight_decay': wd},
                {'params': self.torso_deform_net.parameters(), 'lr': lr_net, 'weight_decay': wd},
            ]

            if self.individual_dim_torso > 0:
                params.append({'params': self.individual_codes_torso, 'lr': lr_net, 'weight_decay': wd})

            return params

        params = [
            {'params': self.audio_net.parameters(), 'lr': lr_net, 'weight_decay': wd}, 
            {'params': self.encoder.parameters(), 'lr': lr},
            {'params': self.encoder_ambient.parameters(), 'lr': lr},
            {'params': self.ambient_net.parameters(), 'lr': lr_net, 'weight_decay': wd},
            {'params': self.sigma_net.parameters(), 'lr': lr_net, 'weight_decay': wd},
            {'params': self.color_net.parameters(), 'lr': lr_net, 'weight_decay': wd}, 
        ]
        if self.att > 0:
            params.append({'params': self.audio_att_net.parameters(), 'lr': lr_net * 5, 'weight_decay': wd})
        if self.emb:
            params.append({'params': self.embedding.parameters(), 'lr': lr})
        if self.individual_dim > 0:
            params.append({'params': self.individual_codes, 'lr': lr_net, 'weight_decay': wd})
        if self.train_camera:
            params.append({'params': self.camera_dT, 'lr': 1e-5, 'weight_decay': 0})
            params.append({'params': self.camera_dR, 'lr': 1e-5, 'weight_decay': 0})

        return params