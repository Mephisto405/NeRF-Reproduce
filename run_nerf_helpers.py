import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from math import pi as PI

# TODO: remove this dependency
#from torchsearchsorted import searchsorted


# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)


# Model
class NeRF(nn.Module):
    
    def __init__(self, inc_x=3, inc_d=3, L_x=10, L_d=4, width=256, embed=True):
        super(NeRF, self).__init__()
        # Implementation according to the paper (Fig. 7)
        # d = 0: inc_x       -> 256
        # d = 1: 256         -> 256
        # d = 2: 256         -> 256
        # d = 3: 256         -> 256
        # d = 4: 256         -> 256
        # d = 5: 256 + inc_x -> 256
        # d = 6: 256         -> 256
        # d = 7: 256         -> 256
        # d = 8: 256         -> 256 + 1 (no act except alpha)
        # d = 9: 256 + inc_d -> 128
        # d =10: 128         -> 3 (sigmoid)

        self.L_x = L_x
        self.L_d = L_d        
        self.embed = embed
        if embed:
            inc_x *= 2*self.L_x + 1
            inc_d *= 2*self.L_d + 1
        self.inc_x = inc_x
        self.inc_d = inc_d
        assert inc_x != 0
        print('Input dimension: ', inc_x)
        print('Input view dimension: ', inc_d)
        
        # model specification
        dims = [[self.inc_x, width], 
                [width, width], 
                [width, width], 
                [width, width], 
                [width, width], 
                [width + self.inc_x, width], 
                [width, width], 
                [width, width], 
                [width, width + 1], # no act except alpha
                [width + self.inc_d, width//2], 
                [width//2, 3]] # sigmoid
        
        layers = []
        for i in range(0,5):
            _inc, _outc = dims[i]
            #layers.append(nn.Conv1d(_inc, _outc, 1, bias=True, groups=1))
            layers.append(nn.Linear(_inc, _outc))
            layers.append(nn.ReLU(inplace=True))
        self.embed_layers = nn.Sequential(*layers)
        
        layers = []
        for i in range(5,9):
            _inc, _outc = dims[i]
            #layers.append(nn.Conv1d(_inc, _outc, 1, bias=True, groups=1))
            layers.append(nn.Linear(_inc, _outc))
            if i != 8:
                layers.append(nn.ReLU(inplace=True))
        self.feat_alpha_layers = nn.Sequential(*layers)
        
        layers = []
        for i in range(9,len(dims)):
            _inc, _outc = dims[i]
            #layers.append(nn.Conv1d(_inc, _outc, 1, bias=True, groups=1))
            layers.append(nn.Linear(_inc, _outc))
            if i == len(dims) - 1:
                pass #layers.append(nn.Sigmoid())
            else:
                layers.append(nn.ReLU(inplace=True))
        self.rgb_layers = nn.Sequential(*layers)
       
    def pos_embed_layer(self, x, L=4):
        # x: (B*S, C)
        exps = torch.tensor([2.0**i for i in range(L)], device=x.device)
        exps = exps.reshape(1, -1) # (1, L)
        
        y = x[...,None] @ exps # (B*S, C, 1) @ (1, L) = (B*S, C, L)
        y = torch.cat([torch.sin(PI * y), torch.cos(PI * y)], dim=-1) # (B*S, C, 2*L)
        y = y.reshape(*(y.shape[:-2]), y.shape[-2]*y.shape[-1]) # (B*S, C*2*L)
        
        return torch.cat([x, y], dim=1)
    
    def forward(self, x):
        # x: (B*S, C) 
        if self.inc_d == 0: # view independent model
            pos = x
        else:
            pos, drc = torch.split(x, [3, 3], dim=-1)
        
        if self.embed:
            pos = self.pos_embed_layer(pos, L=self.L_x)
            if self.inc_d != 0:
                drc = self.pos_embed_layer(drc, L=self.L_d)
        
        feat = self.embed_layers(pos)
        
        feat = torch.cat([feat, pos], 1) # skip-connection
        feat = self.feat_alpha_layers(feat)
        alpha = feat[:,:1,...]
        
        if self.inc_d == 0:
            feat = feat[:,1:,...]
        else:
            feat = torch.cat([feat[:,1:,...], drc], 1)
        color = self.rgb_layers(feat)
        
        outputs = torch.cat([color, alpha], -1)
        return outputs
"""
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
        
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs    

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"
        
        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))    
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears+1]))
        
        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear+1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears+1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear+1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear+1]))
"""


# Ray helpers
def get_rays(H, W, focal, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, focal, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d
    
    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]
    
    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)
    
    return rays_o, rays_d


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    #inds = searchsorted(cdf, u, side='right')
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples
