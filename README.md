# RAD-NERF

----------------------
`WHAT DOES INSTRUCT-PIX2PIX STABLE DIFFUSION MODEL AIM TO ACCOMPLISH?`

In simple words, RAD-NERF aims to make it possible to generate photo-realistic 3D renderings in real-time by introducing novelties in encoding image and audio separately in a grid based manner (inspired mainly from instant-ngp).

----------------------
Note - The repository (unofficial pytorch implementation) uses a mixture of `cuda` along with `pytorch` to design components related to `ray-marching`, `ray-composition`, and `grid-encoding` to take advantage of higher degree of parallelism and also faster-execution of certain MLP networks made possible through the introduction of fully-fused cuda kernels (which was also introduced in instant ngp). Since my exposure to cuda programming is pretty limited, I tried to make sense of whatever I could understand by reading those cuda files. In case, you feel I missed out/ have some mistakes in my understanding of the cuda kernels involved, please feel free to point out in the commets section which will definitely useful to make edits (if required).

----------------------
## LETS LOOK AT IDEAS INVOLVED FROM CODE IMPLEMENTATION POV STEP-BY-STEP

__1.__ `Getting Audio Features`

Pre-trained deepSpeech model from `AD-NeRF` paper is used to get audio features from input `.wav` files.

```
aud_features = np.load(self.opt.aud)
aud_features = torch.from_numpy(aud_features) # torch.Size([588, 16, 44])
if len(aud_features.shape) == 3:
    aud_features = aud_features.float().permute(0, 2, 1) # [N, 16, 29] --> [N, 29, 16] # torch.Size([588, 44, 16])
```

The first dimension of `588` indicate the number of audios slices of 40ms (which are trained with sliding window strategy, i.e., when training this ASR model, some subsection of data from previous and next windows are also considered).

```
def get_audio_features(features, att_mode, index):
    left = index - 4 # -4
    right = index + 4 # 4
    pad_left = 0
    pad_right = 0
    if left < 0:
        pad_left = -left # 4
        left = 0 # 0
    if right > features.shape[0]: # 588
        pad_right = right - features.shape[0]
        right = features.shape[0]
    auds = features[left:right] # torch.Size([4, 44, 16])
    if pad_left > 0:
        auds = torch.cat([torch.zeros_like(auds[:pad_left]), auds], dim=0) # torch.Size([8, 44, 16])
    if pad_right > 0:
        auds = torch.cat([auds, torch.zeros_like(auds[:pad_right])], dim=0) # [8, 16]
    return auds
```

For each audio feature of 40ms audio-sample, padding is done such that the current audio-feature sample is centered in a window of 8 audio-sample features to create a tensor of shape - torch.Size([8, 44, 16]).
Thus, 40ms audio corresponds to tensor of shape ([8, 44, 16]).

__2.__ `Pose data`

The pose data (rotation, translation and scaling) for rendering the image is taken in the form of `4*4` transformation matrix.

The pose data is also designed to be used at 25 fps. (0.04 frames per second)

```
self.poses = []
for f in tqdm.tqdm(frames, desc=f'Loading data'):
    pose = np.array(f['transform_matrix'], dtype=np.float32) # [4, 4] # (4, 4)
    self.poses.append(pose) # 1

def collate():
    # ...
    poses = self.poses[index].to(self.device) 
    # [B, 4, 4] 
    # poses.shape = torch.Size([1, 4, 4]
    # index = [0]
    
    rays = get_rays(poses, self.intrinsics, self.H, self.W, self.num_rays, self.opt.patch_size) 
    # rays.keys() - dict_keys(['i', 'j', 'inds', 'rays_o', 'rays_d']) 
    # rays['i'].shape = torch.Size([1, 202500])
    # rays['j'].shape = torch.Size([1, 202500])
    # rays['inds'].shape = torch.Size([1, 202500])
    # poses.shape = torch.Size([1, 4, 4])
    # self.intrinsics.shape = 4
    # self.intrinsics = array([1200., 1200.,  225.,  225.])
    # self.H = 450
    # self.W = 450; 
    # self.num_rays = -1 

    results['index'] = index 
    # for ind. code [0] # [0]
    
    results['H'] = self.H 
    # 450
    
    results['W'] = self.W 
    # 450
    
    results['rays_o'] = rays['rays_o'] 
    # torch.Size([1, 202500, 3])
    
    results['rays_d'] = rays['rays_d'] 
    # torch.Size([1, 202500, 3])

    # ...

    results['poses'] = convert_poses(poses) 
    # [B, 6] 
    # torch.Size([1, 6])
    
    results['poses_matrix'] = poses 
    # [B, 4, 4] 
    # torch.Size([1, 4, 4])
```

__3.__ `Initialising rays based on pose data`

Let's have a look at `get_rays` function

```
i, j = custom_meshgrid(torch.linspace(0, W-1, W, device=device), torch.linspace(0, H-1, H, device=device)) 
# float 
# i = torch.Size([450, 450]) 
# j - torch.Size([450, 450])

i = i.t().reshape([1, H*W]).expand([B, H*W]) + 0.5 
# torch.Size([1, 202500])

j = j.t().reshape([1, H*W]).expand([B, H*W]) + 0.5 
# torch.Size([1, 202500])
```

pytorch's `mesh_grid` is used to generate x, y spatial coordinates position.
Here, we are generating spatial coordinates for H (450), W (450) size of an image
`0.5` float value is added to add coordinate position.
Therefore, max and min values will range between 0.5 to 449.50 for both H, W dimensions


```
results = {}

inds = torch.arange(H*W, device=device).expand([B, H*W]) 
# torch.Size([1, 202500]) # B = 1

results['i'] = i # torch.Size([1, 202500]) 
# tensor([[  0.5000,   1.5000,   2.5000,  ..., 447.5000, 448.5000, 449.5000]], device='cuda:0')

results['j'] = j # torch.Size([1, 202500]) 
# tensor([[  0.5000,   0.5000,   0.5000,  ..., 449.5000, 449.5000, 449.5000]], device='cuda:0')

results['inds'] = inds # inds.shape - torch.Size([1, 202500]) 
# tensor([[     0,      1,      2,  ..., 202497, 202498, 202499]], device='cuda:0')
```

`results` dictionary is initialized.
The coordinate positions in (H, W) 2d space which we generated previously using pytorch's meshgrid position is then reshaped into (H * W) vector and stored into results dictionary.
Similar we generate unique indices for each element in (H * W) vector, ranging from 0 to ((H * W) - 1) and store in under 'inds' key in results dictionary.

```
zs = torch.ones_like(i) 
# torch.Size([1, 202500])

xs = (i - cx) / fx * zs 
# torch.Size([1, 202500]) 
# cx 225.0 
# fx 1200.0

ys = (j - cy) / fy * zs 
# torch.Size([1, 202500]) 
# cy 225.0 
# fy 1200.0
```

The position of z-coordinate plane is fixed as 1.
Also, the 3d coordinates centered around `0` is computed from the (H * W) spatial coordinate generated previously using the
__(i)__ focal length and 
__(ii)__ 2D projection center positions
parameters of the camera (intrinsic parameters).

```
directions = torch.stack((xs, ys, zs), dim=-1) 
# directions.shape = torch.Size([1, 202500, 3])

directions = directions / torch.norm(directions, dim=-1, keepdim=True) 
# directions.shape = torch.Size([1, 202500, 3])

rays_d = directions @ poses[:, :3, :3].transpose(-1, -2) 
# rays_d.shape = (B, N, 3) 
# poses[:, :3, :3].shape = torch.Size([1, 3, 3]) 
# rays_d.shape = torch.Size([1, 202500, 3])
# poses[:, :3, :3].transpose(-1, -2).shape = torch.Size([1, 3, 3])

rays_o = poses[..., :3, 3] 
# [B, 3] 
# poses[..., :3, 3].shape = torch.Size([1, 3]) 
# poses.shape = torch.Size([1, 4, 4])

rays_o = rays_o[..., None, :].expand_as(rays_d) 
# [B, N, 3] 
# rays_d.shape = torch.Size([1, 202500, 3])
# rays_o[..., None, :].expand_as(rays_d).shape = torch.Size([1, 202500, 3])
# rays_o[..., None, :].shape = torch.Size([1, 1, 3])

results['rays_o'] = rays_o 
# torch.Size([1, 202500, 3])

results['rays_d'] = rays_d 
# torch.Size([1, 202500, 3])
```

The 3d coordinates centered around `0` x,y coordinates is stacked and then processed used to compute direction vector.
The processing involves dividing the above initialized stacked (3D) position tensor by its L2-norm (computed across x,y,z positions for each point).
Then, the normalized tensor is converted into rays direction tensor `rays_d` by matrix multiplying it with the pose transformation matrix.

The ray origin `rays_o` is computed from pose transformation matrix's translation vector values.

The computed ray-directions `rays_d` and ray-origins `rays_o` are stored in results dictionary


```
return results
```

Finally the results which stores all the computed ray data is returned.

__4.__ Background image computation.

```
bg_img = np.ones((self.H, self.W, 3), dtype=np.float32) 
# (450, 450, 3)

bg_img = self.bg_img.view(1, -1, 3).repeat(B, 1, 1).to(self.device) 
# bg_img.shape = torch.Size([1, 202500, 3]) 
# B = 1 
# self.bg_img.view(1, -1, 3).shape = torch.Size([1, 202500, 3])

results['bg_color'] = bg_img 
# torch.Size([1, 202500, 3])
```

Background image is simply computed to be `white`

