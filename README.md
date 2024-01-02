# RAD-NERF

----------------------
`WHAT DOES INSTRUCT-PIX2PIX STABLE DIFFUSION MODEL AIM TO ACCOMPLISH?`

In simple words, RAD-NERF aims to make it possible to generate photo-realistic 3D renderings in real-time by introducing novelties in encoding image and audio separately in a grid based manner (inspired mainly from instant-ngp).

----------------------
Note - The repository (unofficial pytorch implementation) uses a mixture of `cuda` along with `pytorch` to design components related to `ray-marching`, `ray-composition`, and `grid-encoding` to take advantage of higher degree of parallelism and also faster-execution of certain MLP networks made possible through the introduction of fully-fused cuda kernels (which was also introduced in instant ngp). Since my exposure to cuda programming is pretty limited, I tried to make sense of whatever I could understand by reading those cuda files. In case, you feel I missed out/ have some mistakes in my understanding of the cuda kernels involved, please feel free to point out in the commets section which will definitely useful to make edits (if required).

----------------------
## LETS LOOK AT IDEAS INVOLVED FROM CODE IMPLEMENTATION POV STEP-BY-STEP

### DATASET PREPARATION

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

### GENERATING PHOTO-REALISTIC RENDERINGS

__1.__ Iterate through each audio feature which we previously

```
def dataloader(self):
    # ...
    
    size = self.auds.shape[0]
    # size = 588
    # self.auds.shape = torch.Size([588, 44, 16])

    loader = DataLoader(list(range(size)), batch_size=1, collate_fn=self.collate, shuffle=False, num_workers=0) 
    # len(list(range(size))) = 588
    
    # ...

with torch.no_grad():
    for i, data in enumerate(loader):
        with torch.cuda.amp.autocast(enabled=self.fp16):
            preds, preds_depth = self.test_step(data) 
            # dict_keys(['auds', 'index', 'H', 'W', 'rays_o', 'rays_d', 'eye', 'bg_color', 'bg_coords', 'poses', 'poses_matrix'])
```

As you can see from the definition of dataloader, size of the dataset is taken as number of audio features (where each feature corresponds to 40ms time)
Then we iterate through the previously processed audio and corresponding ray information stored under `results` dict as we saw previously to generate photo-realistic renderings.

__2.__ Processings involved in each generation step

__(I)__ Computing nearest and farthest intersection points of each ray in the cube considered

```
nears, fars = raymarching.near_far_from_aabb(rays_o, rays_d, self.aabb_infer, self.min_near) 
# rays_o = torch.Size([202500])
# rays_d = torch.Size([202500]) 
# self.aabb_infer - tensor([-1.0000, -0.5000, -1.0000,  1.0000,  0.5000,  1.0000], device='cuda:0') : (xmin, ymin, zmin, xmax, ymax, zmax)
# self.min_near = 0.05

# ...

void near_far_from_aabb(const at::Tensor rays_o, const at::Tensor rays_d, const at::Tensor aabb, const uint32_t N, const float min_near, at::Tensor nears, at::Tensor fars) {
    static constexpr uint32_t N_THREAD = 128;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    rays_o.scalar_type(), "near_far_from_aabb", ([&] {
        kernel_near_far_from_aabb<<<div_round_up(N, N_THREAD), N_THREAD>>>(rays_o.data_ptr<scalar_t>(), rays_d.data_ptr<scalar_t>(), aabb.data_ptr<scalar_t>(), N, min_near, nears.data_ptr<scalar_t>(), fars.data_ptr<scalar_t>());
    }));
}

__global__ void kernel_near_far_from_aabb(
    const scalar_t * __restrict__ rays_o,
    const scalar_t * __restrict__ rays_d,
    const scalar_t * __restrict__ aabb,
    const uint32_t N,
    const float min_near,
    scalar_t * nears, scalar_t * fars
) 
{
    // parallel per ray
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= N) return;

    // locate
    rays_o += n * 3;
    rays_d += n * 3;

    const float ox = rays_o[0], oy = rays_o[1], oz = rays_o[2];
    const float dx = rays_d[0], dy = rays_d[1], dz = rays_d[2];
    const float rdx = 1 / dx, rdy = 1 / dy, rdz = 1 / dz;

    // get near far (assume cube scene)
    float near = (aabb[0] - ox) * rdx;
    float far = (aabb[3] - ox) * rdx;
    if (near > far) swapf(near, far);

    float near_y = (aabb[1] - oy) * rdy;
    float far_y = (aabb[4] - oy) * rdy;
    if (near_y > far_y) swapf(near_y, far_y);

    if (near > far_y || near_y > far) {
        nears[n] = fars[n] = std::numeric_limits<scalar_t>::max();
        return;
    }

    if (near_y > near) near = near_y;
    if (far_y < far) far = far_y;

    float near_z = (aabb[2] - oz) * rdz;
    float far_z = (aabb[5] - oz) * rdz;
    if (near_z > far_z) swapf(near_z, far_z);

    if (near > far_z || near_z > far) {
        nears[n] = fars[n] = std::numeric_limits<scalar_t>::max();
        return;
    }

    if (near_z > near) near = near_z;
    if (far_z < far) far = far_z;

    if (near < min_near) near = min_near;

    nears[n] = near;
    fars[n] = far;
}
```

This section of code computes the nearest and farthest intersection point of the rays
Assuming the x, y and z are part of a cube, the cube dimensions considered are based on the max and min values of x, y and z coordinates of the data points computed from pose transformation matrix which we saw previously.
As you can see the kernal computation of this `nears` and `fars` tensors computation is written directly in native cuda to take advantage of the thread parallelism possible with native cuda implementation.
From the code, we can see that we spawn 128 threads to compute the nears and fars tensor values of 128 rays parallely.

As you can see the logic seems to be really simple:

We know that our shape of interest is a regular cube.
Hence nearest point of intersection along one dimension cannot be greater than the farthest point of intersection along second or third dimension for the rays to be present in a cube.
The rays which does satisfy this criteria are considered to outside the cube and hence assigned a default value -> `std::numeric_limits<scalar_t>::max()` as the near and far intersection values for that particular ray with the cube of interest.
The initial nears and fars tensor values for the rays are computed along x-direction and subsequently rays are filtered out using the comparison with max and min points of intersection along y and z directions.

At the end, the computed `nears` and `fars` interesection values of each rays is returned.

__(II)__ Encode audio:

```
# encode audio
enc_a = self.encode_audio(auds) 
# enc_a.shape = [1, 64] # torch.Size([1, 64]) 
# auds - torch.Size([8, 44, 16]) 
# Embedding + MLP (Linear at end)

self.audio_net = AudioNet(self.audio_in_dim, self.audio_dim) # 44, 64 # CNN -> reduce spatial to 1, followed by FC layer

def encode_audio(self, a):
    # a: [1, 29, 16] or [8, 29, 16], audio features from deepspeech
    # if emb, a should be: [1, 16] or [8, 16]

    # fix audio traininig
    if a is None: 
        # a.shape = torch.Size([8, 44, 16])
        return None 

    if self.emb:
        a = self.embedding(a).transpose(-1, -2).contiguous() # [1/8, 29, 16]

    enc_a = self.audio_net(a) # [1/8, 64] 
    # enc_a.shape = torch.Size([8, 64])

    if self.att > 0:
        enc_a = self.audio_att_net(enc_a.unsqueeze(0)) 
        # enc_a.shape = torch.Size([1, 64]) 
        # CNN - stride 1, pad 1 -> reduce 8 channels to 1 channel while maintaining spatial dim
        
    return enc_a

# Audio feature extractor
class AudioNet(nn.Module):
    def __init__(self, dim_in=29, dim_aud=64, win_size=16):
        super(AudioNet, self).__init__()
        self.win_size = win_size 
        # 16
        
        self.dim_aud = dim_aud # dim_aud = 64 
        # dim_in = 44
        
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
        half_w = int(self.win_size/2) 
        # half_w = 8 
        # x.shape - torch.Size([8, 44, 16])
        
        x = x[:, :, 8-half_w:8+half_w] 
        # x.shape = torch.Size([8, 44, 16])
        
        x = self.encoder_conv(x).squeeze(-1) 
        # torch.Size([8, 64, 1]).squeeze(-1) = torch.Size([8, 64])
        
        x = self.encoder_fc1(x) 
        # torch.Size([8, 64])

        return x

self.audio_att_net = AudioAttNet(self.audio_dim) 
# 64

# Audio feature extractor
class AudioAttNet(nn.Module):
    def __init__(self, dim_aud=64, seq_len=8):
        super(AudioAttNet, self).__init__()
        self.seq_len = seq_len 
        # 8

        self.dim_aud = dim_aud 
        # 64

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
            nn.Linear(in_features=self.seq_len, out_features=self.seq_len, bias=True), 
            # self.seq_len = 8
            
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # x: [1, seq_len, dim_aud]
        x.shape = torch.Size([1, 8, 64])
        
        y = x.permute(0, 2, 1)  
        # y.shape = [1, dim_aud, seq_len] 
        # y.shape = torch.Size([1, 64, 8])

        y = self.attentionConvNet(y)  
        # y.shape = torch.Size([1, 1, 8])
        
        y = self.attentionNet(y.view(1, self.seq_len)).view(1, self.seq_len, 1) 
        # y.shape = torch.Size([1, 8, 1])
        # y.view(1, self.seq_len).shape = torch.Size([1, 8])
        # self.attentionNet(y.view(1, self.seq_len)).shape = torch.Size([1, 8])

        return torch.sum(y * x, dim=1) # [1, dim_aud] # torch.Size([1, 64])
```

The input audio feature is encoded via an `AudioNet` to produce an embedding of shape -> `[8, 64]` from input audio features of shape -> `[8, 44, 16]`

This is achieved by a series of 1D-CNN layers with ReLU activations which reduces `[8, 44, 16]` into `[8, 64, 1]` which is then squeezed at last output dimension to generate embedding of shape -> `[8, 64]`. 
This is then fed to Linear + ReLU + Linear layers to get a `[8, 64]` embedding.

This is then unsqueezed at 0th dimension to get an embedding of shape -> `[1, 8, 64]`. This is fed as input to the `AudioAttNet`
Here, this embedding is permuted into `[1, 64, 8]` shape.
This is again passed through a series of 1D-CNN + ReLU activations to get an embedding of shape - `[1, 8, 1]` 
We could see from the previous two points that by changing the batching dimension by unsqueezing at the 0th dimension we are basically exposing the embedding computed across each audio feature (corresponding to 40ms time) to be subjected to pass through a neural network to generate a embedding which considers the features across all audios samples (8) in the attention window to compute the attention context.
This context vector of shape -> `[1, 8, 1]` is then multiplied with input embedding of shape `[1, 8, 64]` and summed across `1-st dimension` of the resultant  to get the attention embedding of shape -> `[1, 64]` which is returned.

Thus audio encoding returned from `audio_encode` method is of shape -> `[1, 64]`.

__(III)__ `Ray-Marching`

```
max_steps = 16
```
Maximum number of steps taken along each ray is set to 16

For each step
```
n_alive = rays_alive.shape[0] 
# 202500, 63206
                
# exit loop
if n_alive <= 0:
    break

# decide compact_steps
n_step = max(min(N // n_alive, 8), 1) 
# 1
```
In case, the numnber of active rays reaches 0 before max steps, the `ray_marching` is terminated.

```
xyzs, dirs, deltas = raymarching.march_rays(n_alive, n_step, rays_alive, rays_t, rays_o, rays_d, self.bound, self.density_bitfield, self.cascade, self.grid_size, nears, fars, 128, perturb if step == 0 else False, dt_gamma, max_steps) 
# xyzs.shape = torch.Size([202624, 3])
# dirs.shape = torch.Size([202624, 3])
# deltas.shape = torch.Size([202624, 2])
# self.bound = 1
# self.density_bitfield.shape = torch.Size([262144])
# self.cascade - 1
# self.grid_size - 128
# {nears, fars}.shape - torch.Size([202500])
# perturb = False
# dt_gamma - 0.00390625
# max_steps - 16

def march_rays(...):

    # ...
    
    M = n_alive * n_step # 202500, n_step = 1 # 189618, n_step = 2, n_alive = 63206 # iter 5 - M: 201216, n_alive = 40232

    # ...

    xyzs = torch.zeros(M, 3, dtype=rays_o.dtype, device=rays_o.device) # torch.Size([202624, 3])
    dirs = torch.zeros(M, 3, dtype=rays_o.dtype, device=rays_o.device) # torch.Size([202624, 3])
    deltas = torch.zeros(M, 2, dtype=rays_o.dtype, device=rays_o.device) # torch.Size([202624, 2]) # 2 vals, one for rgb, one for depth # torch.Size([202624, 2])

    if perturb: # False
        # torch.manual_seed(perturb) # test_gui uses spp index as seed
        noises = torch.rand(n_alive, dtype=rays_o.dtype, device=rays_o.device)
    else:
        noises = torch.zeros(n_alive, dtype=rays_o.dtype, device=rays_o.device) # torch.Size([202500]); iter 2: n_alive = 63206, noises.shape = torch.Size([63206])

    _backend.march_rays(n_alive, n_step, rays_alive, rays_t, rays_o, rays_d, bound, dt_gamma, max_steps, C, H, density_bitfield, near, far, xyzs, dirs, deltas, noises)

    # ....

void march_rays_train(const at::Tensor rays_o, const at::Tensor rays_d, const at::Tensor grid, const float bound, const float dt_gamma, const uint32_t max_steps, const uint32_t N, const uint32_t C, const uint32_t H, const uint32_t M, const at::Tensor nears, const at::Tensor fars, at::Tensor xyzs, at::Tensor dirs, at::Tensor deltas, at::Tensor rays, at::Tensor counter, at::Tensor noises) {

    static constexpr uint32_t N_THREAD = 128;
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    rays_o.scalar_type(), "march_rays_train", ([&] {
        kernel_march_rays_train<<<div_round_up(N, N_THREAD), N_THREAD>>>(rays_o.data_ptr<scalar_t>(), rays_d.data_ptr<scalar_t>(), grid.data_ptr<uint8_t>(), bound, dt_gamma, max_steps, N, C, H, M, nears.data_ptr<scalar_t>(), fars.data_ptr<scalar_t>(), xyzs.data_ptr<scalar_t>(), dirs.data_ptr<scalar_t>(), deltas.data_ptr<scalar_t>(), rays.data_ptr<int>(), counter.data_ptr<int>(), noises.data_ptr<scalar_t>());
    }));
}
```
Here, we are going to take a deeper look at `ray_marching`.
We initialize the xyzs, dirs and deltas tensors to zeros.
Lets first see what these xyzs, dirs and deltas indicate

`xyzs` - holds the (x,y,z) coordinates of each sampled point along the rays when we perform `ray_marching`.
Since there are `n_alive` rays in contention (after performing ray_marching where we took `n-1` steps previously) and since we are going to take `n_steps` in the current iteration of forward ray_marching, we would need:
`n_alive * n_steps` 3-dim tensors to hold the coordinates of each sampled point along the rays

`dirs` - holds the 3D direction vector of each sampled point along the rays when we do `ray_marching`
For many ray marching scenarios, especially in simpler models or when dealing with straight rays from the camera, the direction of each ray might not change as it travels through the scene. However, in more complex scenarios involving effects like refraction or in more advanced models, the direction might change at different points along the ray.
Again, as we say previously with `xyzs`, we would need `n_alive * n_steps` 3-dim tensors to store the `dirs`

`deltas` - holds the step_size taken for the current step and the accumulated distance along the rays when we do `ray_marching`
Since, we would be storing these two scalar information for each sampled point along the rays, and there are `n_alive * n_steps` total sampled points, we will require `n_alive * n_steps` 2-dim tensors to store the `deltas`

As you can see, the ray-marching core logic is written in native cuda.
We will be spawning 128 parallel threads to perform ray_marching for 128 rays simulataneously.

Steps involved in ray-marching can be explained diagrammatically as shown below:

![alt text](https://github.com/Karthik-Ragunath/RAD-NeRF/blob/master/assets/ray_marching_screenshot.png?raw=true)

Each circle described in the diagram represents one step taken during ray-marching.

__(IV)__ `Computing RGBs and densities`

```
sigmas, rgbs, ambient = self(xyzs, dirs, enc_a, ind_code, eye) 
# xyzs.shape = torch.Size([202624, 3])
# dirs.shape = torch.Size([202624, 3])
# enc_a.shape = torch.Size([1, 64])
# ind_code.shape = torch.Size([4])
# eye.shape = torch.Size([1, 1])
# sigmas.shape = torch.Size([202624])
# rgbs.shape = torch.Size([202624, 3])
# ambient.shape = torch.Size([202624, 2])
# ind_code = tensor([-0.0718, -0.1488, -0.0244, -0.0537], device='cuda:0', requires_grad=True)
# eye = tensor([0.25])
```
The `xyzs` and `dirs` computed previously from `ray-marching`; 
`enc_a` computed from encoding audio features; 
`ind_code` is identification tensor associated with the particular avatar being trained; 
`eye` represents the approximate area covered by the eyes in the image.

These are fed as input to the NeRF neural network to compute `color`, `density` and `ambient` values.

```
def forward(self, x, d, enc_a, c, e=None):
    # ...

    enc_a = enc_a.repeat(x.shape[0], 1) 
    # enc_a.shape = torch.Size([202624, 64]) 
    # x.shape[0] = 202624
```

The audio encoding `enc_a` tensor is repeated along 0th dimension for the count of number of rays.

```
    enc_x = self.encoder(x, bound=self.bound)
    # enc_x.shape = torch.Size([202624, 32]) 
    # self.bound = 1
```
The `x` tensor which contains information on the points sampled along the rays is then encoded by a grid encoder.

This is one of the fundamental novelties introduced in the `instant-ngp` which is also used in this paper.
The native cuda code involved in generating this encoding is also from the `instant-ngp` repository.

Let's look at the implementation of this `grid-encoder` in detail:
```
self.encoder, self.in_dim = get_encoder(
    'tiledgrid', 
    input_dim=3, 
    num_levels=16, 
    level_dim=2, 
    base_resolution=16, 
    log2_hashmap_size=16, 
    desired_resolution=2048 * self.bound, 
    interpolation='linear',
    align_corners=False
) 
# self.bound = 1 # GridEncoder, 32

def get_encoder(...):
    # ...
    from gridencoder import GridEncoder
    encoder = GridEncoder(...)
    # ...
    return encoder, encoder.output_dim

def forward(self, inputs, bound=1):
    # allocate parameters
    offsets = []
    offset = 0
    self.max_params = 2 ** log2_hashmap_size # 2 ** 16 = 65536 (both)
    for i in range(num_levels): # num_levels = 16 
        resolution = int(np.ceil(base_resolution * per_level_scale ** i)) # 16, 23, 31, 43, 59, 81, 112, 154, 213, ... # 32 * 32 * 32 = 32768, # 44 * 44 * 44 = 85184
        params_in_level = min(self.max_params, (resolution if align_corners else resolution + 1) ** input_dim) # limit max number
        params_in_level = int(np.ceil(params_in_level / 8) * 8) # make divisible
        offsets.append(offset)
        offset += params_in_level
    offsets.append(offset) # [0,4920,18744,51512,117048,182584,248120,313656,379192,444728,510264,575800,641336,706872,772408,837944,903480]
    offsets = torch.from_numpy(np.array(offsets, dtype=np.int32)) 
    self.register_buffer('offsets', offsets)
    
    self.n_params = offsets[-1] * level_dim # 903480 * 2 = tensor(1806960, dtype=torch.int32)

    # parameters
    self.embeddings = nn.Parameter(torch.empty(offset, level_dim)) # (903480, 2) - torch.Size([903480, 2]) # Torso = torch.Size([555520, 2])

    inputs = (inputs + bound) / (2 * bound) 
    # map to [0, 1] 
    # inputs.shape = torch.Size([202624, 2])
    # bound = 1
    
    prefix_shape = list(inputs.shape[:-1]) 
    # inputs.shape[:-1] = torch.Size([202624])
    # len(prefix_shape) = 1
    
    inputs = inputs.view(-1, self.input_dim) 
    # inputs.shape = torch.Size([202624, 2])

    outputs = grid_encode_wrapper(inputs, self.embeddings, self.offsets, self.per_level_scale, self.base_resolution, inputs.requires_grad, self.gridtype_id, self.align_corners, self.interp_id) 
    # outputs.shape = torch.Size([202624, 32])
    
    outputs = outputs.view(prefix_shape + [self.output_dim]) 
    # outputs.shape = torch.Size([202624, 32])

    return outputs

def grid_encode_wrapper(inputs, embeddings, offsets, per_level_scale, base_resolution, calc_grad_inputs=False, gridtype=0, align_corners=False, interpolation=0):
        inputs = inputs.contiguous()
        # torch.Size([202624, 2]) - for audio, 
        # torch.Size([202624, 3]) - for sampled points in rays

        B, D = inputs.shape # batch size, coord dim 
        # (B, D) for audio = (202624, 2) 
        # (B, D) for sampled points on rays = (202624, 3)

        L = offsets.shape[0] - 1 
        # level # L = 16

        C = embeddings.shape[1] 
        # embedding dim for each level # C = 2

        S = np.log2(per_level_scale) 
        # S = 0.4666666666666666, resolution multiplier at each level, apply log2 for later CUDA exp2f 
        # per_level_scale = 1.381912879967776

        H = base_resolution 
        # H = 16

        outputs = torch.empty(L, B, C, device=inputs.device, dtype=embeddings.dtype) 
        # outputs.shape = torch.Size([16, 202624, 2])

        if calc_grad_inputs:
            dy_dx = torch.empty(B, L * D * C, device=inputs.device, dtype=embeddings.dtype)
        else:
            dy_dx = None

        _backend.grid_encode_forward(inputs, embeddings, offsets, outputs, B, D, C, L, S, H, dy_dx, gridtype, align_corners, interpolation)
        # outputs.shape = torch.Size([16, 202624, 2])
        # permute back to [B, L * C]
        
        outputs = outputs.permute(1, 0, 2).reshape(B, L * C) 
        # outputs.shape = torch.Size([202624, 32])

        return outputs

```
Th `get_encoder` function is called which inturn calls `forward` implementation of the `GridEncoder` class.
Before we explore the `forward` method of the `grid_encoder` its important to get a high level understanding of the concepts involved in designing this encoder.
The `grid_encoder` is implemented based on the idea of `multi-resolution hash encoding`.
In terms of implementation, this means, for different resolutution multiple grids are initialized where `features` corresponding to the `vertices` of each grid is stored in a hash table.
Hence, if we are considering a resolution of `L` levels, there will `L` hash tables holding features associated with each vertex in corresponding grids.
Here, the hash table size is a `hyper-parameter`

For initial coarser levels, number of vertices present in the grid which encodes that level is less, hence there will be a 1:1 correspondence between vertices and entries in hash table.

At finer resolution levels, there can be hash collisions in case number of vertices is greater than the size of the hash table which represents that level.

According to `instant-ngp` paper, it is not needed to resolve these hash collisions manually with concepts like chaining, etc.
Instead, the gradient optimization during backpropagation acts something like a `soft-collision resolution algorithm` in the sense that gradient updates will modify the features present in the hash table in such a way that overall loss gets reduced. 
For example, in case larger error correction needed to be performed on a vertex and a smaller correction is involved with a different vertex and both the vertices points to the same entry in the hash table (hash collision), the overall updates made during the backpropagation for the feature stored in that hash entry with collision is mean of the updates due to the error from these two different vertices. Thus mean value will be dominated by the gradient update due to vertex with larger error when compared to smaller error and hence correction from gradient optimization will also follow the same pattern.

To delve a bit more deeper, the `native-cuda` kernal functions which encodes the incoming input into a postion inside the grid.
Based on the calculated position, feature-encoding associated that particular position is calculated by interpolation (trilinear for positions from rays or bilinear for audio) of features associated with 2^(input_dim) vertices which encloses that particular position in the grid.
Thus, the features computed from grid of different resolutions for a particular input is concatenated to get the `grid-encoding`

Coming back to the implementation aspect, we could see from the `forward` method that the number of parameters associated with each resolution is computed in `offsets` list.
Therefore, the scalar `offsets` list denote the size of the hash-table associated with each resolution. (These hash-tables store the features associated with each vertex in the grid). In turn we call the `grid_encode_wrapper` method which acts as a wrapper-interface which links the pytorch code with the native cuda code.

The `grid_encode_wrapper` function calls the `grid_encode_forward` method in the native cuda executable loaded.
The native cuda code performs the multi-resolution grid encoding exactly as we discussed in the previous paragraphs and returns the grid-encoding of the positions sampled along the rays in the `outputs` tensor.

The `outputs` tensor is of the form `[num_rays_alive * num_steps, grid_encoding_feature_size]`
In our case, number of levels is 16 and feature size associated with each vertex in grid of all resolution is 2.
Since, we concatenate the computed feature encoding from grids of all resolution for each sample along the rays, the concatenated `encoding-feature` size associated with each sample is `16 * 2 = 32`
This, output is of shape -> `[num_rays_alive * num_steps, 32]`

```
# ...

# ambient
ambient = torch.cat([enc_x, enc_a], dim=1) 
# ambient.shape = torch.Size([202624, 96])

# ...
```
Next, `audio-encoding` and `ray-poisitons grid-encoding` are concatenated to get the concatenated feature of size `[num_rays_alive * num_steps, audio_feature_len + grid_encoding_feature_len]` which is basically `[num_rays_alive * num_steps, 64 + 32]` -> `[num_rays_alive * num_steps, 96]`.

The concatenated encoding is then passed through `self.ambient_net` 
```
ambient = self.ambient_net(ambient).float() 
# ambient.shape = torch.Size([202624, 2])

self.ambient_net = MLP(self.in_dim + self.audio_dim, self.ambient_dim, self.hidden_dim_ambient, self.num_layers_ambient) 
# self.in_dim = 32
# self.audio_dim = 64
# self.ambient_dim = 2
# self.hidden_dim_ambient = 64
# self.num_layers_ambient = 3

class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers):
        super().__init__()
        self.dim_in = dim_in 
        # self.dim_in = 96
        
        self.dim_out = dim_out 
        # self.dim_out = 2
        
        self.dim_hidden = dim_hidden 
        # self.dim_hidden = 64
        
        self.num_layers = num_layers 
        # self.num_layers = 3

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
```
As we can see from the architecture of the `MLP`, it is just a bunch of `nn.Linear` layers. The output feature is of size `self.ambient_dim`
Thus, `ambient` encoding is of shape -> `[num_rays_alive * num_steps, 2]`

```
# ...

ambient = torch.tanh(ambient) 
# map to [-1, 1] 
# ambient.shape = torch.Size([202624, 2])

# sigma
enc_w = self.encoder_ambient(ambient, bound=1) 
# enc_w.shape = torch.Size([202624, 32])

# ...
```
The computed `ambient` encoding is then passed through the non-linear `tanh` activation to map the `ambient` encoding in the `[-1, 1]` range.

The `ambient` encoding is then encoded via a `grid_encoder` (inspired from `instant-ngp` paper).
We already saw the details involved in encoding with `grid-encoder` in detail in the previous paragraph. Same process is followed here.
Thus the sigma-encoding `enc_w` is of shape -> `[num_rays_alive * num_steps, L * vertex_feature_size]` -> `[num_rays_alive * num_steps, 16 * 2]` (Same as we saw in previous paragraphs).

```
h = torch.cat([enc_x, enc_w, e.repeat(x.shape[0], 1)], dim=-1) 
# h.shape = torch.Size([202624, 65])
```
sigma encoding feature vectors are computed by concatenating the spatial encoding -> `enc_x` (encoding of points sample along the rays), ambient encoding -> `enc_w` and eye features -> `e` to get concatenated feature size of `[num_rays_alive * num_steps, 65]`

```
h = self.sigma_net(h) 
# torch.Size([202624, 65]) 
# Linear + ReLu

self.sigma_net = MLP(self.in_dim + self.in_dim_ambient + self.eye_dim, 1 + self.geo_feat_dim, self.hidden_dim, self.num_layers) # 65 (32 + 32 + 1), 65, 64, 3

class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers):
        super().__init__()
        self.dim_in = dim_in 
        # 65
        
        self.dim_out = dim_out 
        # 65
        
        self.dim_hidden = dim_hidden 
        # 64
        
        self.num_layers = num_layers 
        # 3

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
```
The sigma encoding is then fed as input to the `sigma_net`.
The `sigma_net` uses the same `MLP` architecture as the ambient net which we discussed in the previous paragraphs.

```
sigma = trunc_exp(h[..., 0]) 
# sigma.shape = torch.Size([202624])

geo_feat = h[..., 1:] 
# geo_feat.shape = torch.Size([202624, 64])
```
The output from `sigma_net` is split into two parts.
sigma[:, 0] represents the `sigma` (or) `density` tensor.
sigma[:, 1:] represents the geometric feature encodings for the give pose+audio data inputs.

```
# color
enc_d = self.encoder_dir(d) 
# enc_d.shape = torch.Size([202624, 16])

self.encoder_dir, self.in_dim_dir = get_encoder('spherical_harmonics') 
# SHEncoder: input_dim=3 degree=4, 16

from shencoder import SHEncoder
encoder = SHEncoder(input_dim=input_dim, degree=degree)

def forward(self, inputs, size=1):
    # inputs: [..., input_dim], normalized real world positions in [-size, size]
    # return: [..., degree^2]

    inputs = inputs / size # [-1, 1]

    prefix_shape = list(inputs.shape[:-1])
    inputs = inputs.reshape(-1, self.input_dim)

    outputs = spherical_harmonics_encode_forward(inputs, self.degree, inputs.requires_grad)
    outputs = outputs.reshape(prefix_shape + [self.output_dim])

    return outputs

def spherical_harmonics_encode_forward(ctx, inputs, degree, calc_grad_inputs=False):
    # inputs: [B, input_dim], float in [-1, 1]
    # RETURN: [B, F], float

    inputs = inputs.contiguous()
    B, input_dim = inputs.shape # batch size, coord dim
    output_dim = degree ** 2
    
    outputs = torch.empty(B, output_dim, dtype=inputs.dtype, device=inputs.device)

    if calc_grad_inputs:
        dy_dx = torch.empty(B, input_dim * output_dim, dtype=inputs.dtype, device=inputs.device)
    else:
        dy_dx = None

    _backend.sh_encode_forward(inputs, outputs, B, input_dim, degree, dy_dx)

    ctx.save_for_backward(inputs, dy_dx)
    ctx.dims = [B, input_dim, degree]

    return outputs

__global__ void kernel_sh(
    const scalar_t * __restrict__ inputs, 
    scalar_t * outputs, 
    uint32_t B, uint32_t D, uint32_t C,
    scalar_t * dy_dx
) {
    # ...
	scalar_t x = inputs[0], y = inputs[1], z = inputs[2];
	scalar_t xy=x*y, xz=x*z, yz=y*z, x2=x*x, y2=y*y, z2=z*z, xyz=xy*z;
	scalar_t x4=x2*x2, y4=y2*y2, z4=z2*z2;
	scalar_t x6=x4*x2, y6=y4*y2, z6=z4*z2;

	auto write_sh = [&]() {
		outputs[0] = 0.28209479177387814f ;                          // 1/(2*sqrt(pi))
		if (C <= 1) { return; }
		outputs[1] = -0.48860251190291987f*y ;                               // -sqrt(3)*y/(2*sqrt(pi))
		outputs[2] = 0.48860251190291987f*z ;                                // sqrt(3)*z/(2*sqrt(pi))
        ...
		outputs[15] = 0.59004358992664352f*x*(-x2 + 3.0f*y2) ;                                // sqrt(70)*x*(-x2 + 3*y2)/(8*sqrt(pi))

	};

	write_sh();

	if (dy_dx) {
		scalar_t *dx = dy_dx + b * D * C2;
		scalar_t *dy = dx + C2;
		scalar_t *dz = dy + C2;

		auto write_sh_dx = [&]() {
			dx[0] = 0.0f ;                             // 0
			dx[1] = 0.0f ;                             // 0
			dx[2] = 0.0f ;                             // 0
            ...
			dx[15] = -1.7701307697799304f*x2 + 1.7701307697799304f*y2 ;                               // 3*sqrt(70)*(-x2 + y2)/(8*sqrt(pi))

		};

		auto write_sh_dy = [&]() {
			dy[0] = 0.0f ;                             // 0
			dy[1] = -0.48860251190291992f ;                          // -sqrt(3)/(2*sqrt(pi))
			dy[2] = 0.0f ;                             // 0
            ...
			dy[15] = 3.5402615395598609f*xy ;                                // 3*sqrt(70)*xy/(4*sqrt(pi))

		};

		auto write_sh_dz = [&]() {
			dz[0] = 0.0f ;                             // 0
			dz[1] = 0.0f ;                             // 0
			dz[2] = 0.48860251190291992f ;                           // sqrt(3)/(2*sqrt(pi))
            ...
			dz[15] = 0.0f ;                            // 0

		};
		write_sh_dx();
		write_sh_dy();
		write_sh_dz();
	}
}
```
The geometric feature encodings are fed as input to the `encoder_dir`.
`encoder_dir` is a `spherical-harmonics` encoder.

To give an idea on what is `spherical harmonics`:
In 3D graphics and modeling, particularly in the context of rendering and texture mapping, SH coefficients typically refer to "Spherical Harmonics" coefficients. Spherical Harmonics are a series of orthogonal functions defined on the surface of a sphere and are used in various fields, including physics, mathematics, and computer graphics. They are particularly useful for efficiently representing complex functions or data defined over a sphere, such as lighting environments or the distribution of light and color on a 3D surface.

Now coming back to the code,
`forward` method of SHEncoder class gets called which inturn calls `spherical_harmonics_encode_forward` method which acts as wrapper-interface between pytorch code and native cuda kernel functions which computes the SH-encodings and its gradients.

The computation involved with calculating SH-encodings is pretty straight-forward.
The geometric feature encodings (tensor of size `64` for each point sampled along the rays) are subjected to scalar manipulations with pretty-defined `SH-coefficients`.
Since the SH-coefficients are fixed, gradients can also be computed based on the scalar manipulation formula involving `SH-coefficients`.
The resultant `enc_d` (sh-encoding) is a tensor of shape -> [num_rays_alive * num_steps, 16].

```
h = torch.cat([enc_d, geo_feat, c.repeat(x.shape[0], 1)], dim=-1) 
# h.shape = torch.Size([202624, 84])
```
The concatenation of `end_d` SH-encoding, `geo_feature` (geometrical feature encodings), `c` (feature encoding associated with a particular avatar or latent appearence embedding) results in `h` (color-encoding) which is fed as input to compute rgb values associated with each points sampled along the rays.
`h` (color-encoding) is of shape - `[num_rays_alive * num_steps, 84]`

```
h = self.color_net(h) 
# h.shape = torch.Size([202624, 3])

self.color_net = MLP(self.in_dim_dir + self.geo_feat_dim + self.individual_dim, 3, self.hidden_dim_color, self.num_layers_color)
# self.in_dim_dir = 16
# self.geo_feat_dim = 64
# self.individual_dim = 4
# output_dim = 3
# self.hidden_dim_color = 64
# self.num_layers_color = 2
```
The color-encoding `h` is fed as input to the `color_net` which is again same as the `MLP` architecture which we discussed previously.
Thus, we know that output of `color-net` is a tensor of shape -> `[num_rays_alive * num_steps, 3]`

```
# sigmoid activation for rgb
color = torch.sigmoid(h) 
# color.shape = torch.Size([202624, 3])

return sigma, color, ambient 
# ambient - torch.Size([202624, 2])
```
The color-features computed via `color-net` (`MLP`) is then passed through `sigmoid` non-linear activation to get (rgb) color values in `[0, 1]` range.
The computed `color`, `sigma` (density) and `ambient` tensors for each point sampled along the rays are returned.

__(V)__ `Calculating ray-composition`

```
raymarching.composite_rays(n_alive, n_step, rays_alive, rays_t, sigmas, rgbs, deltas, weights_sum, depth, image, T_thresh) 
# T_thresh = 0.0001

void composite_rays(const uint32_t n_alive, const uint32_t n_step, const float T_thresh, at::Tensor rays_alive, at::Tensor rays_t, at::Tensor sigmas, at::Tensor rgbs, at::Tensor deltas, at::Tensor weights, at::Tensor depth, at::Tensor image) {
    static constexpr uint32_t N_THREAD = 128;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    image.scalar_type(), "composite_rays", ([&] {
        kernel_composite_rays<<<div_round_up(n_alive, N_THREAD), N_THREAD>>>(n_alive, n_step, T_thresh, rays_alive.data_ptr<int>(), rays_t.data_ptr<scalar_t>(), sigmas.data_ptr<scalar_t>(), rgbs.data_ptr<scalar_t>(), deltas.data_ptr<scalar_t>(), weights.data_ptr<scalar_t>(), depth.data_ptr<scalar_t>(), image.data_ptr<scalar_t>());
    }));
}

template <typename scalar_t>
__global__ void kernel_composite_rays(
    const uint32_t n_alive, 
    const uint32_t n_step, 
    const float T_thresh,
    int* rays_alive, 
    scalar_t* rays_t, 
    const scalar_t* __restrict__ sigmas, 
    const scalar_t* __restrict__ rgbs, 
    const scalar_t* __restrict__ deltas, 
    scalar_t* weights_sum, scalar_t* depth, scalar_t* image
) {
    const uint32_t n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n >= n_alive) return;

    const int index = rays_alive[n]; // ray id
    
    // locate 
    sigmas += n * n_step;
    rgbs += n * n_step * 3;
    deltas += n * n_step * 2;
    
    rays_t += index;
    weights_sum += index;
    depth += index;
    image += index * 3;

    scalar_t t = rays_t[0]; // current ray's t
    
    scalar_t weight_sum = weights_sum[0];
    scalar_t d = depth[0];
    scalar_t r = image[0];
    scalar_t g = image[1];
    scalar_t b = image[2];

    // accumulate 
    uint32_t step = 0;
    while (step < n_step) {
        
        // ray is terminated if delta == 0
        if (deltas[0] == 0) break;
        
        const scalar_t alpha = 1.0f - __expf(- sigmas[0] * deltas[0]);

        /* 
        T_0 = 1; T_i = \prod_{j=0}^{i-1} (1 - alpha_j)
        w_i = alpha_i * T_i
        --> 
        T_i = 1 - \sum_{j=0}^{i-1} w_j
        */
        const scalar_t T = 1 - weight_sum;
        const scalar_t weight = alpha * T;
        weight_sum += weight;

        t = deltas[1];
        d += weight * t;
        r += weight * rgbs[0];
        g += weight * rgbs[1];
        b += weight * rgbs[2];

        //printf("[n=%d] num_steps=%d, alpha=%f, w=%f, T=%f, sum_dt=%f, d=%f\n", n, step, alpha, weight, T, sum_delta, d);

        // ray is terminated if T is too small
        // use a larger bound to further accelerate inference
        if (T < T_thresh) break;

        // locate
        sigmas++;
        rgbs += 3;
        deltas += 2;
        step++;
    }

    //printf("[n=%d] rgb=(%f, %f, %f), d=%f\n", n, r, g, b, d);

    // rays_alive = -1 means ray is terminated early.
    if (step < n_step) {
        rays_alive[n] = -1;
    } else {
        rays_t[0] = t;
    }

    weights_sum[0] = weight_sum; // this is the thing I needed!
    depth[0] = d;
    image[0] = r;
    image[1] = g;
    image[2] = b;
}
```
The computed `rgbs`, `sigmas` values of points sampled along the rays from `NeRF` network are fed as input to the `composite_rays` kernel function implemented in native-cuda.
To give an idea of what is `ray-composition`, the `ray-composition` is the process of taking `rgbs` (color) and `sigmas` (density) of the points sampled along the rays and compose them by using some `emission-absorption` based transmittance algorithm such as `Beer-Lambert Law` (used in this case) to assign a single color (rgb) value and and `depth` value (since we are dealing with 3d-space). This color (rgb) value and depth value of the rays are used to render the 2d image based on input camera-pose matrix (and audio too in our case).

Also, to give am idea of what `emission-absorption` based transmittance algorithm mean:
As the ray marches through the volume (in discrete steps), it encounters points with specific emission, absorption, and scattering properties.
__(i)__ Emission: At each step, the ray accumulates light based on the emission properties of the material at that point.
__(ii)__ Absorption: Simultaneously, the ray's current light is attenuated based on the absorption properties of the material. This attenuated light is what will continue to the next point in the volume.
__(iii)__ Transmittance: The algorithm calculates the transmittance of the ray from its starting point to its current point, which is used to determine how much of the initial light has survived.
At each step, the color and intensity of the light are updated based on the emission and absorption encountered. The accumulated color and intensity at the end of the ray's path represent the pixel's color and brightness in the final image.

Now coming back to code implementation part of `ray-composition`,
we can see that `native-cuda` implementation launches 128 parallel threads, where each thread deals with composing the `rgb` (color) and `sigma` (density) value for that ray.
In our above code, the above mentioned `Beer-Lambert Law` is applied as:
```
const scalar_t alpha = 1.0f - __expf(- sigmas[0] * deltas[0]); 
# computing absorption in the current step

const scalar_t T = 1 - weight_sum;
# fraction of light available after absorption in previous steps

const scalar_t weight = alpha * T;
# computing transmittance from this step (by taking into account the fraction of light absorbed in previous steps)

weight_sum += weight;
# Keeping log of total light absorbed until the current step to compute the transmittance in the next step
```

Taking a deeper look at the code, we could see that we do not compute all the `n-steps` for every ray. We could see that there are two early stopping criterias:

__(i)__ Transmittance Threshold (T_thresh): If the accumulated transmittance (T) of the ray drops below a certain threshold (T_thresh), it implies that the ray is unlikely to contribute further to the image due to being mostly absorbed or scattered. Continuing to march it would waste computational resources.
```
// ray is terminated if delta == 0
if (deltas[0] == 0) break;
```

__(ii)__ Zero Delta: If the delta (change in position along the ray) is zero, it might indicate that the ray has hit an opaque surface or otherwise will not contribute further to the image.
```
// ray is terminated if T is too small
// use a larger bound to further accelerate inference
if (T < T_thresh) break;
```

In both these above conditions, the early stoppage is performed and next step is not taken.

For the rays which were stopped due to early stoppage condition, we set the flag which indicates whether a particular ray (index) is alive to `False` condition.
```
rays_alive[n] = -1
```

```
rays_alive = rays_alive[rays_alive >= 0]
# rays_alive.shape = torch.Size([63206])
```
Now, the rays which were `stopped-early` are removed from consideration for future computations since we have already assigned the `rgb` (color) and `sigma` (density) values for those rays

Now with the ray which are still alive, steps (III) `ray-marching`, (IV) `Computing RGBs and densities` and (V) `Calculating ray-composition` are repeated until either are no more rays alive (or) we have reached `max_steps` (max_resolution) which is `16` in our case.

__(VI)__ `Computing 2D-image to be rendered and also depth involved with each pixel`

From all the computations we performed in previous three steps explained above:
(III) `ray-marching`, (IV) `Computing RGBs and densities` and (V) `Calculating ray-composition`,
We are going to use the `transmittance` associated with each ray which is computed in `ray-composition` step and also the `nears` (nearest intersection point of rays with the grid) and `fars` (farthest intersection point of rays with the grid) to calculate the rendering 2D-pixel `RGB` value and the `depth` associated with each pixel (in the given 3D scene).
```
image = image + (1 - weights_sum).unsqueeze(-1) * bg_color 
# image.shape = torch.Size([1, 202500, 3]) 
# weights_sum.shape = torch.Size([202500])

image = image.view(*prefix, 3) 
# image.shape = torch.Size([1, 202500, 3])

image = image.clamp(0, 1) 
# image.shape = torch.Size([1, 202500, 3])

depth = torch.clamp(depth - nears, min=0) / (fars - nears) 
# depth.shape = torch.Size([202500]) 
# fars, nears, depth - torch.Size([202500])

depth = depth.view(*prefix) 
# depth.shape = torch.Size([1, 202500])
# prefix.shape = torch.Size([1, 202500])

results['depth'] = depth
results['image'] = image 
```
As we can see, the 2D-rendering RGB `image` is computed by blending the `transmittance` associated with each ray which is computed in `ray-composition` step and the background-color `bg_color` (tensor of 1s -> `white`).

__3.__ The above photo-realistic 2D-render generation step is repeated to get `N` frames which are then binded together to create the talking head avatar video driven by incoming audio features.

### LOSSES INVOLVED IN TRAINING


