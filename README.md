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

__(i)__ Computing nearest and farthest intersection points of each ray in the cube considered

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

