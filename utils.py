import os 
import numpy as np 
from PIL import Image 
import torchvision.utils as vutils 
import torch 
import trimesh 
from skimage import measure 
def save_sample(real_A, real_B, fake_B, out_dir, step=0): 
    """ 
    Save a grid: input | target | predicted 
    real_A: tensor [B,3,H,W] 
    real_B: tensor [B,1,H,W] 
    fake_B: tensor [B,1,H,W] 
    """ 
    os.makedirs(out_dir, exist_ok=True) 
    # make a visually comparable RGB grid: stack input, target as 3-ch for display, pred 
    B = real_A.shape[0] 
    rows = [] 
    for i in range(min(4,B)): 
        inp = real_A[i] 
        tgt = real_B[i].repeat(3,1,1) 
        pred = fake_B[i].repeat(3,1,1) 
        rows.extend([inp, tgt, pred]) 
    grid = vutils.make_grid(rows, nrow=3, normalize=True, scale_each=True) 
    out_path = os.path.join(out_dir, f"sample_{step:06d}.png") 
    vutils.save_image(grid, out_path) 
    print("Saved training sample:", out_path) 
def heightmap_to_mesh(heightmap, out_path="out_mesh.glb", scale_xy=1.0, scale_z=1.0, iso_level=0.5): 
    """ 
    Convert a normalized heightmap (H,W) numpy in [0,1] OR a path to image 
    to a mesh using marching cubes on a voxel grid extruded in Z. 
    Returns path to exported .glb 
    """ 
    # If heightmap is a path, load 
    if isinstance(heightmap, str): 
        img = Image.open(heightmap).convert('L').resize((256,256)) 
        arr = np.asarray(img).astype(float)/255.0 
    else: 
        arr = np.array(heightmap, dtype=float) 
        # rescale to a reasonable size if too big 
        if arr.max() <= 1.0 and arr.min() >= 0.0: 
            pass 
        else: 
            arr = (arr - arr.min()) / (arr.ptp() + 1e-8) 
 
    H,W = arr.shape 
    # build voxel volume: shape (H, W, Z) 
    Z = 64  # vertical resolution 
    # replicate along Z using heightmap values -> create a binary solid field 
    grid = np.zeros((H, W, Z), dtype=np.uint8) 
    for z in range(Z): 
        threshold = z / (Z - 1) 
        grid[:, :, z] = (arr >= threshold).astype(np.uint8) 
 
    # marching cubes expects volume with spacing. We'll compute verts/faces from grid. 
    verts, faces, normals, values = measure.marching_cubes(grid.astype(float), level=0.5) 
    # marching_cubes returns verts in voxel coords (z is along third axis). Transform coords: 
    # verts columns = (x, y, z_index). We'll map to (x*scale_xy, y*scale_xy, z*scale_z) 
    verts = verts[:, [1,0,2]]  # swap dims to (x,y,z) 
    verts[:, 0] = (verts[:, 0] / max(W-1,1)) * (W * scale_xy) 
    verts[:, 1] = (verts[:, 1] / max(H-1,1)) * (H * scale_xy) 
    verts[:, 2] = (verts[:, 2] / max(Z-1,1)) * (Z * scale_z) 
 
    # center mesh around origin (optional) 
    verts -= verts.mean(axis=0) 
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals, process=True) 
    # export as glb 
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True) 
    mesh.export(out_path) 
    print("Exported mesh to:", out_path) 
    return out_path