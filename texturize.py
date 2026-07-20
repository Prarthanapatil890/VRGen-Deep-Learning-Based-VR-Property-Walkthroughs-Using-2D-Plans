import os 
import numpy as np 
from PIL import Image 
import trimesh 
def bake_heightmap_as_vertex_color(mesh_path, heightmap_path, out_glb=None): 
    if out_glb is None: 
        out_glb = mesh_path.replace(".glb", "_textured.glb") 
    mesh = trimesh.load(mesh_path) 
    if isinstance(mesh, trimesh.Scene): 
        mesh = trimesh.util.concatenate(mesh.dump()) 
    hmap = Image.open(heightmap_path).convert('L') 
    w,h = hmap.size 
    arr = np.array(hmap) / 255.0 
    verts = mesh.vertices.copy() 
    minxy = verts[:, :2].min(axis=0) 
    maxxy = verts[:, :2].max(axis=0) 
    spans = (maxxy - minxy) + 1e-9 
    uvx = ((verts[:,0] - minxy[0]) / spans[0] * (w-1)).astype(int) 
    uvy = ((verts[:,1] - minxy[1]) / spans[1] * (h-1)).astype(int) 
    uvx = np.clip(uvx, 0, w-1); uvy = np.clip(uvy, 0, h-1) 
    colors = arr[uvy, uvx] 
    colors_rgb = (np.stack([colors, colors, colors], axis=1) * 255).astype(np.uint8) 
    mesh.visual.vertex_colors = colors_rgb 
    os.makedirs(os.path.dirname(out_glb), exist_ok=True) 
    mesh.export(out_glb) 
    print("Saved textured GLB:", out_glb) 
    return out_glb 