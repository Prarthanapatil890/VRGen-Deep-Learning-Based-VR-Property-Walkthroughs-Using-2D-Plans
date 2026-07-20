import os, argparse, imageio, numpy as np 
from infer_enhanced import infer_one 
from depth_est import estimate_depth_midas 
from mesh_refine import refine_mesh 
from texturize import bake_heightmap_as_vertex_color 
from utils import heightmap_to_mesh 
from render_views_with_poses import render_views_with_poses 
 
def main(floorplan, cfg="config.yaml", ckpt=None, tmp_dir="tmp_pipeline", export_nerf=True): 
    # --- FILE PATH RELIABILITY FIX --- 
    if not os.path.exists(floorplan): 
        raise FileNotFoundError(f"Input floorplan not found: {floorplan}")  
    # Resolve to an absolute path for reliability across sub-functions 
    floorplan_path_abs = os.path.abspath(floorplan) 
    # --------------------------------- 
    os.makedirs(tmp_dir, exist_ok=True) 
 
    print("1) Generator inference...") 
    # Use the absolute path 
    mesh_path = infer_one(floorplan_path_abs, cfg_path=cfg, ckpt=ckpt, out_mesh=os.path.join(tmp_dir,"initial_mesh.glb")) 
    heightmap_png = mesh_path.replace(".glb",".png") 
    current_mesh = mesh_path 
 
    print("2) Depth fusion...") 
    depth_out=os.path.join(tmp_dir,"depth.png") 
    # Use the absolute path 
    depth=estimate_depth_midas(floorplan_path_abs,out_path=depth_out) 
     
    try: 
        hm=imageio.imread(heightmap_png).astype(float)/255.0 
        dm=imageio.imread(depth_out).astype(float)/255.0 
        fused=(hm+dm)/2.0; fused_path=os.path.join(tmp_dir,"fused_height.png") 
        imageio.imwrite(fused_path,(fused*255).astype("uint8")) 
        fused_mesh=heightmap_to_mesh(fused,out_path=os.path.join(tmp_dir,"fused_mesh.glb")) 
        current_mesh=fused_mesh 
    except Exception as e: print("Fusion failed:",e) 
 
    print("3) Refining mesh...") 
    refined=refine_mesh(current_mesh,out_glb=os.path.join(tmp_dir,"refined.glb"),smoothing_iterations=8,target_faces=30000) 
 
    print("4) Texturizing mesh...") 
    height_src=os.path.join(tmp_dir,"fused_height.png") if os.path.exists(os.path.join(tmp_dir,"fused_height.png")) else 
heightmap_png 
    final_glb=bake_heightmap_as_vertex_color(refined,height_src,out_glb=os.path.join(tmp_dir,"final_textured.glb")) 
 
    if export_nerf: 
        print("5) Exporting NeRF dataset...") 
        render_views_with_poses(final_glb,out_dir="nerf_dataset",n_views=40,img_size=800,radius=6.0,height=1.5) 
 
    print("Done. Final GLB:",final_glb) 
    return final_glb 
 
if __name__=="__main__": 
    p=argparse.ArgumentParser() 
    p.add_argument("floorplan") 
    p.add_argument("--cfg",default="config.yaml") 
    p.add_argument("--ckpt",default="./checkpoints/netG_final.pth") 
    args=p.parse_args() 
    main(args.floorplan,cfg=args.cfg,ckpt=args.ckpt) 
 
 