import os, argparse, trimesh, pyrender, numpy as np, json 
from PIL import Image 
 
def look_at_matrix(eye, target=[0,0,0], up=[0,0,1]): 
    eye, target, up = map(np.array, (eye,target,up)) 
    f = (target-eye); f /= np.linalg.norm(f) 
    u = up/np.linalg.norm(up) 
    s = np.cross(f,u); s/=np.linalg.norm(s); u = np.cross(s,f) 
    m = np.eye(4, dtype=np.float32) 
    m[0,:3],m[1,:3],m[2,:3] = s,u,-f 
    m[:3,3] = -eye @ m[:3,:3] 
    return m 
 
def render_views_with_poses(mesh_path, out_dir="nerf_dataset", n_views=20, img_size=800, radius=5.0, height=1.5): 
    os.makedirs(out_dir, exist_ok=True); img_dir = os.path.join(out_dir,"images"); os.makedirs(img_dir, exist_ok=True) 
    mesh = trimesh.load(mesh_path);  
    if isinstance(mesh, trimesh.Scene): mesh = trimesh.util.concatenate(mesh.dump()) 
    scene = pyrender.Scene(bg_color=[0,0,0,0], ambient_light=[0.6,0.6,0.6]) 
    scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False)) 
 
    yfov = np.pi/3.0; aspect=1.0 
    fx=fy=0.5*img_size/np.tan(yfov/2); cx=cy=img_size/2 
    frames=[] 
    for i in range(n_views): 
        angle=(i/n_views)*2*np.pi; eye=[radius*np.cos(angle), radius*np.sin(angle), height] 
        c2w=np.linalg.inv(look_at_matrix(eye)) 
        cam=pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=aspect) 
        node=scene.add(cam, pose=c2w); r=pyrender.OffscreenRenderer(img_size,img_size) 
        color,_=r.render(scene); r.delete(); scene.remove_node(node) 
        img_path=os.path.join(img_dir,f"view_{i:03d}.png"); Image.fromarray(color).save(img_path) 
        frames.append({"file_path":f"images/view_{i:03d}.png","transform_matrix":c2w.tolist()}) 
        print("Saved:",img_path) 
    transforms={"camera_angle_x":float(2*np.arctan(img_size/(2*fx))),"fl_x":fx,"fl_y":fy,"cx":cx,"cy":cy,"w":img_size,"h":img_size,"fra
 mes":frames} 
    with open(os.path.join(out_dir,"transforms.json"),"w") as f: json.dump(transforms,f,indent=2) 
    print("Saved transforms.json to",out_dir) 
 
