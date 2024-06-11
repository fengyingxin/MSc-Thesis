import os
import argparse
import torch

import numpy as np

from pathlib import Path
from PIL import Image
from pytorch3d.io import (
    load_obj,
    save_obj,
    load_objs_as_meshes
)

# Setup
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")


def init_args():
    print("=> initializing input arguments...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--obj_name", type=str, required=True)

    args = parser.parse_args()

    return args

def init_mesh(args):
    print("=> loading target mesh...")
    model_path = os.path.join(args.input_dir, "{}.obj".format(args.obj_name))
    
    verts, faces, aux = load_obj(model_path, device=device)
    mesh = load_objs_as_meshes([model_path], device=device)

    texture_map = Image.open("init_texture.png").convert("RGB")
    texture_map = torch.from_numpy(np.array(texture_map)) / 255.

    return mesh, verts, faces, aux, texture_map

def normalize_mesh(mesh):
    bbox = mesh.get_bounding_boxes()
    num_verts = mesh.verts_packed().shape[0]

    # move mesh to origin
    mesh_center = bbox.mean(dim=2).repeat(num_verts, 1)
    mesh = mesh.offset_verts(-mesh_center)

    # scale
    lens = bbox[0, :, 1] - bbox[0, :, 0]
    max_len = lens.max()
    scale = 1 / max_len
    scale = scale.unsqueeze(0).repeat(num_verts)
    mesh.scale_verts_(scale)

    return mesh.verts_packed()

def rotate_verts(verts):
    theta = 90
    theta = theta * np.pi / 180
    A = torch.tensor([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ], device=device).float()

    xz = torch.stack([verts[:, 0], verts[:, 2]], dim=1).float()
    xz = torch.matmul(A, xz.T).T
    verts = torch.stack([xz[:, 0], verts[:, 1], xz[:, 1]], dim=1)
    return verts

if __name__ == "__main__":
    args = init_args()

    mesh, verts, faces, aux, texture_map = init_mesh(args)
    verts = normalize_mesh(mesh)
    verts = rotate_verts(verts)

    save_obj(
        str(Path(args.input_dir) / args.obj_name) + "_test.obj",
        verts=verts,
        faces=faces.verts_idx,
        decimal_places=5,
        verts_uvs=aux.verts_uvs,
        faces_uvs=faces.textures_idx,
        texture_map=texture_map
    )

