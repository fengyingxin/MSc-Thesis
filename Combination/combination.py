import clip
from tqdm import tqdm
import kaolin.ops.mesh
import kaolin as kal
import torch
from XMeshori.XMesh import NeuralStyleField
from XMeshori.utils import device 
from XMeshori.render import Renderer
from XMeshori.mesh import Mesh
from XMeshori.Normalization import MeshNormalizer
import numpy as np
import random
import copy
import torchvision
import os
from PIL import Image
import argparse
from pathlib import Path
from torchvision import transforms
import json
import zipfile
import time

def save_config(args):
    config = {}
    config['obj'] = args.xmesh_obj_path
    config['prompt'] = ' '.join(args.xmesh_prompt)
    config['output_dir'] = args.xmesh_output_dir
    config['color_path'] = os.path.join(args.xmesh_output_dir,'colors_final.pt')
    config['normal_path'] = os.path.join(args.xmesh_output_dir,'normals_final.pt')
    config['frontview_center'] = [0. , 0.]
    
    cfg = json.dumps(config,indent=4)
    f = open(os.path.join(args.xmesh_output_dir,'cfg.json'), 'w')
    f.write(cfg)
    

def run_branched(args):
    Path(args.xmesh_output_dir).mkdir(parents=True, exist_ok=True)
    save_config(args)
    # Constrain all sources of randomness
    torch.manual_seed(args.xmesh_seed)
    torch.cuda.manual_seed(args.xmesh_seed)
    torch.cuda.manual_seed_all(args.xmesh_seed)
    random.seed(args.xmesh_seed)
    np.random.seed(args.xmesh_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    # Load CLIP model 
    clip_model, preprocess = clip.load(args.xmesh_clipmodel, device, jit=args.xmesh_jit)
    
    # Adjust output resolution depending on model type 
    res = 224 
    if args.xmesh_clipmodel == "ViT-L/14@336px":
        res = 336
    if args.xmesh_clipmodel == "RN50x4":
        res = 288
    if args.xmesh_clipmodel == "RN50x16":
        res = 384
    if args.xmesh_clipmodel == "RN50x64":
        res = 448
        
    objbase, extension = os.path.splitext(os.path.basename(args.xmesh_obj_path))
    # Check that isn't already done
    if (not args.xmesh_overwrite) and os.path.exists(os.path.join(args.xmesh_output_dir, "loss.png")) and \
            os.path.exists(os.path.join(args.xmesh_output_dir, f"{objbase}_final.obj")):
        print(f"Already done with {args.xmesh_output_dir}")
        exit()
    elif args.xmesh_overwrite and os.path.exists(os.path.join(args.xmesh_output_dir, "loss.png")) and \
            os.path.exists(os.path.join(args.xmesh_output_dir, f"{objbase}_final.obj")):
        import shutil
        for filename in os.listdir(args.xmesh_output_dir):
            file_path = os.path.join(args.xmesh_output_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

    render = Renderer(dim=(res, res))
    mesh = Mesh(args.xmesh_obj_path)
    MeshNormalizer(mesh)()

    prior_color = torch.full(size=(mesh.faces.shape[0], 3, 3), fill_value=0.5, device=device)

    background = None
    if args.xmesh_background is not None:
        assert len(args.xmesh_background) == 3
        background = torch.tensor(args.xmesh_background).to(device)

    losses = []

    n_augs = args.xmesh_n_augs
    dir = args.xmesh_output_dir
    clip_normalizer = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    # CLIP Transform
    clip_transform = transforms.Compose([
        transforms.Resize((res, res)),
        clip_normalizer
    ])

    # Augmentation settings
    augment_transform = transforms.Compose([
        transforms.RandomResizedCrop(res, scale=(1, 1)),
        transforms.RandomPerspective(fill=1, p=0.8, distortion_scale=0.5),
        clip_normalizer
    ])

    # Augmentations for normal network
    if args.xmesh_cropforward :
        curcrop = args.xmesh_normmincrop
    else:
        curcrop = args.xmesh_normmaxcrop
    normaugment_transform = transforms.Compose([
        transforms.RandomResizedCrop(res, scale=(curcrop, curcrop)),
        transforms.RandomPerspective(fill=1, p=0.8, distortion_scale=0.5),
        clip_normalizer
    ])
    cropiter = 0
    cropupdate = 0
    if args.xmesh_normmincrop < args.xmesh_normmaxcrop and args.xmesh_cropsteps > 0:
        cropiter = round(args.xmesh_n_iter / (args.xmesh_cropsteps + 1))
        cropupdate = (args.xmesh_maxcrop - args.xmesh_mincrop) / cropiter

        if not args.xmesh_cropforward:
            cropupdate *= -1

    # Displacement-only augmentations
    displaugment_transform = transforms.Compose([
        transforms.RandomResizedCrop(res, scale=(args.xmesh_normmincrop, args.xmesh_normmincrop)),
        transforms.RandomPerspective(fill=1, p=0.8, distortion_scale=0.5),
        clip_normalizer
    ])

    normweight = 1.0

    # MLP Settings
    input_dim = 6 if args.xmesh_input_normals else 3
    if args.xmesh_only_z:
        input_dim = 1
    mlp = NeuralStyleField(args.xmesh_sigma, args.xmesh_depth, args.xmesh_width, 'gaussian', args.xmesh_colordepth, args.xmesh_normdepth,
                                args.xmesh_normratio, args.xmesh_clamp, args.xmesh_normclamp, niter=args.xmesh_n_iter,
                                progressive_encoding=args.pe, input_dim=input_dim, exclude=args.xmesh_exclude).to(device)
    mlp.reset_weights()
    

    parameters = [
        {'params': mlp.parameters(), 'weight_decay': args.xmesh_decay, 'lr': args.xmesh_learning_rate},
    ]


    optim = torch.optim.Adam(parameters, args.xmesh_learning_rate, weight_decay=args.xmesh_decay)
    activate_scheduler = args.xmesh_lr_decay < 1 and args.xmesh_decay_step > 0 and not args.xmesh_lr_plateau
    if activate_scheduler:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=args.xmesh_decay_step, gamma=args.xmesh_lr_decay)
    if not args.xmesh_no_prompt:
        if args.xmesh_prompt:
            prompt = ' '.join(args.xmesh_prompt)
            prompt_token = clip.tokenize([prompt]).to(device)
            encoded_text = clip_model.encode_text(prompt_token)

            with open(os.path.join(dir, prompt), "w") as f:
                f.write("")

            # Same with normprompt
            norm_encoded = encoded_text
    if args.xmesh_normprompt is not None:
        prompt = ' '.join(args.xmesh_normprompt)
        prompt_token = clip.tokenize([prompt]).to(device)
        norm_encoded = clip_model.encode_text(prompt_token)

        # Save prompt
        with open(os.path.join(dir, f"NORM {prompt}"), "w") as f:
            f.write("")

    if args.xmesh_image:
        img = Image.open(args.xmesh_image)
        img = preprocess(img).to(device)
        encoded_image = clip_model.encode_image(img.unsqueeze(0))
        if args.xmesh_no_prompt:
            norm_encoded = encoded_image

    loss_check = None
    vertices = copy.deepcopy(mesh.vertices)
    network_input = copy.deepcopy(vertices)
    if args.xmesh_symmetry == True:
        network_input[:,2] = torch.abs(network_input[:,2])

    if args.xmesh_standardize == True:
        # Each channel into z-score
        network_input = (network_input - torch.mean(network_input, dim=0))/torch.std(network_input, dim=0)

    for i in tqdm(range(args.xmesh_n_iter)):
        optim.zero_grad()

        sampled_mesh = mesh


        update_mesh(mlp, network_input, encoded_text, prior_color, sampled_mesh, vertices)
        rendered_images, elev, azim = render.render_front_views(sampled_mesh, num_views=args.xmesh_n_views,
                                                                show=args.xmesh_show,
                                                                center_azim=args.xmesh_frontview_center[0],
                                                                center_elev=args.xmesh_frontview_center[1],
                                                                std=args.xmesh_frontview_std,
                                                                return_views=True,
                                                                background=background)
    
        if n_augs == 0:
            clip_image = clip_transform(rendered_images)
            encoded_renders = clip_model.encode_image(clip_image)
            if not args.xmesh_no_prompt:
                loss = torch.mean(torch.cosine_similarity(encoded_renders, encoded_text))

        # Check augmentation steps
        if args.xmesh_cropsteps != 0 and cropupdate != 0 and i != 0 and i % args.xmesh_cropsteps == 0:
            curcrop += cropupdate
            # print(curcrop)
            normaugment_transform = transforms.Compose([
                transforms.RandomResizedCrop(res, scale=(curcrop, curcrop)),
                transforms.RandomPerspective(fill=1, p=0.8, distortion_scale=0.5),
                clip_normalizer
            ])

        if n_augs > 0:
            loss = 0.0
            for _ in range(n_augs):
                augmented_image = augment_transform(rendered_images)
                encoded_renders = clip_model.encode_image(augmented_image)
                if not args.xmesh_no_prompt:
                    if args.xmesh_prompt:
                        if args.xmesh_clipavg == "view":
                            if encoded_text.shape[0] > 1:
                                loss -= torch.cosine_similarity(torch.mean(encoded_renders, dim=0),
                                                                torch.mean(encoded_text, dim=0), dim=0)
                            else:
                                loss -= torch.cosine_similarity(torch.mean(encoded_renders, dim=0, keepdim=True),
                                                                encoded_text)
                        else:
                            loss -= torch.mean(torch.cosine_similarity(encoded_renders, encoded_text))
                if args.xmesh_image:
                    if encoded_image.shape[0] > 1:
                        loss -= torch.cosine_similarity(torch.mean(encoded_renders, dim=0),
                                                        torch.mean(encoded_image, dim=0), dim=0)
                    else:
                        loss -= torch.cosine_similarity(torch.mean(encoded_renders, dim=0, keepdim=True),
                                                        encoded_image)
        if args.xmesh_splitnormloss:
            for param in mlp.mlp_normal.parameters():
                param.requires_grad = False
        loss.backward(retain_graph=True)

        if args.xmesh_n_normaugs > 0:
            normloss = 0.0
            for _ in range(args.xmesh_n_normaugs):
                augmented_image = normaugment_transform(rendered_images)
                encoded_renders = clip_model.encode_image(augmented_image)
                if not args.xmesh_no_prompt:
                    if args.xmesh_prompt:
                        if args.xmesh_clipavg == "view":
                            if norm_encoded.shape[0] > 1:
                                normloss -= normweight * torch.cosine_similarity(torch.mean(encoded_renders, dim=0),
                                                                                 torch.mean(norm_encoded, dim=0),
                                                                                 dim=0)
                            else:
                                normloss -= normweight * torch.cosine_similarity(
                                    torch.mean(encoded_renders, dim=0, keepdim=True),
                                    norm_encoded)
                        else:
                            normloss -= normweight * torch.mean(
                                torch.cosine_similarity(encoded_renders, norm_encoded))
                if args.xmesh_image:
                    if encoded_image.shape[0] > 1:
                        loss -= torch.cosine_similarity(torch.mean(encoded_renders, dim=0),
                                                        torch.mean(encoded_image, dim=0), dim=0)
                    else:
                        loss -= torch.cosine_similarity(torch.mean(encoded_renders, dim=0, keepdim=True),
                                                        encoded_image)
            if args.xmesh_splitnormloss:
                for param in mlp.mlp_normal.parameters():
                    param.requires_grad = True
            if args.xmesh_splitcolorloss:
                for param in mlp.mlp_rgb.parameters():
                    param.requires_grad = False
            if not args.xmesh_no_prompt:
                normloss.backward(retain_graph=True)

        # Also run separate loss on the uncolored displacements
        if args.xmesh_geoloss:
            default_color = torch.zeros(len(mesh.vertices), 3).to(device)
            default_color[:, :] = torch.tensor([0.5, 0.5, 0.5]).to(device)
            sampled_mesh.face_attributes = kaolin.ops.mesh.index_vertices_by_faces(default_color.unsqueeze(0),
                                                                                   sampled_mesh.faces)
            geo_renders, elev, azim = render.render_front_views(sampled_mesh, num_views=args.xmesh_n_views,
                                                                show=args.xmesh_show,
                                                                center_azim=args.xmesh_frontview_center[0],
                                                                center_elev=args.xmesh_frontview_center[1],
                                                                std=args.xmesh_frontview_std,
                                                                return_views=True,
                                                                background=background)
            if args.xmesh_n_normaugs > 0:
                normloss = 0.0
                ### avgview != aug
                for _ in range(args.xmesh_n_normaugs):
                    augmented_image = displaugment_transform(geo_renders)
                    encoded_renders = clip_model.encode_image(augmented_image)
                    if norm_encoded.shape[0] > 1:
                        normloss -= torch.cosine_similarity(torch.mean(encoded_renders, dim=0),
                                                            torch.mean(norm_encoded, dim=0), dim=0)
                    else:
                        normloss -= torch.cosine_similarity(torch.mean(encoded_renders, dim=0, keepdim=True),
                                                            norm_encoded)
                    if args.xmesh_image:
                        if encoded_image.shape[0] > 1:
                            loss -= torch.cosine_similarity(torch.mean(encoded_renders, dim=0),
                                                            torch.mean(encoded_image, dim=0), dim=0)
                        else:
                            loss -= torch.cosine_similarity(torch.mean(encoded_renders, dim=0, keepdim=True),
                                                            encoded_image)
                normloss.backward(retain_graph=True)
        optim.step()

        for param in mlp.mlp_normal.parameters():
            param.requires_grad = True
        for param in mlp.mlp_rgb.parameters():
            param.requires_grad = True

        if activate_scheduler:
            lr_scheduler.step()

        with torch.no_grad():
            losses.append(loss.item())

        # Adjust normweight if set
        if args.xmesh_decayfreq is not None:
            if i % args.xmesh_decayfreq == 0:
                normweight *= args.xmesh_cropdecay

        export_iter_results(args, dir, losses, mesh, mlp, network_input,encoded_text, vertices,str(i))
        if i % 100 == 0:
            report_process(args, dir, i, loss, loss_check, losses, rendered_images)
           
    output_model_obj=export_final_results(args, dir, losses, mesh, mlp, network_input,encoded_text, vertices)
    print('finish xmesh edit')
    return output_model_obj


def report_process(args, dir, i, loss, loss_check, losses, rendered_images):
    print('iter: {} loss: {}'.format(i, loss.item()))
    torchvision.utils.save_image(rendered_images, os.path.join(dir, 'iter_{}.jpg'.format(i)))
    if args.xmesh_lr_plateau and loss_check is not None:
        new_loss_check = np.mean(losses[-100:])
        # If avg loss increased or plateaued then reduce LR
        if new_loss_check >= loss_check:
            for g in torch.optim.param_groups:
                g['lr'] *= 0.5
        loss_check = new_loss_check

    elif args.xmesh_lr_plateau and loss_check is None and len(losses) >= 100:
        loss_check = np.mean(losses[-100:])

def export_iter_results(args, dir, losses, mesh, mlp, network_input,encoded_text, vertices,iter):
    with torch.no_grad():
        color_dir = os.path.join(dir,'colors')
        normal_dir = os.path.join(dir,'normals')
        img_dir = os.path.join(dir,'imgs')
        if(not os.path.exists(color_dir)):
            os.makedirs(color_dir)
        if(not os.path.exists(normal_dir)):
            os.makedirs(normal_dir)
        if(not os.path.exists(img_dir)):
            os.makedirs(img_dir)
        
        pred_rgb, pred_normal = mlp(network_input,encoded_text)
        pred_rgb = pred_rgb.detach().cpu()
        pred_normal = pred_normal.detach().cpu()

        torch.save(pred_rgb, os.path.join(color_dir, f"colors_{iter}iter.pt"))
        torch.save(pred_normal, os.path.join(normal_dir, f"normals_{iter}iter.pt"))

        base_color = torch.full(size=(mesh.vertices.shape[0], 3), fill_value=0.5)
        final_color = torch.clamp(pred_rgb + base_color, 0, 1)

        mesh.vertices = vertices.detach().cpu() + mesh.vertex_normals.detach().cpu() * pred_normal

        objbase, extension = os.path.splitext(os.path.basename(args.xmesh_obj_path))
        # mesh.export(os.path.join(dir, f"{objbase}_{iter}iter.obj"), color=final_color)

        # Run renders
        if args.xmesh_save_render:
            save_rendered_results_iter(args, img_dir, final_color, mesh,iter)


def export_final_results(args, dir, losses, mesh, mlp, network_input,encoded_text, vertices):
    with torch.no_grad():
        pred_rgb, pred_normal = mlp(network_input,encoded_text)
        pred_rgb = pred_rgb.detach().cpu()
        pred_normal = pred_normal.detach().cpu()

        torch.save(pred_rgb, os.path.join(dir, f"colors_final.pt"))
        torch.save(pred_normal, os.path.join(dir, f"normals_final.pt"))

        base_color = torch.full(size=(mesh.vertices.shape[0], 3), fill_value=0.5)
        final_color = torch.clamp(pred_rgb + base_color, 0, 1)

        mesh.vertices = vertices.detach().cpu() + mesh.vertex_normals.detach().cpu() * pred_normal

        objbase, extension = os.path.splitext(os.path.basename(args.xmesh_obj_path))
        mesh.export(os.path.join(dir, f"{objbase}_final.obj"), color=final_color)
        output_model_obj=os.path.join(dir, f"{objbase}_final.obj")

        # Run renders
        if args.xmesh_save_render:
            save_rendered_results(args, dir, final_color, mesh)

        # Save final losses
        torch.save(torch.tensor(losses), os.path.join(dir, "losses.pt"))
        return output_model_obj



def save_rendered_results_iter(args, dir, final_color, mesh,iter):
    default_color = torch.full(size=(mesh.vertices.shape[0], 3), fill_value=0.5, device=device)
    mesh.face_attributes = kaolin.ops.mesh.index_vertices_by_faces(default_color.unsqueeze(0),
                                                                   mesh.faces.to(device))
    kal_render = Renderer(
        camera=kal.render.camera.generate_perspective_projection(np.pi / 4, 224 / 224).to(device),
        dim=(224, 224))

    MeshNormalizer(mesh)()
    # Vertex colorings
    mesh.face_attributes = kaolin.ops.mesh.index_vertices_by_faces(final_color.unsqueeze(0).to(device),
                                                                   mesh.faces.to(device))
    img, mask = kal_render.render_single_view(mesh, args.xmesh_frontview_center[1], args.xmesh_frontview_center[0],
                                              radius=2.5,
                                              background=torch.tensor([1, 1, 1]).to(device).float(),
                                              return_mask=True)
    img = img[0].cpu()
    mask = mask[0].cpu()
    # Manually add alpha channel using background color
    alpha = torch.ones(img.shape[1], img.shape[2])
    alpha[torch.where(mask == 0)] = 0
    img = torch.cat((img, alpha.unsqueeze(0)), dim=0)
    img = transforms.ToPILImage()(img)
    img.save(os.path.join(dir, f"final_cluster_{iter}iter.png"))


def save_rendered_results(args, dir, final_color, mesh):
    default_color = torch.full(size=(mesh.vertices.shape[0], 3), fill_value=0.5, device=device)
    mesh.face_attributes = kaolin.ops.mesh.index_vertices_by_faces(default_color.unsqueeze(0),
                                                                   mesh.faces.to(device))
    kal_render = Renderer(
        camera=kal.render.camera.generate_perspective_projection(np.pi / 4, 1280 / 720).to(device),
        dim=(1280, 720))
    MeshNormalizer(mesh)()
    img, mask = kal_render.render_single_view(mesh, args.xmesh_frontview_center[1], args.xmesh_frontview_center[0],
                                              radius=2.5,
                                              background=torch.tensor([1, 1, 1]).to(device).float(),
                                              return_mask=True)
    img = img[0].cpu()
    mask = mask[0].cpu()
    # Manually add alpha channel using background color
    alpha = torch.ones(img.shape[1], img.shape[2])
    alpha[torch.where(mask == 0)] = 0
    img = torch.cat((img, alpha.unsqueeze(0)), dim=0)
    img = transforms.ToPILImage()(img)
    img.save(os.path.join(dir, f"init_cluster.png"))
    MeshNormalizer(mesh)()
    # Vertex colorings
    mesh.face_attributes = kaolin.ops.mesh.index_vertices_by_faces(final_color.unsqueeze(0).to(device),
                                                                   mesh.faces.to(device))
    img, mask = kal_render.render_single_view(mesh, args.xmesh_frontview_center[1], args.xmesh_frontview_center[0],
                                              radius=2.5,
                                              background=torch.tensor([1, 1, 1]).to(device).float(),
                                              return_mask=True)
    img = img[0].cpu()
    mask = mask[0].cpu()
    # Manually add alpha channel using background color
    alpha = torch.ones(img.shape[1], img.shape[2])
    alpha[torch.where(mask == 0)] = 0
    img = torch.cat((img, alpha.unsqueeze(0)), dim=0)
    img = transforms.ToPILImage()(img)
    img.save(os.path.join(dir, f"final_cluster.png"))


def update_mesh(mlp, network_input,prompt, prior_color, sampled_mesh, vertices):
    # network_input=vertices here
    pred_rgb, pred_normal = mlp(network_input,prompt)
    sampled_mesh.face_attributes = prior_color + kaolin.ops.mesh.index_vertices_by_faces(
        pred_rgb.unsqueeze(0),
        sampled_mesh.faces)
    sampled_mesh.vertices = vertices +  pred_normal
    MeshNormalizer(sampled_mesh)()

def seed_everything(seed=42):
    '''
    :param seed:
    :param device:
    :return:
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True

# preprocess for text2tex
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


def preprocess_init_mesh(output_model_obj):
    print("=> loading target mesh...")
    model_path = output_model_obj
    
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

def rotate_verts(verts,theta):
    theta = theta
    theta = theta * np.pi / 180
    A = torch.tensor([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ], device=device).float()

    xz = torch.stack([verts[:, 0], verts[:, 2]], dim=1).float()
    xz = torch.matmul(A, xz.T).T
    verts = torch.stack([xz[:, 0], verts[:, 1], xz[:, 1]], dim=1)
    return verts

def text2tex_preprocess(output_model_obj):
    mesh, verts, faces, aux, texture_map = preprocess_init_mesh(output_model_obj)
    verts = normalize_mesh(mesh)
    verts = rotate_verts(verts,90)
    new_output_model_obj=output_model_obj.replace('.obj', '_new.obj')
    save_obj(
        new_output_model_obj,
        verts=verts,
        faces=faces.verts_idx,
        decimal_places=5,
        verts_uvs=aux.verts_uvs,
        faces_uvs=faces.textures_idx,
        texture_map=texture_map
    )
    print('finish_preprocessing')
    return new_output_model_obj


# text2tex
from pytorch3d.renderer import TexturesUV
import sys

from Text2Tex.lib.mesh_helper import (
    init_mesh,
    apply_offsets_to_mesh,
    adjust_uv_map
)
from Text2Tex.lib.render_helper import render
from Text2Tex.lib.io_helper import (
    save_backproject_obj,
    save_args,
    save_viewpoints
)
from Text2Tex.lib.vis_helper import (
    visualize_outputs, 
    visualize_principle_viewpoints, 
    visualize_refinement_viewpoints
)
from Text2Tex.lib.diffusion_helper import (
    get_controlnet_depth,
    get_inpainting,
    apply_controlnet_depth,
    apply_inpainting_postprocess
)
from Text2Tex.lib.projection_helper import (
    backproject_from_image,
    render_one_view_and_build_masks,
    select_viewpoint,
    build_similarity_texture_cache_for_all_views
)
from Text2Tex.lib.camera_helper import init_viewpoints

# Setup
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    torch.cuda.set_device(DEVICE)
else:
    print("no gpu avaiable")
    exit()

def text2tex_ori(new_output_model_obj,args):
    # args
    args.text2tex_input_dir=os.path.dirname(new_output_model_obj)
    args.text2tex_obj_name=new_output_model_obj.split('/')[-1].replace('.obj', '')
    args.text2tex_obj_file=new_output_model_obj.split('/')[-1]

    # save
    output_dir = os.path.join(
        args.text2tex_output_dir, 
        "{}-{}-{}-{}-{}-{}".format(
            str(args.text2tex_seed),
            args.text2tex_viewpoint_mode[0]+str(args.text2tex_num_viewpoints),
            args.text2tex_update_mode[0]+str(args.text2tex_update_steps),
            str(args.text2tex_new_strength),
            str(args.text2tex_update_strength),
            str(args.text2tex_view_threshold)
        ),
    )
    if args.text2tex_no_repaint: output_dir += "-norepaint"
    if args.text2tex_no_update: output_dir += "-noupdate"

    os.makedirs(output_dir, exist_ok=True)
    print("=> OUTPUT_DIR:", output_dir)

    # init resources
    # init mesh
    mesh, _, faces, aux, principle_directions, mesh_center, mesh_scale = init_mesh(
        os.path.join(args.text2tex_input_dir, args.text2tex_obj_file),
        os.path.join(output_dir, args.text2tex_obj_file), 
        DEVICE
    )

    # gradient texture
    init_texture = Image.open("init_texture.png").convert("RGB").resize((args.text2tex_uv_size, args.text2tex_uv_size))

    # HACK adjust UVs for multiple materials
    if args.text2tex_use_multiple_objects:
        new_verts_uvs, init_texture = adjust_uv_map(faces, aux, init_texture, args.text2tex_uv_size)
    else:
        new_verts_uvs = aux.verts_uvs

    # update the mesh
    mesh.textures = TexturesUV(
        maps=transforms.ToTensor()(init_texture)[None, ...].permute(0, 2, 3, 1).to(DEVICE),
        faces_uvs=faces.textures_idx[None, ...],
        verts_uvs=new_verts_uvs[None, ...]
    )

    # back-projected faces
    exist_texture = torch.from_numpy(np.zeros([args.text2tex_uv_size, args.text2tex_uv_size]).astype(np.float32)).to(DEVICE)

    # initialize viewpoints
    # including: principle viewpoints for generation + refinement viewpoints for updating
    (
        dist_list, 
        elev_list, 
        azim_list, 
        sector_list,
        view_punishments
    ) = init_viewpoints(args.text2tex_viewpoint_mode, args.text2tex_num_viewpoints, args.text2tex_dist, args.text2tex_elev, principle_directions, 
                            use_principle=True, 
                            use_shapenet=args.text2tex_use_shapenet,
                            use_objaverse=args.text2tex_use_objaverse)

    # save args
    save_args(args, output_dir)

    # initialize depth2image model
    controlnet, ddim_sampler = get_controlnet_depth()


    # ------------------- OPERATION ZONE BELOW ------------------------

    # 1. generate texture with RePaint 
    # NOTE no update / refinement

    generate_dir = os.path.join(output_dir, "generate")
    os.makedirs(generate_dir, exist_ok=True)

    update_dir = os.path.join(output_dir, "update")
    os.makedirs(update_dir, exist_ok=True)

    init_image_dir = os.path.join(generate_dir, "rendering")
    os.makedirs(init_image_dir, exist_ok=True)

    normal_map_dir = os.path.join(generate_dir, "normal")
    os.makedirs(normal_map_dir, exist_ok=True)

    mask_image_dir = os.path.join(generate_dir, "mask")
    os.makedirs(mask_image_dir, exist_ok=True)

    depth_map_dir = os.path.join(generate_dir, "depth")
    os.makedirs(depth_map_dir, exist_ok=True)

    similarity_map_dir = os.path.join(generate_dir, "similarity")
    os.makedirs(similarity_map_dir, exist_ok=True)

    inpainted_image_dir = os.path.join(generate_dir, "inpainted")
    os.makedirs(inpainted_image_dir, exist_ok=True)

    mesh_dir = os.path.join(generate_dir, "mesh")
    os.makedirs(mesh_dir, exist_ok=True)

    interm_dir = os.path.join(generate_dir, "intermediate")
    os.makedirs(interm_dir, exist_ok=True)

    # prepare viewpoints and cache
    NUM_PRINCIPLE = 10 if args.text2tex_use_shapenet or args.text2tex_use_objaverse else 6
    pre_dist_list = dist_list[:NUM_PRINCIPLE]
    pre_elev_list = elev_list[:NUM_PRINCIPLE]
    pre_azim_list = azim_list[:NUM_PRINCIPLE]
    pre_sector_list = sector_list[:NUM_PRINCIPLE]
    pre_view_punishments = view_punishments[:NUM_PRINCIPLE]

    pre_similarity_texture_cache = build_similarity_texture_cache_for_all_views(mesh, faces, new_verts_uvs,
        pre_dist_list, pre_elev_list, pre_azim_list,
        args.text2tex_image_size, args.text2tex_image_size * args.text2tex_render_simple_factor, args.text2tex_uv_size, args.text2tex_fragment_k,
        DEVICE
    )


    # start generation
    print("=> start generating texture...")
    start_time = time.time()
    for view_idx in range(NUM_PRINCIPLE):
        print("=> processing view {}...".format(view_idx))

        # sequentially pop the viewpoints
        dist, elev, azim, sector = pre_dist_list[view_idx], pre_elev_list[view_idx], pre_azim_list[view_idx], pre_sector_list[view_idx] 
        prompt = " the {} view of {}".format(sector, args.text2tex_prompt) if args.text2tex_add_view_to_prompt else args.text2tex_prompt
        print("=> generating image for prompt: {}...".format(prompt))

        # 1.1. render and build masks
        (
            view_score,
            renderer, cameras, fragments,
            init_image, normal_map, depth_map, 
            init_images_tensor, normal_maps_tensor, depth_maps_tensor, similarity_tensor, 
            keep_mask_image, update_mask_image, generate_mask_image, 
            keep_mask_tensor, update_mask_tensor, generate_mask_tensor, all_mask_tensor, quad_mask_tensor,
        ) = render_one_view_and_build_masks(dist, elev, azim, 
            view_idx, view_idx, view_punishments, # => actual view idx and the sequence idx 
            pre_similarity_texture_cache, exist_texture,
            mesh, faces, new_verts_uvs,
            args.text2tex_image_size, args.text2tex_fragment_k,
            init_image_dir, mask_image_dir, normal_map_dir, depth_map_dir, similarity_map_dir,
            DEVICE, save_intermediate=True, smooth_mask=args.text2tex_smooth_mask, view_threshold=args.text2tex_view_threshold
        )

        # 1.2. generate missing region
        # NOTE first view still gets the mask for consistent ablations
        if args.text2tex_no_repaint and view_idx != 0:
            actual_generate_mask_image = Image.fromarray((np.ones_like(np.array(generate_mask_image)) * 255.).astype(np.uint8))
        else:
            actual_generate_mask_image = generate_mask_image

        print("=> generate for view {}".format(view_idx))
        generate_image, generate_image_before, generate_image_after = apply_controlnet_depth(controlnet, ddim_sampler, 
            init_image.convert("RGBA"), prompt, args.text2tex_new_strength, args.text2tex_ddim_steps,
            actual_generate_mask_image, keep_mask_image, depth_maps_tensor.permute(1, 2, 0).repeat(1, 1, 3).cpu().numpy(), 
            args.text2tex_a_prompt, args.text2tex_n_prompt, args.text2tex_guidance_scale, args.text2tex_seed, args.text2tex_eta, 1, DEVICE, args.text2tex_blend)

        generate_image.save(os.path.join(inpainted_image_dir, "{}.png".format(view_idx)))
        generate_image_before.save(os.path.join(inpainted_image_dir, "{}_before.png".format(view_idx)))
        generate_image_after.save(os.path.join(inpainted_image_dir, "{}_after.png".format(view_idx)))

        # 1.2.2 back-project and create texture
        # NOTE projection mask = generate mask
        init_texture, project_mask_image, exist_texture = backproject_from_image(
            mesh, faces, new_verts_uvs, cameras, 
            generate_image, generate_mask_image, generate_mask_image, init_texture, exist_texture, 
            args.text2tex_image_size * args.text2tex_render_simple_factor, args.text2tex_uv_size, args.text2tex_fragment_k,
            DEVICE
        )

        project_mask_image.save(os.path.join(mask_image_dir, "{}_project.png".format(view_idx)))

        # update the mesh
        mesh.textures = TexturesUV(
            maps=transforms.ToTensor()(init_texture)[None, ...].permute(0, 2, 3, 1).to(DEVICE),
            faces_uvs=faces.textures_idx[None, ...],
            verts_uvs=new_verts_uvs[None, ...]
        )

        # 1.2.3. re: render 
        # NOTE only the rendered image is needed - masks should be re-used
        (
            view_score,
            renderer, cameras, fragments,
            init_image, *_,
        ) = render_one_view_and_build_masks(dist, elev, azim, 
            view_idx, view_idx, view_punishments, # => actual view idx and the sequence idx 
            pre_similarity_texture_cache, exist_texture,
            mesh, faces, new_verts_uvs,
            args.text2tex_image_size, args.text2tex_fragment_k,
            init_image_dir, mask_image_dir, normal_map_dir, depth_map_dir, similarity_map_dir,
            DEVICE, save_intermediate=False, smooth_mask=args.text2tex_smooth_mask, view_threshold=args.text2tex_view_threshold
        )

        # 1.3. update blurry region
        # only when: 1) use update flag; 2) there are contents to update; 3) there are enough contexts.
        if not args.text2tex_no_update and update_mask_tensor.sum() > 0 and update_mask_tensor.sum() / (all_mask_tensor.sum()) > 0.05:
            print("=> update {} pixels for view {}".format(update_mask_tensor.sum().int(), view_idx))
            diffused_image, diffused_image_before, diffused_image_after = apply_controlnet_depth(controlnet, ddim_sampler, 
                init_image.convert("RGBA"), prompt, args.text2tex_update_strength, args.text2tex_ddim_steps,
                update_mask_image, keep_mask_image, depth_maps_tensor.permute(1, 2, 0).repeat(1, 1, 3).cpu().numpy(), 
                args.text2tex_a_prompt, args.text2tex_n_prompt, args.text2tex_guidance_scale, args.text2tex_seed, args.text2tex_eta, 1, DEVICE, args.text2tex_blend)

            diffused_image.save(os.path.join(inpainted_image_dir, "{}_update.png".format(view_idx)))
            diffused_image_before.save(os.path.join(inpainted_image_dir, "{}_update_before.png".format(view_idx)))
            diffused_image_after.save(os.path.join(inpainted_image_dir, "{}_update_after.png".format(view_idx)))
        
            # 1.3.2. back-project and create texture
            # NOTE projection mask = generate mask
            init_texture, project_mask_image, exist_texture = backproject_from_image(
                mesh, faces, new_verts_uvs, cameras, 
                diffused_image, update_mask_image, update_mask_image, init_texture, exist_texture, 
                args.text2tex_image_size * args.text2tex_render_simple_factor, args.text2tex_uv_size, args.text2tex_fragment_k,
                DEVICE
            )
            
            # update the mesh
            mesh.textures = TexturesUV(
                maps=transforms.ToTensor()(init_texture)[None, ...].permute(0, 2, 3, 1).to(DEVICE),
                faces_uvs=faces.textures_idx[None, ...],
                verts_uvs=new_verts_uvs[None, ...]
            )


        # 1.4. save generated assets
        # save backprojected OBJ file
        save_backproject_obj(
            mesh_dir, "{}.obj".format(view_idx),
            mesh_scale * mesh.verts_packed() + mesh_center if args.text2tex_use_unnormalized else mesh.verts_packed(),
            faces.verts_idx, new_verts_uvs, faces.textures_idx, init_texture, 
            DEVICE
        )

        # save the intermediate view
        inter_images_tensor, *_ = render(mesh, renderer)
        inter_image = inter_images_tensor[0].cpu()
        inter_image = inter_image.permute(2, 0, 1)
        inter_image = transforms.ToPILImage()(inter_image).convert("RGB")
        inter_image.save(os.path.join(interm_dir, "{}.png".format(view_idx)))

        # save texture mask
        exist_texture_image = exist_texture * 255. 
        exist_texture_image = Image.fromarray(exist_texture_image.cpu().numpy().astype(np.uint8)).convert("L")
        exist_texture_image.save(os.path.join(mesh_dir, "{}_texture_mask.png".format(view_idx)))

    print("=> total generate time: {} s".format(time.time() - start_time))

    # visualize viewpoints
    visualize_principle_viewpoints(output_dir, pre_dist_list, pre_elev_list, pre_azim_list)

    # 2. update texture with RePaint 

    if args.text2tex_update_steps > 0:

        update_dir = os.path.join(output_dir, "update")
        os.makedirs(update_dir, exist_ok=True)

        init_image_dir = os.path.join(update_dir, "rendering")
        os.makedirs(init_image_dir, exist_ok=True)

        normal_map_dir = os.path.join(update_dir, "normal")
        os.makedirs(normal_map_dir, exist_ok=True)

        mask_image_dir = os.path.join(update_dir, "mask")
        os.makedirs(mask_image_dir, exist_ok=True)

        depth_map_dir = os.path.join(update_dir, "depth")
        os.makedirs(depth_map_dir, exist_ok=True)

        similarity_map_dir = os.path.join(update_dir, "similarity")
        os.makedirs(similarity_map_dir, exist_ok=True)

        inpainted_image_dir = os.path.join(update_dir, "inpainted")
        os.makedirs(inpainted_image_dir, exist_ok=True)

        mesh_dir = os.path.join(update_dir, "mesh")
        os.makedirs(mesh_dir, exist_ok=True)

        interm_dir = os.path.join(update_dir, "intermediate")
        os.makedirs(interm_dir, exist_ok=True)

        dist_list = dist_list[NUM_PRINCIPLE:]
        elev_list = elev_list[NUM_PRINCIPLE:]
        azim_list = azim_list[NUM_PRINCIPLE:]
        sector_list = sector_list[NUM_PRINCIPLE:]
        view_punishments = view_punishments[NUM_PRINCIPLE:]

        similarity_texture_cache = build_similarity_texture_cache_for_all_views(mesh, faces, new_verts_uvs,
            dist_list, elev_list, azim_list,
            args.text2tex_image_size, args.text2tex_image_size * args.text2tex_render_simple_factor, args.text2tex_uv_size, args.text2tex_fragment_k,
            DEVICE
        )
        selected_view_ids = []

        print("=> start updating...")
        start_time = time.time()
        for view_idx in range(args.text2tex_update_steps):
            print("=> processing view {}...".format(view_idx))
            
            # 2.1. render and build masks

            # heuristically select the viewpoints
            dist, elev, azim, sector, selected_view_ids, view_punishments = select_viewpoint(
                selected_view_ids, view_punishments,
                args.text2tex_update_mode, dist_list, elev_list, azim_list, sector_list, view_idx,
                similarity_texture_cache, exist_texture,
                mesh, faces, new_verts_uvs,
                args.text2tex_image_size, args.text2tex_fragment_k,
                init_image_dir, mask_image_dir, normal_map_dir, depth_map_dir, similarity_map_dir,
                DEVICE, False
            )

            (
                view_score,
                renderer, cameras, fragments,
                init_image, normal_map, depth_map, 
                init_images_tensor, normal_maps_tensor, depth_maps_tensor, similarity_tensor, 
                old_mask_image, update_mask_image, generate_mask_image, 
                old_mask_tensor, update_mask_tensor, generate_mask_tensor, all_mask_tensor, quad_mask_tensor,
            ) = render_one_view_and_build_masks(dist, elev, azim, 
                selected_view_ids[-1], view_idx, view_punishments, # => actual view idx and the sequence idx 
                similarity_texture_cache, exist_texture,
                mesh, faces, new_verts_uvs,
                args.text2tex_image_size, args.text2tex_fragment_k,
                init_image_dir, mask_image_dir, normal_map_dir, depth_map_dir, similarity_map_dir,
                DEVICE, save_intermediate=True, smooth_mask=args.text2tex_smooth_mask, view_threshold=args.text2tex_view_threshold
            )

            # 2.2. update existing region
            prompt = " the {} view of {}".format(sector, args.text2tex_prompt) if args.text2tex_add_view_to_prompt else args.text2tex_prompt
            print("=> updating image for prompt: {}...".format(prompt))

            if not args.text2tex_no_update and update_mask_tensor.sum() > 0 and update_mask_tensor.sum() / (all_mask_tensor.sum()) > 0.05:
                print("=> update {} pixels for view {}".format(update_mask_tensor.sum().int(), view_idx))
                update_image, update_image_before, update_image_after = apply_controlnet_depth(controlnet, ddim_sampler, 
                    init_image.convert("RGBA"), prompt, args.text2tex_update_strength, args.text2tex_ddim_steps,
                    update_mask_image, old_mask_image, depth_maps_tensor.permute(1, 2, 0).repeat(1, 1, 3).cpu().numpy(), 
                    args.text2tex_a_prompt, args.text2tex_n_prompt, args.text2tex_guidance_scale, args.text2tex_seed, args.text2tex_eta, 1, DEVICE, args.text2tex_blend)

                update_image.save(os.path.join(inpainted_image_dir, "{}.png".format(view_idx)))
                update_image_before.save(os.path.join(inpainted_image_dir, "{}_before.png".format(view_idx)))
                update_image_after.save(os.path.join(inpainted_image_dir, "{}_after.png".format(view_idx)))
            else:
                print("=> nothing to update for view {}".format(view_idx))
                update_image = init_image

                old_mask_tensor += update_mask_tensor
                update_mask_tensor[update_mask_tensor == 1] = 0 # HACK nothing to update

                old_mask_image = transforms.ToPILImage()(old_mask_tensor)
                update_mask_image = transforms.ToPILImage()(update_mask_tensor)


            # 2.3. back-project and create texture
            # NOTE projection mask = update mask
            init_texture, project_mask_image, exist_texture = backproject_from_image(
                mesh, faces, new_verts_uvs, cameras, 
                update_image, update_mask_image, update_mask_image, init_texture, exist_texture, 
                args.text2tex_image_size * args.text2tex_render_simple_factor, args.text2tex_uv_size, args.text2tex_fragment_k,
                DEVICE
            )

            project_mask_image.save(os.path.join(mask_image_dir, "{}_project.png".format(view_idx)))

            # update the mesh
            mesh.textures = TexturesUV(
                maps=transforms.ToTensor()(init_texture)[None, ...].permute(0, 2, 3, 1).to(DEVICE),
                faces_uvs=faces.textures_idx[None, ...],
                verts_uvs=new_verts_uvs[None, ...]
            )

            # 2.4. save generated assets
            # save backprojected OBJ file            
            save_backproject_obj(
                mesh_dir, "{}.obj".format(view_idx),
                mesh_scale * mesh.verts_packed() + mesh_center if args.text2tex_use_unnormalized else mesh.verts_packed(),
                faces.verts_idx, new_verts_uvs, faces.textures_idx, init_texture, 
                DEVICE
            )

            # save the intermediate view
            inter_images_tensor, *_ = render(mesh, renderer)
            inter_image = inter_images_tensor[0].cpu()
            inter_image = inter_image.permute(2, 0, 1)
            inter_image = transforms.ToPILImage()(inter_image).convert("RGB")
            inter_image.save(os.path.join(interm_dir, "{}.png".format(view_idx)))

            # save texture mask
            exist_texture_image = exist_texture * 255. 
            exist_texture_image = Image.fromarray(exist_texture_image.cpu().numpy().astype(np.uint8)).convert("L")
            exist_texture_image.save(os.path.join(mesh_dir, "{}_texture_mask.png".format(view_idx)))

        print("=> total update time: {} s".format(time.time() - start_time))

        # post-process
        if args.text2tex_post_process:
            del controlnet
            del ddim_sampler

            inpainting = get_inpainting(DEVICE)
            post_texture = apply_inpainting_postprocess(inpainting, 
                init_texture, 1-exist_texture[None, :, :, None], "", args.text2tex_uv_size, args.text2tex_uv_size, DEVICE)

            save_backproject_obj(
                mesh_dir, "{}_post.obj".format(view_idx),
                mesh_scale * mesh.verts_packed() + mesh_center if args.text2tex_use_unnormalized else mesh.verts_packed(),
                faces.verts_idx, new_verts_uvs, faces.textures_idx, post_texture, 
                DEVICE
            )
    
        # save viewpoints
        save_viewpoints(args, output_dir, dist_list, elev_list, azim_list, selected_view_ids)

        # visualize viewpoints
        visualize_refinement_viewpoints(output_dir, selected_view_ids, dist_list, elev_list, azim_list)
        print('finish text2tex edit')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # xmesh parameters
    parser.add_argument('--xmesh_obj_path', type=str, default='')
    parser.add_argument('--xmesh_prompt', nargs="+", default='building')
    parser.add_argument('--xmesh_normprompt', nargs="+", default=None)
    parser.add_argument('--xmesh_promptlist', nargs="+", default=None)
    parser.add_argument('--xmesh_normpromptlist', nargs="+", default=None)
    parser.add_argument('--xmesh_image', type=str, default=None)
    parser.add_argument('--xmesh_output_dir', type=str, default='./outputs/combination/xmesh')
    parser.add_argument('--xmesh_traintype', type=str, default="shared")
    parser.add_argument('--xmesh_sigma', type=float, default=10.0)
    parser.add_argument('--xmesh_normsigma', type=float, default=10.0)
    parser.add_argument('--xmesh_depth', type=int, default=4)
    parser.add_argument('--xmesh_width', type=int, default=256)
    parser.add_argument('--xmesh_colordepth', type=int, default=2)
    parser.add_argument('--xmesh_normdepth', type=int, default=2)
    parser.add_argument('--xmesh_normwidth', type=int, default=256)
    parser.add_argument('--xmesh_learning_rate', type=float, default=0.0005)
    parser.add_argument('--xmesh_normal_learning_rate', type=float, default=0.0005)
    parser.add_argument('--xmesh_decay', type=float, default=0)
    parser.add_argument('--xmesh_lr_decay', type=float, default=1)
    parser.add_argument('--xmesh_lr_plateau', action='store_true')
    parser.add_argument('--xmesh_no_pe', dest='pe', default=True, action='store_false')
    parser.add_argument('--xmesh_decay_step', type=int, default=100)
    parser.add_argument('--xmesh_n_views', type=int, default=5)
    parser.add_argument('--xmesh_n_augs', type=int, default=0)
    parser.add_argument('--xmesh_n_normaugs', type=int, default=0)
    parser.add_argument('--xmesh_n_iter', type=int, default=6000)
    parser.add_argument('--xmesh_encoding', type=str, default='gaussian')
    parser.add_argument('--xmesh_normencoding', type=str, default='xyz')
    parser.add_argument('--xmesh_layernorm', action="store_true")
    parser.add_argument('--xmesh_run', type=str, default=None)
    parser.add_argument('--xmesh_gen', action='store_true')
    parser.add_argument('--xmesh_clamp', type=str, default="tanh")
    parser.add_argument('--xmesh_normclamp', type=str, default="tanh")
    parser.add_argument('--xmesh_normratio', type=float, default=0.1)
    parser.add_argument('--xmesh_frontview', action='store_true')
    parser.add_argument('--xmesh_no_prompt', default=False, action='store_true')
    parser.add_argument('--xmesh_exclude', type=int, default=0)

    # Training settings 
    parser.add_argument('--xmesh_frontview_std', type=float, default=8)
    parser.add_argument('--xmesh_frontview_center', nargs=2, type=float, default=[0., 0.])
    parser.add_argument('--xmesh_clipavg', type=str, default=None)
    parser.add_argument('--xmesh_geoloss', action="store_true")
    parser.add_argument('--xmesh_samplebary', action="store_true")
    parser.add_argument('--xmesh_promptviews', nargs="+", default=None)
    parser.add_argument('--xmesh_mincrop', type=float, default=1)
    parser.add_argument('--xmesh_maxcrop', type=float, default=1)
    parser.add_argument('--xmesh_normmincrop', type=float, default=0.1)
    parser.add_argument('--xmesh_normmaxcrop', type=float, default=0.1)
    parser.add_argument('--xmesh_splitnormloss', action="store_true")
    parser.add_argument('--xmesh_splitcolorloss', action="store_true")
    parser.add_argument("--xmesh_nonorm", action="store_true")
    parser.add_argument('--xmesh_cropsteps', type=int, default=0)
    parser.add_argument('--xmesh_cropforward', action='store_true')
    parser.add_argument('--xmesh_cropdecay', type=float, default=1.0)
    parser.add_argument('--xmesh_decayfreq', type=int, default=None)
    parser.add_argument('--xmesh_overwrite', action='store_true')
    parser.add_argument('--xmesh_show', action='store_true')
    parser.add_argument('--xmesh_background', nargs=3, type=float, default=None)
    parser.add_argument('--xmesh_seed', type=int, default=0)
    parser.add_argument('--xmesh_save_render', action="store_true")
    parser.add_argument('--xmesh_input_normals', default=False, action='store_true')
    parser.add_argument('--xmesh_symmetry', default=False, action='store_true')
    parser.add_argument('--xmesh_only_z', default=False, action='store_true')
    parser.add_argument('--xmesh_standardize', default=False, action='store_true')

    # CLIP model settings 
    parser.add_argument('--xmesh_clipmodel', type=str, default='ViT-B/32')
    parser.add_argument('--xmesh_jit', action="store_true")
    
    # text2tex parameters
    parser.add_argument("--text2tex_output_dir", type=str, default="./outputs/combination/text2tex")
    parser.add_argument("--text2tex_prompt", type=str, required=True)
    parser.add_argument("--text2tex_a_prompt", type=str, default="best quality, high quality, extremely detailed, good geometry")
    parser.add_argument("--text2tex_n_prompt", type=str, default="deformed, extra digit, fewer digits, cropped, worst quality, low quality, smoke")
    parser.add_argument("--text2tex_new_strength", type=float, default=1)
    parser.add_argument("--text2tex_update_strength", type=float, default=0.5)
    parser.add_argument("--text2tex_ddim_steps", type=int, default=20)
    parser.add_argument("--text2tex_guidance_scale", type=float, default=10)
    parser.add_argument("--text2tex_output_scale", type=float, default=1)
    parser.add_argument("--text2tex_view_threshold", type=float, default=0.1)
    parser.add_argument("--text2tex_num_viewpoints", type=int, default=8)
    parser.add_argument("--text2tex_viewpoint_mode", type=str, default="predefined", choices=["predefined", "hemisphere"])
    parser.add_argument("--text2tex_update_steps", type=int, default=8)
    parser.add_argument("--text2tex_update_mode", type=str, default="heuristic", choices=["sequential", "heuristic", "random"])
    parser.add_argument("--text2tex_blend", type=float, default=0.5)
    parser.add_argument("--text2tex_eta", type=float, default=0.0)
    parser.add_argument("--text2tex_seed", type=int, default=42)

    parser.add_argument("--text2tex_use_patch", action="store_true", help="apply repaint during refinement to patch up the missing regions")
    parser.add_argument("--text2tex_use_multiple_objects", action="store_true", help="operate on multiple objects")
    parser.add_argument("--text2tex_use_principle", action="store_true", help="poperate on multiple objects")
    parser.add_argument("--text2tex_use_shapenet", action="store_true", help="operate on ShapeNet objects")
    parser.add_argument("--text2tex_use_objaverse", action="store_true", help="operate on Objaverse objects")
    parser.add_argument("--text2tex_use_unnormalized", action="store_true", help="save unnormalized mesh")

    parser.add_argument("--text2tex_add_view_to_prompt", action="store_true", help="add view information to the prompt")
    parser.add_argument("--text2tex_post_process", action="store_true", help="post processing the texture")

    parser.add_argument("--text2tex_smooth_mask", action="store_true", help="smooth the diffusion mask")

    parser.add_argument("--text2tex_force", action="store_true", help="forcefully generate more image")

    # negative options
    parser.add_argument("--text2tex_no_repaint", action="store_true", help="do NOT apply repaint")
    parser.add_argument("--text2tex_no_update", action="store_true", help="do NOT apply update")

    # device parameters
    parser.add_argument("--text2tex_device", type=str, choices=["a6000", "2080"], default="a6000")

    # camera parameters NOTE need careful tuning!!!
    parser.add_argument("--text2tex_test_camera", action="store_true")
    parser.add_argument("--text2tex_dist", type=float, default=1, 
        help="distance to the camera from the object")
    parser.add_argument("--text2tex_elev", type=float, default=0,
        help="the angle between the vector from the object to the camera and the horizontal plane")
    parser.add_argument("--text2tex_azim", type=float, default=180,
        help="the angle between the vector from the object to the camera and the vertical plane")

    args = parser.parse_args()

    # preprocess of text2tex parameters
    if args.text2tex_device == "a6000":
        setattr(args, "text2tex_render_simple_factor", 12)
        setattr(args, "text2tex_fragment_k", 1)
        setattr(args, "text2tex_image_size", 768)
        setattr(args, "text2tex_uv_size", 3000)
    else:
        setattr(args, "text2tex_render_simple_factor", 4)
        setattr(args, "text2tex_fragment_k", 1)
        setattr(args, "text2tex_image_size", 768)
        setattr(args, "text2tex_uv_size", 1000)

    # run xmesh
    seed_everything()
    output_model_obj=run_branched(args)

    # run text2tex preprocess
    new_output_model_obj=text2tex_preprocess(output_model_obj)

    # run text2tex
    text2tex_ori(new_output_model_obj,args)

